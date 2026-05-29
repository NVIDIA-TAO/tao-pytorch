# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OCRNet utils module."""
import os
import pickle
import struct
import tempfile
import logging
from eff.codec import encrypt_stream, decrypt_stream
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """Convert between text-label and text-index for CTC."""

    def __init__(self, character):
        """Initialize CTCLabelConverter.

        Args:
            character (str): A string containing the set of possible characters.
        """
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """Convert text-label into text-index.

        Args:
            text (list): text labels of each image. [batch_size]
            batch_max_length (int): max length of text label in the batch. 25 by default

        Return:
            text (Torch.tensor): text index for CTCLoss. [batch_size, batch_max_length]
            length (Torch.tensor): length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """Convert text-index into text-label.

        Args:
            text_index (numpy.ndarray): the batch of predicted text_index.
            length (list): the length of the predicted text.

        Return:
            list : the list of decoded text.
        """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts


class CTCLabelConverterForBaiduWarpctc(object):
    """Convert between text-label and text-index for baidu warpctc."""

    def __init__(self, character):
        """Initialize CTCLabelConverterForBaiduWarpctc.

        Args:
            character (str): A string containing the set of possible characters.
        """
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """Convert text-label into text-index.

        Args:
            text (list): text labels of each image. [batch_size]

        Return:
            text (torch.Tensor): concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length (torch.Tensor): length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        """Convert text-index into text-label.

        Args:
            text_index (numpy.ndarray): the batch of predicted text_index.
            length (list): the length of the predicted text.

        Return:
            list : the list of decoded text.
        """
        texts = []
        index = 0
        for ll in length:
            t = text_index[index:index + ll]

            char_list = []
            for i in range(ll):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += ll
        return texts


class AttnLabelConverter(object):
    """Convert between text-label and text-index for Attention"""

    def __init__(self, character):
        """Initialize AttnLabelConverter.

        Args:
            character (str): A string containing the set of possible characters.
        """
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """Convert text-label into text-index.

        Args:
            text (list): text labels of each image. [batch_size]
            batch_max_length (int): max length of text label in the batch. 25 by default

        Return:
            text (torch.Tensor): the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length (torch.Tensor): the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """Convert text-index into text-label.

        Args:
            text_index (numpy.ndarray): the batch of predicted text_index.
            length (list): the length of the predicted text.

        Return:
            list : the list of decoded text.
        """
        texts = []
        for index, _ in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        """init."""
        self.reset()

    def add(self, v):
        """add."""
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        """reset."""
        self.n_count = 0
        self.sum = 0

    def val(self):
        """val."""
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def encrypt_onnx(tmp_file_name, output_file_name, key):
    """Encrypts an ONNX model.

    Args:
        tmp_file_name (str): The name of the temporary file containing the ONNX model.
        output_file_name (str): The name of the output file to write the encrypted ONNX model to.
        key (str): The passphrase to use for encryption.
    """
    with open(tmp_file_name, "rb") as open_temp_file, open(output_file_name,
                                                           "wb") as open_encoded_file:
        # set the input name magic number
        open_encoded_file.write(struct.pack("<i", 0))
        encrypt_stream(
            input_stream=open_temp_file, output_stream=open_encoded_file,
            passphrase=key, encryption=True
        )


def encrypt_pytorch(tmp_file_name, output_file_name, key):
    """Encrypts a PyTorch model.

    Args:
        tmp_file_name (str): The name of the temporary file containing the PyTorch model.
        output_file_name (str): The name of the output file to write the encrypted PyTorch model to.
        key (str): The passphrase to use for encryption.
    """
    with open(tmp_file_name, "rb") as open_temp_file, open(output_file_name,
                                                           "wb") as open_encoded_file:
        encrypt_stream(
            input_stream=open_temp_file, output_stream=open_encoded_file,
            passphrase=key, encryption=True
        )


def save_checkpoint(state, filename, key):
    """Saves a PyTorch checkpoint.

    Args:
        state (dict): The state dictionary to save.
        filename (str): The name of the output file to write the encrypted checkpoint to.
        key (str): The passphrase to use for encryption.

    """
    handle, temp_name = tempfile.mkstemp(".tlt")
    os.close(handle)
    torch.save(state, temp_name)
    encrypt_pytorch(temp_name, filename, key)
    os.remove(temp_name)


def decrypt_pytorch(input_file_name, output_file_name, key):
    """Decrypts a TAO model to a PyTorch model.

    Args:
        input_file_name (str): The name of the input file containing the encrypted TAO model.
        output_file_name (str): The name of the output file to write the decrypted PyTorch model to.
        key (str): The passphrase to use for decryption.
    """
    with open(input_file_name, "rb") as open_temp_file, open(output_file_name,
                                                             "wb") as open_encoded_file:
        decrypt_stream(
            input_stream=open_temp_file, output_stream=open_encoded_file,
            passphrase=key, encryption=True
        )


def load_checkpoint(model_path, key, to_cpu=False):
    """Loads a saved PyTorch checkpoint.

    Args:
        model_path (str): The path to the saved checkpoint file.
        key (str): The passphrase to use for decryption.
        to_cpu (bool, optional): Whether to load the model onto the CPU. Defaults to False.

    Returns:
        dict: The loaded state dictionary.

    """
    loc_type = torch.device('cpu') if to_cpu else None
    if model_path.endswith(".tlt"):
        handle, temp_name = tempfile.mkstemp(".pth")
        os.close(handle)
        decrypt_pytorch(model_path, temp_name, key)
        loaded_state = torch.load(temp_name, map_location=loc_type, weights_only=False)
        os.remove(temp_name)
    else:
        loaded_state = torch.load(model_path, map_location=loc_type, weights_only=False)
        if isinstance(loaded_state, dict):
            if "whole_model" in loaded_state:
                loaded_state = pickle.loads(loaded_state["whole_model"])

    return loaded_state


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    """Creates a logger object.

    Args:
        log_file (str, optional): The name of the log file to write to. Defaults to None.
        rank (int, optional): The rank of the process. Defaults to 0.
        log_level (int, optional): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: The logger object.

    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    if log_file is not None:
        if (logger.hasHandlers()):
            logger.handlers.clear()
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def quantize_model(ocr_model, dm):
    """Quantize the OCRNetModel.

    Args:
        ocr_model (OCRNetModel): The candidate to be quantized.
        dm (DataModule): DataModule to get calibration dataset.
    """
    # Init the stuff for QDQ
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization import QuantModuleRegistry
    import torch.nn as nn
    # To avoid the conversion from nn.LSTM to QuantLSTM.
    # The QuantLSTM is using LSTMCell which will be exported to _thnn_lstm_cell in ONNX.
    # And the _thnn_lstm_cell cannot be consumed by TensorRT.
    QuantModuleRegistry.unregister(nn.LSTM)
    model_quant_config = mtq.INT8_DEFAULT_CFG.copy()
    calib_dataset = dm.val_dataloader()
    model = ocr_model.model
    batch_max_length = ocr_model.max_label_length
    model.to(device)

    # A fake model forward loop function for mtq.quantize.
    # @TODO(tylerz): Why do we need such calibration in QAT?
    def forward_loop(model):
        text_for_pred = torch.LongTensor(1, batch_max_length + 1).fill_(0).to(device)
        for batch in calib_dataset:
            image, _ = batch
            image = image.to(device)
            model(image, text_for_pred)

    model_quant_config["quant_cfg"]["*rnn*"] = {"enable": False}

    model = mtq.quantize(model, model_quant_config, forward_loop)
    ocr_model.model = model
