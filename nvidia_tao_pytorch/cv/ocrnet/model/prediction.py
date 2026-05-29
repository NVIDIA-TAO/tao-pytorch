# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OCRNet prediction module."""
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention(nn.Module):
    """Attention module."""

    def __init__(self, input_size, hidden_size, num_classes, batch_max_length=25):
        """Init.

        Args:
            input_size (int): The size of the input.
            hidden_size (int): The size of the hidden layer.
            num_classes (int): The number of classes.
        """
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)
        self.register_buffer("initial_target", torch.LongTensor(1).fill_(0).to(device))
        self.register_buffer("initial_prob", torch.FloatTensor(batch_max_length + 1, self.num_classes).fill_(0))
        self.register_buffer("one_hot", torch.FloatTensor(1, self.num_classes).zero_())
        self.register_buffer("initial_hidden", torch.FloatTensor(1, self.hidden_size).fill_(0))

    def _char_to_onehot(self, input_char, onehot_dim=38):
        """Convert char label to onehot."""
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = self.one_hot.repeat(batch_size, 1, 1)
        one_hot = one_hot.scatter_(2, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=True, batch_max_length=25):
        """ Perform attention forward.
        Args:
            batch_H (torch.Tensor): contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text (torch.Tensor): the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
            is_train (bool, optional): Set it to be true if in train phase.
            batch_max_length (bool, optional): the maximum length of text in this batch.

        Returns:
            prob: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        # print(batch_H.shape)
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        hidden = (self.initial_hidden.repeat(1, batch_size, 1),
                  self.initial_hidden.repeat(1, batch_size, 1))

        if is_train:
            output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
            targets = self.initial_target.repeat(batch_size, 1)  # [GO] token
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i:i + 1], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, _ = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)
        else:
            targets = self.initial_target.repeat(batch_size, 1)  # [GO] token
            probs = self.initial_prob.repeat(batch_size, 1, 1)
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, _ = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs_step = probs_step.permute(1, 0, 2)
                probs[:, i:i + 1, :] = probs_step
                _, next_input = probs_step.max(2)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionCell(nn.Module):
    """Attention Cell."""

    def __init__(self, input_size, hidden_size, num_embeddings):
        """Init.

        Args:
            input_size (int): The size of the input.
            hidden_size (int): The size of the hidden layer.
            num_embeddings (int): The number of embeddings.
        """
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTM(input_size + num_embeddings, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        """Performs a forward pass through the AttentionCell network.

        Args:
            prev_hidden (tuple): The previous hidden state.
            batch_H (torch.Tensor): The input tensor.
            char_onehots (torch.Tensor): The one-hot encoded character tensor.

        Returns:
            tuple: The current hidden state and the attention weights.
        """
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0])
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj.permute(1, 0, 2)))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H)  # batch_size x num_channel
        concat_context = torch.cat([context, char_onehots], 2)  # batch_size x (num_channel + num_embedding)
        _, cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha
