# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OCRNet sequence module."""
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """Bi-SLTM."""

    def __init__(self, input_size, hidden_size, output_size):
        """Init.

        Args:
            input_size (int): The size of the input.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output.
        """
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):  # pylint: disable=redefined-builtin
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
