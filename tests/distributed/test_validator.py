# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging
from unittest.mock import patch
import socket

from nvidia_tao_pytorch.core.distributed.validator import validate_configs

@pytest.fixture
def logger():
    return logging.getLogger("test_logger")

def test_invalid_num_nodes(logger):
    with patch.dict('os.environ', {'WORLD_SIZE': '0'}):
        with pytest.raises(ValueError, match="Number of nodes must be greater than 0"):
            validate_configs(logger)

def test_invalid_node_rank(logger):
    with patch.dict('os.environ', {
        'WORLD_SIZE': '2',
        'NODE_RANK': '2'  # Invalid as it's >= num_nodes
    }):
        with pytest.raises(ValueError, match="Node rank must be between 0"):
            validate_configs(logger)

def test_invalid_port(logger):
    with patch.dict('os.environ', {
        'WORLD_SIZE': '2',
        'NODE_RANK': '0',
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '100'  # Invalid as it's < 1024
    }):
        with pytest.raises(ValueError, match="Port must be between 1024 and 65535"):
            validate_configs(logger)

def test_missing_multinode_params(logger):
    with patch.dict('os.environ', {'WORLD_SIZE': '2'}):
        with pytest.raises(ValueError, match="Master address, port, and node rank must be specified"):
            validate_configs(logger)

def test_single_node_with_multinode_params(logger, caplog):
    with patch.dict('os.environ', {
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '1234',
        'NODE_RANK': '0'
    }):
        validate_configs(logger)
        assert "Multinode training is not enabled" in caplog.text

@patch('socket.gethostbyname')
def test_invalid_master_address(mock_gethostbyname, logger):
    mock_gethostbyname.side_effect = socket.gaierror
    with patch.dict('os.environ', {
        'WORLD_SIZE': '2',
        'NODE_RANK': '0',
        'MASTER_ADDR': 'invalid_host',
        'MASTER_PORT': '1234'
    }):
        with pytest.raises(ValueError, match="Invalid master address"):
            validate_configs(logger)

@patch('socket.socket')
def test_port_in_use(mock_socket, logger):
    mock_socket_instance = mock_socket.return_value.__enter__.return_value
    mock_socket_instance.bind.side_effect = socket.error
    
    with patch.dict('os.environ', {
        'WORLD_SIZE': '2',
        'NODE_RANK': '0',
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '1234'
    }):
        with pytest.raises(ValueError, match="Port .* is already in use"):
            validate_configs(logger)

@patch('torch.cuda.device_count', return_value=4)
def test_successful_validation(logger):
    with patch('socket.gethostbyname'), \
         patch('socket.socket') as mock_socket:
        mock_socket.return_value.__enter__.return_value.bind.return_value = None
        
        with patch.dict('os.environ', {
            'WORLD_SIZE': '2',
            'NODE_RANK': '0',
            'MASTER_ADDR': 'localhost',
            'MASTER_PORT': '1234',
            'NUM_GPU_PER_NODE': '4',
        }):
            validate_configs(logger)