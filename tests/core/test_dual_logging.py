# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dual logging functionality."""

import json
import os
import shutil
import tempfile

import pytest

from nvidia_tao_pytorch.core.tlt_logging import logging, logger, StatusLoggerHandler
from nvidia_tao_pytorch.core.loggers.api_logging import StatusLogger, set_status_logger, get_status_logger


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for log files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def reset_logger():
    """Reset logger handlers after each test."""
    # Store original handlers
    original_handlers = logger.handlers.copy()

    yield

    # Restore original handlers
    logger.handlers = original_handlers


def test_status_logger_handler_added_once(temp_log_dir, reset_logger):
    """Test that StatusLoggerHandler is only added once, even with multiple set_status_logger calls."""
    # Create first status logger
    status_log_1 = os.path.join(temp_log_dir, "status1.json")
    set_status_logger(StatusLogger(filename=status_log_1, verbosity=10, append=False))

    # Count StatusLoggerHandler instances
    status_handlers_1 = [h for h in logger.handlers if isinstance(h, StatusLoggerHandler)]
    assert len(status_handlers_1) == 1, "Should have exactly one StatusLoggerHandler after first set_status_logger"

    # Create second status logger
    status_log_2 = os.path.join(temp_log_dir, "status2.json")
    set_status_logger(StatusLogger(filename=status_log_2, verbosity=10, append=False))

    # Count StatusLoggerHandler instances again
    status_handlers_2 = [h for h in logger.handlers if isinstance(h, StatusLoggerHandler)]
    assert len(status_handlers_2) == 1, "Should still have exactly one StatusLoggerHandler after second set_status_logger"

    # Verify it's the same handler instance
    assert status_handlers_1[0] is status_handlers_2[0], "Should be the same handler instance"


def test_dual_logging_writes_to_both_outputs(temp_log_dir, reset_logger, monkeypatch):
    """Test that log messages are written to both console and status file."""
    # Set logging level to DEBUG to capture all messages
    monkeypatch.setenv('TAO_LOGGING_LEVEL', 'DEBUG')
    
    # Reconfigure logger with DEBUG level
    from nvidia_tao_pytorch.core.tlt_logging import get_logging_level
    new_level = get_logging_level()
    logger.setLevel(new_level)
    for handler in logger.handlers:
        handler.setLevel(new_level)
    
    status_log_file = os.path.join(temp_log_dir, "status.json")

    # Setup status logger (automatically enables dual logging)
    set_status_logger(StatusLogger(filename=status_log_file, verbosity=10, append=False))

    # Write various log levels
    test_messages = [
        ("debug", "Test DEBUG message"),
        ("info", "Test INFO message"),
        ("warning", "Test WARNING message"),
        ("error", "Test ERROR message"),
    ]

    for level, message in test_messages:
        getattr(logging, level)(message)

    # Verify status file exists and contains the messages
    assert os.path.exists(status_log_file), "Status log file should exist"

    with open(status_log_file, 'r') as f:
        lines = f.readlines()

    assert len(lines) >= len(test_messages), f"Status file should have at least {len(test_messages)} entries"

    # Parse and verify messages
    found_messages = []
    for line in lines:
        try:
            data = json.loads(line.strip())
            if 'message' in data:
                found_messages.append(data['message'])
        except json.JSONDecodeError:
            pass

    # Check that our test messages appear in the log
    for level, expected_msg in test_messages:
        assert any(expected_msg in msg for msg in found_messages), \
            f"Expected message '{expected_msg}' not found in status log"


def test_status_logger_switching(temp_log_dir, reset_logger):
    """Test that switching status loggers routes messages to the current logger."""
    # Setup first status logger
    status_log_1 = os.path.join(temp_log_dir, "status1.json")
    set_status_logger(StatusLogger(filename=status_log_1, verbosity=10, append=False))

    # Write to first logger
    logging.info("Message to first logger")

    # Switch to second status logger
    status_log_2 = os.path.join(temp_log_dir, "status2.json")
    set_status_logger(StatusLogger(filename=status_log_2, verbosity=10, append=False))

    # Write to second logger
    logging.info("Message to second logger")

    # Switch to third status logger
    status_log_3 = os.path.join(temp_log_dir, "status3.json")
    set_status_logger(StatusLogger(filename=status_log_3, verbosity=10, append=False))

    # Write to third logger
    logging.info("Message to third logger")

    # Verify all files exist
    assert os.path.exists(status_log_1), "First status log should exist"
    assert os.path.exists(status_log_2), "Second status log should exist"
    assert os.path.exists(status_log_3), "Third status log should exist"

    # Verify the third logger is currently active
    assert get_status_logger().log_path == status_log_3, "Third status logger should be active"

    # Helper function to extract messages from a log file
    def get_messages_from_log(log_file):
        messages = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'message' in data:
                            messages.append(data['message'])
                    except json.JSONDecodeError:
                        pass
        return messages

    # Get messages from each log
    messages_1 = get_messages_from_log(status_log_1)
    messages_2 = get_messages_from_log(status_log_2)
    messages_3 = get_messages_from_log(status_log_3)

    # Verify messages went to the right logs
    assert any("Message to first logger" in msg for msg in messages_1), \
        "First log should contain 'Message to first logger'"
    assert any("Message to second logger" in msg for msg in messages_2), \
        "Second log should contain 'Message to second logger'"
    assert any("Message to third logger" in msg for msg in messages_3), \
        "Third log should contain 'Message to third logger'"


def test_log_level_mapping(temp_log_dir, reset_logger, monkeypatch):
    """Test that Python logging levels are correctly mapped to StatusLogger verbosity levels."""
    # Set logging level to DEBUG to capture all messages
    monkeypatch.setenv('TAO_LOGGING_LEVEL', 'DEBUG')
    
    # Reconfigure logger with DEBUG level
    from nvidia_tao_pytorch.core.tlt_logging import get_logging_level
    new_level = get_logging_level()
    logger.setLevel(new_level)
    for handler in logger.handlers:
        handler.setLevel(new_level)
    
    status_log_file = os.path.join(temp_log_dir, "status.json")
    set_status_logger(StatusLogger(filename=status_log_file, verbosity=10, append=False))

    # Test different log levels
    logging.debug("Debug message")
    logging.info("Info message")
    logging.warning("Warning message")
    logging.error("Error message")
    logging.critical("Critical message")

    # Read and parse the log file
    with open(status_log_file, 'r') as f:
        entries = [json.loads(line.strip()) for line in f if line.strip()]

    # Verify that entries have the correct verbosity and status levels
    assert len(entries) >= 5, "Should have at least 5 log entries"

    # Check that error and critical messages have FAILURE status
    error_entries = [e for e in entries if 'Error message' in e.get('message', '')]
    critical_entries = [e for e in entries if 'Critical message' in e.get('message', '')]

    if error_entries:
        assert error_entries[0]['status'] == 'FAILURE', "Error messages should have FAILURE status"
    if critical_entries:
        assert critical_entries[0]['status'] == 'FAILURE', "Critical messages should have FAILURE status"

    # Check that other messages have RUNNING status
    info_entries = [e for e in entries if 'Info message' in e.get('message', '')]
    if info_entries:
        assert info_entries[0]['status'] == 'RUNNING', "Info messages should have RUNNING status"


def test_dual_logging_silent_on_multiple_calls(temp_log_dir, reset_logger, caplog):
    """Test that multiple set_status_logger calls don't generate warnings."""
    import logging as std_logging

    # Set logging level to capture warnings
    caplog.set_level(std_logging.WARNING)

    # Make multiple calls
    for i in range(3):
        status_log = os.path.join(temp_log_dir, f"status{i}.json")
        set_status_logger(StatusLogger(filename=status_log, verbosity=10, append=False))

    # Check that no warnings about duplicate handlers were logged
    duplicate_handler_warnings = [
        record for record in caplog.records
        if "StatusLoggerHandler already added" in record.getMessage()
    ]

    assert len(duplicate_handler_warnings) == 0, \
        "Should not generate warnings about duplicate handlers"


def test_status_logger_handler_fetches_current_logger(temp_log_dir, reset_logger):
    """Test that StatusLoggerHandler always uses the current status logger."""
    # This is implicitly tested in test_status_logger_switching,
    # but we'll verify the handler behavior more directly

    status_log_1 = os.path.join(temp_log_dir, "handler_test_1.json")
    status_log_2 = os.path.join(temp_log_dir, "handler_test_2.json")

    # Set first logger
    set_status_logger(StatusLogger(filename=status_log_1, verbosity=10, append=False))
    current_logger_1 = get_status_logger()

    # Get the handler
    status_handlers = [h for h in logger.handlers if isinstance(h, StatusLoggerHandler)]
    assert len(status_handlers) == 1
    handler = status_handlers[0]

    # Set second logger
    set_status_logger(StatusLogger(filename=status_log_2, verbosity=10, append=False))
    current_logger_2 = get_status_logger()

    # Verify the status logger changed
    assert current_logger_1 is not current_logger_2, "Status loggers should be different instances"
    assert current_logger_1.log_path != current_logger_2.log_path, "Status loggers should have different paths"

    # Verify handler is still the same instance
    status_handlers_after = [h for h in logger.handlers if isinstance(h, StatusLoggerHandler)]
    assert len(status_handlers_after) == 1
    assert status_handlers_after[0] is handler, "Handler should be the same instance"

    # Write a message - it should go to the second logger
    logging.info("Test message to verify handler uses current logger")

    # Verify the message went to the second log file
    with open(status_log_2, 'r') as f:
        lines = f.readlines()

    messages = []
    for line in lines:
        try:
            data = json.loads(line.strip())
            if 'message' in data:
                messages.append(data['message'])
        except json.JSONDecodeError:
            pass

    assert any("Test message to verify handler uses current logger" in msg for msg in messages), \
        "Message should appear in the second log file"
