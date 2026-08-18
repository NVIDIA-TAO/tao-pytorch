# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for quantization registry management."""

import pytest

from nvidia_tao_pytorch.core.quantization.registry import (
    RegistryManager,
    register_observer,
    register_fake_quant,
    register_backend,
    get_available_backends,
    get_backend_class,
    get_available_observers,
    get_available_fake_quants,
    get_observer_class,
    get_fake_quant_class,
    get_registry_manager,
)


class MockObserver:
    """Mock observer class for testing."""

    pass


class MockFakeQuant:
    """Mock fake quant class for testing."""

    pass


class MockBackend:
    """Mock backend class for testing."""

    pass


class TestRegistryManager:
    """Test cases for the RegistryManager class."""

    def setup_method(self):
        """Set up a fresh registry manager for each test."""
        self.registry = RegistryManager()

    def test_register_observer(self):
        """Test registering an observer."""
        self.registry.register_observer("test_observer", MockObserver)
        assert self.registry.is_observer_registered("test_observer")
        assert self.registry.get_observer("test_observer") == MockObserver

    def test_register_fake_quant(self):
        """Test registering a fake quant."""
        self.registry.register_fake_quant("test_fake_quant", MockFakeQuant)
        assert self.registry.is_fake_quant_registered("test_fake_quant")
        assert self.registry.get_fake_quant("test_fake_quant") == MockFakeQuant

    def test_register_backend(self):
        """Test registering a backend."""
        self.registry.register_backend("test_backend", MockBackend)
        assert self.registry.is_backend_registered("test_backend")
        assert self.registry.get_backend("test_backend") == MockBackend

    def test_register_duplicate_observer(self):
        """Test that registering a duplicate observer raises ValueError."""
        self.registry.register_observer("test_observer", MockObserver)
        with pytest.raises(
            ValueError, match="Observer 'test_observer' is already registered"
        ):
            self.registry.register_observer("test_observer", MockObserver)

    def test_register_duplicate_fake_quant(self):
        """Test that registering a duplicate fake quant raises ValueError."""
        self.registry.register_fake_quant("test_fake_quant", MockFakeQuant)
        with pytest.raises(
            ValueError, match="Fake quant 'test_fake_quant' is already registered"
        ):
            self.registry.register_fake_quant("test_fake_quant", MockFakeQuant)

    def test_register_duplicate_backend(self):
        """Test that registering a duplicate backend raises ValueError."""
        self.registry.register_backend("test_backend", MockBackend)
        with pytest.raises(
            ValueError, match="Backend 'test_backend' is already registered"
        ):
            self.registry.register_backend("test_backend", MockBackend)

    def test_register_non_class_observer(self):
        """Test that registering a non-class observer raises TypeError."""
        with pytest.raises(TypeError, match="Observer 'test' must be a class"):
            self.registry.register_observer("test", "not_a_class")

    def test_register_non_class_fake_quant(self):
        """Test that registering a non-class fake quant raises TypeError."""
        with pytest.raises(TypeError, match="Fake quant 'test' must be a class"):
            self.registry.register_fake_quant("test", "not_a_class")

    def test_register_non_class_backend(self):
        """Test that registering a non-class backend raises TypeError."""
        with pytest.raises(TypeError, match="Backend 'test' must be a class"):
            self.registry.register_backend("test", "not_a_class")

    def test_get_nonexistent_observer(self):
        """Test that getting a nonexistent observer raises ValueError."""
        with pytest.raises(
            ValueError, match="Observer 'nonexistent' is not registered"
        ):
            self.registry.get_observer("nonexistent")

    def test_get_nonexistent_fake_quant(self):
        """Test that getting a nonexistent fake quant raises ValueError."""
        with pytest.raises(
            ValueError, match="Fake quant 'nonexistent' is not registered"
        ):
            self.registry.get_fake_quant("nonexistent")

    def test_get_nonexistent_backend(self):
        """Test that getting a nonexistent backend raises ValueError."""
        with pytest.raises(ValueError, match="Backend 'nonexistent' is not registered"):
            self.registry.get_backend("nonexistent")

    def test_get_available_observers(self):
        """Test getting available observer names."""
        assert not self.registry.get_available_observers()

        self.registry.register_observer("obs1", MockObserver)
        self.registry.register_observer("obs2", MockObserver)

        available = self.registry.get_available_observers()
        assert "obs1" in available
        assert "obs2" in available
        assert len(available) == 2

    def test_get_available_fake_quants(self):
        """Test getting available fake quant names."""
        assert not self.registry.get_available_fake_quants()

        self.registry.register_fake_quant("fq1", MockFakeQuant)
        self.registry.register_fake_quant("fq2", MockFakeQuant)

        available = self.registry.get_available_fake_quants()
        assert "fq1" in available
        assert "fq2" in available
        assert len(available) == 2

    def test_get_available_backends(self):
        """Test getting available backend names."""
        assert not self.registry.get_available_backends()

        self.registry.register_backend("backend1", MockBackend)
        self.registry.register_backend("backend2", MockBackend)

        available = self.registry.get_available_backends()
        assert "backend1" in available
        assert "backend2" in available
        assert len(available) == 2

    def test_unregister_observer(self):
        """Test unregistering an observer."""
        self.registry.register_observer("test_observer", MockObserver)
        assert self.registry.is_observer_registered("test_observer")

        self.registry.unregister_observer("test_observer")
        assert not self.registry.is_observer_registered("test_observer")

    def test_unregister_fake_quant(self):
        """Test unregistering a fake quant."""
        self.registry.register_fake_quant("test_fake_quant", MockFakeQuant)
        assert self.registry.is_fake_quant_registered("test_fake_quant")

        self.registry.unregister_fake_quant("test_fake_quant")
        assert not self.registry.is_fake_quant_registered("test_fake_quant")

    def test_unregister_backend(self):
        """Test unregistering a backend."""
        self.registry.register_backend("test_backend", MockBackend)
        assert self.registry.is_backend_registered("test_backend")

        self.registry.unregister_backend("test_backend")
        assert not self.registry.is_backend_registered("test_backend")

    def test_unregister_nonexistent_observer(self):
        """Test that unregistering a nonexistent observer raises ValueError."""
        with pytest.raises(
            ValueError, match="Observer 'nonexistent' is not registered"
        ):
            self.registry.unregister_observer("nonexistent")

    def test_unregister_nonexistent_fake_quant(self):
        """Test that unregistering a nonexistent fake quant raises ValueError."""
        with pytest.raises(
            ValueError, match="Fake quant 'nonexistent' is not registered"
        ):
            self.registry.unregister_fake_quant("nonexistent")

    def test_unregister_nonexistent_backend(self):
        """Test that unregistering a nonexistent backend raises ValueError."""
        with pytest.raises(ValueError, match="Backend 'nonexistent' is not registered"):
            self.registry.unregister_backend("nonexistent")

    def test_clear_all(self):
        """Test clearing all registries."""
        self.registry.register_observer("obs", MockObserver)
        self.registry.register_fake_quant("fq", MockFakeQuant)
        self.registry.register_backend("backend", MockBackend)

        assert len(self.registry.get_available_observers()) == 1
        assert len(self.registry.get_available_fake_quants()) == 1
        assert len(self.registry.get_available_backends()) == 1

        self.registry.clear_all()

        assert len(self.registry.get_available_observers()) == 0
        assert len(self.registry.get_available_fake_quants()) == 0
        assert len(self.registry.get_available_backends()) == 0


class TestRegistryDecorators:
    """Test cases for the registry decorators."""

    def setup_method(self):
        """Set up a fresh registry manager for each test."""
        # Get the global registry manager and clear it
        registry_manager = get_registry_manager()
        registry_manager.clear_all()

    def test_register_observer_decorator(self):
        """Test the register_observer decorator."""

        @register_observer("test_observer")
        class TestObserver:
            pass

        assert get_observer_class("test_observer") == TestObserver
        assert "test_observer" in get_available_observers()

    def test_register_fake_quant_decorator(self):
        """Test the register_fake_quant decorator."""

        @register_fake_quant("test_fake_quant")
        class TestFakeQuant:
            pass

        assert get_fake_quant_class("test_fake_quant") == TestFakeQuant
        assert "test_fake_quant" in get_available_fake_quants()

    def test_register_backend_decorator(self):
        """Test the register_backend decorator."""

        @register_backend("test_backend")
        class TestBackend:
            pass

        assert get_backend_class("test_backend") == TestBackend
        assert "test_backend" in get_available_backends()

    def test_duplicate_registration_decorator(self):
        """Test that duplicate registration with decorators raises ValueError."""

        @register_observer("test_observer")
        class TestObserver1:  # pylint: disable=unused-variable
            pass
        del TestObserver1

        with pytest.raises(
            ValueError, match="Observer 'test_observer' is already registered"
        ):

            @register_observer("test_observer")
            class TestObserver2:  # pylint: disable=unused-variable
                pass


class TestConvenienceFunctions:
    """Test cases for the convenience functions."""

    def setup_method(self):
        """Set up a fresh registry manager for each test."""
        registry_manager = get_registry_manager()
        registry_manager.clear_all()

    def test_get_registry_manager(self):
        """Test getting the global registry manager."""
        registry_manager = get_registry_manager()
        assert isinstance(registry_manager, RegistryManager)

    def test_convenience_functions_work_with_global_manager(self):
        """Test that convenience functions work with the global registry manager."""

        @register_observer("test_observer")
        class TestObserver:
            pass

        @register_fake_quant("test_fake_quant")
        class TestFakeQuant:
            pass

        @register_backend("test_backend")
        class TestBackend:
            pass

        # Test convenience functions
        assert get_observer_class("test_observer") == TestObserver
        assert get_fake_quant_class("test_fake_quant") == TestFakeQuant
        assert get_backend_class("test_backend") == TestBackend

        assert "test_observer" in get_available_observers()
        assert "test_fake_quant" in get_available_fake_quants()
        assert "test_backend" in get_available_backends()
