# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry management for quantization framework."""

from typing import Dict, Type, List
import threading


class RegistryManager:
    """A thread-safe registry manager for quantization components.

    This class manages registries for observers, fake quantizers, and backends,
    providing a centralized way to register and retrieve quantization components.
    """

    def __init__(self):
        """Initialize the registry manager with empty registries."""
        self._observers: Dict[str, Type] = {}
        self._fake_quants: Dict[str, Type] = {}
        self._backends: Dict[str, Type] = {}
        self._lock = threading.RLock()

    def register_observer(self, name: str, observer_class: Type) -> None:
        """
        Register an observer class under a specified name.

        Parameters
        ----------
        name : str
            The unique name to register the observer class under.
        observer_class : Type
            The observer class to register.

        Raises
        ------
        ValueError
            If the given name is already registered.
        TypeError
            If the observer_class is not a class.
        """
        with self._lock:
            if name in self._observers:
                raise ValueError(f"Observer '{name}' is already registered.")
            if not isinstance(observer_class, type):
                raise TypeError(
                    f"Observer '{name}' must be a class, but got {observer_class}."
                )

            self._observers[name] = observer_class

    def register_fake_quant(self, name: str, fake_quant_class: Type) -> None:
        """
        Register a fake quant class under a specified name.

        Parameters
        ----------
        name : str
            The unique name to register the fake quant class under.
        fake_quant_class : Type
            The fake quant class to register.

        Raises
        ------
        ValueError
            If the given name is already registered.
        TypeError
            If the fake_quant_class is not a class.
        """
        with self._lock:
            if name in self._fake_quants:
                raise ValueError(f"Fake quant '{name}' is already registered.")
            if not isinstance(fake_quant_class, type):
                raise TypeError(
                    f"Fake quant '{name}' must be a class, but got {fake_quant_class}."
                )

            self._fake_quants[name] = fake_quant_class

    def register_backend(self, name: str, backend_class: Type) -> None:
        """
        Register a backend class under a specified name.

        Parameters
        ----------
        name : str
            The unique name to register the backend class under.
        backend_class : Type
            The backend class to register.

        Raises
        ------
        ValueError
            If the given name is already registered.
        TypeError
            If the backend_class is not a class.
        """
        with self._lock:
            if name in self._backends:
                raise ValueError(f"Backend '{name}' is already registered.")
            if not isinstance(backend_class, type):
                raise TypeError(
                    f"Backend '{name}' must be a class, but got {backend_class}."
                )

            self._backends[name] = backend_class

    def get_observer(self, name: str) -> Type:
        """
        Get the observer class for a given name.

        Parameters
        ----------
        name : str
            Name of the observer to retrieve.

        Returns
        -------
        Type
            The observer class if registered.

        Raises
        ------
        ValueError
            If the observer is not registered.
        """
        with self._lock:
            if name not in self._observers:
                raise ValueError(f"Observer '{name}' is not registered.")
            return self._observers[name]

    def get_fake_quant(self, name: str) -> Type:
        """
        Get the fake quant class for a given name.

        Parameters
        ----------
        name : str
            Name of the fake quant to retrieve.

        Returns
        -------
        Type
            The fake quant class if registered.

        Raises
        ------
        ValueError
            If the fake quant is not registered.
        """
        with self._lock:
            if name not in self._fake_quants:
                raise ValueError(f"Fake quant '{name}' is not registered.")
            return self._fake_quants[name]

    def get_backend(self, name: str) -> Type:
        """
        Get the backend class for a given name.

        Parameters
        ----------
        name : str
            Name of the backend to retrieve.

        Returns
        -------
        Type
            The backend class if registered.

        Raises
        ------
        ValueError
            If the backend is not registered.
        """
        with self._lock:
            if name not in self._backends:
                raise ValueError(f"Backend '{name}' is not registered.")
            return self._backends[name]

    def get_available_observers(self) -> List[str]:
        """
        Get all available observer names.

        Returns
        -------
        List[str]
            List of observer names that are currently registered.
        """
        with self._lock:
            return list(self._observers.keys())

    def get_available_fake_quants(self) -> List[str]:
        """
        Get all available fake quant names.

        Returns
        -------
        List[str]
            List of fake quant names that are currently registered.
        """
        with self._lock:
            return list(self._fake_quants.keys())

    def get_available_backends(self) -> List[str]:
        """
        Get all available backend names.

        Returns
        -------
        List[str]
            List of backend names that are currently registered.
        """
        with self._lock:
            return list(self._backends.keys())

    def unregister_observer(self, name: str) -> None:
        """
        Unregister an observer by name.

        Parameters
        ----------
        name : str
            Name of the observer to unregister.

        Raises
        ------
        ValueError
            If the observer is not registered.
        """
        with self._lock:
            if name not in self._observers:
                raise ValueError(f"Observer '{name}' is not registered.")
            del self._observers[name]

    def unregister_fake_quant(self, name: str) -> None:
        """
        Unregister a fake quant by name.

        Parameters
        ----------
        name : str
            Name of the fake quant to unregister.

        Raises
        ------
        ValueError
            If the fake quant is not registered.
        """
        with self._lock:
            if name not in self._fake_quants:
                raise ValueError(f"Fake quant '{name}' is not registered.")
            del self._fake_quants[name]

    def unregister_backend(self, name: str) -> None:
        """
        Unregister a backend by name.

        Parameters
        ----------
        name : str
            Name of the backend to unregister.

        Raises
        ------
        ValueError
            If the backend is not registered.
        """
        with self._lock:
            if name not in self._backends:
                raise ValueError(f"Backend '{name}' is not registered.")
            del self._backends[name]

    def clear_all(self) -> None:
        """Clear all registries."""
        with self._lock:
            self._observers.clear()
            self._fake_quants.clear()
            self._backends.clear()

    def is_observer_registered(self, name: str) -> bool:
        """
        Check if an observer is registered.

        Parameters
        ----------
        name : str
            Name of the observer to check.

        Returns
        -------
        bool
            True if the observer is registered, False otherwise.
        """
        with self._lock:
            return name in self._observers

    def is_fake_quant_registered(self, name: str) -> bool:
        """
        Check if a fake quant is registered.

        Parameters
        ----------
        name : str
            Name of the fake quant to check.

        Returns
        -------
        bool
            True if the fake quant is registered, False otherwise.
        """
        with self._lock:
            return name in self._fake_quants

    def is_backend_registered(self, name: str) -> bool:
        """
        Check if a backend is registered.

        Parameters
        ----------
        name : str
            Name of the backend to check.

        Returns
        -------
        bool
            True if the backend is registered, False otherwise.
        """
        with self._lock:
            return name in self._backends


# Global registry manager instance
_registry_manager = RegistryManager()


# Convenience functions that use the global registry manager
def register_observer(name: str):
    """Register an observer class under a specified name.

    This decorator adds the given observer class to the global registry manager,
    enabling it to be referenced by name in quantization configurations.

    Parameters
    ----------
    name : str
        The unique name to register the observer class under.

    Returns
    -------
    decorator : callable
        A decorator that registers the observer class.
    """

    def decorator(observer):
        _registry_manager.register_observer(name, observer)
        return observer

    return decorator


def register_fake_quant(name: str):
    """Register a fake quant class under a specified name.

    This decorator adds the given fake quant class to the global registry manager,
    enabling it to be referenced by name in quantization configurations.

    Parameters
    ----------
    name : str
        The unique name to register the fake quant class under.

    Returns
    -------
    decorator : callable
        A decorator that registers the fake quant class.
    """

    def decorator(fake_quant):
        _registry_manager.register_fake_quant(name, fake_quant)
        return fake_quant

    return decorator


def register_backend(name: str):
    """Register a backend class under a specified name.

    This decorator adds the given backend class to the global registry manager,
    enabling it to be referenced by name in quantization configurations.

    Parameters
    ----------
    name : str
        The unique name to register the backend class under.

    Returns
    -------
    decorator : callable
        A decorator that registers the backend class.
    """

    def decorator(backend):
        _registry_manager.register_backend(name, backend)
        return backend

    return decorator


def get_available_backends():
    """
    Get all available quantization backends.

    Returns
    -------
    list
        List of backend names that are currently registered.
    """
    return _registry_manager.get_available_backends()


def get_backend_class(backend_name: str):
    """
    Get the backend class for a given backend name.

    Parameters
    ----------
    backend_name : str
        Name of the backend to retrieve.

    Returns
    -------
    class
        The backend class if registered.

    """
    return _registry_manager.get_backend(backend_name)


# Additional convenience functions for the new functionality
def get_available_observers():
    """
    Get all available observer names.

    Returns
    -------
    list
        List of observer names that are currently registered.
    """
    return _registry_manager.get_available_observers()


def get_available_fake_quants():
    """
    Get all available fake quant names.

    Returns
    -------
    list
        List of fake quant names that are currently registered.
    """
    return _registry_manager.get_available_fake_quants()


def get_observer_class(observer_name: str):
    """
    Get the observer class for a given observer name.

    Parameters
    ----------
    observer_name : str
        Name of the observer to retrieve.

    Returns
    -------
    class
        The observer class if registered.
    """
    return _registry_manager.get_observer(observer_name)


def get_fake_quant_class(fake_quant_name: str):
    """
    Get the fake quant class for a given fake quant name.

    Parameters
    ----------
    fake_quant_name : str
        Name of the fake quant to retrieve.

    Returns
    -------
    class
        The fake quant class if registered.
    """
    return _registry_manager.get_fake_quant(fake_quant_name)


def get_registry_manager() -> RegistryManager:
    """
    Get the global registry manager instance.

    Returns
    -------
    RegistryManager
        The global registry manager instance.
    """
    return _registry_manager
