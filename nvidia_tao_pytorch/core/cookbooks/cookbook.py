# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File containing cookbook base class and methods"""


import os
from typing import Union

from eff.callbacks import BinaryContentCallback, StringContentCallback
from eff.core.archive import Archive
from eff.utils.object_class import generate_obj_cls
from eff.validator.validator import validate_metadata
from ruamel.yaml import YAML, yaml_object

__all__ = ['Cookbook', 'ArtifactPathType']

yaml = YAML()


@yaml_object(yaml)
class ArtifactPathType():
    """
    ArtifactPathType refers to the type of the path that the artifact is located at.

    LOCAL_PATH: A user local filepath that exists on the file system.
    TAR_PATH: A (generally flattened) filepath that exists inside of an archive (that may have its own full path).
    """

    LOCAL_PATH = 'LOCAL_PATH'
    TAR_PATH = 'TAR_PATH'


class classproperty(object):
    """Helper class, defining a combined classmethod+property decorator """

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class Cookbook(object):
    """Parent, abstract cookbook class.

    This class cannot be used as is. You will need to derive a class for your application.
    """

    def __init_subclass__(cls, **kwargs):
        """ Adds required properties to the (sub)classes. """
        # Class attribute: a phrase used to generate encryption/decryption key.
        cls._passphrase = None
        # Class attribute: additional medatata that will be added to any instantiated object of that class.
        cls._class_metadata = {}
        # Class attribute: additional files that will be added to any instantiated object of that class.
        cls._class_file_content = {}

    @classmethod
    def get_passphrase(cls):
        """Property getter, returning the passphrase (scope: given class)."""
        return cls._passphrase

    @classmethod
    def set_passphrase(cls, passphrase: str):
        """Property setter, setting the passphrase (scope: given class).

        Args:
            passphrase (str): a phrase used to generate encryption/decryption key.
        """
        cls._passphrase = passphrase

    @classmethod
    def add_class_metadata(cls, force: bool = False, **metadata) -> None:
        """
        Method responsible for adding a new key-value pairs to class metadata.
        Those pairs will be next added to every instance of the class, so then can be used in recipes, e.g.
        added to every created archive.

        Args:
            force: Setting to True enables to overwrite the values for existing keys(DEFAULT:False)
            metadata: keyword args (key-value pairs) that will be added to metadata.

        Raises:
            KeyError when variable with a given key is already present in the cookbook metadata (unless forced=True)
        """
        # if self._state != Archive.States.SAVE_INIT:
        #    raise xxx

        # Iterate through named arguments one by one.
        for key, value in metadata.items():
            # Handle the key duplicate.
            if not force and key in cls._class_metadata.keys():
                raise KeyError("Variable `{}` already present in class metadata".format(key))
            # Add argument to metadata.
            cls._class_metadata[key] = value

    @classproperty
    def class_metadata(cls):
        """
        Property getter for class_metadata.

        Returns:
            Class metadata.
        """
        return cls._class_metadata

    @classmethod
    def add_class_file_content(
        cls,
        name: str,
        content: str,
        description: str,
        encryption: Union[bool, str] = False,
        binary: bool = False,
        **properties
    ) -> str:
        """
        Method responsible for adding new file (virual file, i.e. "name" with content and additional properties) to class.
        Those files will be passed to Archive on save.

        Args:
            name: Name of the file (can be relative/absolute path).
            content: Content of the file.
            description: Description of the content of the file
            encryption: flag indicating whether file will be encrypted or not.
            binary: flag indicating whether file will be binary or text (DEFAULT: False).
            properties: list of additional named params that will be added to file properties.

        Raises:
            KeyError when file with a given name is already present in the archive.
        """
        # Use file name as key (not the whole path).
        _, file_key = os.path.split(name)

        # Handle the key duplicate.
        if name in cls._class_file_content.keys():
            raise KeyError("File `{}` already present in the cookbook".format(name))

        # Add "default" file properties.
        props = {"description": description, "encryption": encryption, "binary": binary}
        # Iterate throught the additional properties.
        for key, value in properties.items():
            props[key] = value

        # Set properties.
        cls._class_file_content[file_key] = (content, props)

    def add_class_file_properties(self, name: str, force: bool = False, **properties) -> None:
        """
        Method responsible for adding a new key-value pairs to a given class file properties.

        Args:
            name: Name of the file (can be relative/absolute path).
            force: Setting to True enables to overwrite the values for existing keys (DEFAULT:False)
            properties: list of additional named params that will be added to file properties.

        Raises:
            KeyError: when file with a name is not present in the archive.
            KeyError: when property with a given name is already present in file properties.
        """
        # if self._state != Archive.States.SAVE_INIT:
        #    raise xxx

        # Use file name as key (not the whole path).
        _, file_key = os.path.split(name)

        # Check if file exists.
        if file_key not in self._class_file_content.keys():
            raise KeyError("Class file `{}` not present in the archive".format(file_key))

        # Iterate through named arguments one by one.
        for key, value in properties.items():
            # Handle the key duplicate.
            if not force and key in self._class_file_content[file_key][1].keys():
                raise KeyError("Variable `{}` already present in file `{}` properties".format(key, file_key))
            # Add/update properties.
            self._class_file_content[file_key][1][key] = value

    @classproperty
    def class_file_content(cls):
        """
        Property getter for class_file_content.

        Returns:
            Class dict with "files with content", key = filename: (content, properies).
        """
        return cls._class_file_content

    @classmethod
    def validate_archive(cls, restore_path: str, *v_conditions, obj_cls=None, **kv_conditions) -> bool:
        """Opens the indicated archive and tries to validate it by comparing the metadata agains the provided conditions.

        Args:
            restore_path: Path to the file/archive to be validated.
            obj_cls: Object class, if not None, it will be used as additional condition (DEFAULT: None)
            v_conditions: Conditions.
            kv_conditions: List of named conditions (key-values, key-Expressions)

        Returns:
            True if all conditions are fullfilled.
        """
        # Extend key-value conditions.
        conds = kv_conditions
        if obj_cls is not None:
            # Add target class to be validated.
            conds["obj_cls"] = generate_obj_cls(obj_cls)

        # Try to retrieve the manifest from the archive.
        manifest = Archive.restore_manifest(restore_path=restore_path)

        # Validate metadata using the provided conditions.
        return validate_metadata(*v_conditions, metadata=manifest["metadata"], **conds)

    @classmethod
    def get_metadata(cls, archive_path: str):
        """
        Elementary class method enabling the user to access metadata of the existing archive.

        Args:
            archive_path: Path to the archive.

        Returns:
            Dictionary showing the current content of the metadata object.
            Note changes to this object won't affect the original metadata.
        """
        # Open the archive, using the class encryption key.
        with Archive.restore_from(restore_path=archive_path, passphrase=cls.get_passphrase()) as effa:
            return effa.metadata

    @classmethod
    def get_files(cls, archive_path: str, **filter_properties):
        """
        Elementary class method enabling the user to access list of files of the existing archive.

        Args:
            archive_path: Path to the archive.
            filter_properties: key-value pairs that will be used to filter the files.

        Returns:
            Dictionary showing the files, in format (filename:properties), where properties is a dictionary, containing
            file properties, starting from `description`.
        """
        # Open the archive, using the class encryption key.
        with Archive.restore_from(restore_path=archive_path, passphrase=cls.get_passphrase()) as effa:
            return effa.artifacts.filter(filter_properties)

    @classmethod
    def get_file_properties(cls, archive_path: str, filename: str):
        """
        Elementary class method enabling the user to access properties of a given file in the existing archive.

        Args:
            archive_path: Path to the archive.
            filename: Name of the file in the archive.

        Returns:
            file properties, as dict of (key:value) pairs.
        """
        # Open the archive, using the class encryption key.
        with Archive.restore_from(restore_path=archive_path, passphrase=cls.get_passphrase()) as effa:
            # Retrieve the "properties" of the file in archive.
            return effa.artifacts[filename].properties

    @classmethod
    def get_file_content(cls, archive_path: str, filename: str, binary: bool = False):
        """
        Elementary class method enabling the user to access content of a given file in the existing archive.

        Args:
            archive_path: Path to the archive.
            filename: Name of the file in the archive.
            binary: Flag indicating that we want to read/return the content in the binary format (DEFAULT: False).

        Returns:
            File content, as a "raw" string or bytes (depending on binary).

        Raises:
            UnicodeDecodeError: When trying to return the binary file in the text form.
        """
        # Open the archive, using the class encryption key.
        with Archive.restore_from(restore_path=archive_path, passphrase=cls.get_passphrase()) as effa:
            if binary:
                content_callback = BinaryContentCallback()
            else:
                content_callback = StringContentCallback()
            # Return the content.
            return effa.artifacts[filename].get_content(content_callback=content_callback)
