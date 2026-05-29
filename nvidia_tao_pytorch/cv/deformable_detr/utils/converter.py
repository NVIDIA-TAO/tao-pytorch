# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for converting datasets to json"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import logging
import os
import random
import six
import numpy as np
from PIL import Image, ImageOps
import json
from json.decoder import JSONDecodeError

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.deformable_detr.utils.converter_lib import _shard, _shuffle
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import DEFAULT_TARGET_CLASS_MAPPING, DEFAULT_CLASS_LOOKUP_TABLE, get_categories

logger = logging.getLogger(__name__)


def create_info_images(image_id, file_name, img_size):
    """create image info in COCO format"""
    images = {
        "id": image_id,
        "file_name": file_name,
        "height": img_size[1],
        "width": img_size[0]
    }
    return images


def create_info_categories():
    """create categories info in COCO format"""
    categories = [{"supercategory": "person", "id": 1, "name": "person"},
                  {"supercategory": "face", "id": 2, "name": "face"},
                  {"supercategory": "bag", "id": 3, "name": "bag"}]
    return categories


def create_info_annotations(image_id, category_id, annotation_id, bbox, area):
    """create annotation info in COCO format"""
    annotation = {
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area,
        'iscrowd': 0
    }
    return annotation


class DatasetConverter(six.with_metaclass(ABCMeta, object)):
    """Converts an object detection dataset to Json.

    This class needs to be subclassed, and the convert() and
    create_example_proto() methods overridden to do the dataset
    conversion. Splitting of partitions to shards, shuffling and
    writing Json are implemented here.
    """

    @abstractmethod
    def __init__(self, data_root, input_source, num_partitions, num_shards,
                 output_dir, mapping_path=None):
        """Initialize the converter.

        Args:
            input_source (string): Dataset directory path relative to data root.
            num_partitions (int): Number of partitions (folds).
            num_shards (int): Number of shards.
            output_dir (str): Path for the output file.
            mapping_path (str): Path to a JSON file containing the class mapping.
        """
        if data_root is not None:
            self.img_root_dir = os.path.join(data_root, input_source)
        else:
            self.img_root_dir = input_source
        self.img_root_dir = os.path.abspath(self.img_root_dir)  # data_root/sqeuence_name
        self.input_source = os.path.basename(input_source)
        self.output_partitions = num_partitions
        self.output_shards = num_shards
        self.output_dir = output_dir

        self.output_filename = os.path.join(output_dir, self.input_source, self.input_source)

        # Make the output directory to write the shard.
        if not os.path.exists(output_dir):
            logger.info("Creating output directory {}".format(output_dir))
            os.makedirs(output_dir)

        if not os.path.exists(os.path.join(output_dir, self.input_source)):
            logger.info("Creating output directory {}".format(os.path.join(output_dir, self.input_source)))
            os.makedirs(os.path.join(output_dir, self.input_source))
        self.class_map = {}
        self.log_warning = {}

        # check the previous image id and annotation id for offset
        self.image_id_offset = 0
        self.annotation_id_offset = 0

        if mapping_path:
            if not os.path.exists(mapping_path):
                raise FileNotFoundError(f"Provided class mapping path {mapping_path} does not exist!")
            with open(mapping_path, "r", encoding='utf-8') as f:
                self.target_class_mapping = json.load(f)
            self.category_info, self.class_lookup_table = get_categories(self.target_class_mapping)
            logger.info("Load category mapping from {}".format(mapping_path))
        else:
            # Use the default values for PeopleNet
            self.target_class_mapping = DEFAULT_TARGET_CLASS_MAPPING
            self.class_lookup_table = DEFAULT_CLASS_LOOKUP_TABLE
            self.category_info = create_info_categories()
        logger.info("Category mapping: \n {}".format(self.class_lookup_table))
        # Set a fixed seed to get a reproducible sequence.
        random.seed(42)

    def convert(self):
        """Do the dataset conversion."""
        # Divide dataset into partitions and shuffle them.
        partitions = self._partition()
        _shuffle(partitions)

        # Shard and write the partitions to Json.
        global_image_id, global_ann_id = self._write_partitions(partitions)
        stats_filename = os.path.join(self.output_dir, 'stats.txt')
        with open(stats_filename, 'a+') as f:
            print('writing')
            print(stats_filename)
            out_str = "{},{},{}\n".format(self.input_source, global_image_id, global_ann_id)
            print(out_str)
            f.write(out_str)

        # Log how many objects per class got written in total.
        logger.info("Cumulative object statistics")
        s_logger = status_logging.get_status_logger()

        # Print out the class map
        log_str = "Class map. \nLabel in GT: Label in Json file "
        for key, value in six.iteritems(self.class_map):
            log_str += "\n{}: {}".format(key, value)
        logger.info(log_str)
        s_logger.write(message=log_str)
        note_string = (
            "For the dataset_config in the experiment_spec, "
            "please use labels in the Json file, while writing the classmap.\n"
        )
        logger.info(note_string)
        s_logger.write(message=note_string)

        logger.info("Json generation complete.")
        s_logger.write(
            status_level=status_logging.Status.RUNNING,
            message="Json generation complete."
        )

        # Save labels with error to a JSON file
        self._save_log_warnings()

    def _write_partitions(self, partitions):
        """Shard and write partitions into Json.

        Args:
            partitions (list): A list of list of frame IDs.

        Returns:
        """
        # Divide partitions into shards.
        sharded_partitions = _shard(partitions, self.output_shards)

        # Write .Json to disk for each partition and shard.
        stats_filename = os.path.join(self.output_dir, 'stats.txt')
        if os.path.exists(stats_filename):
            with open(stats_filename, 'r') as f:
                line = f.readlines()[-1].split(",")
            global_image_id = int(line[1])
            global_ann_id = int(line[2])
        else:
            global_image_id = 0
            global_ann_id = 0

        for p, partition in enumerate(sharded_partitions):
            for s, shard in enumerate(partition):
                shard_image_counter, shrad_ann_counter = self._write_shard(shard, p, s, global_image_id, global_ann_id)
                global_image_id += shard_image_counter
                global_ann_id += shrad_ann_counter
        return global_image_id, global_ann_id

    def _write_shard(self, shard, partition_number, shard_number, global_image_id, global_ann_id):
        """Write a single shard into the Json file.

        Note that the dataset-specific part is captured in function
        create_example_proto() which needs to be overridden for each
        specific dataset.

        Args:
            shard (list): A list of frame IDs for this shard.
            partition_number (int): Current partition (fold) index.
            shard_number (int): Current shard index.

        Returns:
        """
        logger.info('Writing partition {}, shard {}'.format(partition_number, shard_number))
        status_logging.get_status_logger().write(
            message='Writing partition {}, shard {}'.format(partition_number, shard_number)
        )
        output = self.output_filename

        if self.output_partitions != 0 and self.output_partitions != 1:
            output = '{}-fold-{:03d}-of-{:03d}'.format(output, partition_number,
                                                       self.output_partitions)
        if self.output_shards != 0:
            output = '{}-shard-{:05d}-of-{:05d}.json'.format(output, shard_number, self.output_shards)

        # # Store all the data for the shard.
        json_output = {
            "images": [],
            "annotations": [],
            "categories": self.category_info
        }
        image_count = 0
        ann_count = 0

        for frame_id in shard:
            image_file = os.path.join(self.input_source, self.images_dir, frame_id + self.extension)

            width, height = self._get_image_size(frame_id)
            shard_imgage_id = self.image_id_offset + global_image_id + image_count
            images_info = create_info_images(shard_imgage_id, image_file, (width, height))
            json_output["images"].append(images_info)

            # Create the Example with all labels for this frame_id.
            shard_ann_id = self.annotation_id_offset + global_ann_id + ann_count
            json_output, shard_ann_count = self._create_info_dict(json_output, frame_id, shard_imgage_id, shard_ann_id)
            image_count = image_count + 1
            ann_count = ann_count + shard_ann_count
        with open(output, 'w+') as outfile:
            try:
                json.dump(json_output, outfile)
            except JSONDecodeError:
                pass

        return image_count, ann_count

    def _get_image_size(self, frame_id):
        """Read image size from the image file, image sizes vary in KITTI."""
        image_file = os.path.join(self.img_root_dir, self.images_dir, frame_id + self.extension)
        width, height = ImageOps.exif_transpose(Image.open(image_file)).size

        return width, height

    @abstractmethod
    def _partition(self):
        """Return dataset partitions."""
        pass

    @abstractmethod
    def _create_info_dict(self, json_output, frame_id, image_id, ann_id):
        """Generate the example for this frame."""
        pass

    def _save_log_warnings(self):
        """Store out of bound bounding boxes to a json file."""
        if self.log_warning:
            logger.info("Writing the log_warning.json")
            with open(f"{self.output_dir}_warning.json", "w") as f:
                json.dump(self.log_warning, f, indet=2)
            logger.info("There were errors in the labels. Details are logged at"
                        " %s_waring.json", self.output_dir)


class KITTIConverter(DatasetConverter):
    """Converts a KITTI detection dataset to jsons."""

    def __init__(self, data_root, input_source, num_partitions, num_shards,
                 output_dir,
                 image_dir_name=None,
                 label_dir_name=None,
                 extension='.png',
                 partition_mode='random',
                 val_split=None,
                 mapping_path=None):
        """Initialize the converter.

        Args:
            input_source (string): Dataset directory path relative to data root.
            num_partitions (int): Number of partitions (folds).
            num_shards (int): Number of shards.
            output_dir (str): Path for the output file.
            image_dir_name (str): Name of the subdirectory containing images.
            label_dir_name (str): Name of the subdirectory containing the label files for the
                respective images in image_dir_name
            extension (str): Extension of the images in the dataset directory.
            partition_mode (str): Mode to partitition the dataset. We only support sequence or
                random split mode. In the sequence mode, it is mandatory to instantiate the
                kitti sequence to frames file. Also, any arbitrary number of partitions maybe
                used. However, for random split, the sequence map file is ignored and only 2
                partitions can every be used. Here, the data is divided into two folds
                    1. validation fold
                    2. training fold
                Validation fold (defaults to fold=0) contains val_split% of data, while train
                fold contains (100-val_split)% of data.
            val_split (int): Percentage split for validation. This is used with the random
                partition mode only.
            mapping_path (str): Path to a JSON file containing the class mapping.
                If not specified, default to DEFAULT_TARGET_CLASS_MAPPING
        """
        super(KITTIConverter, self).__init__(
            data_root=data_root,
            input_source=input_source,
            num_partitions=num_partitions,
            num_shards=num_shards,
            output_dir=output_dir,
            mapping_path=mapping_path)

        # KITTI defaults.
        self.images_dir = image_dir_name
        self.labels_dir = label_dir_name
        self.extension = extension
        self.partition_mode = partition_mode
        self.val_split = val_split / 100.

        # check the previous image id and annotation id for offset
        self.image_id_offset = 0
        self.annotation_id_offset = 0

    def _partition(self):
        """Partition KITTI dataset to self.output_partitions partitions based on sequences.

        The following code is a modified version of the KITTISplitter class in Rumpy.

        Returns:
            partitions (list): A list of lists of frame ids, one list per partition.
        """
        logger.debug("Generating partitions")
        s_logger = status_logging.get_status_logger()
        s_logger.write(message="Generating partitions")
        partitions = [[] for _ in six.moves.range(self.output_partitions)]
        if self.partition_mode is None and self.output_partitions == 1:
            images_root = os.path.join(self.img_root_dir, self.images_dir)
            images_list = [os.path.splitext(imfile)[0] for imfile in os.listdir(images_root)
                           if imfile.endswith(self.extension)]
            partitions[0].extend(images_list)
        elif self.partition_mode == 'random':
            assert self.output_partitions == 2, "Invalid number of partitions ({}) "\
                   "for random split mode.".format(self.output_partitions)
            assert 0 <= self.val_split < 1, (
                "Validation split must satisfy the criteria, 0 <= val_split < 100. "
            )
            images_root = os.path.join(self.img_root_dir, self.images_dir)
            images_list = [os.path.splitext(imfile)[0] for imfile in os.listdir(images_root)
                           if imfile.endswith(self.extension)]
            total_num_images = len(images_list)
            num_val_images = (int)(self.val_split * total_num_images)
            logger.debug("Validation percentage: {}".format(self.val_split))
            partitions[0].extend(images_list[:num_val_images])
            partitions[1].extend(images_list[num_val_images:])
            for part in partitions:
                random.shuffle(part)
            logger.info("Num images in\nTrain: {}\tVal: {}".format(len(partitions[1]),
                                                                   len(partitions[0])))

            s_logger.kpi = {
                "num_images": total_num_images
            }
            s_logger.write(
                message="Num images in\nTrain: {}\tVal: {}".format(
                    len(partitions[1]),
                    len(partitions[0])
                )
            )

            if self.val_split == 0:
                logger.info("Skipped validation data...")
                s_logger.write(message="Skipped validation data.")
            else:
                validation_note = (
                    "Validation data in partition 0. Hence, while choosing the validation"
                    "set during training choose validation_fold 0."
                )
                logger.info(validation_note)
                s_logger.write(message=validation_note)
        else:
            raise NotImplementedError("Unknown partition mode. Please stick to either "
                                      "random or sequence")

        return partitions

    def _create_info_dict(self, json_output, frame_id, image_id, global_ann_id):
        """Generate the example proto for this frame.

        Args:
            frame_id (string): The frame id.

        Returns:
            example : An Example containing all labels for the frame.
        """
        # Create proto for the training example. Populate with frame attributes.
        json_output = self._process_info(json_output, frame_id, image_id, global_ann_id)
        return json_output

    def _process_info(self, json_output, frame_id, image_id, global_ann_id):
        """Add KITTI target features such as bbox to the Example protobuf.

        Reads labels from KITTI txt files with following fields:
        (From Kitti devkit's README)
        1    type         Describes the type of object: 'Car', 'Van',
                          'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist',
                          'Tram', 'Misc' or 'DontCare'
        1    truncated    Float from 0 (non-truncated) to 1 (truncated),
                          where truncated refers to the object leaving image
                          boundaries
        1    occluded     Integer (0,1,2,3) indicating occlusion state:
                          0 = fully visible, 1 = partly occluded
                          2 = largely occluded, 3 = unknown
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    bbox         2D bounding box of object in the image (0-based
                          index): contains left, top, right, bottom pixel
                          coordinates
        3    dimensions   3D object dimensions: height, width, length (in
                          meters)
        3    location     3D object location x,y,z in camera coordinates (in
                          meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates
                          [-pi..pi]

        Args:
            example (tf.train.Example): The Example protobuf for this frame.
            frame_id (string): Frame id.
        """
        # reads the labels as a list of tuples
        label_file = os.path.join(self.img_root_dir, self.labels_dir, '{}.txt'.format(frame_id))
        labels = np.genfromtxt(label_file, dtype=None).tolist()
        if isinstance(labels, tuple):
            labels = [labels]

        ann_counter = 0
        for label in labels:
            assert len(label) == 15, 'Ground truth kitti labels should have only 15 fields.'
            x1 = label[4]
            y1 = label[5]
            x2 = label[6]
            y2 = label[7]

            # Map object classes as they are in the dataset to target classes of the model
            # self.class_map[label[0]] = label[0].lower()
            object_class = label[0].lower().decode('utf-8')
            mapped_class = self.target_class_mapping.get(object_class, 'unknown')
            self.class_map[label[0]] = mapped_class
            if mapped_class == 'unknown':
                continue
            category_id = self.class_lookup_table[mapped_class]

            # Check to make sure the coordinates are 'ltrb' format.
            error_string = "Top left coordinate must be less than bottom right."\
                           "Error in object {} of label_file {}. \nCoordinates: "\
                           "x1 = {}, x2 = {}, y1: {}, y2: {}".format(labels.index(label),
                                                                     label_file,
                                                                     x1, x2, y1, y2)
            if not (x1 < x2 and y1 < y2):
                logger.debug(error_string)
                logger.debug("Skipping this object")
                continue

            # coco bbox format (x1, y1, w, h)
            bbox = [x1, y1, (x2 - x1), (y2 - y1)]
            area = bbox[2] * bbox[3]
            annotation_id = global_ann_id + ann_counter
            annotation_info = create_info_annotations(image_id, category_id, annotation_id, bbox, area)
            ann_counter = ann_counter + 1
            json_output["annotations"].append(annotation_info)

        return json_output, ann_counter
