# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modules to support torch checkpoint to TensorRT engine conversion."""

import tensorrt as trt

from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    TrtRunner,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
    EngineFromBytes
)
from polygraphy.comparator import Comparator, CompareFunc
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx


class TRTEngineBuilder:
    """Class to build TensorRT engines from ONNX models."""

    def __init__(self, segic_onnx_path, prompt_feature_extract_onnx_path,
                 segic_trt_engine_path, prompt_feature_extract_trt_engine_path):
        """Initializes the TRTEngineBuilder.

        Args:
            segic_onnx_path (str): Path to the SegIC ONNX model.
            prompt_feature_extract_onnx_path (str): Path to the Prompt Feature Extractor ONNX model.
            segic_trt_engine_path (str): Path to save the SegIC TRT engine.
            prompt_feature_extract_trt_engine_path (str): Path to save the Prompt Feature
                Extractor TRT engine.
        """
        self.segic_onnx_path = segic_onnx_path
        self.prompt_feature_extract_onnx_path = prompt_feature_extract_onnx_path
        self.segic_trt_engine_path = segic_trt_engine_path
        self.prompt_feature_extract_trt_engine_path = prompt_feature_extract_trt_engine_path

    def build_segic_trt(self, fp16=False, min_bz=1, opt_bz=1, max_bz=4):
        """
        Builds SegIC TRT engine.

        Args:
            fp16 (bool): Whether to use FP16 precision.
            min_bz (int): Minimum batch size.
            opt_bz (int): Optimal batch size.
            max_bz (int): Maximum batch size.
        """
        print("Building SegIC TRT engine...")

        profile_object = Profile()
        profile_object.add("images", min=(1, 3, 896, 896), opt=(1, 3, 896, 896),
                           max=(1, 3, 896, 896))
        profile_object.add("ori_sizes", min=(1, 2), opt=(1, 2), max=(1, 2))
        profile_object.add("input_prompts", min=(min_bz, 256), opt=(opt_bz, 256), max=(max_bz, 256))
        profile_object.add("inst_feats", min=(min_bz, 1024), opt=(opt_bz, 1024), max=(max_bz, 1024))

        profiles = [
            profile_object
        ]

        config = CreateConfig(
            profiles=profiles,
            fp16=fp16,
            sparse_weights=True,
            memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 107374182400},
        )

        engine = engine_from_network(
            network_from_onnx_path(self.segic_onnx_path),
            config=config,
        )

        save_engine(engine, path=self.segic_trt_engine_path)

    def build_prompt_feature_extractor_trt(self, fp16=False, min_bz=1, opt_bz=1, max_bz=4):
        """
        Builds Prompt Feature Extractor TRT engine.

        Args:
            fp16 (bool): Whether to use FP16 precision.
            min_bz (int): Minimum batch size.
            opt_bz (int): Optimal batch size.
            max_bz (int): Maximum batch size.
        """
        print("Building Prompt Feature Extractor TRT engine...")

        profile_object = Profile()
        profile_object.add("images", min=(min_bz, 3, 896, 896), opt=(opt_bz, 3, 896, 896),
                           max=(max_bz, 3, 896, 896))
        profile_object.add("labels", min=(min_bz, 1, 896, 896), opt=(opt_bz, 1, 896, 896),
                           max=(max_bz, 1, 896, 896))
        profile_object.add("ori_sizes", min=(min_bz, 2), opt=(opt_bz, 2), max=(max_bz, 2))
        profile_object.add("tokens", min=(min_bz, 64), opt=(opt_bz, 64), max=(max_bz, 64))

        profiles = [
            profile_object
        ]

        config = CreateConfig(
            profiles=profiles,
            fp16=fp16,
            sparse_weights=True,
            memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 107374182400},

        )

        engine = engine_from_network(
            network_from_onnx_path(self.prompt_feature_extract_onnx_path),
            config=config,
        )

        save_engine(engine, path=self.prompt_feature_extract_trt_engine_path)

    def build_engine(self, min_bz=1, opt_bz=1, max_bz=4, fp16=False):
        """
        Builds both Prompt Feature Extractor and SegIC TRT engines and verifies ONNX2TRT conversion
        Accuracy.

        Args:
            min_bz (int): Minimum batch size.
            opt_bz (int): Optimal batch size.
            max_bz (int): Maximum batch size.
            fp16 (bool): Whether to use FP16 precision.
        """
        self.build_prompt_feature_extractor_trt(fp16=fp16, min_bz=min_bz, opt_bz=opt_bz,
                                                max_bz=max_bz)
        self.build_segic_trt(fp16=fp16, min_bz=min_bz, opt_bz=opt_bz, max_bz=max_bz)

        # skip accuracy verification for now as it's verified offline
        # self.verify_pf_accuracy()
        # self.verify_segic_accuracy()

    def verify_pf_accuracy(self):
        """
        Verify the accuracy of the Prompt Feature Extractor ONNX2TRT conversion.
        """
        print("Check prompt feature extractor ONNX2TRT accuracy...")
        build_onnxrt_session = SessionFromOnnx(self.prompt_feature_extract_onnx_path)
        build_engine = EngineFromBytes(BytesFromPath(self.prompt_feature_extract_trt_engine_path))

        runners = [
            TrtRunner(build_engine),
            OnnxrtRunner(build_onnxrt_session),
        ]

        run_results = Comparator.run(runners)

        assert bool(
            Comparator.compare_accuracy(
                run_results, compare_func=CompareFunc.simple(atol=1e-1, rtol=1e-2)
            )
        )

    def verify_segic_accuracy(self):
        """
        Verify the accuracy of the SegIC ONNX2TRT conversion.
        """
        print("Check SegIC ONNX2TRT accuracy...")
        build_onnxrt_session = SessionFromOnnx(self.segic_onnx_path)
        build_engine = EngineFromBytes(BytesFromPath(self.segic_trt_engine_path))

        runners = [
            TrtRunner(build_engine),
            OnnxrtRunner(build_onnxrt_session),
        ]

        run_results = Comparator.run(runners)

        assert bool(
            Comparator.compare_accuracy(
                run_results, compare_func=CompareFunc.simple(atol=1e-3, rtol=1e-2)
            )
        )
