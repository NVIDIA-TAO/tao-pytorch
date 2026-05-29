# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gradio app for SegIC."""

import argparse
import os
import time
import datetime
import json
from contextlib import ExitStack

from detectron2.utils.logger import setup_logger
import gradio as gr

from nvidia_tao_pytorch.cv.segic.modeling.inference import SegicTRTInferencer
from nvidia_tao_pytorch.cv.segic.gradio.onnx_to_trt import TRTEngineBuilder

setup_logger()
logger = setup_logger(name="segic")

TITLE = "SegIC"
DESCRIPTION = """
Gradio demo for SegIC: Visual Prompt In-Context Semantic Segmentation with SegIC models.
You may click one of the examples or upload your own images.

SegIC could perform in-context segmentation. By taking in visual prompt, SegIC will 
segment target items in test images. The prompt consists of an image, a mask, and an 
optional meta desciotion. The prompt features can be extracted and stored by clicking 
`Extract Features`. After that, you can upload test images and click `Predict` to get 
segmented test image. By preserving the prompt features, you can get consistent 
segmentation results for different test images.

Meta description is optional. If no meta description is given, by default the meta 
description would be `instance`. This works fine if the visual prompt (image + mask) 
is good enough to guide the segmentation.
"""  # noqa

ARTICLE = """
<p style='text-align: center'><a href='https://arxiv.org/abs/2311.14671' target='_blank'>SEGIC: Unleashing the Emergent Correspondence for In-Context Segmentation</a></p>
"""  # noqa


prompt_examples = [
    [
        "examples/prompt_imgs/example_prompt1.jpg",
        "examples/masks/example_prompt1.png",
        "bandaid variety",
    ],
    [
        "examples/prompt_imgs/person.jpg",
        "examples/masks/person.png",
        "person",
    ],
    [
        "examples/prompt_imgs/bicycle.jpg",
        "examples/masks/bicycle.png",
        "bicycle",
    ],
    [
        "examples/prompt_imgs/bird.jpg",
        "examples/masks/bird.png",
        "bird",
    ],
    [
        "examples/prompt_imgs/cat.jpg",
        "examples/masks/cat.png",
        "cat",
    ],
]

test_examples = [
    ["examples/tgt_imgs/test1_1.jpg"],
    ["examples/tgt_imgs/test1_2.jpg"],
    ["examples/tgt_imgs/person_test1.jpg"],
    ["examples/tgt_imgs/person_test2.jpg"],
    ["examples/tgt_imgs/bicycle_test1.jpg"],
    ["examples/tgt_imgs/bicycle_test2.jpg"],
    ["examples/tgt_imgs/bird_test1.jpg"],
    ["examples/tgt_imgs/bird_test2.jpg"],
    ["examples/tgt_imgs/cat_test1.jpg"],
    ["examples/tgt_imgs/cat_test2.jpg"],
]


def exatract_features(image_input, seg_input, meta_input, model_name=None):
    """
    Extracts prompt features from prompt image, mask and meta description.

    Args:
        image_input (str): Path to the prompt image.
        seg_input (str): Path to the prompt mask.
        meta_input (str): Meta description of the in-context object.
        model_name (str): Model name for prompt feature extraction.

    Returns:
        meta_desc(str): Meta description of the prompt feature extraction.
    """
    logger.info("extracting prompt features")
    if model_name is None:
        model_name = "SegIC(DINOv2-L-v1)"
    with ExitStack():
        logger.info("extracting prompt features by %s", model_name)
        start_time = time.time()
        # fixed batch size to 4
        trt_inferencer.extract_prompt_features(image_input, seg_input, meta_input, batch_size=4)
        finish_time = time.time()
        print(
            "Prompt feature extracted in {:.2f}s".format(
                finish_time - start_time,
            )
        )
        utc_time = datetime.datetime.utcfromtimestamp(finish_time).strftime('%Y-%m-%d %H:%M:%S')
        meta_desc = f"Prompt feature extraction completed at {utc_time} UTC."
        image_name = os.path.basename(image_input)
        seg_name = os.path.basename(seg_input)
        meta_desc += f"\nPrompt image: {image_name}"
        meta_desc += f"\nprompt mask: {seg_name}"
        meta_desc += f"\nmeta description: {meta_input}"
        return meta_desc


def inference(target_input):
    """
    Runs inference on target images using the extracted prompt features.

    Args:
        target_input (str): Path to the target image.

    Returns:
        np.ndarray: Image with segmentation mask overlayed.
    """
    logger.info("running inference")
    with ExitStack():
        start_time = time.time()
        visualized_output = trt_inferencer.inference_batch(target_input)
        print(
            "{}: finished in {:.2f}s".format(
                target_input,
                time.time() - start_time,
            )
        )
        return visualized_output[0]  # only one image


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="SegIC Gradio app."
    )
    parser.add_argument('-tp', '--trt_engine_prompt_extractor', type=str, default=None,
                        help="SegIC prompt extractor TRT Engine model path.")
    parser.add_argument('-t', '--trt_engine', type=str, default=None,
                        help="SegIC TRT Engine model path.")
    parser.add_argument('-op', '--onnx_path_prompt_extractor', type=str,
                        help="SegIC prompt extractor ONNX file path.")
    parser.add_argument('-o', '--onnx_path', type=str, help="SegIC ONNX file path.")
    args = parser.parse_args()
    models = {}
    if args.trt_engine and args.trt_engine_prompt_extractor and \
            os.path.exists(args.trt_engine) and os.path.exists(args.trt_engine_prompt_extractor):
        feature_extract_trt = args.trt_engine_prompt_extractor
        segic_trt = args.trt_engine
    elif args.onnx_path and args.onnx_path_prompt_extractor and \
            os.path.exists(args.onnx_path) and os.path.exists(args.onnx_path_prompt_extractor):

        feature_extract_trt = "prompt_feature_extract.trt"
        segic_trt = "segic.trt"  # save trt engines to current directory

        trt_builder = TRTEngineBuilder(
            segic_onnx_path=args.onnx_path,
            prompt_feature_extract_onnx_path=args.onnx_path_prompt_extractor,
            segic_trt_engine_path=segic_trt,
            prompt_feature_extract_trt_engine_path=feature_extract_trt
        )

        # build fp16 trt engines
        trt_builder.build_engine(min_bz=1, opt_bz=4, max_bz=4, fp16=True)  # hard coded batch size

    else:
        raise ValueError("Please provide valid trt engine paths or onnx paths for prompt " +
                         "extractor and segic model.")

    batch_size = 4   # limited by trt engine
    inference_size = (896, 896)

    trt_inferencer = SegicTRTInferencer(
        feature_extract_trt,
        segic_trt,
        inference_size)

    block = gr.Blocks(title=TITLE).queue()
    with block as demo:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>" + TITLE + "</h1>")
        gr.Markdown(DESCRIPTION)
        input_components = []
        output_components = []

        with gr.Column():
            with gr.Row() as input_component_row:
                prompt_image_gr = gr.Image(type="filepath", label="Prompt Image")
                prompt_mask_gr = gr.Image(type="filepath", label="Prompt Mask")
                input_image_gr = gr.Image(type="filepath", label="Target Image")
                output_image_gr = gr.Image(label="In-Context Segmentation", type="numpy",
                                           image_mode="RGB")

            with gr.Column(scale=3, variant="panel") as meta_component_column:
                model_name_gr = gr.Dropdown(
                    label="Model", choices=["SegIC(DINOv2-L-v1)"], value="SegIC(DINOv2-L-v1)"
                )
                meta_desc_gr = gr.Textbox(value="", label="Meta Description")

            with gr.Column(scale=3) as status_column:
                feature_extract_gr = gr.Textbox(value="Feature not extracted", label="Status")

            input_components.extend([prompt_image_gr, prompt_mask_gr, meta_desc_gr])

            with gr.Column(scale=2):

                with gr.Row():
                    examples_handler = gr.Examples(
                        examples=prompt_examples,
                        inputs=[c for c in input_components if not isinstance(c, gr.State)],
                        outputs=[feature_extract_gr],
                        fn=exatract_features,
                        cache_examples=False,
                        examples_per_page=5,
                        label="Prompt examples"
                    )

                    examples_handler2 = gr.Examples(
                        examples=test_examples,
                        inputs=[input_image_gr],
                        outputs=[],
                        fn=inference,
                        cache_examples=False,
                        examples_per_page=12,
                        label="Test examples"
                    )
                output_components.extend([feature_extract_gr, output_image_gr])

                with gr.Row():
                    clear_btn = gr.Button("Clear")
                    feature_extract_btn = gr.Button("Extract Features", variant="primary")
                    predict_btn = gr.Button("Predict", variant="primary")

        gr.Markdown(ARTICLE)

        feature_extract_btn.click(
            exatract_features,
            [prompt_image_gr, prompt_mask_gr, meta_desc_gr],
            [feature_extract_gr],
            api_name="extract_features",
            scroll_to_output=False,
        )  # mage_input, seg_input, meta_input

        predict_btn.click(
            inference,
            [input_image_gr],
            [output_image_gr],
            api_name="predict",
            scroll_to_output=True,
        )  # target_input

        clear_btn.click(
            None,
            [],
            (input_components + [input_image_gr] + output_components +
             [input_component_row, meta_component_column]),
            js=f"""() => {json.dumps(
                [component.cleared_value if hasattr(component, "cleared_value") else None
                 for component in input_components + output_components])}""",
        )

    block.launch(debug=True, server_name='0.0.0.0', server_port=8890, inline=False)
