# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import numpy as np
import os
import time
import itertools
import json
from contextlib import ExitStack
import gradio as gr
import torch

from detectron2.config import LazyConfig, instantiate
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.evaluation import inference_context
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer, random_color
from nvidia_tao_pytorch.cv.odise.data.datasets.openseg_categories import ADE20K_150_CATEGORIES
from PIL import Image

from nvidia_tao_pytorch.cv.odise.checkpoint import ODISECheckpointer
from nvidia_tao_pytorch.cv.odise.config import instantiate_odise
from nvidia_tao_pytorch.cv.odise.data import get_openseg_labels
from nvidia_tao_pytorch.cv.odise.gradio.pano_wrapper import OpenPanopticInference

setup_logger()
logger = setup_logger(name="odise")

COCO_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 1
]
COCO_THING_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 1]
COCO_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 0
]
COCO_STUFF_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 0]

ADE_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 1
]
ADE_THING_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 1]
ADE_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 0
]
ADE_STUFF_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 0]

LVIS_CLASSES = get_openseg_labels("lvis_1203", True)
# use beautiful coco colors
LVIS_COLORS = list(
    itertools.islice(itertools.cycle([c["color"] for c in COCO_CATEGORIES]), len(LVIS_CLASSES))
)


class VisualizationDemo(object):
    def __init__(self, model, metadata, aug, instance_mode=ColorMode.IMAGE):
        """
        Args:
            model (nn.Module):
            metadata (MetadataCatalog): image metadata.
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.model = model
        self.metadata = metadata
        self.aug = aug
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

    def predict(self, original_image, vocab=None):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        height, width = original_image.shape[:2]
        aug_input = T.AugInput(original_image, sem_seg=None)
        self.aug(aug_input)
        image = aug_input.image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        predictions = self.model([inputs], vocab, self.metadata)[0]
        return predictions

    def run_on_image(self, path, demo_classes=None, extra_classes=None, extra_ids=None, only_extra=False):
        """
        Args:
            path (str): path to the original image
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        # use PIL, to be consistent with evaluation
        image = utils.read_image(path, format="RGB")
        start_time = time.time()
        predictions = self.predict(image, demo_classes)
        print(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            panoptic_seg.to(self.cpu_device)

            new_segments_info = []
            if extra_ids:
                for i in segments_info:
                    if i['category_id'] in extra_ids:
                        new_segments_info.append(i)

            vis_output = visualizer.draw_panoptic_seg(
                panoptic_seg.cpu(), new_segments_info if extra_ids and only_extra else segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output


title = "ODISE"
description = """
Gradio demo for ODISE: Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models. \n
You may click on of the examples or upload your own image. \n

ODISE could perform open vocabulary segmentation, you may input more classes (separate by comma).
The expected format is 'a1,a2;b1,b2', where a1,a2 are synonyms vocabularies for the first class. 
The first word will be displayed as the class name.
"""  # noqa

article = """
<p style='text-align: center'><a href='https://arxiv.org/abs/2303.04803' target='_blank'>Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models</a> | <a href='https://github.com/NVlab/ODISE' target='_blank'>Github Repo</a></p>
"""  # noqa


examples = [
    [
        "examples/coco.jpg",
        "black pickup truck, pickup truck; blue sky, sky",
        ["COCO (133 categories)", "ADE (150 categories)", "LVIS (1203 categories)"],
    ],
    [
        "examples/ade.jpg",
        "luggage, suitcase, baggage;handbag",
        ["LVIS (1203 categories)"],
    ],
    [
        "examples/ego4d.jpg",
        "faucet, tap; kitchen paper, paper towels",
        ["COCO (133 categories)"],
    ],
]


def build_demo_classes_and_metadata(vocab, label_list):
    extra_classes = []

    if vocab:
        for words in vocab.split(";"):
            extra_classes.append([word.strip() for word in words.split(",")])

    demo_thing_classes = []
    demo_stuff_classes = []
    demo_thing_colors = []
    demo_stuff_colors = []
    only_extra = False
    if not label_list:
        label_list = ["COCO", "LVIS", "ADE"]
        only_extra = True
    if any("COCO" in label for label in label_list):
        demo_thing_classes += COCO_THING_CLASSES
        demo_stuff_classes += COCO_STUFF_CLASSES
        demo_thing_colors += COCO_THING_COLORS
        demo_stuff_colors += COCO_STUFF_COLORS
    if any("ADE" in label for label in label_list):
        demo_thing_classes += ADE_THING_CLASSES
        demo_stuff_classes += ADE_STUFF_CLASSES
        demo_thing_colors += ADE_THING_COLORS
        demo_stuff_colors += ADE_STUFF_COLORS
    if any("LVIS" in label for label in label_list):
        demo_thing_classes += LVIS_CLASSES
        demo_thing_colors += LVIS_COLORS

    real_extra_classes = []
    extra_classes_ids = set()
    if extra_classes:
        all_classes = demo_thing_classes + demo_stuff_classes
        m = {}
        for i, c in enumerate(all_classes):
            for j in c:
                if j not in m:
                    m[j] = [i]
                else:
                    m[j].append(i)

        for ex in extra_classes: # [[], []]
            if ex:
                included = False
                for e in ex:
                    if e in m:
                        included = True
                        idx = m[e]
                        extra_classes_ids = extra_classes_ids.union(set(idx))
                    elif e not in m and included:
                        all_classes[idx].append(e)
                        extra_classes_ids = extra_classes_ids.union(set(idx))
                if not included:
                    real_extra_classes.append(ex)

    demo_thing_classes = real_extra_classes + demo_thing_classes
    extra_colors = [random_color(rgb=True, maximum=1) for _ in range(len(real_extra_classes))]
    demo_thing_colors = extra_colors + demo_thing_colors

    extra_classes_ids = list(map(lambda x: x+len(real_extra_classes), list(extra_classes_ids)))
    extra_classes_ids += list(range(len(real_extra_classes)))

    MetadataCatalog.pop("odise_demo_metadata", None)
    demo_metadata = MetadataCatalog.get("odise_demo_metadata")
    demo_metadata.thing_classes = [c[0] for c in demo_thing_classes]
    demo_metadata.stuff_classes = [
        *demo_metadata.thing_classes,
        *[c[0] for c in demo_stuff_classes],
    ]
    demo_metadata.thing_colors = demo_thing_colors
    demo_metadata.stuff_colors = demo_thing_colors + demo_stuff_colors
    demo_metadata.stuff_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.stuff_classes))
    }
    demo_metadata.thing_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.thing_classes))
    }
    demo_metadata.thing_dataset_id_to_contiguous_id_value_tensor = \
        torch.tensor(list(demo_metadata.thing_dataset_id_to_contiguous_id.values()), device='cuda')

    demo_classes = demo_thing_classes + demo_stuff_classes

    return demo_classes, demo_metadata, extra_classes, extra_classes_ids, only_extra


def inference(image_path, vocab, label_list, model_name=None):

    logger.info("building class names")

    demo_classes, demo_metadata, extra_classes, extra_ids, only_extra = build_demo_classes_and_metadata(vocab, label_list)
    if model_name is None:
        model_name = "ODISE(Label)"
    with ExitStack() as stack:
        logger.info(f"loading model {model_name}")

        stack.enter_context(inference_context(inference_model))
        stack.enter_context(torch.no_grad())

        demo = VisualizationDemo(inference_model, demo_metadata, aug)
        _, visualized_output = demo.run_on_image(image_path, demo_classes, extra_classes, extra_ids, only_extra)
        return Image.fromarray(visualized_output.get_image())

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ODISE Gradio app."
    )
    parser.add_argument('-p', '--checkpoint_path', type=str, help="ODISE model path.")
    args = parser.parse_args()
    models = {}
    spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = args.checkpoint_path

    cfg = LazyConfig.load(os.path.join(spec_root, "config/common/category_odise.py"))
    cfg.model.overlap_threshold = 0
    cfg.model.precision = 'fp16'
    cfg.model.is_inference = True
    cfg.train.device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all_rng(42)

    dataset_cfg = cfg.dataloader.test
    wrapper_cfg = cfg.dataloader.wrapper
    aug = instantiate(dataset_cfg.mapper).augmentations
    model = instantiate_odise(cfg.model)

    model.to(cfg.train.device)
    ODISECheckpointer(model).load(checkpoint_path)
    models["ODISE(Label)"] = model

    inference_model = OpenPanopticInference(
        model=models["ODISE(Label)"],
        labels=None,
        metadata=None,
        semantic_on=False,
        instance_on=False,
        panoptic_on=True,
    )

    block = gr.Blocks(title=title).queue()
    with block as demo:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>" + title + "</h1>")
        gr.Markdown(description)
        input_components = []
        output_components = []

        with gr.Row():
            output_image_gr = gr.Image(label="Panoptic Segmentation", type="pil")
            output_components.append(output_image_gr)

        with gr.Row():  # .style(equal_height=True, mobile_collapse=True)
            with gr.Column(scale=3, variant="panel") as input_component_column:
                input_image_gr = gr.Image(type="filepath")
                model_name_gr = gr.Dropdown(
                    label="Model", choices=["ODISE(Label)"], value="ODISE(Label)"
                )
                extra_vocab_gr = gr.Textbox(value="", label="Extra Vocabulary")
                category_list_gr = gr.CheckboxGroup(
                    choices=["COCO (133 categories)", "ADE (150 categories)", "LVIS (1203 categories)"],
                    value=["COCO (133 categories)", "ADE (150 categories)", "LVIS (1203 categories)"],
                    label="Category to use",
                )
                input_components.extend([input_image_gr, extra_vocab_gr, category_list_gr])

            with gr.Column(scale=2):
                examples_handler = gr.Examples(
                    examples=examples,
                    inputs=[c for c in input_components if not isinstance(c, gr.State)],
                    outputs=[c for c in output_components if not isinstance(c, gr.State)],
                    fn=inference,
                    cache_examples=torch.cuda.is_available(),
                    examples_per_page=5,
                )
                with gr.Row():
                    clear_btn = gr.Button("Clear")
                    submit_btn = gr.Button("Submit", variant="primary")

        gr.Markdown(article)

        submit_btn.click(
            inference,
            input_components,
            output_components,
            api_name="predict",
            scroll_to_output=True,
        )  # image_path, vocab, label_list, model_name

        clear_btn.click(
            None,
            [],
            (input_components + output_components + [input_component_column]),
            js=f"""() => {json.dumps(
                        [component.cleared_value if hasattr(component, "cleared_value") else None
                        for component in input_components + output_components]
                    )}
                    """,
        )

    # demo.launch(share=True, debug=True, server_name='0.0.0.0', server_port=8889)
    block.launch(debug=True, server_name='0.0.0.0', server_port=8889, inline=False)
