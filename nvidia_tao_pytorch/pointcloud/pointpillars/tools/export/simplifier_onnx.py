# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ONNX model simplifications."""
import numpy as np
import onnx
import onnx_graphsurgeon as gs


@gs.Graph.register()
def replace_with_scatter(self, inputs, outputs):
    """Insert Scatter plugin."""
    # Disconnect output nodes of all input tensors
    dense_shape = outputs[0].shape[2:4]
    for inp in inputs:
        inp.outputs.clear()
    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()
    # Insert the new node.
    attrs = {"dense_shape": dense_shape}
    return self.layer(
        op="PillarScatterPlugin",
        name="PillarScatterPlugin_0",
        inputs=inputs,
        outputs=outputs,
        attrs=attrs
    )


def recursive_set_shape(node):
    """Recursively set shape."""
    for ot in node.outputs:
        ot.shape = tuple(["batch"] + list(ot.shape[1:]))
        for on in ot.outputs:
            recursive_set_shape(on)


def simplify_onnx(onnx_model, cfg):
    """Simplify ONNX model."""
    graph = gs.import_onnx(onnx_model)
    tmap = graph.tensors()
    MAX_VOXELS = tmap["input"].shape[1]
    MAX_POINTS = tmap["input"].shape[2]
    # (point_feats, cluster, center)
    NUM_FEATS = tmap["input"].shape[3] + 3 + 3
    input_new = gs.Variable(name="input", dtype=np.float32, shape=("batch", MAX_VOXELS, MAX_POINTS, NUM_FEATS))
    X = gs.Variable(name="coords_", dtype=np.int32, shape=("batch", MAX_VOXELS, 4))
    Y = gs.Variable(name="params", dtype=np.int32, shape=("batch",))
    first_node_after_pillarscatter = [node for node in graph.nodes if node.op == "Conv"][0]
    first_node_pillarvfe = [node for node in graph.nodes if node.op == "MatMul"][0]
    first_node_pillarvfe = first_node_pillarvfe.i()
    current_node = first_node_pillarvfe
    for _ in range(7):
        current_node = current_node.o()
    last_node_pillarvfe = current_node
    # merge some layers into one layer between inputs and outputs as below
    graph.inputs.append(Y)
    inputs = [last_node_pillarvfe.outputs[0], X, Y]
    outputs = [first_node_after_pillarscatter.inputs[0]]
    graph.replace_with_scatter(inputs, outputs)
    # Remove the now-dangling subgraph.
    graph.cleanup().toposort()
    # just keep some layers between inputs and outputs as below
    graph.inputs = [first_node_pillarvfe.inputs[0], X, Y]
    graph.outputs = [tmap["cls_preds"], tmap["box_preds"], tmap["dir_cls_preds"]]
    # Notice that we do not need to manually modify the rest of the graph. ONNX GraphSurgeon will
    # take care of removing any unnecessary nodes or tensors, so that we are left with only the subgraph.
    graph.cleanup()
    graph.inputs = [input_new, X, Y]
    first_add = [node for node in graph.nodes if node.op == "MatMul"][0]
    first_add = first_add.i()
    first_add.inputs[0] = input_new
    graph.cleanup().toposort()
    scatter_node = [n for n in graph.nodes if n.op == "PillarScatterPlugin"][0]
    lidar_point_features = cfg.dataset.data_augmentor.aug_config_list[0].num_point_features
    points = gs.Variable(
        name="points",
        dtype=np.float32,
        shape=("batch", cfg.inference.max_points_num, lidar_point_features)
    )
    num_points = gs.Variable(name="num_points", dtype=np.int32, shape=("batch",))
    voxels = gs.Variable(
        name="voxels", dtype=np.float32,
        shape=("batch", MAX_VOXELS, MAX_POINTS, NUM_FEATS)
    )
    voxel_coords = gs.Variable(name="voxel_coords", dtype=np.int32, shape=("batch", MAX_VOXELS, 4))
    num_pillar = gs.Variable(name="num_pillar", dtype=np.int32, shape=("batch",))
    pfp_attrs = dict()
    pfp_attrs["max_voxels"] = MAX_VOXELS
    pfp_attrs["max_num_points_per_voxel"] = MAX_POINTS
    pfp_attrs["voxel_feature_num"] = NUM_FEATS
    pfp_attrs["point_cloud_range"] = cfg.dataset.point_cloud_range
    pfp_attrs["voxel_size"] = cfg.dataset.data_processor[2].voxel_size
    VoxelGenerator_plugin = gs.Node(
        op="VoxelGeneratorPlugin",
        name="VoxelGeneratorPlugin_0",
        inputs=[points, num_points],
        outputs=[voxels, voxel_coords, num_pillar],
        attrs=pfp_attrs
    )
    first_add.inputs[0] = VoxelGenerator_plugin.outputs[0]
    scatter_node.inputs = [
        scatter_node.inputs[0],
        VoxelGenerator_plugin.outputs[1],
        VoxelGenerator_plugin.outputs[2]
    ]
    graph.nodes.append(VoxelGenerator_plugin)
    graph.inputs = [points, num_points]
    graph.cleanup().toposort()
    # Append postprocessing node
    num_boxes = gs.Variable(name="num_boxes", dtype=np.int32, shape=("batch",))
    decodebbox_attrs = dict()
    decodebbox_attrs["point_cloud_range"] = cfg.dataset.point_cloud_range
    decodebbox_attrs["num_dir_bins"] = cfg.model.dense_head.num_dir_bins
    decodebbox_attrs["dir_offset"] = cfg.model.dense_head.dir_offset
    decodebbox_attrs["dir_limit_offset"] = cfg.model.dense_head.dir_limit_offset
    decodebbox_attrs["score_thresh"] = cfg.model.post_processing.score_thresh
    decodebbox_attrs["anchor_bottom_height"] = []
    decodebbox_attrs["anchors"] = []
    for anchor in cfg.model.dense_head.anchor_generator_config:
        decodebbox_attrs["anchor_bottom_height"].extend(
            anchor["anchor_bottom_heights"]
        )
        for anc_size in anchor["anchor_sizes"]:
            for anc_rot in anchor["anchor_rotations"]:
                _anc_size = anc_size.copy()
                _anc_size.append(anc_rot)
                decodebbox_attrs["anchors"].extend(
                    _anc_size
                )
    num_classes = len(decodebbox_attrs["anchor_bottom_height"])
    nms_2d_size = graph.outputs[0].shape[1] * graph.outputs[0].shape[2]
    output_boxes = gs.Variable(
        name="output_boxes",
        dtype=np.float32,
        shape=("batch", nms_2d_size * num_classes * 2, 9)
    )
    DecodeBbox_plugin = gs.Node(
        op="DecodeBbox3DPlugin",
        name="DecodeBbox3DPlugin_0",
        inputs=graph.outputs,
        outputs=[output_boxes, num_boxes],
        attrs=decodebbox_attrs
    )
    graph.nodes.append(DecodeBbox_plugin)
    graph.outputs = DecodeBbox_plugin.outputs
    graph.cleanup().toposort()
    # Recursively set shape[0] = "batch"
    recursive_set_shape(scatter_node)
    return gs.export_onnx(graph)


if __name__ == '__main__':
    mode_file = "pointpillars-native-sim.onnx"
    simplify_onnx(onnx.load(mode_file))
