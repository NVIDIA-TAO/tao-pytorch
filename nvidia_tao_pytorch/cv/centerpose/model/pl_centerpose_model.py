# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

""" Main PTL model file for CenterPose. """

import datetime
import os
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import MultiStepLR


from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils import common_utils

from nvidia_tao_pytorch.cv.centerpose.model.centerpose import create_model
from nvidia_tao_pytorch.cv.centerpose.model.criterion import ObjectPoseLoss
from nvidia_tao_pytorch.cv.centerpose.model.post_processing import HeatmapDecoder, TransformOutputs, MergeOutput, PnPProcess
from nvidia_tao_pytorch.cv.centerpose.model.post_processing_utils import save_inference_prediction
from nvidia_tao_pytorch.cv.centerpose.utils.centerpose_evaluator import Evaluator


# pylint:disable=too-many-ancestors
class CenterPosePlModel(TAOLightningModule):
    """ PTL module for CenterPose Model."""

    def __init__(self, experiment_spec):
        """Init training for CenterPose Model."""
        super().__init__(experiment_spec)
        self.batch_size = self.experiment_spec.dataset.batch_size
        self.training_config = self.experiment_spec.train
        self.infer_config = self.experiment_spec.inference
        self.eval_config = self.experiment_spec.evaluate

        # init the model and loss functions
        self._build_criterion()
        self._build_model()

        # post-processing
        self.hm_decoder = HeatmapDecoder(self.infer_config.num_select)
        self.transform_outputs = TransformOutputs(self.dataset_config.output_res, self.dataset_config.output_res)
        self.merge_output = MergeOutput(self.infer_config.visualization_threshold)
        self.pnp_process = PnPProcess(self.experiment_spec)

        self.status_logging_dict = {}
        self.val_cp_evaluator = Evaluator(self.experiment_spec)

        self.checkpoint_filename = 'centerpose_model'

    def _build_model(self):
        """Internal function to build the model."""
        self.model = create_model(self.experiment_spec.model)

    def _build_criterion(self):
        """Internal function to build the criterion."""
        self.loss = ObjectPoseLoss(self.experiment_spec.train.loss_config)

    def configure_optimizers(self):
        """Configure optimizers for training"""
        optimizer = torch.optim.Adam(self.model.parameters(), self.experiment_spec.train.optim.lr)
        lr_scheduler = MultiStepLR(optimizer=optimizer,
                                   milestones=self.experiment_spec.train.optim.lr_steps,
                                   gamma=self.experiment_spec.train.optim.lr_decay)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }

    def training_step(self, batch, batch_idx):
        """Training step."""
        outputs = self.model(batch['input'])
        loss = self.loss(outputs, batch, 'train')
        loss = loss.mean()
        if isinstance(self.model, torch.nn.DataParallel):
            torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.training_config.clip_grad_val)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.clip_grad_val)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        return loss

    def on_train_epoch_end(self):
        """Log Training metrics to status.json"""
        average_train_loss = self.trainer.logged_metrics["train_loss_epoch"].item()

        self.status_logging_dict = {}
        self.status_logging_dict["train_loss"] = average_train_loss

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def on_validation_epoch_start(self) -> None:
        """
        Validation epoch start.
        Reset CenterPose evaluator for each epoch.
        """
        self.val_cp_evaluator.reset()

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self.model(batch['input'])

        # Heatmap Decoder
        dets = self.hm_decoder(outputs)

        # Transform and merge the decoded outputs
        transformed_det = self.transform_outputs(dets.copy(), batch['principle_points'], batch['max_axis'])

        # Filter and merge the output
        merged_output = self.merge_output(transformed_det)

        # Set up the testing intrinsic matrix and the 3D keypoint format
        self.pnp_process.set_intrinsic_matrix(batch['intrinsic_matrix'])
        self.pnp_process.set_3d_keypoints_format(self.eval_config)

        final_output = self.pnp_process(merged_output)

        # Launch the evaluation
        self.val_cp_evaluator.evaluate(final_output, batch)

    def on_validation_epoch_end(self):
        """Validation epoch end.
        Compute 3D IoU@0.5 and 2D MPE (mean pixel error) at the end of epoch.
        """
        self.val_cp_evaluator.finalize()
        iou, mpe = self.val_cp_evaluator.get_accuracy()

        if self.trainer.is_global_zero:
            print("\n Validation 3DIoU : {}\n".format(iou))
            print("\n Validation 2DMPE : {}\n".format(mpe))

        self.log("val_3DIoU", iou, rank_zero_only=True, sync_dist=True)
        self.log("val_2DMPE", mpe, rank_zero_only=True, sync_dist=True)

        if not self.trainer.sanity_checking:
            self.status_logging_dict = {}
            self.status_logging_dict["val_3DIoU"] = str(iou)
            self.status_logging_dict["val_2DMPE"] = str(mpe)
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING
            )

        pl.utilities.memory.garbage_collection_cuda()

    def forward(self, x):
        """Forward of the CenterPose model."""
        outputs = self.model(x['input'])
        return outputs

    def on_test_epoch_start(self) -> None:
        """ Test epoch start.
        reset CenterPose evaluator at start
        """
        self.cp_evaluator = Evaluator(self.experiment_spec)

    def test_step(self, batch, batch_idx):
        """Test step. Evaluate """
        outputs = self.model(batch['input'])

        # Heatmap Decoder
        dets = self.hm_decoder(outputs)

        # Transform and merge the decoded outputs
        transformed_det = self.transform_outputs(dets.copy(), batch['principle_points'], batch['max_axis'])

        # Filter and merge the output
        merged_output = self.merge_output(transformed_det)

        # Set up the testing intrinsic matrix and the 3D keypoint format
        self.pnp_process.set_intrinsic_matrix(batch['intrinsic_matrix'])
        self.pnp_process.set_3d_keypoints_format(self.eval_config)

        final_output = self.pnp_process(merged_output)

        # Launch the evaluation
        self.cp_evaluator.evaluate(final_output, batch)

    def on_test_epoch_end(self):
        """Test epoch end.
        compute 3D IoU at the end of epoch
        """
        self.cp_evaluator.finalize()
        self.cp_evaluator.write_report()

        iou, mpe = self.cp_evaluator.get_accuracy()

        # Log the evaluation results to a file
        log_file = os.path.join(self.experiment_spec.results_dir, 'log_eval_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
        logger = common_utils.create_logger(log_file, rank=0)
        if self.trainer.is_global_zero:
            logger.info('**********************Start logging Evaluation Results **********************')
            logger.info('*************** 3D IoU *****************')
            logger.info('3D IoU: %.5f' % iou)
            logger.info('*************** 2D MPE *****************')
            logger.info('2D MPE: %.5f' % mpe)
        self.status_logging_dict = {}
        self.status_logging_dict["test_3DIoU"] = str(iou)
        self.status_logging_dict["test_2DMPE"] = str(mpe)
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Test metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def predict_step(self, batch, batch_idx):
        """Predict step. Inference """
        outputs = self.model(batch['input'])
        # Heatmap Decoder
        dets = self.hm_decoder(outputs)
        # Transform and merge the decoded outputs
        transformed_det = self.transform_outputs(dets.copy(), batch['principle_points'], batch['max_axis'])
        # Filter and merge the output
        merged_output = self.merge_output(transformed_det)

        if self.infer_config.use_pnp is True:
            merged_output = self.pnp_process(merged_output)
        return merged_output

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """Predict batch end.
        save the result inferences at the end of batch
        """
        output_dir = self.experiment_spec.results_dir
        save_inference_prediction(outputs, output_dir, batch, self.infer_config)
