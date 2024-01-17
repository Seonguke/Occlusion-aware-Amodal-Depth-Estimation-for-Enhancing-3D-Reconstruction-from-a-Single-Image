from collections import OrderedDict
from typing import Dict, List

import torch
from torch import nn

import MinkowskiEngine as Me

from lib import utils, modeling
from lib.modeling import backbone, depth, detector, projection, frustum
from lib.modeling.utils import ModuleResult
from lib.structures import FieldList, DepthMap
from lib.utils import logger
from lib.config import config
import random
from scipy import ndimage as ndi
import copy
import numpy as np

class Masking_pretrain(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # 2D modules
        backbone2d = backbone.build_backbone()
        self.encoder2d: nn.Module = backbone.ResNetEncoder(backbone2d)
        self.depth2d: nn.Module = depth.DepthPrediction()
        self.instance2d: nn.Module = detector.GeneralizedRCNN(self.encoder2d.out_channels[:3])

        # 2D to 3D
        self.projection: nn.Module = projection.SparseProjection()

        # 3D modules
        self.frustum3d: nn.Module = frustum.FrustumCompletion()
        self.postprocess: nn.Module = frustum.PostProcess()  # TODO: pass thing and stuff classes
        
        # Define hierarchical training status
        self.training_stages = OrderedDict([("LEVEL-64", config.MODEL.FRUSTUM3D.IS_LEVEL_64),
                                           ("LEVEL-128", config.MODEL.FRUSTUM3D.IS_LEVEL_128),
                                           ("LEVEL-256", config.MODEL.FRUSTUM3D.IS_LEVEL_256),
                                           ("FULL", True)])

    def masking_Occupancy(self, results, per: int):
        # 0,1 ,11,12
        # TODO Add Occupancy
        occ = results["frustum"]["geometry"]
        occ_shape = occ.shape
        i_shape = results["panoptic"]["panoptic_instances"].shape
        occ = occ.view(-1)

        instance = results["panoptic"]["panoptic_instances"].cpu()
        # make a little 3D diamond:
        # TODO Dilated Mask############################################
        mask3d = instance > 0
        mask3d = mask3d.reshape((256, 256, 256)).cpu().numpy()
        diamond = ndi.generate_binary_structure(rank=3, connectivity=1)
        # dilate 30x with it
        dilated = ndi.binary_dilation(np.array(mask3d), diamond, iterations=3)
        # 0,1 ,11,12
        dilated = ~dilated
        dilated = torch.from_numpy(dilated).to('cuda')
        dilated = dilated.view(-1)
        ###########################################################3

        instance = instance.view(-1)
        # = torch.mul(occ,wall_mask.to('cuda'))
        un = np.unique(np.array(instance))
        for i in un:
            if i == 0:
                continue
            cnt = instance == i
            cnt = cnt.sum().item()
            idx = np.where(instance == i)
            sam = random.sample(list(idx[0]), int(cnt * (per / 100)))
            instance[sam] = 0  # instance to Occupancy
        v_mask = (instance > 0)
        v_mask = (v_mask).to('cuda')
        v_mask = torch.logical_or(v_mask, dilated)
        occ = torch.mul(occ, v_mask)

        # np.multiply(v_mask,instance)
        occ = occ.reshape(occ_shape)

        results["frustum"]["geometry"] = occ
        return results
    def forward(self, images: torch.Tensor, targets: List[FieldList], is_validate=False) -> ModuleResult:
        losses = {}
        results = {}
        m_inp = {}

        # 2D features
        _, image_features = self.encoder2d(images)

        # Depth
        depth_targets = [target.get_field("depth") for target in targets]
        depth_losses, depth_results = self.depth2d(image_features["blocks"], depth_targets)
        losses.update({"depth": depth_losses})
        results.update({"depth": depth_results})

        # Instances (Detection & Matching)
        instance_losses, instance_results = self.instance2d(image_features, targets, is_validate)
        losses.update({"instance": instance_losses})
        results.update({"instance": instance_results})

        # 2D to 3D
        # Projection
        feature2d = results["depth"]["features"]
        projection_results = self.projection(results["depth"]["prediction"], feature2d, instance_results, targets)
        results.update({"projection": projection_results})

        frustum_losses, frustum_results = self.frustum3d(projection_results, targets)
        losses.update({"frustum": frustum_losses})
        results.update({"frustum": frustum_results})

        if self.get_current_training_stage() == "FULL":
            _, panoptic_results = self.postprocess(instance_results, frustum_results)
            results.update({"panoptic": panoptic_results})

        return losses, results

    def inference(self, image: torch.Tensor, intrinsic, frustum_mask):
        results = {
            "input": image,
            "intrinsic": intrinsic
        }

        # Inference 2d
        _, image_features = self.encoder2d(image)

        # inference depth
        depth_result, depth_features = self.depth2d.inference(image_features["blocks"])
        results["depth"] = DepthMap(depth_result.squeeze(), intrinsic)

        # inference maskrcnn
        instance_result = self.instance2d.inference(image_features)
        results["instance"] = instance_result

        # inference projection
        projection_result: Me.SparseTensor = self.projection.inference(depth_result, depth_features, instance_result, intrinsic)
        results["projection"] = projection_result

        # inference 3d
        frustum_result = self.frustum3d.inference(projection_result, frustum_mask)
        results["frustum"] = frustum_result

        # Merge geometry, semantics & instances to panoptic output
        _, panoptic_result = self.postprocess(instance_result, frustum_result)
        results["panoptic"] = panoptic_result

        return results

    def log_model_info(self) -> None:
        if config.MODEL.INSTANCE2D.USE:
            logger.info(f"number of weights in detection network: {utils.count_parameters(self.instance2d):,}")
        if config.MODEL.DEPTH2D.USE or not config.MODEL.PROJECTION.OCC_IN:
            logger.info(f"number of weights in depth network: {utils.count_parameters(self.depth2d):,}")
        if config.MODEL.FRUSTUM3D.USE:
            logger.info(f"number of weights in 3D network: {utils.count_parameters(self.frustum3d):,}")

    def fix_weights(self) -> None:
        if config.MODEL.FIX2D:
            modeling.fix_weights(self, "2d")
        if config.MODEL.INSTANCE2D.FIX:
            modeling.fix_weights(self, "instance2d")
        if config.MODEL.DEPTH2D.FIX:
            modeling.fix_weights(self, "depth2d")
        if config.MODEL.FRUSTUM3D.FIX:
            modeling.fix_weights(self, "frustum3d")

    # def load_state_dict(self, loaded_state_dict: Dict) -> None:
    #     model_state_dict = self.state_dict()
    #     loaded_state_dict = modeling.strip_prefix_if_present(loaded_state_dict, prefix="module.")
    #     modeling.align_and_update_state_dicts(model_state_dict, loaded_state_dict)
    #
    #     super().load_state_dict(model_state_dict)

    def switch_training(self) -> None:
        # set parts in training mode and keep other fixed
        self.train()

        if config.MODEL.FIX2D:
            self.encoder2d.eval()

            if config.MODEL.DEPTH2D.USE:
                self.depth2d.eval()

            if config.MODEL.INSTANCE2D.USE:
                self.instance2d.eval()

        if config.MODEL.FRUSTUM3D.FIX:
            self.frustum3d.eval()

        if config.MODEL.DEPTH2D.FIX:
            self.depth2d.eval()

        if config.MODEL.INSTANCE2D.FIX:
            self.instance2d.eval()

    def switch_test(self) -> None:
        self.eval()

    def get_current_training_stage(self) -> str:
        # Return first element that is true
        for level, status in self.training_stages.items():
            if status:
                return level

    def set_current_training_stage(self, iteration: int) -> str:
        # Set proper hierarchy training level, globally affects 3D UNet
        num_iterations = config.MODEL.FRUSTUM3D.LEVEL_ITERATIONS_64
        last_training_stage = None
        if iteration >= num_iterations and self.training_stages["LEVEL-64"]:
            self.training_stages["LEVEL-64"] = False
            config.MODEL.FRUSTUM3D.IS_LEVEL_64 = False
            last_training_stage = "level_64"

        num_iterations += config.MODEL.FRUSTUM3D.LEVEL_ITERATIONS_128
        if iteration >= num_iterations and config.MODEL.FRUSTUM3D.IS_LEVEL_128:
            self.training_stages["LEVEL-128"] = False
            config.MODEL.FRUSTUM3D.IS_LEVEL_128 = False
            last_training_stage = "level_128"

        num_iterations += config.MODEL.FRUSTUM3D.LEVEL_ITERATIONS_256
        if iteration >= num_iterations and config.MODEL.FRUSTUM3D.IS_LEVEL_256:
            self.training_stages["LEVEL-256"] = False
            config.MODEL.FRUSTUM3D.IS_LEVEL_256 = False
            last_training_stage = "level_256"

        return last_training_stage
