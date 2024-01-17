from collections import OrderedDict
from typing import List
import matplotlib.pyplot as plt
import MinkowskiEngine as Me
import torch
from lib import utils, modeling
from lib.config import config
from lib.modeling import backbone, depth, projection, frustum
from lib.modeling.utils import ModuleResult
from lib.structures import FieldList, DepthMap
from lib.utils import logger
from torch import nn
from lib.modeling.amodal_model import PartialCompletionContentDPT

import torch.nn.functional as F


class PanopticReconstruction(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model2d: nn.Module = PartialCompletionContentDPT()
        amodal_depth = './weights/amodal_depth.pth.tar'
        amodal_mask = './weights/amodal_mask.pth'
        self.model2d.load_model_demo(amodal_depth, amodal_mask)
        self.model2d.switch_to('eval')
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

    def forward(self, images: torch.Tensor, targets: List[FieldList], is_validate=False) -> ModuleResult:
        losses = {}
        results = {}

        init_mask = targets[0].get_field('init_mask')

        hint = targets[0].get_field('hint')
        hint2 = targets[0].get_field('hint2')

        images = images.clone()
        output_depth_1, _, dfeat1 = self.model2d.inf(images.cuda(), torch.ones_like(init_mask).cuda(),
                                                     torch.ones_like(init_mask).cuda())

        images2 = images.clone()
        images2[init_mask.repeat(1, 3, 1, 1) > 0] = -1
        output_depth_2, comp_mask_2, dfeat2 = self.model2d.inf(images2.cuda(), hint.cuda(), init_mask.cuda())

        images3 = images2.clone()
        images3[comp_mask_2.repeat(1, 3, 1, 1) > 0] = -1
        output_depth_3, comp_mask_3, dfeat3 = self.model2d.inf(images3.cuda(), hint2.cuda(),
                                                               init_mask.cuda() + hint.cuda())

        # 2D to 3D
        # Projection
        feature2d = [dfeat1, dfeat2, dfeat3]
        depth_pred = [output_depth_1, output_depth_2, output_depth_3]
        results.update({"depth": depth_pred})

        projection_results = self.projection(depth_pred, feature2d, targets)
        results.update({"projection": projection_results})

        # 3D
        frustum_losses, frustum_results = self.frustum3d(projection_results, targets)
        losses.update({"frustum": frustum_losses})
        results.update({"frustum": frustum_results})

        return losses, results

    def inference(self, image: torch.Tensor, intrinsic, frustum_mask, init_mask, hint, hint2):
        results = {
            "input": image,
            "intrinsic": intrinsic
        }
        image = image.clone()
        output_depth_1, _, dfeat1 = self.model2d.inf(image.cuda(), torch.ones_like(init_mask).cuda(),
                                                     torch.ones_like(init_mask).cuda())

        images2 = image.clone()
        images2[init_mask.repeat(1, 3, 1, 1) > 0] = -1
        output_depth_2, comp_mask_2, dfeat2 = self.model2d.inf(images2.cuda(), hint.cuda(), init_mask.cuda())

        images3 = images2.clone()
        images3[comp_mask_2.repeat(1, 3, 1, 1) > 0] = -1
        output_depth_3, comp_mask_3, dfeat3 = self.model2d.inf(images3.cuda(), hint2.cuda(),
                                                               init_mask.cuda() + hint.cuda())

        feature2d = [dfeat1, dfeat2, dfeat3]
        depth_pred = [output_depth_1, output_depth_2, output_depth_3]  # orginal 0111

        results["input_mask"] = [init_mask, hint, hint2]
        results["rgb"] = [image, images2, images3]
        results["output_mask"] = [comp_mask_2.clone(), comp_mask_3.clone()]

        results.update({"depth": depth_pred})

        # inference projection
        projection_result: Me.SparseTensor = self.projection.inference(depth_pred, feature2d,
                                                                       intrinsic)

        results["projection"] = projection_result

        # inference 3d
        frustum_result = self.frustum3d.inference(projection_result, frustum_mask)
        results["frustum"] = frustum_result

        '''-------------------------------------------'''

        return results

    def log_model_info(self) -> None:
        # if config.MODEL.INSTANCE2D.USE:
        #    logger.info(f"number of weights in detection network: {utils.count_parameters(self.instance2d):,}")
        # if config.MODEL.DEPTH2D.USE or not config.MODEL.PROJECTION.OCC_IN:
        #     logger.info(f"number of weights in depth network: {utils.count_parameters(self.depth2d):,}")
        if config.MODEL.FRUSTUM3D.USE:
            logger.info(f"number of weights in 3D network: {utils.count_parameters(self.frustum3d):,}")

    def fix_weights(self) -> None:
        # if config.MODEL.FIX2D:
        #     modeling.fix_weights(self, "2d")
        # # if config.MODEL.INSTANCE2D.FIX:
        # #   modeling.fix_weights(self, "instance2d")
        # if config.MODEL.DEPTH2D.FIX:
        #     modeling.fix_weights(self, "depth2d")
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
        self.model2d.eval()
        # if config.MODEL.FIX2D:
        #     self.encoder2d.eval()
        #
        #     if config.MODEL.DEPTH2D.USE:
        #         self.depth2d.eval()

        # if config.MODEL.INSTANCE2D.USE:
        #     self.instance2d.eval()

        # if config.MODEL.DEPTH2D.FIX:
        #     self.depth2d.eval()

        # if config.MODEL.INSTANCE2D.FIX:
        #     self.instance2d.eval()

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
