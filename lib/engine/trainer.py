import time
from collections import OrderedDict
from pathlib import Path

import torch
from lib.structures.field_list import collect

from lib import utils, logger, config, modeling, solver, data

import lib.data.transforms2d as t2d
from lib.config import config
from lib.utils.intrinsics import adjust_intrinsic
from lib.structures import DepthMap

from pathlib import Path
import os
from typing import Dict, Any
import lib.visualize as vis
import json
from lib.visualize.image import write_detection_image, write_depth
from lib.structures.frustum import compute_camera2frustum_transform
import numpy as np
from PIL import Image
import random
from scipy import ndimage as ndi
import copy
class Trainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.checkpointer = None
        self.dataloader = None
        self.logger = logger
        self.meters = utils.MetricLogger(delimiter="  ")
        self.checkpoint_arguments = {}
        self.pretrain3d_path='weights/pretrained_frustum.pth'

        self.setup()

    def setup(self) -> None:
        # Setup model
        #TODO Change Traning mode

        self.model = modeling.PanopticReconstruction()

        device = torch.device(config.MODEL.DEVICE)
        self.model.to(device, non_blocking=True)

        Non_pret = self.model.state_dict()
        update_dict = copy.deepcopy(Non_pret)


        # Setup optimizer, scheduler, checkpointer
        self.optimizer = torch.optim.Adam(self.model.parameters(), config.SOLVER.BASE_LR,
                                          betas=(config.SOLVER.BETA_1, config.SOLVER.BETA_2),
                                          weight_decay=config.SOLVER.WEIGHT_DECAY)
        self.scheduler = solver.WarmupMultiStepLR(self.optimizer, config.SOLVER.STEPS, config.SOLVER.GAMMA,
                                                  warmup_factor=1,
                                                  warmup_iters=0,
                                                  warmup_method="linear")

        output_path = Path(config.OUTPUT_DIR)
        self.checkpointer = utils.DetectronCheckpointer(self.model, self.optimizer, self.scheduler, output_path)


        #Additionally load a 2D model which overwrites the previously loaded weights
        print(config.MODEL.PRETRAIN2D)
        if config.MODEL.PRETRAIN2D:

            prt_3d = ['frustum3d']


            pretrain_3d = torch.load(self.pretrain3d_path)

            for k, v in pretrain_3d["model"].items():
                frustum_weight = k.split('.')[0]

                if frustum_weight in prt_3d:
                    update_dict[k]=v

            self.model.load_state_dict(update_dict)

        self.checkpoint_arguments["iteration"] = 0



        # Dataloader
        self.dataloader = data.setup_dataloader(config.DATASETS.TRAIN)
        self.valid_loader = data.setup_dataloader(config.DATASETS.VAL)
    def do_valid(self,iteration) -> None:
        for idx, (image_ids, targets) in enumerate(self.valid_loader):

            # Get input images
            images = collect(targets, "color")

            # Pass through model
            # losses, results = self.model(images, targets)
            # print(image_ids[0])
            try:
                with torch.no_grad():
                    losses, results = self.model(images, targets)
            except Exception as e:
                print(e, "skipping", image_ids[0])
                del targets, images
                continue

            ##
            input_image = collect(targets, "color")
            results["input"] = input_image
            output_path = config.OUTPUT_DIR + str(iteration) + '/'+str(idx)
            print(losses)
            self.vis_occ(results, output_path, [target.get_field("depth") for target in targets][0].intrinsic_matrix)


    def do_train(self) -> None:
        # Log start logging

        self.logger.info(f"Start training {self.checkpointer.output_path.name}")

        # Switch training mode
        self.model.switch_training()

        # Main loop
        iteration = 0
        #
        iteration_end = time.time()
        for idx, (image_ids, targets) in enumerate(self.dataloader):
            assert targets is not None, "error during data loading"
            data_time = time.time() - iteration_end

            # Get input images
            images = collect(targets, "color")

            # Pass through model

            try:
                losses, results = self.model(images, targets)
            except Exception as e:
                print(e, "skipping", image_ids[0])
                del targets, images
                continue

            # Accumulate total loss
            total_loss: torch.Tensor = 0.0
            log_meters = OrderedDict()

            for loss_group in losses.values():
                for loss_name, loss in loss_group.items():
                    if torch.is_tensor(loss) and not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss
                        log_meters[loss_name] = loss.item()

            # Loss backpropagation, optimizer & scheduler step
            self.optimizer.zero_grad()

            if torch.is_tensor(total_loss):
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                log_meters["total"] = total_loss.item()
            else:
                log_meters["total"] = total_loss

            # Minkowski Engine recommendation
            torch.cuda.empty_cache()
            if iteration % 2000 == 0 and iteration >20000 :
                self.do_valid(iteration)
            # Save checkpoint
            if iteration % config.SOLVER.CHECKPOINT_PERIOD == 0:
                self.checkpointer.save(f"model_{iteration:07d}", **self.checkpoint_arguments)

            last_training_stage = self.model.set_current_training_stage(iteration)

            # Save additional checkpoint after hierarchy level
            if last_training_stage is not None:
                self.checkpointer.save(f"model_{last_training_stage}_{iteration:07d}", **self.checkpoint_arguments)
                self.logger.info(f"Finish {last_training_stage} hierarchy level")

            # Gather logging information
            self.meters.update(**log_meters)
            batch_time = time.time() - iteration_end
            self.meters.update(time=batch_time, data=data_time)
            current_learning_rate = self.scheduler.get_lr()[0]
            current_training_stage = self.model.get_current_training_stage()

            self.logger.info(self.meters.delimiter.join([f"IT: {iteration:06d}", current_training_stage,
                                                         f"{str(self.meters)}", f"LR: {current_learning_rate}"]))

            iteration += 1
            iteration_end = time.time()

        self.checkpointer.save("model_final", **self.checkpoint_arguments)


    def vis_occ(self,results: Dict[str, Any], output_path: os.PathLike,intrinsic_matrix) -> None:
        device = results["input"].device
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)


        # Visualize depth prediction
        depth_map,depth_map2,depth_map3= results["depth"]
        write_depth(depth_map.squeeze().squeeze(), output_path / "depth_map.png")
        write_depth(depth_map2.squeeze().squeeze(), output_path / "depth_map2.png")
        write_depth(depth_map3.squeeze().squeeze(), output_path / "depth_map3.png")
        # Visualize projection
        vis.write_pointcloud(results["projection"].C[:, 1:], None, output_path / "projection.ply")

        # Visualize 3D outputs
        dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
        min_coordinates = torch.IntTensor([0, 0, 0]).to(device)
        truncation = config.MODEL.FRUSTUM3D.TRUNCATION
        iso_value = config.MODEL.FRUSTUM3D.ISO_VALUE

        geometry = results["frustum"]["geometry"]
        surface, _, _ = geometry.dense(dense_dimensions, min_coordinates, default_value=truncation)

        # Main outputs
        camera2frustum = compute_camera2frustum_transform(intrinsic_matrix.cpu(),
                                                          torch.tensor(depth_map.squeeze().squeeze().size()) ,
                                                          config.MODEL.PROJECTION.DEPTH_MIN,
                                                          config.MODEL.PROJECTION.DEPTH_MAX,
                                                          config.MODEL.PROJECTION.VOXEL_SIZE)

        # remove padding: original grid size: [256, 256, 256] -> [231, 174, 187]
        camera2frustum[:3, 3] += (torch.tensor([256, 256, 256]) - torch.tensor([231, 174, 187])) / 2
        frustum2camera = torch.inverse(camera2frustum)
        # print(frustum2camera)
        vis.write_distance_field(surface.squeeze(), None, output_path / "mesh_geometry.ply", transform=frustum2camera)

        # Visualize auxiliary outputs
        vis.write_pointcloud(geometry.C[:, 1:], None, output_path / "sparse_coordinates.ply")

        surface_mask = surface.squeeze() < iso_value
        points = surface_mask.squeeze().nonzero()

        vis.write_pointcloud(points, None, output_path / "points_geometry.ply")



