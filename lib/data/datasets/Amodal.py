import os
import random
import zipfile
from pathlib import Path
from typing import Dict, Union, List, Tuple
import torch.nn.functional as F
import numpy as np
import torch.utils.data
from PIL import Image
import pyexr
import cv2
import torch.nn as nn
from lib.data import transforms2d as t2d
from lib.data import transforms3d as t3d
from lib.structures import FieldList
from lib.config import config
from lib.utils.intrinsics import adjust_intrinsic
from torchvision.transforms import Compose
from lib.modeling.amodal_model.dpt import Resize, NormalizeImage, PrepareForNet

_imagenet_stats = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}


class Amodal(torch.utils.data.Dataset):
    def __init__(self, file_list_path: os.PathLike, dataset_root_path: os.PathLike, fields: List[str],
                 num_samples: int = None, shuffle: bool = False) -> None:
        super().__init__()

        self.dataset_root_path = Path(dataset_root_path+'filtering_rgb/')
        self.geo_root= Path(dataset_root_path+'weight/')
        self.samples: List = self.load_and_filter_file_list(file_list_path)

        if shuffle:
            random.shuffle(self.samples)

        self.samples = self.samples[:num_samples]

        # Fields defines which data should be loaded
        if fields is None:
            fields = []
        self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform_rgb = Compose(
            [
                Resize(
                    384,
                    384,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                self.normalization,
                PrepareForNet(),
            ]
        )
        self.fields = fields
        self.model_2d_sz=(384,384)
        self.image_size = (320, 240)
        self.depth_image_size = (160, 120)
        self.intrinsic = self.prepare_intrinsic()
        self.voxel_size = config.MODEL.PROJECTION.VOXEL_SIZE
        self.depth_min = config.MODEL.PROJECTION.DEPTH_MIN
        self.depth_max = config.MODEL.PROJECTION.DEPTH_MAX
        self.grid_dimensions = config.MODEL.FRUSTUM3D.GRID_DIMENSIONS
        self.truncation = config.MODEL.FRUSTUM3D.TRUNCATION
        self.max_instances = config.MODEL.INSTANCE2D.MAX
        self.num_min_instance_pixels = config.MODEL.INSTANCE2D.MIN_PIXELS
        self.stuff_classes = [0, 10, 11, 12]

        self.frustum_mask: torch.Tensor = self.load_frustum_mask()

        self.transforms: Dict = self.define_transformations()

    def __getitem__(self, index) -> Tuple[str, FieldList]:
        sample_path = self.samples[index]
        scene_id = sample_path.split("/")[0]
        image_id = sample_path.split("/")[1]
        prev_id=image_id
        mask_id = str(int(image_id) -1).zfill(4)
        mask_id2 =  str(int(image_id) -2 ).zfill(4)
        sample = FieldList(self.image_size, mode="xyxy")
        sample.add_field("index", index)
        sample.add_field("name", sample_path)

        try:

            # 2D data
            if "color" in self.fields:
                color = Image.open(self.dataset_root_path / scene_id / f"rgb_{prev_id}.png", formats=["PNG"])
                color = color.resize(self.model_2d_sz)
                color = np.array(color) / 255.
                #color = color[:, ::-1, :]
                color = self.transform_rgb({"image": color})["image"]
                color = torch.from_numpy(color.astype(np.float32))
                #color = self.transforms["color"](color)
                #color = torch.flip(color, (2,))

                sample.add_field("color", color)
                sample.add_field("rgb", self.dataset_root_path / scene_id / f"rgb_{prev_id}.png")
            if "depth" in self.fields:
                depth = pyexr.read(str(self.dataset_root_path / scene_id / f"depth_{prev_id}.exr")).squeeze().copy()
                try:
                    depth = depth[:, :, 0]
                except:
                    depth=depth
                #depth = depth[:, ::-1]
                depth = torch.from_numpy(
                    depth.astype(np.float32)).unsqueeze(0)
                depth=F.interpolate(depth.unsqueeze(dim=0),size=(120, 160)).squeeze(dim=0)
                depth = self.transforms["depth"](depth)
                sample.add_field("depth", depth)

            if "instance2d" in self.fields:
                # segmentation2d = np.load(self.dataset_root_path / scene_id / f"segmap_{image_id}.mapped.npz")["data"]
                # instance2d = self.transforms["instance2d"](segmentation2d)
                # sample.add_field("instance2d", instance2d)
                init_mask = Image.open(self.dataset_root_path / scene_id / f"mask_{prev_id}.png")
                hint = Image.open(self.dataset_root_path / scene_id / f"mask_{mask_id}.png")
                hint2 = Image.open(self.dataset_root_path / scene_id / f"mask_{mask_id2}.png")

                init_mask = init_mask.resize(self.model_2d_sz)
                hint = hint.resize(self.model_2d_sz)
                hint2 = hint2.resize(self.model_2d_sz)

                hint = np.array(hint) / 255.
                hint2 = np.array(hint2) / 255.
                mask = np.array(init_mask) / 255.

                mask = Image.fromarray(mask)
                hint = Image.fromarray(hint)
                hint2 = Image.fromarray(hint2)

                hint = hint.resize((384, 384))
                hint2 = hint2.resize((384, 384))
                mask = mask.resize((384, 384))

                hint = np.array(hint)
                hint2 = np.array(hint2)
                mask = np.array(mask)
                # normalize

                mask = mask.astype(np.float32)
                hint = hint.astype(np.float32)
                hint2 = hint2.astype(np.float32)
                mask[mask > 0] = 1
                hint[hint > 0] = 1
                hint2[hint2 > 0] = 1

                hint[mask > 0] = 0
                hint2[mask > 0] = 0
                hint2[hint > 0] = 0

                init_mask = torch.from_numpy(
                    mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                hint = torch.from_numpy(
                    hint.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                hint2 = torch.from_numpy(
                    hint2.astype(np.float32)).unsqueeze(0).unsqueeze(0)

                sample.add_field("init_mask", init_mask)
                sample.add_field("hint", hint)
                sample.add_field("hint2", hint2)

            # 3D data
            needs_weighting = False

            if "geometry" in self.fields:
                geometry_path = self.geo_root / scene_id / f"geometry_{prev_id}.npz"
                #geometry_path = self.dataset_root_path / scene_id / f"geometry_{prev_id}.npz"
                geometry = np.load(geometry_path)["data"]
                geometry = np.ascontiguousarray(
                    np.flip(geometry, axis=[0, 1]))  # Flip order, thanks for pointing that out.
                geometry = self.transforms["geometry"](geometry)

                # process hierarchy
                sample.add_field("occupancy_256", self.transforms["occupancy_256"](geometry))
                sample.add_field("occupancy_128", self.transforms["occupancy_128"](geometry))
                sample.add_field("occupancy_64", self.transforms["occupancy_64"](geometry))
                geometry = self.transforms["geometry_truncate"](geometry)
                sample.add_field("geometry", geometry)

                # add frustum mask
                sample.add_field("frustum_mask", self.frustum_mask.clone())

                needs_weighting = True

            # if "semantic3d" or "instance3d" in self.fields:
            #     segmentation3d_path = self.dataset_root_path / scene_id / f"segmentation_{prev_id}.mapped.npz"
            #     segmentation3d_data = np.load(segmentation3d_path)["data"]
            #     segmentation3d_data = np.copy(np.flip(segmentation3d_data, axis=[1, 2]))  # Flip order, thanks for pointing that out.
            #     semantic3d, instance3d = segmentation3d_data
            #     needs_weighting = True
            #
            #     if "semantic3d" in self.fields:
            #         semantic3d = self.transforms["semantic3d"](semantic3d)
            #         sample.add_field("semantic3d", semantic3d)
            #
            #         # process semantic3d hierarchy
            #         sample.add_field("semantic3d_64", self.transforms["segmentation3d_64"](semantic3d))
            #         sample.add_field("semantic3d_128", self.transforms["segmentation3d_128"](semantic3d))
            #
            #     if "instance3d" in self.fields:
            #         # Ensure consistent instance id shuffle between 2D and 3D instances
            #         instance_mapping = sample.get_field("instance2d").get_field("instance_mapping")
            #         instance3d = self.transforms["instance3d"](instance3d, mapping=instance_mapping)
            #         sample.add_field("instance3d", instance3d)
            #
            #         # process instance3d hierarchy
            #         sample.add_field("instance3d_64", self.transforms["segmentation3d_64"](instance3d))
            #         sample.add_field("instance3d_128", self.transforms["segmentation3d_128"](instance3d))

            if needs_weighting:
                weighting_path = self.geo_root / scene_id / f"weighting_{prev_id}.npz"
                weighting = np.load(weighting_path)["data"]
                weighting = np.copy(np.flip(weighting, axis=[0, 1]))  # Flip order, thanks for pointing that out.
                weighting = self.transforms["weighting3d"](weighting)
                sample.add_field("weighting3d", weighting)

                # Process weighting mask hierarchy
                sample.add_field("weighting3d_64", self.transforms["weighting3d_64"](weighting))
                sample.add_field("weighting3d_128", self.transforms["weighting3d_128"](weighting))

            return sample_path, sample
        except Exception as e:
            print(sample_path)
            print(e)
            return None, sample

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def load_and_filter_file_list(file_list_path: os.PathLike) -> List[str]:
        with open(file_list_path) as f:
            content = f.readlines()

        images = [line.strip() for line in content]

        return images

    def load_frustum_mask(self) -> torch.Tensor:
        mask_path = self.dataset_root_path / "frustum_mask.npz"
        mask = np.load(str(mask_path))["mask"]
        mask = torch.from_numpy(mask).bool()

        return mask

    def define_transformations(self) -> Dict:
        transforms = dict()

        # 2D transforms
        transforms["color"] = t2d.Compose([
            t2d.ToTensor(),
            t2d.Normalize(_imagenet_stats["mean"], _imagenet_stats["std"])
        ])

        transforms["depth"] = t2d.Compose([
            #t2d.ToImage(),
            #t2d.Resize(self.depth_image_size, Image.NEAREST),
            #t2d.ToNumpyArray(),
            #t2d.ToTensorFromNumpy(),
            t2d.ToDepthMap(self.intrinsic)  # 3D-Front has single intrinsic matrix
        ])

        transforms["instance2d"] = t2d.Compose([
            t2d.SegmentationToMasks(self.image_size, self.num_min_instance_pixels, self.max_instances, True,
                                    self.stuff_classes)
        ])

        # 3D transforms
        transforms["geometry"] = t3d.Compose([
            t3d.ToTensor(dtype=torch.float),
            t3d.Unsqueeze(0),
            t3d.ToTDF(truncation=12)
        ])

        transforms["geometry_truncate"] = t3d.ToTDF(truncation=self.truncation)

        transforms["occupancy_64"] = t3d.Compose(
            [t3d.ResizeTrilinear(0.25), t3d.ToBinaryMask(8), t3d.ToTensor(dtype=torch.float)])
        transforms["occupancy_128"] = t3d.Compose(
            [t3d.ResizeTrilinear(0.5), t3d.ToBinaryMask(6), t3d.ToTensor(dtype=torch.float)])
        transforms["occupancy_256"] = t3d.Compose([t3d.ToBinaryMask(self.truncation), t3d.ToTensor(dtype=torch.float)])

        transforms["weighting3d"] = t3d.Compose([t3d.ToTensor(dtype=torch.float), t3d.Unsqueeze(0)])
        transforms["weighting3d_64"] = t3d.ResizeTrilinear(0.25)
        transforms["weighting3d_128"] = t3d.ResizeTrilinear(0.5)

        transforms["semantic3d"] = t3d.Compose([t3d.ToTensor(dtype=torch.long)])

        transforms["instance3d"] = t3d.Compose(
            [t3d.ToTensor(dtype=torch.long), t3d.Mapping(mapping={}, ignore_values=[0])])

        transforms["segmentation3d_64"] = t3d.Compose([t3d.ResizeMax(8, 4, 2)])
        transforms["segmentation3d_128"] = t3d.Compose([t3d.ResizeMax(4, 2, 1)])

        return transforms

    def prepare_intrinsic(self) -> torch.Tensor:
        # image_size = (384, 384)
        # depth_image_size = (384, 384)
        intrinsic = np.array(config.MODEL.PROJECTION.INTRINSIC).reshape((4, 4))
        intrinsic_adjusted = adjust_intrinsic(intrinsic, self.image_size, self.depth_image_size)
        intrinsic_adjusted = torch.from_numpy(intrinsic_adjusted).float()

        return intrinsic_adjusted
