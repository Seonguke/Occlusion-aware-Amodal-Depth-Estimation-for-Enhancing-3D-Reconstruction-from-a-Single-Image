import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from typing import Dict, Any

from lib import modeling

import lib.data.transforms2d as t2d
from lib.config import config
from lib.utils.intrinsics import adjust_intrinsic
from lib.structures import DepthMap

import copy
import lib.visualize as vis
from lib.visualize.image import write_detection_image, write_depth
from lib.structures.frustum import compute_camera2frustum_transform
from torchvision.transforms import Compose
from lib.modeling.amodal_model.dpt import Resize, NormalizeImage, PrepareForNet
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt
from lib.data import setup_dataloader
from tqdm import tqdm
from lib.structures.field_list import collect
import pyexr
import torch.nn.functional as F
import pandas as pd

# ------------- panoptic-reconstruction IoU -----------
def intersection(ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return (ground_truth.bool() & prediction.bool()).float()


def union(ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return (ground_truth.bool() | prediction.bool()).float()

def compute_iou(ground_truth: torch.Tensor, prediction: torch.Tensor) -> float:
    num_intersection = float(torch.sum(intersection(ground_truth, prediction)))
    num_union = float(torch.sum(union(ground_truth, prediction)))
    iou = 0.0 if num_union == 0 else num_intersection / num_union
    return iou

# ------------------------------

def dishow(disp):
    plt.imshow(disp)
    plt.jet()
    plt.colorbar(label='Distance to Camera')
    # plt.title('Depth2Disparity image')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.plot
    plt.show()


def mask_input(opts):
    scene_id = opts.line[0]
    prev_id = opts.line[1]
    mask_id = str(int(prev_id) - 1).zfill(4)
    mask_id2 = str(int(prev_id) - 2).zfill(4)
    dataset_root_path = '/data2/0511/filtering_rgb/'

    model_2d_sz = (384, 384)
    init_mask = Image.open(dataset_root_path + '/' + scene_id + '/' + f"mask_{prev_id}.png")
    hint = Image.open(dataset_root_path + '/' + scene_id + '/' + f"mask_{mask_id}.png")
    hint2 = Image.open(dataset_root_path + '/' + scene_id + '/' + f"mask_{mask_id2}.png")

    init_mask = init_mask.resize(model_2d_sz)
    hint = hint.resize(model_2d_sz)
    hint2 = hint2.resize(model_2d_sz)


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
    return init_mask, hint, hint2

def gt_depth_input(opts):
    dataset_root_path = '/data2/0511/filtering_rgb/'

    scene_id = opts[0]
    prev_id = opts[1]
    mask_id = str(int(prev_id) - 1).zfill(4)
    mask_id2 = str(int(prev_id) - 2).zfill(4)

    init_depth = pyexr.read(dataset_root_path + '/' + scene_id + '/' + f"depth_{prev_id}.exr").squeeze().copy()
    depth = pyexr.read(dataset_root_path + '/' + scene_id + '/' + f"depth_{mask_id}.exr").squeeze().copy()
    depth2 = pyexr.read(dataset_root_path + '/' + scene_id + '/' + f"depth_{mask_id2}.exr").squeeze().copy()

    #depth = pyexr.read(str(dataset_root_path / scene_id / f"depth_{prev_id}.exr")).squeeze().copy()
    if len(init_depth.shape)>2:
        init_depth = init_depth[:, :, 0]
        depth = depth[:,:,0]
        depth2  = depth2[:,:,0]


    init_depth = torch.from_numpy(
        init_depth.astype(np.float32)).unsqueeze(0)
    init_depth = F.interpolate(init_depth.unsqueeze(dim=0), size=(120, 160)).squeeze(dim=0)
    depth = torch.from_numpy(
        depth.astype(np.float32)).unsqueeze(0)
    depth = F.interpolate(depth.unsqueeze(dim=0), size=(120, 160)).squeeze(dim=0)
    depth2 = torch.from_numpy(
        depth2.astype(np.float32)).unsqueeze(0)
    depth2 = F.interpolate(depth2.unsqueeze(dim=0), size=(120, 160)).squeeze(dim=0)

    return init_depth,depth,depth2
def disave( disp, name, cat):
    for x in range(disp.shape[0]):
        plt.imshow(disp[x, 0, :, :])
        plt.jet()
        plt.colorbar()
        plt.plot()
        plt.savefig(fname=name+ '/'+ cat + '.jpg', bbox_inches='tight',
                    pad_inches=0)
        plt.close()


def visualize_2d(results: Dict[str, Any], output_path: os.PathLike) -> None:
    a = 0
    output_path = str(output_path)
    output_depth = results['depth']
    output_rgb = results['rgb']

    output_rg = ['10_rgb']
    cat = ['7_depth']


    for j, ct in enumerate(output_depth):
        ct = ct.detach().cpu()
        # ct = torch.clamp(ct, 0, 10)
        ct = ct.numpy()
        disave(ct,output_path,  cat[j])
    for j, mt in enumerate(output_rgb):
        mt = mt.permute(0, 2, 3, 1)
        mt += 1
        mt /= 2

        mt = mt.float().detach().cpu() * 255
        for x in range(mt.shape[0]):
            cv2.imwrite("{}/{}.png".format(output_path, output_rg[j]),
                        np.array(mt[x, ...])[:, :, ::-1])



def main(opts):
    configure_inference(opts)

    device = torch.device("cuda:0")

    # Define model and load checkpoint.
    print("Load model...")
    model = modeling.PanopticReconstruction_base()
    checkpoint = torch.load(opts.model)
    update_dict = copy.deepcopy(model.state_dict())
    prt_2d = ['frustum3d']

    for k, v in checkpoint["model"].items():
        zz = k.split('.')[0]
        # if 'frustum3d.model.model.encoder_features' in k:
        #    continue
        if zz in prt_2d:
            update_dict[k] = v

    model.load_state_dict(update_dict)
    #model.load_state_dict(checkpoint["model"])  # load model checkpoint
    model = model.to(device)  # move to gpu
    #model.switch_test()

    # Define image transformation.
    color_image_size = (320, 240)
    depth_image_size = (160, 120)
    model_2d_sz = (384, 384)
    # imagenet_stats = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # image_transforms = t2d.Compose([
    #     t2d.Resize(color_image_size),
    #     t2d.ToTensor(),
    #     t2d.Normalize(imagenet_stats[0], imagenet_stats[1]),  # use imagenet stats to normalize image
    # ])
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform_rgb = Compose(
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
            normalization,
            PrepareForNet(),
        ]
    )
    # Open and prepare input image.
    print("Load input image...")
    # color = Image.open(opts.input)
    # color = color.resize((model_2d_sz))
    # color = np.array(color) / 255.
    # # color = color[:, ::-1, :]
    # color = transform_rgb({"image": color})["image"]
    # input_image = torch.from_numpy(color.astype(np.float32))

    # init_mask, hint, hint2 = mask_input(opts)

    # Prepare intrinsic matrix.
    front3d_intrinsic = np.array(config.MODEL.PROJECTION.INTRINSIC)
    front3d_intrinsic = adjust_intrinsic(front3d_intrinsic, color_image_size, depth_image_size)
    front3d_intrinsic = torch.from_numpy(front3d_intrinsic).to(device).float()

    # Prepare frustum mask.
    front3d_frustum_mask = np.load(str("/data2/0511/filtering_rgb/frustum_mask.npz"))["mask"]
    front3d_frustum_mask = torch.from_numpy(front3d_frustum_mask).bool().to(device).unsqueeze(0).unsqueeze(0)
    dataloader = setup_dataloader(config.DATASETS.VAL, False, is_iteration_based=False, shuffle=False)



    print("Perform panoptic 3D scene reconstruction...")

    result_IoU = []
    for idx, (image_ids, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        if targets is None:
            print(f"Error, {image_ids[0]}")
            continue
        config.OUTPUT_DIR = config.OUTPUT_DIR + image_ids[0] + '/'
        suffix = image_ids[0].split('/')
        # Get input images
        input_image = collect(targets, "color")
        init_mask = collect(targets, "init_mask").squeeze(dim=0)
        hint = collect(targets, "hint").squeeze(dim=0)
        hint2 = collect(targets, "hint2").squeeze(dim=0)
        try:
            init_depth, depth, depth2 = gt_depth_input(suffix)
        except:
            continue
        with torch.no_grad():  # image: torch.Tensor, intrinsic, frustum_mask,init_mask,hint,hint2
            # results = model.inference(input_image, front3d_intrinsic, front3d_frustum_mask, init_mask,
            #                           hint, hint2)
            results = model.inference(input_image, front3d_intrinsic, front3d_frustum_mask, init_mask,
                                      hint, hint2, init_depth)
        coordinates = results['frustum']['coordinate']
        gt_occupancy = collect(targets, "occupancy_256").squeeze(dim=0)

        pred_occ = torch.zeros_like(gt_occupancy).cuda()
        pred_occ[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]] = 1

        gt_occupancy = gt_occupancy[front3d_frustum_mask.squeeze(dim=0)]
        pred_occ = pred_occ[front3d_frustum_mask.squeeze(dim=0)]


        #IoU Code gogo
        gt_sum = torch.sum(gt_occupancy)
        pred_sum = torch.sum(pred_occ)
        #IoU Code gogo
        if (gt_sum < 1000):
            out_dict = {
                'class': suffix[0],
                'idx': suffix[1],
                'gt_sum': gt_sum,
                'pred_sum': pred_sum,
            }
            result_IoU.append(out_dict)
            continue

        IoU = compute_iou(gt_occupancy, pred_occ)
        out_dict = {
            'class': suffix[0],
            'idx': suffix[1],
            'IoU': IoU,
            'gt_sum': gt_sum,
            'pred_sum': pred_sum,
        }
        result_IoU.append(out_dict)


        #print(torch.sum(gt_occupancy))  # < 1000  blank
        #print(torch.sum(pred_occ))

    out_file = '/data2/uk/access/01131_eval_meshes_full_IoU_base.csv'
    out_file_class = '/data2/uk/access/01131_eval_meshes_IoU_base.csv'

    # Create pandas dataframe and save
    eval_df = pd.DataFrame(result_IoU)
    # eval_df.set_index(['idx'], inplace=True)
    eval_df.set_index(['class'], inplace=True)

    eval_df.to_csv(out_file)

    # Create CSV file  with main statistics
    eval_df_class = eval_df.mean()
    eval_df_class.to_csv(out_file_class)

    # Print results
    print(eval_df_class)


def configure_inference(opts):
    # load config
    config.OUTPUT_DIR = opts.output
    config.merge_from_file(opts.config_file)
    config.merge_from_list(opts.opts)
    # inference settings
    config.MODEL.FRUSTUM3D.IS_LEVEL_64 = False
    config.MODEL.FRUSTUM3D.IS_LEVEL_128 = False
    config.MODEL.FRUSTUM3D.IS_LEVEL_256 = False
    config.MODEL.FRUSTUM3D.FIX = True


def visualize_results(results: Dict[str, Any], output_path: os.PathLike) -> None:
    device = results["input"].device
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # # Visualize depth prediction
    # depth_map: DepthMap = results["depth"]
    # depth_map.to_pointcloud(output_path / "depth_prediction.ply")
    # write_depth(depth_map, output_path / "depth_map.png")

    # Visualize depth prediction
    depth_map = results["depth"][0]
    write_depth(depth_map.squeeze().squeeze(), output_path / "depth_map.png")
    #write_depth(depth_map2.squeeze().squeeze(), output_path / "depth_map2.png")
    #write_depth(depth_map3.squeeze().squeeze(), output_path / "depth_map3.png")
    # Visualize projection
    vis.write_pointcloud(results["projection"].C[:, 1:], None, output_path / "projection.ply")

    # Visualize 3D outputs
    dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
    min_coordinates = torch.IntTensor([0, 0, 0]).to('cuda')
    truncation = config.MODEL.FRUSTUM3D.TRUNCATION
    iso_value = config.MODEL.FRUSTUM3D.ISO_VALUE

    geometry = results["frustum"]["geometry"] # Point
    surface, _, _ = geometry.dense(dense_dimensions, min_coordinates, default_value=truncation) # TSDF
    intrinsic_matrix = [[138.5641, 0.0000, 79.5000, 0.0000],
                        [0.0000, 138.5641, 59.5000, 0.0000],
                        [0.0000, 0.0000, 1.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 1.0000]]
    # Main outputs
    camera2frustum = compute_camera2frustum_transform(torch.tensor(intrinsic_matrix).cpu(),
                                                      torch.tensor(depth_map.squeeze().squeeze().size()),
                                                      config.MODEL.PROJECTION.DEPTH_MIN,
                                                      config.MODEL.PROJECTION.DEPTH_MAX,
                                                      config.MODEL.PROJECTION.VOXEL_SIZE)# Projection Matrix ( Camera to World)

    # remove padding: original grid size: [256, 256, 256] -> [231, 174, 187]
    camera2frustum[:3, 3] += (torch.tensor([256, 256, 256]) - torch.tensor([231, 174, 187])) / 2
    frustum2camera = torch.inverse(camera2frustum)
    # print(frustum2camera)
    vis.write_distance_field(surface.squeeze(), None, output_path / "mesh_geometry.ply", transform=frustum2camera)#VOXEL to MC MESH

    # Visualize auxiliary outputs
    vis.write_pointcloud(geometry.C[:, 1:], None, output_path / "sparse_coordinates.ply")# PCL

    surface_mask = surface.squeeze() < iso_value
    points = surface_mask.squeeze().nonzero()

    vis.write_pointcloud(points, None, output_path / "points_geometry.ply")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    pth = '/media/iccv/nvme0n1p2//data2/0511/filtering_rgb/'
    f = open("/media/iccv/nvme0n1p2//data2/0511/filtering_rgb/valid3.txt", 'r')
    lines = f.readlines()
    cnt = 0

    # print(zz)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, default="/media/iccv/nvme0n1p2//data2/uk/access/1/")
    parser.add_argument("--config-file", "-c", type=str, default="configs/front3d_evaluate.yaml")
    parser.add_argument("--model", "-m", type=str, default="/media/iccv/nvme0n1p2//data2/uk/0520_depth_1/model_0090000.pth")

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cnt = cnt + 1
    main(args)
