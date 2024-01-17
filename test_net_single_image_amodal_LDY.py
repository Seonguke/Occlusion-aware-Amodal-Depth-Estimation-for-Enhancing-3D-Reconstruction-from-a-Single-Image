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
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from lib.data import setup_dataloader
from tqdm import tqdm
from lib.structures.field_list import collect
import pyexr
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
def disave(disp, name, cat):
    for x in range(disp.shape[0]):
        plt.imshow(disp[x, 0, :, :])
        plt.jet()
        plt.colorbar()
        plt.plot()
        plt.savefig(fname=name + '/' + cat + '.jpg', bbox_inches='tight',
                    pad_inches=0)
        plt.close()


def visualize_2d(results: Dict[str, Any], output_path: os.PathLike) -> None:
    a = 0
    output_path = str(output_path)
    output_depth = results['depth']
    output_rgb = results['rgb']
    inf_mask = results['output_mask']
    gt_mask = results['input_mask']

    output_rg = ['10_rgb', '11_rgb', '12_rgb']
    cat = ['7_depth', '8_depth', '9_depth']
    mac = ['4_inf_mask', '5_inf_mask']
    gtc = ['1_mask', '2_mask', '3_mask']

    for j, ct in enumerate(output_depth):
        ct = ct.detach().cpu()
        # ct = torch.clamp(ct, 0, 10)
        ct = ct.numpy()
        disave(ct, output_path, cat[j])
    for j, mt in enumerate(output_rgb):
        mt = mt.permute(0, 2, 3, 1)
        mt += 1
        mt /= 2

        mt = mt.float().detach().cpu() * 255
        for x in range(mt.shape[0]):
            cv2.imwrite("{}/{}.png".format(output_path, output_rg[j]),
                        np.array(mt[x, ...])[:, :, ::-1])
    for j, mt in enumerate(gt_mask):

        if mt.size(1) == 1:
            mt = mt.repeat(1, 3, 1, 1)
        mt = mt.permute(0, 2, 3, 1)
        # mt /= 2
        # mt += 0.5
        mt = mt.float().detach().cpu() * 255
        for x in range(mt.shape[0]):
            cv2.imwrite("{}/{}.png".format(output_path, gtc[j]),
                        np.array(mt[x, ...])[:, :, ::-1])
    for j, mt in enumerate(inf_mask):

        if mt.size(1) == 1:
            mt = mt.repeat(1, 3, 1, 1)
        mt = mt.permute(0, 2, 3, 1)
        # mt /= 2
        # mt += 0.5
        mt = mt.float().detach().cpu() * 255

        for x in range(mt.shape[0]):
            cv2.imwrite("{}/{}.png".format(output_path, mac[j]),
                        np.array(mt[x, ...])[:, :, ::-1])


def main(opts):
    configure_inference(opts)

    device = torch.device("cuda:0")

    # Define model and load checkpoint.
    print("Load model...")
    model = modeling.PanopticReconstruction()
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
    model = model.to(device)  # move to gpu

    color_image_size = (320, 240)
    depth_image_size = (160, 120)


    # Open and prepare input image.
    print("Load input image...")


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
                                      hint, hint2, init_depth, depth, depth2)

        print(f"Visualize results, save them at {config.OUTPUT_DIR}")
        visualize_results(results, config.OUTPUT_DIR)
        visualize_2d(results, config.OUTPUT_DIR)


        coordinates = results['frustum']['coordinate']
        gt_occupancy = collect(targets, "occupancy_256").squeeze(dim=0)

        pred_occ = torch.zeros_like(gt_occupancy).cuda()
        pred_occ[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]] = 1

        gt_occupancy=gt_occupancy[front3d_frustum_mask.squeeze(dim=0)]
        pred_occ=pred_occ[front3d_frustum_mask.squeeze(dim=0)]


        # IoU Code gogo
        gt_sum = torch.sum(gt_occupancy)
        pred_sum = torch.sum(pred_occ)
        # IoU Code gogo
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

        '''
        if (len(result_IoU) == 10):
            break
        '''
        #print(torch.sum(gt_occupancy)) #  < 1000  blank
        #print(torch.sum(pred_occ))


    #save root
    out_file = './01133_eval_meshes_full_IoU.csv'
    out_file_class = './01133_eval_meshes_IoU.csv'

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


    # Visualize depth prediction
    depth_map, depth_map2, depth_map3 = results["depth"]
    write_depth(depth_map.squeeze().squeeze(), output_path / "depth_map.png")
    write_depth(depth_map2.squeeze().squeeze(), output_path / "depth_map2.png")
    write_depth(depth_map3.squeeze().squeeze(), output_path / "depth_map3.png")
    # Visualize projection
    vis.write_pointcloud(results["projection"].C[:, 1:], None, output_path / "projection.ply")

    # Visualize 3D outputs
    dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
    min_coordinates = torch.IntTensor([0, 0, 0]).to('cuda')
    truncation = config.MODEL.FRUSTUM3D.TRUNCATION
    iso_value = config.MODEL.FRUSTUM3D.ISO_VALUE

    geometry = results["frustum"]["geometry"]
    surface, _, _ = geometry.dense(dense_dimensions, min_coordinates, default_value=truncation)
    intrinsic_matrix = [[138.5641, 0.0000, 79.5000, 0.0000],
                        [0.0000, 138.5641, 59.5000, 0.0000],
                        [0.0000, 0.0000, 1.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 1.0000]]
    # Main outputs
    camera2frustum = compute_camera2frustum_transform(torch.tensor(intrinsic_matrix).cpu(),
                                                      torch.tensor(depth_map.squeeze().squeeze().size()),
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


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    f = open("./resources/Amodal_front3d/demo.txt", 'r')
    lines = f.readlines()
    cnt = 0

    # print(zz)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, default="./access/3/")
    parser.add_argument("--config-file", "-c", type=str, default="configs/amodal_front3d_evaluate.yaml")
    parser.add_argument("--model", "-m", type=str, default="./weights/SG3N_Amodal.pth")

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cnt = cnt + 1
    main(args)
