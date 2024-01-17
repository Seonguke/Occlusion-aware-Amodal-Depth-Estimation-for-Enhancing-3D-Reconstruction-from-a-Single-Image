import argparse
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any
import copy
import cv2
import matplotlib.pyplot as plt
import pyexr

from lib import modeling
from lib.config import config
from lib.utils.intrinsics import adjust_intrinsic
import lib.visualize as vis
from lib.visualize.image import write_detection_image, write_depth
from lib.structures.frustum import compute_camera2frustum_transform
from lib.modeling.amodal_model.dpt import Resize, NormalizeImage, PrepareForNet

from torchvision.transforms import Compose
import torch.nn.functional as F


def mask_input(dataset_root_path, scene_id, img_id):
    prev_id = img_id

    model_2d_sz = (384, 384)
    init_mask = Image.open(dataset_root_path + '/' + scene_id + '/' + f"mask_{prev_id}.png")

    init_mask = init_mask.resize(model_2d_sz)
    init_mask = np.array(init_mask) / 255.
    init_mask = Image.fromarray(init_mask)
    init_mask = init_mask.resize((384, 384))
    init_mask = np.array(init_mask)
    # normalize
    init_mask = init_mask.astype(np.float32)
    init_mask = torch.from_numpy(
        init_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    return init_mask


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
    output_path = str(output_path)
    output_depth = results['depth']
    output_rgb = results['rgb']

    output_rg = ['10_rgb']
    cat = ['7_depth']

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


def load_pretrained3d(opts, device):
    print("Load model...")

    model = modeling.PanopticReconstruction_base()
    checkpoint = torch.load(opts.model)
    update_dict = copy.deepcopy(model.state_dict())
    prt_3d = ['frustum3d']

    for k, v in checkpoint["model"].items():
        weight_name = k.split('.')[0]

        if weight_name in prt_3d:
            update_dict[k] = v

    model.load_state_dict(update_dict)
    model = model.to(device)  # move to gpu

    return model


def prepare_input_img(color):
    model_2d_sz = (384, 384)
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
    # prepare input image.
    print("Prepare input image...")

    color = color.resize((model_2d_sz))
    color = np.array(color) / 255.
    # color = color[:, ::-1, :]
    color = transform_rgb({"image": color})["image"]
    input_image = torch.from_numpy(color.astype(np.float32))
    return input_image


def main(opts):
    configure_inference(opts)
    device = torch.device("cuda:0")
    data_root = 'data/Amodal_front3d/filtering_rgb/'
    # Define image transformation.
    color_image_size = (320, 240)
    depth_image_size = (160, 120)

    model = load_pretrained3d(opts, device)

    f = open(opts.input_list, 'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip().split('/')
        scene_id = line[0]
        image_id = line[1]
        color = Image.open(data_root + scene_id + '/rgb_' + image_id + '.png')
        input_image = prepare_input_img(color)
        init_mask = mask_input(data_root, scene_id, image_id)
        # Prepare intrinsic matrix.
        front3d_intrinsic = np.array(config.MODEL.PROJECTION.INTRINSIC)
        front3d_intrinsic = adjust_intrinsic(front3d_intrinsic, color_image_size, depth_image_size)
        front3d_intrinsic = torch.from_numpy(front3d_intrinsic).to(device).float()

        # Prepare frustum mask.
        front3d_frustum_mask = np.load(str(data_root + "frustum_mask.npz"))["mask"]
        front3d_frustum_mask = torch.from_numpy(front3d_frustum_mask).bool().to(device).unsqueeze(0).unsqueeze(0)
        print("Perform panoptic 3D scene reconstruction...")
        with torch.no_grad():
            results = model.inference(input_image.unsqueeze(dim=0), front3d_intrinsic, front3d_frustum_mask, init_mask)

        print(f"Visualize results, save them at {config.OUTPUT_DIR + scene_id + '/' + image_id}")
        visualize_results(results, config.OUTPUT_DIR + scene_id + '/' + image_id)
        visualize_2d(results, config.OUTPUT_DIR + scene_id + '/' + image_id)


def configure_inference(opts):
    # load config
    config.OUTPUT_DIR = opts.output_path
    config.merge_from_file(opts.config_file)
    config.merge_from_list(opts.opts)
    # inference settings
    config.MODEL.FRUSTUM3D.IS_LEVEL_64 = False
    config.MODEL.FRUSTUM3D.IS_LEVEL_128 = False
    config.MODEL.FRUSTUM3D.IS_LEVEL_256 = False
    config.MODEL.FRUSTUM3D.FIX = True


def visualize_results(results: Dict[str, Any], output_path: os.PathLike) -> None:
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # Visualize depth prediction
    depth_map = results["depth"][0]
    write_depth(depth_map.squeeze().squeeze(), output_path / "depth_map.png")
    # Visualize projection
    vis.write_pointcloud(results["projection"].C[:, 1:], None, output_path / "projection.ply")

    # Visualize 3D outputs
    dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
    min_coordinates = torch.IntTensor([0, 0, 0]).to('cuda')
    truncation = config.MODEL.FRUSTUM3D.TRUNCATION
    iso_value = config.MODEL.FRUSTUM3D.ISO_VALUE

    geometry = results["frustum"]["geometry"]  # Point
    surface, _, _ = geometry.dense(dense_dimensions, min_coordinates, default_value=truncation)  # TSDF
    intrinsic_matrix = [[138.5641, 0.0000, 79.5000, 0.0000],
                        [0.0000, 138.5641, 59.5000, 0.0000],
                        [0.0000, 0.0000, 1.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 1.0000]]
    # Main outputs
    camera2frustum = compute_camera2frustum_transform(torch.tensor(intrinsic_matrix).cpu(),
                                                      torch.tensor(depth_map.squeeze().squeeze().size()),
                                                      config.MODEL.PROJECTION.DEPTH_MIN,
                                                      config.MODEL.PROJECTION.DEPTH_MAX,
                                                      config.MODEL.PROJECTION.VOXEL_SIZE)  # Projection Matrix ( Camera to World)

    # remove padding: original grid size: [256, 256, 256] -> [231, 174, 187]
    camera2frustum[:3, 3] += (torch.tensor([256, 256, 256]) - torch.tensor([231, 174, 187])) / 2
    frustum2camera = torch.inverse(camera2frustum)
    # print(frustum2camera)
    vis.write_distance_field(surface.squeeze(), None, output_path / "mesh_geometry.ply",
                             transform=frustum2camera)  # VOXEL to MC MESH

    # Visualize auxiliary outputs
    vis.write_pointcloud(geometry.C[:, 1:], None, output_path / "sparse_coordinates.ply")  # PCL

    surface_mask = surface.squeeze() < iso_value
    points = surface_mask.squeeze().nonzero()

    vis.write_pointcloud(points, None, output_path / "points_geometry.ply")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_list", "-i", type=str, default=str("resources/Amodal_front3d/demo.txt"))
    parser.add_argument("--output_path", "-o", type=str, default="output/base/")
    parser.add_argument("--config-file", "-c", type=str, default="configs/amodal_front3d_evaluate.yaml")
    parser.add_argument("--model", "-m", type=str, default="weights/SG3N_base.pth")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    main(args)
