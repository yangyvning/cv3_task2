import torch
from scene import Scene
import os
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, GaussianModel
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.cameras import Camera
from utils.graphics_utils import focal2fov, fov2focal
from PIL import Image

# 检查是否可用 SparseGaussianAdam
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def read_colmap_cameras_center(images_txt_path):
    centers = []
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#') or len(line.strip()) == 0:
            continue
        parts = line.strip().split()
        if len(parts) >= 9:
            TX, TY, TZ = float(parts[5]), float(parts[6]), float(parts[7])
            centers.append([TX, TY, TZ])
    centers = np.array(centers)
    return centers

def generate_camera_path(scene_cameras, scene_center, num_frames=30, radius=3.0, height_offset=0.0):
    cameras = []
    ref_cam = scene_cameras[0]
    FoVx = ref_cam.FoVx
    FoVy = ref_cam.FoVy
    image_width = ref_cam.image_width
    image_height = ref_cam.image_height
    resolution = (image_width, image_height)
    dummy_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    pil_image = Image.fromarray(dummy_image)

    cx, cy, cz = scene_center
    traj_z = cz + height_offset

    for i in range(num_frames):
        theta = 2 * np.pi * i / num_frames
        cam_pos = np.array([
            cx + radius * np.cos(theta),
            cy + radius * np.sin(theta),
            traj_z
        ])

        forward = scene_center - cam_pos
        forward = forward / np.linalg.norm(forward)

        # 更加鲁棒的上方向计算
        world_up = np.array([0.0, 0.0, 1.0])
        if np.abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])

        right = np.cross(world_up, forward)
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        # 构造旋转矩阵（列向量为 right, up, forward）
        R = np.stack([right, up, forward], axis=1)

        camera = Camera(
            resolution=resolution,
            colmap_id=i,
            R=R,
            T=cam_pos,
            FoVx=FoVx,
            FoVy=FoVy,
            depth_params=None,
            image=pil_image,
            invdepthmap=None,
            image_name=f"custom_{i:05d}",
            uid=i,
            trans=np.array([0.0, 0.0, 0.0]),
            scale=1.0,
            data_device="cuda",
            train_test_exp=False,
            is_test_dataset=False,
            is_test_view=False
        )
        camera.camera_center = torch.from_numpy(cam_pos).float().cuda()
        cameras.append(camera)

    return cameras

def render_set(model_path, name, iteration, views, gaussians, pipeline, background,
               train_test_exp, separate_sh, is_custom_path=False):
    if is_custom_path:
        render_path = os.path.join(model_path, "custom_path", "renders")
        gts_path = os.path.join(model_path, "custom_path", "gt")
    else:
        render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
        gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        result = render(view, gaussians, pipeline, background,
                        use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = result["render"]
        gt = view.original_image[0:3, :, :] if hasattr(view, 'original_image') else rendering

        if train_test_exp:
            mid = rendering.shape[-1] // 2
            rendering = rendering[..., mid:]
            gt = gt[..., mid:]

        torchvision.utils.save_image(rendering,
                                     os.path.join(render_path, f'{idx:05d}.png'))
        torchvision.utils.save_image(gt,
                                     os.path.join(gts_path, f'{idx:05d}.png'))

def render_sets(dataset: ModelParams, iteration: int,
                pipeline: PipelineParams, skip_train: bool,
                skip_test: bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if args.custom_path:
            print(f"Rendering custom camera path ({args.path_frames} frames)...")
            images_txt_path = os.path.join(dataset.model_path, "cameras", "images.txt")
            centers = read_colmap_cameras_center(images_txt_path)
            scene_center = np.median(centers, axis=0)
            print(f"Computed scene center from COLMAP camera centers: {scene_center}")

            train_cams = scene.getTrainCameras() or scene.getTestCameras()
            custom_cameras = generate_camera_path(
                scene_cameras=train_cams,
                scene_center=scene_center,
                num_frames=args.path_frames,
                radius=args.path_radius,
                height_offset=args.path_height
            )
            render_set(dataset.model_path, "custom", scene.loaded_iter,
                       custom_cameras, gaussians, pipeline,
                       background, dataset.train_test_exp,
                       separate_sh, is_custom_path=True)

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter,
                       scene.getTrainCameras(), gaussians, pipeline,
                       background, dataset.train_test_exp,
                       separate_sh)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter,
                       scene.getTestCameras(), gaussians, pipeline,
                       background, dataset.train_test_exp,
                       separate_sh)

if __name__ == "__main__":
    parser = ArgumentParser(description="Custom Camera Path Rendering")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--custom_path", action="store_true")
    parser.add_argument("--path_radius", type=float, default=3.0)
    parser.add_argument("--path_frames", type=int, default=30)
    parser.add_argument("--path_height", type=float, default=0.0)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration,
                pipeline.extract(args), args.skip_train,
                args.skip_test, SPARSE_ADAM_AVAILABLE)
