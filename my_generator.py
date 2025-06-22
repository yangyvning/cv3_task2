#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from my_generator import get_new_train_json

# import numpy as np
# import matplotlib.pyplot as plt
# from scene.cameras import Camera
# from utils.graphics_utils import focal2fov, fov2focal
# from PIL import Image
try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


# 单数据集渲染
def render_set(model_path, name, iteration, views, gaussians, pipeline, background,
               train_test_exp, separate_sh, is_custom_path=False):
    if is_custom_path:
        # 自定义路径的输出目录
        render_path = os.path.join(model_path, "custom_path", "renders")
        gts_path = os.path.join(model_path, "custom_path", "gt")
    else:
        # 原始训练/测试集的输出目录
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    # 创建输出目录：renders（渲染结果）和 gt（真实图像）
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # 遍历每个相机视角进行渲染
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 核心渲染调用：传入当前相机、高斯模型、管线参数和背景
        rendering = \
        render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]

        # 自定义路径没有真实图像
        gt = view.original_image[0:3, :, :] if hasattr(view, 'original_image') else rendering

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        # 保存渲染结果和真实图像
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


# 核心渲染函数
def render_sets(args, dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                separate_sh: bool):
    with torch.no_grad():
        # 自定义路径，先替换
        if args.custom_path:
            get_new_train_json(args.json_path, args.path_frames, args.path_radius, args.path_height)

        gaussians = GaussianModel(dataset.sh_degree)  # 初始化高斯模型
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)  # 加载场景数据

        # 设置背景色（纯白或纯黑）
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 自定义渲染 或 渲染训练集
        if args.custom_path or (not skip_train):
            render_set(model_path=dataset.model_path, name="train", iteration=scene.loaded_iter,
                       views=scene.getTrainCameras(),
                       gaussians=gaussians, pipeline=pipeline, background=background,
                       train_test_exp=dataset.train_test_exp, separate_sh=separate_sh)
        if not skip_test:  # 渲染测试集
            render_set(model_path=dataset.model_path, name="test", iteration=scene.loaded_iter,
                       views=scene.getTestCameras(),
                       gaussians=gaussians, pipeline=pipeline, background=background,
                       train_test_exp=dataset.train_test_exp, separate_sh=separate_sh)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)  # 加载模型参数配置
    pipeline = PipelineParams(parser)  # 加载渲染管线配置
    parser.add_argument("--iteration", default=-1, type=int)  # 指定训练迭代次数
    parser.add_argument("--skip_train", action="store_true")  # 跳过训练集渲染
    parser.add_argument("--skip_test", action="store_true")  # 跳过测试集渲染
    parser.add_argument("--quiet", action="store_true")  # 静默模式（不输出日志）

    parser.add_argument("--custom_path", action="store_true", help="Render custom camera path")  # 自定义路径标志
    parser.add_argument("--path_radius", type=float, default=3.0, help="Radius for circular camera path")  # 指定渲染路径半径
    parser.add_argument("--path_frames", type=int, default=30, help="Number of frames in camera path")  # 指定渲染帧数
    parser.add_argument("--path_height", type=float, default=0.0, help="Height for circular camera path")  # 指定渲染相机高度
    parser.add_argument("--json_path", type=str, help="Path to transforms_train.json")  # 指定渲染相机高度
    args = get_combined_args(parser)  # 合并所有参数
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)  # 初始化系统状态（如随机种子）

    # 调用核心渲染函数
    render_sets(args, model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                SPARSE_ADAM_AVAILABLE)
