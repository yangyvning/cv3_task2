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
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def psnr(mse):
    return -10.0 * torch.log10(mse + 1e-8)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    losses, psnrs = [], []

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name} set")):
        result = render(view, gaussians, pipeline, background,
                        use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = result["render"]
        gt = view.original_image[0:3, :, :]

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f'{idx:05d}.png'))

        # 计算 MSE 和 PSNR
        mse_loss = F.mse_loss(rendering, gt)
        losses.append(mse_loss.item())
        psnrs.append(psnr(mse_loss).item())

    avg_mse = sum(losses) / len(losses)
    avg_psnr = sum(psnrs) / len(psnrs)
    print(f"[{name}] Avg MSE: {avg_mse:.6f}, Avg PSNR: {avg_psnr:.2f} dB")

    # 写入 TensorBoard
    writer = SummaryWriter(log_dir=os.path.join("runs", os.path.basename(model_path), name))
    writer.add_scalar("MSE", avg_mse, iteration)
    writer.add_scalar("PSNR", avg_psnr, iteration)
    writer.close()


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter,
                       scene.getTrainCameras(), gaussians, pipeline,
                       background, dataset.train_test_exp, separate_sh)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter,
                       scene.getTestCameras(), gaussians, pipeline,
                       background, dataset.train_test_exp, separate_sh)


if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering with Loss Evaluation and TensorBoard Logging")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print("Rendering " + args.model_path)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)
