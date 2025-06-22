# 基于 3D Gaussian Splatting 的物体重建与新视图合成
### 环境配置
- 操作系统: `Windows 11` 
- GPU: `NVIDIA GeForce RTX 3050 Laptop GPU` 
- CUDA: `Cuda V11.8`
- 根据README_3DGS.md文件配置3DGS环境依赖：

复制3DGS官方代码库
```shell
# SSH
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
```
创建虚拟环境
```shell
- SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda create --name 3dgs -y python=3.8
conda activate 3dgs
```
安装相关依赖
```shell
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install plyfile tqdm opencv-python joblib
```
编译子模块
```shell
cd submodules/diff-gaussian-rasterization
python setup.py install
cd ../fused-ssim
python setup.py install
cd ../simple-knn/
python setup.py install
```


### 训练和评估
```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```
默认情况下，训练好的模型会使用数据集中所有可用的图像。如果需要在训练过程中保留测试集进行评估，应使用该--eval标志
```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval # 
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```
这样可得到在训练集/测试集的loss数据和测试集上的PSNR、SSIM、LPIPS（Alex）指标。
train.py通过--test_iterations参数指定测试间隔（默认情况是7000,30000）。


通过以下命令进行tensorboard可视化查看
```shell
tensorboard --logdir runs
```
### 渲染
由于3DGS官方的render.py不支持自定义相机参数导入，因此本项目自定义my_generator.py自动生成环绕轨迹，my_render.py来实现自定义轨迹环绕
```shell
python my_render.py -m ./项目目录 --custom_path --skip_test --json_path ./path_to_json --path_radius 3.2 --path_frames 30 --path_height 1.0
```
my_render.py命令行参数
1. --custom_path
作用：是否启用自定义相机轨道（如环绕物体轨迹）。
使用该 flag 后，将触发 generate_camera_path() 生成一圈轨道；
会渲染 custom_path/renders 下的图片；

2. --path_radius（浮点数，默认 3.0）
作用：控制自定义轨道（即环绕路径）距离物体中心的半径。
值越小，离物体越近；
3. --path_frames（整数，默认 30）
作用：轨道中总共要渲染多少张图片（即一圈的帧数）。

4. --path_height（浮点数，默认 0.0）
作用：相机轨道高度相对物体中心 z 坐标的偏移。

最后使用ffmpeg，即可合成环绕物体一周的视频。
```shell
ffmpeg -framerate 30 -i render_path /%05d.png -c:v libx264 -pix_fmt yuv420p -y output_path
```
