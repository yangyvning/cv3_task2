
import numpy as np
import json

FILE = './data/deerqq/transforms_train.json'
ORIGINAL = './data/deerqq/transforms_train_original.json'

def generate_transform_matrix(theta, radius=3.0, height=0.0, center=[0,0,0]):
    theta_rad = np.radians(theta)
    
    # 1. 计算相机在世界坐标系中的位置
    x = center[0] + radius * np.cos(theta_rad)
    y = center[2] + radius * np.sin(theta_rad)
    z = height
    camera_pos = np.array([x, y, z])
    
    # 2. 计算相机朝向（指向场景中心）
    forward = (center - camera_pos)
    forward /= np.linalg.norm(forward)
    
    # 3. 计算右向量（假设世界 Z 轴向上）
    right = np.cross(np.array([0, 0, 1]), forward)
    right /= np.linalg.norm(right)
    
    # 4. 重新计算上向量保证正交
    up = np.cross(forward, right)

    # 5. 构建相机到世界的变换矩阵
    c2w = np.eye(4)
    c2w[:3, 0] = right    # X轴
    c2w[:3, 1] = up       # Y轴 
    c2w[:3, 2] = -forward # Z轴（相机看向-Z方向）
    c2w[:3, 3] = camera_pos
    
    return c2w.tolist()


def get_new_train_json(path_frames, path_radius, path_height):
    new_frames = []
    for i in range(path_frames):
        theta = i * 360 / path_frames   # 一圈
        transform = generate_transform_matrix(theta, path_radius, path_height)
        
        new_frames.append({
            "file_path": f"train/frame01_0001",
            "sharpness": 100.0,         # 统一锐度值
            "transform_matrix": transform
        })


    with open(FILE, 'r') as f:
        data = json.load(f)
    with open(ORIGINAL, 'w') as f:  # 备份
        json.dump(data, f, indent=4)

    data['frames'] = new_frames

    with open(FILE, 'w') as f:
        json.dump(data, f, indent=4)
    
    print('======================================================================')
    print(f'!!! The original file {FILE} be replaced, cpoied to {ORIGINAL}, remember to rename it before the next rendering process so that you will not lose the original file {FILE}')
    print('======================================================================')

