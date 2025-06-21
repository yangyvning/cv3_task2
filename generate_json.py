import json
import numpy as np

# 读入官方 train.py 生成的 cameras.json
with open('output/e16520d1-3/cameras.json', 'r') as f:
    cams = json.load(f)

frames = []
for cam in cams:
    # 构造 4×4 的相机到世界矩阵
    c2w = np.eye(4, dtype=float)
    c2w[:3, :3] = cam['rotation']    # 3×3 旋转
    c2w[:3, 3]  = cam['position']    # XYZ 平移
    frames.append({'transform_matrix': c2w.tolist()})

# 输出到 trajectory.json
with open('trajectory.json', 'w') as f:
    json.dump({'frames': frames}, f, indent=2)

print(f"Generated trajectory.json with {len(frames)} frames")
