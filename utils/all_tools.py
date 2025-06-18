import os
import numpy as np
import open3d as o3d
import plyfile
from plyfile import PlyElement, PlyData
import pyransac3d as pyrsc
import matplotlib.pyplot as plt

# ply -> np
def read_ply2np(path):
    ply_read = PlyData.read(path)
    name = [ply_read["vertex"].properties[i].name for i in range(len(ply_read["vertex"].properties))]
    data = np.array(ply_read["vertex"][name[0]]).reshape(-1, 1)
    for i, name in enumerate(name[1:]):
        temp_i = np.array(ply_read["vertex"][name]).reshape(-1, 1)
        data = np.concatenate([data, temp_i], axis=1)
    return data


def read_pcd2np(path,type="XYZRGB"):
    pcd = o3d.io.read_point_cloud(path)

    points = np.asarray(pcd.points)
    if type=="XYZRGB":
        colors = np.asarray(pcd.colors)
        colors.astype(int)
        np_pcd = np.concatenate([points, colors],axis=1)
        return np_pcd
    else:
        return points


# np -> ply 파일 저장
def save_ply_from_np(np_input, ply_path):
    dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'int16'), ('green', 'int16'),
                  ('blue', 'int16')]
    points = [tuple(x) for x in np_input.tolist()]
    if np_input.shape[1] == 6:
        vertex = np.array(points, dtype=dtype_list)
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el]).write(ply_path)
    elif np_input.shape[1] == 7:
        dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'int16'), ('green', 'int16'),
                      ('blue', 'int16'), ('scalar_sf', 'f4')]
        vertex = np.array(points, dtype=dtype_list)
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el]).write(ply_path)
    elif np_input.shape[1] > 7:
        dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'int16'), ('green', 'int16'),
                      ('blue', 'int16'), ('scalar_sf', 'f4')]
        for i in range(np_input.shape[1] - 7):
            dtype_list.append((f'scalar_sf{i + 1}', 'f4'))
        vertex = np.array(points, dtype=dtype_list)
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el]).write(ply_path)
    elif np_input.shape[1] < 6:
        dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        if np_input.shape[1] >= 4:
            for i in range(np_input.shape[1] - 3):
                dtype_list.append((f'scalar_sf{i}', 'f4'))
        vertex = np.array(points, dtype=dtype_list)
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el]).write(ply_path)


# label별 색상을 할당하여 PLY 파일로 저장
# data: numpy 배열 (X,Y,Z,R,G,B,label)
def save_ply_with_color_by_label(data, output_path):
    # 마지막 열이 label 정보
    labels = data[:, 6]
    
    # 고유한 label 값 추출
    unique_labels = np.unique(labels)
    
    cmap = plt.get_cmap('Set1')
    colors = cmap.colors
    
    # 색상 맵 생성 (label -> RGB)
    color_map = {}
    for i, label in enumerate(unique_labels):
        if i < len(colors):
            rgba = colors[i]
            rgb = [int(255 * c) for c in rgba[:3]]
            color_map[label] = rgb
        else:
            # 컬러맵의 색상 수를 초과하는 경우 랜덤 색상 사용
            color_map[label] = [
                np.random.randint(0, 256),
                np.random.randint(0, 256),
                np.random.randint(0, 256)
            ]
    
    new_data = data.copy()

    # 각 포인트에 해당 label 색상 할당
    for label, color in color_map.items():
        mask = (labels == label)
        new_data[mask, 3:6] = color
    
    # PLY 파일 저장
    save_ply_from_np(new_data, output_path)


# train.txt와 test.txt 파일에서 파일 경로 목록 읽기
def get_files_from_txt(txt_path):
    with open(txt_path, 'r') as file:
        lines = file.readlines()
    # 경로에서 'data/Crops3D/' 제거하고 줄바꿈 제거
    return [line.replace('data/Crops3D/', '').strip().replace('\\', '/') for line in lines if line.strip()]


if __name__ == '__main__':
    data = read_ply2np("11-2_1-1.ply")
    print(data.shape)