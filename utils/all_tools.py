import os
import numpy as np
import open3d as o3d
import plyfile
from plyfile import PlyElement, PlyData
import pyransac3d as pyrsc
import matplotlib.pyplot as plt

def read_ply2np(path):
    """
    :rtype: numpy
    """
    ply_read = PlyData.read(path)
    name = [ply_read["vertex"].properties[i].name for i in range(len(ply_read["vertex"].properties))]
    data = np.array(ply_read["vertex"][name[0]]).reshape(-1, 1)
    for i, name in enumerate(name[1:]):
        temp_i = np.array(ply_read["vertex"][name]).reshape(-1, 1)
        data = np.concatenate([data, temp_i], axis=1)
    return data


def read_pcd2np(path,type="XYZRGB"):
    """

    :rtype: numpy
    """
    pcd = o3d.io.read_point_cloud(path)

    # 将点云数据转换为NumPy数组
    points = np.asarray(pcd.points)
    if type=="XYZRGB":
        colors = np.asarray(pcd.colors)
        colors.astype(int)
        np_pcd = np.concatenate([points, colors],axis=1)
        return np_pcd
    else:
        return points


def save_ply_from_np(np_input, ply_path):
    """

    Args:
        np_input: numpy point cloud
        ply_path: save path

    Returns:

    """
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
            # 컬러맵에서 색상 가져오기 (차례대로)
            rgba = colors[i]
            # RGB 값을 0-255 범위로 변환
            rgb = [int(255 * c) for c in rgba[:3]]
            color_map[label] = rgb
        else:
            # 컬러맵의 색상 수를 초과하는 경우 랜덤 색상 사용
            color_map[label] = [
                np.random.randint(0, 256),
                np.random.randint(0, 256),
                np.random.randint(0, 256)
            ]
    
    # 새로운 데이터 배열 생성
    new_data = data.copy()
    
    # 각 포인트에 해당 label 색상 할당
    for label, color in color_map.items():
        mask = (labels == label)
        new_data[mask, 3:6] = color
    
    # PLY 파일 저장
    save_ply_from_np(new_data, output_path)

if __name__ == '__main__':
    data = read_ply2np("11-2_1-1.ply")
    print(data.shape)