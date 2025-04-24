import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from plyfile import PlyElement, PlyData
import hdbscan
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='ply 파일을 HDBSCAN 클러스터링으로 처리합니다.')
parser.add_argument('input', help='입력 PLY 파일 경로')

def read_ply2np(path):
    ply_read = PlyData.read(path)
    name = [ply_read["vertex"].properties[i].name for i in range(len(ply_read["vertex"].properties))]
    data = np.array(ply_read["vertex"][name[0]]).reshape(-1, 1)
    
    for i, name in enumerate(tqdm(name[1:], desc="PLY 데이터 읽는 중")):
        temp_i = np.array(ply_read["vertex"][name]).reshape(-1, 1)
        data = np.concatenate([data, temp_i], axis=1)
    return data

def save_ply_from_np(np_input, ply_path):
    dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'int16'), ('green', 'int16'),
                 ('blue', 'int16')]
    
    # 데이터 변환 진행도 표시
    points = [tuple(x) for x in tqdm(np_input.tolist(), desc="데이터 변환 중")]
    
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

def perform_hdbscan(point_cloud, min_cluster_size=10, min_samples=5):
    # 좌표만 추출 (x, y, z)
    coords = point_cloud[:, :3]
    
    # 데이터 정규화
    coords_scaled = StandardScaler().fit_transform(coords)
    
    # HDBSCAN 클러스터링 수행
    with tqdm(total=100, desc="HDBSCAN 진행 중") as pbar:
        pbar.update(10)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                  min_samples=min_samples,
                                  gen_min_span_tree=True)
        pbar.update(40)
        labels = clusterer.fit_predict(coords_scaled)
        pbar.update(50)
    
    return labels

def assign_colors_to_clusters(labels):

    unique_labels = np.unique(labels)
    
    colors = np.zeros((len(labels), 3), dtype=np.int16)
    
    # 노이즈 포인트는 검은색으로 설정
    noise_mask = (labels == -1)
    colors[noise_mask] = [0, 0, 0]
    
    # 다양한 색상 생성 (matplotlib의 탭 색상 + 추가 색상)
    cmap = plt.get_cmap('tab20')
    extra_cmap = plt.get_cmap('hsv')
    
    color_idx = 0
    for label in tqdm(unique_labels, desc="클러스터별 색상 할당 중"):
        if label == -1:
            continue
            
        if color_idx < 20:
            rgb = np.array(cmap(color_idx)[:3]) * 255
        else:
            hsv_idx = (color_idx - 20) / (len(unique_labels) - 1 - 20) if len(unique_labels) > 21 else 0
            rgb = np.array(extra_cmap(hsv_idx)[:3]) * 255
            
        mask = (labels == label)
        colors[mask] = rgb.astype(np.int16)
        color_idx += 1
    
    return colors

def main(ply_file_path, min_cluster_size=10, min_samples=5, output_dir=None):
    steps = ["파일 로딩", "클러스터링", "색상 할당", "결과 생성", "파일 저장"]
    total_steps = len(steps)
    
    with tqdm(total=total_steps, desc="전체 처리 진행 상황") as main_pbar:
        # 포인트 클라우드 데이터 읽기
        print(f"{ply_file_path}에서 포인트 클라우드 읽는 중...")
        point_cloud = read_ply2np(ply_file_path)
        main_pbar.update(1)
        
        print(f"포인트 클라우드 형태: {point_cloud.shape}")
        
        # HDBSCAN 클러스터링 수행
        print("HDBSCAN 클러스터링 수행 중...")
        labels = perform_hdbscan(
            point_cloud, 
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        main_pbar.update(1)
        
        # 클러스터에 색상 할당
        colors = assign_colors_to_clusters(labels)
        main_pbar.update(1)
        
        # 결과 포인트 클라우드 생성
        # XYZ 좌표 유지 + 새로운 RGB 색상 + 레이블
        result_point_cloud = np.zeros((point_cloud.shape[0], 7), dtype=np.float32)
        
        # 진행 상황 표시와 함께 데이터 복사
        for i, name in enumerate(tqdm(["XYZ", "RGB", "레이블"], desc="데이터 복사 중")):
            if i == 0:
                result_point_cloud[:, :3] = point_cloud[:, :3]  # XYZ 좌표
            elif i == 1:
                result_point_cloud[:, 3:6] = colors  # RGB 색상
            else:
                result_point_cloud[:, 6] = labels  # 클러스터 레이블
        main_pbar.update(1)
        
        # 결과 저장
        if output_dir is None:
            output_dir = os.path.dirname(ply_file_path)
        
        base_name = os.path.basename(ply_file_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_clustered{ext}")
        
        print(f"output path: {output_path}")
        save_ply_from_np(result_point_cloud, output_path)
        main_pbar.update(1)
    
    return labels, result_point_cloud, output_path

if __name__ == '__main__':
    args = parser.parse_args()
    ply_file = args.input
    
    labels, result_pc, output_path = main(
        ply_file, 
        min_cluster_size=10,
        min_samples=5
    )
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)
    
    print(f"총 포인트 수: {len(labels)}")
    print(f"클러스터 수: {n_clusters}")
    print(f"노이즈 포인트 수: {n_noise} ({n_noise/len(labels)*100:.2f}%)")
