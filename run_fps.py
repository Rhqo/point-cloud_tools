import torch
import numpy as np
import utils.all_tools as all_tools
from fps.fps import farthest_point_sample
import argparse

parser = argparse.ArgumentParser(description='FPS 샘플링을 사용하여 PLY 파일을 다운샘플링합니다.')
parser.add_argument('input', help='입력 GLB 파일 경로')

def fps_downsample_single_file(ply_file_path):
    point_num = 100000
    print(f"Processing: {ply_file_path}")
    print(f"Target point number: {int(point_num/1000)}k")
    
    # 데이터 로드
    data = all_tools.read_ply2np(ply_file_path)
    
    # FPS 샘플링
    xyz = data[:, :3][np.newaxis, ...]  # 배치 차원 추가
    xyz_tensor = torch.from_numpy(xyz).to("cuda")
    centroids = farthest_point_sample(xyz_tensor, point_num)
    
    # 결과 추출 및 저장
    sampled_data = data[centroids.cpu().numpy().squeeze(0)]  # 배치 차원 제거
    save_path = ply_file_path.replace(".ply", f'_{int(point_num/1000)}k.ply')
    all_tools.save_ply_from_np(sampled_data, save_path)
    print(f"Saved: {save_path}")

# 사용 예시
if __name__ == "__main__":
    args = parser.parse_args()

    input_file = args.input
    fps_downsample_single_file(input_file)
