import os
import numpy as np
import open3d as o3d
import plyfile
from plyfile import PlyElement, PlyData
import pyransac3d as pyrsc
import matplotlib.pyplot as plt
import tqdm

from utils.all_tools import read_ply2np, save_ply_with_color_by_label, get_files_from_txt

# dataset 순회하며 PLY 파일 처리
def process_dataset(database_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # train.txt와 test.txt 파일 경로
    train_txt = os.path.join(database_path, 'train.txt')
    test_txt = os.path.join(database_path, 'test.txt')
    
    # 파일 경로 존재 확인
    if not os.path.exists(train_txt):
        raise FileNotFoundError(f"Train file list not found: {train_txt}")
    if not os.path.exists(test_txt):
        raise FileNotFoundError(f"Test file list not found: {test_txt}")
    
    # 파일 목록 읽기
    train_files = get_files_from_txt(train_txt)
    test_files = get_files_from_txt(test_txt)

    # 작물 종류별 디렉토리 분류를 위한 기본 경로 지정
    train_base_folder = os.path.join(output_folder, 'train')
    test_base_folder = os.path.join(output_folder, 'test')
    os.makedirs(train_base_folder, exist_ok=True)
    os.makedirs(test_base_folder, exist_ok=True)
    
    datasets = [
        {"files": train_files, "base_folder": train_base_folder, "name": "train"},
        {"files": test_files, "base_folder": test_base_folder, "name": "test"}
    ]
    
    # train, test 파일 처리
    for dataset in datasets:
        for rel_path in tqdm.tqdm(dataset["files"], desc=f"Processing {dataset['name']} files"):
            try:
                # 전체 파일 경로 구성
                full_path = os.path.join(database_path, rel_path)
                
                if os.path.exists(full_path):
                    # 작물 종류 추출 (파일의 상위 디렉토리)
                    crop_type = os.path.dirname(rel_path).split('/')[0]
                    
                    # 작물 종류별 디렉토리 생성
                    crop_type_folder = os.path.join(dataset['base_folder'], crop_type)
                    os.makedirs(crop_type_folder, exist_ok=True)
                    
                    # Read the PLY file into a np
                    np_data = read_ply2np(full_path)

                    # Extract the filename without the path
                    file_name = os.path.basename(rel_path)

                    # Create the output file path
                    output_path = os.path.join(crop_type_folder, file_name)

                    save_ply_with_color_by_label(np_data, output_path)    
                else:
                    print(f"Warning: File not found - {full_path}")
            except Exception as e:
                print(f"Error processing file {rel_path}: {e}")


if __name__ == '__main__':
    dataset_path = "Crops3D_10k"
    process_dataset(dataset_path, "output")
