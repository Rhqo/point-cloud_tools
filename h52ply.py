import h5py
import numpy as np
import open3d as o3d
import os

base_dir = './Crops3D_10k-C'

output_root_ply_dir = 'output_ply_files'

corrupt_dir_path = os.path.join(base_dir, 'corrupt')

def save_sample_as_ply(points, colors, output_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # RGB 값을 0-255에서 0-1 범위로 정규화
    colors = colors.astype(np.float32) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_path, pcd)

for filename in os.listdir(corrupt_dir_path):
    if filename.endswith('.h5'):
        h5_file_path = os.path.join(corrupt_dir_path, filename)

        parts = os.path.splitext(filename)[0].split('_')
        
        corruption_type = ""
        severity_level = ""

        if 'clean' in filename:
            corruption_type = 'clean'
            severity_level = '0'
        elif len(parts) >= 2 and parts[-1].isdigit():
            severity_level = parts[-1]
            corruption_type = '_'.join(parts[:-1])
        else:
            corruption_type = '_'.join(parts)
            severity_level = 'unknown'

        output_ply_type_dir = os.path.join(output_root_ply_dir, corruption_type)
        output_ply_severity_dir = os.path.join(output_ply_type_dir, f'level_{severity_level}')
        os.makedirs(output_ply_severity_dir, exist_ok=True)

        with h5py.File(h5_file_path, 'r') as f:
            full_data = f['data'][()]
            
            num_samples = full_data.shape[0]
            print(f"{num_samples} Samples in {h5_file_path}")

            for i in range(num_samples):
                sample_data = full_data[i]
                points = sample_data[:, :3]
                colors = sample_data[:, 3:6]

                output_filename = f"{os.path.splitext(filename)[0]}_sample_{i:03d}.ply"
                current_output_ply_path = os.path.join(output_ply_severity_dir, output_filename)

                save_sample_as_ply(points, colors, current_output_ply_path)