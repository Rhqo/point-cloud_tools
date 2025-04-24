from plyfile import PlyData, PlyElement
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='PLY 파일에서 label이 -1인 점들을 제외하고 저장합니다.')
parser.add_argument('input', help='입력 PLY 파일 경로')

def filter_and_save_ply(input_path, output_path):
    ply_data = PlyData.read(input_path)
    vertex_data = ply_data['vertex'].data

    np_data = np.array([(x['x'], x['y'], x['z'], x['red'], x['green'], x['blue'], x['scalar_sf']) for x in vertex_data],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'i2'), ('green', 'i2'), ('blue', 'i2'), ('scalar_sf', 'f4')])

    # 'scalar_sf'가 -1인 점들을 제외
    filtered_data = np_data[np_data['scalar_sf'] != -1]

    points = [tuple(row) for row in tqdm(filtered_data, desc='Filtering points')]

    dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'i2'), ('green', 'i2'), ('blue', 'i2'), ('scalar_sf', 'f4')]

    vertex = np.array(points, dtype=dtype_list)
    el = PlyElement.describe(vertex, 'vertex', comments=['filtered vertices'])

    PlyData([el]).write(output_path)
    print(f'저장 완료: {output_path}')


args = parser.parse_args()

input_ply_path = args.input
output_ply_path = args.input.replace('.ply', '_filtered.ply')

filter_and_save_ply(input_ply_path, output_ply_path)
