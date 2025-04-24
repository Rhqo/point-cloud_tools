import open3d as o3d
import numpy as np

# 1. PLY 파일 로드 (레이블 정보 포함)
pcd = o3d.io.read_point_cloud("11-2_1-1_clustered_filtered.ply")

# 2. 레이블 정보 추출 (예시: colors의 R 채널을 레이블로 가정)
labels = (np.asarray(pcd.colors)[:, 0] * 255).astype(int)

def highlight_cluster(indices):
    if not indices:
        return
    
    # 선택된 포인트의 레이블 확인
    selected_label = labels[indices[0]]
    
    # 원본 색상 복원
    original_colors = np.copy(np.asarray(pcd.colors))
    
    # 디밍 처리 (선택된 클러스터 제외)
    dimmed_colors = original_colors * 0.2
    dimmed_colors[labels == selected_label] = original_colors[labels == selected_label]
    
    pcd.colors = o3d.utility.Vector3dVector(dimmed_colors)

# 3. 시각화 파이프라인
vis = o3d.visualization.VisualizerWithEditing()
vis.create_window()
vis.add_geometry(pcd)

# 4. 사용자 상호작용 대기
vis.run()  # 사용자가 포인트 선택 후 'Q' 누르기

# 5. 선택된 포인트 처리
picked_points = vis.get_picked_points()
highlight_cluster(picked_points)

# 6. 최종 결과 표시
o3d.visualization.draw_geometries([pcd])
vis.destroy_window()
