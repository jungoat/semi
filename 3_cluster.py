# 필요한 라이브러리 설치
# !pip install scikit-learn matplotlib seaborn

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def assign_labels(input_csv='data/processed/defect_attributes.csv', 
                  output_dir='data/processed', 
                  n_clusters_x=2, 
                  n_clusters_y=3, 
                  n_clusters_aspect=2):
    """
    K-means 클러스터링을 사용하여 불량 속성에 라벨을 부여합니다.
    """
    df = pd.read_csv(input_csv)
    
    # 시각화 결과 저장 폴더 생성
    viz_dir = os.path.join(output_dir, 'cluster_visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # --- 클러스터링을 위한 데이터 준비 ---
    # 각 속성을 2D 배열 형태로 변환 (KMeans 입력 형식)
    x_coords = df[['x']]
    y_coords = df[['y']]
    aspect_ratios = df[['aspect_ratio']]
    
    # --- 각 속성별 K-means 클러스터링 ---
    kmeans_x = KMeans(n_clusters=n_clusters_x, random_state=42, n_init=10).fit(x_coords)
    kmeans_y = KMeans(n_clusters=n_clusters_y, random_state=42, n_init=10).fit(y_coords)
    kmeans_aspect = KMeans(n_clusters=n_clusters_aspect, random_state=42, n_init=10).fit(aspect_ratios)

    # --- 클러스터 라벨을 데이터프레임에 추가 ---
    df['x_cluster'] = kmeans_x.labels_
    df['y_cluster'] = kmeans_y.labels_
    df['aspect_cluster'] = kmeans_aspect.labels_

    # --- (중요) 라벨에 의미 부여하기 (Optional but Recommended) ---
    # K-means 라벨(0,1,2..)은 실행시마다 바뀔 수 있으므로, 중심값 기준으로 정렬해 일관성 부여
    x_centers = kmeans_x.cluster_centers_.flatten()
    x_map = {label: rank for rank, label in enumerate(np.argsort(x_centers))}
    df['x_cluster'] = df['x_cluster'].map(x_map)

    y_centers = kmeans_y.cluster_centers_.flatten()
    y_map = {label: rank for rank, label in enumerate(np.argsort(y_centers))}
    df['y_cluster'] = df['y_cluster'].map(y_map)
    
    aspect_centers = kmeans_aspect.cluster_centers_.flatten()
    aspect_map = {label: rank for rank, label in enumerate(np.argsort(aspect_centers))}
    df['aspect_cluster'] = df['aspect_cluster'].map(aspect_map)

    # --- 최종 클래스 라벨 생성 ---
    # 각 클러스터 라벨을 조합하여 하나의 고유한 라벨로 만듭니다.
    # 예: 12개 클래스 (0~11)
    df['final_class'] = (df['x_cluster'] * (n_clusters_y * n_clusters_aspect) +
                         df['y_cluster'] * n_clusters_aspect +
                         df['aspect_cluster'])

    # --- 결과 저장 ---
    output_csv_path = os.path.join(output_dir, 'labeled_defects.csv')
    df.to_csv(output_csv_path, index=False)

    print("\n" + "="*50)
    print(f" 라벨링 완료! 총 {df['final_class'].nunique()}개의 클래스가 생성되었습니다.")
    print(f"라벨링된 CSV 파일: {output_csv_path}")
    print("="*50)
    
    # --- 시각화로 결과 확인 ---
    plt.figure(figsize=(18, 5))
    
    # X 좌표 클러스터링 시각화
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df, x='x', y='y', hue='x_cluster', palette='viridis')
    plt.title('X-Coordinate Clusters')
    plt.savefig(os.path.join(viz_dir, 'x_clusters.png'))

    # Y 좌표 클러스터링 시각화
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df, x='x', y='y', hue='y_cluster', palette='viridis')
    plt.title('Y-Coordinate Clusters')
    plt.savefig(os.path.join(viz_dir, 'y_clusters.png'))

    # 가로세로비 클러스터링 시각화
    plt.subplot(1, 3, 3)
    sns.histplot(data=df, x='aspect_ratio', hue='aspect_cluster', multiple='stack', palette='viridis')
    plt.title('Aspect Ratio Clusters')
    plt.savefig(os.path.join(viz_dir, 'aspect_ratio_clusters.png'))

    plt.tight_layout()
    plt.show()
    print(f"클러스터링 시각화 결과가 {viz_dir} 폴더에 저장되었습니다.")
    
    return df

# 함수 실행
labeled_df = assign_labels()
print("\n라벨이 추가된 데이터 샘플:")
print(labeled_df.head())