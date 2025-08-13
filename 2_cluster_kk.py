import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

def find_optimal_k(data, attribute_name, max_k=10):
    """
    실루엣 스코어를 사용하여 특정 속성에 대한 최적의 k값을 찾습니다.
    """
    k_values = range(2, max_k + 1)
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        score = silhouette_score(data, cluster_labels)
        silhouette_scores.append(score)

    # 가장 높은 점수를 받은 k 찾기
    best_k = k_values[np.argmax(silhouette_scores)]
    
    # 결과 시각화
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette Score for {attribute_name}')
    # 최고점에 빨간색 별 표시
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best k = {best_k}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_k

# 1단계에서 생성한 CSV 파일 로드
df = pd.read_csv('data/processed/defect_attributes.csv')

# 각 속성별 최적의 k 찾기
print("--- Finding optimal k for X-coordinate ---")
best_k_x = find_optimal_k(df[['x']], 'X-coordinate')

print("\n--- Finding optimal k for Y-coordinate ---")
best_k_y = find_optimal_k(df[['y']], 'Y-coordinate')

print("\n--- Finding optimal k for Aspect Ratio ---")
best_k_aspect = find_optimal_k(df[['aspect_ratio']], 'Aspect Ratio')

print("\n" + "="*50)
print("🎉 Optimal K values found:")
print(f"  - X-coordinate: k = {best_k_x}")
print(f"  - Y-coordinate: k = {best_k_y}")
print(f"  - Aspect Ratio: k = {best_k_aspect}")
print("\n이 값들을 `02_cluster_and_label.py` 파일에 적용하세요.")
print("="*50)