# 필요한 라이브러리 설치
# !pip install opencv-python pandas

import cv2
import numpy as np
import pandas as pd
from glob import glob
import os
from tqdm import tqdm # 진행 상황을 보여주기 위한 라이브러리

def preprocess_defects(base_dir='data/train', output_dir='data/processed'):
    """
    원본 이미지와 GT 마스크를 기반으로 불량 패치와 속성 정보를 추출합니다.
    """
    img_dir = os.path.join(base_dir, 'img')
    patch_dir = os.path.join(output_dir, 'patches')
    
    # 결과 저장용 디렉토리 생성
    os.makedirs(patch_dir, exist_ok=True)

    # Ground Truth(_GT.png) 파일 목록 가져오기
    gt_files = sorted(glob(os.path.join(img_dir, '*_GT.png')))
    
    # 각 불량의 속성을 저장할 리스트
    defect_attributes = []

    print(f"총 {len(gt_files)}개의 Ground Truth 파일을 처리합니다...")

    # 각 GT 파일을 순회하며 처리
    for gt_path in tqdm(gt_files):
        # 마스크 이미지 로드 (흑백으로)
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # 마스크에 불량이 없는 경우(이미지가 전부 검은색) 건너뛰기
        if np.sum(mask) == 0:
            continue

        # --- 핵심 로직: Connected Components ---
        # 이미지에서 서로 분리된 각 불량 영역(흰색 덩어리)을 찾아 번호를 매기고 정보를 반환합니다.
        # num_labels: 찾은 객체 개수 (배경 포함)
        # labels: 각 픽셀에 객체 번호가 매겨진 이미지
        # stats: 각 객체의 x, y, 너비, 높이, 면적 정보
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # 배경(레이블 0)을 제외하고 각 불량 영역(레이블 1부터)을 순회
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # 논문처럼 너무 작은 노이즈성 영역은 무시할 수 있습니다 (예: 10픽셀 미만)
            if area < 10:
                continue

            # 원본 이미지 경로 찾기
            original_img_path = gt_path.replace('_GT.png', '.png')
            
            # --- 1. 불량 패치 추출 및 저장 ---
            original_img = cv2.imread(original_img_path)
            # 불량 영역의 바운딩 박스를 이용해 원본 이미지에서 패치 자르기
            patch = original_img[y:y+h, x:x+w]
            
            # 패치 파일 이름 생성 및 저장
            patch_filename = f"{os.path.basename(original_img_path).split('.')[0]}_patch_{i}.png"
            cv2.imwrite(os.path.join(patch_dir, patch_filename), patch)

            # --- 2. 불량 속성 정보 기록 ---
            defect_attributes.append({
                'patch_file': patch_filename, # 어떤 패치 파일에 해당하는지
                'original_file': os.path.basename(original_img_path),
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'aspect_ratio': w / h if h > 0 else 0
            })

    # --- 최종 결과 저장 ---
    # 수집된 속성 정보를 pandas DataFrame으로 변환
    attributes_df = pd.DataFrame(defect_attributes)
    # CSV 파일로 저장
    attributes_df.to_csv(os.path.join(output_dir, 'defect_attributes.csv'), index=False)
    
    print("\n" + "="*50)
    print(f" 처리 완료! 총 {len(attributes_df)}개의 불량 패치를 추출했습니다.")
    print(f"패치 이미지 저장 위치: {patch_dir}")
    print(f"속성 정보 CSV 파일: {os.path.join(output_dir, 'defect_attributes.csv')}")
    print("="*50)
    
    return attributes_df

# 함수 실행
attributes_df = preprocess_defects()
print("\n추출된 속성 정보 샘플:")
print(attributes_df.head())