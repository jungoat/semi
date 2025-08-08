# scripts/step4_build_patch_splits.py
# 목적: step3 결과(조건 라벨 포함)에서 학습에 바로 쓸 패치 메타 CSV 생성
# 입력: patches/defect_instances_labeled.csv
# 출력: patches/train.csv, patches/test.csv
# 컬럼: [patch_path, cond_id, split]

import os
import pandas as pd
from dataclasses import dataclass

labeled_csv = "patches/defect_instances_labeled.csv"
train_out = "patches/train.csv"
test_out = "patches/test.csv"

@dataclass
class PatchRecord:
    patch_path: str
    cond_id: int
    split: str

class PatchSplitBuilder:
    def __init__(self, labeled_csv_path: str):
        self.labeled_csv_path = labeled_csv_path

    def build(self):
        df = pd.read_csv(self.labeled_csv_path)
        need_cols = {"patch_path", "cond_id", "split"}
        if not need_cols.issubset(df.columns):
            raise ValueError(f"필요 컬럼 누락: {need_cols} ⊄ {set(df.columns)}")

        # 실제 파일 존재하는 것만 사용
        df = df[df["patch_path"].apply(lambda p: isinstance(p, str) and os.path.isfile(p))].copy()

        # 최소 컬럼만 남기고 정렬
        df = df[["patch_path", "cond_id", "split"]].sort_values(["split", "cond_id", "patch_path"]).reset_index(drop=True)

        # split별 저장
        df_train = df[df["split"] == "train"].copy()
        df_test  = df[df["split"] == "test"].copy()

        os.makedirs(os.path.dirname(train_out) or ".", exist_ok=True)
        df_train.to_csv(train_out, index=False)
        df_test.to_csv(test_out, index=False)

        # 요약 출력
        print(f"saved: {train_out} ({len(df_train)})")
        print(f"saved: {test_out}  ({len(df_test)})")
        print("train cond counts:", df_train["cond_id"].value_counts().sort_index().to_dict())
        print("test  cond counts:", df_test["cond_id"].value_counts().sort_index().to_dict())


def main():
    builder = PatchSplitBuilder(labeled_csv)
    builder.build()

if __name__ == "__main__":
    main()
