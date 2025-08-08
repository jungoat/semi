import os
import pandas as pd

# 고정 경로 (소문자)
train_csv = "data/meta/train.csv"
test_csv  = "data/meta/test.csv"
train_img = "data/train/img"
train_ann = "data/train/ann"
test_img  = "data/test/img"
test_ann  = "data/test/ann"

def build_index():
    def load_split(split_csv, img_dir, ann_dir, split_name):
        df = pd.read_csv(split_csv)
        records = []
        for _, row in df.iterrows():
            fname = os.path.basename(row['filename'])  # CSV에 filename, label 가정
            label = int(row['label'])

            img_path = os.path.join(img_dir, fname)
            mask_path = os.path.join(img_dir, fname.replace('.png', '_GT.png'))
            ann_path = os.path.join(ann_dir, fname.replace('.png', '.json'))

            records.append({
                'split': split_name,
                'basename': fname,
                'label': label,
                'img_path': img_path,
                'mask_path': mask_path,
                'ann_path': ann_path
            })
        return records

    train_recs = load_split(train_csv, train_img, train_ann, 'train')
    test_recs  = load_split(test_csv, test_img, test_ann, 'test')

    index_df = pd.DataFrame(train_recs + test_recs)
    index_df.to_csv("data/meta/index.csv", index=False)
    print(f"saved index.csv with {len(index_df)} rows")

if __name__ == "__main__":
    build_index()
