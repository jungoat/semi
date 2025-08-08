# scripts/make_table1_report.py
# 목적: FID + 분류기 결과를 모아 Table 1 형태의 요약 CSV 생성
# 입력:
#   - runs/flow_match/fid.txt                        (scripts/eval_fid.py 출력)
#   - runs/cls_full/{real,synth,real_plus_synth}/metrics.txt  (scripts/eval_classifier.py 출력)
# 출력:
#   - runs/table1_report.csv            (긴 형식: metric, value, mode)
#   - runs/table1_classifier_wide.csv   (넓은 형식: acc/prec/rec/f1 × mode)

import os
import re
import pandas as pd

fid_txt = "runs/flow_match/fid.txt"
cls_modes = ["real", "synth", "real_plus_synth"]
cls_dir = "runs/cls_full"

out_long = "runs/table1_report.csv"
out_wide = "runs/table1_classifier_wide.csv"


def parse_fid(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # 예: "FID_all: 15.1234" 라인 파싱
    m = re.search(r"FID_all\s*:\s*([0-9]+\.?[0-9]*)", text)
    return float(m.group(1)) if m else None


def parse_metrics(path: str):
    """metrics.txt 파싱 → dict(acc,prec,rec,f1)"""
    if not os.path.isfile(path):
        return None
    vals = {"acc": None, "prec": None, "rec": None, "f1": None}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            for k in list(vals.keys()):
                if line.startswith(k + ":"):
                    try:
                        vals[k] = float(line.split(":", 1)[1])
                    except Exception:
                        pass
    return vals if all(v is not None for v in vals.values()) else None


def main():
    os.makedirs(os.path.dirname(out_long) or ".", exist_ok=True)

    rows = []

    # 1) FID
    fid = parse_fid(fid_txt)
    if fid is not None:
        rows.append({"metric": "FID_all", "value": fid, "mode": "flow_match"})

    # 2) 분류기(모드별)
    wide = {"metric": ["acc", "prec", "rec", "f1"]}
    for m in cls_modes:
        p = os.path.join(cls_dir, m, "metrics.txt")
        d = parse_metrics(p)
        if d is None:
            continue
        # long rows
        for k, v in d.items():
            rows.append({"metric": k, "value": v, "mode": m})
        # wide cols
        for k in ["acc", "prec", "rec", "f1"]:
            wide.setdefault(m, []).append(d[k])

    # 저장: long
    df = pd.DataFrame(rows)
    df = df.sort_values(["metric", "mode"]).reset_index(drop=True)
    df.to_csv(out_long, index=False)

    # 저장: wide (분류기만 피벗)
    wide_df = pd.DataFrame(wide)
    wide_df.to_csv(out_wide, index=False)

    print(f"saved: {out_long} ({len(df)} rows)")
    print(f"saved: {out_wide} ({len(wide_df)} rows)")


if __name__ == "__main__":
    main()
