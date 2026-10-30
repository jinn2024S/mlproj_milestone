"""Builds malignant/benign labels and a train/val/test split."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

MALIGNANT = {"mel", "bcc", "akiec"}
BENIGN = {"nv", "bkl", "df", "vasc"}
TEST_DIAG_COLS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
SEED = 42 # fixed for identical splits
DATA_DIR = Path(".") # folder containing the HAM10000 downloads
OUT_DIR = Path("splits") # where to save train.csv, val.csv, test.csv
VAL_FRACTION = 0.15 # approximate share of HAM10000 used for validation

def find_images(folders):
    """Return {image_id: full path} for every .jpg inside the given folders."""
    paths = {}
    for folder in folders:
        folder = Path(folder)

        if not folder.exists():
            raise SystemExit(f"Folder not found: {folder}")
        
        for p in folder.rglob("*.jpg"):
            paths[p.stem] = str(p.resolve())

    return paths


def attach_paths(df, paths, name):
    """Add a 'path' column and drop rows whose image file is missing."""
    
    df["path"] = df["image_id"].map(paths) # look up each image_id file location
    missing = df["path"].isna().sum() # blank path if missing

    if missing:
        print(f"WARNING: {missing} {name} images listed in the labels were not found.")

    return df.dropna(subset=["path"]).reset_index(drop=True)


def describe(df, name):
    """Return a short text summary of one split that includes image counts, malignant share, lesions, and diagnoses."""
    n_mal = int(df["label"].sum()) # count malignant images
    lines = [f"{name}: {len(df)} images, {n_mal} malignant ({n_mal / len(df):.1%}), {len(df) - n_mal} benign", ]

    # if the table has lesion ids, then only count unique lesions
    if df["lesion_id"].notna().any():
        lines.append(f"unique lesions: {df['lesion_id'].nunique()}")

    counts = df["dx"].value_counts() # number of images per diagnosis
    lines.append("by diagnosis: " + ", ".join(f"{dx}={n}" for dx, n in counts.items()))
    return "\n".join(lines)


def load_ham10000(data_dir):
    """Read the HAM10000 labels and add malignant/benign labels and image paths"""
    candidates = sorted(p for p in data_dir.glob("HAM10000_metadata*") if p.is_file()) # match files starting with "HAM10000_metadata"
    if not candidates:
        raise SystemExit(f"Could not find HAM10000_metadata in {data_dir}")
    
    meta = pd.read_csv(candidates[0], sep=None, engine="python")
    meta.columns = [c.strip() for c in meta.columns] # remove spaces around column names
    for col in ("image_id", "lesion_id", "dx"):
        if col not in meta.columns:
            raise SystemExit(f"Metadata is missing column '{col}'. Columns found: {list(meta.columns)}")

    meta["dx"] = meta["dx"].astype(str).str.strip().str.lower() # clean diagnosis codes (" MEL" to "mel")
    # fail early if diagnosis code is unknown
    unknown = set(meta["dx"]) - MALIGNANT - BENIGN
    if unknown:
        raise SystemExit(f"Unexpected diagnosis codes in metadata: {unknown}")
    meta["label"] = meta["dx"].isin(MALIGNANT).astype(int) # 1 for malignant, otherwise 0

    paths = find_images([data_dir / "HAM10000_images_part_1", data_dir / "HAM10000_images_part_2"]) # get HAM10000 images from its two folders in one set
    meta = attach_paths(meta, paths, "HAM10000")
    return meta[["image_id", "lesion_id", "dx", "label", "path"]]


def load_isic2018_test(data_dir):
    """Read the ISIC 2018 Task 3 test labels and add malignant/benign labels and image paths."""
    gt = pd.read_csv(data_dir / "ISIC2018_Task3_Test_GroundTruth.csv", sep=None, engine="python")
    gt.columns = [c.strip() for c in gt.columns]

    if "dx" in gt.columns:
        # layout 1 matching HAM10000_metadata
        if "image_id" not in gt.columns:
            raise SystemExit(f"Ground-truth file has 'dx' but no 'image_id'. Columns found: {list(gt.columns)}")
        test = pd.DataFrame({
            "image_id": gt["image_id"].astype(str).str.strip(),
            "lesion_id": gt["lesion_id"] if "lesion_id" in gt.columns else pd.NA,
            "dx": gt["dx"].astype(str).str.strip().str.lower(),
        })
    else:
        # layout 2 with one column per diagnosis
        id_col = "image" if "image" in gt.columns else "image_id"
        missing_cols = [c for c in [id_col] + TEST_DIAG_COLS if c not in gt.columns]
        if missing_cols:
            raise SystemExit(f"Ground-truth file is missing columns {missing_cols}. Columns found: {list(gt.columns)}")
        
        # check each image has one diagnosis column set to 1
        row_sums = gt[TEST_DIAG_COLS].sum(axis=1)
        if not (row_sums == 1).all():
            raise SystemExit(f"{(row_sums != 1).sum()} test rows do not have exactly one diagnosis marked.")
        
        test = pd.DataFrame({
            "image_id": gt[id_col].astype(str).str.strip(),
            "lesion_id": pd.NA, # layout 2 has no lesion IDs
            "dx": gt[TEST_DIAG_COLS].idxmax(axis=1).str.lower(), # the name of the column marked 1
        })

    unknown = set(test["dx"]) - MALIGNANT - BENIGN
    if unknown:
        raise SystemExit(f"Unexpected diagnosis codes in test ground truth: {unknown}")
    test["label"] = test["dx"].isin(MALIGNANT).astype(int)

    paths = find_images([data_dir / "ISIC2018_Task3_Test_Images"])
    test = attach_paths(test, paths, "ISIC 2018 test")
    return test[["image_id", "lesion_id", "dx", "label", "path"]]


def main():
    data_dir = DATA_DIR
    out_dir = OUT_DIR
    n_splits = max(2, round(1 / VAL_FRACTION))

    # split HAM10000 into training and validation
    ham = load_ham10000(data_dir)
    n_splits = max(2, round(1 / VAL_FRACTION)) # split into 7 parts with 1 part (about 14%) becoming validation
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED) # photos of the same lesion stay together and each part gets the same share of malignant images
    # first fold becomes validation and the rest of the folds become training
    train_idx, val_idx = next(splitter.split(ham, ham["label"], groups=ham["lesion_id"]))
    train, val = ham.iloc[train_idx].reset_index(drop=True), ham.iloc[val_idx].reset_index(drop=True)

    # check that no lesions appear in multiple sets
    overlap = set(train["lesion_id"]) & set(val["lesion_id"])
    if overlap:
        raise SystemExit(f"Leak detected: {len(overlap)} lesions appear in both train and val.")

    test = load_isic2018_test(data_dir)

    # check that no test images appear in HAM10000
    shared = set(test["image_id"]) & set(ham["image_id"])
    if shared:
        raise SystemExit(f"Leak detected: {len(shared)} test images also appear in HAM10000.")

    # save the splits
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    # write summary of the split to summary.txt and print it
    summary = "\n\n".join([
        f"Split seed: {SEED}. HAM10000 split by lesion_id ({n_splits}-fold, one fold used as validation).",
        describe(train, "Train (HAM10000)"),
        describe(val, "Validation (HAM10000)"),
        describe(test, "Test (ISIC 2018 Task 3)")    
        ])
    
    (out_dir / "summary.txt").write_text(summary + "\n")
    print(summary)
    print(f"\nSaved train.csv, val.csv, test.csv and summary.txt to {out_dir.resolve()}")


if __name__ == "__main__":
    main()