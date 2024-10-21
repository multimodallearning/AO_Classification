from shutil import copy
from pathlib import Path
import pandas as pd
from tqdm import tqdm

img_src = Path("/home/ron/Documents/KidsBoneChecker/datasets/data/GRAZPEDWRI-DX/img8bit")
lbl_src = Path("data/yolo_labels")

dst = Path("/home/ron/Desktop/GrazPedWriBVM")

val_fold = 1
fold_series = pd.read_csv('data/dataset_cv_splits.csv', index_col='filestem')['fold']


for file_stem, fold in tqdm(fold_series.items(), desc="Copying images and labels", total=len(fold_series)):
    is_validation = fold == val_fold
    sub_dir = "validation" if is_validation else "training"

    img_file = img_src / f"{file_stem}.png"
    lbl_file = lbl_src / f"{file_stem}.txt"

    copy(img_file, dst / "images" / sub_dir / img_file.name)
    copy(lbl_file, dst / "labels" / sub_dir / lbl_file.name)




