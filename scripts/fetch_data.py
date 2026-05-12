import argparse
from pathlib import Path
import shutil

from fmri_forecasting.utils.nsd_utils import NSDDataHandler


BUCKET = "natural-scenes-dataset"
ROOT = "nsddata_timeseries/ppdata"
NIFTI_REL_PATH = 'func1pt8mm/timeseries'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download runs even when they already exist locally.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for i in range(1, 9):  # 1
        subj = f"subj0{i}"
        print("--------------------")
        print(f"Started {subj}")
        print("--------------------")

        handler = NSDDataHandler() # default handler
        atlas_path = str(Path(handler.template_dir) / "ROIs" / "visual_sphere_atlas_23.nii.gz")
        handler = NSDDataHandler(roi_atlas_path = atlas_path)
        handler.download_runs(subj, 9999, prefix="session", overwrite=args.overwrite)
        handler.get_t1_to_epi_warp(subj)
        handler.get_warped_atlas(subj, warp_type="func")

        handler.extract_all_roi_timeseries(subj, verbose=True)

        shutil.rmtree(handler.download_dir)

    # Encrypt Subjects & Sessions
    handler = NSDDataHandler(roi_atlas_path = atlas_path)
    handler.train_test_split(test_size=0.2, random_state=46, method="pool", enc_key=b"some_key")


if __name__ == "__main__":
    main()

