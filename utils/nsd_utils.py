# Tools related to fetching and processing data from NSD dataset
# Check out nilearn

import os
import shutil
import random
import re

from tqdm import tqdm
from pathlib import Path
import nibabel as nib
import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.client import Config

import ants
# import antspynet
import hmac
import base64
import hashlib
import json

from nilearn import datasets
from sklearn.model_selection import train_test_split
from collections import defaultdict

from src.process import compute_rs_graph

# S3 database definitions
BUCKET = "natural-scenes-dataset"
TS_ROOT = "nsddata_timeseries/ppdata"
TS_REL_PATH = 'func1pt8mm/timeseries'
RESOLUTION = None  # 1.8  # mm
ANAT_ROOT = "nsddata/ppdata"
ANAT_REL_PATH = "anat"
ANAT_FILE = "T1_0pt5_masked.nii.gz"  # Less aggressive skull stripping


class NSDDataHandler:
    """ Class used for downloading scans from NSD Dataset"""

    def __init__(self, download_dir=None, warp_dir=None, template_dir=None, template_path=None,
                 roi_atlas_path=None, deriv_dir=None, resolution=RESOLUTION):
        self.s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        self.resolution = resolution
        self.base_data_dir = Path("data")

        # Local directory definitions
        if download_dir is None:
            self.download_dir = str(self.base_data_dir / "nsd_downloads")
        else:
            self.download_dir = download_dir

        if deriv_dir is None:
            self.deriv_dir = str(self.base_data_dir / "nsd_derivatives")
        else:
            self.deriv_dir = deriv_dir

        if warp_dir is None:
            self.warp_dir = str(self.base_data_dir / "nsd_derivatives")
        else:
            self.warp_dir = warp_dir

        if template_dir is None:
            self.template_dir = str(self.base_data_dir / "templates")
        else:
            self.template_dir = template_dir

        if template_path is None:
            self.template_path = str(self.base_data_dir / "templates" / "MNI152_template.nii.gz")
        else:
            self.template_path = template_path

        if roi_atlas_path is None:
            self.roi_atlas_path = str(self.base_data_dir / "templates" / "ROIs" / "BN19.nii")
        else:
            if type(roi_atlas_path) == Path:
                roi_atlas_path = str(roi_atlas_path)
            self.roi_atlas_path = roi_atlas_path

    def list_runs(self, subj):
        """ List runs for the subject"""
        prefix = f"{TS_ROOT}/{subj}/{TS_REL_PATH}"
        resp = self.s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)

        if "Contents" not in resp:
            return []

        runs = {item["Key"].split("/")[-1] for item in resp.get("Contents", []) if
                item["Key"].split("/")[-1].endswith(".nii.gz")}
        return sorted(runs)

    def download_runs(self, subj, n_runs=None, prefix=None, overwrite=False):
        """
        Download first n BOLD runs for a subject from NSD.

        Parameters
        ----------
        subj : str
            Subject ID (e.g., 'subj01')
        n_runs : int or None
            Number of runs to download (None = download all)
        prefix : str or None
            Optional filename filter (e.g., 'session01', 'task-nsd')
        overwrite : bool
            Whether to re-download runs that already exist locally.
        """
        if prefix is None:
            prefix = "session"

        runs_save_dir = os.path.join(self.download_dir, subj, "func")
        os.makedirs(runs_save_dir, exist_ok=True)

        s3_prefix = f"{TS_ROOT}/{subj}/{TS_REL_PATH}"
        resp = self.s3.list_objects_v2(Bucket=BUCKET, Prefix=s3_prefix)

        assert "Contents" in resp, "Contents not in s3 resp"

        # Collect run files
        run_keys = [
            item["Key"]
            for item in resp["Contents"]
            if item["Key"].endswith(".nii.gz")
               and prefix in os.path.basename(item["Key"])
        ]

        # Sort deterministically
        run_keys = sorted(run_keys)
        if n_runs is not None:
            run_keys = run_keys[:n_runs]

        print(f"Starting download of {len(run_keys)} runs for {subj}...")
        for run_key in tqdm(run_keys):
            self.download_file_path(
                run_key,
                out_dir=runs_save_dir,
                verbose=False,
                overwrite=overwrite,
            )
        return runs_save_dir

    def get_subject_anat_scan(self, subj, recompute=False):
        """ Get subject anatomical scan path and file name in s3 database"""
        # anat_save_dir = os.path.join(os.getcwd(), r"data\nsd_downloads", subj, "anat")

        anat_save_dir = os.path.join(self.download_dir, subj, "anat")
        anat_save_path = os.path.join(anat_save_dir, ANAT_FILE)

        # Precomputed
        if os.path.exists(anat_save_path) and not recompute:
            return anat_save_path

        s3_prefix = f"{ANAT_ROOT}/{subj}/{ANAT_REL_PATH}/{ANAT_FILE}"
        resp = self.s3.list_objects_v2(Bucket=BUCKET, Prefix=s3_prefix)

        assert "Contents" in resp, "Contents not in s3 resp"

        anat_file_pth = resp.get("Contents", [])[0]["Key"]  # S3 database path
        anat_file = anat_file_pth.split('/')[-1]
        assert anat_file == ANAT_FILE, f"Dedicated anatomical file name and the one in database do not match " \
                                       f"({ANAT_FILE} != {anat_file})"
        anat_dwnld_path = self.download_file_path(anat_file_pth, out_dir=anat_save_dir)
        assert anat_save_path == anat_dwnld_path, f"Download path and expected save paths do not match " \
                                                  f"({anat_dwnld_path}, {anat_save_path})"
        return anat_save_path

    def download_file_path(self, s3_path, out_dir=None, verbose=True, overwrite=True):
        """ Download the file path and return the local-path"""

        if out_dir is None:
            out_dir = self.download_dir

        file_name = s3_path.split('/')[-1]
        extension = '.'.join(file_name.split('.')[1:])
        local_path = os.path.join(out_dir, file_name)
        os.makedirs(out_dir, exist_ok=True)

        if os.path.exists(local_path) and not overwrite:
            print("Skipping existing file:", local_path) if verbose else None
            return local_path

        print("Downloading:", s3_path) if verbose else None
        self.s3.download_file(BUCKET, s3_path, local_path)
        return local_path

    def get_rand_func_path(self, subj, seed=2345):
        """ Given subject returns random functional scan"""

        random.seed(seed)
        func_dir = Path(self.download_dir) / subj / "func"
        func_template = random.choice(list(func_dir.glob("*.nii.gz")))
        func_template = func_template.name
        target_scan_path = str(Path(func_dir) / func_template)  # run path
        epi_4d = ants.image_read(target_scan_path)
        subj_scan = epi_4d[..., 0]  # first volume (3D)
        return target_scan_path

    def get_mni_to_t1_warp(self, subj, recompute=False, recompute_template=False):
        """ Computes the subject space <-> MNI space warp using ANT toolbox and saves it. Returns the save path """

        subj_save_dir = str(Path(self.warp_dir) / subj / "anat")
        target_scan_path = self.get_subject_anat_scan(subj)

        # Dirs and paths
        affine_object_path = os.path.join(subj_save_dir, "mni_to_subj_affine.mat")
        warp_object_path = os.path.join(subj_save_dir, "mni_to_subj_warp.nii.gz")
        inv_affine_object_path = os.path.join(subj_save_dir, "subj_to_mni_affine.mat")
        inv_warp_object_path = os.path.join(subj_save_dir, "subj_to_mni_warp.nii.gz")

        if os.path.exists(warp_object_path) and os.path.exists(affine_object_path) and not recompute:
            return affine_object_path, warp_object_path
        print("Computing MNI to subject space transform ...")

        if recompute_template or not os.path.exists(self.template_path):
            print("Downloading MNI template ...")
            mni_nib = datasets.load_mni152_template(resolution=self.resolution)
            os.makedirs(self.template_dir, exist_ok=True)
            os.makedirs(str(Path(self.template_dir) / "ROIs"), exist_ok=True)
            nib.save(mni_nib, self.template_path)
        mni = ants.image_read(self.template_path)
        subj_scan = ants.image_read(target_scan_path)  # Get Anatomical Scan

        # Normalizing Intensities
        # mni = ants.histogram_match_image(mni, subj_t1) # Doesnt really improve ...

        # Run SyN registration (equivalent to antsRegistrationSyNQuick)
        reg = ants.registration(
            fixed=subj_scan,
            moving=mni,
            type_of_transform='SyNRA', verbose=False)  # Syn is not robust to 3T -> 7T

        # Save transforms
        os.makedirs(subj_save_dir, exist_ok=True)

        affine_path = warp_path = None
        for t in reg['fwdtransforms']:
            if t.endswith(".mat"):
                affine_path = t
            elif t.endswith(".nii.gz"):
                warp_path = t

        # Save inverse warps as well
        inv_affine_path = inv_warp_path = None
        for t in reg['invtransforms']:
            if t.endswith(".mat"):
                inv_affine_path = t
            elif t.endswith(".nii.gz"):
                inv_warp_path = t

        if affine_path is None or warp_path is None:
            raise RuntimeError("Could not identify affine and warp transforms")

        if inv_affine_path is None or inv_warp_path is None:
            raise RuntimeError("Could not identify inverse affine and inverse warp transforms")

        shutil.copy(affine_path, affine_object_path)
        shutil.copy(warp_path, warp_object_path)
        print(f"Saved MNI→T1 warp: {affine_object_path, warp_object_path}")

        shutil.copy(inv_affine_path, inv_affine_object_path)
        shutil.copy(inv_warp_path, inv_warp_object_path)
        print(f"Saved T1→MNI warp: {inv_affine_object_path, inv_warp_object_path}")

        return affine_object_path, warp_object_path

    def get_t1_to_epi_warp(self, subj, run_path=None, recompute=False):
        """
        Compute T1 → EPI transform using a reference run.

        Saves the affine (and optionally nonlinear) transforms for reuse.

        Parameters
        ----------
        subj : str
            Subject ID
        run_path : str
            Path to the BOLD run NIfTI file. If None, pick randomly
        recompute : bool
            If True, recompute warp even if cached

        Returns
        -------
        affine_object_path : str
            Path to the saved T1 → EPI affine transform
        """

        if run_path is None:
            run_path = self.get_rand_func_path(subj)  # Pick Run file

        # Directories
        subj_save_dir = str(Path(self.warp_dir) / subj / "func")
        os.makedirs(subj_save_dir, exist_ok=True)

        run_id = os.path.basename(run_path).replace(".nii.gz", "")
        affine_object_path = os.path.join(subj_save_dir, f"{run_id}_t1_to_epi_affine.mat")
        inv_affine_object_path = os.path.join(subj_save_dir, f"{run_id}_epi_to_t1_affine.mat")

        # If already computed
        if os.path.exists(affine_object_path) and os.path.exists(inv_affine_object_path) and not recompute:
            return affine_object_path, inv_affine_object_path

        print(f"Computing T1 → EPI warp for subject {subj}, run {run_id} ...")

        # Load images
        t1_path = self.get_subject_anat_scan(subj)
        subj_t1 = ants.image_read(t1_path)

        # Load 4D EPI and compute mean
        epi_nib = nib.load(run_path)
        epi_data = epi_nib.get_fdata()
        epi_mean_data = np.mean(epi_data, axis=3)
        epi_ref = ants.from_numpy(epi_mean_data)
        epi_ref.set_spacing(epi_nib.header.get_zooms()[:3])

        # Optionally reorient images to canonical axes
        # subj_t1 = ants.reorient_image2(subj_t1, orientation="RAS")
        # epi_ref = ants.reorient_image2(epi_ref, orientation="RAS")

        # Run registration: usually affine is sufficient for T1 → EPI
        reg = ants.registration(
            fixed=epi_ref,
            moving=subj_t1,
            type_of_transform='Affine',
            verbose=False
        )

        # Identify forward and inverse affine transforms
        affine_path = inv_affine_path_tmp = None
        for t in reg['fwdtransforms']:
            if t.endswith(".mat"):
                affine_path = t
        for t in reg['invtransforms']:
            if t.endswith(".mat"):
                inv_affine_path_tmp = t

        if affine_path is None or inv_affine_path_tmp is None:
            raise RuntimeError("Could not identify T1→EPI or EPI→T1 affine transforms")

        # Copy to final paths
        shutil.copy(affine_path, affine_object_path)
        shutil.copy(inv_affine_path_tmp, inv_affine_object_path)

        print(f"Saved T1→EPI warp: {affine_object_path}")
        print(f"Saved inverse EPI→T1 warp: {inv_affine_object_path}")

        return affine_object_path, inv_affine_object_path

    def get_subject_warps(self, subj, recompute=False, recompute_template=False):
        """ Compute MNI -> Subj T1 -> Subj EPI warps"""
        self.get_mni_to_t1_warp(subj, recompute=recompute, recompute_template=recompute_template)
        self.get_t1_to_epi_warp(subj, recompute=recompute)

    def get_warped_atlas(self, subj, warp_type='anat', func_template=None, roi_atlas_path=None, recompute=False):
        """
        Computes and saves the ROI atlas warped to the subject space.

        Parameters
        ----------
        subj : str
            Subject ID
        warp_type : str
            'anat' -> MNI → T1
            'func' -> MNI → T1 → EPI
        func_template : str or None
            Optional functional reference run path (used for func warp)
        roi_atlas_path : str or None
            Path to ROI atlas in MNI space
        recompute : bool
            Whether to recompute warp even if cached

        Returns
        -------
        warped_atlas_path : str
            Path to atlas warped into subject space (T1 or EPI)
        """
        if roi_atlas_path is None:
            roi_atlas_path = self.roi_atlas_path
        atlas_name = Path(roi_atlas_path).name

        if warp_type == 'anat':
            subj_warp_dir = Path(self.warp_dir) / subj / "anat"
            os.makedirs(subj_warp_dir, exist_ok=True)
            subj_scan_path = self.get_subject_anat_scan(subj)
        elif warp_type == 'func':
            warp_type = 'func'
            subj_warp_dir = Path(self.warp_dir) / subj / "func"
            os.makedirs(subj_warp_dir, exist_ok=True)
        else:
            raise ValueError(f"Unknown warp_type: {warp_type}")

        # Build output path
        warped_atlas_path = subj_warp_dir / f"atlas_{atlas_name}_in_{warp_type}_subject_space.nii.gz"
        if os.path.exists(warped_atlas_path) and not recompute:
            return str(warped_atlas_path)

        # Select target scan & warp directory
        if warp_type == 'anat':
            subj_scan = ants.image_read(subj_scan_path)

            # Get MNI → T1 transforms
            affine_object_path, warp_object_path = self.get_mni_to_t1_warp(subj, recompute=recompute)
            transform_list = [warp_object_path, affine_object_path]
        else:
            # Pick functional reference
            if func_template is None:
                func_template = self.get_rand_func_path(subj)  # random run

            # Compute T1 → EPI warp for this run
            t1_to_epi_affine, _ = self.get_t1_to_epi_warp(subj, func_template, recompute=recompute)

            # Get MNI → T1 transforms
            affine_mni_to_t1, warp_mni_to_t1 = self.get_mni_to_t1_warp(subj, recompute=recompute)

            subj_scan = load_epi_mean(func_template)

            # Compose transform list: T1 → EPI first, then MNI → T1
            transform_list = [t1_to_epi_affine, warp_mni_to_t1, affine_mni_to_t1]

        # Load atlas
        atlas = ants.image_read(roi_atlas_path)

        print(f"Warping ROI atlas to {warp_type} space ...")

        # Apply transforms
        warped_atlas = ants.apply_transforms(
            fixed=subj_scan,
            moving=atlas,
            transformlist=transform_list,
            interpolator="nearestNeighbor"
        )

        ants.image_write(warped_atlas, str(warped_atlas_path))
        return str(warped_atlas_path)

    # TODO: extract_brain for Visualization
    def extract_brain(self, subj, recompute=False):
        """ Strip skull and dura from anatomical scan, save the brain and mask images """

        # If already computed
        brain_path = f"{self.deriv_dir}/{subj}/T1_brain.nii.gz"
        mask_path = f"{self.deriv_dir}/{subj}/T1_brain_mask.nii.gz"

        if os.path.exists(brain_path) and os.path.exists(mask_path) and not recompute:
            return brain_path, mask_path

        # Get the anatomical scan
        anat_save_path = self.get_subject_anat_scan(subj)
        t1 = ants.image_read(anat_save_path)

        # Perform brain extraction
        brain = ants.brain_extraction(  # HERE: Currently not supported by antspyx
            t1,
            modality="t1",  # use "t2" for T2-weighted
        )

        # Get mask
        mask = brain > 0

        # Save result
        ants.image_write(brain, brain_path)
        ants.image_write(mask, mask_path)

        return brain_path, mask_path

    def extract_roi_timeseries(self, subj, run_name, roi_atlas_path=None, recompute=False, verbose=False):
        """
        Extract mean ROI timeseries from a single BOLD run and save per-session.

        The output is now organized as:
            {deriv_dir}/{subj}/roi_timeseries/sessionXX/roi_timeseries_runYY.npz

        where sessionXX is parsed from the run filename (e.g. "session20").

        Parameters
        ----------
        subj : str
            Subject ID
        run_name : str
            Filename of the BOLD run
        roi_atlas_path : str or None
            Path to ROI atlas in MNI space
        recompute : bool
            If True, recompute ROI timeseries even if cached
        verbose : bool
            If True, print actions

        Returns
        -------
        subj_roi_ts_path : str
            Path to directory containing ROI timeseries for this subject
        """
        # Get atlas name
        atlas_name = Path(self.roi_atlas_path).stem.replace(".nii", "")

        # Parse session and run number from filename
        match = re.match(r".*session(\d+)_run(\d+).*?\.nii\.gz$", run_name)
        if not match:
            if verbose:
                print(f"Warning: Could not parse session/run from filename: {run_name}")
                print("  → falling back to flat structure")
            session_id = "unknown"
            run_id = "unknown"
        else:
            session_num, run_num = match.groups()
            session_id = f"session{session_num}"
            run_id = f"run{run_num}"

        # Build output path and filename
        subj_save_dir = Path(self.deriv_dir) / subj
        session_dir = subj_save_dir / "roi_timeseries" / atlas_name / session_id
        os.makedirs(session_dir, exist_ok=True)
        ts_save_file = session_dir / f"roi_timeseries_{session_id}_{run_id}.npz"

        # Skip if already exists and not recompute
        if ts_save_file.exists() and not recompute:
            if verbose:
                print(f"Already exists (skipping): {ts_save_file}")
            return str(session_dir)

        # Get warped atlas in functional space of this run
        run_path = Path(self.download_dir) / subj / "func" / run_name
        if not run_path.exists():
            if verbose:
                print(f"Run file not found: {run_path}")
            return None

        warped_atlas_path = self.get_warped_atlas(
            subj,
            warp_type='func',
            func_template=str(run_path),
            roi_atlas_path=roi_atlas_path or self.roi_atlas_path,
            recompute=recompute
        )

        if not warped_atlas_path or not Path(warped_atlas_path).exists():
            if verbose:
                print(f"Failed to obtain warped atlas for {run_name}")
            return None

        # Load Nifti Images
        try:
            bold_img = nib.load(str(run_path))
            bold_data = bold_img.get_fdata()
            if bold_data.ndim != 4:
                raise RuntimeError(f"BOLD data must be 4D, got shape {bold_data.shape}")

            atlas_img = nib.load(str(warped_atlas_path))
            atlas_data = atlas_img.get_fdata().astype(int)

            if bold_data.shape[:3] != atlas_data.shape:
                raise RuntimeError(
                    f"Spatial mismatch: BOLD {bold_data.shape[:3]} vs "
                    f"atlas {atlas_data.shape}"
                )
        except Exception as e:
            if verbose:
                print(f"Error loading images for {run_name}: {e}")
            return None

        # Extract unique ROI labels (skip 0)
        roi_labels = np.unique(atlas_data)
        roi_labels = roi_labels[roi_labels != 0]

        roi_ts = {}
        roi_voxel_counts = {}

        for roi in roi_labels:
            mask = atlas_data == roi
            if not np.any(mask):
                continue
            vox_ts = bold_data[mask, :]
            roi_ts[int(roi)] = np.mean(vox_ts, axis=0)
            roi_voxel_counts[int(roi)] = int(mask.sum())

        if not roi_ts:
            if verbose:
                print(f"No valid ROIs found in atlas for {run_name}")
            return None

        # Save ROI timeseries
        try:
            np.savez_compressed(
                ts_save_file,
                roi_labels=np.array(list(roi_ts.keys()), dtype=int),
                timeseries=np.stack(list(roi_ts.values()), axis=0),
                voxel_counts=np.array(list(roi_voxel_counts.values()), dtype=int),
                run_name=run_name,
                session=session_id,
                run=run_id,
                atlas_name=atlas_name
            )
            if verbose:
                print(f"Saved: {ts_save_file}  ({len(roi_ts)} ROIs)")
            return str(ts_save_file)

        except Exception as e:
            if verbose:
                print(f"Failed to save {ts_save_file}: {e}")
            return None

    def extract_all_roi_timeseries(self, subj, roi_atlas_path=None, recompute=False, verbose=False):
        """
        Extract ROI timeseries for **all** downloaded functional runs of the given subject.

        For each downloaded BOLD run (`.nii.gz` file) in the subject's `func` directory,
        this method calls `extract_roi_timeseries` to compute and save mean timeseries
        per ROI (using the specified or default atlas after warping).

        Parameters
        ----------
        subj : str
            Subject ID (e.g., 'subj01')
        roi_atlas_path : str or Path, optional
            Path to the ROI atlas file in template space.
            If None, uses the default atlas path stored in the handler (`self.roi_atlas_path`).
        recompute : bool, default False
            If True, recomputes the warped atlas and ROI timeseries even if output files exist.
        verbose : bool, default False
            If True, prints progress information for each run.

        Returns
        -------
        list of str
            List of paths to the paths where ROI timeseries were saved
            (one per run), or empty list if no runs were found/processed.

        See Also
        --------
        extract_roi_timeseries : Method used to process each individual run.
        list_runs : Method to list available runs on S3 (before download).
        download_runs : Method to download runs from NSD S3 bucket.
        """
        func_dir = Path(self.download_dir) / subj / "func"

        if not func_dir.exists() or not func_dir.is_dir():
            if verbose:
                print(f"No functional runs directory found for {subj} at: {func_dir}")
            return []

        # Find all .nii.gz files (BOLD runs)
        run_paths = sorted(func_dir.glob("*.nii.gz"))

        if not run_paths:
            if verbose:
                print(f"No .nii.gz run files found in {func_dir}")
            return []

        if verbose:
            print(f"Found {len(run_paths)} BOLD runs for {subj}. Processing ROI timeseries...")

        saved_files = []

        for run_path in tqdm(run_paths, desc=f"Extracting ROI timeseries {subj}", disable=not verbose):
            run_name = run_path.name

            saved_file = self.extract_roi_timeseries(
                subj=subj,
                run_name=run_name,
                roi_atlas_path=roi_atlas_path,
                recompute=recompute,
                verbose=False
            )

            if saved_file:
                saved_files.append(saved_file)

        if verbose:
            print(f"Completed. Processed {len(saved_files)}/{len(run_paths)} runs successfully.")

        return saved_files

    def clean_ts_data(self, subj: str, atlas_name: str = None, confirm: bool = True, verbose: bool = True) -> None:
        """
        Deletes all previously computed ROI timeseries data for the given subject.

        This removes the entire `roi_timeseries` directory tree under:
            {warp_dir}/{subj}/roi_timeseries/

        If atlas_name is given, only deletes data under that atlas subfolder.

        Useful when changing ROI atlas, MNI template, warping parameters, etc.

        Parameters
        ----------
        subj : str
            Subject ID (e.g. 'subj01')
        atlas_name : str
            Name of the roi_atlas for which timeseries data was extracted
        confirm : bool, default True
            If True, asks for interactive confirmation before deletion
            Set to False for non-interactive / scripted usage
        verbose : str
            If True, print actions & information
        """
        base_dir = Path(self.deriv_dir) / subj / "roi_timeseries"

        if not base_dir.exists():
            if verbose:
                print(f"No timeseries directory found for {subj}: {base_dir}")
                print("Nothing to clean.")
            return

        if atlas_name:
            target_dirs = [base_dir / atlas_name]
        else:
            target_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

        for target_dir in target_dirs:
            if not target_dir.exists():
                continue

            # Count how much would be deleted (for user feedback)
            npz_files = list(target_dir.glob("**/*.npz"))
            n_files = len(npz_files)
            n_runs = len([d for d in target_dir.iterdir() if d.is_dir()])

            if n_files == 0:
                if verbose:
                    print(f"No .npz files found: {target_dir}")
                return

            if verbose:
                print(f"For atlas {atlas_name} found {n_files} ROI timeseries files across ~{n_runs} runs")
                print(f"Target directory to remove: {target_dir}")
                print(f"Total size ≈ {sum(f.stat().st_size for f in npz_files) / 1e6:.1f} MB")

            if confirm:
                response = input("Delete all ROI timeseries data? [y/N] ").strip().lower()
                if response not in ('y', 'yes'):
                    print("Cleanup cancelled.")
                    return

            # Perform deletion
            print("Deleting...", end=" ", flush=True)
            try:
                if target_dir.is_dir():
                    shutil.rmtree(target_dir, ignore_errors=False)
                if verbose:
                    print("done.")
            except Exception as e:
                if verbose:
                    print(f"\nError during deletion: {e}")
                    print("Some files may remain. Please check manually.")
                return

        if verbose:
            print(f"Successfully removed all ROI timeseries for subject {subj}.")
            print("You can now re-run extract_roi_timeseries() with the new settings.")

    ############################################
    # TRAIN TEST SPLIT & ENCRYPTION
    ############################################

    def parse_dataset(self, atlas_name):

        root = Path(self.base_data_dir)

        run_pattern = re.compile(r"roi_timeseries_session(\d+)_run(\d+)\.npz")

        records = []

        for file_path in root.rglob("roi_timeseries_session*_run*.npz"):

            parts = file_path.parts

            subj = next((p for p in parts if p.startswith("subj")), None)
            session = next((p for p in parts if p.startswith("session")), None)
            atlas = parts[-3]

            if atlas_name and atlas != atlas_name:
                continue

            m = run_pattern.fullmatch(file_path.name)
            if not m:
                continue

            sess, run = m.groups()

            records.append({
                "path": file_path,
                "subject": subj.replace("subj", ""),
                "session": sess,
                "run": run,
                "atlas": atlas
            })

        return records

    def build_encrypted_records(self, records, enc_key):
        """ Given parsed records encrypts them with given key """

        encrypted_records = []

        for r in records:
            enc_subj = encrypt_val(r["subject"], enc_key)
            enc_sess = encrypt_val(r["session"], enc_key)
            enc_run = encrypt_val(r["run"], enc_key)

            encrypted_records.append(
                {
                    **r,
                    "enc_subject": enc_subj,
                    "enc_session": enc_sess,
                    "enc_run": enc_run,
                }
            )

        return encrypted_records

    def write_structure(self, records, base_output):
        """ Given encrypted records, write the records """

        for r in records:
            dest_dir = (
                    Path(base_output)
                    / f"subj{r['enc_subject']}"
                    / f"session{r['enc_session']}"
            )

            dest_dir.mkdir(parents=True, exist_ok=True)

            filename = f"session{r['enc_session']}_run{r['enc_run']}.npz"

            dest = dest_dir / filename

            shutil.copy2(r["path"], dest)

    def subject_wise_split(self, records, test_size=0.2, random_state=44):
        """ Subject-wise split method. Partition the dataset into train and test sets by keeping
         subject data intact """

        subjects = sorted({r["subject"] for r in records})

        train_subj, test_subj = train_test_split(
            subjects,
            test_size=test_size,
            random_state=random_state,
        )

        train = [r for r in records if r["subject"] in train_subj]
        test = [r for r in records if r["subject"] in test_subj]

        return train, test

    def pooled_stratified_split(self, records, test_size=0.2, random_state=45):
        """ Pool split method. Pools all the subjects together and does the train test split by stratifying subjects"""

        subject_labels = [r["subject"] for r in records]

        train_idx, test_idx = train_test_split(
            range(len(records)),
            test_size=test_size,
            random_state=random_state,
            stratify=subject_labels,
        )

        train = [records[i] for i in train_idx]
        test = [records[i] for i in test_idx]

        return train, test

    def train_test_split(self, atlas_name=None, test_size=0.2, random_state=46, method="pool", enc_key=b"some_key"):
        """ Performs a train test split on the dataset. method == subj gives subject-wise partitioning,
        method == pool yields pooled subject stratified partitioning"""

        if atlas_name is None:
            atlas_name = Path(self.roi_atlas_path).stem

        output_root = self.deriv_dir

        records = self.parse_dataset(atlas_name)
        enc_records = self.build_encrypted_records(records, enc_key)

        if method == 'subj':  # subject-wise
            print("Performing subject-wise split...")
            train_sw, test_sw = self.subject_wise_split(enc_records, test_size=test_size, random_state=random_state)

            self.write_structure(train_sw, f"{output_root}/subjectwise/train")
            self.write_structure(test_sw, f"{output_root}/subjectwise/test")

        else:  # pooled stratified
            print("Performing pooled-stratified split...")
            train_ps, test_ps = self.pooled_stratified_split(enc_records, test_size=test_size,
                                                             random_state=random_state)

            self.write_structure(train_ps, f"{output_root}/pooled_stratified/train")
            self.write_structure(test_ps, f"{output_root}/pooled_stratified/test")

        print("Train-test split completed.")

def load_epi_mean(run_path):
    """Load 4D EPI and return mean image as 3D ANTs image"""
    epi_nib = nib.load(run_path)
    epi_data = epi_nib.get_fdata()  # shape: (X,Y,Z,T)
    epi_mean_data = np.mean(epi_data, axis=3)  # collapse time
    epi_mean = ants.from_numpy(epi_mean_data)
    epi_mean.set_spacing(epi_nib.header.get_zooms()[:3])
    return epi_mean


############################################
# ENCRYPTION
############################################

def encrypt_val(value: str, key: bytes, length: int = 10):
    digest = hmac.new(key, value.encode(), hashlib.sha256).digest()
    encoded = base64.urlsafe_b64encode(digest).decode()
    return encoded[:length]


def compute_fc(ts, z_trans=False):
    """ Given M x N timeseries data of M regions, return the functional connectivity"""
    ts = np.asarray(ts)

    if ts.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {ts.shape}")

    # ensure time is first dimension
    if ts.shape[0] > ts.shape[1]:
        ts = ts.T

    fc = np.corrcoef(np.nan_to_num(ts), rowvar=True)
    if z_trans:
        fc = np.arctanh(fc)

    return fc


# TODO: Fix ROI related issues
def compute_subject_fc(root_dir, data_key="timeseries", method='signed', roi_labels=None):
    """
    Loop through subject/session/run structure, compute FC matrices for each run,
    and store them per subject.

    Parameters
    ----------
    root_dir : str
        Root directory containing subject folders.
    data_key : str or None
        Key inside npz file containing the timeseries matrix.
        If None, the first array in the npz file is used.
    method : str
        compute_rs_graph method
    z_trans : bool
        If True, perform Fisher's z-transform on the FC matrices
    roi_labels : list-like
        Order according to roi_label list

    Returns
    -------
    subject_fc : dict
        Dictionary mapping subject -> dictionary of FC matrices (keys are runs)
    """

    subject_fc = {}

    for subj in tqdm(sorted(os.listdir(root_dir))):
        subj_path = os.path.join(root_dir, subj)
        if not os.path.isdir(subj_path):
            continue

        fc_dict = {}
        i = 0
        for ses in sorted(os.listdir(subj_path)):
            ses_path = os.path.join(subj_path, ses)
            if not os.path.isdir(ses_path):
                continue

            for file in sorted(os.listdir(ses_path)):

                if not file.endswith(".npz"):
                    continue

                run_name = file.split(".npz")[0]
                run_path = os.path.join(ses_path, file)
                data = np.load(run_path)

                if data_key is None:
                    ts = data[list(data.keys())[0]]
                else:
                    ts = data[data_key]

                run_roi_labels = np.array(data["roi_labels"])

                # Filter and fill missing ROIs with NaNs
                if roi_labels is not None:
                    roi_labels = np.array(roi_labels)

                    # map ROI label -> index in current run
                    label_to_idx = {lab: i for i, lab in enumerate(run_roi_labels)}

                    n_target = len(roi_labels)
                    T = ts.shape[1]

                    ts_all = np.full((n_target, T), np.nan)

                    for i, lab in enumerate(roi_labels):
                        if lab in label_to_idx:
                            ts_all[i, :] = ts[label_to_idx[lab], :]
                    ts = ts_all

                # if i < 5:
                #     print(ts.shape)
                # assume shape (regions, timepoints )
                # fc = compute_fc(ts, z_trans=z_trans)
                fc = compute_rs_graph({i: ts[i, :] for i in range(ts.shape[0])}, method=method)
                fc_dict[run_name] = fc
                i += 1

        subject_fc[subj] = fc_dict

    return subject_fc


def randomize_nsd_subj_fc(subj_fc, seed=None, verbose=True):
    """ Randomizes the roi filtered & ordered NSD subject functional connectivity to empty and full states. The output
     can be readily used with the process.pool_rs_graphs_activations """

    if seed is not None:
        np.random.seed(seed)

    all_graphs = {}
    if verbose:
        print("Subject FC matrix shapes before and after removing NaN entries:")

    for subj in subj_fc:
        all_graphs[subj] = {}
        subj_conn = []
        for run in subj_fc[subj]:
            fc = subj_fc[subj][run]
            if fc is None:
                continue

            subj_conn.append(fc)

        subj_conn = np.array(subj_conn)
        valid_mask = ~np.isnan(subj_conn).any(axis=(1, 2))
        subj_valid_conn = subj_conn[valid_mask]
        if verbose:
            print(f"{subj} Before: {subj_conn.shape}, After: {subj_valid_conn.shape}")

        if subj_valid_conn.shape[0] == 0:
            del all_graphs[subj]
            continue

        # Randomize Empty and Full states
        n = subj_valid_conn.shape[0]
        ind = np.random.permutation(n)
        empty_ind = ind[:n // 2 + 1]
        full_ind = ind[n // 2 + 1:]

        all_graphs[subj]['empty'] = subj_valid_conn[empty_ind]
        all_graphs[subj]['full'] = subj_valid_conn[full_ind]
    return all_graphs


def pool_rs_graphs(subj_fc, remove_nans=True):
    """ Pools the subject functional connectivity graphs into data matrix, state labels and subject id array"""
    mat_x = []
    states = []
    subjects = []

    for subj in subj_fc:

        for run in subj_fc[subj]:
            fc = subj_fc[subj][run]
            if fc is None:
                continue

            mat_x.append(fc)
            states.append(run)
            subjects.append(subj)

    mat_x = np.stack(mat_x, axis=0)
    states = np.array(states)
    subjects = np.array(subjects)

    # Remove nans
    if remove_nans:
        valid_mask = ~np.isnan(mat_x).any(axis=(1, 2))
        mat_x = mat_x[valid_mask, :, :]
        states = states[valid_mask]
        subjects = subjects[valid_mask]

    return mat_x, states, subjects


# ROI Creation

VISUAL_CORTEX_ROI_COORDS = {
    "V1": [(0, -90, 0)],
    "V2_L": [(-10, -85, 5)],
    "V2_R": [(10, -85, 5)],
    "V3_L": [(-15, -80, 10)],
    "V3_R": [(15, -80, 10)],
    "hV4_L": [(-25, -75, -10)],
    "hV4_R": [(25, -75, -10)],
    "V3A_L": [(-20, -85, 25)],
    "V3A_R": [(20, -85, 25)],
    "V3B_L": [(-25, -80, 30)],
    "V3B_R": [(25, -80, 30)],
    "LO1_L": [(-35, -75, -5)],
    "LO1_R": [(35, -75, -5)],
    "LO2_L": [(-40, -70, -5)],
    "LO2_R": [(40, -70, -5)],
    "VO1_L": [(-25, -70, -15)],
    "VO1_R": [(25, -70, -15)],
    "VO2_L": [(-30, -65, -15)],
    "VO2_R": [(30, -65, -15)],
    "PPA_L": [(-28, -45, -12)],
    "PPA_R": [(28, -45, -12)],
    "FFA_L": [(-40, -55, -15)],
    "FFA_R": [(40, -55, -15)],
}

VISUAL_CORTEX_ROI_RADIUS = {
    "V1": 5,
    "V2_L": 5,
    "V2_R": 5,
    "V3_L": 5,
    "V3_R": 5,

    "hV4_L": 6,
    "hV4_R": 6,
    "LO1_L": 6,
    "LO1_R": 6,
    "LO2_L": 6,
    "LO2_R": 6,

    "VO1_L": 6,
    "VO1_R": 6,
    "VO2_L": 6,
    "VO2_R": 6,
    "PPA_L": 6,
    "PPA_R": 6,
    "FFA_L": 6,
    "FFA_R": 6,
}

def create_spherical_roi_atlas(
    template_path,
    output_path,
    roi_coords,
    default_radius=5.0,
    radius_by_roi=None,
    overwrite_overlaps=False,
    labels_json_path=None,
):
    """
    Create one integer-labeled NIfTI atlas from spherical MNI ROIs.

    Atlas values:
        0 = background
        1, 2, 3, ... = ROI labels

    Parameters
    ----------
    template_path : str or Path
        MNI-space reference/template NIfTI.
    output_path : str or Path
        Output atlas path, e.g. visual_roi_atlas.nii.gz.
    roi_coords : dict
        {"ROI_NAME": [(x, y, z), ...], ...}
    default_radius : float
        Default radius in mm.
    radius_by_roi : dict or None
        Optional per-ROI radius override.
    overwrite_overlaps : bool
        If False, earlier ROIs keep overlapping voxels.
        If True, later ROIs overwrite earlier ones.
    labels_json_path : str or Path or None
        Optional path to save label metadata as JSON.

    Returns
    -------
    label_map : dict
        Mapping from label integer to ROI metadata.
    """
    template_path = Path(template_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    radius_by_roi = radius_by_roi or {}

    ref_img = nib.load(str(template_path))
    shape = ref_img.shape[:3]
    affine = ref_img.affine
    header = ref_img.header.copy()

    # Coordinates of every voxel in MNI/world space
    ijk = np.indices(shape).reshape(3, -1).T
    xyz = nib.affines.apply_affine(affine, ijk)

    atlas_flat = np.zeros(len(ijk), dtype=np.int16)
    label_map = {}

    for label_id, (roi_name, centers) in enumerate(roi_coords.items(), start=1):
        radius = float(radius_by_roi.get(roi_name, default_radius))

        roi_mask = np.zeros(len(ijk), dtype=bool)

        for center in centers:
            center = np.asarray(center, dtype=float)
            dist = np.linalg.norm(xyz - center, axis=1)
            roi_mask |= dist <= radius

        if overwrite_overlaps:
            atlas_flat[roi_mask] = label_id
        else:
            atlas_flat[roi_mask & (atlas_flat == 0)] = label_id

        label_map[label_id] = {
            "roi_name": roi_name,
            "centers_mni": [tuple(map(float, c)) for c in centers],
            "radius_mm": radius,
            "n_voxels": int(np.sum(atlas_flat == label_id)),
        }

    atlas = atlas_flat.reshape(shape)

    atlas_img = nib.Nifti1Image(atlas, affine, header)
    atlas_img.set_data_dtype(np.int16)
    nib.save(atlas_img, str(output_path))

    if labels_json_path is not None:
        labels_json_path = Path(labels_json_path)
        labels_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(labels_json_path, "w") as f:
            json.dump(label_map, f, indent=2)

    return label_map


