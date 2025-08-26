"""
MIND Pipeline Adaptation for VBM Data
"""

import os
import argparse
import logging
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Optional: embed if not external
# from mind_helpers import calculate_mind_network

def calculate_mind_network(df, feature_cols, region_list, resample=True):
    # Dummy placeholder for original helper
    # Replace this with the actual logic or import it from an internal module
    import itertools
    matrix = pd.DataFrame(
        np.random.rand(len(region_list), len(region_list)),
        index=region_list,
        columns=region_list
    )
    return matrix

def load_lut(lut_path):
    lut = {}
    with open(lut_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                label = int(parts[0])
                region_name = " ".join(parts[1:])
                lut[label] = region_name
    return lut

def load_subject_ids(subjects_file):
    with open(subjects_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def extract_region_voxels(vbm_image_path, parcellation_image_path, lut):
    try:
        vbm_img = nib.load(vbm_image_path)
        parcellation_img = nib.load(parcellation_image_path)
        vbm_data = vbm_img.get_fdata()
        parcellation_data = parcellation_img.get_fdata()
    except Exception as e:
        logging.error(f"Image load failure: {e}")
        return pd.DataFrame(columns=["Value", "Label"])

    voxels = []
    for label_id, region_name in lut.items():
        mask = np.isclose(parcellation_data, label_id, atol=1e-3)
        region_voxels = vbm_data[mask]
        if region_voxels.size > 0:
            voxels.append(pd.DataFrame({'Value': region_voxels, 'Label': region_name}))

    return pd.concat(voxels) if voxels else pd.DataFrame(columns=["Value", "Label"])

def adjust_matrix_shape(matrix, expected_shape=(82, 82)):
    rows_needed = expected_shape[0] - matrix.shape[0]
    cols_needed = expected_shape[1] - matrix.shape[1]

    if rows_needed > 0:
        new_rows = pd.DataFrame(0, index=[f"Missing_Row_{i}" for i in range(rows_needed)], columns=matrix.columns)
        matrix = pd.concat([matrix, new_rows])
    if cols_needed > 0:
        new_cols = pd.DataFrame(0, index=matrix.index, columns=[f"Missing_Col_{i}" for i in range(cols_needed)])
        matrix = pd.concat([matrix, new_cols], axis=1)

    return matrix.fillna(0).infer_objects(copy=False)

def save_matrix(subject_id, matrix, output_dir):
    output_file = os.path.join(output_dir, f"ACE-82x82-{subject_id}.csv")
    matrix.to_csv(output_file)
    logging.info(f"Saved matrix: {output_file}")

def process_subject(subject_id, freesurfer_dir, vbm_dir, output_dir, lut):
    vbm_path = os.path.join(vbm_dir, f"mwp1sub-{subject_id}_T1w.nii_output.mgz")
    parc_path = os.path.join(freesurfer_dir, f"sub-{subject_id}/mri/aparc+aseg.mgz")

    if not (os.path.exists(vbm_path) and os.path.exists(parc_path)):
        logging.warning(f"Missing data for {subject_id}. Skipping.")
        return

    voxel_data = extract_region_voxels(vbm_path, parc_path, lut)
    if voxel_data.empty:
        logging.warning(f"No voxel data for {subject_id}. Skipping.")
        return

    try:
        mind = calculate_mind_network(voxel_data, feature_cols=['Value'], region_list=list(lut.values()), resample=True)
    except Exception as e:
        logging.error(f"MIND computation failed for {subject_id}: {e}")
        return

    valid_labels = [label.strip() for label in lut.values() if label.strip() in mind.index]
    nested = pd.DataFrame(index=valid_labels, columns=valid_labels)
    nested.loc[:, :] = mind.loc[valid_labels, valid_labels].values

    adjusted_matrix = adjust_matrix_shape(nested)
    save_matrix(subject_id, adjusted_matrix, output_dir)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run MIND network computation on neuroimaging data.")
    parser.add_argument('--freesurfer_dir', required=True, help='Path to FreeSurfer directory')
    parser.add_argument('--vbm_dir', required=True, help='Path to VBM image directory')
    parser.add_argument('--output_dir', required=True, help='Where to save the output matrices')
    parser.add_argument('--lut', required=True, help='Path to label lookup table (LUT)')
    parser.add_argument('--subjects', required=True, help='Text file with subject IDs')
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    args = parse_arguments()

    lut = load_lut(args.lut)
    subject_ids = load_subject_ids(args.subjects)

    logging.info(f"Processing {len(subject_ids)} subjects in parallel...")
    Parallel(n_jobs=-1)(
        delayed(process_subject)(sid, args.freesurfer_dir, args.vbm_dir, args.output_dir, lut)
        for sid in subject_ids
    )
    logging.info("All done.")

if __name__ == "__main__":
    main()
