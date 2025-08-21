"""
circular_connectome_plot.py

Generates circular connectome plots (positive and negative polarity)
from significant connectivity results, using MNE-Python and Nilearn.

Author: Neuroimaging Tools Collective
License: MIT

Dependencies:
    - mne
    - nilearn
    - matplotlib
    - seaborn
    - pandas
    - numpy
    - tqdm

Required Files:
    - demo_csv: CSV file with subject IDs and metadata (must contain "SUB_ID" and "Age")
    - connectivity_matrices: Folder with connectivity matrices per subject (e.g., {subject_id}_atlas-aparc_mind.csv)
    - node_labels.txt: Text file with region names, one per line
    - significant_results_*.txt: One or more TSV files containing significant links
    - atlas_image.nii.gz: NIfTI image of the atlas (used to find cut coordinates)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from nilearn.image import load_img
from nilearn.plotting import find_parcellation_cut_coords
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout
import mne

# ----------------------------
# Entry Data Points (Edit These)
# ----------------------------

# Replace these with your actual file paths
demo_csv = "path/to/model.csv"
conn_folder = "path/to/connectivity_matrices/"
node_label_file = "path/to/node_labels.txt"
atlas_img_file = "path/to/atlas_image.nii.gz"
result_files = [
    "path/to/significant_results_file_1.txt",
    "path/to/significant_results_file_2.txt",
    "path/to/significant_results_file_3.txt"
]
output_folder = "path/to/output_folder/"

# Visualization parameters
LINK_THRESHOLD = 0.05
CIRCLE_COLORMAP = 'coolwarm'

# ----------------------------
# Processing Steps
# ----------------------------

# 1. Load subject labels
print("Loading labels...")
data_path = mne.datasets.sample.data_path()
subjects_dir = data_path / "subjects"
labels = mne.read_labels_from_annot("sample", parc="aparc", subjects_dir=subjects_dir)
label_colors = [label.color for label in labels]

# 2. Load demographic info
print("Loading demographic data...")
demo_df = pd.read_csv(demo_csv)
sub_ids = demo_df.loc[~demo_df["Age"].isna(), "SUB_ID"]

# 3. Load connectivity matrices
print("Loading connectivity matrices...")
con_mats = []
for sid in tqdm(sub_ids):
    mat_file = Path(conn_folder) / f"{sid}_atlas-aparc_mind.csv"
    con_mats.append(pd.read_csv(mat_file).iloc[:, 1:].to_numpy())
con_mats = np.array(con_mats)

# 4. Load region names
print("Loading region names...")
reg_names = np.loadtxt(node_label_file, dtype=str).tolist()

# 5. Compute circular layout
lh_names = [n for n in reg_names if n.startswith("ctx-lh-")]
rh_names = [n for n in reg_names if n.startswith("ctx-rh-")]
other_names = [n for n in reg_names if n not in lh_names + rh_names]
lh_order = sorted(lh_names)
rh_order = sorted(rh_names)
node_order = lh_order + rh_order[::-1] + other_names
name_to_index = {name: idx for idx, name in enumerate(reg_names)}
order_indices = [name_to_index[name] for name in node_order]
node_names_clean = [name.replace("ctx-lh-", "").replace("ctx-rh-", "") for name in node_order]
node_angles = circular_layout(node_names=node_order, node_order=node_order, start_pos=90, group_boundaries=[0, 34], group_sep=5)

# 6. Set node colors
half = len(node_order) // 2
node_colors = half * [(0.5, 0.5, 0)] + (len(node_order) - half) * [(0., 0.5, 0.5)]

# 7. Load atlas coords (not used in plotting here, but may be useful)
_ = find_parcellation_cut_coords(load_img(atlas_img_file))

# 8. Process each result file
for res_file in result_files:
    file_stem = Path(res_file).stem
    print(f"\nProcessing: {file_stem}")

    res_df = pd.read_csv(res_file, sep="\t")

    for polarity in ['positive', 'negative']:
        if polarity == 'positive':
            res_df_polar = res_df[res_df["strn"] > LINK_THRESHOLD]
        else:
            res_df_polar = res_df[res_df["strn"] < -LINK_THRESHOLD]

        print(f"Number of {polarity} connections to plot: {len(res_df_polar)}")

        if len(res_df_polar) == 0:
            print(f"Skipping {polarity} plot — no connections.")
            continue

        # Build adjacency matrix
        t_mat = np.zeros((len(reg_names), len(reg_names)))
        for _, row in res_df_polar.iterrows():
            i, j = int(row["3Drow"] - 1), int(row["3Dcol"] - 1)
            t_mat[i, j] = t_mat[j, i] = row["strn"]

        t_mat_ordered = t_mat[np.ix_(order_indices, order_indices)]
        vmax = np.max(np.abs(t_mat_ordered))

        # Hide labels for unconnected nodes
        node_names_plot = [node_names_clean[i] if np.any(t_mat_ordered[i, :]) else '' for i in range(len(node_names_clean))]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 8), facecolor="black", subplot_kw=dict(projection="polar"))
        ax.set_facecolor('black')
        ax.add_artist(plt.Circle((0, 0), radius=1.0, facecolor='black', zorder=0))

        plot_connectivity_circle(
            t_mat_ordered,
            node_names=node_names_plot,
            node_angles=node_angles,
            node_colors=node_colors,
            fontsize_names=12,
            colormap=CIRCLE_COLORMAP,
            vmin=-vmax, vmax=+vmax,
            colorbar=False,
            ax=ax,
            show=False
        )

        plt.tight_layout()
        out_path = Path(output_folder) / f"sig_links_{file_stem}_{polarity}.png"
        fig.savefig(out_path, dpi=300, facecolor='black')
        print(f"Saved: {out_path}")

print("\nAll result files processed. Visualization complete!")
