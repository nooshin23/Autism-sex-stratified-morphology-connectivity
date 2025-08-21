"""
PCA-Based CBCL Subscale Analysis and Regression with Brain Connectivity
=======================================================================

What this script does
---------------------
1) Load and merge:
   - Demographics CSV (SUB_ID, Age, Cohort, Gender, ICV, ...)
   - CBCL Excel (with named columns for subscales)
2) Load subject-level brain connectivity matrices from a folder
3) Save Pearson/Spearman correlation matrices + heatmaps for CBCL
4) PCA on CBCL subscales (per sex), then OLS regression:
   connectivity_strength ~ PCA components + (Age, Site ID, ICV)
5) Save PCA explained variance, regression summary, and observed vs predicted plots

Inputs you must provide (replace paths below)
---------------------------------------------
- BASE_PATH/
    ├─ demographics_file.csv        # Subject metadata (includes SUB_ID, Age, Cohort, Gender, ICV)
    ├─ cbcl_scores.xlsx             # CBCL subscales with column names (see CBCL_COL_NAME_MAP below)
    ├─ sig_edges_male.csv           # columns: 3Drow, 3Dcol  (significant edges for males)
    ├─ sig_edges_female.csv         # columns: 3Drow, 3Dcol  (significant edges for females)
    └─ connectivity_folder/         # CSV per subject: {SUB_ID}_atlas-aparc_mind.csv; first col is index/label, the rest is a square matrix

Outputs (auto-created)
----------------------
- BASE_PATH/results_pca/
    - cbcl_pearson_correlation_matrix.csv, cbcl_spearman_correlation_matrix.csv
    - cbcl_pearson_heatmap.png, cbcl_spearman_heatmap.png
    - M_pca_explained_variance.csv, F_pca_explained_variance.csv
    - M_pca_variance_plot.png, F_pca_variance_plot.png
    - M_pca_regression_summary_with_covariates.txt, F_pca_regression_summary_with_covariates.txt
    - M_pca_predicted_vs_observed_with_covariates.png, F_pca_predicted_vs_observed_with_covariates.png
"""

# --- Libraries ---
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# =========================
#   USER CONFIGURE THESE
# =========================
BASE_PATH = "path/to/your_project_folder"

# Input files/folders (describe content, replace with your real paths in the same folder or elsewhere)
MODEL_PATH = os.path.join(BASE_PATH, "demographics_file.csv")        # CSV with columns like: SUB_ID, Age, Cohort, Gender, ICV, ...
CBCL_EXCEL_PATH = os.path.join(BASE_PATH, "cbcl_scores.xlsx")        # Excel with CBCL subscale columns (see CBCL_COL_NAME_MAP)
RES_MALE_PATH = os.path.join(BASE_PATH, "sig_edges_male.csv")        # CSV with columns: 3Drow, 3Dcol
RES_FEMALE_PATH = os.path.join(BASE_PATH, "sig_edges_female.csv")    # CSV with columns: 3Drow, 3Dcol
CONNECTIVITY_FOLDER = os.path.join(BASE_PATH, "connectivity_folder") # Folder of subject matrices named: {SUB_ID}_atlas-aparc_mind.csv

# Output folder
OUTPUT_DIR = os.path.join(BASE_PATH, "results_pca")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
#   COLUMN NAME SETTINGS
# =========================
# We map common CBCL labels (case-insensitive) -> standardized names used downstream.
# Edit the keys to match *your* Excel header names if different.
CBCL_COL_NAME_MAP = {
    # left side = any of these label variants in your Excel; right side = our internal standardized column
    "anxiety problems": "cbcl_anxiety",
    "anxious/depressed": "cbcl_anxiety",           # sometimes appears as syndrome scale
    "affective problems": "cbcl_affective",
    "depressed": "cbcl_affective",
    "adhd problems": "cbcl_adhd",
    "attention problems": "cbcl_adhd",
    "oppositional defiant problems": "cbcl_oppositional",
    "oppositional": "cbcl_oppositional",
    "conduct problems": "cbcl_conduct",
    "conduct": "cbcl_conduct",
    "internalizing problems": "cbcl_internal",
    "internalizing": "cbcl_internal",
}

# Acceptable ID column names in the CBCL Excel (first column with subject/site IDs).
CBCL_ID_CANDIDATES = ["site_id", "subject", "subject_id", "sub_id", "participant_id", "id"]

# Demographics must contain these columns at minimum
REQUIRED_DEMOGRAPHIC_COLS = ["SUB_ID", "Age", "Cohort", "Gender", "ICV"]

# The final standardized CBCL columns we'll use downstream (right-hand side of the map)
CBCL_COLS = ["cbcl_anxiety", "cbcl_affective", "cbcl_adhd", "cbcl_oppositional", "cbcl_conduct", "cbcl_internal"]

# =========================
#        HELPERS
# =========================
def _normalize_col(s: str) -> str:
    """Lowercase, strip, and collapse spaces for robust column matching."""
    return " ".join(str(s).strip().lower().split())

def _find_header_row_excel(xl_path: str, candidates=(0, 1, 2, 10, 11, 12, 13)):
    """
    Try multiple header rows to find one that contains at least the ID column or any known CBCL columns.
    Returns (df, used_header_row).
    """
    for hdr in candidates:
        try:
            df = pd.read_excel(xl_path, header=hdr)
        except Exception as e:
            logger.warning(f"Failed reading Excel with header={hdr}: {e}")
            continue

        normalized = {_normalize_col(c): c for c in df.columns}
        # If we see any CBCL names or a plausible ID, accept this header row.
        has_id = any(name in normalized for name in CBCL_ID_CANDIDATES)
        has_cbcl = any(name in normalized for name in map(_normalize_col, CBCL_COL_NAME_MAP.keys()))
        if has_id or has_cbcl:
            logger.info(f"Using Excel header row {hdr}")
            return df, hdr

    raise ValueError("Could not find a suitable header row in the CBCL Excel. "
                     "Please ensure your CBCL file has column names and update 'CBCL_COL_NAME_MAP' if needed.")

def _rename_cbcl_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename CBCL columns based on CBCL_COL_NAME_MAP (case-insensitive, whitespace-normalized).
    Returns a new DataFrame with standardized CBCL cols plus a SITE_ID column.
    """
    # Build a mapping from current columns -> normalized form
    current_cols_norm = {_normalize_col(c): c for c in df.columns}

    # Find the ID column in CBCL sheet and standardize to 'SITE_ID'
    site_id_col = None
    for cand in CBCL_ID_CANDIDATES:
        if cand in current_cols_norm:
            site_id_col = current_cols_norm[cand]
            break
    if site_id_col is None:
        # fallback: assume first column is the ID
        site_id_col = df.columns[0]
        logger.warning(f"No explicit ID column found in CBCL sheet; assuming first column '{site_id_col}' is SITE_ID.")

    # Start with SITE_ID
    out = pd.DataFrame()
    out["SITE_ID"] = df[site_id_col].astype(str)

    # Map CBCL columns by name
    # Create reverse index for quick lookup of source column by normalized key
    for raw_label, std_name in CBCL_COL_NAME_MAP.items():
        raw_norm = _normalize_col(raw_label)
        if raw_norm in current_cols_norm:
            source_col = current_cols_norm[raw_norm]
            out[std_name] = pd.to_numeric(df[source_col], errors="coerce")

    # Ensure all required CBCL_COLS are present (even if missing -> NaN)
    for col in CBCL_COLS:
        if col not in out.columns:
            out[col] = np.nan

    return out

# =========================
#     CORE PIPELINE
# =========================
def load_demographic_data() -> pd.DataFrame:
    """
    Demographics CSV must include at least: SUB_ID, Age, Cohort, Gender, ICV.
    We derive SITE_ID by stripping a leading 'sub-' prefix if present.
    Filters to ASD cohort.
    """
    df = pd.read_csv(MODEL_PATH)
    missing = [c for c in REQUIRED_DEMOGRAPHIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Demographics CSV missing required columns: {missing}")

    df = df.loc[~df["Age"].isna()].copy()
    df["SITE_ID"] = df["SUB_ID"].astype(str).str.replace("sub-", "", regex=False)
    df = df[df["Cohort"].astype(str).str.upper() == "ASD"].copy()
    return df

def load_cbcl_data() -> pd.DataFrame:
    """
    Load CBCL subscales by column *names*, not hard-coded indices.
    Tries multiple header rows to locate labels; renames to standardized CBCL_COLS.
    Returns a DataFrame with columns: SITE_ID + CBCL_COLS
    """
    df_raw, _used_header = _find_header_row_excel(CBCL_EXCEL_PATH)
    cbcl_std = _rename_cbcl_columns(df_raw)
    # Make sure ID is string for merging
    cbcl_std["SITE_ID"] = cbcl_std["SITE_ID"].astype(str)
    return cbcl_std

def merge_datasets(demo: pd.DataFrame, cbcl: pd.DataFrame) -> pd.DataFrame:
    """Merge demographics and CBCL on SITE_ID; drop rows missing CBCL subscales."""
    merged = demo.merge(cbcl, on="SITE_ID", how="inner", validate="many_to_one")
    merged = merged.dropna(subset=CBCL_COLS)
    if merged.empty:
        raise ValueError("Merged dataset is empty after dropping missing CBCL rows. "
                         "Check that SITE_IDs align between demographics and CBCL.")
    return merged

def load_valid_connectivity_matrices(subject_ids):
    """
    For each SUB_ID, look for {SUB_ID}_atlas-aparc_mind.csv in CONNECTIVITY_FOLDER.
    Assumes first column is a label/index, and the remaining columns form a square matrix.
    Returns: (np.array of matrices [N, R, R], valid_ids list in same order)
    """
    con_mats = []
    valid_ids = []
    for sid in tqdm(subject_ids, desc="Loading connectivity matrices"):
        file_path = os.path.join(CONNECTIVITY_FOLDER, f"{sid}_atlas-aparc_mind.csv")
        if os.path.exists(file_path):
            mat_df = pd.read_csv(file_path)
            if mat_df.shape[1] < 2:
                logger.warning(f"Matrix file has <2 columns (bad format): {file_path}")
                continue
            mat = mat_df.iloc[:, 1:].to_numpy()
            # optional sanity check: square
            if mat.shape[0] != mat.shape[1]:
                logger.warning(f"Matrix is not square ({mat.shape}) in {file_path}; skipping.")
                continue
            con_mats.append(mat)
            valid_ids.append(sid)
        else:
            logger.debug(f"Missing connectivity file for {sid}: {file_path}")
    if not con_mats:
        raise ValueError("No valid connectivity matrices found. Check CONNECTIVITY_FOLDER and filenames.")
    return np.array(con_mats), valid_ids

def compute_connectivity_strength(con_mats, sig_edges):
    """
    Sum weights on significant edges for each subject matrix.
    sig_edges: list of (i, j) zero-based indices into matrix.
    """
    strengths = []
    for mat in con_mats:
        total = 0.0
        for i, j in sig_edges:
            total += mat[i, j]
        strengths.append(total)
    return strengths

def save_cbcl_correlations(df: pd.DataFrame):
    """Compute and save Pearson and Spearman correlation matrices + heatmaps for CBCL subscales."""
    pearson_corr = df[CBCL_COLS].corr()
    spearman_corr = df[CBCL_COLS].corr(method="spearman")

    pearson_corr.to_csv(os.path.join(OUTPUT_DIR, "cbcl_pearson_correlation_matrix.csv"))
    spearman_corr.to_csv(os.path.join(OUTPUT_DIR, "cbcl_spearman_correlation_matrix.csv"))

    for method, matrix in [("Pearson", pearson_corr), ("Spearman", spearman_corr)]:
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
        plt.title(f"CBCL {method} Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"cbcl_{method.lower()}_heatmap.png"))
        plt.close()

def perform_pca_and_regression(df: pd.DataFrame, group: str, edge_path: str, con_mats: np.ndarray):
    """
    Per-sex pipeline:
      - Filter to Gender == group ('M' or 'F')
      - Compute connectivity_strength via sig edges
      - PCA on standardized CBCL vars
      - OLS: connectivity_strength ~ [PCA components] + [scaled Age, SITE_ID (encoded), ICV]
      - Save explained variance, regression summary, and predicted vs observed plot
    """
    df_g = df[df["Gender"].astype(str).str.upper() == group.upper()].copy()
    if df_g.empty:
        logger.warning(f"No rows for gender group '{group}'. Skipping.")
        return

    # Align connectivity matrices by row order of df_g within original merged df
    indices = df_g.index.to_list()
    con_group = con_mats[indices, :, :]

    # Load significant edges (assume columns: 3Drow, 3Dcol are zero-based indices)
    edges_df = pd.read_csv(edge_path)
    if not {"3Drow", "3Dcol"}.issubset(set(edges_df.columns)):
        raise ValueError(f"{edge_path} must contain columns '3Drow' and '3Dcol'.")
    sig_edges = list(zip(edges_df["3Drow"].astype(int), edges_df["3Dcol"].astype(int)))

    # Compute connectivity strength
    df_g["connectivity_strength"] = compute_connectivity_strength(con_group, sig_edges)

    # Prepare CBCL data
    X_cbcl = df_g[CBCL_COLS].apply(pd.to_numeric, errors="coerce")
    valid_index = X_cbcl.dropna().index
    df_g = df_g.loc[valid_index]
    X_cbcl = X_cbcl.loc[valid_index]
    y = df_g["connectivity_strength"].astype(float)

    # Standardize CBCL
    scaler_cbcl = StandardScaler()
    X_cbcl_scaled = scaler_cbcl.fit_transform(X_cbcl)

    # PCA (keep all components by default; downstream users can select subset using explained variance CSV)
    pca = PCA()
    X_pca = pca.fit_transform(X_cbcl_scaled)

    # Save explained variance
    explained_var = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "ExplainedVariance": pca.explained_variance_ratio_
    })
    explained_var.to_csv(os.path.join(OUTPUT_DIR, f"{group}_pca_explained_variance.csv"), index=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(x="PC", y="ExplainedVariance", data=explained_var, color="#4C78A8")
    plt.title(f"Explained Variance by PCA Components ({group})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{group}_pca_variance_plot.png"))
    plt.close()

    # Covariates: Age, SITE_ID (factorized numeric), ICV
    covariates = df_g[["Age", "SITE_ID", "ICV"]].copy()
    covariates["SITE_ID"] = pd.factorize(covariates["SITE_ID"].astype(str))[0]
    covariates = covariates.apply(pd.to_numeric, errors="coerce")
    covariates_scaled = StandardScaler().fit_transform(covariates)

    # Combine PCA + Covariates
    X_full = np.hstack([X_pca, covariates_scaled])
    Xc = sm.add_constant(X_full, has_constant="add")

    # OLS
    model = sm.OLS(y, Xc).fit()

    with open(os.path.join(OUTPUT_DIR, f"{group}_pca_regression_summary_with_covariates.txt"), "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    y_pred = model.predict(Xc)
    r2 = model.rsquared
    pval = model.f_pvalue

    plt.figure(figsize=(5.5, 5.5))
    sns.regplot(x=y, y=y_pred, ci=95, scatter_kws={"alpha": 0.6})
    plt.xlabel("Observed Connectivity Strength")
    plt.ylabel("Predicted Connectivity Strength")
    plt.title(f"{group}: R²={r2:.3f}, p={pval:.3g}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{group}_pca_predicted_vs_observed_with_covariates.png"))
    plt.close()

    logger.info(f"[{group}] OLS complete | R²={r2:.3f}, p={pval:.3g} | n={len(y)}")

# =========================
#        MAIN
# =========================
if __name__ == "__main__":
    # 1) Load inputs
    demo = load_demographic_data()
    cbcl = load_cbcl_data()

    # 2) Merge on SITE_ID
    merged = merge_datasets(demo, cbcl)

    # 3) Save CBCL correlations
    save_cbcl_correlations(merged)

    # 4) Load connectivity matrices for the subjects that survived merging
    con_mats, valid_ids = load_valid_connectivity_matrices(merged["SUB_ID"].astype(str).tolist())

    # 5) Keep only rows with available connectivity files
    merged = merged[merged["SUB_ID"].astype(str).isin(valid_ids)].reset_index(drop=True)

    # 6) Per-sex PCA + OLS
    perform_pca_and_regression(merged, "M", RES_MALE_PATH, con_mats)
    perform_pca_and_regression(merged, "F", RES_FEMALE_PATH, con_mats)
