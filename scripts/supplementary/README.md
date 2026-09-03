# scripts/supplementary/ — Supplementary Figure Scripts & AI Validation Pipeline

## Purpose
Contains scripts for supplementary figures and the end-to-end AI validation pipeline that scores SpCas9 variants using ProteinMPNN (CB) and VESM, then integrates results with MD-derived hub data and published DMS data.

## Sub-folder Structure

```
supplementary/
├── AI-dms-validation/
│   ├── VESM-CB/
│   │   ├── 01_clean_cas9.py
│   │   ├── SpCas9_VESM_score.py
│   │   └── run_CB_SpCas9.py
│   ├── plot_CB_state_bias_2rows.py
│   └── plot_integrated_rank.py
└── build_md_key_residue_table.py
```

---

## AI Validation Pipeline (`AI-dms-validation/VESM-CB/`)

Run scripts in this order:

### Step 1 — `01_clean_cas9.py`
Prepares and cleans SpCas9 PDB structures for each R-loop state. Removes non-protein atoms, renumbers residues, and saves cleaned PDBs to `data/AI-validation-[CB,VESM]/raw_pdb/`.

### Step 2 — `SpCas9_VESM_score.py`
Runs saturation mutagenesis scoring using the VESM model (ESM2-3B backbone + Ntranos lab distilled weights).

**Usage:**
```bash
# Score only MD-identified switch residues (fast):
python SpCas9_VESM_score.py --mode subset

# Full saturation scan across all 1368 positions:
python SpCas9_VESM_score.py --mode all
```

**Hardware target:** 4× GPU (RTX-class, ≥ 24 GB VRAM each). Adjust batch size for smaller GPUs.  
**Output:** `data/AI-validation-[CB,VESM]/SpCas9_VESM3B_full_variants.csv` and `SpCas9_VESM3B_full_position_summary.csv`

### Step 3 — `run_CB_SpCas9.py`
Runs ProteinMPNN on each state's PDB to compute conditional probabilities (CB scores) for every residue. Generates `CB_log_{n}_vs_{n+2}.txt` files and `CB_results_*_proteinmpnn/` folders.

---

## Supplementary Figure Scripts

### `plot_CB_state_bias_2rows.py`
Generates a two-row supplementary panel showing the directional bias of ProteinMPNN CB scores across each R-loop state transition (Figure S — CB Directionality).

**Input:** `data/AI-validation-[CB,VESM]/CB_results_*_proteinmpnn/variants.csv`  
**Outputs:**
- `figures/supplenmentary/CB_Directionality.png`
- `data/AI-validation-[CB,VESM]/CB_directionality/cb_state_bias_table.csv`
- `data/AI-validation-[CB,VESM]/CB_directionality/cb_direction_flippers.csv`
- `data/AI-validation-[CB,VESM]/CB_directionality/cb_directionality_enrichment.csv`

### `plot_integrated_rank.py`
Generates the integrated CB + VESM + MD percentile rank figure (Figure S — integrated validation):
- Panel A: Scatter of CB percentile vs VESM substitution-intolerance percentile, dot size = number of supporting MD evidence categories, coloured by evidence count.
- Panel B: Top 30 residues by combined score, stacked bar of percentile contributions.

**Inputs:**
- `data/AI-validation-[CB,VESM]/CB_results_*_proteinmpnn/position_summary.csv`
- `data/AI-validation-[CB,VESM]/SpCas9_VESM3B_full_position_summary.csv`
- `data/Interaction_Network/SB_All_Dynamic_Hubs_Ranked.csv`

**Output:** `Integrated_Rank.png/.pdf` in the working directory.

### `build_md_key_residue_table.py`
Applies the revision's cross-method filter to `full_superset.csv`. The five category sizes are 84, 54, 46, 90, and 89; their union contains 311 residues. Requiring at least two supporting categories produces `MD_key_residues_multi_evidence.csv` with 52 residues.

## Execution

```bash
# Step 1–3: AI scoring pipeline
cd scripts/supplementary/AI-dms-validation/VESM-CB
python 01_clean_cas9.py
python SpCas9_VESM_score.py --mode all
python run_CB_SpCas9.py
```

> **Note:** `plot_CB_state_bias_2rows.py` and `plot_integrated_rank.py` use bare relative paths (`CB_results_*_proteinmpnn/`, `SpCas9_VESM3B_full_position_summary.csv`) and must be run from `data/AI-validation-[CB,VESM]/`:

```bash
cd data/AI-validation-[CB,VESM]
python ../../scripts/supplementary/build_md_key_residue_table.py
python ../../scripts/supplementary/AI-dms-validation/plot_CB_state_bias_2rows.py
python ../../scripts/supplementary/AI-dms-validation/plot_integrated_rank.py
```

> **`plot_integrated_rank.py`** also requires the full-superset MD annotation file at `full_superset.csv` (bare filename, resolved from `data/AI-validation-[CB,VESM]/` working directory).

> **Figure 10** — three-way CB × VESM × DMS validation — is produced by a standalone script at `data/AI-validation-[CB,VESM]/cb_vesm_dms_triple.py`. Run it from `data/AI-validation-[CB,VESM]/CB_VESM_discordance/` after `scripts/Figure7/plot_CB_VESM_discordance.py` has generated `cb_vesm_quadrant_table.csv`:

```bash
cd data/AI-validation-[CB,VESM]/CB_VESM_discordance
python ../cb_vesm_dms_triple.py
```

## Requirements

```
torch
transformers
huggingface_hub
pandas
numpy
matplotlib
scipy
adjustText
```

> Note: `SpCas9_VESM_score.py` sets `HF_ENDPOINT=https://hf-mirror.com` for model download. Change this to `https://huggingface.co` if you are not in a region where the mirror is needed.
