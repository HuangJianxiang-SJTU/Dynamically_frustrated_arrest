# raw_data/

This folder contains **raw computation outputs** produced directly by MD simulation tools and analysis pipelines, before any aggregation or post-processing.

## Sub-folders

### `RMSF/`
Per-replica RMSF files in GROMACS `.xvg` format.

```
RMSF/
└── {state}/          # 6-nt, 8-nt, 10-nt, 12-nt, 14-nt, 16-nt, 18-nt
    ├── replica1/rmsf.xvg
    ├── replica2/rmsf.xvg
    ├── replica3/rmsf.xvg
    ├── replica4/rmsf.xvg
    └── replica5/rmsf.xvg
```

Each `rmsf.xvg` file is the direct GROMACS `gmx rmsf` output: two columns, residue index and RMSF (nm).  
These 35 files (7 states × 5 replicas) are averaged to produce `data/RMSF/rmsf_summary_all_states.csv`.

### `Interaction_Network/`
Per-state interaction files from the contact analysis pipeline.

```
Interaction_Network/
└── {state}/          # 6-nt through 18-nt
    ├── competitive_salt_bridges.csv
    └── competitive_hydrophobic.csv
```

Each CSV contains pairwise residue interactions with per-frame occupancy values.  
Columns: `Pos_Residue`, `Target_Residue`, `Interaction_Type`, `Occupancy`.  
These are aggregated by `scripts/Figure5/Allosteric_Interaction_Pipeline_final.py`.

### `GCCM_Network/`
Raw Generalised Correlation Contact Map (GCCM) matrices.

```
GCCM_Network/
└── {state}/          # 6-nt through 18-nt
    └── gccm_full.dat
```

Each `gccm_full.dat` file is a plain-text square matrix (1368 × 1368) of pairwise generalised correlation coefficients computed with the Nierzwicki method from MD trajectories.  
These are read by `scripts/Figure6/build_network.py` to construct the NetworkX graphs stored in `data/GCCM_network/`.

### `Global_Secondary_Structure/`
| File | Description |
|------|-------------|
| `Global_SS_RawData.csv` | Per-residue ordered-structure fraction (%) across all 7 states. Columns: `Residue_Number`, `Domain`, `{state}_Ordered_Pct` for each of the seven states. |

This file is processed by `scripts/Figure4/plot_ss_heatmap.py` and `scripts/Figure4/classify_switches_final.py`.

---

## Notes

- MD trajectories (`.xtc` / `.trr`) and topology files (`.tpr`) are **not** included in this repository due to file size. They are deposited on Zenodo under the version-independent DOI [10.5281/zenodo.21065807](https://doi.org/10.5281/zenodo.21065807).
- The `build_network.py` script in `scripts/Figure6/` requires the original trajectory files to regenerate the `.pkl` graphs; all pre-computed `.pkl` files are already provided in `data/GCCM_network/gccm/`.
