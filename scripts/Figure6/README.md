# scripts/Figure6/ — GCCM Network Analysis

## Purpose
Builds and analyses the Generalised Correlation Contact Map (GCCM)-weighted residue interaction networks for each R-loop state. Extracts allosteric information-flow properties including betweenness centrality evolution, community structure, and optimal path impedance.

## Scripts

| Script | Execution order | Description |
|--------|-----------------|-------------|
| `build_network.py` | 1 (optional) | Constructs NetworkX graph objects from raw GCCM matrices and MD trajectory contact data. Saves one `.pkl` per state to `data/GCCM_network/gccm/`. **Skip this step** if pre-computed `.pkl` files are already present. Requires original trajectory files. |
| `plot_gccm_compact.py` | 2 | Reads the `.pkl` graphs and generates a compact summary figure of GCCM network statistics across states. |
| `community_girvan_newman_weighted.py` | 3a (optional recomputation) | Recomputes the reported weighted Girvan–Newman partitions from the precomputed graphs, using GCCM-derived impedance as the edge distance and retaining the maximum-modularity partition for each state. |
| `plot_community.py` | 3b | Reads the deposited weighted Girvan–Newman `community_summary.txt` and generates the community membership heatmap and community size plots. |
| `plot_path_impedance.py` | 4 | Computes and plots optimal allosteric path impedance for three functional residue groups across seven states. |
| `plot_centrality_evolution.py` | 5 | Plots the betweenness centrality evolution of key hub residues across R-loop states. |

## Inputs

| File | Description |
|------|-------------|
| `data/GCCM_network/gccm/{state}/network_G_nierzwicki.pkl` | Pre-computed NetworkX graphs (one per state) |
| `data/GCCM_network/community_summary.txt` | Weighted Girvan–Newman community assignments used for the reported figures (Q = 0.79–0.82) |
| `raw_data/GCCM_Network/{state}/gccm_full.dat` | Raw GCCM matrices (required only for `build_network.py`) |

## Outputs

| File | Description |
|------|-------------|
| `figures/Figure6/GCCM_Compact_Summary.png` | Network statistics summary |
| `figures/Figure6/Path_Impedance_Groups.png` | Allosteric path impedance by group |
| `data/GCCM_network/Betweenness_Ranked.csv` | Residues ranked by betweenness centrality across states |
| `data/GCCM_network/Dynamic_Variance_Hubs_Ranked.csv` | Residues ranked by variance in betweenness centrality (dynamic hubs) |
| Community heatmap / size figures | Saved to working directory by `plot_community.py` |
| Centrality evolution figure | Saved to working directory by `plot_centrality_evolution.py` |

## Execution

```bash
# build_network.py is the only script that requires trajectory files.
# Skip if pre-computed .pkl files already exist in data/GCCM_network/gccm/
cd scripts/Figure6
python build_network.py

# The remaining scripts use bare relative paths and must run from data/GCCM_network/
# (plot_gccm_compact.py → Dynamic_Variance_Hubs_Ranked.csv)
# (plot_centrality_evolution.py → DATA_DIR='.', reads {state}/network_G_nierzwicki.pkl)
# (plot_community.py → community_summary.txt)
cd data/GCCM_network

# Run from gccm/ subdirectory for pkl-dependent scripts (DATA_DIR='.')
cd gccm
python ../../../scripts/Figure6/plot_centrality_evolution.py
python ../../../scripts/Figure6/community_girvan_newman_weighted.py

# Run from data/GCCM_network/ for txt/csv-dependent scripts
cd ..
python ../../scripts/Figure6/plot_gccm_compact.py
python ../../scripts/Figure6/plot_community.py

# plot_path_impedance.py expects pathway files in DATA_DIR subdirectories;
# run from the directory containing the pathway data
python ../../scripts/Figure6/plot_path_impedance.py
```

## Notes on `build_network.py`
- Uses **24-core multiprocessing**; adjust `N_WORKERS` in the script to match your hardware.
- The script sets `OMP_NUM_THREADS=1` before importing NumPy to prevent thread over-subscription.
- GCCM edge weight = |generalised correlation coefficient| (Nierzwicki method), filtered by contact occupancy (cutoff 4.5 Å, ≥ 75 % of frames).
- The stored NetworkX `weight` attribute is GCCM-derived impedance, `-log(|GC|)`, and is used as the path distance for weighted betweenness and weighted Girvan–Newman decomposition.
- Louvain was examined during exploratory work but was not used to generate the deposited community assignments or the reported community figures.
