# data/

This folder contains **processed, analysis-ready data files** derived from the raw MD outputs.  
All files here are used directly as inputs to the plotting scripts in `scripts/`.

## Sub-folders

### `RMSF/`
| File | Description |
|------|-------------|
| `rmsf_summary_all_states.csv` | Per-residue RMSF (Å) aggregated across 5 replicas for each R-loop state. Columns: `Residue`, then `{state}_Mean` and `{state}_Std` for each of the seven states (6nt–18nt). |

### `Secondary_Structure_Shift/`
| File | Description |
|------|-------------|
| `All_Residues_SS_Shift.csv` | Ordered-structure fraction (%) per residue per state, plus summary statistics (`Max_Shift_Pct`, `Early_Avg`, `Late_Avg`, `Delta_Structure`, `Trend`). |
| `Classified_Switches.csv` | Subset of residues identified as allosteric switches (ordered-fraction shift ≥ 40 %). Classified as "Folder" (disorder → order) or "Unfolder" (order → disorder). |

### `Interaction_Network/`
| File | Description |
|------|-------------|
| `SB_Constitutive_Interactions.csv` | Salt-bridge interactions present at occupancy = 1.0 in **all** seven states. |
| `SB_All_Dynamic_Hubs_Ranked.csv` | Ranked list of salt-bridge hub residues showing state-dependent changes in connectivity. |
| `SB_Clustered_Data.csv` | Salt-bridge interaction binary occupancy matrix (per interaction × per state) with cluster assignment. |
| `SB_Stage_Classified_Data.csv` | Salt-bridge interactions classified by the R-loop stage at which they are active (e.g., "Stage 1 Only", "Constitutive"). |
| `SB_Stage_Summary.csv` | Summary statistics per stage-category for salt bridges: count, mean occupancies per state, and top contributing domains. |
| `Hydro_Constitutive_Interactions.csv` | Hydrophobic contacts present in all seven states. |
| `Hydro_All_Dynamic_Hubs_Ranked.csv` | Ranked dynamic hub residues for hydrophobic contacts. |
| `Hydro_Clustered_Data.csv` | Hydrophobic interaction binary occupancy matrix (per interaction × per state) with cluster assignment. |
| `Hydro_Stage_Classified_Data.csv` | Hydrophobic contacts classified by the R-loop stage at which they are active. |
| `Hydro_Stage_Summary.csv` | Summary statistics per stage-category for hydrophobic contacts: count, mean occupancies per state, and top contributing domains. |

### `GCCM_network/`
| File | Description |
|------|-------------|
| `gccm/{state}/network_G_nierzwicki.pkl` | NetworkX graph object encoding the GCCM-weighted contact network for each state (6-nt through 18-nt). Edge weights reflect generalised correlation × contact frequency. |
| `community_summary.txt` | Reported weighted Girvan–Newman community assignments for each residue and state. GCCM-derived impedance is used as the edge distance, and the maximum-modularity partition along the edge-betweenness dendrogram is retained (Q = 0.79–0.82). |
| `Betweenness_Ranked.csv` | Ranked list of residues by betweenness centrality across R-loop states. |
| `Dynamic_Variance_Hubs_Ranked.csv` | Residues ranked by variance in betweenness centrality across states (dynamic centrality hubs). |
| `Full_Betweenness_Centrality.csv` | Weighted betweenness centrality for all 1,368 residues in all seven states. |
| `Betweenness_Elbow.png/.pdf` | Kneedle check on the complete peak-centrality ranking; the verified elbow is rank 89 (peak centrality 0.0823457336). |
| `Elbow_Analysis.png/.pdf` | Elbow/threshold audit for the other independently filtered MD evidence categories. |

### `AI-validation-[CB,VESM]/`
| File / Sub-folder | Description |
|-------------------|-------------|
| `raw_pdb/` | Cleaned PDB structures for each R-loop state (6-nt through 18-nt, one chain), used as input to ProteinMPNN. |
| `CB_log_{n}_vs_{n+2}.txt` | ProteinMPNN conditional probability (CB) log files for each consecutive state-pair transition. |
| `CB_results_{n}_vs_{n+2}_proteinmpnn/` | `variants.csv` and `position_summary.csv` for each transition: per-variant CB scores and per-position summaries. |
| `SpCas9_VESM3B_full_variants.csv` | Full saturation-mutagenesis log-likelihood ratios from VESM (ESM2-3B backbone loaded with VESM distilled weights) for all 1,368 positions × 19 substitutions. |
| `SpCas9_VESM3B_full_position_summary.csv` | Per-position VESM summary. The reported sign-inverted mean LLR is model-inferred substitution intolerance, not direct sequence conservation. |
| `full_superset.csv` | Union of the five independently filtered MD evidence categories (311 unique residues). Column `n` records the number of supporting categories. |
| `MD_key_residues_multi_evidence.csv` | Final stringent MD key-residue set: the 52 residues supported by at least two independent MD categories (`n >= 2`). |
| `cb_vesm_dms_triple.py` | Script that reads `CB_VESM_discordance/cb_vesm_quadrant_table.csv` and `DMS_reference/spencer-zhang-data.csv` to produce the three-way CB × VESM × DMS validation figure. Run from the `AI-validation-[CB,VESM]/CB_VESM_discordance/` directory. |
| `DMS_reference/spencer-zhang-data.csv` | Deep mutational scanning (DMS) data for SpCas9 from Spencer & Zhang 2017 (see reference below). Used as an orthogonal experimental reference for AI-predicted functional importance scores. |

#### `AI-validation-[CB,VESM]/CB_VESM_discordance/`
| File | Description |
|------|-------------|
| `cb_vesm_quadrant_table.csv` | Per-position table with CB driving force, VESM substitution intolerance, MD hub annotations, and quadrant assignment (Q1–Q4). Produced by `scripts/Figure7/plot_CB_VESM_discordance.py`. |
| `cb_vesm_dms_merged.csv` | Merged table adding DMS tolerance columns to the quadrant table. Produced by `cb_vesm_dms_triple.py`. |
| `variant_level_merged.csv` | Variant-level (position × mutation) table merging CB scores, VESM LLR, DMS log₂FC, and MD annotations. |
| `per_domain_correlations.csv` | Spearman correlation between VESM substitution intolerance and DMS tolerance computed per SpCas9 domain. |

#### `AI-validation-[CB,VESM]/CB_directionality/`
| File | Description |
|------|-------------|
| `cb_state_bias_table.csv` | All position-transition records with CB bias z-score and direction assignment (state1_biased / neutral / state2_biased). Produced by `scripts/supplementary/AI-dms-validation/plot_CB_state_bias_2rows.py`. |
| `cb_direction_flippers.csv` | Residues that switch directional bias across transitions (wide format, one column per transition). |
| `cb_directionality_enrichment.csv` | Fisher's exact test enrichment of MD role categories within each directional group per transition. |

## Source of Processed Files

| Folder | Derived from |
|--------|-------------|
| `RMSF/` | `raw_data/RMSF/` — averaged across 5 replicas per state |
| `Secondary_Structure_Shift/` | `raw_data/Global_Secondary_Structure/Global_SS_RawData.csv` |
| `Interaction_Network/` | `raw_data/Interaction_Network/` — aggregated across states by `scripts/Figure5/Allosteric_Interaction_Pipeline_final.py` |
| `GCCM_network/` | `raw_data/GCCM_Network/` + trajectory files via `scripts/Figure6/build_network.py` |
| `AI-validation-[CB,VESM]/` | Scripts in `scripts/supplementary/AI-dms-validation/` and `scripts/Figure7/` |

## References

- Spencer, J.M., Zhang, X. Deep mutational scanning of S. pyogenes Cas9 reveals important functional domains. *Sci Rep* 7, 16836 (2017). https://doi.org/10.1038/s41598-017-17081-y
