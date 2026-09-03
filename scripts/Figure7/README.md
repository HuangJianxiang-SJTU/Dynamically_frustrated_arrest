# scripts/Figure7/ — CB vs VESM Discordance Analysis

## Purpose
Integrates ProteinMPNN conditional-probability (CB) scores with VESM substitution-intolerance scores to identify residues showing discordance between state-conditioned sequence compatibility and model-inferred substitution tolerance. VESM uses an ESM2-3B backbone loaded with VESM distilled weights; its sign-inverted mean LLR is not a direct multiple-sequence-alignment conservation measure.

## Scripts

| Script | Description |
|--------|-------------|
| `plot_CB_VESM_discordance.py` | Generates a two-panel figure: (A) CB driving-force vs VESM substitution-intolerance scatter plot coloured by MD hub category and quadrant-annotated at medians; (B) quadrant composition bar plot (total / MD-annotated / multi-evidence MD key residue). |

## Inputs

| File | Description |
|------|-------------|
| `data/AI-validation-[CB,VESM]/CB_results_*_proteinmpnn/position_summary.csv` | Per-position CB score summaries for each state transition |
| `data/AI-validation-[CB,VESM]/SpCas9_VESM3B_full_position_summary.csv` | VESM per-position mean LLR scores |
| `data/AI-validation-[CB,VESM]/full_superset.csv` | Provenance table for the MD annotation fields embedded in the CB result tables; its five-category union contains 311 residues, and support from at least two categories defines the 52 multi-evidence MD key residues |
| `data/AI-validation-[CB,VESM]/DMS_reference/spencer-zhang-data.csv` | DMS reference data from Spencer & Zhang 2017 (see reference below) |

## Outputs

The 600-dpi PNG/PDF and the quadrant assignment table are written directly to `data/AI-validation-[CB,VESM]/CB_VESM_discordance/`. The table is used downstream by `data/AI-validation-[CB,VESM]/cb_vesm_dms_triple.py`.

## Execution

```bash
cd data/AI-validation-[CB,VESM]
python ../../scripts/Figure7/plot_CB_VESM_discordance.py
```

> **Note:** The script uses bare relative paths (`CB_results_*_proteinmpnn/`, `SpCas9_VESM3B_full_position_summary.csv`) and must therefore be executed from the `data/AI-validation-[CB,VESM]/` directory, not from `scripts/Figure7/`.  
> Outputs are written to the `CB_VESM_discordance/` subfolder automatically.

## References

- Spencer, J.M., Zhang, X. Deep mutational scanning of S. pyogenes Cas9 reveals important functional domains. *Sci Rep* 7, 16836 (2017). https://doi.org/10.1038/s41598-017-17081-y
