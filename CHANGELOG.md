# Changelog

## Version 2.0 — 2026-09-02

This release aligns the deposited data, scripts, figures, and documentation with the revised manuscript under review at *Nucleic Acids Research*.

- Replaced the stale exploratory Louvain community table with the weighted Girvan–Newman assignments used for the reported results. The deposited partitions use GCCM-derived impedance as the edge distance and retain the maximum-modularity partition for each state (Q = 0.79–0.82).
- Added the weighted Girvan–Newman recomputation script and regenerated community-membership, community-size, and modularity outputs.
- Added the complete 1,368-residue weighted betweenness-centrality matrix and Kneedle audit. The verified centrality elbow is rank 89, with peak centrality 0.0823457336; the earlier value of 98 was a preliminary caption error.
- Documented all five independently filtered MD evidence categories: 84 structural switches, 54 GCCM-variance hubs, 46 salt-bridge partner-switching residues, 90 hydrophobic partner-switching residues, and 89 centrality hubs. Their union contains 311 residues.
- Added `MD_key_residues_multi_evidence.csv`, containing the final 52 MD key residues supported by at least two independent categories, together with a script that recreates the table from `full_superset.csv`.
- Replaced "evolutionary constraint" with "VESM substitution intolerance" in code, tables, figures, and documentation. VESM uses an ESM2-3B backbone loaded with VESM distilled weights; its sign-inverted mean LLR is not direct sequence conservation.
- Corrected the Figure 7 aggregation to enforce one record per SpCas9 position. Historical synonyms for the structural-switch category had duplicated some rows in the plotting workflow. The corrected output contains exactly 1,368 positions and reproduces the manuscript's super-hub counts: Q1 = 13, Q2 = 15, Q3 = 13, and Q4 = 11.

The underlying MD trajectories and raw simulation outputs are unchanged from version 1.
