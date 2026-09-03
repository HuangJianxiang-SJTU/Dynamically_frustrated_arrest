"""
Figure 10: Three-way cross-check — CB × VESM × DMS (Spencer & Zhang 2017)
--------------------------------------------------------------------------
External validation of the Q1/Q4 partition using experimental mutational
tolerance from a deep mutational scan of SpCas9 (Spencer & Zhang, Sci Rep 2017).

Key axis:
    DMS_tolerance = mean(Log2 Fold Change after Positive Selection) per position.
        > 0  -> substitutions RETAIN on-target activity (tolerant)
        < 0  -> substitutions DEPLETED under positive selection (deleterious)

Hypotheses being tested:
    H1 (validation): Q1 residues (high CB, high VESM) show LOW DMS tolerance.
    H2 (engineering): Q4 residues (high CB, low VESM) show HIGH DMS tolerance
                      i.e. they are mutatable AND MD-selective, the target profile
                      for fidelity engineering.

Outputs:
    - cb_vesm_dms_merged.csv           (per-position triple-axis table)
    - q4_priority_triple_convergent.csv (triple-convergent engineering set)
    - CB_VESM_DMS.png / .pdf

Style: Helvetica, no panel titles, 600 DPI, top/right spines off.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import mannwhitneyu, spearmanr, kruskal

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# ==========================================
# CONFIG
# ==========================================
QUADRANT_TABLE = "cb_vesm_quadrant_table.csv"      # produced by Figure 7 script
DMS_FILE       = "../DMS_reference/spencer-zhang-data.csv"  # Spencer & Zhang 2017 supp

# For position-level tolerance, require at least this many mutations
MIN_N_MUTATIONS = 1        # dataset is already sparse (~2 per position)
# Minimum mutations to use the stronger filter in the priority set
PRIORITY_MIN_N  = 1

C_Q1 = '#E15759'   # core drivers (validation)
C_Q2 = '#4E79A7'   # functional invariants
C_Q3 = '#BBBBBB'   # background
C_Q4 = '#F28E2B'   # plastic drivers (engineering)

# Domain boundary definitions (consistent with manuscript)
DOMAIN_BOUNDS = [
    (1,    59,   'RuvC-I'),
    (60,   94,   'BH'),
    (95,   176,  'REC1-A'),
    (177,  305,  'REC2'),
    (306,  495,  'REC1-B'),
    (496,  717,  'REC3'),
    (718,  764,  'RuvC-II'),
    (765,  780,  'L1'),
    (781,  905,  'HNH'),
    (906,  918,  'L2'),
    (919,  1100, 'RuvC-III'),
    (1101, 1368, 'PI'),
]

def assign_domain(pos):
    for lo, hi, name in DOMAIN_BOUNDS:
        if lo <= pos <= hi:
            return name
    return 'Unknown'

# ==========================================
# 1. LOAD & AGGREGATE DMS PER POSITION
# ==========================================
dms = pd.read_csv(DMS_FILE)
dms.columns = dms.columns.str.strip()

# Drop synonymous if any ended up in the table (shouldn't, but safe)
dms = dms[dms['Synonymous Mutation'].astype(str).str.strip() != 'Synonymous']

# Drop rows with missing log2FC
dms = dms.dropna(subset=['Log2 Fold Change after Positive Selection'])

# Aggregate per residue position
agg = (dms
       .groupby('AA Position')
       .agg(DMS_tol_mean     = ('Log2 Fold Change after Positive Selection', 'mean'),
            DMS_tol_median   = ('Log2 Fold Change after Positive Selection', 'median'),
            DMS_tol_max      = ('Log2 Fold Change after Positive Selection', 'max'),
            DMS_n_mutations  = ('Log2 Fold Change after Positive Selection', 'size'),
            DMS_n_tolerated  = ('Log2 Fold Change after Positive Selection',
                                lambda x: int((x > 0).sum())))
       .reset_index()
       .rename(columns={'AA Position': 'position'}))

# Assign domain from manuscript-consistent boundaries (not DMS file)
agg['Domain'] = agg['position'].apply(assign_domain)

# Fraction of sampled substitutions that are tolerated (>0)
agg['DMS_frac_tolerated'] = agg['DMS_n_tolerated'] / agg['DMS_n_mutations']
agg = agg[agg['DMS_n_mutations'] >= MIN_N_MUTATIONS].copy()
print(f"Aggregated DMS: {len(agg)} positions covered (of 1368).")
print(f"  median mutations/position = {agg['DMS_n_mutations'].median():.1f}")

# ==========================================
# 2. MERGE WITH CB x VESM QUADRANT TABLE
# ==========================================
quad = pd.read_csv(QUADRANT_TABLE)
if 'constraint' not in quad.columns and 'vesm_substitution_intolerance' in quad.columns:
    quad = quad.rename(columns={'vesm_substitution_intolerance': 'constraint'})
# Expected columns from Figure 9 script:
#   position, wt, mean_CB_force, constraint, Hub_Overlap_Count,
#   is_MD_switch, MD_Roles, quadrant
print(f"Quadrant table: {len(quad)} positions.")

merged = pd.merge(quad, agg, on='position', how='inner')
print(f"Merged (CB x VESM x DMS): {len(merged)} positions "
      f"({len(merged)/len(quad)*100:.1f}% of CB-VESM set).")

merged.to_csv("cb_vesm_dms_merged.csv", index=False)

# ==========================================
# 3. STATISTICAL TESTS OF H1 / H2
# ==========================================
q1 = merged[merged['quadrant'] == 'Q1_core_driver']
q2 = merged[merged['quadrant'] == 'Q2_functional_invariant']
q3 = merged[merged['quadrant'] == 'Q3_background']
q4 = merged[merged['quadrant'] == 'Q4_plastic_driver']

print("\n=== DMS tolerance by quadrant (mean log2FC, positive selection) ===")
for name, sub in [('Q1 core drivers', q1), ('Q2 invariants', q2),
                  ('Q3 background', q3), ('Q4 plastic drivers', q4)]:
    print(f"  {name:22s}  n={len(sub):4d}  "
          f"median={sub['DMS_tol_median'].median():+.3f}  "
          f"mean={sub['DMS_tol_mean'].mean():+.3f}")

# Kruskal-Wallis across all 4 quadrants
kw_stat, kw_p = kruskal(q1['DMS_tol_median'], q2['DMS_tol_median'],
                        q3['DMS_tol_median'], q4['DMS_tol_median'])
print(f"\nKruskal-Wallis across quadrants: H={kw_stat:.3f}, p={kw_p:.2e}")

# Pairwise Mann-Whitney (one-sided): Q4 > Q1, Q4 > Q2, Q1 < Q3 (sanity)
def mw_one_sided(a, b, alt='greater', label=''):
    s, p = mannwhitneyu(a, b, alternative=alt)
    print(f"  {label:40s}  U={s:.0f}, p={p:.2e}")
    return p

print("\nOne-sided Mann-Whitney:")
mw_one_sided(q4['DMS_tol_median'], q1['DMS_tol_median'], 'greater',
             'Q4 > Q1 (engineering vs core)')
mw_one_sided(q4['DMS_tol_median'], q2['DMS_tol_median'], 'greater',
             'Q4 > Q2 (engineering vs invariant)')
mw_one_sided(q1['DMS_tol_median'], q3['DMS_tol_median'], 'less',
             'Q1 < Q3 (core vs background, sanity)')

# Spearman correlations: DMS tolerance vs each axis
r_dms_cb,  p_dms_cb  = spearmanr(merged['DMS_tol_median'], merged['mean_CB_force'])
r_dms_ves, p_dms_ves = spearmanr(merged['DMS_tol_median'], merged['constraint'])
print(f"\nSpearman DMS vs CB force:     r={r_dms_cb:+.3f}, p={p_dms_cb:.2e}")
print(f"Spearman DMS vs VESM -LLR:    r={r_dms_ves:+.3f}, p={p_dms_ves:.2e}")
print("  (negative r with VESM constraint is the expected sanity check: "
      "conserved positions are less tolerant to mutation)")

# ==========================================
# 4. TRIPLE-CONVERGENT PRIORITY SET (engineering)
# ==========================================
priority = merged[
    (merged['quadrant'] == 'Q4_plastic_driver') &
    (merged['Hub_Overlap_Count'] >= 2) &
    (merged['DMS_n_mutations'] >= PRIORITY_MIN_N) &
    (merged['DMS_tol_median'] > 0)
].copy()
priority = priority.sort_values('mean_CB_force', ascending=False)

print(f"\n=== Triple-convergent engineering priority set: "
      f"{len(priority)} residues ===")
print(f"(Q4 super-hub with >={PRIORITY_MIN_N} sampled substitution(s) AND median log2FC > 0)")
cols = ['position', 'wt', 'Domain', 'mean_CB_force', 'constraint',
        'DMS_tol_median', 'DMS_n_mutations', 'DMS_n_tolerated',
        'Hub_Overlap_Count', 'MD_Roles']
print(priority[cols].to_string(index=False))
priority[cols].to_csv("q4_priority_triple_convergent.csv", index=False)

# Also report the less-strict set (drop DMS median filter, keep Q4 super-hubs)
relaxed = merged[
    (merged['quadrant'] == 'Q4_plastic_driver') &
    (merged['Hub_Overlap_Count'] >= 2)
].copy().sort_values('mean_CB_force', ascending=False)
print(f"\n=== Q4 super-hubs with any DMS coverage: {len(relaxed)} residues ===")
print(relaxed[cols].to_string(index=False))

# ==========================================
# 5. LOOKUP KNOWN FIDELITY-VARIANT RESIDUES
# ==========================================
fidelity_positions = {
    539:  'Sniper-Cas9 F539S',
    691:  'HiFi Cas9 R691A',
    810:  'eSpCas9(1.0) K810A',
    848:  'eSpCas9(1.1) K848A',
    855:  'Slaymaker K855A',
    1003: 'eSpCas9 K1003A',
    1007: 'Sniper2 E1007L/P',
    1010: 'SuperFi Y1010D',
    1013: 'SuperFi Y1013D',
    1016: 'SuperFi Y1016D',
    1018: 'SuperFi V1018D',
    1019: 'SuperFi R1019D',
    1027: 'SuperFi Q1027D',
    1031: 'SuperFi K1031D',
    1060: 'eSpCas9 R1060A',
    526:  'rCas9HF K526D',
}
print("\n=== DMS coverage of published fidelity-variant residues ===")
for pos, label in fidelity_positions.items():
    row = merged[merged['position'] == pos]
    if len(row) == 0:
        # may be absent either from CB-VESM set or from DMS
        in_quad = (quad['position'] == pos).any()
        in_dms  = (agg['position']  == pos).any()
        print(f"  {pos:5d} {label:26s}  "
              f"in_CBVESM={in_quad}  in_DMS={in_dms}")
    else:
        r = row.iloc[0]
        print(f"  {pos:5d} {label:26s}  quadrant={r['quadrant']:24s}  "
              f"DMS_med={r['DMS_tol_median']:+.3f}  n={int(r['DMS_n_mutations'])}")

# ==========================================
# 6. FIGURE
# ==========================================
fig = plt.figure(figsize=(18, 6.5))
gs  = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.4, 1.4], wspace=0.42)
ax_box = fig.add_subplot(gs[0])
ax_sc1 = fig.add_subplot(gs[1])
ax_sc2 = fig.add_subplot(gs[2])

# --- Panel A: DMS tolerance by quadrant (boxplot + strip) ---
order  = ['Q1_core_driver', 'Q2_functional_invariant',
          'Q3_background', 'Q4_plastic_driver']
labels = ['Q1\nCore\ndrivers', 'Q2\nFunct.\ninvariants',
          'Q3\nBack-\nground', 'Q4\nPlastic\ndrivers']
colors = [C_Q1, C_Q2, C_Q3, C_Q4]

data   = [merged[merged['quadrant'] == q]['DMS_tol_median'].values for q in order]
bp = ax_box.boxplot(data, positions=range(len(order)), widths=0.55,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color='black', lw=1.4),
                    whiskerprops=dict(color='#555555'),
                    capprops=dict(color='#555555'))
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.55)
    patch.set_edgecolor('black')

# strip plot of individual points
rng = np.random.default_rng(0)
for i, (d, c) in enumerate(zip(data, colors)):
    xj = rng.normal(i, 0.06, size=len(d))
    ax_box.scatter(xj, d, s=10, c=c, alpha=0.55, lw=0, zorder=3)

ax_box.axhline(0, color='#444444', ls='--', lw=0.9, alpha=0.7)
ax_box.set_xticks(range(len(order)))
ax_box.set_xticklabels(labels, fontsize=11, rotation=15, ha='right')
ax_box.set_ylabel('DMS tolerance (median log$_2$FC, positive selection)',
                  fontsize=15)
ax_box.tick_params(axis='y', labelsize=13)
ax_box.spines['top'].set_visible(False)
ax_box.spines['right'].set_visible(False)

# Annotate Kruskal-Wallis p
ax_box.text(0.02, 0.97,
            f"Kruskal–Wallis p = {kw_p:.1e}\n"
            f"Q4 > Q1: p = {mannwhitneyu(q4['DMS_tol_median'], q1['DMS_tol_median'], alternative='greater')[1]:.1e}",
            transform=ax_box.transAxes, fontsize=11, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.85, edgecolor='#AAAAAA'))

# --- Panel B: DMS tolerance vs CB force, colored by VESM constraint ---
sc = ax_sc1.scatter(merged['mean_CB_force'], merged['DMS_tol_median'],
                    c=merged['constraint'], cmap='viridis',
                    s=np.clip(merged['DMS_n_mutations'] * 8, 8, 60),
                    alpha=0.75, lw=0.3, edgecolors='#333333')
ax_sc1.axhline(0, color='#444444', ls='--', lw=0.9, alpha=0.6)
ax_sc1.axvline(merged['mean_CB_force'].median(),
               color='#444444', ls='--', lw=0.9, alpha=0.6)

# Highlight triple-convergent priority residues
if len(priority) > 0:
    for _, row in priority.iterrows():
        ax_sc1.scatter(row['mean_CB_force'], row['DMS_tol_median'],
                       s=110, facecolor='none', edgecolor=C_Q4, lw=1.8,
                       zorder=5)
        ax_sc1.annotate(f"{row['wt']}{int(row['position'])}",
                        (row['mean_CB_force'], row['DMS_tol_median']),
                        xytext=(5, 4), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='#8B4500')

cb = fig.colorbar(sc, ax=ax_sc1, pad=0.02, fraction=0.045)
cb.set_label('VESM substitution\nintolerance (−meanLLR)', fontsize=12)
cb.ax.tick_params(labelsize=10)

ax_sc1.set_xlabel('Mean CB driving force (stepwise)', fontsize=14)
ax_sc1.set_ylabel('DMS tolerance (median log$_2$FC)', fontsize=14)
ax_sc1.tick_params(labelsize=12)
ax_sc1.spines['top'].set_visible(False)
ax_sc1.spines['right'].set_visible(False)

# Quadrant label for the engineering sweet spot
ax_sc1.text(0.98, 0.97,
            f"Engineering sweet spot\n(high CB, high DMS)\nρ(DMS,CB) = {r_dms_cb:+.3f}",
            transform=ax_sc1.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF5D0',
                      alpha=0.85, edgecolor='#AAAAAA'))

# --- Panel C: DMS tolerance vs VESM constraint (sanity) ---
# Color by CB force so the reader sees that Q4 (warm colors, high CB) clusters top-left
sc2 = ax_sc2.scatter(merged['constraint'], merged['DMS_tol_median'],
                     c=merged['mean_CB_force'], cmap='coolwarm',
                     s=np.clip(merged['DMS_n_mutations'] * 8, 8, 60),
                     alpha=0.75, lw=0.3, edgecolors='#333333')
ax_sc2.axhline(0, color='#444444', ls='--', lw=0.9, alpha=0.6)

# Highlight priority residues
if len(priority) > 0:
    for _, row in priority.iterrows():
        ax_sc2.scatter(row['constraint'], row['DMS_tol_median'],
                       s=110, facecolor='none', edgecolor=C_Q4, lw=1.8,
                       zorder=5)
        ax_sc2.annotate(f"{row['wt']}{int(row['position'])}",
                        (row['constraint'], row['DMS_tol_median']),
                        xytext=(5, 4), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='#8B4500')

cb2 = fig.colorbar(sc2, ax=ax_sc2, pad=0.02, fraction=0.045)
cb2.set_label('CB driving force', fontsize=12)
cb2.ax.tick_params(labelsize=10)

ax_sc2.set_xlabel('VESM Substitution Intolerance (−meanLLR)', fontsize=14)
ax_sc2.set_ylabel('DMS tolerance (median log$_2$FC)', fontsize=14)
ax_sc2.tick_params(labelsize=12)
ax_sc2.spines['top'].set_visible(False)
ax_sc2.spines['right'].set_visible(False)

ax_sc2.text(0.98, 0.97,
            f"ρ(DMS, VESM −meanLLR) = {r_dms_ves:+.3f}\n(expected negative)",
            transform=ax_sc2.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.85, edgecolor='#AAAAAA'))

for _ax,_lab in zip((ax_box,ax_sc1,ax_sc2),'ABC'):
    _ax.set_title(_lab, loc='left', fontsize=20, fontweight='bold')
plt.savefig('CB_VESM_DMS.png', dpi=600, bbox_inches='tight')
plt.savefig('CB_VESM_DMS.pdf', bbox_inches='tight')
plt.close()
print("\nSaved: CB_VESM_DMS.png/.pdf")
print("Saved: cb_vesm_dms_merged.csv")
print("Saved: q4_priority_triple_convergent.csv")

