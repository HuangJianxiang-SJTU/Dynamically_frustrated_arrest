"""
Figure 7: CB vs VESM Discordance Analysis
------------------------------------------
Panel A: CB driving force vs VESM substitution-intolerance scatter,
         colored by MD hub category, quadrant crosshairs at medians.
Panel B: Quadrant composition bar (total / MD-annotated / super-hub).

Style: Helvetica, panel titles A/B, dpi=600, legend below panel A.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from adjustText import adjust_text
from scipy.stats import pearsonr, spearmanr

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# ==========================================
# CONFIG
# ==========================================
STEPWISE_PAIRS = ["6_vs_8", "8_vs_10", "10_vs_12", "12_vs_14", "14_vs_16", "16_vs_18"]
VESM_FILE = "SpCas9_VESM3B_full_position_summary.csv"
MD_FILE = "full_superset.csv"
OUTPUT_DIR = "CB_VESM_discordance"
os.makedirs(OUTPUT_DIR, exist_ok=True)

C_BG     = '#BBBBBB'
C_SINGLE = '#4E79A7'
C_MULTI  = '#E15759'

# ==========================================
# 1. CB: mean stepwise driving force
# ==========================================
dfs_all = []   # all positions (for background scatter)
for pair in STEPWISE_PAIRS:
    path = f"CB_results_{pair}_proteinmpnn/position_summary.csv"
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping.")
        continue
    tmp = pd.read_csv(path)
    tmp['driving_force'] = -tmp['CB_bias_zscore']
    dfs_all.append(tmp[['position', 'wt', 'driving_force']])

# All positions aggregated (for scatter background)
cb_all = (pd.concat(dfs_all)
          .groupby(['position', 'wt'], as_index=False)['driving_force'].mean()
          .rename(columns={'driving_force': 'mean_CB_force'}))
print(f"CB all positions: {len(cb_all)}")

# Use the verified one-row-per-residue MD table as annotation source. Historical
# CB result files contain two names for the switch category, which must not be
# allowed to duplicate positions during aggregation.
md = pd.read_csv(MD_FILE)
category_labels = {
    'Switch': 'Structural_Switch',
    'GCCM': 'GCCM_Hub',
    'SB_hub': 'SaltBridge_Hub',
    'Hydro_hub': 'Hydrophobic_Hub',
    'BC': 'Centrality_Hub',
}
md['Hub_Overlap_Count'] = md[list(category_labels)].eq('Y').sum(axis=1)
md['MD_Roles'] = md.apply(
    lambda row: ' | '.join(
        label for column, label in category_labels.items() if row[column] == 'Y'
    ),
    axis=1,
)
md['is_MD_switch'] = md['Hub_Overlap_Count'] > 0
md_meta = md.rename(columns={'Residue': 'position'})[
    ['position', 'is_MD_switch', 'MD_Roles', 'Hub_Overlap_Count']
]
print(f"MD union: {len(md_meta)} positions; "
      f"multi-evidence set: {(md_meta['Hub_Overlap_Count'] >= 2).sum()}")

# ==========================================
# 2. VESM
# ==========================================
vesm = pd.read_csv(VESM_FILE)
vesm['vesm_substitution_intolerance'] = -vesm['mean_LLR']
vesm_sub = vesm[['position', 'vesm_substitution_intolerance']].copy()
print(f"VESM: {len(vesm_sub)} positions")

# ==========================================
# 3. MERGE & QUADRANT
# ==========================================
# Full scatter merge: all positions with both CB and VESM scores
merged_full = pd.merge(cb_all, vesm_sub, on='position', how='inner')
# Add one verified MD metadata record per position for coloring and filtering.
merged_full = pd.merge(merged_full, md_meta, on='position', how='left',
                       validate='one_to_one')
merged_full['Hub_Overlap_Count'] = merged_full['Hub_Overlap_Count'].fillna(0).astype(int)
merged_full['is_MD_switch'] = merged_full['Hub_Overlap_Count'] > 0
merged_full['MD_Roles'] = merged_full['MD_Roles'].fillna('')
if len(merged_full) != 1368 or not merged_full['position'].is_unique:
    raise ValueError("Expected one merged record for each of 1,368 SpCas9 positions")
print(f"Merged full (scatter): {len(merged_full)} positions")

# Quadrant analysis on the same full set
cb_med  = merged_full['mean_CB_force'].median()
ves_med = merged_full['vesm_substitution_intolerance'].median()
merged_quad = merged_full  # alias for downstream code

cb_med  = merged_quad['mean_CB_force'].median()
ves_med = merged_quad['vesm_substitution_intolerance'].median()

def assign_quadrant(row):
    hi_cb  = row['mean_CB_force'] > cb_med
    hi_ves = row['vesm_substitution_intolerance'] > ves_med
    if   hi_cb and hi_ves:      return 'Q1_core_driver'
    elif not hi_cb and hi_ves:  return 'Q2_functional_invariant'
    elif hi_cb and not hi_ves:  return 'Q4_plastic_driver'
    else:                       return 'Q3_background'

merged_quad['quadrant'] = merged_quad.apply(assign_quadrant, axis=1)
quadrant_output = os.path.join(OUTPUT_DIR, "cb_vesm_quadrant_table.csv")
merged_quad.to_csv(quadrant_output, index=False)
print(f"Saved: {quadrant_output}")

print("\nQuadrant counts:")
print(merged_quad['quadrant'].value_counts())

for q in ['Q1_core_driver', 'Q2_functional_invariant', 'Q4_plastic_driver', 'Q3_background']:
    sub    = merged_quad[(merged_quad['quadrant'] == q) & (merged_quad['Hub_Overlap_Count'] >= 1)]
    n_sup  = merged_quad[(merged_quad['quadrant'] == q) & (merged_quad['Hub_Overlap_Count'] >= 2)].shape[0]
    print(f"{q}: {merged_quad[merged_quad['quadrant']==q].shape[0]} total, "
          f"{len(sub)} MD-annotated, {n_sup} super-hubs")

# ==========================================
# 4. CORRELATION
# ==========================================
r_p, p_p = pearsonr(merged_quad['mean_CB_force'], merged_quad['vesm_substitution_intolerance'])
r_s, p_s = spearmanr(merged_quad['mean_CB_force'], merged_quad['vesm_substitution_intolerance'])
print(f"\nPearson  r = {r_p:.3f}, p = {p_p:.2e}")
print(f"Spearman r = {r_s:.3f}, p = {p_s:.2e}")

# ==========================================
# 5. FIGURE
# ==========================================
fig = plt.figure(figsize=(18, 7))
gs  = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.34)
ax_main = fig.add_subplot(gs[0])
ax_quad = fig.add_subplot(gs[1])

# --- Panel A: scatter ---
bg     = merged_quad[merged_quad['Hub_Overlap_Count'] == 0]
single = merged_quad[merged_quad['Hub_Overlap_Count'] == 1]
multi  = merged_quad[merged_quad['Hub_Overlap_Count'] >= 2]

ax_main.scatter(bg['mean_CB_force'],     bg['vesm_substitution_intolerance'],
                s=8,  c=C_BG,     alpha=0.35, lw=0, zorder=1,
                label=f'Non-hub residues (n={len(bg)})')
ax_main.scatter(single['mean_CB_force'], single['vesm_substitution_intolerance'],
                s=22, c=C_SINGLE, alpha=0.70, lw=0, zorder=2,
                label=f'Single-category MD hub (n={len(single)})')
ax_main.scatter(multi['mean_CB_force'],  multi['vesm_substitution_intolerance'],
                s=70, c=C_MULTI,  alpha=0.95, edgecolors='black', lw=0.7,
                zorder=4, label=f'Multi-category MD hub / super-hub (n={len(multi)})')

# Median crosshairs
ax_main.axvline(cb_med,  color='#444444', ls='--', lw=1.0, alpha=0.6)
ax_main.axhline(ves_med, color='#444444', ls='--', lw=1.0, alpha=0.6)

# Quadrant labels
x_min, x_max = merged_quad['mean_CB_force'].min(), merged_quad['mean_CB_force'].max()
y_min, y_max = (merged_quad['vesm_substitution_intolerance'].min(),
                merged_quad['vesm_substitution_intolerance'].max())
px = (x_max - x_min) * 0.02
py = (y_max - y_min) * 0.02

ax_main.text(x_max - px, y_max - py, 'Q1: Core drivers\n(high CB, high VESM)',
             ha='right', va='top', fontsize=13, color='#8B0000',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE0E0', alpha=0.7))
ax_main.text(x_min + px, y_max - py, 'Q2: Functional invariants\n(low CB, high VESM)',
             ha='left',  va='top', fontsize=13, color='#1A4D8F',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E0EEFF', alpha=0.7))
ax_main.text(x_max - px, y_min + py, 'Q4: Plastic drivers\n(high CB, low VESM)',
             ha='right', va='bottom', fontsize=13, color='#5A3E00',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF5D0', alpha=0.7))
ax_main.text(x_min + px, y_min + py, 'Q3: Background\n(low CB, low VESM)',
             ha='left',  va='bottom', fontsize=13, color='#555555',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0', alpha=0.7))

# Label super-hubs
texts = []
for _, row in multi.iterrows():
    t = ax_main.text(row['mean_CB_force'], row['vesm_substitution_intolerance'],
                     f"{row['wt']}{int(row['position'])}",
                     fontsize=9, fontweight='bold', color='#8B0000')
    texts.append(t)
adjust_text(texts, ax=ax_main,
            arrowprops=dict(arrowstyle='-', color='#888888', lw=0.6),
            expand_points=(1.8, 2.5))

# Spearman annotation — bottom-right, away from Q2 quadrant label
ax_main.text(0.97, 0.75,
             f"Spearman r = {r_s:.3f}\np = {p_s:.2e}",
             transform=ax_main.transAxes,
             fontsize=14, va='bottom', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

ax_main.set_xlabel('Mean CB Driving Force (stepwise)', fontsize=20)
ax_main.set_ylabel('VESM Substitution Intolerance (\u2212meanLLR)', fontsize=20)
ax_main.tick_params(labelsize=18)
ax_main.spines['top'].set_visible(False)
ax_main.spines['right'].set_visible(False)

# Legend below panel A
ax_main.legend(loc='upper center', bbox_to_anchor=(0.44, -0.12),
               fontsize=12, frameon=False, ncol=3)

# --- Panel B: quadrant composition bar ---
q_keys   = ['Q1_core_driver', 'Q2_functional_invariant', 'Q4_plastic_driver', 'Q3_background']
q_labels = ['Q1\nCore drivers', 'Q2\nFunctional\ninvariants',
            'Q4\nPlastic drivers', 'Q3\nBackground']
q_colors = [C_MULTI, C_SINGLE, '#F28E2B', C_BG]

for i, (key, label, color) in enumerate(zip(q_keys, q_labels, q_colors)):
    sub     = merged_quad[merged_quad['quadrant'] == key]
    n_total = len(sub)
    n_md    = (sub['Hub_Overlap_Count'] >= 1).sum()
    n_super = (sub['Hub_Overlap_Count'] >= 2).sum()
    # Three bars: total (transparent), MD (solid color), super-hub (hatched)
    ax_quad.bar(i, n_total, color=color, alpha=0.25, edgecolor='black', lw=0.8)
    ax_quad.bar(i, n_md,    color=color, alpha=0.75, edgecolor='black', lw=0.8)
    ax_quad.bar(i, n_super, color=color, alpha=1.00, edgecolor='black', lw=1.2,
                hatch='///')
    ax_quad.text(i, n_total + 4, str(n_total),
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

# Legend for bar shading — use actual visual encoding
from matplotlib.patches import Patch
bar_legend = [
    Patch(facecolor='#888888', alpha=0.25, edgecolor='black',
          label='All residues'),
    Patch(facecolor='#888888', alpha=0.75, edgecolor='black',
          label='MD-annotated (≥1 hub)'),
    Patch(facecolor='#888888', alpha=1.00, edgecolor='black', hatch='///',
          label='Super-hub (≥2 hubs)'),
]
ax_quad.legend(handles=bar_legend, loc='upper center', bbox_to_anchor=(0.3, -0.15),
               fontsize=11, frameon=False, ncol=1)

ax_quad.set_xticks(range(4))
ax_quad.set_xticklabels(q_labels, fontsize=12, rotation=30, ha='right')
ax_quad.set_ylabel('Number of residues', fontsize=18)
ax_quad.tick_params(axis='y', labelsize=16)
ax_quad.spines['top'].set_visible(False)
ax_quad.spines['right'].set_visible(False)

ax_main.set_title('A', loc='left', fontsize=24, fontweight='bold')
ax_quad.set_title('B', loc='left', fontsize=24, fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, 'CB_VESM_Discordance.png'),
            dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'CB_VESM_Discordance.pdf'),
            bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR}/CB_VESM_Discordance.png/.pdf")
plt.close()

# ==========================================
# 6. TOP RESIDUES PER QUADRANT
# ==========================================
print("\n=== Top 10 per quadrant (MD-annotated, sorted by CB force) ===")
for key, label in zip(q_keys, q_labels):
    sub = merged_quad[(merged_quad['quadrant'] == key) & (merged_quad['Hub_Overlap_Count'] >= 1)]
    sub = sub.sort_values('mean_CB_force', ascending=False).head(10)
    print(f"\n--- {label.replace(chr(10), ' ')} ---")
    print(sub[['position', 'wt', 'mean_CB_force', 'vesm_substitution_intolerance',
               'Hub_Overlap_Count', 'MD_Roles']].to_string(index=False))
