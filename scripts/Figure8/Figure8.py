#!/usr/bin/env python3
"""
================================================================================
Figure X  ::  SpCas9 flow-cytometry validation  (main-text figure)
================================================================================

Produces a 4-panel figure:
    Panel a (full width) : ranked stacked-bar of GFP-low + GFP-high at 72 h,
                           with WT reference line and predicted-class markers.
    Panel b              : GFP-high / total ratio at 72 h (the discrimination
                           index; R1210D and eSpCas9 form a low-ratio group).
    Panel c              : R1210 substitution series (WT, A, Q, D, eSpCas9).
    Panel d              : time course (24/48/72 h) for selected constructs.

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------
    pip install matplotlib numpy
    (statistics is in the Python standard library)

Run:
    python figure_x_validation.py
Outputs (written next to the script):
    fig_main.png   (300 dpi raster)
    fig_main.pdf   (vector, fonts embedded as Type-42 / editable in Illustrator)

--------------------------------------------------------------------------------
QUICK EDITING GUIDE  (search for the tag in brackets)
--------------------------------------------------------------------------------
    [COLORS]   - all hex colours in one block. GFP palette is here.
    [LAYOUT]   - figure size, panel grid, row heights, spacing.
    [DATA]     - raw triplicate values. Edit numbers here to update everything.
    [CLASS]    - which construct belongs to which predicted class (dot colour).
    [PANEL A]  - ranked stacked bar.  [PANEL B] [PANEL C] [PANEL D] follow.
    To change which constructs appear in panel c or d, edit the `series` /
    `tc` lists in those blocks.
    To turn the "eSpCas9-like low-ratio group" annotation off, comment out the
    two axB.text / axB.annotate lines flagged [ANNOTATION] in PANEL B.
================================================================================
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import statistics
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# Global style
# ----------------------------------------------------------------------------
mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
mpl.rcParams['pdf.fonttype'] = 42          # editable text in vector output
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['xtick.major.width'] = 0.8
mpl.rcParams['ytick.major.width'] = 0.8

# ----------------------------------------------------------------------------
# [COLORS]   GFP-themed palette  (RGB shown in comments)
# ----------------------------------------------------------------------------
C_LOW   = "#4379D1"   #  67,121,209  GFP-low bar segment (blue)
C_HIGH  = "#55AD7D"   #  85,173,125  GFP-high bar segment (GFP green)
C_Q4    = "#1D9E75"   #  teal-green  : plastic-driver (Q4) class marker
C_Q1    = "#D85A30"   #  coral       : core-driver (Q1) class marker
C_CTRL  = "#7A7A7A"   #  grey        : control / benchmark class marker
C_REF   = "#993C1D"   #  dark coral  : WT reference dashed line
C_LOWRATIO = "#993C1D"  # dark coral : low-ratio cluster bars in panel b
C_VALUE_LABEL = "#0F6E56"  # dark teal : numeric labels above bars in panel c

# ----------------------------------------------------------------------------
# [DATA]   Raw triplicate values  (GFP-low and GFP-high %, at 72 h)
#          Each entry: "l72" = three GFP-low replicates,
#                      "h72" = three GFP-high replicates.
#          Edit these numbers to regenerate the whole figure.
# ----------------------------------------------------------------------------
raw = {
 "WT":          {"l72":[17.5,16.9,17.7], "h72":[7.36,7.5,7.23]},
 "Y271A":       {"l72":[19.7,19.9,19.8], "h72":[9.29,9.82,9.45]},
 "Y271D":       {"l72":[21.1,20.3,20.8], "h72":[11.2,11.6,11.0]},
 "Y271F":       {"l72":[21.5,20.8,20.4], "h72":[10.2,10.2,9.64]},
 "M495A":       {"l72":[20,19.4,20.2],   "h72":[7.97,8.03,8.06]},
 "A538G":       {"l72":[19.5,20.2,19.5], "h72":[9.97,11.1,11.7]},
 "L625A":       {"l72":[16.0,17.0,17.0], "h72":[9.04,8.78,8.94]},
 "L625D":       {"l72":[16.1,16.3,16.1], "h72":[7.79,7.72,8.39]},
 "Q771A":       {"l72":[21.0,21.2,21.0], "h72":[12.2,12.4,12.9]},
 "K772A":       {"l72":[20.7,20.4,20.3], "h72":[12.8,11.9,12.1]},
 "R925A":       {"l72":[19.5,18.8,19.6], "h72":[10.9,9.75,9.72]},
 "K961A":       {"l72":[19.0,18.2,18.3], "h72":[12.3,11.8,11.3]},
 "V1018D":      {"l72":[15.6,15.1,15.6], "h72":[6.51,6.78,7.04]},
 "A1034G":      {"l72":[16.7,17.1,16.1], "h72":[7.09,6.73,7.02]},
 "F1037A":      {"l72":[18.3,17.9,18.0], "h72":[8.60,8.11,8.29]},
 "R1210A":      {"l72":[19.1,19.1,18.7], "h72":[9.97,9.67,9.80]},
 "R1210D":      {"l72":[12.3,13.1,13.2], "h72":[0.88,1.05,1.13]},
 "R1210Q":      {"l72":[18.8,18.3,18.2], "h72":[9.38,9.54,9.50]},
 "Y1242A":      {"l72":[14.3,15.1,13.9], "h72":[8.36,7.87,7.89]},
 "eSpCas9(1.1)":{"l72":[12.4,12.4,12.6], "h72":[1.02,1.06,1.02]},
}

# Time-course totals (low+high) at 24, 48, 72 h for panel d constructs.
# If you change panel-d membership, add/remove entries here too.
tc = {
 "WT":          {"t":[7.20, 21.57, 24.73]},
 "R1210D":      {"t":[7.18, 15.17, 13.89]},
 "eSpCas9(1.1)":{"t":[3.93, 10.74, 13.50]},
 "Q771A":       {"t":[10.44, 29.30, 33.57]},
 "L625A":       {"t":[10.23, 23.50, 25.59]},
}

def m(x):   return statistics.mean(x)
def sem(x): return statistics.stdev(x) / (len(x) ** 0.5)

# ----------------------------------------------------------------------------
# [CLASS]   predicted class of each construct -> marker colour above its bar
#           "q4" plastic driver | "q1" core driver | "ctrl" control/benchmark
# ----------------------------------------------------------------------------
cls = {"Y271A":"q4","Y271D":"q4","Y271F":"q4","A538G":"q4","L625A":"q4","L625D":"q4",
       "Q771A":"q4","K772A":"q4","R1210A":"q4","R1210Q":"q4","R1210D":"q4","F1037A":"q4",
       "A1034G":"q4","Y1242A":"q4","V1018D":"ctrl","K961A":"ctrl","eSpCas9(1.1)":"ctrl",
       "WT":"ctrl","M495A":"q1","R925A":"q1"}
clsdot = {"q4": C_Q4, "q1": C_Q1, "ctrl": C_CTRL}

# ----------------------------------------------------------------------------
# [LAYOUT]   figure size and panel grid
#   Row 0 = panel a (spans all 3 columns).
#   Row 1 = panels b, c, d side by side.
#   height_ratios -> increase second value to make b/c/d taller.
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 8.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2], hspace=0.30, wspace=0.30,
                      left=0.065, right=0.985, top=0.94, bottom=0.08)

# ============================================================================
# [PANEL A]   ranked stacked bar, GFP-low + GFP-high at 72 h
# ============================================================================
axA = fig.add_subplot(gs[0, :])
items = [(n, m(d["l72"]), m(d["h72"]), m(d["l72"]) + m(d["h72"]),
          sem([d["l72"][i] + d["h72"][i] for i in range(3)])) for n, d in raw.items()]
items.sort(key=lambda x: -x[3])                       # rank by total, descending
names = [x[0] for x in items]
los   = [x[1] for x in items]
his   = [x[2] for x in items]
tots  = [x[3] for x in items]
sems  = [x[4] for x in items]
xpos  = np.arange(len(names))

axA.bar(xpos, los, color=C_LOW,  edgecolor='white', linewidth=0.4, width=0.72)
axA.bar(xpos, his, bottom=los, color=C_HIGH, edgecolor='white', linewidth=0.4, width=0.72)
axA.errorbar(xpos, tots, yerr=sems, fmt='none', ecolor='#333',
             elinewidth=1, capsize=6, capthick=1.5)

# class marker dot above each bar
for i, n in enumerate(names):
    axA.plot(i, tots[i] + sems[i] + 1.0, marker='o', markersize=8,
             color=clsdot[cls[n]], markeredgecolor='white',
             markeredgewidth=0.4, clip_on=False)

# WT reference line
wt_tot = m(raw["WT"]["l72"]) + m(raw["WT"]["h72"])
axA.axhline(wt_tot, color=C_REF, linewidth=1.0, linestyle='--', dashes=(4, 2), zorder=0)
axA.text(len(names) - 0.3, wt_tot + 0.4, f'WT {wt_tot:.1f}%',
         color=C_REF, fontsize=12, ha='right', va='bottom')

axA.set_xticks(xpos)
axA.set_xticklabels(names, rotation=45, ha='right', fontsize=13)
axA.set_ylabel('GFP-positive cells (%)', fontsize=18)
axA.set_ylim(0, 38)
axA.set_xlim(-0.7, len(names) - 0.3)
axA.spines['top'].set_visible(False)
axA.spines['right'].set_visible(False)
# axA.text(-0.045, 1.06, 'a', transform=axA.transAxes, fontsize=15,
         # fontweight='bold', va='top')

# two-part legend: bar segments + class markers
leg1 = [Patch(facecolor=C_LOW, label='GFP-low'),
        Patch(facecolor=C_HIGH, label='GFP-high')]
leg2 = [Line2D([0], [0], marker='o', color='w', markerfacecolor=C_Q4,   markersize=10, label='Plastic driver (Q4)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_Q1,   markersize=10, label='Core driver (Q1)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_CTRL, markersize=10, label='Control / benchmark')]
l1 = axA.legend(handles=leg1, loc='upper right', fontsize=14, frameon=False,
                bbox_to_anchor=(0.7, 1.03))
axA.add_artist(l1)
axA.legend(handles=leg2, loc='upper right', fontsize=12, frameon=False,
           bbox_to_anchor=(0.9, 1.03))

# ============================================================================
# [PANEL B]   GFP-high / total ratio at 72 h  (ascending)
# ============================================================================
axB = fig.add_subplot(gs[1, 0])
ritems = [(n, m(d["h72"]) / (m(d["l72"]) + m(d["h72"]))) for n, d in raw.items()]
ritems.sort(key=lambda x: x[1])
rnames = [x[0] for x in ritems]
ratios = [x[1] for x in ritems]
rcolors = []
for n, r in ritems:
    if r < 0.1:            rcolors.append(C_LOWRATIO)   # eSpCas9-like cluster
    elif n == "WT":        rcolors.append(C_CTRL)
    else:                  rcolors.append(C_HIGH)
ypos = np.arange(len(rnames))
axB.barh(ypos, ratios, color=rcolors, edgecolor='white', linewidth=0.4, height=0.74)
axB.set_yticks(ypos)
axB.set_yticklabels(rnames, fontsize=12)
axB.set_xlabel('GFP-high / total (72 h)', fontsize=18)
axB.set_xlim(0, 0.43)

axB.invert_yaxis()
axB.spines['top'].set_visible(False)
axB.spines['right'].set_visible(False)
# [ANNOTATION] comment these two lines out to remove the cluster label/arrow
axB.text(0.115, 0.5, 'eSpCas9-like\nlow-ratio group', fontsize=12,
         color=C_LOWRATIO, ha='left', va='center', style='italic')
axB.annotate('', xy=(0.090, 0.5), xytext=(0.110, 0.5),
             arrowprops=dict(arrowstyle='->', color=C_LOWRATIO, lw=0.8))
# axB.text(-0.30, 1.05, 'b', transform=axB.transAxes, fontsize=15,
         # fontweight='bold', va='top')

# ============================================================================
# [PANEL C]   R1210 substitution series
# ============================================================================
axC = fig.add_subplot(gs[1, 1])
series  = ["WT", "R1210A", "R1210Q", "R1210D", "eSpCas9(1.1)"]
slabels = ["WT", "R1210A", "R1210Q", "R1210D", "eSpCas9"]
slo = [m(raw[s]["l72"]) for s in series]
shi = [m(raw[s]["h72"]) for s in series]
sx  = np.arange(len(series))
axC.bar(sx, slo, color=C_LOW,  edgecolor='white', linewidth=0.4, width=0.62)
axC.bar(sx, shi, bottom=slo, color=C_HIGH, edgecolor='white', linewidth=0.4, width=0.62)
for i, s in enumerate(series):   # GFP-high value above each bar
    axC.text(i, slo[i] + shi[i] + 0.6, f'{shi[i]:.1f}', ha='center',
             fontsize=13, color=C_VALUE_LABEL)
axC.set_xticks(sx)
axC.set_xticklabels(slabels, rotation=35, ha='right', fontsize=13)
axC.set_ylabel('GFP-positive cells (%)', fontsize=18)
axC.set_ylim(0, 32)
axC.spines['top'].set_visible(False)
axC.spines['right'].set_visible(False)
# axC.text(-0.26, 1.05, 'c', transform=axC.transAxes, fontsize=15,
         # fontweight='bold', va='top')

# ============================================================================
# [PANEL D]   time course (24 / 48 / 72 h)
# ============================================================================
axD = fig.add_subplot(gs[1, 2])
tpts = [24, 48, 72]
tcolors = {"WT": C_CTRL, "R1210D": C_LOW, "eSpCas9(1.1)": C_LOWRATIO,
           "Q771A": C_Q4, "L625A": C_HIGH}
tdash   = {"WT": False, "R1210D": False, "eSpCas9(1.1)": True,
           "Q771A": True, "L625A": False}
for n in ["WT", "R1210D", "eSpCas9(1.1)", "Q771A", "L625A"]:
    ls = '--' if tdash[n] else '-'
    axD.plot(tpts, tc[n]["t"], marker='o', markersize=5, linewidth=1.7,
             color=tcolors[n], linestyle=ls, label=n,
             markeredgecolor='white', markeredgewidth=0.5)
axD.set_xticks(tpts)
axD.set_xticklabels(['24 h', '48 h', '72 h'], fontsize=16)
axD.set_xlim(20, 76)
axD.set_ylim(0, 38)
axD.set_ylabel('Total GFP-positive (%)', fontsize=18)
axD.spines['top'].set_visible(False)
axD.spines['right'].set_visible(False)
axD.legend(loc='upper left', fontsize=11.5, frameon=False, ncol=1)
# axD.text(-0.26, 1.05, 'd', transform=axD.transAxes, fontsize=15,
         # fontweight='bold', va='top')

# move panel b
posB = axB.get_position()
# Move it right by increasing x0 and x1
new_x0 = posB.x0 + 0.03  # shift right by 3% of figure width
new_x1 = posB.x1 + 0.03
axB.set_position([new_x0, posB.y0, posB.width, posB.height])


# Move panel C too
posC = axC.get_position()
new_x0 = posC.x0 + 0.015  # shift right by 1.5% of figure width
new_x1 = posC.x1 + 0.015
axC.set_position([new_x0, posC.y0, posC.width, posC.height])

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
fig.savefig('fig_main.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('fig_main.pdf', bbox_inches='tight', facecolor='white')
print("Wrote fig_main.png and fig_main.pdf")
