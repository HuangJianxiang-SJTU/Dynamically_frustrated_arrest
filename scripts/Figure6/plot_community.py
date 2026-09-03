"""
Community Membership Heatmap (Figure 6D) + Community Sizes (SI)
Shared color scheme for consistency.
Reads the deposited weighted Girvan–Newman assignments from
community_summary.txt; this plotting script does not recompute communities.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from collections import Counter
import re
import os

# --- FONT SETTINGS ---
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# ==========================================
# 1. CONFIGURATION
# ==========================================
SUMMARY_FILE = 'community_summary.txt'
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

STATE_LABELS = ['6nt', '8nt', '10nt', '12nt', '14nt', '16nt', '18nt']
DISPLAY_STATES = [s.replace('nt', '-nt') for s in STATE_LABELS]

'''
KEY_GROUPS = {
    r'$\alpha$37 helix (692-698)': [692, 694, 695, 698],
    'Y450 (TS sensor)':            [450],
    'L1 linker (765-780)':         list(range(765, 781)),
    'REC-HNH hinge (789/794)':     [789, 794],
    'H840 (HNH catalytic)':        [840],
    'HNH-RuvC (841/858)':          [841, 858],
    'L2 linker (906-918)':         list(range(906, 919)),
    'K1200 (NTS sensor)':          [1200],
}
'''

KEY_GROUPS = { 
    r'$\alpha$37 helix': [692, 694, 695, 698],
    'Y450':            [450],
    'L1 linker':         list(range(765, 781)),
    'REC-HNH hinge':     [789, 794],
    'H840':        [840],
    'HNH-RuvC':          [841, 858],
    'L2 linker':         list(range(906, 919)),
    'K1200':          [1200],
}

# ==========================================
# 2. SHARED COLOR PALETTE (max 20 communities)
# ==========================================
# Hand-picked palette: distinct, colorblind-friendly, publication-quality
COMM_COLORS = [
    '#4E79A7',  # C1  steel blue
    '#F28E2B',  # C2  orange
    '#E15759',  # C3  coral red
    '#76B7B2',  # C4  teal
    '#59A14F',  # C5  green
    '#EDC948',  # C6  gold
    '#B07AA1',  # C7  purple
    '#FF9DA7',  # C8  pink
    '#9C755F',  # C9  brown
    '#BAB0AC',  # C10 warm gray
    '#2CA02C',  # C11 bright green
    '#1F77B4',  # C12 blue
    '#D62728',  # C13 red
    '#9467BD',  # C14 violet
    '#8C564B',  # C15 dark brown
    '#E377C2',  # C16 magenta
    '#7F7F7F',  # C17 gray
    '#BCBD22',  # C18 olive
    '#17BECF',  # C19 cyan
    '#AEC7E8',  # C20 light blue
]

def get_comm_color(comm_id):
    if comm_id < 0:
        return '#EEEEEE'
    return COMM_COLORS[comm_id % len(COMM_COLORS)]

# ==========================================
# 3. PARSE SUMMARY FILE
# ==========================================
print(f"Parsing {SUMMARY_FILE}...")

all_mems = []
all_comm_sizes = []   # list of lists: [[size_C1, size_C2, ...], ...]
all_Q = []
all_k = []

with open(SUMMARY_FILE, 'r') as f:
    content = f.read()

state_blocks = re.split(r'State:\s+', content)[1:]

for block in state_blocks:
    lines = block.strip().split('\n')

    header = lines[0]
    match_q = re.search(r'Q=(\d+\.\d+)', header)
    match_k = re.search(r'(\d+)\s+communities', header)
    Q = float(match_q.group(1)) if match_q else 0.0
    k = int(match_k.group(1)) if match_k else 0

    all_Q.append(Q)
    all_k.append(k)

    mem = {}
    sizes = []
    current_comm_id = -1

    for line in lines[1:]:
        comm_match = re.match(r'\s+C\s*(\d+)\s+n=\s*(\d+)\s+\[(.+)\]', line)
        if comm_match:
            current_comm_id = int(comm_match.group(1)) - 1
            size = int(comm_match.group(2))
            sizes.append(size)
            continue

        key_match = re.match(r'\s+[\u21b3↳]\s+(.+?):\s+\[(.+)\]', line)
        if key_match and current_comm_id >= 0:
            resids_str = key_match.group(2)
            resids = [int(x.strip()) for x in resids_str.split(',')]
            for r in resids:
                mem[r] = current_comm_id

    all_mems.append(mem)
    all_comm_sizes.append(sizes)

print(f"Parsed {len(all_mems)} states")

# ==========================================
# 4. FIGURE 7D: COMMUNITY MEMBERSHIP HEATMAP
# ==========================================
print("Generating weighted Girvan–Newman community membership heatmap...")

groups = list(KEY_GROUPS.items())
ng = len(groups)
ns = len(STATE_LABELS)

data_comm = np.full((ng, ns), -1, dtype=int)

for s, mem in enumerate(all_mems):
    for g, (gname, resids) in enumerate(groups):
        vals = [mem[r] for r in resids if r in mem]
        if vals:
            data_comm[g, s] = Counter(vals).most_common(1)[0][0]

fig, ax = plt.subplots(figsize=(7, 6))

for i in range(ng):
    for j in range(ns):
        v = data_comm[i, j]
        color = get_comm_color(v)
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                              facecolor=color,
                              edgecolor='white', linewidth=2)
        ax.add_patch(rect)

        if v >= 0:
            rgb = mpl.colors.to_rgb(color)
            brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            text_color = 'black' if brightness > 0.55 else 'white'
            ax.text(j, i, f'C{v + 1}', ha='center', va='center',
                    fontsize=11, color=text_color, fontweight='bold')

ax.set_xlim(-0.5, ns - 0.5)
ax.set_ylim(ng - 0.5, -0.5)

ax.set_xticks(range(ns))
ax.set_xticklabels(DISPLAY_STATES, fontsize=14, rotation=30, ha='right')
ax.xaxis.tick_bottom()

ax.set_yticks(range(ng))
ax.set_yticklabels([g[0] for g in groups], fontsize=13)

ax.tick_params(axis='x', labelsize=18, length=4)
ax.tick_params(axis='y', labelsize=13, length=4)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.set_box_aspect(0.6)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07D_community_membership.png", dpi=600, bbox_inches='tight')
plt.savefig(f"{FIG_DIR}/07D_community_membership.pdf", bbox_inches='tight')
print(f"Saved: {FIG_DIR}/07D_community_membership.png and .pdf")
plt.close()

# ==========================================
# 5. SI: COMMUNITY SIZE DISTRIBUTION
# ==========================================
print("Generating community size distribution...")

max_k = max(len(c) for c in all_comm_sizes)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(ns)
bottom = np.zeros(ns)

for ci in range(max_k):
    sizes = np.array([c[ci] if ci < len(c) else 0 for c in all_comm_sizes])
    color = get_comm_color(ci)
    ax.bar(x, sizes, 0.65, bottom=bottom,
           color=color, edgecolor='white', lw=0.4,
           label=f'C{ci + 1}')
    bottom += sizes

ax.set_xticks(x)
ax.set_xticklabels(DISPLAY_STATES, fontsize=18)
#ax.set_xlabel('R-loop state', fontsize=20)
ax.set_ylabel('Residues', fontsize=20)
ax.tick_params(axis='both', labelsize=18)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', alpha=0.3, ls='--')

ax.legend(fontsize=8, ncol=5, loc='upper center',
          bbox_to_anchor=(0.5, -0.15), frameon=False)

plt.subplots_adjust(bottom=0.25)
plt.savefig(f"{FIG_DIR}/SX_community_sizes.png", dpi=600, bbox_inches='tight')
plt.savefig(f"{FIG_DIR}/SX_community_sizes.pdf", bbox_inches='tight')
print(f"Saved: {FIG_DIR}/SX_community_sizes.png and .pdf")
plt.close()

# ==========================================
# 6. SI: MODULARITY Q
# ==========================================
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(range(ns), all_Q, 'o-', color='#4E79A7', lw=2.2, ms=9, zorder=5)

for xi, q in zip(range(ns), all_Q):
    ax.annotate(f'{q:.3f}', (xi, q), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=11)

ax.set_xticks(range(ns))
ax.set_xticklabels(DISPLAY_STATES, fontsize=18)
ax.set_xlabel('R-loop state', fontsize=20)
ax.set_ylabel('Modularity Q', fontsize=20)
ax.tick_params(axis='both', labelsize=18)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/SX_modularity_Q.png", dpi=600, bbox_inches='tight')
plt.savefig(f"{FIG_DIR}/SX_modularity_Q.pdf", bbox_inches='tight')
print(f"Saved: {FIG_DIR}/SX_modularity_Q.png and .pdf")
plt.close()

# ==========================================
# 7. PRINT SUMMARY
# ==========================================
print("\n--- Community Summary ---")
for i, s in enumerate(DISPLAY_STATES):
    print(f"  {s}: k={all_k[i]}, Q={all_Q[i]:.4f}")

print("\n--- Key Group Community Trajectory ---")
for g, (gname, resids) in enumerate(groups):
    comms = [f'C{data_comm[g, s] + 1}' if data_comm[g, s] >= 0 else '?'
             for s in range(ns)]
    print(f"  {gname}: {' -> '.join(comms)}")
