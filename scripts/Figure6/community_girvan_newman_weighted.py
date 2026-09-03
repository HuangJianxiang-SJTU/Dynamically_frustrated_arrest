"""Weighted Girvan–Newman community analysis for the reported results.

The residue contact network uses GCCM-derived impedance, ``-log(|GC|)``, as
the edge distance. Communities are obtained by edge-betweenness
decomposition, and the maximum-modularity partition along the resulting
dendrogram is retained for each R-loop state.
"""

import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from multiprocessing import Pool, cpu_count
import time

import networkx as nx
from networkx.algorithms.community.quality import modularity
import igraph as ig

from parse_pathways import STATE_LABELS, DPI

# ======================================================================
# USER SETTINGS
# ======================================================================
DATA_DIR       = '.'
PKL_NAME       = 'network_G_nierzwicki.pkl'
MAX_K          = 30
EARLY_STOP_GAP = 0.005
N_WORKERS      = min(7, cpu_count())
USE_WEIGHTS    = True

FIG_DIR = './figures/gn_weighted'
# ======================================================================

DOMAINS = {
    'RuvC': [(1,59),(718,764),(919,1100)],
    'BH':   [(60,94)],
    'REC1': [(95,176),(306,495)],
    'REC2': [(177,305)],
    'REC3': [(496,717)],
    'L1':   [(765,780)],
    'HNH':  [(781,905)],
    'L2':   [(906,918)],
    'PI':   [(1101,1368)],
}
DOMAIN_COLORS = {
    'RuvC':'#DD8452','BH':'#8C564B','REC1':'#4C72B0','REC2':'#64B5CD',
    'REC3':'#55A868','L1':'#E377C2','HNH':'#C44E52','L2':'#F7B6D2',
    'PI':'#937860','Other':'#8C8C8C',
}
KEY_GROUPS = {
    'α37 (692–698)':          [692, 694, 695, 698],
    'REC-HNH (789/794)':      [789, 794],
    'H840 (HNH cat.)':        [840],
    'HNH-RuvC (841/858)':     [841, 858],
    'TypeB loop (795–821)':   list(range(795, 822)),
    'TypeC HNH (841–846)':    list(range(841, 847)),
    'TypeC RuvC (1010–1021)': list(range(1010, 1022)),
    'PI-1200':                [1200],
}

# ── helpers ───────────────────────────────────────────────────────────

def domain_of(resid):
    for name, ranges in DOMAINS.items():
        for lo, hi in ranges:
            if lo <= resid <= hi:
                return name
    return 'Other'


def load_graph(label):
    for p in [
        os.path.join(DATA_DIR, label, PKL_NAME),
        os.path.join(DATA_DIR, f'{label}_{PKL_NAME}'),
    ]:
        if os.path.isfile(p):
            with open(p, 'rb') as f:
                obj = pickle.load(f)
            if isinstance(obj, tuple):
                if len(obj) == 2:
                    G, res_ids = obj
                elif len(obj) == 3:
                    G, _, res_ids = obj
                else:
                    G = next((x for x in obj
                               if isinstance(x, nx.Graph)), None)
                    res_ids = next((x for x in obj
                                    if isinstance(x, list)), None)
            elif isinstance(obj, nx.Graph):
                G = obj
                res_ids = list(range(1, G.number_of_nodes()+1))
            else:
                return None, None
            print(f'  [{label}] nodes={G.number_of_nodes()}, '
                  f'edges={G.number_of_edges()}, '
                  f'resids {res_ids[0]}–{res_ids[-1]}')
            return G, res_ids
    print(f'  [WARN] pkl not found for {label}')
    return None, None


def nx_to_igraph(G):
    nodes     = list(G.nodes())
    node_to_i = {n: i for i, n in enumerate(nodes)}
    edges     = [(node_to_i[u], node_to_i[v]) for u, v in G.edges()]
    weights   = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    gc_vals   = [G[u][v].get('gc', 1.0)     for u, v in G.edges()]

    ig_G = ig.Graph(n=len(nodes), edges=edges, directed=False)
    ig_G.es['weight'] = weights
    ig_G.es['gc']     = gc_vals
    ig_G.vs['node']   = nodes
    return ig_G, nodes


def node_comms_to_resid_comms(comms, res_ids):
    result = []
    for comm in comms:
        converted = set()
        for node in comm:
            if isinstance(node, int) and 0 <= node < len(res_ids):
                converted.add(res_ids[node])
            else:
                converted.add(node)
        result.append(frozenset(converted))
    return result


def membership_from_comms(comms_resid):
    return {r: i for i, comm in enumerate(comms_resid) for r in comm}


# ── igraph GN (unweighted) ────────────────────────────────────────────

def detect_igraph(G_nx):
    """Run Girvan–Newman with GCCM impedance as the edge distance."""
    ig_G, nodes = nx_to_igraph(G_nx)

    # Weighted edge betweenness treats the stored impedance as path distance.
    t0 = time.time()
    dendrogram = ig_G.community_edge_betweenness(
        directed=False,
        weights='weight' if USE_WEIGHTS else None)
    print(f'    igraph dendrogram (weighted): {time.time()-t0:.1f}s')

    # find max-Q partition
    best_Q          = -1.0
    best_membership = None
    n_nodes         = ig_G.vcount()

    for k in range(2, min(MAX_K + 1, n_nodes)):
        try:
            membership = dendrogram.as_clustering(k).membership
        except Exception:
            break

        communities = {}
        for vi, ci in enumerate(membership):
            communities.setdefault(ci, set()).add(nodes[vi])
        partition = [frozenset(v) for v in communities.values()]

        try:
            Q = modularity(G_nx, partition, weight='weight')
        except Exception:
            Q = -1.0

        if k % 3 == 0:
            print(f'    k={k:3d}  Q={Q:.4f}  best_Q={best_Q:.4f}')

        if Q > best_Q:
            best_Q          = Q
            best_membership = membership

        if k > 3 and Q < best_Q - EARLY_STOP_GAP:
            print(f'    Early stop at k={k} '
                  f'(dropped {best_Q-Q:.4f} below best_Q={best_Q:.4f})')
            break

    if best_membership is None:
        return [frozenset(G_nx.nodes())], 0.0

    communities = {}
    for vi, ci in enumerate(best_membership):
        communities.setdefault(ci, set()).add(nodes[vi])
    best_P = sorted([frozenset(v) for v in communities.values()],
                    key=len, reverse=True)

    print(f'  → FINAL k={len(best_P)}, Q={best_Q:.4f}')
    return best_P, best_Q


# ── worker ────────────────────────────────────────────────────────────

def process_one_state(label):
    t0 = time.time()
    print(f'[{label}] START  pid={os.getpid()}')

    G, res_ids = load_graph(label)
    if G is None:
        return label, [], 0.0, []

    comms_node, Q = detect_igraph(G)
    comms_resid   = node_comms_to_resid_comms(comms_node, res_ids)

    elapsed = time.time() - t0
    print(f'[{label}] DONE  k={len(comms_resid)}  '
          f'Q={Q:.4f}  time={elapsed:.1f}s')
    return label, comms_resid, Q, res_ids


# ── plots ─────────────────────────────────────────────────────────────

def plot_strip(mem, comms_resid, res_ids, label, Q, save_dir, dpi):
    n    = len(res_ids)
    k    = len(comms_resid)
    cmap = plt.cm.tab20
    colors = [cmap(mem.get(r, 0) / max(k, 1)) for r in res_ids]

    fig, ax = plt.subplots(figsize=(4.5, max(8, n * 0.065 + 2)))
    for i, (r, c) in enumerate(zip(res_ids, colors)):
        ax.barh(i, 1.0, color=c, edgecolor='none', height=1.0)
        if r % 100 == 0:
            ax.text(1.02, i, str(r), va='center', fontsize=5)

    for dname, ranges in DOMAINS.items():
        for lo, hi in ranges:
            idxs = [i for i, r in enumerate(res_ids) if lo <= r <= hi]
            if not idxs:
                continue
            ax.barh(idxs, [0.10]*len(idxs), left=1.08,
                    color=DOMAIN_COLORS.get(dname, '#999'),
                    edgecolor='none', height=1.0, alpha=0.85)
            mid = (min(idxs)+max(idxs))//2
            ax.text(1.20, mid, dname, va='center',
                    fontsize=4.5, fontweight='bold')

    patches = [mpatches.Patch(
                   color=cmap(i/max(k,1)),
                   label=f'C{i+1} (n={len(comms_resid[i])})')
               for i in range(min(k, 12))]
    ax.legend(handles=patches, fontsize=6,
              loc='lower right', framealpha=0.85)
    ax.set_xlim(0, 1.6); ax.set_ylim(-1, n)
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_title(f'{label}  k={k}  Q={Q:.4f}\n'
                 f'Weighted Girvan–Newman (GCCM impedance)',
                 fontsize=9, fontweight='bold')
    plt.tight_layout()
    out = f'{save_dir}/07_gn_community_{label}.png'
    plt.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'  [SAVED] {out}')


def plot_community_modularity(Q_vals, all_k, state_labels,
                               save_dir, dpi):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    x = np.arange(len(state_labels))

    ax1.plot(x, Q_vals, 'o-', color='steelblue', lw=2.5, ms=9)
    for xi, q in zip(x, Q_vals):
        ax1.annotate(f'{q:.3f}', (xi, q), xytext=(0, 8),
                     textcoords='offset points',
                     ha='center', fontsize=9)
    ax1.axvline(4, color='red', ls='--', lw=1.5,
                alpha=0.6, label='14nt')
    ax1.axhspan(0.3, 0.8, alpha=0.08, color='green',
                label='Literature range (0.3–0.8)')
    ax1.set_ylabel('Modularity Q', fontsize=11)
    ax1.set_ylim(0, max(Q_vals)*1.25+0.05)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, ls='--')
    ax1.set_title('Modularity Q — Weighted Girvan–Newman',
                  fontsize=11, fontweight='bold')

    ax2.bar(x, all_k, color='steelblue', edgecolor='black', lw=0.6)
    for xi, k in zip(x, all_k):
        ax2.text(xi, k+0.05, str(k),
                 ha='center', va='bottom', fontsize=10)
    ax2.axvline(4, color='red', ls='--', lw=1.5, alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(state_labels, rotation=45, ha='right')
    ax2.set_ylabel('Communities k', fontsize=11)
    ax2.set_ylim(0, max(all_k)+2)
    ax2.grid(True, axis='y', alpha=0.3, ls='--')
    ax2.set_title('Number of communities k',
                  fontsize=11, fontweight='bold')

    plt.tight_layout()
    out = f'{save_dir}/07_gn_community_modularity.png'
    plt.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] {out}')


def plot_community_sizes(all_comms, state_labels, save_dir, dpi):
    max_k  = max(len(c) for c in all_comms)
    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(state_labels))
    bottom = np.zeros(len(state_labels))
    cmap   = plt.cm.tab20
    for ci in range(max_k):
        sizes = np.array([len(c[ci]) if ci < len(c) else 0
                          for c in all_comms])
        ax.bar(x, sizes, 0.65, bottom=bottom,
               color=cmap(ci/max(max_k,1)),
               edgecolor='white', lw=0.4, label=f'C{ci+1}')
        bottom += sizes
    ax.set_xticks(x)
    ax.set_xticklabels(state_labels, rotation=45,
                       ha='right', fontsize=9)
    ax.set_ylabel('Residues', fontsize=11)
    ax.set_title('Community size distribution — Weighted Girvan–Newman',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=5, loc='upper right')
    ax.axvline(3.5, color='red', ls='--', lw=1.5, alpha=0.6)
    ax.grid(True, axis='y', alpha=0.3, ls='--')
    plt.tight_layout()
    out = f'{save_dir}/07_gn_community_sizes.png'
    plt.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] {out}')


def plot_hub_community_overlap(all_mems, state_labels,
                                save_dir, dpi):
    groups = list(KEY_GROUPS.items())
    ng, ns = len(groups), len(state_labels)
    data   = np.full((ng, ns), -1, dtype=int)

    for s, mem in enumerate(all_mems):
        for g, (_, resids) in enumerate(groups):
            vals = [mem[r] for r in resids if r in mem]
            if vals:
                data[g, s] = Counter(vals).most_common(1)[0][0]

    nmax = int(data.max())+2
    cmap = matplotlib.colormaps['tab20'].resampled(nmax)
    fig, ax = plt.subplots(figsize=(ns*1.6+2, ng*0.7+2))
    ax.imshow(data, cmap=cmap, vmin=-0.5,
              vmax=nmax-0.5, aspect='auto')

    ax.set_xticks(range(ns)); ax.set_yticks(range(ng))
    ax.set_xticklabels(state_labels, rotation=45,
                       ha='right', fontsize=9)
    ax.set_yticklabels([g[0] for g in groups], fontsize=9)

    for i in range(ng):
        for j in range(ns):
            v = data[i, j]
            if v >= 0:
                ax.text(j, i, f'C{v+1}',
                        ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='white')

    try:
        idx14 = state_labels.index('14nt')
        ax.axvline(idx14-0.5, color='red', lw=2,
                   ls='--', alpha=0.8, label='14nt transition')
        ax.legend(fontsize=9, loc='lower right')
    except ValueError:
        pass

    ax.set_title('Community membership of key residue groups\n'
                 '(weighted Girvan–Newman; GCCM impedance)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('R-loop State', fontsize=10)
    plt.tight_layout()
    out = f'{save_dir}/07_gn_community_hub_types.png'
    plt.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] {out}')


def plot_community_nmi(all_mems, state_labels, save_dir, dpi):
    try:
        from sklearn.metrics import normalized_mutual_info_score
    except ImportError:
        print('[SKIP] NMI: pip install scikit-learn')
        return
    all_r = sorted(set().union(*[set(m.keys()) for m in all_mems]))
    n     = len(state_labels)
    mat   = np.array([
        [normalized_mutual_info_score(
            [all_mems[i].get(r,-1) for r in all_r],
            [all_mems[j].get(r,-1) for r in all_r])
         for j in range(n)] for i in range(n)])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='NMI')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(state_labels, rotation=45,
                       ha='right', fontsize=9)
    ax.set_yticklabels(state_labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            tc = ('white' if mat[i,j] < 0.35
                  or mat[i,j] > 0.85 else 'black')
            ax.text(j, i, f'{mat[i,j]:.2f}',
                    ha='center', va='center',
                    fontsize=9, color=tc)
    ax.set_title('Community structure similarity (NMI)\n'
                 'Weighted Girvan–Newman (GCCM impedance)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = f'{save_dir}/07_gn_community_overlap.png'
    plt.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] {out}')


def export_bfactor_for_vmd(all_mems, state_labels, save_dir):
    vmd_dir = os.path.join(save_dir, 'vmd_community')
    os.makedirs(vmd_dir, exist_ok=True)
    for s, label in enumerate(state_labels):
        mem = all_mems[s]
        if not mem:
            continue
        out = os.path.join(vmd_dir, f'community_{label}.dat')
        with open(out, 'w') as fh:
            fh.write(f'# Community membership (weighted Girvan–Newman) '
                     f'— {label}\n')
            fh.write('# ResID  CommunityIndex (1-based)\n')
            for resid, comm_idx in sorted(mem.items()):
                fh.write(f'{resid:6d}  {comm_idx+1:3d}\n')
        print(f'  [SAVED] {out}')


def write_summary(all_comms, all_Q, all_mems,
                  state_labels, save_dir):
    out = f'{save_dir}/07_gn_community_summary.txt'
    with open(out, 'w') as fh:
        fh.write('SpCas9 R-loop Community Analysis\n')
        fh.write('Method: Weighted Girvan-Newman (igraph, impedance weights)\n')
        fh.write('Network: GC + contact freq >= 75% + '
                 'heavy atom < 4.5A\n')
        fh.write('='*65+'\n\n')
        for s, label in enumerate(state_labels):
            fh.write(f'State: {label}  |  '
                     f'{len(all_comms[s])} communities  '
                     f'|  Q={all_Q[s]:.4f}\n')
            fh.write('-'*50+'\n')
            for ci, comm in enumerate(all_comms[s]):
                clist   = sorted(comm)
                dom_cnt = Counter(domain_of(r) for r in clist)
                dom_str = ', '.join(
                    f'{d}:{n}' for d, n in
                    sorted(dom_cnt.items(), key=lambda x: -x[1]))
                fh.write(f'  C{ci+1:2d}  n={len(comm):4d}'
                         f'  [{dom_str}]\n')
                for gname, gresids in KEY_GROUPS.items():
                    present = [r for r in gresids if r in comm]
                    if present:
                        fh.write(f'       ↳ {gname}: {present}\n')
            fh.write('\n')
    print(f'[SAVED] {out}')


# ── main ──────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print('='*60)
    print('  Community Analysis — weighted Girvan–Newman')
    print('  Edge distance: GCCM-derived impedance')
    print(f'  MAX_K={MAX_K}  EARLY_STOP_GAP={EARLY_STOP_GAP}')
    print(f'  N_WORKERS={N_WORKERS}')
    print(f'  Output: {FIG_DIR}')
    print('='*60)

    os.makedirs(FIG_DIR, exist_ok=True)

    with Pool(processes=N_WORKERS) as pool:
        results = pool.map(process_one_state, STATE_LABELS)

    result_dict = {label: (comms, Q, res_ids)
                   for label, comms, Q, res_ids in results}

    all_comms = []
    all_Q     = []
    all_mems  = []

    for label in STATE_LABELS:
        comms_resid, Q, res_ids = result_dict[label]
        if not comms_resid:
            all_comms.append([])
            all_Q.append(0.0)
            all_mems.append({})
            continue

        mem_resid = membership_from_comms(comms_resid)
        all_comms.append(comms_resid)
        all_Q.append(Q)
        all_mems.append(mem_resid)

        plot_strip(mem_resid, comms_resid, res_ids,
                   label, Q, FIG_DIR, DPI)
        print(f'[{label}] k={len(comms_resid)}  Q={Q:.4f}')

    valid = [i for i, c in enumerate(all_comms) if c]
    if not valid:
        print('\n[ERROR] No communities found.')
        return

    all_k = [len(c) for c in all_comms]

    print('\n--- Summary figures ---')
    plot_community_modularity(all_Q, all_k, STATE_LABELS,
                               FIG_DIR, DPI)
    plot_community_sizes(all_comms, STATE_LABELS, FIG_DIR, DPI)
    plot_hub_community_overlap(all_mems, STATE_LABELS,
                                FIG_DIR, DPI)
    plot_community_nmi(all_mems, STATE_LABELS, FIG_DIR, DPI)
    export_bfactor_for_vmd(all_mems, STATE_LABELS, FIG_DIR)
    write_summary(all_comms, all_Q, all_mems,
                  STATE_LABELS, FIG_DIR)

    total = time.time() - t_start
    print('\n'+'='*60)
    print(f'Total wall time: {total:.1f}s  ({total/60:.1f} min)')
    print(f'\nAll output in: {FIG_DIR}/')
    for label in STATE_LABELS:
        print(f'  07_gn_community_{label}.png')
    print('  07_gn_community_modularity.png')
    print('  07_gn_community_sizes.png')
    print('  07_gn_community_hub_types.png  ← key figure')
    print('  07_gn_community_overlap.png')
    print('  07_gn_community_summary.txt')
    print('  vmd_community/community_*.dat')
    print('='*60)


if __name__ == '__main__':
    main()
