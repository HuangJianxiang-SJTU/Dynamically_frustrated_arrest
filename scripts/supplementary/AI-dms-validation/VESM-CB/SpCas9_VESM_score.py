"""
SpCas9 (Q99ZW2) VESM_3B Saturation Mutagenesis Scorer
======================================================
Hardware target : 4× RTX 5090 (128 GB VRAM total)
Model           : VESM_3B (ESM2-3B backbone + Ntranos lab distilled weights)
Output          : per-variant LLR table + per-position mean-LLR summary

Usage:
    # Score only your MD switch residues (~seconds):
    python SpCas9_VESM3B_score.py --mode subset

    # Full saturation scan across all 1368 positions (~25 sec on 4x5090):
    python SpCas9_VESM3B_score.py --mode all
"""

import os

# ── Mirror MUST be set BEFORE importing huggingface_hub ──────────────────────
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, EsmForMaskedLM
from huggingface_hub import hf_hub_download

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME = "VESM_3B"
ESM_BASE   = "facebook/esm2_t36_3B_UR50D"
LOCAL_DIR  = "./vesm_weights"
MAX_LEN    = 1022      # ESM2 hard limit (excluding BOS/EOS tokens)
BATCH_SIZE = 64       # 4x RTX 5090 DataParallel; reduce to 64 if single GPU
N_GPUS     = torch.cuda.device_count()
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── Your MD-identified switch / sensor / checkpoint residues (1-indexed) ─────
MD_GROUPS = {
    "Allosteric_Switch": [5, 23, 24, 25, 26, 35, 251, 254, 255, 256, 263, 271, 272, 284, 303, 304, 305, 494, 495, 531, 538, 539, 624, 625, 712, 713, 714, 769, 771, 772, 773, 774, 775, 922, 957, 958, 960, 961, 981, 1002, 1003, 1004, 1008, 1017, 1018, 1019, 1020, 1021, 1022, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1049, 1050, 1051, 1057, 1058, 1059, 1087, 1110, 1120, 1150, 1206, 1210, 1215, 1216, 1219, 1220, 1240],
    "GCCM_Hub": [167, 168, 169, 170, 171, 172, 173, 174, 175, 180, 181, 182, 183, 184, 185, 186, 187, 199, 201, 202, 203, 204, 205, 206, 209, 212, 230, 231, 242, 244, 245, 246, 247, 267, 268, 294, 295, 296, 297, 298, 299, 300, 301, 302, 306, 408, 409, 410, 411, 469, 628, 1047, 1048, 1080],
    "SaltBridge_Hub": [63, 69, 71, 115, 215, 220, 234, 253, 273, 274, 304, 397, 457, 586, 627, 629, 653, 661, 705, 762, 765, 772, 778, 782, 783, 809, 848, 849, 850, 895, 905, 910, 919, 923, 925, 969, 999, 1019, 1026, 1099, 1114, 1118, 1123, 1200, 1210, 1341],
    "Hydrophobic_Hub": [9, 11, 136, 237, 241, 252, 256, 258, 262, 266, 282, 286, 290, 301, 305, 308, 411, 414, 450, 451, 464, 491, 492, 495, 529, 530, 537, 538, 539, 620, 625, 626, 631, 632, 659, 693, 694, 704, 727, 733, 737, 763, 784, 830, 842, 846, 900, 911, 914, 917, 921, 922, 931, 955, 958, 962, 993, 996, 997, 1001, 1004, 1008, 1009, 1013, 1015, 1018, 1021, 1022, 1032, 1034, 1036, 1037, 1038, 1039, 1042, 1052, 1074, 1169, 1204, 1206, 1228, 1242, 1290, 1294, 1309, 1310, 1312, 1315, 1324, 1326],
    "Centrality_Hub": [13, 52, 53, 55, 57, 58, 59, 60, 63, 64, 65, 66, 67, 68, 70, 71, 74, 75, 164, 220, 222, 269, 271, 274, 294, 302, 414, 415, 447, 448, 450, 491, 492, 496, 498, 499, 500, 507, 509, 596, 623, 624, 653, 654, 705, 708, 712, 713, 714, 715, 716, 717, 719, 720, 723, 731, 733, 735, 766, 768, 771, 776, 780, 797, 807, 817, 818, 841, 842, 850, 856, 916, 925, 926, 934, 940, 947, 949, 951, 960, 961, 980, 1106, 1108, 1131, 1138, 1139, 1242, 1244],
}

# Flatten the dictionary into a unique list of residues for the ESM scorer
TARGET_POSITIONS = list(set([res for group in MD_GROUPS.values() for res in group]))

# 20 standard amino acids
AAS = list("ACDEFGHIKLMNPQRSTVWY")

# ── SpCas9 Q99ZW2 full sequence (1368 AA) ────────────────────────────────────
CAS9_SEQ = (
    "MDKKYSIGLDIGTNSVGWAVITDEYKVPSKKFKVLGNTDRHSIKKNLIGALLFDSGETAE"
    "ATRLKRTARRRYTRRKNRICYLQEIFSNEMAKVDDSFFHRLEESFLVEEDKKHERHPIFGN"
    "IVDEVAYHEKYPTIYHLRKKLVDSTDKADLRLIYLALAHMIKFRGHFLIEGDLNPDNSDVD"
    "KLFIQLVQTYNQLFEENPINASGVDAKAILSARLSKSRRLENLIAQLPGEKKNGLFGNLIA"
    "LSLGLTPNFKSNFDLAEDAKLQLSKDTYDDDLDNLLAQIGDQYADLFLAAKNLSDAILLSD"
    "ILRVNTEITKAPLSASMIKRYDEHHQDLTLLKALVRQQLPEKYKEIFFDQSKNGYAGYIDG"
    "GASQEEFYKFIKPILEKMDGTEELLVKLNREDLLRKQRTFDNGSIPHQIHLGELHAILRRQ"
    "EDFYPFLKDNREKIEKILTFRIPYYVGPLARGNSRFAWMTRKSEETITPWNFEEVVDKGASA"
    "QSFIERMTNFDKNLPNEKVLPKHSLLYEYFTVYNELTKVKYVTEGMRKPAFLSGEQKKAIV"
    "DLLFKTNRKVTVKQLKEDYFKKIECFDSVEISGVEDRFNASLGTYHDLLKIIKDKDFLDNEE"
    "NEDILEDIVLTLTLFEDREMIEERLKTYAHLFDDKVMKQLKRRRYTGWGRLSRKLINGIRDK"
    "QSGKTILDFLKSDGFANRNFMQLIHDDSLTFKEDIQKAQVSGQGDSLHEHIANLAGSPAIK"
    "KGILQTVKVVDELVKVMGRHKPENIVIEMARENQTTQKGQKNSRERMKRIEEGIKELGSQIL"
    "KEHPVENTQLQNEKLYLYYLQNGRDMYVDQELDINRLSDYDVDHIVPQSFLKDDSIDNKVLT"
    "RSDKNRGKSDNVPSEEVVKKMKNYWRQLLNAKLITQRKFDNLTKAERGGLSELDKAGFIKRQ"
    "LVETRQITKHVAQILDSRMNTKYDENDKLIREVKVITLKSKLVSDFRKDFQFYKVREINNYHH"
    "AHDAYLNAVVGTALIKKYPKLESEFVYGDYKVYDVRKMIAKSEQEIGKATAKYFFYSNIMNFF"
    "KTEITLANGEIRKRPLIETNGETGEIVWDKGRDFATVRKVLSMPQVNIVKKTEVQTGGFSKES"
    "ILPKRNSDKLIARKKDWDPKKYGGFDSPTVAYSVLVVAKVEKGKSKKLKSVKELLGITIMERS"
    "SFEKNPIDFLEAKGYKEVKKDLIIKLPKYSLFELENGRKRMLASAGELQKGNELALPSKYVNF"
    "LYLASHYEKLKGSPEDNEQKQLFVEQHKHYLDEIIEQISEFSKRVILADANLDKVLSAYNKHR"
    "DKPIREQAENIIHLFTLTNLGAPAAFKYFDTTIDRKRYTSTKEVLDATLIHQSITGLYETRIDL"
    "SQLGGD"
)
assert len(CAS9_SEQ) == 1368, (
    f"Sequence length mismatch: got {len(CAS9_SEQ)}, expected 1368. "
    "Please verify the Q99ZW2 FASTA sequence."
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_window(seq, pos, max_len=MAX_LEN):
    """
    Return (window_seq, local_pos) for a centered sliding window around pos.
    Handles boundary cases at N- and C-termini.
    pos : 0-indexed
    """
    half  = max_len // 2
    start = max(0, pos - half)
    end   = min(len(seq), start + max_len)
    if end - start < max_len:          # C-terminal boundary: shift window left
        start = max(0, end - max_len)
    return seq[start:end], pos - start


def load_model():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    print(f"📥 Downloading {MODEL_NAME} weights...")
    weight_path = hf_hub_download(
        repo_id="ntranoslab/vesm",
        filename=f"{MODEL_NAME}.pth",
        local_dir=LOCAL_DIR
    )

    print(f"🧠 Loading ESM2-3B backbone ({ESM_BASE})...")
    tokenizer = AutoTokenizer.from_pretrained(ESM_BASE)
    model     = EsmForMaskedLM.from_pretrained(ESM_BASE)

    # ── Verify VESM weight keys before patching ───────────────────────────
    print("⚙️  Inspecting VESM weight file...")
    vesm_state = torch.load(weight_path, map_location="cpu")
    vesm_keys  = set(vesm_state.keys())
    model_keys = set(model.state_dict().keys())
    matched    = vesm_keys & model_keys
    unmatched  = vesm_keys - model_keys

    print(f"   VESM .pth total keys : {len(vesm_keys)}")
    print(f"   Matched to ESM2-3B   : {len(matched)}")
    print(f"   Unmatched (ignored)  : {len(unmatched)}")
    if unmatched:
        print(f"   Unmatched sample     : {list(unmatched)[:5]}")
    if len(matched) == 0:
        raise RuntimeError(
            "VESM .pth has NO matching keys to ESM2-3B. "
            "Check that MODEL_NAME='VESM_3B' is paired with "
            "ESM_BASE='facebook/esm2_t36_3B_UR50D'."
        )

    model.load_state_dict(vesm_state, strict=False)
    print(f"   ✓ Patched {len(matched)} parameter tensors with VESM weights")

    # ── Multi-GPU setup ───────────────────────────────────────────────────
    if N_GPUS > 1:
        print(f"🔥 Distributing across {N_GPUS} GPUs via DataParallel...")
        model = torch.nn.DataParallel(model)
    else:
        print(f"🔥 Running on single GPU: {torch.cuda.get_device_name(0)}")

    model = model.to(DEVICE).eval()
    return tokenizer, model


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_positions(tokenizer, model, seq, positions):
    """
    Batched masked-language-model scoring.

    For each position in `positions` (0-indexed):
      1. Extract a centered window of MAX_LEN residues
      2. Mask the target position
      3. Run a single forward pass per batch of BATCH_SIZE positions
      4. Compute LLR = log P(mut|context) - log P(wt|context) for all 19 muts

    Returns a DataFrame with columns:
        uniprot_id, protein_variant, wt, mut, position, LLR
    """
    results = []
    total   = len(positions)

    for batch_start in range(0, total, BATCH_SIZE):
        batch_pos   = positions[batch_start : batch_start + BATCH_SIZE]
        seqs_masked = []
        meta        = []   # (pos_0idx, local_pos, wt_aa)

        for pos in batch_pos:
            window, local = get_window(seq, pos)
            wt_aa  = seq[pos]
            masked = window[:local] + tokenizer.mask_token + window[local + 1:]
            seqs_masked.append(masked)
            meta.append((pos, local, wt_aa))

        enc = tokenizer(
            seqs_masked,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN + 2    # +2 for BOS / EOS tokens
        ).to(DEVICE)

        with torch.no_grad():
            # DataParallel returns on cuda:0 automatically
            logits = model(**enc).logits    # (batch, seq_len, vocab_size)

        # Numerically stable LLR via log-softmax over vocabulary
        log_probs = F.log_softmax(logits, dim=-1)

        for i, (pos, local, wt_aa) in enumerate(meta):
            tok_pos   = local + 1           # +1 accounts for BOS token
            wt_tok_id = tokenizer.convert_tokens_to_ids(wt_aa)
            wt_lp     = log_probs[i, tok_pos, wt_tok_id].item()

            for mut_aa in AAS:
                if mut_aa == wt_aa:
                    continue
                mut_tok_id = tokenizer.convert_tokens_to_ids(mut_aa)
                mut_lp     = log_probs[i, tok_pos, mut_tok_id].item()
                llr        = mut_lp - wt_lp

                results.append({
                    "uniprot_id":      "Q99ZW2",
                    "protein_variant": f"{wt_aa}{pos + 1}{mut_aa}",
                    "wt":              wt_aa,
                    "mut":             mut_aa,
                    "position":        pos + 1,    # back to 1-indexed for output
                    "LLR":             llr,
                })

        done = min(batch_start + BATCH_SIZE, total)
        print(f"  ✓ {done}/{total} positions scored...", end="\r")

    print()
    return pd.DataFrame(results)


# ── Per-position summary ──────────────────────────────────────────────────────
# ── Per-position summary ──────────────────────────────────────────────────────

def summarise(df, tag):
    """
    Aggregate per-variant LLR to per-position mean LLR.
    Most negative mean LLR = most evolutionarily constrained position.
    Also flags which MD roles a residue plays and counts its Hub Overlaps.
    """
    summary = (
        df.groupby(["position", "wt"])["LLR"]
        .agg(mean_LLR="mean", min_LLR="min", max_LLR="max", n_muts="count")
        .reset_index()
    )

    # 映射你在 MD 中发现的角色 (需要脚本开头已定义 MD_GROUPS 字典)
    def get_md_roles(pos):
        roles = []
        for group_name, res_list in MD_GROUPS.items():
            if pos in res_list:
                roles.append(group_name)
        return " | ".join(roles) if roles else "None"

    # 添加新列：角色标签、是否为关键残基、以及重叠次数 (Super-Hub 分数)
    summary["MD_Roles"] = summary["position"].apply(get_md_roles)
    summary["is_MD_switch"] = summary["MD_Roles"] != "None"
    
    summary["Hub_Overlap_Count"] = summary["position"].apply(
        lambda pos: sum(1 for res_list in MD_GROUPS.values() if pos in res_list)
    )

    # 按进化受限程度 (越负越保守) 排序
    summary = summary.sort_values("mean_LLR")

    out = f"SpCas9_VESM3B_{tag}_position_summary.csv"
    summary.to_csv(out, index=False)
    print(f"✓ Position summary → {out}")

    print("\n── Top 30 most evolutionarily constrained positions ──")
    # 为了终端显示整洁，这里选取几个核心列展示
    cols_to_print = ["position", "wt", "mean_LLR", "Hub_Overlap_Count", "MD_Roles"]
    print(summary[cols_to_print].head(30).to_string(index=False))

    if tag == "all":
        md_hits = summary[summary["is_MD_switch"]].copy()
        # 将你的 MD 残基优先按照 Super-Hub 重叠次数排序，其次按进化保守度排序
        md_hits = md_hits.sort_values(by=["Hub_Overlap_Count", "mean_LLR"], ascending=[False, True])
        
        print(f"\n── MD residues ranked by category overlap and VESM substitution intolerance ──")
        print(f"   ({len(md_hits)} residues found in full scan)")
        print(md_hits[cols_to_print].to_string(index=False))

    return summary


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SpCas9 VESM_3B scorer")
    parser.add_argument(
        "--mode", choices=["all", "subset"], default="subset",
        help=(
            "'subset' scores only TARGET_POSITIONS (~seconds); "
            "'all' runs the full 1368-position saturation scan (~25 sec on 4x5090)"
        )
    )
    args = parser.parse_args()

    # ── GPU summary ──────────────────────────────────────────────────────
    if torch.cuda.is_available():
        print(f"\n🖥️  Detected {N_GPUS} CUDA device(s):")
        for i in range(N_GPUS):
            props = torch.cuda.get_device_properties(i)
            print(f"   [{i}] {props.name}  {props.total_memory // 1024**3} GB VRAM")
    else:
        print("⚠️  No CUDA device found — running on CPU (will be slow)")

    # ── Load model ───────────────────────────────────────────────────────
    tokenizer, model = load_model()

    # ── Choose positions ─────────────────────────────────────────────────
    if args.mode == "all":
        positions_0idx = list(range(len(CAS9_SEQ)))
        tag = "full"
        print(f"\n🚀 Full saturation scan: {len(positions_0idx)} positions, "
              f"batch_size={BATCH_SIZE}")
    else:
        positions_0idx = sorted(set(p - 1 for p in TARGET_POSITIONS
                                    if 1 <= p <= len(CAS9_SEQ)))
        tag = "switch_residues"
        print(f"\n🚀 Subset scan: {len(positions_0idx)} MD switch residues, "
              f"batch_size={BATCH_SIZE}")

    # ── Score ────────────────────────────────────────────────────────────
    df = score_positions(tokenizer, model, CAS9_SEQ, positions_0idx)

    # ── Save per-variant table ───────────────────────────────────────────
    out_variants = f"SpCas9_VESM3B_{tag}_variants.csv"
    df.to_csv(out_variants, index=False)
    print(f"✓ Variant scores   → {out_variants}  ({len(df)} rows)")

    # ── Save + print summary ─────────────────────────────────────────────
    summarise(df, tag)

    print("\n🎉 Done.")


if __name__ == "__main__":
    main()
