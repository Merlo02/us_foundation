"""Summarise inter-session downstream test accuracy across tuning groups.

For each group_0 dir we read every run's metrics.csv (one non-empty test/acc
row per run = one held-out test session) and its config.yaml hyperparameters,
then report mean +/- std of test accuracy across the runs of the group.
"""
import csv
import glob
import os
import statistics

import yaml

BASE = "/leonardo_scratch/fast/IscrB_WearUsFM/gmerlino/models_downstream/tuning"


def read_test_acc(run_dir):
    """Return the single non-empty test/acc value from a run's metrics.csv."""
    csvs = glob.glob(os.path.join(run_dir, "lightning_logs", "version_*", "metrics.csv"))
    if not csvs:
        return None
    csvs.sort()
    vals = []
    for path in csvs:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = row.get("test/acc", "")
                if v not in ("", None):
                    vals.append(float(v))
    if not vals:
        return None
    # one test phase per run -> last non-empty value is the final trainer.test()
    return vals[-1]


def read_cfg(run_dir):
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    m = cfg.get("model", {})
    ht = m.get("head_type")
    if isinstance(ht, dict):
        fusion = ht.get("type", "concat")
        num_queries = ht.get("num_queries")
    else:
        fusion = "concat"
        num_queries = None
    gu = m.get("gradual_unfreezeing") or {}
    gu_enabled = bool(gu.get("enabled", False))
    return {
        "fusion": fusion,
        "num_queries": num_queries,
        "gu_enabled": gu_enabled,
        "freeze_epochs": gu.get("freeze_epochs"),
        "lr": cfg.get("train", {}).get("lr", m.get("lr")),
        "layerwise_lr_decay": m.get("layerwise_lr_decay"),
    }


def summarise_group(group_dir):
    runs = sorted(glob.glob(os.path.join(group_dir, "run_*")))
    accs = []
    cfg0 = None
    for r in runs:
        a = read_test_acc(r)
        if a is None:
            continue
        accs.append(a)
        if cfg0 is None:
            cfg0 = read_cfg(r)
    if not accs:
        return None
    mean = statistics.mean(accs)
    std = statistics.stdev(accs) if len(accs) > 1 else 0.0
    return {"n": len(accs), "accs": accs, "mean": mean, "std": std, "cfg": cfg0}


def show(label, group_dir):
    s = summarise_group(group_dir)
    if s is None:
        print(f"\n## {label}\n   NO DATA at {group_dir}")
        return s
    c = s["cfg"]
    print(f"\n## {label}")
    print(f"   path: {group_dir}")
    print(f"   fusion={c['fusion']} num_queries={c['num_queries']} "
          f"gu_enabled={c['gu_enabled']} freeze_epochs={c['freeze_epochs']} "
          f"lr={c['lr']} lw_decay={c['layerwise_lr_decay']}")
    print(f"   n={s['n']}  per-run acc: " + ", ".join(f"{a:.4f}" for a in s["accs"]))
    print(f"   mean={s['mean']*100:.2f}%  std={s['std']*100:.2f}%")
    return s


# ---- Fixed single-group targets ----
show("concat", f"{BASE}/fullFinetuning_interp800_gesture/group_0")
show("posenc", f"{BASE}/positional_exp/fulFinetuning_interp800_posenc/tuning_20260605_114050/group_0")

# ---- concat_gradualFreezing: 2 tunings, pick best ----
print("\n==== concat_gradualFreezing candidates ====")
for t in sorted(glob.glob(f"{BASE}/positional_exp/fulFinetuning_interp800_gradualFreeze/tuning_*")):
    show(os.path.basename(t), os.path.join(t, "group_0"))

# ---- posenc_gradualFreezing: 2 tunings, pick best ----
print("\n==== posenc_gradualFreezing candidates ====")
for t in sorted(glob.glob(f"{BASE}/positional_exp/fulFinetuning_interp800_posenc_gradualFreeze/tuning_*")):
    show(os.path.basename(t), os.path.join(t, "group_0"))

# ---- cross-attention: all tunings in both attn dirs ----
print("\n==== cross-attention candidates ====")
attn_dirs = []
attn_dirs += sorted(glob.glob(f"{BASE}/positional_exp/fulFinetuning_interp800_attn/tuning_*"))
attn_dirs += sorted(glob.glob(f"{BASE}/attn_exp/fulFinetuning_interp800_attn/tuning_*"))
for t in attn_dirs:
    rel = t.replace(BASE + "/", "")
    show(rel, os.path.join(t, "group_0"))
