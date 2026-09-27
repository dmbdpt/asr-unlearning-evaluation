"""Organize unlearning checkpoints referenced by a re-eval CSV into ~/unlearned_ckps."""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

CKPT_COL = "param_eval_from_run.checkpoint_path"
METHOD_COL = "param_unlearning.unlearner"
SUBJECT_COL = "forget_subject"
UUID_COL = "source_run_uuid"

# Folder names in ~/unlearned_ckps do not always match the unlearner id.
DEFAULT_METHOD_MAP = {"bad_teacher_smooth": "softbt"}

MANIFEST_METRICS = [
    "forget_post_metrics.cer",
    "forget_post_metrics.wer",
    "retain_post_metrics.wer",
    "test_post_metrics.wer",
    "post_speaker_level_auc",
    "post_utterance_level_auc",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?",
                   default=str(Path(__file__).resolve().parents[1] / "results" / "all_reeval_merged.csv"),
                   help="re-eval CSV to read (default: results/all_reeval_merged.csv)")
    p.add_argument("-o", "--out-dir", default=str(Path.home() / "unlearned_ckps"),
                   help="destination root (default: ~/unlearned_ckps)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--copy", action="store_const", dest="mode", const="copy",
                      help="copy checkpoint bytes instead of hardlinking")
    mode.add_argument("--symlink", action="store_const", dest="mode", const="symlink",
                      help="symlink to the source checkpoint instead of hardlinking")
    p.set_defaults(mode="hardlink")
    p.add_argument("--methods", nargs="+", metavar="NAME",
                   help="only these unlearners (CSV names, e.g. cfk finetune)")
    p.add_argument("--subjects", nargs="+", metavar="ID",
                   help="only these forget subjects")
    p.add_argument("--rename", nargs="+", default=[], metavar="CSV_NAME=FOLDER",
                   help="override the unlearner -> folder mapping")
    p.add_argument("-f", "--force", action="store_true",
                   help="replace destination checkpoints that already exist")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="report what would happen without touching the filesystem")
    return p.parse_args(argv)


def build_method_map(renames):
    mapping = dict(DEFAULT_METHOD_MAP)
    for item in renames:
        if "=" not in item:
            sys.exit(f"--rename expects CSV_NAME=FOLDER, got {item!r}")
        csv_name, folder = item.split("=", 1)
        mapping[csv_name] = folder
    return mapping


def load_rows(csv_path, args):
    """Read the CSV and keep the newest checkpoint per (method, subject)."""
    best, skipped = {}, []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            method, subject = row.get(METHOD_COL), row.get(SUBJECT_COL)
            src = (row.get(CKPT_COL) or "").strip()
            if not method or not subject:
                continue
            if args.methods and method not in args.methods:
                continue
            if args.subjects and subject not in args.subjects:
                continue
            if not src or src.lower() in {"none", "nan"}:
                skipped.append((method, subject, row.get(UUID_COL, ""), "no checkpoint path"))
                continue
            if not os.path.isfile(src):
                skipped.append((method, subject, row.get(UUID_COL, ""), f"missing: {src}"))
                continue
            # Resolve through symlinks: some checkpoint trees (e.g.
            row["_mtime"] = os.stat(src).st_mtime
            row["_src"] = os.path.realpath(src)
            row["_csv_src"] = src
            key = (method, subject)
            if key not in best or row["_mtime"] > best[key]["_mtime"]:
                best[key] = row
    return best, skipped


def place(src, dst, mode, dry_run):
    """Materialize dst from src, returning the action taken."""
    if dry_run:
        return mode
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp-organize")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    try:
        if mode == "hardlink":
            os.link(src, tmp)
        elif mode == "symlink":
            tmp.symlink_to(src)
        else:
            shutil.copy2(src, tmp)
    except OSError as exc:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        if mode == "hardlink":
            raise OSError(f"{exc} (source and destination may be on different "
                          f"filesystems; retry with --copy or --symlink)") from exc
        raise
    os.replace(tmp, dst)
    return mode


def main(argv=None):
    args = parse_args(argv)
    csv_path = Path(args.csv).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    if not csv_path.is_file():
        sys.exit(f"CSV not found: {csv_path}")

    method_map = build_method_map(args.rename)
    best, skipped = load_rows(csv_path, args)
    if not best:
        sys.exit("No usable rows found in the CSV for the requested filters.")

    placed, existing, manifest = 0, 0, []
    for (method, subject), row in sorted(best.items(), key=lambda kv: (kv[0][0], int(kv[0][1]))):
        folder = method_map.get(method, method)
        dst = out_dir / folder / str(subject) / "last.ckpt"
        src = row["_src"]

        if (dst.exists() or dst.is_symlink()) and not args.force:
            existing += 1
            print(f"skip   {folder}/{subject}/last.ckpt (exists; use --force to replace)")
        else:
            place(src, dst, args.mode, args.dry_run)
            placed += 1
            verb = {"hardlink": "link", "symlink": "symlink", "copy": "copy"}[args.mode]
            prefix = "would " if args.dry_run else ""
            print(f"{prefix}{verb:<7} {folder}/{subject}/last.ckpt <- {src}")

        entry = {"method_folder": folder, "unlearner": method, "forget_subject": subject,
                 "run_uuid": row.get(UUID_COL, ""), "source_checkpoint": row["_csv_src"],
                 "resolved_checkpoint": src, "dest": str(dst), "mode": args.mode}
        entry.update({m: row.get(m, "") for m in MANIFEST_METRICS})
        manifest.append(entry)

    if skipped:
        print(f"\n{len(skipped)} row(s) skipped:")
        for method, subject, uuid, why in skipped:
            print(f"  {method}/{subject} [{uuid[:8]}]: {why}")

    manifest_path = out_dir / "manifest.csv"
    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
            writer.writeheader()
            writer.writerows(manifest)

    print(f"\n{placed} placed, {existing} already present, {len(skipped)} skipped"
          f"{' (dry run, nothing written)' if args.dry_run else f'; manifest -> {manifest_path}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
