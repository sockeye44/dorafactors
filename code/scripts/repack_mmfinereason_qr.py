#!/usr/bin/env python3
# repack_mmfinereason_qr.py
#
# Converts:
#   question -> query
#   qwen3vl_235b_thinking_response -> response
#   image -> images = [image]
#   tok_len = len(tokenizer(query + joiner + response, add_special_tokens=False))
#
# Supports:
#   - source = HF dataset repo id OR local dataset snapshot path
#   - save_to_disk()
#   - push_to_hub()
#   - copy README.md from source repo/path to destination repo
#
# Example:
#   python3 repack_mmfinereason_qr.py \
#     --source OpenDataArena/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking \
#     --dest-repo eyes-ml/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking-QR \
#     --tokenizer Qwen/Qwen3-8B \
#     --save-to-disk /data/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking-QR \
#     --push
#
# Or from your local snapshot:
#   python3 repack_mmfinereason_qr.py \
#     --source /root/.cache/huggingface/hub/datasets--OpenDataArena--MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking/snapshots/cda6140bfba449bef9e57065a528b7424966e144 \
#     --dest-repo eyes-ml/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking-QR \
#     --tokenizer Qwen/Qwen3-8B \
#     --save-to-disk /data/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking-QR \
#     --push

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Optional

from datasets import DatasetDict, Image, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

# Datasets changed some feature helpers across versions, so be mildly defensive.
try:
    from datasets import List as HFList
except Exception:
    try:
        from datasets.features import List as HFList
    except Exception:
        HFList = None

try:
    from datasets import Sequence as HFSequence
except Exception:
    try:
        from datasets.features import Sequence as HFSequence
    except Exception:
        HFSequence = None


def build_images_feature():
    if HFList is not None:
        return HFList(Image())
    if HFSequence is not None:
        return HFSequence(Image())
    raise RuntimeError(
        "Could not import a list/sequence feature type from datasets. "
        "Please upgrade `datasets`."
    )


def is_local_source(source: str) -> bool:
    return Path(source).exists()


def load_source_dataset(source: str, config: Optional[str], token: Optional[str]) -> DatasetDict:
    if is_local_source(source):
        if config is not None:
            # Local snapshots generally don't need config, but keep the branch explicit.
            return load_dataset(source, name=config)
        return load_dataset(source)
    else:
        if config is not None:
            return load_dataset(source, name=config, token=token)
        return load_dataset(source, token=token)


def maybe_cast_image_decode_false(ds):
    # Avoid decoding every image into PIL during map().
    if "image" in ds.column_names:
        try:
            return ds.cast_column("image", Image(decode=False))
        except Exception:
            # Fall back gracefully if the source feature is already in a weird-but-usable shape.
            return ds
    return ds


def inject_note_after_yaml_front_matter(readme_text: str, note_md: str) -> str:
    """
    Preserve HF dataset-card YAML front matter.
    If README starts with:
      ---
      ...
      ---
    then insert note AFTER that block.
    """
    if not readme_text.startswith("---\n"):
        return note_md.rstrip() + "\n\n" + readme_text

    # Find the closing front-matter fence.
    parts = readme_text.split("\n")
    if not parts or parts[0] != "---":
        return note_md.rstrip() + "\n\n" + readme_text

    for i in range(1, len(parts)):
        if parts[i] == "---":
            head = "\n".join(parts[: i + 1]).rstrip() + "\n\n"
            tail = "\n".join(parts[i + 1 :]).lstrip("\n")
            return head + note_md.rstrip() + "\n\n" + tail

    # Malformed front matter; just prepend.
    return note_md.rstrip() + "\n\n" + readme_text


def load_source_readme(source: str, token: Optional[str]) -> Optional[str]:
    if is_local_source(source):
        p = Path(source) / "README.md"
        if p.exists():
            return p.read_text(encoding="utf-8")
        return None

    try:
        readme_path = hf_hub_download(
            repo_id=source,
            filename="README.md",
            repo_type="dataset",
            token=token,
        )
        return Path(readme_path).read_text(encoding="utf-8")
    except Exception:
        return None


def save_temp_readme(text: str) -> str:
    fd, path = tempfile.mkstemp(prefix="hf_readme_", suffix=".md")
    os.close(fd)
    Path(path).write_text(text, encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        required=True,
        help="HF dataset repo id or local dataset snapshot path.",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="Optional dataset config/subset name.",
    )
    ap.add_argument(
        "--dest-repo",
        default=None,
        help="Destination dataset repo id, e.g. username/my-dataset. Required if --push.",
    )
    ap.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="If set, keep only rows with tok_len <= max_len.",
    )
    ap.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-8B",
        help="Tokenizer repo to use for tok_len.",
    )
    ap.add_argument(
        "--joiner",
        default="\n\n",
        help=r'String inserted between query and response before tokenization. Default: "\n\n"',
    )
    ap.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Include tokenizer special tokens in tok_len. Default: off.",
    )
    ap.add_argument(
        "--keep-extra-cols",
        action="store_true",
        help="Keep original source columns in addition to id/query/response/images/tok_len.",
    )
    ap.add_argument(
        "--save-to-disk",
        default=None,
        help="Optional local output path for DatasetDict.save_to_disk().",
    )
    ap.add_argument(
        "--push",
        action="store_true",
        help="Push processed dataset to Hub.",
    )
    ap.add_argument(
        "--private",
        action="store_true",
        help="Create destination repo as private if it doesn't already exist.",
    )
    ap.add_argument(
        "--token",
        default=None,
        help="HF token. If omitted, uses the local HF login/token env.",
    )
    ap.add_argument(
        "--readme-mode",
        choices=["prepend-note", "copy", "skip"],
        default="prepend-note",
        help="How to handle source README.md when pushing.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Tokenization map batch size.",
    )
    args = ap.parse_args()

    if args.push and not args.dest_repo:
        raise ValueError("--dest-repo is required when --push is set")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"[1/6] Loading source dataset: {args.source}")
    ds = load_source_dataset(args.source, args.config, args.token)

    print(f"[2/6] Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def transform_batch(batch):
        questions = batch["question"]
        responses = batch["qwen3vl_235b_thinking_response"]
        images_in = batch["image"]

        queries = [q if q is not None else "" for q in questions]
        outputs = [r if r is not None else "" for r in responses]
        texts = [q + args.joiner + r for q, r in zip(queries, outputs)]

        enc = tokenizer(
            texts,
            add_special_tokens=args.add_special_tokens,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        tok_len = [len(x) for x in enc["input_ids"]]

        # images becomes a single-item list per row
        images = [[img] if img is not None else [] for img in images_in]

        out = {
            "query": queries,
            "response": outputs,
            "images": images,
            "tok_len": tok_len,
        }
        return out

    out = DatasetDict()
    image_list_feature = build_images_feature()

    print("[3/6] Transforming splits")
    for split_name, split_ds in ds.items():
        print(f"  - split={split_name}, rows={len(split_ds)}")
        split_ds = maybe_cast_image_decode_false(split_ds)

        mapped = split_ds.map(
            transform_batch,
            batched=True,
            batch_size=args.batch_size,
            desc=f"Transforming {split_name}",
        )
        
        if args.max_len is not None:
            before_n = len(mapped)
            mapped = mapped.filter(
                lambda ex: ex["tok_len"] <= args.max_len,
                desc=f"Filtering {split_name} by tok_len <= {args.max_len}",
            )
            after_n = len(mapped)
            print(
                f"  - split={split_name}: kept {after_n}/{before_n} "
                f"rows ({after_n / before_n:.2%}) with tok_len <= {args.max_len}"
            )
        
        if not args.keep_extra_cols:
            keep_cols = ["query", "response", "images", "tok_len"]
            if "id" in mapped.column_names:
                keep_cols = ["id"] + keep_cols
            remove_cols = [c for c in mapped.column_names if c not in keep_cols]
            mapped = mapped.remove_columns(remove_cols)
        
        mapped = mapped.cast_column("images", image_list_feature)
        out[split_name] = mapped
    print("[4/6] Result")
    print(out)
    for split_name, split_ds in out.items():
        print(f"  - {split_name}: {split_ds.features}")

    if args.save_to_disk:
        print(f"[5/6] Saving to disk: {args.save_to_disk}")
        out.save_to_disk(args.save_to_disk)

    if args.push:
        print(f"[6/6] Pushing dataset to Hub: {args.dest_repo}")
        out.push_to_hub(
            args.dest_repo,
            private=args.private,
            token=args.token,
        )

        if args.readme_mode != "skip":
            source_readme = load_source_readme(args.source, args.token)
            if source_readme is not None:
                if args.readme_mode == "copy":
                    final_readme = source_readme
                else:
                    note = f"""# Derived dataset note

This dataset was derived from `{args.source}`.

Field changes:
- `question` -> `query`
- `qwen3vl_235b_thinking_response` -> `response`
- `image` -> `images` (single-item list)
- added `tok_len`, computed with tokenizer `{args.tokenizer}` on `query + {args.joiner!r} + response`
- `add_special_tokens={args.add_special_tokens}`

The original README content is preserved below.
"""
                    final_readme = inject_note_after_yaml_front_matter(source_readme, note)

                tmp_readme = save_temp_readme(final_readme)
                api = HfApi(token=args.token)
                api.upload_file(
                    path_or_fileobj=tmp_readme,
                    path_in_repo="README.md",
                    repo_id=args.dest_repo,
                    repo_type="dataset",
                )
                print("README.md uploaded.")
            else:
                print("No source README.md found; skipping README upload.")

    print("Done.")


if __name__ == "__main__":
    main()

