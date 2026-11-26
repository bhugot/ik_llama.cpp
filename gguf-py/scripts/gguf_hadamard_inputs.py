#!/usr/bin/env python3
"""Fold a randomized block Hadamard rotation into the input side of selected weights.

For each selected weight W (ggml ne0 = input features) this writes W' = W diag(s) H_b, where H_b is the
orthonormal Walsh-Hadamard matrix of size b applied block-wise and s is a +/-1 sign vector (one per
input width). The output file carries the `prism.hadamard.*` metadata, so at inference the matching
activation transform x' = H_b (s * x) is inserted before each selected matmul and W' x' == W x.

Only F32 / F16 / BF16 weights can be rotated here. To get a quantized model whose weights are
quantized *after* rotation, rotate the unquantized GGUF with this script, then run llama-quantize.

With --metadata-only the weights are left untouched (they were produced directly in the rotated basis,
e.g. by quantization-aware training) and only the metadata is written; --signs-json then gives the
block size and the sign vectors that were used: {"block": 256, "signs": {"2048": [1, -1, ...], ...}}.

    python gguf_hadamard_inputs.py in.gguf out.gguf --block 256
    python gguf_hadamard_inputs.py in.gguf out.gguf --metadata-only --signs-json signs.json --max-layer 24

Gated delta-net models with more value heads than key heads (e.g. Qwen3.5 4B/9B/35B-A3B): convert_hf_to_gguf
stores the ssm_out input columns in tiled [head_dim, n_k, rep] order, while a rotation learned on the source
checkpoint is defined in its grouped [head_dim, rep, n_k] order. --regroup-ssm-out puts the columns of the
selected ssm_out weights back in grouped order and sets prism.hadamard.gdn_v_grouped, so that the runtime
regroups the activation the same way before the Hadamard transform:

    python gguf_hadamard_inputs.py in.gguf out.gguf --metadata-only --signs-json signs.json --regroup-ssm-out
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

import gguf  # noqa: E402


def get_field_data(reader: gguf.GGUFReader, key: str):
    field = reader.get_field(key)
    if not field or not field.types:
        return None
    main_type = field.types[0]
    if main_type == gguf.GGUFValueType.ARRAY:
        if field.types[-1] == gguf.GGUFValueType.STRING:
            return [str(bytes(field.parts[idx]), encoding="utf-8") for idx in field.data]
        return [pv for idx in field.data for pv in field.parts[idx].tolist()]
    if main_type == gguf.GGUFValueType.STRING:
        return str(bytes(field.parts[-1]), encoding="utf-8")
    return field.parts[-1][0]


logger = logging.getLogger("gguf-hadamard-inputs")

DEFAULT_PATTERNS = (
    r"\.ffn_down\.weight$",
    r"\.ffn_down_exps\.weight$",
    r"\.ffn_down_shexp\.weight$",
    r"\.ssm_out\.weight$",
    r"\.attn_output\.weight$",
)


def hadamard(n: int) -> np.ndarray:
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(np.float32)


def bf16_to_f32(a: np.ndarray) -> np.ndarray:
    return (a.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(a: np.ndarray) -> np.ndarray:
    u = a.astype(np.float32).view(np.uint32)
    u = u + (0x7FFF + ((u >> 16) & 1))  # round to nearest even
    return (u >> 16).astype(np.uint16)


def rotate(w: np.ndarray, signs: np.ndarray, hb: np.ndarray) -> np.ndarray:
    """w [..., n] float32 -> (w * s) @ blockdiag(H_b) along the last axis."""
    b = hb.shape[0]
    shp = w.shape
    x = (w * signs).reshape(-1, shp[-1] // b, b)
    return (x @ hb).reshape(shp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--block", type=int, default=256, help="Hadamard block size (power of 2)")
    ap.add_argument("--pattern", action="append", default=None,
                    help="regex on tensor names to rotate (repeatable); default: down / ssm_out / attn_output")
    ap.add_argument("--seed", type=int, default=0, help="seed of the sign vectors")
    ap.add_argument("--no-signs", action="store_true", help="plain block Hadamard (no random signs)")
    ap.add_argument("--signs-json", type=Path, default=None, help="block size and sign vectors to use")
    ap.add_argument("--metadata-only", action="store_true", help="weights are already rotated: metadata only")
    ap.add_argument("--max-layer", type=int, default=-1, help="only blk.i with i < N (e.g. skip an MTP layer)")
    ap.add_argument("--gdn-v-grouped", action="store_true",
                    help="set prism.hadamard.gdn_v_grouped (rotation defined in grouped value-head order)")
    ap.add_argument("--regroup-ssm-out", action="store_true",
                    help="put ssm_out input columns back in grouped value-head order (rotation learned on the "
                         "source checkpoint) and set prism.hadamard.gdn_v_grouped")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    given = json.loads(args.signs_json.read_text()) if args.signs_json else None
    b = int(given["block"]) if given else args.block
    if b < 2 or b & (b - 1):
        sys.exit("--block must be a power of 2")
    if args.output.exists() and not args.force:
        sys.exit(f"{args.output} exists (use --force)")
    pats = [re.compile(p) for p in (args.pattern or DEFAULT_PATTERNS)]

    reader = gguf.GGUFReader(args.input, "r")
    if reader.get_field("prism.hadamard.version") is not None:
        sys.exit("input already carries prism.hadamard metadata")
    arch = get_field_data(reader, gguf.Keys.General.ARCHITECTURE)
    ok_types = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16)

    def in_range(name: str) -> bool:
        m = re.match(r"blk\.(\d+)\.", name)
        return args.max_layer < 0 or (m is not None and int(m.group(1)) < args.max_layer)

    selected = [t for t in reader.tensors if any(p.search(t.name) for p in pats) and in_range(t.name)]
    for t in selected:
        if t.tensor_type not in ok_types and not args.metadata_only:
            sys.exit(f"{t.name} is {t.tensor_type.name}: only F32/F16/BF16 weights can be rotated")
        if int(t.shape[0]) % b:
            sys.exit(f"{t.name}: ne0={int(t.shape[0])} is not a multiple of {b}")
    if not selected:
        sys.exit("no tensor matches the patterns")
    widths = sorted({int(t.shape[0]) for t in selected})
    if given:
        signs = {int(k): np.asarray(v, dtype=np.float32) for k, v in given["signs"].items()}
        missing = [w for w in widths if w not in signs]
        if missing:
            sys.exit(f"--signs-json has no sign vector for widths {missing}")
        signs = {w: signs[w] for w in widths}
    else:
        rng = np.random.default_rng(args.seed)
        signs = {w: (np.ones(w, dtype=np.float32) if args.no_signs
                     else rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=w)) for w in widths}
    hb = hadamard(b)
    names = {t.name for t in selected}
    regroup = set()
    if args.regroup_ssm_out:
        n_k = get_field_data(reader, f"{arch}.ssm.group_count")
        n_v = get_field_data(reader, f"{arch}.ssm.time_step_rank")
        if not n_k or not n_v or int(n_v) % int(n_k):
            sys.exit("--regroup-ssm-out: missing or inconsistent ssm.group_count / ssm.time_step_rank")
        n_k, n_v = int(n_k), int(n_v)
        regroup = {t.name for t in selected if ".ssm_out." in t.name} if n_v != n_k else set()
        for t in selected:
            if t.name in regroup and t.tensor_type not in ok_types:
                sys.exit(f"{t.name} is {t.tensor_type.name}: --regroup-ssm-out needs F32/F16/BF16 weights")
            if t.name in regroup and int(t.shape[0]) % n_v:
                sys.exit(f"{t.name}: ne0={int(t.shape[0])} is not a multiple of {n_v} value heads")
        logger.info(f"regrouping {len(regroup)} ssm_out weights (key heads {n_k}, value heads {n_v})")
    logger.info(f"{'tagging' if args.metadata_only else 'rotating'} {len(selected)} tensors, block {b}, widths {widths}")

    writer = gguf.GGUFWriter(args.output, arch=arch)
    alignment = get_field_data(reader, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val = get_field_data(reader, field.name)
        if val is not None:
            sub = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
            writer.add_key_value(field.name, val, field.types[0], sub_type=sub)
    writer.add_uint32("prism.hadamard.version", 1)
    writer.add_uint32("prism.hadamard.block_size", b)
    writer.add_array("prism.hadamard.weight_names", sorted(names))
    writer.add_array("prism.hadamard.sign_widths", [int(w) for w in widths])
    writer.add_array("prism.hadamard.sign_values", [int(v) for w in widths for v in signs[w]])
    if args.gdn_v_grouped or regroup:
        writer.add_bool("prism.hadamard.gdn_v_grouped", True)

    for t in reader.tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        data = t.data
        if t.name in regroup:
            n = int(t.shape[0])
            raw = np.asarray(data)
            a = raw.view(np.uint16) if t.tensor_type == gguf.GGMLQuantizationType.BF16 else raw
            a = a.reshape(-1, n)
            o, hd = a.shape[0], n // n_v
            # tiled [hd, n_k, rep] -> grouped [hd, rep, n_k] (ggml order, fastest first)
            a = np.ascontiguousarray(a.reshape(o, n_v // n_k, n_k, hd).transpose(0, 2, 1, 3).reshape(o, n))
            data = a.view(np.uint8).reshape(raw.shape) if t.tensor_type == gguf.GGMLQuantizationType.BF16 else a.reshape(raw.shape)
            logger.debug(f"regrouped {t.name}")
        if t.name in names and not args.metadata_only:
            n = int(t.shape[0])
            if t.tensor_type == gguf.GGMLQuantizationType.BF16:
                data = f32_to_bf16(rotate(bf16_to_f32(np.asarray(data).view(np.uint16)), signs[n], hb)).view(np.uint8)
            else:
                data = rotate(np.asarray(data, dtype=np.float32), signs[n], hb).astype(data.dtype)
            logger.debug(f"rotated {t.name} {tuple(t.data.shape)}")
        writer.write_tensor_data(data)
    writer.close()
    logger.info(f"wrote {args.output}")


if __name__ == "__main__":
    main()
