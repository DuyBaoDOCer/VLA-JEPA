"""Inspect a VLA-JEPA checkpoint state_dict and classify keys by module prefix.

This is a structural check, not a model build: it only loads a state_dict
(a mapping of parameter name -> tensor) and counts keys per prefix. It never
constructs a model.

The project's `load_pretrained_backbones(..., reload_modules=...)` mechanism
assumes the checkpoint's module tree is exactly these nine prefixes: six that
get reloaded, three that get dropped (state/action encoder and action
decoder, whose dimensions differ between the pretrain task and this
finetune). If any key falls outside these nine prefixes, that assumption is
wrong and the reload logic cannot be trusted to know what it is loading.

Usage:
    python inspect_ckpt.py <checkpoint_path>
    python inspect_ckpt.py --self-test
"""

import argparse
import sys

import torch

# (prefix, role) -- role is informational only, printed in the report.
PREFIXES = [
    ("qwen_vl_interface", "LOAD"),
    ("vj_encoder", "LOAD"),
    ("vj_predictor", "LOAD"),
    ("action_model.model", "LOAD"),
    ("action_model.future_tokens", "LOAD"),
    ("action_model.position_embedding", "LOAD"),
    ("action_model.state_encoder", "DROP"),
    ("action_model.action_encoder", "DROP"),
    ("action_model.action_decoder", "DROP"),
]


def _looks_like_state_dict(obj):
    if not isinstance(obj, dict) or not obj:
        return False
    sample_value = next(iter(obj.values()))
    return torch.is_tensor(sample_value)


def load_state_dict(checkpoint_path):
    """Load a checkpoint and return the underlying state_dict.

    Handles both a raw state_dict (torch.save(model.state_dict(), path))
    and a wrapper dict with the state_dict nested under a common key.
    """
    obj = torch.load(checkpoint_path, map_location="cpu")
    if _looks_like_state_dict(obj):
        return obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in obj and _looks_like_state_dict(obj[key]):
                return obj[key]
        raise ValueError(
            "Could not find a state_dict inside checkpoint dict; "
            f"top-level keys: {list(obj.keys())[:20]}"
        )
    raise ValueError(f"Loaded checkpoint object is not a state_dict-like mapping: {type(obj)}")


def build_fake_state_dict():
    """A synthetic state_dict with exactly the nine expected prefixes."""
    torch.manual_seed(0)
    fake_params = {
        "qwen_vl_interface": [("weight", (4, 4)), ("bias", (4,))],
        "vj_encoder": [("layer.weight", (4, 4))],
        "vj_predictor": [("layer.weight", (4, 4)), ("layer.bias", (4,))],
        "action_model.model": [("block.weight", (4, 4))],
        "action_model.future_tokens": [("embedding", (4, 4))],
        "action_model.position_embedding": [("weight", (4, 4))],
        "action_model.state_encoder": [("weight", (4, 4))],
        "action_model.action_encoder": [("weight", (4, 4))],
        "action_model.action_decoder": [("weight", (4, 4))],
    }
    state_dict = {}
    for prefix, params in fake_params.items():
        for suffix, shape in params:
            state_dict[f"{prefix}.{suffix}"] = torch.randn(*shape)
    return state_dict


def classify(state_dict):
    """Return (counts per prefix, list of keys matching no prefix)."""
    counts = {prefix: 0 for prefix, _ in PREFIXES}
    matched_keys = set()
    for key in state_dict:
        for prefix, _ in PREFIXES:
            if key == prefix or key.startswith(prefix + "."):
                counts[prefix] += 1
                matched_keys.add(key)
                break
    unaccounted = [key for key in state_dict if key not in matched_keys]
    return counts, unaccounted


def report(state_dict):
    total_keys = len(state_dict)
    counts, unaccounted = classify(state_dict)

    print(f"Total keys in state_dict: {total_keys}")
    print("Prefix counts:")
    for prefix, role in PREFIXES:
        print(f"  {prefix:40s} {role:5s} {counts[prefix]}")

    for prefix, _ in PREFIXES:
        assert counts[prefix] > 0, f"Prefix '{prefix}' has zero matching keys"

    total_counted = sum(counts.values())
    if total_counted != total_keys:
        print(
            f"MISMATCH: prefix counts sum to {total_counted}, "
            f"but state_dict has {total_keys} keys"
        )
        print(f"Unaccounted keys ({len(unaccounted)}):")
        for key in sorted(unaccounted):
            print(f"  {key}")
        sys.exit(1)

    assert total_counted == total_keys, "sum of prefix counts must equal len(state_dict)"
    print(f"OK: prefix counts sum to {total_counted}, matches len(state_dict)")

    total_params = sum(tensor.numel() for tensor in state_dict.values())
    sample_key = next(iter(state_dict))
    sample_dtype = state_dict[sample_key].dtype
    print(f"Total parameters: {total_params}")
    print(f"Sample tensor dtype ({sample_key}): {sample_dtype}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a VLA-JEPA checkpoint state_dict by module prefix."
    )
    parser.add_argument(
        "checkpoint_path", nargs="?", help="Path to the checkpoint .pt file"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run against a synthetic state_dict, no checkpoint file needed",
    )
    args = parser.parse_args()

    if args.self_test:
        print("Running in --self-test mode: synthetic state_dict, no checkpoint file used")
        state_dict = build_fake_state_dict()
    else:
        if not args.checkpoint_path:
            parser.error("checkpoint_path is required unless --self-test is given")
        print(f"Loading checkpoint: {args.checkpoint_path}")
        state_dict = load_state_dict(args.checkpoint_path)

    report(state_dict)


if __name__ == "__main__":
    main()
