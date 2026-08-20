"""Numerically prove the multi-view embedding bug fixed by upstream 0dd5281.

Old code (VLA_JEPA.py, pre-fix):
    batch_videos = batch_videos.reshape(B * V, T, C, H, W)          # from [B, V, ...]
    ...
    video_embeddings = torch.cat(torch.chunk(video_embeddings, chunks=V, dim=0), dim=2)

The initial reshape from [B, V, ...] to [B*V, ...] is B-major: flat index
i = b * V + v. torch.chunk(x, chunks=V, dim=0) then slices the leading
dimension into V *contiguous* blocks of size B each, which is only the
correct (b, v) grouping when V == 1 or B == 1. For B, V > 1 it mixes views
from different batch elements.

New code (post-fix):
    x = video_embeddings.reshape(B, V, T_tok, P, D)
    x = x.permute(0, 2, 3, 1, 4)
    x = x.reshape(B, T_tok * P, V * D)

This reverses the original B-major collapse directly, so (b, v) pairing is
always preserved regardless of B and V.

This script builds an identifiable fake embedding tensor and computes both
expressions with no model, no checkpoint, torch only. It asserts four
concrete claims; any assertion failure means the analysis behind TIP-007c is
wrong and must be reported, not patched around.
"""

import torch


def build_fake_embeddings(B, V, tokens_per_clip, embed_dim):
    """emb[i], i = b*V + v, filled with value b*10+v (identifiable per (b, v))."""
    emb = torch.zeros(B * V, tokens_per_clip, embed_dim)
    for b in range(B):
        for v in range(V):
            i = b * V + v
            emb[i] = float(b * 10 + v)
    return emb


def old_expr(emb, V):
    """Pre-fix: chunk assumes V-major layout, cat along the embed-dim axis."""
    return torch.cat(torch.chunk(emb, chunks=V, dim=0), dim=2)


def new_expr(emb, B, V, tokens_per_clip, embed_dim):
    """Post-fix: reshape/permute/reshape recovers correct (b, v) pairing."""
    T_tok, P = 1, tokens_per_clip
    x = emb.reshape(B, V, T_tok, P, embed_dim)
    x = x.permute(0, 2, 3, 1, 4)
    x = x.reshape(B, T_tok * P, V * embed_dim)
    return x


def batch_row_values(row):
    """Distinct b*10+v values present in a [tokens, V*D] batch row."""
    return sorted(set(row.flatten().tolist()))


def main():
    tokens_per_clip, embed_dim = 4, 3

    # --- Case 1: B=2, V=2 (the case we actually train with) ---
    B, V = 2, 2
    emb = build_fake_embeddings(B, V, tokens_per_clip, embed_dim)
    old = old_expr(emb, V)
    new = new_expr(emb, B, V, tokens_per_clip, embed_dim)

    print(f"=== B={B}, V={V} ===")
    print("old[batch=0] distinct values:", batch_row_values(old[0]))
    print("old[batch=1] distinct values:", batch_row_values(old[1]))
    print("new[batch=0] distinct values:", batch_row_values(new[0]))
    print("new[batch=1] distinct values:", batch_row_values(new[1]))

    differ = not torch.equal(old, new)
    print("old != new:", differ)
    assert differ, "ASSERTION FAILED: old and new expressions agree at B=2,V=2"

    # new: batch row b must contain only values whose tens digit is b
    new_correct = True
    for b in range(B):
        vals = batch_row_values(new[b])
        if any(int(v) // 10 != b for v in vals):
            new_correct = False
    print("new rows contain only their own batch's values:", new_correct)
    assert new_correct, "ASSERTION FAILED: new result mixes batches"

    # old: batch row 0 must contain a value belonging to b=1 (the bug)
    old_row0_vals = batch_row_values(old[0])
    old_bug_present = any(int(v) // 10 == 1 for v in old_row0_vals)
    print("old row 0 contains a b=1 value (bug present):", old_bug_present)
    assert old_bug_present, "ASSERTION FAILED: old result does not exhibit the bug"

    # --- Case 2: B=1, V=2 (bug does not manifest) ---
    B1, V1 = 1, 2
    emb1 = build_fake_embeddings(B1, V1, tokens_per_clip, embed_dim)
    old1 = old_expr(emb1, V1)
    new1 = new_expr(emb1, B1, V1, tokens_per_clip, embed_dim)

    print(f"\n=== B={B1}, V={V1} ===")
    print("old[batch=0] distinct values:", batch_row_values(old1[0]))
    print("new[batch=0] distinct values:", batch_row_values(new1[0]))

    agree_at_b1 = torch.equal(old1, new1)
    print("old == new at B=1:", agree_at_b1)
    assert agree_at_b1, "ASSERTION FAILED: old and new expressions disagree at B=1"
    print(
        "Explanation: at B=1 there is only one batch element, so chunking "
        "V contiguous blocks of size B=1 along dim 0 happens to align with "
        "each block being exactly one view of that same batch element. The "
        "B-major/V-major mismatch only produces wrong pairings when B > 1."
    )

    print("\nAll four assertions PASSED.")


if __name__ == "__main__":
    main()
