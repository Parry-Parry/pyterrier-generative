"""Test to verify TDPart fix - trace through the algorithm logic."""
import numpy as np
from pyterrier_generative.algorithms.common import RankedList

# Simulate the state after iteration 0 trimming
# We had 20 docs, after iteration 0:
# - c_keep = 10 docs (kept in candidates)
# - iteration_backfill = 10 docs (accumulated in backfill_accumulator)

print("=== Simulating TDPart Algorithm ===\n")

# Initial state: 20 docs
all_docs = [f'd{i}' for i in range(20)]
print(f"Initial: {len(all_docs)} docs")

# After iteration 0 trimming
c_keep_docs = [f'd{i}' for i in [9, 8, 7, 6, 18, 17, 16, 15, 14, 13]]
iteration_backfill_docs = [f'd{i}' for i in [12, 11, 10, 5, 4, 3, 2, 1, 0, 19]]

print(f"\nAfter Iteration 0:")
print(f"  candidates (c_keep): {len(c_keep_docs)} docs: {c_keep_docs}")
print(f"  backfill_accumulator: {len(iteration_backfill_docs)} docs: {iteration_backfill_docs}")
print(f"  Total accounted: {len(c_keep_docs) + len(iteration_backfill_docs)} docs")

# Iteration 1: process c_keep (10 docs)
# ranking = c_keep (this is the fix!)
# After ranking first 10 docs with pivot at position 4:
candidates_new = [f'd{i}' for i in [19, 0, 1, 2]]  # 4 docs (before pivot)
pivot = 'd3'  # 1 doc
backfill_new = [f'd{i}' for i in [4, 5, 10, 11, 12]]  # 5 docs (after pivot)

print(f"\nIteration 1 (processing c_keep):")
print(f"  After ranking 10 docs:")
print(f"    candidates: {len(candidates_new)} docs: {candidates_new}")
print(f"    pivot: {pivot}")
print(f"    backfill: {len(backfill_new)} docs: {backfill_new}")
print(f"    remainder: 0 docs (exhausted)")

# Since remainder is exhausted, we're done
# Final assembly:
# - candidates = candidates_new + pivot
# - backfill_accumulator += backfill_new

final_candidates = candidates_new + [pivot]
final_backfill = iteration_backfill_docs + backfill_new

print(f"\nFinal assembly:")
print(f"  candidates: {len(final_candidates)} docs: {final_candidates}")
print(f"  backfill_accumulator: {len(final_backfill)} docs: {final_backfill}")
print(f"  Total: {len(final_candidates) + len(final_backfill)} docs")

# Verify we have all 20 docs
all_final = set(final_candidates + final_backfill)
expected = set(all_docs)

if all_final == expected and len(all_final) == 20:
    print("\n✓ SUCCESS: All 20 documents accounted for!")
else:
    print(f"\n✗ FAILED: Missing or extra documents!")
    print(f"  Expected: {sorted(expected)}")
    print(f"  Got: {sorted(all_final)}")
    print(f"  Missing: {expected - all_final}")
    print(f"  Extra: {all_final - expected}")
