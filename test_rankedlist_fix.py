"""Quick test for RankedList slice assignment fix."""
import numpy as np
from pyterrier_generative.algorithms.common import RankedList

# Test 1: Slice assignment from different RankedList
rl = RankedList(
    doc_idx=np.array(['d0', 'd1', 'd2', 'd3']),
    doc_texts=np.array(['t0', 't1', 't2', 't3'])
)
new_rl = RankedList(
    doc_idx=np.array(['d10', 'd11']),
    doc_texts=np.array(['t10', 't11'])
)
rl[1:3] = new_rl
print("Test 1 - Slice assignment:")
print(f"  Expected: ['d0', 'd10', 'd11', 'd3']")
print(f"  Got:      {list(rl.doc_idx)}")
assert list(rl.doc_idx) == ['d0', 'd10', 'd11', 'd3'], f"FAILED: {list(rl.doc_idx)}"
assert list(rl.doc_texts) == ['t0', 't10', 't11', 't3'], f"FAILED: {list(rl.doc_texts)}"
print("  PASSED")

# Test 2: Self-assignment with array indexing (like in apply_order)
rl2 = RankedList(
    doc_idx=np.array(['a', 'b', 'c', 'd']),
    doc_texts=np.array(['t_a', 't_b', 't_c', 't_d'])
)
order = np.array([3, 2, 1, 0])  # Reverse order
orig = np.arange(4)
rl2[orig] = rl2[order]
print("\nTest 2 - Self-assignment (reverse order):")
print(f"  Expected: ['d', 'c', 'b', 'a']")
print(f"  Got:      {list(rl2.doc_idx)}")
assert list(rl2.doc_idx) == ['d', 'c', 'b', 'a'], f"FAILED: {list(rl2.doc_idx)}"
assert list(rl2.doc_texts) == ['t_d', 't_c', 't_b', 't_a'], f"FAILED: {list(rl2.doc_texts)}"
print("  PASSED")

print("\nAll tests passed!")
