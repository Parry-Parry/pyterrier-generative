"""Common data structures and utilities for ranking algorithms."""
from functools import lru_cache
import numpy as np
from numpy import concatenate as concat
from tqdm.auto import tqdm


def iter_windows(n, window_size, stride, verbose: bool = False):
    """Iterate over sliding windows with given size and stride."""
    for start_idx in tqdm(range((n // stride) * stride, -1, -stride), disable=not verbose, unit='window'):
        end_idx = start_idx + window_size
        if end_idx > n:
            end_idx = n
        window_len = end_idx - start_idx
        if start_idx == 0 or window_len > stride:
            yield start_idx, end_idx, window_len


@lru_cache(maxsize=128)
def iter_windows_cached(n: int, window_size: int, stride: int):
    """
    Cached list of window tuples for a given (n, window_size, stride).
    Returns a tuple of (start_idx, end_idx, window_len) tuples.
    """
    return tuple(iter_windows(n, window_size, stride, verbose=True))


def split(ranked_list, i):
    """Split a list at index i."""
    return ranked_list[:i], ranked_list[i:]


class RankedList(object):
    """
    A ranked list of documents with parallel arrays for IDs and texts.

    This data structure maintains two parallel numpy arrays:
    - doc_idx: document identifiers
    - doc_texts: document text content

    It supports slicing, indexing, and concatenation operations.
    """

    def __init__(self, doc_idx=None, doc_texts=None) -> None:
        self.doc_texts = np.asarray(doc_texts) if doc_texts is not None else np.array([])
        self.doc_idx = np.asarray(doc_idx) if doc_idx is not None else np.array([])

    def __len__(self):
        return len(self.doc_idx)

    def view(self, start: int, end: int):
        """
        Return views (no copies) of the ranked list slice [start:end].

        Returns:
            (doc_idx_view, doc_texts_view): numpy array views
        """
        return self.doc_idx[start:end], self.doc_texts[start:end]

    def permute_window_inplace(self, start: int, end: int, order: np.ndarray) -> None:
        """
        In-place permutation of the window [start:end] according to `order`,
        where `order` is a permutation of 0..(end-start-1).

        This avoids allocating intermediate RankedList objects and avoids the
        slow list/ndarray __setitem__ path.
        """
        window_len = end - start
        if window_len <= 1:
            return

        # Make sure order is a numpy integer array
        order = np.asarray(order)
        if order.shape[0] != window_len:
            raise ValueError(f"order length {order.shape[0]} != window length {window_len}")

        # Copy out window then write back permuted
        tmp_idx = self.doc_idx[start:end].copy()
        tmp_txt = self.doc_texts[start:end].copy()
        self.doc_idx[start:end] = tmp_idx[order]
        self.doc_texts[start:end] = tmp_txt[order]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return RankedList(self.doc_idx[key].copy(), self.doc_texts[key].copy())
        elif isinstance(key, int):
            return RankedList(np.array([self.doc_idx[key]]), np.array([self.doc_texts[key]]))
        elif isinstance(key, list) or isinstance(key, np.ndarray):
            key_arr = np.asarray(key)
            return RankedList(self.doc_idx[key_arr].copy(), self.doc_texts[key_arr].copy())
        else:
            raise TypeError("Invalid key type. Please use int, slice, list, or numpy array.")

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.doc_idx[key], self.doc_texts[key] = value.doc_idx[0], value.doc_texts[0]
        elif isinstance(key, slice):
            # Extract the value arrays first to avoid issues with in-place modification
            new_idx = value.doc_idx.copy()
            new_texts = value.doc_texts.copy()

            if self.doc_idx.dtype.kind == 'O' or self.doc_texts.dtype.kind == 'O':
                self.doc_idx[key] = new_idx
                self.doc_texts[key] = new_texts
                return

            if self.doc_idx.dtype == new_idx.dtype and self.doc_texts.dtype == new_texts.dtype:
                self.doc_idx[key] = new_idx
                self.doc_texts[key] = new_texts
                return

            # Handle dtype issues: if new values have longer strings, we need to recreate the array
            # Get the slice indices
            start, stop, step = key.indices(len(self.doc_idx))
            slice_len = len(range(start, stop, step))

            if len(new_idx) != slice_len:
                raise ValueError(f"Slice assignment requires same length: slice has {slice_len} elements, value has {len(new_idx)}")

            # Check if we need to resize the dtype for string arrays
            if self.doc_idx.dtype.kind in ('U', 'S', 'O'):  # Unicode, bytes, or object
                # Create new arrays with appropriate dtype if needed
                max_idx_len = max(len(str(x)) for x in concat([self.doc_idx, new_idx]))
                max_text_len = max(len(str(x)) for x in concat([self.doc_texts, new_texts]))

                if self.doc_idx.dtype.kind == 'U' and self.doc_idx.dtype.itemsize // 4 < max_idx_len:
                    self.doc_idx = self.doc_idx.astype(f'U{max_idx_len}')
                if self.doc_texts.dtype.kind == 'U' and self.doc_texts.dtype.itemsize // 4 < max_text_len:
                    self.doc_texts = self.doc_texts.astype(f'U{max_text_len}')

            self.doc_idx[key] = new_idx
            self.doc_texts[key] = new_texts
        elif isinstance(key, list) or isinstance(key, np.ndarray):
            if len(key) != len(value):
                raise ValueError("Assigning RankedList requires the same length as the key.")
            # Extract values first to avoid issues when value is self
            new_idx = value.doc_idx.copy()
            new_texts = value.doc_texts.copy()

            # Handle dtype issues for string arrays
            if self.doc_idx.dtype.kind in ('U', 'S', 'O'):
                max_idx_len = max(len(str(x)) for x in concat([self.doc_idx, new_idx]))
                max_text_len = max(len(str(x)) for x in concat([self.doc_texts, new_texts]))

                if self.doc_idx.dtype.kind == 'U' and self.doc_idx.dtype.itemsize // 4 < max_idx_len:
                    self.doc_idx = self.doc_idx.astype(f'U{max_idx_len}')
                if self.doc_texts.dtype.kind == 'U' and self.doc_texts.dtype.itemsize // 4 < max_text_len:
                    self.doc_texts = self.doc_texts.astype(f'U{max_text_len}')

            for i, idx in enumerate(key):
                self.doc_idx[idx] = new_idx[i]
                self.doc_texts[idx] = new_texts[i]

    def __add__(self, other):
        if not isinstance(other, RankedList):
            raise TypeError("Unsupported operand type(s) for +: 'RankedList' and '{}'".format(type(other)))
        return RankedList(concat([self.doc_idx, other.doc_idx]), concat([self.doc_texts, other.doc_texts]))

    def __str__(self):
        return f"{self.doc_idx}, {self.doc_texts}"


__all__ = ['RankedList', 'iter_windows', 'iter_windows_cached', 'split']
