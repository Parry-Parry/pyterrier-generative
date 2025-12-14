"""LiT5 ranking model implementation."""

from typing import Optional, Union, List
import re
import numpy as np
import torch
from transformers import T5Tokenizer

from pyterrier_generative.modelling.base import GenerativeRanker
from pyterrier_generative._algorithms import Algorithm


class LiT5Backend:
    """
    Backend for LiT5 model using FiD (Fusion-in-Decoder) architecture.

    This backend is specific to LiT5 and doesn't follow the standard Backend interface
    since LiT5 requires special handling with its FiD architecture.
    """

    supports_message_input = False

    def __init__(
        self,
        model_path: str = 'castorini/LiT5-Distill-large',
        batch_size: int = 16,
        bfloat16: Optional[bool] = None,
        window_size: int = 20,
        device: Optional[Union[str, torch.device]] = None,
    ):
        from pyterrier_t5.modeling_fid import FiD

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.tokenizer = T5Tokenizer.from_pretrained(
            model_path,
            return_dict=False,
            legacy=False,
            use_fast=True
        )
        self.model = FiD.from_pretrained(model_path, from_flax=False).to(device).eval()
        self.model.encoder.config.n_passages = window_size
        self.model.encoder.config.batch_size = batch_size

        # Try to use bfloat16 if available
        if bfloat16 is None:
            try:
                self.model = self.model.bfloat16()
                bfloat16 = True
            except:
                bfloat16 = False
        elif bfloat16:
            self.model = self.model.bfloat16()

        self.bfloat16 = bfloat16
        self.window_size = window_size
        self.template = "Search Query: {q} Passage: [{i}] {d} Relevance Ranking: "

    def generate(self, prompts: List[str]) -> List[str]:
        """
        Generate rankings for the given prompts.

        Note: LiT5 doesn't use prompts in the traditional sense - it expects
        a special format. This method is here for interface compatibility but
        the actual ranking is done via the score method.
        """
        raise NotImplementedError(
            "LiT5Backend doesn't support standard generate(). "
            "Use the LiT5 ranker directly instead."
        )


class LiT5(GenerativeRanker):
    """
    LiT5 (Listwise T5) ranking model.

    LiT5 is a listwise ranking model based on T5 with Fusion-in-Decoder architecture.
    It's specifically designed for ranking and uses a different architecture than
    standard seq2seq models.

    See: https://huggingface.co/castorini/LiT5-Distill-large

    Parameters:
        model_path (str): HuggingFace model path. Defaults to 'castorini/LiT5-Distill-large'.
        batch_size (int): Batch size for encoding passages.
        bfloat16 (bool): Use bfloat16 precision. Auto-detected if None.
        window_size (int): Number of passages to rank at once.
        stride (int): Stride for sliding window algorithm.
        device (str or torch.device): Device to run on.
        algorithm (Algorithm): Ranking algorithm. Defaults to SLIDING_WINDOW.

    Example::

        from pyterrier_generative import LiT5
        import pyterrier as pt

        # Simple usage
        ranker = LiT5(window_size=20)

        # In a pipeline
        pipeline = bm25 % 20 >> ranker
        results = pipeline.search("What is information retrieval?")

        # With custom model
        ranker = LiT5(
            model_path='castorini/LiT5-Distill-base',
            window_size=10,
            bfloat16=True
        )
    """

    def __init__(
        self,
        model_path: str = 'castorini/LiT5-Distill-large',
        batch_size: int = 16,
        bfloat16: Optional[bool] = None,
        window_size: int = 20,
        stride: int = 10,
        device: Optional[Union[str, torch.device]] = None,
        algorithm: Algorithm = Algorithm.SLIDING_WINDOW,
        **kwargs
    ):
        """Initialize LiT5 ranker."""

        # Create LiT5-specific backend
        backend = LiT5Backend(
            model_path=model_path,
            batch_size=batch_size,
            bfloat16=bfloat16,
            window_size=window_size,
            device=device,
        )

        # We don't use a prompt for LiT5 - it has its own template
        # Pass a dummy callable that won't be used
        dummy_prompt = lambda **kwargs: ""

        # Initialize parent, but we'll override _rank_window
        super().__init__(
            model=backend,
            prompt=dummy_prompt,
            system_prompt="",
            algorithm=algorithm,
            window_size=window_size,
            stride=stride,
            **kwargs
        )

        self.model_path = model_path
        self.backend = backend  # Keep reference to LiT5-specific backend

    def _rank_window(self, **kwargs) -> list[int]:
        """
        Rank documents in a window using LiT5's FiD architecture.

        Overrides the base class method since LiT5 has special requirements.
        """
        doc_texts = kwargs.get('doc_text', [])
        query = kwargs.get('query', '')
        start_idx = kwargs.get('start_idx', 0)
        end_idx = kwargs.get('end_idx', len(doc_texts))
        window_len = kwargs.get('window_len', len(doc_texts))

        # Pad to window_size if needed
        padded_docs = doc_texts + ["" for _ in range(window_len, self.window_size)]

        # Format passages with LiT5's template
        passages = [
            self.backend.template.format(q=query, i=i+1, d=text)
            for i, text in enumerate(padded_docs)
        ]

        # Tokenize
        inputs = self.backend.tokenizer.batch_encode_plus(
            passages,
            return_tensors="pt",
            padding='max_length',
            max_length=150,
            truncation=True
        )

        # Generate ranking with FiD
        with torch.no_grad():
            outputs = self.backend.model.generate(
                input_ids=inputs['input_ids'].to(self.backend.device).reshape(1, -1),
                attention_mask=inputs['attention_mask'].to(self.backend.device).reshape(1, -1),
                max_length=100,
                do_sample=False,
            )

        # Decode output
        output = self.backend.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse ranking (same logic as base class parse_output)
        output = re.sub(r'[^0-9]', ' ', output)  # keep only digits
        output = [int(x)-1 for x in output.split()]  # convert to 0-indexed integers
        output = list({x: 0 for x in output if 0 <= x < window_len}.keys())  # dedupe and filter
        order = output + [i for i in range(window_len) if i not in output]  # backfill

        return order

    def _rank_windows_batch(self, windows_kwargs: list[dict]) -> list[list[int]]:
        """
        Batch-process multiple windows using LiT5's FiD architecture.

        This is more efficient than the base implementation because FiD can
        process multiple queries in parallel at the encoder level.
        """
        if not windows_kwargs:
            return []

        batch_size = len(windows_kwargs)
        all_passages = []
        window_lens = []

        # Prepare all windows
        for kwargs in windows_kwargs:
            doc_texts = kwargs.get('doc_text', [])
            query = kwargs.get('query', '')
            window_len = kwargs.get('window_len', len(doc_texts))
            window_lens.append(window_len)

            # Pad to window_size if needed
            padded_docs = doc_texts + ["" for _ in range(window_len, self.window_size)]

            # Format passages with LiT5's template
            passages = [
                self.backend.template.format(q=query, i=i+1, d=text)
                for i, text in enumerate(padded_docs)
            ]
            all_passages.extend(passages)

        # Tokenize all passages at once
        inputs = self.backend.tokenizer.batch_encode_plus(
            all_passages,
            return_tensors="pt",
            padding='max_length',
            max_length=150,
            truncation=True
        )

        # Reshape for FiD: (batch_size, window_size, seq_len)
        seq_len = inputs['input_ids'].shape[1]
        input_ids = inputs['input_ids'].to(self.backend.device).reshape(batch_size, self.window_size, seq_len)
        attention_mask = inputs['attention_mask'].to(self.backend.device).reshape(batch_size, self.window_size, seq_len)

        # Generate rankings for all windows in batch
        with torch.no_grad():
            outputs = self.backend.model.generate(
                input_ids=input_ids.reshape(batch_size, -1),
                attention_mask=attention_mask.reshape(batch_size, -1),
                max_length=100,
                do_sample=False,
            )

        # Parse outputs
        orders = []
        for i, window_len in enumerate(window_lens):
            output = self.backend.tokenizer.decode(outputs[i], skip_special_tokens=True)

            # Parse ranking (same logic as base class parse_output)
            output = re.sub(r'[^0-9]', ' ', output)  # keep only digits
            output = [int(x)-1 for x in output.split()]  # convert to 0-indexed integers
            output = list({x: 0 for x in output if 0 <= x < window_len}.keys())  # dedupe and filter
            order = output + [i for i in range(window_len) if i not in output]  # backfill
            orders.append(order)

        return orders

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"model_path={self.model_path!r}, "
            f"algorithm={self.algorithm.value!r}, "
            f"window_size={self.window_size})"
        )


__all__ = ['LiT5', 'LiT5Backend']
