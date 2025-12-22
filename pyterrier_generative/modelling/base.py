import re
from typing import Optional

import pyterrier as pt
import pyterrier_alpha as pta
from pyterrier_rag.backend import Backend
from pyterrier_rag.prompt.jinja import jinja_formatter
import pandas as pd

from pyterrier_generative._algorithms import (
    Algorithm,
    collect_windows_for_batching,
    apply_batched_results,
    sliding_window,
    single_window,
    setwise,
    tdpart
)
from pyterrier_generative.truncation import (
    get_token_counter,
    truncate_documents_iterative
)

class GenerativeRanker(pt.Transformer):
    """
    Base class for generative rankers in PyTerrier Generative.

    This class provides a template for implementing generative ranking models.
    Subclasses should implement the `generate` method to define how documents
    are ranked based on the input queries.

    Attributes:
        model_name (str): The name of the generative model to be used.
        prompt (str): The prompt template for the generative model.
        system_prompt (str): The system prompt for the generative model.
    """

    def __init__(
        self,
        model: Backend,
        prompt: str,
        system_prompt: str = "",
        algorithm: Algorithm = Algorithm.SLIDING_WINDOW,
        # Algorithm-specific parameters with reasonable defaults
        window_size: int = 20,
        stride: int = 10,
        buffer: int = 20,
        cutoff: int = 10,
        k: int = 10,
        max_iters: int = 10,
        # Document truncation parameters
        truncate_docs: bool = False,
        max_prompt_length: Optional[int] = None,
        truncate_tokens_per_iter: int = 50,
        truncate_max_iters: int = 100
    ):
        """
        Initializes the GenerativeRanker with the specified model name.

        Args:
            model (Backend): The backend model to be used for ranking.
            prompt (str or callable): Prompt template (Jinja2) or custom function.
            system_prompt (str): System instructions for the LLM.
            algorithm (Algorithm): Ranking algorithm to use.
            window_size (int): Size of ranking window for windowed algorithms.
            stride (int): Stride for sliding window algorithm.
            buffer (int): Buffer size for tdpart algorithm.
            cutoff (int): Cutoff position for tdpart algorithm.
            k (int): Top-k cutoff for setwise algorithm.
            max_iters (int): Maximum iterations for tdpart algorithm.
            truncate_docs (bool): Enable automatic document truncation when prompts exceed max length.
            max_prompt_length (int, optional): Maximum prompt length in tokens. If None, uses backend's max_input_length.
            truncate_tokens_per_iter (int): Number of tokens to remove per document per iteration.
            truncate_max_iters (int): Maximum truncation iterations before giving up.
        """
        self.model = model
        self.prompt = prompt if callable(prompt) else jinja_formatter(prompt)
        self.make_prompt_from = (
            self.callable_prompt
            if callable(prompt)
            else self.string_prompt
        )
        self.system_prompt = system_prompt
        self.algorithm = algorithm
        self.window_size = window_size
        self.stride = stride
        self.buffer = buffer
        self.cutoff = cutoff
        self.k = k
        self.max_iters = max_iters

        # Document truncation settings
        self.truncate_docs = truncate_docs
        self.max_prompt_length = max_prompt_length
        self.truncate_tokens_per_iter = truncate_tokens_per_iter
        self.truncate_max_iters = truncate_max_iters

        # Initialize token counter lazily (only if truncation is enabled)
        self._token_counter = None

    def string_prompt(self, docs, **query_columns):
        prompt_text = self.prompt(docs=docs, **query_columns)
        if self.model.supports_message_input:
            messages = []
            if self.system_prompt is not None:
                messages.append({'role': 'system', 'content': self.system_prompt})
            messages.append({'role': 'user', 'content': prompt_text})
            return messages
        else:
            if self.system_prompt is not None:
                prompt_text = self.system_prompt + "\n\n" + prompt_text
            return prompt_text

    def callable_prompt(self, **query_columns):
        # Callable prompts receive query_columns but not docs (which is Jinja-specific)
        # Remove docs from query_columns if present (it's only used for Jinja templates)
        query_columns.pop('docs', None)
        prompt_output = self.prompt(**query_columns)
        if self.model.supports_message_input:
            messages = []
            if self.system_prompt is not None:
                messages.append({'role': 'system', 'content': self.system_prompt})
            if isinstance(prompt_output, str):
                messages.append({'role': 'user', 'content': prompt_output})
            else:
                messages.extend(prompt_output)
            return messages
        else:
            if isinstance(prompt_output, str):
                if self.system_prompt is not None:
                    return self.system_prompt + "\n\n" + prompt_output
                return prompt_output
            else:
                # For callable prompts that return messages, extract content
                content = ""
                for msg in prompt_output:
                    if msg.get('role') == 'system':
                        content += msg.get('content', '') + "\n\n"
                    else:
                        content += msg.get('content', '')
                if self.system_prompt is not None:
                    content = self.system_prompt + "\n\n" + content
                return content

    def parse_output(self, output : str, length : int) -> list[int]:
        output = re.sub(r'[^0-9]', ' ', output) # clean outputs (keep only digits)
        output = [int(x)-1 for x in output.split()] # convert to integer
        output = list({x: 0 for x in output if 0 <= x < length}.keys()) # remove duplicates (but keep order) and remove anything out of range
        order = output + [i for i in range(length) if i not in output] # backfill missing passages
        return order

    def _get_token_counter(self):
        """Lazily initialize and return the token counter."""
        if self._token_counter is None:
            self._token_counter = get_token_counter(self.model)
        return self._token_counter

    def _build_prompt_for_texts(self, doc_texts: list, query: str):
        """
        Build the actual prompt for the given documents and query.

        Args:
            doc_texts: List of document text strings
            query: Query string

        Returns:
            The built prompt (string or list of messages)
        """
        # Create DocRow objects (same as in _rank_window)
        class DocRow:
            def __init__(self, idx, text):
                self.text = text
                self._idx = idx

        doc_rows = [(i, DocRow(i, text)) for i, text in enumerate(doc_texts)]

        # Build prompt using the same logic as _rank_window
        prompt = self.make_prompt_from(
            docs=doc_rows,
            query=query,
            num=len(doc_texts),
            passages=doc_texts
        )

        return prompt

    def _count_prompt_tokens(self, prompt, token_counter) -> int:
        """
        Count tokens in a prompt (handles both string and message formats).

        Args:
            prompt: Either a string or list of message dicts
            token_counter: TokenCounter instance

        Returns:
            Total token count for the prompt
        """
        if isinstance(prompt, str):
            return token_counter.count_tokens(prompt)
        elif isinstance(prompt, list):
            # Message format: [{'role': 'system', 'content': '...'}, ...]
            total = 0
            for msg in prompt:
                content = msg.get('content', '')
                total += token_counter.count_tokens(content)
                # Add small overhead for message formatting (role, structure, etc.)
                total += 4  # Approximate tokens for {"role": "...", "content": ""}
            return total
        else:
            raise ValueError(f"Unknown prompt format: {type(prompt)}")

    def _apply_truncation(self, doc_texts: list, query: str) -> list:
        """
        Apply document truncation if enabled and necessary.

        Args:
            doc_texts: List of document text strings
            query: Query string

        Returns:
            List of potentially truncated document texts
        """
        if not self.truncate_docs:
            return doc_texts

        # Get max prompt length (use backend's max_input_length if not specified)
        max_length = self.max_prompt_length
        if max_length is None:
            max_length = getattr(self.model, 'max_input_length', None)
            if max_length is None:
                # No max length available, skip truncation
                return doc_texts

        # Get token counter
        token_counter = self._get_token_counter()

        # Create a callback that builds the actual prompt and counts tokens
        def build_and_count(texts):
            prompt = self._build_prompt_for_texts(texts, query)
            return self._count_prompt_tokens(prompt, token_counter)

        # Apply iterative truncation with actual prompt building
        truncated_texts, success = truncate_documents_iterative(
            doc_texts=doc_texts,
            prompt_builder_and_counter=build_and_count,
            max_length=max_length,
            token_counter=token_counter,
            tokens_to_remove_per_iter=self.truncate_tokens_per_iter,
            max_iterations=self.truncate_max_iters
        )

        if not success:
            # Truncation failed - final prompt still exceeds max_length
            final_tokens = build_and_count(truncated_texts)
            raise ValueError(
                f"Document truncation failed to fit within max_length={max_length}. "
                f"Final prompt size is {final_tokens} tokens (exceeds by {final_tokens - max_length}). "
                f"Consider: (1) reducing window_size, (2) increasing max_prompt_length, "
                f"(3) increasing truncate_tokens_per_iter, or (4) increasing truncate_max_iters."
            )

        return truncated_texts

    def _rank_window(self, **kwargs) -> list[int]:
        """
        Callable wrapper for algorithms. Takes window kwargs and returns ranking order.

        Args from algorithms:
            qid: query ID
            query: query string
            doc_text: list of document texts
            doc_idx: list of document IDs (docnos)
            start_idx, end_idx, window_len: window metadata

        Returns:
            list[int]: 0-indexed ordering of documents in the window
        """
        # Extract what we need
        doc_texts = kwargs.get('doc_text', [])
        query = kwargs.get('query', '')

        # Apply truncation if enabled
        doc_texts = self._apply_truncation(doc_texts, query)

        # Build prompt using existing prompt construction logic
        # The prompt methods expect docs as (index, row) iterator
        # We create a simple row object that exposes text attribute
        class DocRow:
            def __init__(self, idx, text):
                self.text = text
                self._idx = idx

        # Create iterator of (index, DocRow) tuples
        doc_rows = [(i, DocRow(i, text)) for i, text in enumerate(doc_texts)]

        # Call appropriate prompt method (string_prompt or callable_prompt)
        # Pass both docs iterator AND extracted fields for template flexibility
        prompt = self.make_prompt_from(
            docs=doc_rows,
            query=query,
            num=len(doc_texts),
            passages=doc_texts
        )

        # Generate using backend
        # backend.generate() expects list of prompts/messages, returns list of output objects
        # Each output object must have a .text attribute
        outputs = self.model.generate([prompt])

        # Extract the output text from the output object
        output_text = outputs[0]
        text = output_text.text

        # Parse output to get ranking order
        order = self.parse_output(text, len(doc_texts))

        return order

    def _rank_windows_batch(self, windows_kwargs: list[dict]) -> list[list[int]]:
        """
        Batch-process multiple windows at once for improved efficiency.

        Args:
            windows_kwargs: List of kwargs dicts, each containing:
                - query: query string
                - doc_text: list of document texts
                - (other metadata fields are ignored for ranking)

        Returns:
            list[list[int]]: List of 0-indexed orderings, one per window
        """
        if not windows_kwargs:
            return []

        # Apply truncation to all windows and build prompts
        prompts = []
        truncated_windows = []

        for kwargs in windows_kwargs:
            doc_texts = kwargs.get('doc_text', [])
            query = kwargs.get('query', '')

            # Apply truncation if enabled
            doc_texts_truncated = self._apply_truncation(doc_texts, query)
            truncated_windows.append(doc_texts_truncated)

            # Build prompt using existing prompt construction logic
            class DocRow:
                def __init__(self, idx, text):
                    self.text = text
                    self._idx = idx

            doc_rows = [(i, DocRow(i, text)) for i, text in enumerate(doc_texts_truncated)]

            prompt = self.make_prompt_from(
                docs=doc_rows,
                query=query,
                num=len(doc_texts_truncated),
                passages=doc_texts_truncated
            )
            prompts.append(prompt)

        # Batch generate using backend - this is where efficiency gains come from
        # Backend returns list of output objects, each with a .text attribute
        outputs = self.model.generate(prompts)

        orders = []
        for output_text, doc_texts in zip(outputs, truncated_windows):
            # Extract text from output object
            text = output_text.text
            order = self.parse_output(text, len(doc_texts))
            orders.append(order)

        return orders

    def __call__(self, **kwargs):
        """Allow algorithms to call this instance as model(**kwargs)"""
        return self._rank_window(**kwargs)

    def _apply_algorithm(self, query: str, query_results: pd.DataFrame):
        """
        Apply the selected algorithm to rank documents for a single query.

        Returns:
            tuple: (doc_idx_array, doc_texts_array) - ALWAYS in this order
        """

        # Dispatch based on algorithm type
        if self.algorithm == Algorithm.SLIDING_WINDOW:
            result = sliding_window(self, query, query_results)
        elif self.algorithm == Algorithm.SINGLE_WINDOW:
            result = single_window(self, query, query_results)
        elif self.algorithm == Algorithm.SETWISE:
            result = setwise(self, query, query_results)
        elif self.algorithm == Algorithm.TDPART:
            result = tdpart(self, query, query_results)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return result

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by applying the generative ranking model.

        Args:
            inp (pd.DataFrame): Input DataFrame with columns: qid, query, docno, text, score

        Returns:
            pd.DataFrame: DataFrame with re-ranked documents
        """
        pta.validate.columns(inp, includes=['qid', 'query', 'docno', 'text'])

        if inp is None or inp.empty:
            return pd.DataFrame(columns=["qid", "query", "docno", "text", "rank", "score"])

        # Always use cross-query batching for efficiency
        return self._transform_with_batching(inp)

    def _transform_with_batching(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Transform queries with cross-query batching for efficiency."""

        # Collect all windows from all queries
        # For SLIDING_WINDOW and TDPART, this also executes the algorithm
        all_windows_data = collect_windows_for_batching(self, inp)

        if not all_windows_data:
            return pd.DataFrame(
                columns=["qid", "query", "docno", "text", "rank", "score"]
            )

        # Check if algorithm already processed everything
        # (SLIDING_WINDOW and TDPART process during collection)
        already_processed = (
            'tdpart_state' in all_windows_data[0] or
            'sliding_window_state' in all_windows_data[0]
        )

        if already_processed:
            # Rankings already finalized, just build results
            orders = None
        else:
            # SINGLE_WINDOW: batch process all windows
            windows_kwargs = [w['kwargs'] for w in all_windows_data]
            orders = self._rank_windows_batch(windows_kwargs)

        # Apply results back to each query
        results = apply_batched_results(all_windows_data, orders)

        # Combine all query results
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame(
            columns=["qid", "query", "docno", "text", "rank", "score"]
        )