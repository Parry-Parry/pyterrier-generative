import re

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
        max_iters: int = 10
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
        # backend.generate() expects list of prompts/messages, returns list of outputs
        # It handles batching internally and supports both string and message format
        outputs = self.model.generate([prompt])

        # Extract the output text
        # Assuming generate() returns list of strings
        output_text = outputs[0]

        # Parse output to get ranking order
        order = self.parse_output(output_text, len(doc_texts))

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

        # Build prompts for all windows
        prompts = []
        for kwargs in windows_kwargs:
            doc_texts = kwargs.get('doc_text', [])
            query = kwargs.get('query', '')

            # Build prompt using existing prompt construction logic
            class DocRow:
                def __init__(self, idx, text):
                    self.text = text
                    self._idx = idx

            doc_rows = [(i, DocRow(i, text)) for i, text in enumerate(doc_texts)]

            prompt = self.make_prompt_from(
                docs=doc_rows,
                query=query,
                num=len(doc_texts),
                passages=doc_texts
            )
            prompts.append(prompt)

        # Batch generate using backend - this is where efficiency gains come from
        outputs = self.model.generate(prompts)

        # Parse all outputs
        orders = []
        for output_text, kwargs in zip(outputs, windows_kwargs):
            doc_texts = kwargs.get('doc_text', [])
            order = self.parse_output(output_text, len(doc_texts))
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

        # Check if we should use cross-query batching
        if hasattr(self, '_rank_windows_batch'):
            return self._transform_with_batching(inp)
        else:
            return self._transform_sequential(inp)

    def _transform_sequential(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Transform queries one at a time without batching."""
        results = []

        # Process each query independently
        for qid, query_group in inp.groupby('qid'):
            query = query_group['query'].iloc[0]

            # Apply the selected algorithm
            ranked_docnos, ranked_texts = self._apply_algorithm(query, query_group)

            # Build output DataFrame for this query
            num_docs = len(ranked_docnos)
            query_results = pd.DataFrame({
                'qid': [qid] * num_docs,
                'query': [query] * num_docs,
                'docno': ranked_docnos,
                'text': ranked_texts,
                'rank': range(num_docs),  # 0-indexed ranks
                'score': [num_docs - i for i in range(num_docs)]  # score = negative rank + 1
            })

            results.append(query_results)

        # Combine all query results
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame(
            columns=["qid", "query", "docno", "text", "rank", "score"]
        )

    def _transform_with_batching(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Transform queries with cross-query batching for efficiency."""

        # Collect all windows from all queries
        all_windows_data = collect_windows_for_batching(self, inp)

        if not all_windows_data:
            return pd.DataFrame(columns=["qid", "query", "docno", "text", "rank", "score"])

        # Batch process all windows at once across all queries
        windows_kwargs = [w['kwargs'] for w in all_windows_data]
        orders = self._rank_windows_batch(windows_kwargs)

        # Apply results back to each query
        results = apply_batched_results(all_windows_data, orders)

        # Combine all query results
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame(
            columns=["qid", "query", "docno", "text", "rank", "score"]
        )