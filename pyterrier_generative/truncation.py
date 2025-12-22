"""
Document truncation utilities for managing prompt lengths across different backends.
"""

from typing import List, Optional, Tuple
import warnings


class TokenCounter:
    """Base class for token counting across different backends."""

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        raise NotImplementedError

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for a batch of texts."""
        return [self.count_tokens(text) for text in texts]


class TiktokenCounter(TokenCounter):
    """Token counter for OpenAI models using tiktoken."""

    def __init__(self, model_id: str):
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for OpenAI token counting. "
                "Install it with: pip install tiktoken"
            )

        # Map model IDs to tiktoken encoding names
        # https://github.com/openai/tiktoken/blob/main/tiktoken/model.py
        try:
            self.encoding = tiktoken.encoding_for_model(model_id)
        except KeyError:
            # Default to cl100k_base for unknown models (GPT-4, GPT-3.5-turbo default)
            warnings.warn(
                f"Unknown model {model_id}, using cl100k_base encoding. "
                "This may not be accurate for all models."
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.encoding.encode(text))


class HuggingFaceCounter(TokenCounter):
    """Token counter for HuggingFace models using transformers tokenizer."""

    def __init__(self, tokenizer):
        """
        Initialize with a HuggingFace tokenizer.

        Args:
            tokenizer: Either a tokenizer instance or model_id string
        """
        if isinstance(tokenizer, str):
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens using HuggingFace tokenizer."""
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for a batch of texts efficiently."""
        encoded = self.tokenizer(texts, add_special_tokens=True)
        return [len(ids) for ids in encoded['input_ids']]


def get_token_counter(backend, model_id: Optional[str] = None) -> TokenCounter:
    """
    Get the appropriate token counter for the given backend.

    Args:
        backend: Backend instance (VLLMBackend, HuggingFaceBackend, OpenAIBackend)
        model_id: Optional model ID override

    Returns:
        TokenCounter instance appropriate for the backend
    """
    backend_class_name = backend.__class__.__name__

    if backend_class_name == 'OpenAIBackend':
        # Use tiktoken for OpenAI models
        model = model_id or backend.model_id
        return TiktokenCounter(model)

    elif backend_class_name in ('HuggingFaceBackend', 'Seq2SeqLMBackend'):
        # Use the backend's tokenizer directly
        return HuggingFaceCounter(backend.tokenizer)

    elif backend_class_name == 'VLLMBackend':
        # vLLM uses HuggingFace tokenizers under the hood
        # Access the tokenizer from the LLM instance
        try:
            tokenizer = backend.model.get_tokenizer()
            return HuggingFaceCounter(tokenizer)
        except Exception:
            # Fallback: load tokenizer from model_id
            model = model_id or backend.model_id
            return HuggingFaceCounter(model)

    else:
        raise ValueError(f"Unknown backend type: {backend_class_name}")


def truncate_documents_iterative(
    doc_texts: List[str],
    prompt_builder_and_counter,
    max_length: int,
    token_counter: TokenCounter,
    tokens_to_remove_per_iter: int = 50,
    max_iterations: int = 100
) -> Tuple[List[str], bool]:
    """
    Iteratively truncate documents to fit within max_length budget.

    This algorithm removes a fixed number of tokens from each document in each iteration
    until the total prompt fits within max_length or max_iterations is reached.

    Args:
        doc_texts: List of document text strings
        prompt_builder_and_counter: Callable that takes doc_texts and returns token count of built prompt
        max_length: Maximum allowed token count for the full prompt
        token_counter: TokenCounter instance for counting tokens (used for per-doc truncation)
        tokens_to_remove_per_iter: Number of tokens to remove from each document per iteration
        max_iterations: Maximum number of iterations to prevent infinite loops

    Returns:
        Tuple of (truncated_doc_texts, success)
        - truncated_doc_texts: List of potentially truncated document texts
        - success: True if we successfully fit within max_length, False otherwise
    """
    if not doc_texts:
        return doc_texts, True

    # Start with original texts
    current_texts = list(doc_texts)

    for iteration in range(max_iterations):
        # Get actual prompt token count by building the prompt
        total_tokens = prompt_builder_and_counter(current_texts)

        # Check if we're within budget
        if total_tokens <= max_length:
            return current_texts, True

        # Calculate how much we need to remove
        excess_tokens = total_tokens - max_length

        if iteration == 0:
            warnings.warn(
                f"Prompt exceeds max length by {excess_tokens} tokens "
                f"({total_tokens} > {max_length}). Starting iterative truncation..."
            )

        # Count tokens for each document (for per-doc truncation logic)
        doc_token_counts = token_counter.count_tokens_batch(current_texts)

        # Truncate each document by removing tokens_to_remove_per_iter tokens
        new_texts = []
        something_changed = False

        for text, token_count in zip(current_texts, doc_token_counts):
            if token_count <= tokens_to_remove_per_iter:
                # Document is too short to truncate further, keep as is
                new_texts.append(text)
            else:
                # Calculate target token count
                target_tokens = max(10, token_count - tokens_to_remove_per_iter)  # Keep at least 10 tokens

                # Truncate by character approximation, then verify
                # Rough heuristic: 1 token ≈ 4 characters
                char_ratio = len(text) / token_count if token_count > 0 else 4
                target_chars = int(target_tokens * char_ratio)

                # Truncate and add ellipsis
                truncated = text[:target_chars].rsplit(' ', 1)[0]  # Cut at word boundary
                if truncated != text:
                    truncated = truncated + "..."
                    something_changed = True

                new_texts.append(truncated)

        if not something_changed:
            # No more truncation possible
            warnings.warn(
                f"Cannot truncate further. Final prompt has {total_tokens} tokens "
                f"(exceeds max of {max_length} by {excess_tokens} tokens)."
            )
            return current_texts, False

        current_texts = new_texts

    # Reached max iterations without success
    final_tokens = prompt_builder_and_counter(current_texts)
    warnings.warn(
        f"Reached max iterations ({max_iterations}) without fitting within token budget. "
        f"Final prompt has {final_tokens} tokens (max: {max_length})."
    )
    return current_texts, False


def estimate_prompt_overhead(
    query: str,
    num_docs: int,
    token_counter: TokenCounter,
    template_tokens_per_doc: int = 10
) -> int:
    """
    Estimate the token overhead from the prompt template (query, formatting, etc.).

    Args:
        query: Query string
        num_docs: Number of documents
        token_counter: TokenCounter instance
        template_tokens_per_doc: Estimated template tokens per document (numbering, brackets, etc.)

    Returns:
        Estimated token count for template overhead
    """
    # Count query tokens
    query_tokens = token_counter.count_tokens(query)

    # Add template overhead: query appears in template, plus formatting per document
    # Based on RANKPROMPT template:
    # - Query appears twice
    # - Each document has: "[N] " prefix and newline
    # - Instruction text
    instruction_overhead = 100  # Conservative estimate for instruction text

    return query_tokens * 2 + (template_tokens_per_doc * num_docs) + instruction_overhead
