"""Pre-configured standard ranking models."""

from typing import Optional, Union
import torch

from pyterrier_generative.modelling.base import GenerativeRanker
from pyterrier_generative._algorithms import Algorithm
from pyterrier_generative.prompts import RANKPROMPT, RANKGPT_SYSTEM_PROMPT, RANKLLM_SYSTEM_PROMPT
from pyterrier_generative.modelling.util import Variants


class _GenerativeRanker(GenerativeRanker, metaclass=Variants):
    """
    Base class for standard pre-configured rankers with variants.

    Subclasses should define a VARIANTS dict mapping variant names to model IDs.
    """

    VARIANTS = None  # To be defined by subclasses

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        prompt: Union[str, callable] = RANKPROMPT,
        system_prompt: str = "",
        algorithm: Algorithm = Algorithm.SLIDING_WINDOW,
        window_size: int = 20,
        stride: int = 10,
        buffer: int = 20,
        cutoff: int = 10,
        k: int = 10,
        max_iters: int = 100,
        max_new_tokens: int = 100,
        backend: Optional[str] = None,
        model_args: Optional[dict] = None,
        generation_args: Optional[dict] = None,
        device: Optional[Union[str, torch.device]] = None,
        api_key: Optional[str] = None,
        verbose: bool = False,
        backend_kwargs: Optional[dict] = None,
    ):
        """
        Initialize StandardRanker with the specified model.

        Args:
            model_id: Model identifier (uses first variant if None)
            prompt: Prompt template (Jinja2 string or callable)
            system_prompt: System prompt for the model
            algorithm: Ranking algorithm to use
            window_size: Size of ranking window
            stride: Stride for sliding window
            buffer: Buffer size for tdpart
            cutoff: Cutoff position for tdpart
            k: Top-k for setwise
            max_iters: Max iterations for tdpart
            max_new_tokens: Max tokens to generate
            backend: Backend to use ('vllm', 'hf', 'openai')
            model_args: Backend-specific model arguments
            generation_args: Backend-specific generation arguments
            device: Device for HF backend
            api_key: API key for OpenAI backend
            verbose: Enable verbose logging
            backend_kwargs: Additional backend-specific kwargs (e.g., batch_size)
        """

        # Use first variant as default if no model_id provided
        if model_id is None:
            if not self.VARIANTS:
                raise ValueError("model_id is required when no VARIANTS are defined")
            model_id = next(iter(self.VARIANTS.values()))

        self.model_id = model_id

        # Auto-detect backend based on model_id if not specified
        if backend is None:
            if model_id.startswith('gpt-'):
                backend = 'openai'
            else:
                backend = 'vllm'

        # Handle API key for OpenAI
        if api_key and backend == 'openai':
            generation_args = generation_args or {}
            generation_args['api_key'] = api_key

        # Prepare backend kwargs
        extra_kwargs = backend_kwargs or {}

        # Select and initialize backend
        if backend == 'vllm':
            from pyterrier_rag.backend import VLLMBackend
            backend_instance = VLLMBackend(
                model_id=model_id,
                model_args=model_args or {},
                generation_args=generation_args,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
                **extra_kwargs,
            )
        elif backend == 'hf':
            from pyterrier_rag.backend import HuggingFaceBackend
            backend_instance = HuggingFaceBackend(
                model_id=model_id,
                model_args=model_args or {},
                generation_args=generation_args,
                max_new_tokens=max_new_tokens,
                device=device,
                verbose=verbose,
                **extra_kwargs,
            )
        elif backend == 'openai':
            from pyterrier_rag.backend import OpenAIBackend
            backend_instance = OpenAIBackend(
                model_id=model_id,
                generation_args=generation_args,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
                **extra_kwargs,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'vllm', 'hf', or 'openai'.")

        # Initialize parent GenerativeRanker
        super().__init__(
            model=backend_instance,
            prompt=prompt,
            system_prompt=system_prompt,
            algorithm=algorithm,
            window_size=window_size,
            stride=stride,
            buffer=buffer,
            cutoff=cutoff,
            k=k,
            max_iters=max_iters,
        )

        self.backend_type = backend

    def __repr__(self):
        # Check if this is a known variant
        inv_variants = {v: k for k, v in self.VARIANTS.items()}
        if self.model_id in inv_variants:
            return f'{self.__class__.__name__}.{inv_variants[self.model_id]}()'

        return (
            f"{self.__class__.__name__}("
            f"model_id={self.model_id!r}, "
            f"backend={self.backend_type!r}, "
            f"algorithm={self.algorithm.value!r}, "
            f"window_size={self.window_size})"
        )


class RankGPT(_GenerativeRanker):
    """
    RankGPT ranker using OpenAI's GPT models.

    Provides easy access to GPT-3.5 and GPT-4 models for ranking.

    Example::

        from pyterrier_generative import RankGPT
        import pyterrier as pt

        # Use default (GPT-3.5-turbo)
        ranker = RankGPT.gpt35()

        # Use GPT-4
        ranker = RankGPT.gpt4(api_key="sk-...")

        # Use with custom parameters
        ranker = RankGPT.gpt35(window_size=10, stride=5)

        # Pass backend-specific kwargs
        ranker = RankGPT.gpt4(
            backend_kwargs={'timeout': 60}
        )

        # In a pipeline
        pipeline = bm25 % 20 >> RankGPT.gpt4()
        results = pipeline.search("What is information retrieval?")

    .. automethod:: gpt35()
    .. automethod:: gpt35_16k()
    .. automethod:: gpt4()
    .. automethod:: gpt4_turbo()
    """

    VARIANTS = {
        'gpt35': 'gpt-3.5-turbo',
        'gpt35_16k': 'gpt-3.5-turbo-16k',
        'gpt4': 'gpt-4',
        'gpt4_turbo': 'gpt-4-turbo-preview',
    }

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        prompt: Union[str, callable] = RANKPROMPT,
        system_prompt: str = RANKGPT_SYSTEM_PROMPT,
        **kwargs
    ):
        """Initialize RankGPT with the RANKGPT system prompt."""
        super().__init__(
            model_id=model_id,
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        )


class RankZephyr(_GenerativeRanker):
    """
    RankZephyr ranker using the Zephyr-7B model.

    Model: ``castorini/rank_zephyr_7b_v1_full``

    Example::

        from pyterrier_generative import RankZephyr

        # Use default variant
        ranker = RankZephyr.v1()

        # With custom backend
        ranker = RankZephyr.v1(backend='hf')

        # Pass backend-specific kwargs like batch_size
        ranker = RankZephyr.v1(
            backend='vllm',
            backend_kwargs={'batch_size': 32}
        )

    .. automethod:: v1()
    """

    VARIANTS = {
        'v1': 'castorini/rank_zephyr_7b_v1_full',
    }

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        prompt: Union[str, callable] = RANKPROMPT,
        system_prompt: str = RANKLLM_SYSTEM_PROMPT,
        **kwargs
    ):
        """Initialize RankZephyr with the RANKLLM system prompt."""
        super().__init__(
            model_id=model_id,
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        )


class RankVicuna(_GenerativeRanker):
    """
    RankVicuna ranker using the Vicuna-7B model.

    Model: ``castorini/rank_vicuna_7b_v1``

    Example::

        from pyterrier_generative import RankVicuna

        # Use default variant
        ranker = RankVicuna.v1()

        # Pass backend-specific kwargs like batch_size
        ranker = RankVicuna.v1(
            backend='vllm',
            backend_kwargs={'batch_size': 32}
        )

    .. automethod:: v1()
    """

    VARIANTS = {
        'v1': 'castorini/rank_vicuna_7b_v1',
    }

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        prompt: Union[str, callable] = RANKPROMPT,
        system_prompt: str = RANKLLM_SYSTEM_PROMPT,
        **kwargs
    ):
        """Initialize RankVicuna with the RANKLLM system prompt."""
        super().__init__(
            model_id=model_id,
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        )


# Alias for backwards compatibility
StandardRanker = _GenerativeRanker

__all__ = ['StandardRanker', 'RankGPT', 'RankZephyr', 'RankVicuna']
