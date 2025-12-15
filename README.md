# 🤖 PyTerrier Generative

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![PyTerrier](https://img.shields.io/badge/PyTerrier-Compatible-orange)](https://github.com/terrier-org/pyterrier)

Generative **listwise ranking** with [PyTerrier](https://github.com/terrier-org/pyterrier).
PyTerrier Generative brings large language models to information retrieval through unified ranking algorithms and efficient batching.

## 📘 Overview

**PyTerrier Generative** provides:
- **Pre-configured rankers**: RankZephyr, RankVicuna, RankGPT, LiT5.
- **Flexible algorithms**: Sliding window, single window, top-down partitioning, setwise.
- **Efficient batching**: Automatic batching of ranking windows for 6-8x speedup.
- **Customizable prompts**: Jinja2 templates or Python callables.
- **Multiple backends**: vLLM, HuggingFace Transformers, OpenAI.

Workflow:
1) Choose a ranker (e.g., `RankZephyr()`) or create a custom `GenerativeRanker`.
2) Integrate into your PyTerrier pipeline.
3) Rank documents using LLM-based listwise scoring.

## 🚀 Getting Started

### Install from PyPI
```bash
pip install pyterrier-generative
```

### Install from source
```bash
git clone https://github.com/Parry-Parry/pyterrier-generative.git
cd pyterrier-generative
pip install -e .
```

### Quick Example
```python
import pyterrier as pt
from pyterrier_generative import RankZephyr

# Initialize PyTerrier
pt.init()

# Create ranker
ranker = RankZephyr(window_size=20)

# Use in pipeline
pipeline = pt.BatchRetrieve(index) % 100 >> ranker

# Search
results = pipeline.search("machine learning")
```

## 🎯 Pre-configured Rankers

### RankZephyr
```python
from pyterrier_generative import RankZephyr

ranker = RankZephyr(
    algorithm=Algorithm.SLIDING_WINDOW,
    window_size=20,
    stride=10
)
```
**Model**: `castorini/rank_zephyr_7b_v1_full`
**Backend**: vLLM (default)

### RankVicuna
```python
from pyterrier_generative import RankVicuna

ranker = RankVicuna(
    algorithm=Algorithm.SLIDING_WINDOW,
    window_size=20
)
```
**Model**: `castorini/rank_vicuna_7b_v1`
**Backend**: vLLM (default)

### RankGPT
```python
from pyterrier_generative import RankGPT

ranker = RankGPT(
    api_key="your-openai-key",
    algorithm=Algorithm.SLIDING_WINDOW
)
```
**Model**: `gpt-3.5-turbo`
**Backend**: OpenAI

### LiT5
```python
from pyterrier_generative import LiT5

ranker = LiT5(
    model_path='castorini/LiT5-Distill-large',
    window_size=20
)
```
**Architecture**: Fusion-in-Decoder (FiD)
**Backend**: PyTerrier-T5

## ⚙️ Custom Rankers

Build your own ranker with custom prompts and backends:

```python
from pyterrier_generative import GenerativeRanker, Algorithm
from pyterrier_rag.backend.vllm import VLLMBackend

# Create custom backend
backend = VLLMBackend(
    model_id="meta-llama/Llama-3-8B-Instruct",
    max_new_tokens=100
)

# Custom Jinja2 prompt
prompt = """
Rank these passages for: {{ query }}

{% for p in passages %}
[{{ loop.index }}] {{ p }}
{% endfor %}

Ranking:
"""

# Create ranker
ranker = GenerativeRanker(
    model=backend,
    prompt=prompt,
    algorithm=Algorithm.SLIDING_WINDOW,
    window_size=20,
    stride=10
)
```

### Callable Prompts
```python
def custom_prompt(query, passages, num):
    lines = [f"Query: {query}", f"Documents ({num}):"]
    for i, passage in enumerate(passages, 1):
        lines.append(f"[{i}] {passage[:200]}...")
    lines.append("Provide ranking:")
    return "\n".join(lines)

ranker = GenerativeRanker(
    model=backend,
    prompt=custom_prompt,
    algorithm=Algorithm.SLIDING_WINDOW
)
```

## 🔄 Ranking Algorithms

### Sliding Window
Processes documents in overlapping windows, refining rankings iteratively.

```python
ranker = RankZephyr(
    algorithm=Algorithm.SLIDING_WINDOW,
    window_size=20,   # Documents per window
    stride=10         # Window overlap
)
```
**Best for**: Exhaustive Search
**Complexity**: O(n/stride) windows

### Top-Down Partitioning
Recursively partitions documents around pivot elements.

```python
ranker = RankZephyr(
    algorithm=Algorithm.TDPART,
    window_size=20,
    buffer=20,
    cutoff=10,
    max_iters=100
)
```
**Best for**: Efficient Top-k Search
**Complexity**: O(log n) windows (best case)

### Single Window
Ranks top-k documents in one pass.

```python
ranker = RankZephyr(
    algorithm=Algorithm.SINGLE_WINDOW,
    window_size=20    # Top-k to rank
)
```
**Best for**: Small candidate sets, speed-critical applications
**Complexity**: O(1) window

### Setwise
Pairwise comparison using heapsort.

```python
ranker = RankZephyr(
    algorithm=Algorithm.SETWISE,
    k=10              # Top-k to extract
)
```
**Best for**: High-precision top-k ranking
**Complexity**: O(n log k) comparisons

## Backend Selection

**vLLM** (fastest for local models):
```python
ranker = RankZephyr(backend='vllm')  # Default
```

**HuggingFace** (maximum compatibility):
```python
ranker = RankZephyr(backend='hf')
```

**OpenAI** (no local GPU needed):
```python
ranker = RankGPT(api_key="...")
```

## 🔌 Integration with PyTerrier

### Basic Pipeline
```python
import pyterrier as pt
from pyterrier_generative import RankZephyr

# First-stage retrieval
bm25 = pt.BatchRetrieve(index, wmodel="BM25")

# Re-rank top 100 with RankZephyr
ranker = RankZephyr(window_size=20)

pipeline = bm25 % 100 >> ranker

results = pipeline.search("information retrieval")
```

### Multi-stage Pipeline
```python
from pyterrier_dr import ElectraScorer
from pyterrier_generative import RankZephyr

# Stage 1: BM25 retrieval (top 1000)
bm25 = pt.BatchRetrieve(index) % 1000

# Stage 2: Dense re-ranking (top 100)
dense = ElectraScorer()

# Stage 3: Generative re-ranking (final 20)
generative = RankZephyr(window_size=20)

pipeline = bm25 >> dense % 100 >> generative

results = pipeline.search("neural networks")
```

### Experiment
```python
from pyterrier_generative import RankZephyr, RankVicuna, LiT5

rankers = {
    "BM25": bm25,
    "BM25 >> RankZephyr": bm25 % 100 >> RankZephyr(),
    "BM25 >> RankVicuna": bm25 % 100 >> RankVicuna(),
    "BM25 >> LiT5": bm25 % 100 >> LiT5(),
}

results = pt.Experiment(
    rankers,
    dataset.get_topics(),
    dataset.get_qrels(),
    eval_metrics=["map", "ndcg_cut_10", "recip_rank"]
)
```

## 🎨 Advanced Features

### StandardRanker Variants
Access multiple model sizes through the variant system:

```python
from pyterrier_generative import StandardRanker

# All GPT variants
ranker_35 = StandardRanker.RankGPT35(api_key="...")
ranker_4 = StandardRanker.RankGPT4(api_key="...")
ranker_4_turbo = StandardRanker.RankGPT4Turbo(api_key="...")

# Access any variant programmatically
ranker = StandardRanker(
    model_id='gpt-4',
    backend='openai',
    api_key="..."
)
```

### System Prompts (for chat models)
```python
from pyterrier_generative import GenerativeRanker

ranker = GenerativeRanker(
    model=backend,
    system_prompt="You are an expert search engine. Rank documents by relevance.",
    prompt="Query: {{ query }}\n...",
    algorithm=Algorithm.SLIDING_WINDOW
)
```

### Custom Generation Parameters
```python
from pyterrier_rag.backend.vllm import VLLMBackend

backend = VLLMBackend(
    model_id="castorini/rank_zephyr_7b_v1_full",
    max_new_tokens=100,
    generation_args={
        'temperature': 0.0,
        'top_p': 1.0,
        'max_tokens': 100
    }
)
```

## 📊 How It Works

### Listwise Ranking
Traditional pointwise/pairwise rankers score documents independently or in pairs. **Listwise ranking** considers all documents together:

```
Input:  Query + [Doc1, Doc2, ..., DocN]
Model:  "Rank these documents: 3, 1, 5, 2, 4"
Output: Reordered documents by LLM preference
```

### Sliding Window Algorithm
For large document sets, sliding windows enable manageable ranking:

```
Documents: [D1, D2, D3, ..., D100]

Window 1: [D1...D20]  → Rank → [D5, D2, D8, ...]
Window 2: [D11...D30] → Rank → [D15, D12, D18, ...]
Window 3: [D21...D40] → Rank → [D25, D22, D28, ...]
...

Final: Merge rankings → [D5, D15, D25, D2, ...]
```

### Batching
Multiple windows are processed in parallel:

```
Traditional:
  Window1 → Model → Output1
  Window2 → Model → Output2
  Window3 → Model → Output3

Batched (6-8x faster):
  [Window1, Window2, Window3] → Model → [Output1, Output2, Output3]
```

## 🔬 Research

If you use PyTerrier Generative in your research, please cite:

```bibtex
@software{pyterrier_generative,
  title = {PyTerrier Generative: Listwise Ranking with Large Language Models},
  author = {Parry, Andrew},
  year = {2025},
  url = {https://github.com/Parry-Parry/pyterrier-generative}
}
```
Related work:
- RankGPT: [Is ChatGPT Good at Search?](https://arxiv.org/abs/2304.09542)
- RankZephyr: [Rank-without-GPT](https://arxiv.org/abs/2312.03648)
- LiT5: [Listwise Reranker for Multi-Stage IR](https://arxiv.org/abs/2110.09416)

## 👥 Authors

- [Andrew Parry](mailto:a.parry.1@research.gla.ac.uk) - University of Glasgow

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass (`pytest`)
5. Submit a pull request

See [tests/README.md](tests/README.md) for testing guidelines.

## 🧾 Version History

| Version | Date       | Changes                                        |
|--------:|------------|------------------------------------------------|
|     0.1 | 2025-01-14 | Initial release with batching and 4 algorithms |

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.
