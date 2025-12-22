# 🤖 PyTerrier Generative

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![PyTerrier](https://img.shields.io/badge/PyTerrier-Compatible-orange)](https://github.com/terrier-org/pyterrier)

Generative **listwise ranking** with [PyTerrier](https://github.com/terrier-org/pyterrier).
PyTerrier Generative supports the use of generative rankers and list-wise algorithms.

## 📘 Overview

**PyTerrier Generative** provides:
- **Pre-configured rankers**: RankZephyr, RankVicuna, RankGPT, LiT5.
- **Flexible algorithms**: Sliding window, single window, top-down partitioning, setwise.
- **Efficient batching**: Automatic batching of ranking windows.
- **Customizable prompts**: Jinja2 templates or Python callables.
- **Multiple backends**: vLLM, HuggingFace Transformers, OpenAI.

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

pt.init()

# Create ranker
ranker = RankZephyr.v1(window_size=20)

# Use in pipeline
pipeline = pt.BatchRetrieve(index) % 100 >> ranker
results = pipeline.search("machine learning")
```

## 🎯 Pre-configured Rankers

### RankZephyr
```python
from pyterrier_generative import RankZephyr

# Use default variant
ranker = RankZephyr(window_size=20)

# Or specify algorithm and parameters
ranker = RankZephyr(
    algorithm=Algorithm.SLIDING_WINDOW,
    window_size=20,
    stride=10
)
```
**Variants**: `v1` → `castorini/rank_zephyr_7b_v1_full`
**Backend**: vLLM (default), HuggingFace

### RankVicuna
```python
from pyterrier_generative import RankVicuna

ranker = RankVicuna(window_size=20)
```
**Variants**: `v1` → `castorini/rank_vicuna_7b_v1`
**Backend**: vLLM (default), HuggingFace

### RankGPT
```python
from pyterrier_generative import RankGPT

# Use GPT-3.5 (default)
ranker = RankGPT.gpt35(api_key="sk-...")

# Or GPT-4
ranker = RankGPT.gpt4(api_key="sk-...")
```
**Variants**: `gpt35`, `gpt35_16k`, `gpt4`, `gpt4_turbo`
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

Build your own ranker with custom prompts and backends, you can find more details on backends in PyTerrier RAG:

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
ranker = RankGPT.gpt35(api_key="...")
```

## 🔌 PyTerrier Integration

### Basic Re-ranking
```python
import pyterrier as pt
from pyterrier_generative import RankZephyr

bm25 = pt.BatchRetrieve(index)
ranker = RankZephyr(window_size=20)

pipeline = bm25 % 100 >> ranker
results = pipeline.search("information retrieval")
```

### Multi-stage Pipeline
```python
from pyterrier_generative import RankGPT

# Three-stage ranking: BM25 → Dense → Generative
pipeline = (
    bm25 % 1000
    >> dense_ranker % 100
    >> RankGPT.gpt35(api_key="...")
)
```

### Comparative Evaluation
```python
from pyterrier_generative import RankZephyr, RankVicuna

rankers = {
    "BM25": bm25,
    "BM25 >> RankZephyr": bm25 % 100 >> RankZephyr(),
    "BM25 >> RankVicuna": bm25 % 100 >> RankVicuna(),
}

pt.Experiment(rankers, topics, qrels, eval_metrics=["map", "ndcg_cut_10"])
```

## 🎨 Advanced Features

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

## 👥 Authors

- [Andrew Parry](mailto:a.parry.1@research.gla.ac.uk) - University of Glasgow

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass (`pytest`)
5. Submit a pull request

## 🧾 Version History

| Version | Date       | Changes                                        |
|--------:|------------|------------------------------------------------|
|     0.1.3 | 2025-12-22 | Allow Truncation of Documents |
|     0.1.2 | 2025-12-17 | Bug Fixes |
|     0.1.1 | 2025-12-14 | Bug Fixes |
|     0.1 | 2025-12-14 | Initial release with batching and 4 algorithms |

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.
