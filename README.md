# HybridKG: Reproducible Knowledge Graph Construction with LLM Enrichment

A hybrid framework for knowledge graph (KG) extraction from scientific text that combines the **reproducibility of deterministic NLP** with the **semantic richness of large language models**.

> **Companion to [Arachne-Scholar](https://github.com/jack89-ML/Arachne-Scholar).**
> This repository is the research companion of the Arachne engine: it shares the same
> deterministic SVO backbone (spaCy dependency parsing) and quantifies what LLM
> enrichment adds on top of it. Arachne-Scholar is the production engine (local-first,
> OCR, HUD, Gephi export); this repo is the controlled experiment behind the method —
> see the [Arachne README](https://github.com/jack89-ML/Arachne-Scholar) for the
> deterministic-manifesto context.

## Key Results

| Metric | Value |
|--------|-------|
| **Reproducibility** | GED = 0.0 (perfect) across 5 papers × 3 seeds |
| **Semantic relation types** | 10+ (causal, temporal, definitional, procedural...) vs 2 (spaCy only) |
| **Entity categories** | ~20 LLM-enriched (Software, Concept, Method, Metric...) vs 18 NER labels |
| **Implicit relations** | 42 multi-hop inferences |
| **Cost** | $0.82 total for 5 papers (~$0.16/paper) |

## Method

### Phase A: Deterministic Backbone (spaCy)
- SVO triple extraction via dependency parsing
- NER entity extraction
- SHA-256 hash verification for perfect reproducibility

### Phase B: LLM Enrichment (gpt-4o-mini)
- Batched entity enrichment (50/batch): disambiguation + categorization
- Batched triple enrichment (50/batch): semantic relation typing
- Implicit relation inference via multi-hop reasoning

## Repository Structure

```
paper/               # LaTeX source (NeurIPS 2025 template)
  main.tex           # Paper draft
  main.pdf           # Compiled PDF
corpus/              # 5 arXiv papers processed
results/             # Graph JSON output
  <paper>/B1_spacy/  # Deterministic backbone (3 seeds)
  <paper>/B4_hybrid/ # Hybrid pipeline (1 seed)
code/                # Core implementation
experiments/         # Experiment runners
run_b4_all.py        # Batched LLM enrichment pipeline
experiment_log.md    # Full experiment documentation
```

## Running

```bash
# Reproducibility experiment (B1)
python experiments/run_reproducibility.py

# Hybrid enrichment (B4)
python run_b4_all.py

# Aggregate results
python aggregate_results.py
```

**Requirements**: Python 3.11+, spaCy (`en_core_web_lg`), OpenRouter API key

## Citation

```bibtex
@misc{hybridkg2026,
  title     = {HybridKG: Reproducible Knowledge Graph Construction with Deterministic Backbone and LLM Enrichment},
  author    = {Peracchio, Jacopo},
  year      = {2026}
}
```

## License

MIT
