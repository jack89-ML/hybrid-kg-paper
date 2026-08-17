# Experiment Design

## One-Sentence Contribution
Hybrid framework: deterministic NLP (spaCy SVO) + constrained LLM enrichment for reproducible,
cost-effective KG construction as knowledge bases for LLMs.

---

## Experiment 1: Reproducibility

**Claim testata**: C1 — L'ibrido è più riproducibile del LLM-only

**Setup**:
- 5 paper accademici (arXiv: 2502.09956, 2501.00309, 2407.04363, 2306.08302, 2510.20345)
- Ogni paper processato N=5 volte con ogni metodo
- Metodi: B1 (spaCy-only), B2 (LLM zero-shot), B3 (LLM schema-guided), B4 (ibrido)

**Metrica**: Graph Edit Distance normalizzata tra coppie di run
```
GED_norm(A, B) = edit_distance(A, B) / max(|A|, |B|)
```
dove edit_distance è il minimo numero di operazioni (add/remove node, add/remove edge, change label)
per trasformare il grafo A in B.

**Evidenza attesa**:
- B1 (spaCy): GED_norm = 0.0 (deterministico puro)
- B4 (ibrido): GED_norm < 0.05 (leggera variazione dalla disambiguazione LLM)
- B2/B3 (LLM-only): GED_norm > 0.2 (variazione significativa)

**Script**: `experiments/run_reproducibility.py`

---

## Experiment 2: Relation Coverage

**Claim testata**: C2 — L'ibrido cattura relazioni più ricche del deterministico-only

**Setup**:
- Stessi 5 paper di Exp 1
- Analisi statistica dei tipi di relazione estratti

**Metriche**:
- **Relazioni distinte**: conteggio tipi di relazione unici
- **Diversità di Shannon**: H = -Σ p_i * log(p_i) sui tipi di relazione
- **Relazioni n-arie**: triple con >2 argomenti (es. "X published Y in Z at W")
- **Relazioni implicite**: catturate solo da LLM (Fase B.3)

**Evidenza attesa**:
- B1 (spaCy): 3-5 tipi di relazione (direct_action, attribute, prepositional...)
- B4 (ibrido): 8-15 tipi (aggiunge causal, temporal, compositional, procedural...)
- LLM-only: molti tipi ma rumorosi (hallucination)

---

## Experiment 3: Cost Analysis

**Claim testata**: C3 — L'ibrido è più economico del LLM-only

**Setup**:
- Stessi 5 paper
- Tracciamento input/output token per ogni chiamata LLM

**Metriche**:
- **USD per triple**: costo API / numero triple valide
- **USD per paper**: costo totale per processare un paper
- **Token efficiency**: triple valide per 1000 token LLM

**Modello LLM**: gpt-4o-mini via OpenRouter
- Input: $0.15/1M token
- Output: $0.60/1M token

**Evidenza attesa**:
- B4 (ibrido): 5-10x più economico di B2/B3
  - LLM processa solo nodi/archi esistenti (non testo grezzo)
  - Prompt più corti e focalizzati

---

## Experiment 4: Hallucination Rate

**Claim testata**: C4 — LLM vincolato su scaffold deterministico allucina meno

**Setup**:
- Campione di 200 triple da ogni metodo
- Valutazione umana (dual annotator + adjudicator) su:
  - **Veridicità**: la tripla è vera rispetto al testo originale?
  - **Rilevanza**: la tripla è informativa o banale?
  - **Copertura**: il set di triple copre i concetti chiave del paper?

**Metriche**:
- **Hallucination Rate**: triple_false / triple_totali
- **Cohen's κ**: inter-annotator agreement
- **F1**: rispetto a gold standard

**Evidenza attesa**:
- B2 (LLM zero-shot): hallucination rate 15-30%
- B3 (LLM schema-guided): 8-15%
- B4 (ibrido): 3-8%
- B1 (spaCy): 0% (deterministico, ma copertura bassa)

---

## Corpus

5 paper arXiv per esperimenti (scaricati via API):

| # | Paper | arXiv ID | Dominio | Pagine |
|---|-------|----------|---------|--------|
| 1 | KGGen | 2502.09956 | NLP/KG | ~10 |
| 2 | GraphRAG Survey | 2501.00309 | IR/KG | ~20 |
| 3 | AriGraph | 2407.04363 | AI/Agents | ~10 |
| 4 | LLM-KG Unifying | 2306.08302 | Survey | ~15 |
| 5 | LLM-empowered KGC Survey | 2510.20345 | Survey | ~15 |

---

## Note per Phase 3 (Execution)

- `experiments/run_reproducibility.py`: lancia tutti i 4 metodi N=5 volte
- `experiments/run_baselines.py`: varianti LLM-only per confronto
- `experiments/download_corpus.py`: scarica i 5 paper da arXiv
- `results/`: output per ogni paper/metodo/seed
- Cost tracker: ogni chiamata LLM loggata in `results/cost_log.jsonl`
