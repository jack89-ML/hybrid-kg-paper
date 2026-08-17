# Hybrid KG: A Reproducible, Semantically-Rich Knowledge Graph Extraction Pipeline

## Outline

### Abstract (5-sentence formula)
1. **What**: Introduciamo HybridKG, un framework ibrido per estrazione di knowledge graph da testo scientifico.
2. **Why**: I KG esistenti sono o puramente deterministici (riproducibili ma semanticamente poveri) o puramente LLM-based (ricchi ma costosi e non riproducibili).
3. **How**: Combiniamo spaCy dependency parsing (backbone deterministico) con arricchimento LLM vincolato (gpt-4o-mini, batching 50, max_tokens 16000).
4. **Evidence**: 5 paper arXiv × 3 seed, GED=0.0 per riproducibilità. Confronto B1 (solo spaCy) vs B4 (ibrido).
5. **Number**: 10+ tipi relazione semantica (vs 2 del backbone), 42 relazioni implicite, costo $0.82/5paper.

### 1. Introduction (1-1.5 pp)
- **Problema**: Knowledge graph construction ha un trade-off tra riproducibilità e ricchezza semantica
- **Approcci esistenti**: metodi deterministici (OpenIE, spaCy) sono riproducibili ma producono solo relazioni grammaticali; metodi LLM (GPT, Claude) producono relazioni ricche ma non riproducibili e costosi
- **Nostro approccio**: ibrido — backbone deterministico per riproducibilità, LLM per arricchimento vincolato
- **Contributions**:
  1. Framework ibrido che separa estrazione (deterministica) da arricchimento (LLM)
  2. Dimostrazione riproducibilità perfetta (GED=0.0) su 5 paper
  3. 10+ tipi relazione semantica a $0.16/paper
  4. Pipeline batched per scalare a grandi testi

### 2. Related Work
- **Deterministic KG Construction**: OpenIE (Angeli et al.), spaCy dependency parsing
- **LLM-based KG Construction**: GPT-based extraction, GraphRAG
- **Hybrid Approaches**: lavori che combinano pattern matching + LLM
- **GED Metric**: Graph Edit Distance per valutare riproducibilità

### 3. Method

#### 3.1 Phase A: Deterministic Backbone
- spaCy `en_core_web_lg`
- Dependency parsing → SVO triples
- NER → entity extraction with categories (PERSON, ORG, GPE, etc.)
- Hash-based reproducibility verification

#### 3.2 Phase B: LLM Enrichment
- **Entity enrichment**: disambiguazione + categorizzazione (Software, Concept, Method, Metric...)
- **Triple enrichment**: assegnazione relation_subtype (causal, temporal, definitional, procedural...)
- **Implicit inference**: multi-hop relation inference
- **Batching**: 50 entità/triple per chiamata per controllare costi e latenza

#### 3.3 Export
- JSON format: nodes (id, label, category, confidence), edges (source, target, relation_type, relation_subtype, confidence, source)

### 4. Experiments

#### 4.1 Setup
- 5 paper arXiv: KGGen, AriGraph, LLM_KGC_Survey, LLM_KG_Roadmap, GraphRAG_Survey
- 3 seed per B1, 1 seed per B4 (limite API)
- Modello LLM: gpt-4o-mini via OpenRouter
- 30K chars per paper per B4 (limite API)

#### 4.2 RQ1: Riproducibilità (B1)
- **Risultato**: GED=0.0 su tutti i paper e seed
- 9.648 nodi, 2.331 archi, idempotenti per hash

#### 4.3 RQ2: Arricchimento Semantico (B4 vs B1)
- **Relazioni**: B1 solo 2 tipi (direct_action, prepositional); B4: 10+ tipi
- **Categorie**: B1 solo NER labels; B4: +20 categorie LLM (Software, Concept, Metric...)
- **Relazioni implicite**: 42 inferenze multi-hop in B4
- **Copertura sottotipi**: ~100% archi B4 con sottotipo vs 0% B1

#### 4.4 RQ3: Costo
- $0.82 totale per 5 paper (~$0.16/paper)
- Breakdown: entity enrichment ~40%, triple enrichment ~50%, implicit ~10%

### 5. Discussion & Limitations
- **Truncation**: B4 usa 30K chars per limiti API; B1 usa testo completo
- **Single seed B4**: riproducibilità B4 non misurata (costo proibitivo per 3 seed)
- **Qualità LLM**: categorie non validate da umani
- **Costo**: $0.16/paper è accettabile per ricerca ma potrebbe scalare

### 6. Conclusion & Future Work
- Framework ibrido funziona: riproducibilità deterministica + ricchezza semantica LLM
- **Futuro**: validazione umana categorie, confronto B2/B3, multi-seed B4 con modelli più economici

---

## Struttura LaTeX (NeurIPS 2025 template)

```
paper/
  main.tex          # Paper completo
  neurips.sty       # Style file
  figures/          # Figure da generare
    fig1_pipeline.pdf
    fig2_comparison.pdf
  references.bib    # Bibliografia
```

## Figure da generare
1. **Pipeline diagram**: Phase A (spaCy) → Phase B (LLM batched) → Export
2. **Comparison chart**: B1 vs B4 — relation types, categories, implicit relations
3. **Cost breakdown**: per-paper cost, phase distribution
