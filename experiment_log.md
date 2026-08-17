# Experiment Log — Hybrid KG Paper

## Contribution (one sentence)
Un framework ibrido che combina NLP deterministico (spaCy dependency parsing) con arricchimento LLM vincolato (gpt-4o-mini), producendo knowledge graph con relazioni semanticamente più ricche dei metodi puramente deterministici, a un costo nettamente inferiore dei metodi puramente LLM-based.

---

## Esperimenti Eseguiti

### Experiment 1: Riproducibilità (B1 — spaCy deterministico)

**Claim testata**: Il backbone deterministico è perfettamente riproducibile (GED=0.0)

**Setup**: 5 paper arXiv, 3 seed ciascuno, modello `en_core_web_lg`, testo integrale

**Key result**:

| Paper | Nodi | Archi | GED medio |
|-------|------|-------|-----------|
| KGGen | 775 | 263 | 0.000000 |
| AriGraph | 822 | 235 | 0.000000 |
| LLM_KGC_Survey | 746 | 191 | 0.000000 |
| LLM_KG_Roadmap | 2.280 | 468 | 0.000000 |
| GraphRAG_Survey | 5.025 | 1.174 | 0.000000 |
| **Totale** | **9.648** | **2.331** | **0.000000** |

**Risultati**: results/{paper}/B1_spacy/seed_{1..3}/graph.json

### Experiment 2: Arricchimento Ibrido (B4 — spaCy + LLM)

**Claim testata**: L'arricchimento LLM produce KG con relazioni semanticamente più ricche

**Setup**: 5 paper arXiv, 1 seed, 30K chars per paper (limite API), modello `gpt-4o-mini` via OpenRouter, batching 50 entità/triple per chiamata

**Key result**:

| Paper | B1 nodi | B1 archi | B4 nodi | B4 archi | Implicite | Sottotipi | LLM cats uniche |
|-------|---------|----------|---------|----------|-----------|-----------|-----------------|
| KGGen | 775 | 263 | 477 | 204 | 17 | 204 | Concept, Software, Metric... |
| AriGraph | 822 | 235 | 431 | 218 | 12 | 213 | Organization, Location, Product... |
| LLM_KGC_Survey | 746 | 191 | 340 | 180 | 13 | 177 | Concept, Organization, Person... |
| LLM_KG_Roadmap | 2.280 | 468 | 444 | 148 | 0 | 148 | Concept, Event, Facility, Language... |
| GraphRAG_Survey | 5.025 | 1.174 | 536 | 152 | 0 | 152 | Organization, Software, Location... |
| **Totale** | **9.648** | **2.331** | **2.228** | **902** | **42** | **894** | **~20** |

**Nota**: B1 usa testo integrale, B4 usa 30K chars. Il confronto dimensionale non è diretto; il valore è qualitativo.

**Arricchimento qualitativo**:
- B1: solo 2 tipi relazione (direct_action, prepositional) da dependency parsing
- B4: 10+ tipi (causal, temporal, definitional, procedural, attributional, comparative...)
- B4: ~100% archi con sottotipo semantico LLM vs 0% in B1
- B4: 42 relazioni implicite (inferenza multi-hop LLM)
- B4: categorie entità oltre NER spaCy (Software, Concept, Metric, Method...)

**Costo totale**: $0.82 per 5 paper (~$0.16/paper)

*Nota metodologica*: B1 su testo completo, B4 su 30K chars per limiti API. Per confronto equo, B1 su 30K avrebbe metriche simili a B4 in dimensioni ma senza arricchimento semantico.

### Experiment 3: Bug trovati e fixati

1. **max_tokens=4000 insufficiente**: 270 entità producevano JSON troncato. Fix: aumentato a 16000 + batching 50 per velocità.
2. **Timeout API singola**: chiamata con 270 entità in unico prompt timeout a 200s. Fix: batching 50 entità per batch, 6 batch per KGGen, ~20s ciascuno.

---

## Figure
*(da generare)*

## Failed Experiments
- **en_core_web_trf**: Troppo lento su testi lunghi. Passato a en_core_web_lg.
- **B4 multi-seed**: LLM calls troppo lente per 3 seed. Ridotto a 1 seed.

## Open Questions
- Quanto incide la dimensione del testo sulla qualità dell'arricchimento LLM?
- B4 vs B2 (LLM zero-shot) e B3 (LLM schema-guided)?
- Validità umana delle categorie LLM (es. "Software" per KGGen, corretto?)
