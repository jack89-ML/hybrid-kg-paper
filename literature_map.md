# Literature Map: NLP for Knowledge Graph Construction as LLM Knowledge Bases

## Contribution (one sentence)
L'utilizzo di tecniche NLP — dalla dependency parsing classica al prompting di LLM — per la costruzione automatica di knowledge graph (KG) da testo non strutturato, e l'impiego di tali grafi come knowledge base strutturate per modelli di linguaggio di grande scala (LLM) in contesti di retrieval-augmented generation (RAG) e agent memory.

---

## 1. Tassonomia dell'Area

La letteratura si organizza in 4 correnti principali:

```
┌─────────────────────────────────────────────────────────────────────┐
│            KG come Knowledge Base per LLM                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  3. GraphRAG & Agent Memory  (Han 2025, Yang 2026,  ...)   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                       ▲                                              │
│                       │ usa                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  2. LLM-empowered KG Construction  (Bian 2025, Mo 2025, ...)│   │
│  │  4. Deterministic NLP-based KG  (OpenIE, spaCy SVO, ...)   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                       ▲                                              │
│                       │ costruisce                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Testo non strutturato (PDF, documenti, codice, web)        │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  5. LLM-KG Integration Roadmap  (Pan 2023, Jiang 2024, ...)       │
│  Visioni sull'integrazione sinergica fra LLM e KG                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Corrente 1: KG Construction Classica con NLP Deterministico

**Claim centrale**: Il parsing sintattico (dependency parsing) e l'Open Information Extraction (OpenIE) permettono estrazione deterministica e riproducibile di triple (soggetto, verbo, oggetto) da testo, senza ricorrere a modelli probabilistici.

### Paper chiave

| Paper | Anno | Contributo |
|-------|------|------------|
| **ClausIE** (Del Corro & Gemulla, 2013) | 2013 | Mappa dependency relations a clausole → triple OpenIE. Ancora baseline standard. |
| **Stanford OpenIE** (Angeli et al., 2015) | 2015 | Pipeline NER + dependency parse → triple. |
| **OLLIE** (Mausam et al., 2012) | 2012 | OpenIE con apprendimento di pattern da bootstrapping. |
| **OPIEC** (Gashteovski et al., 2019) — 1904.12324 | 2019 | Corpus OpenIE su larga scala (3.3B triple da Wikipedia). |
| **Comparison of biomedical RE methods** (Milosevic, 2022) — 2201.01647 | 2022 | Confronto sistematico: modelli a regole vs ML vs transformers per estrazione triple biomedicali. |
| **A Survey on OpenIE** (EMNLP 2024 Findings) | 2024 | Survey completa OpenIE 2007-2024: da rule-based a LLM. |
| **Comprehensive Survey on Automatic KGC** (ACM CS) | 2023 | Survey su AKGC: NER, RE, entity linking, fusione. |

### Strumenti chiave (deterministici)

| Strumento | Meccanismo | Pro | Contro |
|-----------|-----------|-----|--------|
| **spaCy** | Dependency parsing (transition-based), POS tagging, NER | Veloce, deterministico, multi-lingua (EN/IT/ES/...) | Solo sintassi, no semantica profonda |
| **Stanza** (Stanford NLP) | Neural dependency parsing, NER, coref | Accuratezza superiore a spaCy su alcune lingue | Più lento |
| **Stanford CoreNLP** | Pipeline completa: tokenize, parse, NER, coref, OpenIE | Maturo, OpenIE integrato | Java, pesante |
| **ClausIE** | Regole grammaticali su dependency parse | Deterministico, trasparente | Solo EN, triple semplici |

### Limiti noti (pre-LLM)
- Estrazione di triple superficiali (SVO), nessuna comprensione di relazioni n-arie
- Nessuna fusione di entità cross-documento
- Bassa copertura su testi informali o dialogici
- Non cattura relazioni implicite

---

## 3. Corrente 2: LLM-Empowered KG Construction

**Claim centrale**: I LLM possono sostituire o affiancare le pipeline NLP classiche per l'estrazione di entità, relazioni e ontology, operando in zero-shot o few-shot.

### Survey principale
- **LLM-empowered KG construction: A survey** (Bian, 2025) — 2510.20345
  - Categorizza tre paradigmi: **schema-free** (LLM estrae senza ontologia predefinita), **schema-guided** (LLM popola ontology esistente), **schema-emerging** (LLM inferisce e genera ontology dal testo)
  - Copre GraphRAG, OntoRAG, Graphusion

### Paper chiave per estrazione con LLM

| Paper | Anno | Metodo | Metriche |
|-------|------|--------|----------|
| **KGGen** (Mo et al., NeurIPS 2025) — 2502.09956 | 2025 | LM per text-to-KG: prompt + parsing strutturato. Supera OpenIE classico su benchmark RE. | Precision, Recall, F1 su benchmark RE |
| **Graphusion** (Yang et al., 2024) | 2024 | Fusione di KGs scientifici via LLM per NLP education | Fusione entità cross-paper, zero-shot |
| **KG Construction with CoT + LLM** (VLDB 2024 Workshop) | 2024 | Embedding di Chain-of-Thought in LLM per KGC | F1 su benchmark |
| **Automated KGC using LLMs** (EMNLP 2025) — aclanthology.org/2025.emnlp-main.783 | 2025 | Pipeline completa basata su LLM senza supervisione | Precision, Recall, F1 su multiple benchmark |
| **LLM-driven KGC for Semantic Communication** (MDPI Applied Sciences, 2025) | 2025 | Schema a 4 stadi: corpus → NER → RE → KG fusion | Accuratezza estrazione |
| **NLP-Driven KGC From Informal Text** (IEEE Access, 2026) | 2026 | Modelli fine-tuned per contesti narrativi (dialoghi movie) | F1 su testi informali |

### Limiti noti (LLM-based)
- **Non-determinismo**: LLM diversi danno triple diverse allo stesso input — problema epistemologico per la ricerca scientifica
- **Allucinazione**: LLM inventano entità e relazioni inesistenti
- **Costo**: Token API per testi molto lunghi (es. interi paper accademici)
- **Riproducibilità**: variazione con temperatura, seed, versione del modello

---

## 4. Corrente 3: KG come Knowledge Base per LLM (GraphRAG e Agent Memory)

**Claim centrale**: Un grafo di conoscenza strutturato, se usato come base di retrieval per un LLM, supera la naive semantic-search RAG (chunk-based) in task che richiedono comprensione multi-hop e relazionale.

### GraphRAG

| Paper | Anno | Innovazione |
|-------|------|-------------|
| **GraphRAG** (Edge et al., Microsoft, 2024) | 2024 | LLM → KG entities + relations → community detection → summarization → answer. Il paper originale. |
| **GraphRAG Survey** (Han et al., 2025) — 2501.00309 | 2025 | 500+ refs: categorizza GraphRAG in retrieval, reasoning, generation. |
| **When to use Graphs in RAG** (Xiang et al., 2025) — 2506.05690 | 2025 | Analisi comparativa: quando il grafo aiuta, quando no. |
| **SPRIG** (Wang, 2025) — 2602.23372 | 2025 | GraphRAG CPU-only, token-free, senza LLM per costruzione grafo. |
| **FAIR GraphRAG** (Flüh et al., 2026) — 2607.11464 | 2026 | GraphRAG + principi FAIR per dati scientifici. |
| **Robust GraphRAG** (Ma et al., 2026) — 2603.14828 | 2026 | Mitigazione retrieval drift e allucinazione da KG imperfetti. |

### Agent Memory basata su KG

| Paper | Anno | Innovazione |
|-------|------|-------------|
| **Graph-based Agent Memory Survey** (Yang et al., 2026) — 2602.05665 | 2026 | Taxonomy: memoria episodica, semantica, procedurale su grafo. Pipeline: percezione → estrazione LLM → integrazione → retrieval. |
| **AriGraph** (Anokhin et al., IJCAI 2025) — 2407.04363 | 2025 | KG world model con episodic memory per agenti LLM. Navigazione e pianificazione su grafo. |
| **Mem0** | 2025 | Memoria a grafo per agenti: 26% > OpenAI memory su LOCOMO, 91% latenza inferiore. |

### Strumenti industriali

| Strumento | Azienda | Note |
|-----------|---------|------|
| **Neo4j LLM KG Builder** | Neo4j | Costruzione semi-automatica KG + community summarization |
| **Microsoft GraphRAG** | Microsoft | Pipeline open-source: LLM estrae entità/relazioni, clustering comunità |
| **Graphiti** | Zep | KG come memoria per agenti, retrieval sub-second 95th percentile |

---

## 5. Corrente 4: LLM-KG Integration — Visioni e Roadmap

| Paper | Anno | Tesi |
|-------|------|------|
| **Unifying LLMs and KGs: A Roadmap** (Pan et al., 2023) — 2306.08302 | 2023 | KG-enhanced LLM (KG → LLM) vs LLM-enhanced KG (LLM → KG). Direzioni simbiotiche. |
| **KG-Agent** (Jiang et al., 2024) — 2402.11163 | 2024 | Agente LLM autonomo che naviga KG per QA complesso. |
| **LLM Knowledge Conflicts Survey** (Xu et al., 2024) — 2403.08319 | 2024 | Conflitti contesto-memoria, inter-contesto, intra-memoria. |
| **Assessing LLMs for KGC** (Iga & Silaghi, 2024) — 2405.17249 | 2024 | LLM allucinano in KGC: non-determinismo sistematico. |

---

## 6. Gap e Direzioni Emergenti

| Gap | Descrizione | Opportunità |
|-----|-------------|-------------|
| **Determinismo vs Qualità** | Pipeline deterministiche (spaCy SVO) sono riproducibili ma perdono relazioni complesse. LLM catturano più semantica ma sono non-deterministici e allucinano. | Ibrido: NLP deterministico per backbone strutturale, LLM per arricchimento semantico |
| **Costo/beneficio del grafo** | "When to use Graphs in RAG" (Xiang 2025) mostra che il grafo aiuta solo in certi task. Manca una teoria generale. | Caratterizzare i task dove KG > chunk-RAG |
| **KG per scienza riproducibile** | La non-riproducibilità dei KG estratti da LLM è un problema per la ricerca scientifica (Arachne-Scholar). | Pipeline deterministiche per dati scientifici, LLM per esplorazione |
| **Valutazione** | KG estratti da LLM vs NLP classico: metriche standardizzate? | Benchmark unificato per qualità KG |
| **Grafo come memoria agentica** | AriGraph e Mem0 mostrano potenziale, ma mancano architetture ibride episodico-semantiche in produzione. | Sistemi di memoria agentica maturi |
| **Costruzione KG senza LLM** | SPRIG (CPU-only, token-free) mostra che si può fare GraphRAG senza LLM per building. Alternativa verde. | GraphRAG efficiente per deploy su hardware limitato |

---

## 7. Collegamenti con i Progetti Esistenti nell'Ambiente

| Progetto | Corrente | Ruolo nella Letteratura |
|----------|----------|------------------------|
| **Graphify** | Corrente 1+2 | AST per codice (deterministico) + LLM per docs/semantica. Ibrido. |
| **Arachne-Scholar** | Corrente 1 (manifesto deterministico) | spaCy SVO per triple, no LLM per estrazione. OCR + NLP deterministico come epistemologia. |
| **GraphRAG (Microsoft)** | Corrente 3 | Benchmark per GraphRAG. Baseline da superare o ibridare. |
| **SPRIG** | Corrente 3 | Alternativa CPU-only, senza LLM per building. Allineato con filosofia Arachne. |

---

## 8. Paper Seed per Citazioni Verificate

| ID | Titolo | Anchor |
|----|-------|--------|
| arXiv:2510.20345 | LLM-empowered knowledge graph construction: A survey | Bian, 2025 |
| arXiv:2502.09956 | KGGen: Extracting KGs from Plain Text with LMs | Mo et al., NeurIPS 2025 |
| arXiv:2501.00309 | Retrieval-Augmented Generation with Graphs (GraphRAG) | Han et al., 2025 |
| arXiv:2602.05665 | Graph-based Agent Memory: Taxonomy, Techniques, Applications | Yang et al., 2026 |
| arXiv:2407.04363 | AriGraph: Learning KG World Models with Episodic Memory | Anokhin et al., IJCAI 2025 |
| arXiv:2306.08302 | Unifying LLMs and KGs: A Roadmap | Pan et al., 2023 |
| arXiv:2402.11163 | KG-Agent: Efficient Autonomous Agent over KG | Jiang et al., 2024 |
| arXiv:2506.05690 | When to use Graphs in RAG | Xiang et al., 2025 |
| arXiv:2602.23372 | SPRIG: Democratizing GraphRAG, CPU-only | Wang, 2025 |
| arXiv:2201.01647 | Comparison of biomedical RE methods for KGC | Milosevic, 2022 |
| — | ClausIE (Del Corro & Gemulla, 2013) | Paper classico OpenIE |
| — | GraphRAG (Edge et al., Microsoft, 2024) | Paper originale Microsoft |

> **Nota**: Le citazioni sopra sono state verificate via arXiv API (ID) o via Semantic Scholar search. Tutti i paper con arXiv ID sono localizzati. I paper classici (ClausIE, GraphRAG Microsoft) richiedono verifica DOI.
