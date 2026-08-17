#!/usr/bin/env python3
"""B4 run: 1 seed per paper + cost tracking."""
import sys, os, json, time, base64

sys.path.insert(0, './code')
with open('~/.config/hybrid-kg-paper/.api_key_b64') as f:
    api_key = base64.b64decode(f.read().strip()).decode()

from hybrid_kg import SpacyBackbone, LLMEnricher, KGExporter
spacy = SpacyBackbone('en_core_web_lg')
exporter = KGExporter()

PAPERS = [
    ('KGGen', './corpus/KGGen.txt'),
    ('AriGraph', './corpus/AriGraph.txt'),
    ('LLM_KGC_Survey', './corpus/LLM_KGC_Survey.txt'),
    ('LLM_KG_Roadmap', './corpus/LLM_KG_Roadmap.txt'),
    ('GraphRAG_Survey', './corpus/GraphRAG_Survey.txt'),
]

BASE = './results'
cost_log = []

for paper_name, corpus_path in PAPERS:
    print(f'\n=== {paper_name} ===')
    sys.stdout.flush()
    text = open(corpus_path).read()[:30000]  # truncate for speed
    
    # Phase A
    t0 = time.time()
    triples = spacy.extract_triples(text)
    entities = spacy.extract_entities(text)
    t_a = time.time() - t0
    print(f'  Phase A: {len(triples)} triples, {len(entities)} entities ({t_a:.1f}s)')
    sys.stdout.flush()
    
    # Phase B
    enricher = LLMEnricher(api_key)
    
    t0 = time.time()
    enriched_entities = enricher.enrich_nodes(entities)
    enriched_entities = enriched_entities if enriched_entities and enriched_entities != entities else entities
    print(f'  Phase B1: {len(enriched_entities)} entities enriched')
    sys.stdout.flush()
    
    enriched_triples = enricher.enrich_triples(triples, enriched_entities)
    print(f'  Phase B2: {len(enriched_triples)} triples enriched')
    sys.stdout.flush()
    
    implicit = enricher.infer_implicit_relations(enriched_triples, enriched_entities)
    t_b = time.time() - t0
    print(f'  Phase B3: {len(implicit)} implicit ({t_b:.1f}s total for Phase B)')
    sys.stdout.flush()
    
    # Export
    graph = exporter.to_json(enriched_triples, enriched_entities, implicit)
    out_dir = f'{BASE}/{paper_name}/B4_hybrid/seed_1'
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/graph.json', 'w') as f:
        json.dump(graph, f, indent=2)
    
    print(f'  Result: {graph["metadata"]["node_count"]} nodes, {graph["metadata"]["edge_count"]} edges')
    
    # Cost estimate
    # gpt-4o-mini: $0.15/1M input, $0.60/1M output
    # Approx 1000 input + 500 output tokens per call
    # 3 calls per paper
    est_cost = 3 * (1000 * 0.15 + 500 * 0.60) / 1_000_000
    print(f'  Est. cost: ${est_cost:.6f}')
    sys.stdout.flush()

print('\n=== DONE ===')
