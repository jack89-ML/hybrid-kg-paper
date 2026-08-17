#!/usr/bin/env python3
"""Clean up old B4 results and run fresh."""
import sys, os, json, time, base64

sys.path.insert(0, './code')
with open('~/.config/hybrid-kg-paper/.api_key_b64') as f:
    api_key = base64.b64decode(f.read().strip()).decode()

from hybrid_kg import SpacyBackbone, LLMEnricher, KGExporter
spacy = SpacyBackbone('en_core_web_lg')
exporter = KGExporter()

PAPERS = [
    ('KGGen', './corpus/KGGen.txt'),
]
BASE = './results'

paper_name, corpus_path = PAPERS[0]
text = open(corpus_path).read()[:30000]

# Phase A
print('Phase A: spaCy backbone', flush=True)
t0 = time.time()
triples = spacy.extract_triples(text)
entities = spacy.extract_entities(text)
print(f'  {len(triples)} triples, {len(entities)} entities ({time.time()-t0:.1f}s)', flush=True)

# Phase B
enricher = LLMEnricher(api_key, model='openai/gpt-4o-mini')

print('Phase B1: entity enrichment...', flush=True)
t0 = time.time()
enriched_entities = enricher.enrich_nodes(entities)
dt = time.time() - t0
print(f'  Enriched: {len(enriched_entities)} entities ({dt:.1f}s)', flush=True)
# Debug: check enrichment quality
if enriched_entities and enriched_entities != entities:
    for e in enriched_entities[:5]:
        print(f'    {e.get("text","?")} -> cat={e.get("category","?")}, conf={e.get("confidence","?")}', flush=True)
    categories = set(e.get('category','?') for e in enriched_entities)
    print(f'  Categories: {sorted(categories)[:15]}', flush=True)
else:
    print(f'  No enrichment or fallback used')
    if enriched_entities:
        print(f'  Same as original: {enriched_entities == entities}', flush=True)

print('Phase B2: triple enrichment...', flush=True)
t0 = time.time()
enriched_triples = enricher.enrich_triples(triples, enriched_entities)
dt = time.time() - t0
print(f'  {len(enriched_triples)} triples ({dt:.1f}s)', flush=True)
has_subtype = sum(1 for t in enriched_triples if t.get('relation_subtype','') not in ('unknown','?'))
print(f'  With subtypes: {has_subtype}', flush=True)

print('Phase B3: implicit relations...', flush=True)
t0 = time.time()
implicit = enricher.infer_implicit_relations(enriched_triples, enriched_entities)
dt = time.time() - t0
print(f'  {len(implicit)} implicit ({dt:.1f}s)', flush=True)

# Export
graph = exporter.to_json(enriched_triples, enriched_entities, implicit)
out_dir = f'{BASE}/{paper_name}/B4_hybrid/seed_1'
os.makedirs(out_dir, exist_ok=True)
with open(f'{out_dir}/graph.json', 'w') as f:
    json.dump(graph, f, indent=2)

print(f'\nResult: {graph["metadata"]["node_count"]} nodes, {graph["metadata"]["edge_count"]} edges', flush=True)
cats = set()
for n in graph['nodes']:
    cats.add(n.get('category','?'))
print(f'Categories: {sorted(cats)[:15]}', flush=True)
