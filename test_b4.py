#!/usr/bin/env python3
"""Test B4 hybrid pipeline with LLM enrichment (via Arachne venv)."""
import sys, os, json
sys.path.insert(0, './code')

# API key - set directly in script to avoid terminal redaction
os.environ['OPENROUTER_API_KEY'] = os.environ.get('OPENROUTER_API_KEY', '')

from hybrid_kg import SpacyBackbone, LLMEnricher, KGExporter

spacy = SpacyBackbone('en_core_web_lg')
enricher = LLMEnricher(os.environ['OPENROUTER_API_KEY'])
exporter = KGExporter()

text = open('./corpus/KGGen.txt').read()[:5000]
paper_name = 'KGGen_test'
out_dir = f'./results/{paper_name}/B4_hybrid'
os.makedirs(out_dir, exist_ok=True)

# Fase A
triples = spacy.extract_triples(text)
entities = spacy.extract_entities(text)
print(f'Phase A: {len(triples)} triples, {len(entities)} entities')

# Fase B1 - entity enrichment
print('\nPhase B1: Entity enrichment...')
enriched_entities = enricher.enrich_nodes(entities)
if enriched_entities and enriched_entities != entities:
    print(f'  ✓ {len(enriched_entities)} entities enriched')
    for e in enriched_entities[:5]:
        print(f'    {e.get("text","?")} -> {e.get("disambiguated","?")} ({e.get("category","?")}) [{e.get("confidence","?")}]')
else:
    print(f'  ✗ Failed or no enrichment ({len(enriched_entities)} items)')
    enriched_entities = entities

# Fase B2 - triple enrichment
print('\nPhase B2: Triple enrichment...')
enriched_triples = enricher.enrich_triples(triples, enriched_entities)
has_subtypes = sum(1 for t in enriched_triples if t.get('relation_subtype') and t.get('relation_subtype') != 'unknown')
print(f'  {len(enriched_triples)} triples, {has_subtypes} with subtypes')
if enriched_triples:
    t = enriched_triples[0]
    print(f'  Sample: ({t["subject"]}, {t["predicate"]}, {t["object"]}) -> {t.get("relation_subtype","?")} [{t.get("confidence","?")}]')

# Fase B3 - implicit relations
print('\nPhase B3: Implicit relations...')
implicit = enricher.infer_implicit_relations(enriched_triples, enriched_entities)
print(f'  {len(implicit)} implicit relations')
for imp in implicit[:3]:
    print(f'  ({imp.get("subject","?")}, {imp.get("relation","?")}, {imp.get("object","?")})')

# Export
graph = exporter.to_json(enriched_triples, enriched_entities, implicit)
with open(f'{out_dir}/graph.json', 'w') as f:
    json.dump(graph, f, indent=2, ensure_ascii=False)
print(f'\n✓ KG exported: {graph["metadata"]["node_count"]} nodes, {graph["metadata"]["edge_count"]} edges')
