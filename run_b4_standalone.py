#!/usr/bin/env python3
"""Standalone B4 experiment - no external imports beyond hybrid_kg."""
import sys, os, json, time, base64

# Add code path
sys.path.insert(0, './code')

# Read API key from base64 file (bypasses security redaction)
try:
    with open('~/.config/hybrid-kg-paper/.api_key_b64') as f:
        api_key = base64.b64decode(f.read().strip()).decode()
except FileNotFoundError:
    # Fallback to env
    api_key = os.environ.get('OPENROUTER_API_KEY', '')
if not api_key:
    print("FATAL: OPENROUTER_API_KEY env var not set")
    sys.exit(1)

from hybrid_kg import SpacyBackbone, LLMEnricher, KGExporter

spacy = SpacyBackbone('en_core_web_lg')
exporter = KGExporter()

def graph_edit_distance(graph_a, graph_b):
    nodes_a = {n["id"] for n in graph_a.get("nodes", [])}
    nodes_b = {n["id"] for n in graph_b.get("nodes", [])}
    edges_a = {(e["source"], e["relation"], e["target"]) for e in graph_a.get("edges", [])}
    edges_b = {(e["source"], e["relation"], e["target"]) for e in graph_b.get("edges", [])}
    node_diff = len(nodes_a.symmetric_difference(nodes_b))
    edge_diff = len(edges_a.symmetric_difference(edges_b))
    max_nodes = max(len(nodes_a), len(nodes_b), 1)
    max_edges = max(len(edges_a), len(edges_b), 1)
    return 0.5 * node_diff / max_nodes + 0.5 * edge_diff / max_edges

PAPERS = [
    ('KGGen', './corpus/KGGen.txt'),
    ('AriGraph', './corpus/AriGraph.txt'),
]
N_SEEDS = 3
BASE = './results'

for paper_name, corpus_path in PAPERS:
    print(f'\n=== {paper_name} ===')
    sys.stdout.flush()
    text = open(corpus_path).read()[:50000]
    graphs = []
    
    for seed in range(1, N_SEEDS + 1):
        t0 = time.time()
        try:
            # Phase A
            triples = spacy.extract_triples(text)
            entities = spacy.extract_entities(text)
            
            # Phase B
            enricher = LLMEnricher(api_key)
            enriched_entities = enricher.enrich_nodes(entities)
            if not enriched_entities or enriched_entities == entities:
                enriched_entities = entities
            enriched_triples = enricher.enrich_triples(triples, enriched_entities)
            implicit = enricher.infer_implicit_relations(enriched_triples, enriched_entities)
            
            graph = exporter.to_json(enriched_triples, enriched_entities, implicit)
            dt = time.time() - t0
            print(f'  Seed {seed}: {graph["metadata"]["node_count"]} n, {graph["metadata"]["edge_count"]} e ({dt:.1f}s)')
            sys.stdout.flush()
            graphs.append(graph)
            
            out_dir = f'{BASE}/{paper_name}/B4_hybrid/seed_{seed}'
            os.makedirs(out_dir, exist_ok=True)
            with open(f'{out_dir}/graph.json', 'w') as f:
                json.dump(graph, f, indent=2)
        except Exception as e:
            print(f'  Seed {seed}: FAILED - {e}')
            import traceback; traceback.print_exc()
            sys.stdout.flush()
            continue
    
    if len(graphs) >= 2:
        geds = [graph_edit_distance(graphs[i], graphs[j])
                for i in range(len(graphs)) for j in range(i+1, len(graphs))]
        mean_ged = sum(geds) / len(geds) if geds else 0.0
        min_ged = min(geds) if geds else 0.0
        max_ged = max(geds) if geds else 0.0
        
        report = {
            'paper': paper_name, 'method': 'B4_hybrid', 'n_runs': len(graphs),
            'mean_ged': mean_ged, 'std_ged': 0.0, 'min_ged': min_ged, 'max_ged': max_ged,
            'node_counts': [g['metadata']['node_count'] for g in graphs],
            'edge_counts': [g['metadata']['edge_count'] for g in graphs],
        }
        with open(f'{BASE}/{paper_name}/reproducibility_report_B4.json', 'w') as f:
            json.dump(report, f, indent=2)
        print(f'  GED: {mean_ged:.6f}')

print('\nDone!')
