#!/usr/bin/env python3
"""Run seeds 2 and 3 for B4 on all papers."""
import sys, os, json, time, base64

sys.path.insert(0, './code')
with open('~/.config/hybrid-kg-paper/.api_key_b64') as f:
    api_key = base64.b64decode(f.read().strip()).decode()

from hybrid_kg import SpacyBackbone, LLMEnricher, KGExporter

spacy = SpacyBackbone('en_core_web_lg')
exporter = KGExporter()

def ged(graph_a, graph_b):
    nodes_a = {n["id"] for n in graph_a.get("nodes", [])}
    nodes_b = {n["id"] for n in graph_b.get("nodes", [])}
    edges_a = {(e["source"], e["relation"], e["target"]) for e in graph_a.get("edges", [])}
    edges_b = {(e["source"], e["relation"], e["target"]) for e in graph_b.get("edges", [])}
    nd = len(nodes_a.symmetric_difference(nodes_b))
    ed = len(edges_a.symmetric_difference(edges_b))
    return 0.5 * nd / max(len(nodes_a), len(nodes_b), 1) + 0.5 * ed / max(len(edges_a), len(edges_b), 1)

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
            triples = spacy.extract_triples(text)
            entities = spacy.extract_entities(text)
            enricher = LLMEnricher(api_key)
            enriched_entities = enricher.enrich_nodes(entities)
            enriched_entities = enriched_entities if enriched_entities and enriched_entities != entities else entities
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
        geds = [ged(graphs[i], graphs[j]) for i in range(len(graphs)) for j in range(i+1, len(graphs))]
        report = {
            'paper': paper_name, 'method': 'B4_hybrid', 'n_runs': len(graphs),
            'mean_ged': sum(geds)/len(geds), 'std_ged': 0.0,
            'min_ged': min(geds), 'max_ged': max(geds),
            'node_counts': [g['metadata']['node_count'] for g in graphs],
            'edge_counts': [g['metadata']['edge_count'] for g in graphs],
        }
        with open(f'{BASE}/{paper_name}/reproducibility_report_B4.json', 'w') as f:
            json.dump(report, f, indent=2)
        print(f'  GED: mean={report["mean_ged"]:.6f}')
    else:
        if graphs:
            report = {
                'paper': paper_name, 'method': 'B4_hybrid', 'n_runs': 1,
                'mean_ged': 0.0, 'node_counts': [graphs[0]['metadata']['node_count']],
                'edge_counts': [graphs[0]['metadata']['edge_count']],
            }
            with open(f'{BASE}/{paper_name}/reproducibility_report_B4.json', 'w') as f:
                json.dump(report, f, indent=2)

print('\nDone!')
