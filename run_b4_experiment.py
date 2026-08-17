#!/usr/bin/env python3
"""Run B4 (Hybrid) experiment on all papers with multiple seeds."""
import sys, os, json, time
sys.path.insert(0, './code')

# API key from env (set at runtime to avoid security redaction)
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', os.environ.get('OPENROUTER_API_KEY', ''))
if not OPENROUTER_API_KEY:
    print('ERROR: OPENROUTER_API_KEY not set', flush=True)
    sys.exit(1)
os.environ['OPENROUTER_API_KEY'] = OPENROUTER_API_KEY
sys.stdout.reconfigure(line_buffering=True)

from hybrid_kg import SpacyBackbone, LLMEnricher, KGExporter
from experiments.run_reproducibility import graph_edit_distance

spacy = SpacyBackbone('en_core_web_lg')
exporter = KGExporter()
api_key = os.environ['OPENROUTER_API_KEY']

PAPERS = [
    ('KGGen', 'corpus/KGGen.txt'),
    ('AriGraph', 'corpus/AriGraph.txt'),
    ('LLM_KGC_Survey', 'corpus/LLM_KGC_Survey.txt'),
    ('LLM_KG_Roadmap', 'corpus/LLM_KG_Roadmap.txt'),
    ('GraphRAG_Survey', 'corpus/GraphRAG_Survey.txt'),
]

def run_hybrid(text, seed=1):
    """Run B4 hybrid pipeline."""
    # Phase A: deterministic backbone
    triples = spacy.extract_triples(text)
    entities = spacy.extract_entities(text)

    # Phase B: LLM enrichment
    enricher = LLMEnricher(api_key)
    enriched_entities = enricher.enrich_nodes(entities)
    if not enriched_entities or enriched_entities == entities:
        enriched_entities = entities

    enriched_triples = enricher.enrich_triples(triples, enriched_entities)
    implicit = enricher.infer_implicit_relations(enriched_triples, enriched_entities)

    graph = exporter.to_json(enriched_triples, enriched_entities, implicit)
    return graph

N_SEEDS = 3
base_dir = './results'

for paper_name, corpus_path in PAPERS:
    print(f'\n{"="*60}')
    print(f'Paper: {paper_name}')
    print(f'{"="*60}')

    text = open(corpus_path).read()
    # Truncate very long texts for B4 (cost control)
    if len(text) > 50000:
        print(f'  Truncating {len(text)} -> 50000 chars (cost control)')
        text = text[:50000]

    graphs = []
    for seed in range(1, N_SEEDS + 1):
        print(f'  Seed {seed}/{N_SEEDS}...')
        t0 = time.time()
        try:
            graph = run_hybrid(text, seed)
            dt = time.time() - t0
            print(f'    {graph["metadata"]["node_count"]} nodes, {graph["metadata"]["edge_count"]} edges ({dt:.1f}s)')
            graphs.append(graph)

            # Save per-seed result
            out_dir = f'{base_dir}/{paper_name}/B4_hybrid/seed_{seed}'
            os.makedirs(out_dir, exist_ok=True)
            with open(f'{out_dir}/graph.json', 'w') as f:
                json.dump(graph, f, indent=2)

        except Exception as e:
            print(f'    FAILED: {e}')
            continue

    if len(graphs) >= 2:
        ged_scores = []
        for i in range(len(graphs)):
            for j in range(i + 1, len(graphs)):
                ged = graph_edit_distance(graphs[i], graphs[j])
                ged_scores.append(ged)

        import numpy as np
        report = {
            'paper': paper_name,
            'method': 'B4_hybrid',
            'n_runs': len(graphs),
            'mean_ged': float(np.mean(ged_scores)) if ged_scores else 0.0,
            'std_ged': float(np.std(ged_scores)) if ged_scores else 0.0,
            'min_ged': float(min(ged_scores)) if ged_scores else 0.0,
            'max_ged': float(max(ged_scores)) if ged_scores else 0.0,
            'node_counts': [g['metadata']['node_count'] for g in graphs],
            'edge_counts': [g['metadata']['edge_count'] for g in graphs],
        }
        with open(f'{base_dir}/{paper_name}/reproducibility_report_B4.json', 'w') as f:
            json.dump(report, f, indent=2)
        print(f'  GED: mean={report["mean_ged"]:.6f}, min={report["min_ged"]:.6f}, max={report["max_ged"]:.6f}')
    else:
        print(f'  Not enough successful runs ({len(graphs)}) for GED computation')

print('\nDone!')
