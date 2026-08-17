#!/usr/bin/env python3
"""Check which B4 results have real LLM enrichment vs spaCy-only."""
import json, os, sys

base = './results'

# All spaCy NER labels
ner_labels = {'CARDINAL','DATE','EVENT','FAC','GPE','LAW','LOC','MONEY',
              'NORP','ORDINAL','ORG','PERCENT','PERSON','PRODUCT','TIME',
              'WORK_OF_ART','QUANTITY','LANGUAGE','unknown'}

for p in ['KGGen', 'AriGraph', 'LLM_KGC_Survey', 'LLM_KG_Roadmap', 'GraphRAG_Survey']:
    path = f'{base}/{p}/B4_hybrid/seed_1/graph.json'
    if os.path.exists(path):
        g = json.load(open(path))
        cats = set(n.get('category','?') for n in g.get('nodes',[]))
        enriched = [c for c in sorted(cats) if c not in ner_labels]
        print(f'{p:20s} n={g["metadata"]["node_count"]:>4n} e={g["metadata"]["edge_count"]:>4n}  LLM_cats={enriched}')
    else:
        print(f'{p:20s} MISSING')
