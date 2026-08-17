#!/usr/bin/env python3
"""Aggregate B1 vs B4 results for all 5 papers."""
import json, os, sys
from collections import Counter

BASE = './results'
PAPERS = ['KGGen', 'AriGraph', 'LLM_KGC_Survey', 'LLM_KG_Roadmap', 'GraphRAG_Survey']

ner_labels = {'CARDINAL','DATE','EVENT','FAC','GPE','LAW','LOC','MONEY',
              'NORP','ORDINAL','ORG','PERCENT','PERSON','PRODUCT','TIME',
              'WORK_OF_ART','QUANTITY','LANGUAGE','unknown'}

print(f"{'Paper':20s} {'B1_nodes':>8s} {'B1_edges':>8s} {'B4_nodes':>8s} {'B4_edges':>8s} {'LLM_cats':>20s} {'Implicit':>8s} {'Subtypes':>8s}")
print("-"*90)

total_b1_n = total_b1_e = total_b4_n = total_b4_e = 0
for p in PAPERS:
    b1 = json.load(open(f'{BASE}/{p}/B1_spacy/seed_1/graph.json'))
    b4 = json.load(open(f'{BASE}/{p}/B4_hybrid/seed_1/graph.json'))
    
    b1_n, b1_e = b1['metadata']['node_count'], b1['metadata']['edge_count']
    b4_n, b4_e = b4['metadata']['node_count'], b4['metadata']['edge_count']
    
    cats = set(n.get('category','?') for n in b4.get('nodes',[]))
    llm_cats = cats - ner_labels
    
    edges = b4.get('edges',[])
    implicit = sum(1 for e in edges if e.get('source') == 'llm_inferred')
    subtypes = sum(1 for e in edges if e.get('relation_subtype','') not in ('unknown','?'))
    
    print(f"{p:20s} {b1_n:>8n} {b1_e:>8n} {b4_n:>8n} {b4_e:>8n} {str(sorted(llm_cats)[:5]):>20s} {implicit:>8n} {subtypes:>8n}")
    
    total_b1_n += b1_n; total_b1_e += b1_e
    total_b4_n += b4_n; total_b4_e += b4_e

print("-"*90)
print(f"{'TOTAL':20s} {total_b1_n:>8n} {total_b1_e:>8n} {total_b4_n:>8n} {total_b4_e:>8n}")
