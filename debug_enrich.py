#!/usr/bin/env python3
"""Debug: what does LLM return for entity enrichment?"""
import sys, os, base64, json
sys.path.insert(0, './code')
with open('~/.config/hybrid-kg-paper/.api_key_b64') as f:
    api_key = base64.b64decode(f.read().strip()).decode()

from hybrid_kg import SpacyBackbone, LLMEnricher

spacy = SpacyBackbone('en_core_web_lg')
text = open('./corpus/KGGen.txt').read()[:5000]
entities = spacy.extract_entities(text)
print(f'spaCy entities ({len(entities)}):')
for e in entities[:3]:
    print(f'  {json.dumps(e)}')

enricher = LLMEnricher(api_key)
print('\nCalling enrich_nodes...')
import time; t0 = time.time()
response = enricher._call_llm(enricher._build_enrich_prompt(entities))
print(f'Time: {time.time()-t0:.1f}s')
print(f'Raw response ({len(response or "")} chars):')
print((response or '')[:500])
print(end='', flush=True)

if response:
    print('\nParsing...')
    parsed = enricher._parse_structured_response(response, entities)
    print(f'Parsed: {len(parsed)} items')
    if parsed:
        print(f'First item: {json.dumps(parsed[0])}')
        print(f'Same as original: {parsed == entities}')
    else:
        print('Empty parsed result')
