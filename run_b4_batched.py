#!/usr/bin/env python3
"""B4 with batched entity enrichment + full pipeline."""
import sys, os, json, time, base64

sys.path.insert(0, './code')
with open('~/.config/hybrid-kg-paper/.api_key_b64') as f:
    api_key = base64.b64decode(f.read().strip()).decode()

from hybrid_kg import SpacyBackbone, KGExporter

# Re-implement batched version inline
import requests
from typing import Optional

class BatchedLLMEnricher:
    def __init__(self, api_key, model="openai/gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.max_retries = 3
    
    def _call_llm(self, prompt, max_tokens=4000):
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are an assistant that responds ONLY with valid JSON."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.1,
                        "max_tokens": max_tokens,
                    },
                    timeout=120,
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                elif resp.status_code == 429:
                    time.sleep(min(2 ** attempt * 5, 30))
                else:
                    return None
            except Exception as e:
                print(f"  [API error] {e}", flush=True)
                time.sleep(2 ** attempt)
        return None

    def enrich_nodes_batched(self, entities, batch_size=50):
        """Enrich entities in batches to avoid slow large responses."""
        enriched = []
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i+batch_size]
            prompt = f"""For each entity in the JSON array, add disambiguation and category.

Return JSON array with format:
[{{"text": "original", "disambiguated": "canonical name", "category": "Software/Concept/Method/Dataset/Metric/Person/Organization/Location/Other", "confidence": 0.0-1.0}}]

Entities:
{json.dumps(batch, indent=2)}

Respond with ONLY valid JSON array."""
            
            t0 = time.time()
            response = self._call_llm(prompt, max_tokens=4000)
            dt = time.time() - t0
            print(f"  Batch {i//batch_size + 1}/{(len(entities)-1)//batch_size + 1}: {dt:.1f}s", flush=True)
            
            if response:
                import re
                try:
                    parsed = json.loads(response)
                except:
                    match = re.search(r'```(?:json)?\s*\n?(\[.*?\])\s*\n?```', response, re.DOTALL)
                    if match:
                        try:
                            parsed = json.loads(match.group(1))
                        except:
                            parsed = batch
                    else:
                        parsed = batch
                enriched.extend(parsed if isinstance(parsed, list) else batch)
            else:
                enriched.extend(batch)
        return enriched
    
    def enrich_triples(self, triples, entities, batch_size=50):
        enriched = []
        for i in range(0, len(triples), batch_size):
            batch = triples[i:i+batch_size]
            prompt = f"""For each SVO triple below, assign relation_subtype and confidence.

Relation types: causal, temporal, compositional, attributional, comparative, procedural, definitional, other
Confidence: 0.0 (unlikely) - 1.0 (certain)

Triples:
{json.dumps(batch, indent=2)}

Return JSON array: [{{"subject", "predicate", "object", "relation_subtype", "confidence", "semantic_role_subject", "semantic_role_object"}}]"""
            
            t0 = time.time()
            response = self._call_llm(prompt, max_tokens=4000)
            dt = time.time() - t0
            print(f"  Triple batch {i//batch_size + 1}: {dt:.1f}s", flush=True)
            
            if response:
                import re
                try:
                    parsed = json.loads(response)
                except:
                    match = re.search(r'```.*?\n?(\[.*?\]).*?```', response, re.DOTALL)
                    parsed = json.loads(match.group(1)) if match else batch
                if isinstance(parsed, list):
                    enriched_map = {(e.get("subject",""), e.get("predicate",""), e.get("object","")): e for e in parsed}
                    for t in batch:
                        key = (t["subject"], t["predicate"], t["object"])
                        if key in enriched_map:
                            t["relation_subtype"] = enriched_map[key].get("relation_subtype", "unknown")
                            t["confidence"] = enriched_map[key].get("confidence", 0.5)
                        else:
                            t["relation_subtype"] = "unknown"
                            t["confidence"] = 0.5
                enriched.extend(batch)
            else:
                enriched.extend(batch)
        return enriched
    
    def infer_implicit(self, triples, entities):
        sample_triples = triples[:100]  # limit for inference
        prompt = f"""Given the following triples (entity-relation-entity), infer implicit multi-hop relations.
Only connect entities that appear in the triples.
Only add STRONGLY implied relations (e.g., A→B and B→C implies A→C).

Relation types: contributes_to, part_of, specializes, contradicts, exemplifies, follows, leads_to

Triples:
{json.dumps(sample_triples, indent=2)}

Return JSON array: [{{"subject", "relation", "object", "relation_type", "confidence"}}]
or empty array [] if none found."""
        
        t0 = time.time()
        response = self._call_llm(prompt, max_tokens=2000)
        dt = time.time() - t0
        print(f"  Implicit: {dt:.1f}s", flush=True)
        
        if response:
            import re
            try:
                parsed = json.loads(response)
            except:
                match = re.search(r'```.*?\n?(\[.*?\]).*?```', response, re.DOTALL)
                parsed = json.loads(match.group(1)) if match else []
            return parsed if isinstance(parsed, list) else []
        return []


spacy = SpacyBackbone('en_core_web_lg')
exporter = KGExporter()
enricher = BatchedLLMEnricher(api_key)

PAPERS = [
    ('KGGen', './corpus/KGGen.txt'),
]
BASE = './results'

paper_name, corpus_path = PAPERS[0]
text = open(corpus_path).read()[:30000]

print('Phase A: spaCy backbone', flush=True)
t0 = time.time()
triples = spacy.extract_triples(text)
entities = spacy.extract_entities(text)
print(f'  {len(triples)} triples, {len(entities)} entities ({time.time()-t0:.1f}s)', flush=True)

print('Phase B1: entity enrichment (batched)', flush=True)
t0 = time.time()
enriched_entities = enricher.enrich_nodes_batched(entities, batch_size=50)
print(f'  Total: {time.time()-t0:.1f}s', flush=True)
if enriched_entities:
    cats = set(e.get('category','?') for e in enriched_entities)
    print(f'  Categories: {sorted(cats)[:15]}', flush=True)
    for e in enriched_entities[:3]:
        print(f'    {e.get("text","?")} -> {e.get("category","?")} [{e.get("confidence","?")}]', flush=True)

print('Phase B2: triple enrichment', flush=True)
t0 = time.time()
enriched_triples = enricher.enrich_triples(triples, enriched_entities)
print(f'  Total: {time.time()-t0:.1f}s', flush=True)
has_s = sum(1 for t in enriched_triples if t.get('relation_subtype','') not in ('unknown','?'))
print(f'  Subtypes: {has_s}/{len(enriched_triples)}', flush=True)

print('Phase B3: implicit relations', flush=True)
t0 = time.time()
implicit = enricher.infer_implicit(enriched_triples, enriched_entities)
print(f'  {len(implicit)} implicit ({time.time()-t0:.1f}s)', flush=True)

graph = exporter.to_json(enriched_triples, enriched_entities, implicit)
out_dir = f'{BASE}/{paper_name}/B4_hybrid/seed_1'
os.makedirs(out_dir, exist_ok=True)
with open(f'{out_dir}/graph.json', 'w') as f:
    json.dump(graph, f, indent=2)
print(f'\nResult: {graph["metadata"]["node_count"]} nodes, {graph["metadata"]["edge_count"]} edges', flush=True)
