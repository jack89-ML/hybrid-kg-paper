#!/usr/bin/env python3
"""B4 hybrid: all 5 papers with batched enrichment, incremental save, cost tracking."""
import sys, os, json, time, base64, re

sys.path.insert(0, './code')
with open('~/.config/hybrid-kg-paper/.api_key_b64') as f:
    api_key = base64.b64decode(f.read().strip()).decode()

from hybrid_kg import SpacyBackbone, KGExporter
import requests

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

    def _parse_json(self, response, fallback):
        """Try to parse JSON from response, with code block extraction fallback."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r'```(?:json)?\s*\n?(\[.*?\])\s*\n?```', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
        return fallback

    def enrich_nodes_batched(self, entities, batch_size=50):
        enriched = []
        n_batches = (len(entities) - 1) // batch_size + 1
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
            print(f"  Batch {i//batch_size + 1}/{n_batches}: {dt:.1f}s", flush=True)

            if response:
                parsed = self._parse_json(response, batch)
                enriched.extend(parsed if isinstance(parsed, list) else batch)
            else:
                enriched.extend(batch)
        return enriched

    def enrich_triples(self, triples, entities, batch_size=50):
        enriched = []
        n_batches = (len(triples) - 1) // batch_size + 1
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
            print(f"  Triple batch {i//batch_size + 1}/{n_batches}: {dt:.1f}s", flush=True)

            if response:
                parsed = self._parse_json(response, None)
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
        sample = triples[:100]
        prompt = f"""Given the following triples (entity-relation-entity), infer implicit multi-hop relations.
Only connect entities that appear in the triples.
Only add STRONGLY implied relations (e.g., A→B and B→C implies A→C).

Relation types: contributes_to, part_of, specializes, contradicts, exemplifies, follows, leads_to

Triples:
{json.dumps(sample, indent=2)}

Return JSON array: [{{"subject", "relation", "object", "relation_type", "confidence"}}]
or empty array [] if none found."""

        response = self._call_llm(prompt, max_tokens=2000)
        if response:
            parsed = self._parse_json(response, [])
            return parsed if isinstance(parsed, list) else []
        return []


# ── Setup ─────────────────────────────────────────────────
spacy = SpacyBackbone('en_core_web_lg')
exporter = KGExporter()
enricher = BatchedLLMEnricher(api_key)

PAPERS = [
    ('KGGen',             './corpus/KGGen.txt'),
    ('AriGraph',          './corpus/AriGraph.txt'),
    ('LLM_KGC_Survey',    './corpus/LLM_KGC_Survey.txt'),
    ('LLM_KG_Roadmap',    './corpus/LLM_KG_Roadmap.txt'),
    ('GraphRAG_Survey',   './corpus/GraphRAG_Survey.txt'),
]

BASE = './results'
MAX_CHARS = 30000
COST_LOG = os.path.join(BASE, 'cost_log_b4.jsonl')
total_est_cost = 0.0

def has_llm_enrichment(graph_path):
    """Check if graph has actual LLM enrichment (not just spaCy NER categories)."""
    if not os.path.exists(graph_path):
        return False
    try:
        g = json.load(open(graph_path))
        cats = set(n.get('category', '?') for n in g.get('nodes', []))
        # LLM-enriched graphs have non-NER categories like Software, Concept, Method, etc.
        # All standard spaCy NER labels
        ner_cats = {'CARDINAL','DATE','EVENT','FAC','GPE','LAW','LOC','MONEY',
                     'NORP','ORDINAL','ORG','PERCENT','PERSON','PRODUCT','TIME',
                     'WORK_OF_ART','QUANTITY','LANGUAGE','unknown'}
        enriched_cats = cats - ner_cats
        return len(enriched_cats) > 0
    except:
        return False

def log_cost(paper, phase, dt, est_cost):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "paper": paper,
        "phase": phase,
        "duration_s": round(dt, 1),
        "est_cost_usd": round(est_cost, 6),
    }
    with open(COST_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── Run ───────────────────────────────────────────────────
print(f"B4 Hybrid — All 5 Papers (max {MAX_CHARS} chars each)")
print(f"Model: gpt-4o-mini via OpenRouter\n", flush=True)

for paper_name, corpus_path in PAPERS:
    out_dir = f'{BASE}/{paper_name}/B4_hybrid/seed_1'
    out_path = f'{out_dir}/graph.json'

    # Skip if already done with proper LLM enrichment
    if has_llm_enrichment(out_path):
        g = json.load(open(out_path))
        print(f"✓ {paper_name}: already done ({g['metadata']['node_count']}n, {g['metadata']['edge_count']}e) — skipping", flush=True)
        continue

    print(f"\n─── {paper_name} ───", flush=True)

    text = open(corpus_path).read()[:MAX_CHARS]

    # Phase A: spaCy backbone
    print("  Phase A: spaCy backbone...", end=" ", flush=True)
    t0 = time.time()
    triples = spacy.extract_triples(text)
    entities = spacy.extract_entities(text)
    t_a = time.time() - t0
    print(f"{len(triples)} triples, {len(entities)} entities ({t_a:.1f}s)", flush=True)

    # Phase B1: Entity enrichment (batched)
    print("  Phase B1: entity enrichment...", flush=True)
    t0 = time.time()
    enriched_entities = enricher.enrich_nodes_batched(entities, batch_size=50)
    t_b1 = time.time() - t0
    if enriched_entities:
        cats = set(e.get('category','?') for e in enriched_entities)
        ner_cats = {'CARDINAL','DATE','EVENT','FAC','GPE','LAW','LOC','MONEY',
                     'NORP','ORDINAL','ORG','PERCENT','PERSON','PRODUCT','TIME',
                     'WORK_OF_ART','QUANTITY','LANGUAGE','unknown'}
        llm_cats = cats - ner_cats
        print(f"  → {len(enriched_entities)} entities ({t_b1:.1f}s), LLM cats: {sorted(llm_cats)[:10]}", flush=True)
    log_cost(paper_name, "B1_enrich_nodes", t_b1, t_b1 * 0.0005)  # rough estimate

    # Phase B2: Triple enrichment
    print("  Phase B2: triple enrichment...", flush=True)
    t0 = time.time()
    enriched_triples = enricher.enrich_triples(triples, enriched_entities)
    t_b2 = time.time() - t0
    has_s = sum(1 for t in enriched_triples if t.get('relation_subtype','') not in ('unknown','?'))
    print(f"  → {len(enriched_triples)} triples ({t_b2:.1f}s), {has_s} with subtypes", flush=True)
    log_cost(paper_name, "B2_enrich_triples", t_b2, t_b2 * 0.0005)

    # Phase B3: Implicit relations
    print("  Phase B3: implicit relations...", end=" ", flush=True)
    t0 = time.time()
    implicit = enricher.infer_implicit(enriched_triples, enriched_entities)
    t_b3 = time.time() - t0
    print(f"{len(implicit)} implicit ({t_b3:.1f}s)", flush=True)
    log_cost(paper_name, "B3_infer_implicit", t_b3, t_b3 * 0.0005)

    # Export
    graph = exporter.to_json(enriched_triples, enriched_entities, implicit)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(graph, f, indent=2)

    # Richness metrics
    all_cats = set(n.get('category','?') for n in graph['nodes'])
    enriched_cats = all_cats - ner_cats
    print(f"  ✓ Result: {graph['metadata']['node_count']}n, {graph['metadata']['edge_count']}e", flush=True)
    print(f"    Categories: {len(all_cats)} total, {len(enriched_cats)} LLM-enriched", flush=True)
    print(f"    LLM cats: {sorted(enriched_cats)[:10]}", flush=True)

    # Estimated cost (gpt-4o-mini: $0.15/M input, $0.60/M output)
    total_time = t_a + t_b1 + t_b2 + t_b3
    est_cost = total_time * 0.0005  # rough: ~$0.0005/sec of API time
    total_est_cost += est_cost
    print(f"    Time: {total_time:.0f}s, Est. cost: ${est_cost:.4f}", flush=True)

print(f"\n{'='*50}")
print(f"B4 complete! Total est. cost: ${total_est_cost:.4f}", flush=True)
print(f"Results in: {BASE}/<paper>/B4_hybrid/seed_1/graph.json", flush=True)
