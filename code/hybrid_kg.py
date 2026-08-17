#!/usr/bin/env python3
"""
Hybrid KG Construction: Deterministic NLP (spaCy) + LLM Enrichment.

Phase 2 — Experiment Design / Phase 3 — Execution
==================================================
Architettura:
  Fase A: spaCy dependency parsing → SVO triple backbone (deterministico)
  Fase B: LLM arricchisce nodi e archi esistenti (vincolato)
  Fase C: Export KG pronto per GraphRAG

Metodo proposto per il paper "Hybrid KG: Combining Deterministic NLP
and LLM Enrichment for Reproducible Knowledge Base Construction".
"""

import json, os, sys, time, hashlib, argparse
from pathlib import Path
from typing import Optional

# ── Fase A: spaCy backbone ──────────────────────────────────────────

class SpacyBackbone:
    """Estrae triple SVO deterministiche usando spaCy dependency parsing.

    Vocabolario di relazione (mapping dependency → relation type):
      nsubj/ROOT/dobj  →  (subject, action, object)    [azione diretta]
      nsubjpass/ROOT/agent → (subject, action, agent)   [passiva]
      amod            →  (entity, has_attribute, attr) [attributo]
      prep            →  (entity, relation, entity)    [preposizionale]
      conj            →  (entity, conjunction, entity) [coordinazione]
    """

    DEP_RELATION_MAP = {
        ("nsubj", "ROOT", "dobj"): "direct_action",
        ("nsubj", "ROOT", "attr"): "attribute",
        ("nsubj", "ROOT", "xcomp"): "extended_action",
        ("nsubjpass", "ROOT", "agent"): "passive_action",
        ("nsubj", "prep"): "prepositional",
    }

    def __init__(self, model_name: str = "en_core_web_trf"):
        self.model_name = model_name
        self.nlp = None

    def _ensure_model(self):
        if self.nlp is not None:
            return
        try:
            import spacy
            self.nlp = spacy.load(self.model_name)
        except OSError:
            print(f"[spaCy] Model {self.model_name} not found, downloading...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "spacy", "download", self.model_name])
            import spacy
            self.nlp = spacy.load(self.model_name)
        # Add coref for better entity resolution if available
        try:
            self.nlp.add_pipe("experimental_coref")
        except (ValueError, ImportError):
            pass

    def extract_triples(self, text: str) -> list[dict]:
        """Estrae triple SVO da un testo. Output determinista."""
        self._ensure_model()
        doc = self.nlp(text)
        triples = []

        for sent in doc.sents:
            root = sent.root
            if root.pos_ != "VERB":
                continue

            # Trova subject e object via dependency parse
            subject = None
            obj = None
            prep_objects = []

            for child in root.children:
                if child.dep_ in ("nsubj", "nsubjpass", "agent"):
                    subject = self._extract_phrase(child)
                elif child.dep_ in ("dobj", "attr", "xcomp", "pobj"):
                    obj = self._extract_phrase(child)
                elif child.dep_ == "prep":
                    prep_objects.append((child.text, self._extract_phrase(
                        list(child.children)[0] if list(child.children) else None
                    )))

            if subject and root.text and obj:
                triple = {
                    "subject": subject,
                    "predicate": root.lemma_,
                    "object": obj,
                    "relation_type": "direct_action",
                    "sentence": sent.text,
                    "source": "dependency_parse",
                    "span_start": sent.start,
                    "span_end": sent.end,
                }
                triples.append(triple)

            # Aggiungi relazioni preposizionali
            for prep, prep_obj in prep_objects:
                if subject and prep_obj:
                    triples.append({
                        "subject": subject,
                        "predicate": f"{root.lemma_}_{prep}",
                        "object": prep_obj,
                        "relation_type": "prepositional",
                        "sentence": sent.text,
                        "source": "dependency_parse",
                        "span_start": sent.start,
                        "span_end": sent.end,
                    })

        # Hash deterministico dell'output (per riproducibilità)
        return triples

    def _extract_phrase(self, token) -> str:
        """Estrae la frase nominale completa (sottoalbero) per un token.

        Ordina i modifier (det, amod, compound, nummod, poss, quantmod)
        in ordine di apparizione nel testo per ottenere sintassi naturale.
        """
        if token is None:
            return ""
        # Raccogli tutti i modifier (sinistra e destra)
        pre_modifiers = []
        post_modifiers = []
        for child in token.children:
            if child.dep_ in ("compound", "amod", "det", "nummod", "poss", "quantmod"):
                if child.i < token.i:
                    pre_modifiers.append(child)
                else:
                    post_modifiers.append(child)
            # Ricorsione: aggettivi composti (es. "state-of-the-art")
            if child.dep_ == "amod" and list(child.children):
                for sub in child.children:
                    if sub.dep_ == "compound" and sub.i < child.i:
                        pre_modifiers.append(sub)

        # Ordina per posizione nel testo
        pre_modifiers.sort(key=lambda x: x.i)
        post_modifiers.sort(key=lambda x: x.i)

        # Costruisci: pre-modifier + head + post-modifier
        all_tokens = []
        for m in pre_modifiers:
            all_tokens.append(m.text_with_ws.strip())
        all_tokens.append(token.text)
        for m in post_modifiers:
            all_tokens.append(m.text_with_ws.strip())

        phrase = " ".join(all_tokens).strip()
        # Rimuovi det iniziale se presente (per soggetti/oggetti)
        import re
        phrase = re.sub(r'^(the|an?|this|that|these|those) ', '', phrase, flags=re.IGNORECASE)
        return phrase

    def extract_entities(self, text: str) -> list[dict]:
        """Estrae entità nominate (NER) deterministiche."""
        self._ensure_model()
        doc = self.nlp(text)
        entities = []
        seen = set()
        for ent in doc.ents:
            key = f"{ent.text}:{ent.label_}"
            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "source": "spacy_ner",
                })
        return entities

    def get_hash(self, text: str) -> str:
        """Hash deterministico dell'estrazione (prova di riproducibilità)."""
        triples = self.extract_triples(text)
        return hashlib.sha256(
            json.dumps(triples, sort_keys=True).encode()
        ).hexdigest()


# ── Fase B: LLM Enrichment ──────────────────────────────────────────

class LLMEnricher:
    """Arricchisce nodi e archi del backbone deterministico.

    L'LLM NON crea nuove entità o relazioni da zero.
    Operazioni consentite:
      - Disambiguare: "Apple" → "Apple Inc." (azienda) vs "apple" (frutto)
      - Tipizzare: relazione "uses" → "causal:instrument"
      - Collegamento implicito: se nodo A ha relazione R1 con B, e B ha R2 con C,
        inferisci relazione A-B-C come pathway
      - Confidenza: assegna score [0,1] a ogni arco esistente

    Prompt engineering: struttura JSON vincolata in output.
    """

    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini",
                 base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = 3

    def enrich_nodes(self, entities: list[dict]) -> list[dict]:
        """Arricchisce entità esistenti con disambiguazione e categorie.

        Args:
            entities: lista da SpacyBackbone.extract_entities()

        Returns:
            entità arricchite con 'disambiguated', 'category', 'confidence'
        """
        prompt = self._build_enrich_prompt(entities)
        response = self._call_llm(prompt)
        return self._parse_structured_response(response, entities)

    def enrich_triples(self, triples: list[dict], entities: list[dict]) -> list[dict]:
        """Arricchisce triple esistenti con tipo relazione e confidenza.

        Args:
            triples: lista da SpacyBackbone.extract_triples()
            entities: entità arricchite

        Returns:
            triple arricchite con 'relation_subtype', 'confidence', 'semantic_roles'
        """
        # Batch processing per risparmiare token
        prompt = self._build_triple_prompt(triples, entities)
        response = self._call_llm(prompt)
        return self._parse_triple_response(response, triples)

    def infer_implicit_relations(self, triples: list[dict], entities: list[dict]) -> list[dict]:
        """Inferisce relazioni implicite tra entità esistenti (multi-hop).

        Esempio: (A, authored, B) + (B, published_in, C) → (A, contributes_to, C)

        Restituisce nuove triple con source='llm_inferred'.
        """
        prompt = self._build_implicit_prompt(triples, entities)
        response = self._call_llm(prompt)
        return self._parse_implicit_response(response)

    def _call_llm(self, prompt: str) -> Optional[str]:
        import requests
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "Sei un assistente che risponde SOLO con JSON valido. Nessun preambolo."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.1,  # bassa per consistenza
                        "max_tokens": 16000,
                    },
                    timeout=60,
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                elif resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 60)
                    time.sleep(wait)
                    continue
                else:
                    print(f"[LLM] HTTP {resp.status_code}: {resp.text[:200]}")
                    return None
            except Exception as e:
                print(f"[LLM] Attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        return None

    def _build_enrich_prompt(self, entities: list[dict]) -> str:
        return f"""You are a knowledge graph enrichment system. Given NER entities extracted
by spaCy, disambiguate and categorize each one.

RULES:
- Do NOT add new entities
- Do NOT remove entities
- For each entity, provide: disambiguated name, category (Person/Organization/
  Location/Concept/Method/Dataset/Metric/Software), and confidence [0,1]
- If entity is already unambiguous, keep original text

Entities:
{json.dumps(entities, indent=2)}

Respond with a JSON array where each element has:
  "text": original text,
  "disambiguated": best canonical name,
  "category": entity category,
  "confidence": float 0-1
"""

    def _build_triple_prompt(self, triples: list[dict], entities: list[dict]) -> str:
        return f"""Given SVO triples from dependency parsing and NER entities,
enrich each triple with: relation subtype, confidence score, and
semantic roles (agent/patient/instrument).

RULES:
- Do NOT add new triples
- Do NOT remove triples
- Relation subtypes: causal, temporal, compositional, attributional,
  comparative, procedural, definitional, or other
- Confidence: 0 (unlikely)-1 (certain)

NODES (entities):
{json.dumps(entities[:20], indent=2)}

TRIPLES:
{json.dumps(triples[:50], indent=2)}

Respond with a JSON array where each element has:
  "subject": original subject text,
  "predicate": original predicate,
  "object": original object text,
  "relation_subtype": str,
  "confidence": float 0-1,
  "semantic_role_subject": str,
  "semantic_role_object": str
"""

    def _build_implicit_prompt(self, triples: list[dict], entities: list[dict]) -> str:
        return f"""Given a set of entity-relation triples (from dependency parsing),
infer implicit (multi-hop) relations between entities. Only add relations
that are STRONGLY implied by the existing graph structure.

RULES:
- Only connect entities that already appear in the triple list
- Only add relations that are logically entailed (A→B→C implies A→C)
- Assign relation type from: contributes_to, part_of, specializes,
  contradicts, exemplifies, follows, leads_to
- Assign confidence [0,1]

EXISTING TRIPLES:
{json.dumps(triples[:100], indent=2)}

Respond with JSON array:
  [{{"subject": ..., "relation": ..., "object": ..., "relation_type": ..., "confidence": ...}}]
  or empty array [] if no implicit relations found.
"""

    def _parse_structured_response(self, response: Optional[str], fallback: list) -> list:
        if not response:
            return fallback
        import re
        try:
            # Try direct JSON parse
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown
            match = re.search(r'```(?:json)?\s*\n?(\[.*?\])\s*\n?```', response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return fallback

    def _parse_triple_response(self, response: Optional[str], original_triples: list) -> list:
        enriched = self._parse_structured_response(response, [])
        if not enriched:
            return original_triples
        # Merge LLM enrichment back into original triples
        enriched_map = {}
        for e in enriched:
            key = (e.get("subject", ""), e.get("predicate", ""), e.get("object", ""))
            enriched_map[key] = e

        result = []
        for t in original_triples:
            key = (t["subject"], t["predicate"], t["object"])
            if key in enriched_map:
                t["relation_subtype"] = enriched_map[key].get("relation_subtype", "unknown")
                t["confidence"] = enriched_map[key].get("confidence", 0.5)
                t["semantic_role_subject"] = enriched_map[key].get("semantic_role_subject", "")
                t["semantic_role_object"] = enriched_map[key].get("semantic_role_object", "")
            else:
                t["relation_subtype"] = "unknown"
                t["confidence"] = 0.5
            result.append(t)
        return result

    def _parse_implicit_response(self, response: Optional[str]) -> list:
        return self._parse_structured_response(response, [])


# ── Fase C: KG Export ───────────────────────────────────────────────

class KGExporter:
    """Esporta KG in formato standard per GraphRAG e analisi."""

    @staticmethod
    def to_json(triples: list[dict], entities: list[dict],
                implicit: list[dict] = None) -> dict:
        graph = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "node_count": 0,
                "edge_count": 0,
                "source": "hybrid_kg",
            }
        }

        # Nodes: deduplicati
        node_map = {}
        for ent in entities:
            node_id = ent["text"]
            if node_id not in node_map:
                node_map[node_id] = {
                    "id": node_id,
                    "label": ent.get("disambiguated", ent["text"]),
                    "category": ent.get("category", ent.get("label", "unknown")),
                    "confidence": ent.get("confidence", 1.0),
                    "source": ent.get("source", "spacy_ner"),
                }

        # Edges: from triples
        for t in triples:
            subj = t["subject"]
            obj = t["object"]
            if subj not in node_map:
                node_map[subj] = {"id": subj, "label": subj, "category": "unknown", "confidence": 0.5, "source": "inferred"}
            if obj not in node_map:
                node_map[obj] = {"id": obj, "label": obj, "category": "unknown", "confidence": 0.5, "source": "inferred"}
            edge = {
                "source": subj,
                "target": obj,
                "relation": t.get("predicate", ""),
                "relation_type": t.get("relation_type", "unknown"),
                "relation_subtype": t.get("relation_subtype", "unknown"),
                "confidence": t.get("confidence", 0.5),
                "source": t.get("source", "dependency_parse"),
                "sentence": t.get("sentence", "")[:200],
            }
            graph["edges"].append(edge)

        # Add implicit edges
        if implicit:
            for imp in implicit:
                if imp["subject"] in node_map and imp["object"] in node_map:
                    graph["edges"].append({
                        "source": imp["subject"],
                        "target": imp["object"],
                        "relation": imp.get("relation", ""),
                        "relation_type": imp.get("relation_type", "implicit"),
                        "relation_subtype": "inferred",
                        "confidence": imp.get("confidence", 0.3),
                        "source": "llm_inferred",
                        "sentence": "",
                    })

        graph["nodes"] = list(node_map.values())
        graph["metadata"]["node_count"] = len(graph["nodes"])
        graph["metadata"]["edge_count"] = len(graph["edges"])
        return graph

    @staticmethod
    def save(graph: dict, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
        print(f"[KG] Saved: {path} ({graph['metadata']['node_count']} nodes, "
              f"{graph['metadata']['edge_count']} edges)")


# ── Pipeline completa ───────────────────────────────────────────────

class HybridKGPipeline:
    """Pipeline completa: spaCy → LLM → Export."""

    def __init__(self, api_key: str, spacy_model: str = "en_core_web_trf",
                 llm_model: str = "openai/gpt-4o-mini"):
        self.backbone = SpacyBackbone(spacy_model)
        self.enricher = LLMEnricher(api_key, model=llm_model)
        self.exporter = KGExporter()

    def run(self, text: str, paper_name: str, output_dir: str = "results",
            enrich: bool = True, save_intermediate: bool = True) -> dict:
        """Esegue pipeline completa su un testo.

        Returns:
            dict con metriche e percorso output
        """
        out = Path(output_dir) / paper_name
        out.mkdir(parents=True, exist_ok=True)

        # Fase A: Backbone deterministico
        print(f"[Phase A] spaCy extraction for '{paper_name}'...")
        t0 = time.time()
        triples = self.backbone.extract_triples(text)
        entities = self.backbone.extract_entities(text)
        backbone_hash = self.backbone.get_hash(text)
        t_a = time.time() - t0
        print(f"  → {len(triples)} triples, {len(entities)} entities in {t_a:.1f}s")
        print(f"  → Determinism hash: {backbone_hash[:16]}...")

        if save_intermediate:
            with open(out / "backbone_triples.json", "w") as f:
                json.dump(triples, f, indent=2, ensure_ascii=False)
            with open(out / "backbone_entities.json", "w") as f:
                json.dump(entities, f, indent=2, ensure_ascii=False)

        results = {
            "paper": paper_name,
            "phase_a": {"triples": len(triples), "entities": len(entities),
                        "time_s": t_a, "hash": backbone_hash},
            "phase_b": {},
            "phase_c": {"output_path": ""},
        }

        # Fase B: LLM Enrichment
        if enrich and self.enricher.api_key:
            print(f"[Phase B] LLM enrichment...")
            t0 = time.time()

            # B.1: Enrich entities
            enriched_entities = self.enricher.enrich_nodes(entities)
            t_b1 = time.time() - t0

            # B.2: Enrich triples
            enriched_triples = self.enricher.enrich_triples(triples, enriched_entities)
            t_b2 = time.time() - t0

            # B.3: Implicit relations
            implicit = self.enricher.infer_implicit_relations(enriched_triples, enriched_entities)
            t_b3 = time.time() - t0

            t_b = time.time() - t0
            print(f"  → {len(enriched_entities)} entities enriched")
            print(f"  → {len(enriched_triples)} triples enriched")
            print(f"  → {len(implicit)} implicit relations in {t_b:.1f}s")

            if save_intermediate:
                with open(out / "enriched_entities.json", "w") as f:
                    json.dump(enriched_entities, f, indent=2, ensure_ascii=False)
                with open(out / "implicit_relations.json", "w") as f:
                    json.dump(implicit, f, indent=2, ensure_ascii=False)

            results["phase_b"] = {
                "entities_enriched": len(enriched_entities),
                "triples_enriched": len(enriched_triples),
                "implicit_relations": len(implicit),
                "time_s": t_b,
            }

            # Use enriched for export
            final_triples = enriched_triples
            final_entities = enriched_entities
        else:
            final_triples = triples
            final_entities = entities
            implicit = []

        # Fase C: Export
        graph = self.exporter.to_json(final_triples, final_entities, implicit)
        path = str(out / "kg_graph.json")
        self.exporter.save(graph, path)

        results["phase_c"] = {"output_path": path}
        results["summary"] = {
            "nodes": graph["metadata"]["node_count"],
            "edges": graph["metadata"]["edge_count"],
            "density": graph["metadata"]["edge_count"] / max(graph["metadata"]["node_count"], 1),
        }

        # Salva report metriche
        with open(out / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Done] Results saved to {out}/")

        return results


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hybrid KG Pipeline")
    parser.add_argument("--text", help="Input text file path")
    parser.add_argument("--paper", default="experiment", help="Paper/experiment name")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--no-enrich", action="store_true", help="Skip LLM enrichment")
    parser.add_argument("--spacy-model", default="en_core_web_trf")
    parser.add_argument("--llm-model", default="openai/gpt-4o-mini")
    parser.add_argument("--api-key", help="OpenRouter API key (or env OPENROUTER_API_KEY)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.no_enrich:
        print("Warning: No API key provided. Running backbone only (no enrichment).")
        args.no_enrich = True

    pipeline = HybridKGPipeline(
        api_key=api_key,
        spacy_model=args.spacy_model,
        llm_model=args.llm_model,
    )

    if args.text:
        with open(args.text) as f:
            text = f.read()
        results = pipeline.run(text, args.paper, args.output, enrich=not args.no_enrich)
        print(f"\nSummary: {results['summary']}")
    else:
        print("Usage: python hybrid_kg.py --text sample.txt --paper test")
        print("       python hybrid_kg.py --text sample.txt --paper test --no-enrich")


if __name__ == "__main__":
    main()
