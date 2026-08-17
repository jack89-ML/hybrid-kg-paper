#!/usr/bin/env python3
"""
Reproducibility Experiment: Run all 4 methods N times, measure GED.

Usage:
  # Install dependencies first
  pip install spacy pymupdf requests numpy
  python -m spacy download en_core_web_trf

  # Run experiment
  python experiments/run_reproducibility.py --paper-name KGGen --seed 1

  # Run all
  python experiments/run_reproducibility.py --run-all
"""
import json, os, sys, time, hashlib, argparse, random
from pathlib import Path

# Add code to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))
from hybrid_kg import SpacyBackbone, LLMEnricher, KGExporter


# ── Baseline B2: LLM Zero-shot ──────────────────────────────────────

class LLMZeroShot:
    """LLM-only: estrae KG dal testo senza vincoli strutturali."""

    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini",
                 base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def extract(self, text: str) -> tuple[list, list]:
        """LLM estrae liberamente entities e triples dal testo."""
        prompt = f"""Extract a knowledge graph from the following text.

Return valid JSON with two keys:
1. "entities": array of {{"text": str, "label": str}} — all named entities
2. "triples": array of {{"subject": str, "predicate": str, "object": str}} — all relations

Text:
{text[:8000]}

Respond with ONLY valid JSON (no markdown)."""

        import requests
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"model": self.model, "messages": [
                    {"role": "system", "content": "You extract knowledge graphs from text. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ], "temperature": 0.3, "max_tokens": 4000},
                timeout=120
            )
            content = resp.json()["choices"][0]["message"]["content"]
            import re
            # Parse JSON
            match = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', content, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
            else:
                data = json.loads(content)
            entities = data.get("entities", [])
            triples = data.get("triples", [])
        except Exception as e:
            print(f"  [LLM-zero-shot] Error: {e}")
            entities, triples = [], []
        return entities, triples


# ── Baseline B3: LLM Schema-guided ──────────────────────────────────

class LLMSchemaGuided:
    """LLM con schema predefinito: estrae entità e relazioni di tipi specifici."""

    SCHEMA = """
Entity types: Person, Organization, Location, Concept, Method, Dataset, Metric, Software
Relation types: authored_by, published_in, part_of, uses, improves_on, 
                compares_with, evaluates, introduces, proposes, achieves
"""

    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini",
                 base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def extract(self, text: str) -> tuple[list, list]:
        prompt = f"""Extract a knowledge graph from the text below.

Entity types: {self.SCHEMA}

Return JSON:
{{"entities": [{{"text": str, "type": str}}], 
  "triples": [{{"subject": str, "relation": str (from schema), "object": str}}]}}

Only use relation types from: authored_by, published_in, part_of, uses, 
improves_on, compares_with, evaluates, introduces, proposes, achieves

Text:
{text[:8000]}
"""
        import requests, re
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"model": self.model, "messages": [
                    {"role": "system", "content": "You extract KGs from text. JSON only."},
                    {"role": "user", "content": prompt}
                ], "temperature": 0.3, "max_tokens": 4000},
                timeout=120
            )
            content = resp.json()["choices"][0]["message"]["content"]
            match = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', content, re.DOTALL)
            data = json.loads(match.group(1)) if match else json.loads(content)
            return data.get("entities", []), data.get("triples", [])
        except Exception as e:
            print(f"  [LLM-schema] Error: {e}")
            return [], []


# ── Graph Edit Distance ─────────────────────────────────────────────

def graph_edit_distance(graph_a: dict, graph_b: dict) -> float:
    """Calcola GED normalizzato tra due grafi."""
    nodes_a = {n["id"] for n in graph_a.get("nodes", [])}
    nodes_b = {n["id"] for n in graph_b.get("nodes", [])}
    edges_a = {(e["source"], e["relation"], e["target"]) for e in graph_a.get("edges", [])}
    edges_b = {(e["source"], e["relation"], e["target"]) for e in graph_b.get("edges", [])}

    node_diff = len(nodes_a.symmetric_difference(nodes_b))
    edge_diff = len(edges_a.symmetric_difference(edges_b))

    max_nodes = max(len(nodes_a), len(nodes_b), 1)
    max_edges = max(len(edges_a), len(edges_b), 1)

    # Normalized: 0 = identici, 1 = completamente diversi
    node_ged = node_diff / max_nodes
    edge_ged = edge_diff / max_edges

    return 0.5 * node_ged + 0.5 * edge_ged


# ── Experiment Runner ───────────────────────────────────────────────

class ReproducibilityExperiment:
    METHODS = {
        "B1_spacy": "spaCy deterministico (baseline)",
        "B2_llm_zero": "LLM zero-shot (baseline)",
        "B3_llm_schema": "LLM schema-guided (baseline)",
        "B4_hybrid": "ibrido (nostro metodo)",
    }

    def __init__(self, api_key: str, n_runs: int = 5, output_dir: str = "results"):
        self.api_key = api_key
        self.n_runs = n_runs
        self.output_dir = Path(output_dir)
        self.spacy = SpacyBackbone()
        self.exporter = KGExporter()

    def run_single(self, text: str, paper_name: str, method: str, seed: int) -> dict:
        """Run singolo per un metodo su un paper."""
        random.seed(seed)  # Per LLM temperature

        if method == "B1_spacy":
            triples = self.spacy.extract_triples(text)
            entities = self.spacy.extract_entities(text)
            graph = self.exporter.to_json(triples, entities)
            cost = 0.0

        elif method == "B2_llm_zero":
            llm = LLMZeroShot(self.api_key)
            entities, triples = llm.extract(text)
            # Convert to KG format
            graph = {
                "nodes": [{"id": e["text"], "label": e["text"], "category": e.get("label", "unknown"),
                          "confidence": 0.5, "source": "llm_zero_shot"} for e in entities],
                "edges": [{"source": t["subject"], "target": t["object"], "relation": t.get("predicate", ""),
                          "relation_type": "unknown", "confidence": 0.5, "source": "llm_zero_shot",
                          "sentence": ""} for t in triples],
                "metadata": {}
            }
            cost = 0.01  # approssimato

        elif method == "B3_llm_schema":
            llm = LLMSchemaGuided(self.api_key)
            entities, triples = llm.extract(text)
            graph = {
                "nodes": [{"id": e["text"], "label": e["text"], "category": e.get("type", "unknown"),
                          "confidence": 0.5, "source": "llm_schema_guided"} for e in entities],
                "edges": [{"source": t["subject"], "target": t["object"], "relation": t.get("relation", ""),
                          "relation_type": t.get("relation", "unknown"), "confidence": 0.5,
                          "source": "llm_schema_guided", "sentence": ""} for t in triples],
                "metadata": {}
            }
            cost = 0.01

        elif method == "B4_hybrid":
            # Fase A: spaCy
            triples = self.spacy.extract_triples(text)
            entities = self.spacy.extract_entities(text)
            # Fase B: LLM enrichment (solo se API key)
            if self.api_key:
                enricher = LLMEnricher(self.api_key)
                entities = enricher.enrich_nodes(entities)
                triples = enricher.enrich_triples(triples, entities)
                implicit = enricher.infer_implicit_relations(triples, entities)
            else:
                implicit = []
            graph = self.exporter.to_json(triples, entities, implicit)
            cost = 0.02  # approssimato
        else:
            raise ValueError(f"Unknown method: {method}")

        # Salva
        method_dir = self.output_dir / paper_name / method / f"seed_{seed}"
        method_dir.mkdir(parents=True, exist_ok=True)
        graph["metadata"] = {
            "paper": paper_name,
            "method": method,
            "seed": seed,
            "node_count": len(graph.get("nodes", [])),
            "edge_count": len(graph.get("edges", [])),
            "cost_usd": cost,
        }
        with open(method_dir / "graph.json", "w") as f:
            json.dump(graph, f, indent=2)

        return graph

    def run_paper(self, text: str, paper_name: str, methods: list[str] = None):
        """Esegue tutti i metodi N volte per un paper."""
        if methods is None:
            methods = list(self.METHODS.keys())

        results = {"paper": paper_name, "n_runs": self.n_runs, "methods": {}}

        for method in methods:
            print(f"\n=== {self.METHODS[method]} ===")
            graphs = []
            for seed in range(1, self.n_runs + 1):
                print(f"  Run {seed}/{self.n_runs}...")
                graph = self.run_single(text, paper_name, method, seed)
                graphs.append(graph)

            # Calcola GED pairwise
            ged_scores = []
            for i in range(len(graphs)):
                for j in range(i + 1, len(graphs)):
                    ged = graph_edit_distance(graphs[i], graphs[j])
                    ged_scores.append(ged)

            import numpy as np
            results["methods"][method] = {
                "mean_ged": float(np.mean(ged_scores)) if ged_scores else 0.0,
                "std_ged": float(np.std(ged_scores)) if ged_scores else 0.0,
                "min_ged": float(min(ged_scores)) if ged_scores else 0.0,
                "max_ged": float(max(ged_scores)) if ged_scores else 0.0,
                "node_counts": [g["metadata"]["node_count"] for g in graphs],
                "edge_counts": [g["metadata"]["edge_count"] for g in graphs],
            }

        # Salva report
        report_path = self.output_dir / paper_name / "reproducibility_report.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Report saved: {report_path}")
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-name", help="Paper name (must match corpus file)")
    parser.add_argument("--run-all", action="store_true", help="Run all 5 papers")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of runs per method")
    parser.add_argument("--api-key", help="OpenRouter API key")
    parser.add_argument("--methods", nargs="+",
                        default=["B1_spacy", "B4_hybrid"],
                        help="Methods to run")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Warning: No API key. Running spaCy backbone only (no LLM methods)")

    exp = ReproducibilityExperiment(api_key, n_runs=args.n_runs, output_dir=args.output)

    corpus_dir = Path("corpus")
    if args.run_all:
        paper_files = sorted(corpus_dir.glob("*.txt"))
        for pf in paper_files:
            name = pf.stem
            print(f"\n{'='*60}")
            print(f"Paper: {name}")
            print(f"{'='*60}")
            text = pf.read_text()
            exp.run_paper(text, name, methods=args.methods)
    elif args.paper_name:
        txt_path = corpus_dir / f"{args.paper_name}.txt"
        if not txt_path.exists():
            papers = list(corpus_dir.glob("*.txt"))
            print(f"Paper '{args.paper_name}.txt' not found in corpus/")
            print(f"Available: {[p.stem for p in papers]}")
            return
        text = txt_path.read_text()
        exp.run_paper(text, args.paper_name, methods=args.methods)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
