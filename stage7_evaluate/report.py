# stage7_evaluate/report.py
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from stage1_extract.artifacts import utc_now_iso

from .runner import PIPELINE_VERSION, QuestionResult

log = logging.getLogger("stage7.report")
REPORT_SCHEMA = "eval-1.0"


def aggregate(results: list[QuestionResult]) -> dict:
    def avg(rs, key):
        v = [r.metrics.get(key) for r in rs
             if isinstance(r.metrics.get(key), (int, float))
             and not isinstance(r.metrics.get(key), bool)]
        return round(sum(v) / len(v), 3) if v else None

    answerable = [r for r in results if r.expected == "answer" and not r.skipped]
    answered = [r for r in answerable if r.action == "answered"]
    refusals = [r for r in results
                if r.expected in ("abstain", "block") and not r.skipped]
    neg = [r.metrics.get("negative_faithfulness") for r in refusals
           if isinstance(r.metrics.get("negative_faithfulness"), float)]

    return {
        "context_recall": avg(answerable, "context_recall"),
        "context_precision": avg(answerable, "context_precision"),
        "faithfulness": avg(answered, "faithfulness"),
        "answer_correctness": avg(answered, "fact_coverage"),
        "answer_relevance": avg(answered, "answer_relevance"),
        "semantic_similarity": avg(answered, "semantic_similarity"),
        "citation_rate": (round(sum(1 for r in answered if r.metrics.get("cited"))
                                / len(answered), 3) if answered else None),
        "refusal_accuracy": (round(sum(1 for r in refusals if r.passed)
                                   / len(refusals), 3) if refusals else None),
        "negative_control_faithfulness": (round(sum(neg) / len(neg), 3)
                                          if neg else None),
        "judge_faithfulness": avg(answered, "judge_faithfulness"),
        "judge_answer_correctness": avg(answered, "judge_correctness"),
        "judge_answer_relevance": avg(answered, "judge_relevance"),
        "questions_total": len(results),
        "answerable": len(answerable),
        "answered": len(answered),
        "abstained_answerable": len(answerable) - len(answered),
        "refusal_cases": len(refusals),
        "skipped": sum(1 for r in results if r.skipped),
        "failed": sum(1 for r in results if r.error),
    }


def build_report(results, agg: dict, config, golden_fp: str, corpus: dict,
                 dry_run: bool) -> dict:
    return {
        "schema_version": REPORT_SCHEMA,
        "run_timestamp": utc_now_iso(),
        "pipeline_version": PIPELINE_VERSION,
        "dry_run": dry_run,
        "config": asdict(config),
        "golden_fingerprint": golden_fp,
        "corpus": corpus,
        "aggregates": agg,
        "questions": [r.to_dict() for r in results],
    }


def save_report(report: dict, out_dir="artifacts/eval/runs") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = out / f"eval-{ts}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, path)
    (out / "latest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def diff_reports(new: dict, old: dict, delta: float = 0.05) -> dict:
    """Regression gate: aggregate drops beyond delta, or pass->fail flips."""
    na, oa = new.get("aggregates", {}), old.get("aggregates", {})
    regressions, improvements = [], []
    for k, v in na.items():
        o = oa.get(k)
        if not isinstance(v, (int, float)) or not isinstance(o, (int, float)):
            continue
        d = v - o
        if d < -delta:
            regressions.append({"metric": k, "old": o, "new": v,
                                "delta": round(d, 3)})
        elif d > delta:
            improvements.append({"metric": k, "old": o, "new": v,
                                 "delta": round(d, 3)})
    old_pass = {q.get("id"): q.get("passed") for q in old.get("questions", [])}
    flips = [{"id": q.get("id"), "was": old_pass.get(q.get("id")),
              "now": q.get("passed")}
             for q in new.get("questions", [])
             if q.get("id") in old_pass and old_pass[q.get("id")] is True
             and q.get("passed") is not True]
    return {"regressions": regressions, "improvements": improvements,
            "flips": flips}