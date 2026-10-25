"""Extract source-referenced attribute candidates without model inference.

Run ``uv run python -m rag.attributes .data/catalog/manifest.json`` after import.
This stores unreviewed evidence in products.attributes.extraction, preserving
original details, and writes a private coverage report beside the manifest.
Nothing extracted here is enabled as a retrieval hard filter.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

VERSION = "rules-v2"
DETAIL_FIELDS = {"brand": "brand", "material": "material", "fabric": "material",
                 "fabric type": "material", "color": "color", "colour": "color",
                 "size": "size", "style": "style", "pattern": "pattern", "fit type": "fit",
                 "closure type": "closure", "sleeve type": "sleeve", "neck style": "neck"}
DETAIL_FIELDS.update({"cup style": "cup_style", "sleeve style": "sleeve_style",
                      "upper material": "material", "lining material": "material",
                      "sole material": "material"})
UNKNOWN = {"", "unknown", "n/a", "na", "none", "not applicable", "not_applicable", "null"}
PATTERNS = {
    "material_mention": r"\b(?:(?:faux|synthetic|vegan|genuine)\s+)?(?:cotton|polyester|spandex|elastane|leather|suede|nylon|wool|rayon|silk|linen|alloy|stainless steel|sterling silver)\b",
    "composition_mention": r"\b\d{1,3}(?:\.\d+)?\s*%\s*(?:cotton|polyester|spandex|elastane|nylon|wool|rayon|silk|linen)\b",
    "closure_mention": r"\b(?:zipper|button|buckle|pull[ -]on|hook[ -]and[ -]loop|magnetic)\s+(?:closure|fastening|clasp)\b",
    "fit_mention": r"\b(?:slim|regular|relaxed|athletic|loose)\s+fit\b",
    "care_instruction": r"\b(?:(?:do not|never)\s+(?:machine wash|hand wash|dry clean|tumble dry)|machine wash(?:able)?|hand wash|dry clean(?: only)?|do not bleach|line dry|tumble dry(?: low)?)\b",
    "measurement_mention": r"\b\d+(?:\.\d+)?\s*(?:cm|mm|inches|inch|oz)\b",
}


def visible_source(text: str) -> str:
    """Mask markup/scripts while retaining original character offsets."""
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1\s*>", lambda m: " " * len(m[0]), text, flags=re.I | re.S)
    return re.sub(r"<[^>]*>", lambda m: " " * len(m[0]), text)


def candidate_scope(field: str, text: str, start: int, end: int, source: dict[str, Any]) -> dict[str, Any]:
    """Attach explicit nearby scope; leave unsupported or tabular relations unknown.

    Scope evidence uses the same source string and offsets as the candidate.
    Detail-key evidence is identified separately. These patterns do not resolve
    variants or interpret a multi-column size table.
    """
    result = {"applies_to": "unknown", "variant_condition": "unknown",
              "measurement_role": "unknown", "size_label": "unknown",
              "scope_evidence": [], "scope_warnings": []}
    masked = visible_source(text)

    def record(match, role):
        """Append the source offsets and text supporting a scope role."""
        result["scope_evidence"].append({"role": role, "matched_text": text[match.start():match.end()],
                                          "start": match.start(), "end": match.end()})

    if field in {"material", "material_mention", "composition_mention"}:
        suffix = re.match(r"\s+(upper|lining|sole|outsole|insole|shell|heel)\b", masked[end:], re.I)
        prefix = re.search(r"\b(upper|lining|sole|outsole|insole|shell|heel)\s*:\s*$", masked[:start], re.I)
        if suffix:
            result["applies_to"] = suffix.group(1).casefold()
            # Translate the suffix match's local offsets to the source string.
            a, b = end + suffix.start(1), end + suffix.end(1)
            result["scope_evidence"].append({"role": "component", "matched_text": text[a:b], "start": a, "end": b})
        elif prefix:
            result["applies_to"] = prefix.group(1).casefold()
            record(prefix, "component")
        elif source.get("field") == "details":
            key = source.get("key", "")
            component = re.match(r"(Upper|Lining|Sole) Material$", key, re.I)
            if component:
                result["applies_to"] = component.group(1).casefold()
                result["scope_evidence"].append({"role": "component", "details_key": key})
    if field in {"material", "material_mention", "composition_mention"}:
        clause_start = max(masked.rfind(";", 0, start), masked.rfind("\n", 0, start)) + 1
        boundaries = list(re.finditer(r"[.!?]\s+", masked[:start]))
        if boundaries:
            clause_start = max(clause_start, boundaries[-1].end())
        clause = masked[clause_start:start]
        variant = re.search(r"\b(?:solid colors?|heather grey|all other heathers|blue|black|white|red|"
                            r"green|grey|gray|pink)\s*:", clause, re.I)
        if variant:
            a, b = clause_start + variant.start(), clause_start + variant.end()
            result["variant_condition"] = text[a:b].rstrip(":").strip()
            result["scope_evidence"].append({"role": "variant", "matched_text": text[a:b], "start": a, "end": b})
        elif re.search(r"\b(?:colors?|colours?|heathers?|variants?)\b", masked, re.I):
            result["scope_warnings"].append("possible_variant_conditions")
    if field == "measurement_mention":
        prefix = masked[max(0, start - 45):start]
        role = re.search(r"\b(chest|bust|waist|length|width|height|shoulder|inseam|heel)\b[^;:.!?\n]{0,25}[:]?\s*$", prefix, re.I)
        if role:
            result["measurement_role"] = role.group(1).casefold()
            a, b = max(0, start - 45) + role.start(1), max(0, start - 45) + role.end(1)
            result["scope_evidence"].append({"role": "measurement", "matched_text": text[a:b], "start": a, "end": b})
        labels = list(re.finditer(r"\bSize\s+(XS|S|M|L|XL|[2-6]XL)\s*:", masked[:start], re.I))
        if labels and start - labels[-1].end() < 80 and not re.search(r"[;\n]", masked[labels[-1].end():start]):
            result["size_label"] = labels[-1].group(1).upper()
            record(labels[-1], "size")
        if re.search(r"\b(?:size (?:table|chart)|Tag\s+[SML]|Shoulder[-]{2,})", masked, re.I):
            result["scope_warnings"].append("size_table_requires_review")
            result["size_label"] = "unknown"
            result["measurement_role"] = "unknown"
    if field == "size":
        if re.search(r"\b(?:years?|months?)\b", text[start:end], re.I):
            result["scope_warnings"].append("age_range_not_standard_size")
        elif re.search(r"\b\d+(?:\.\d+)?\s*(?:cm|mm|inch|inches)\b", text[start:end], re.I):
            result["scope_warnings"].append("size_contains_measurements")
        elif re.search(r"[/,]", text[start:end]):
            result["scope_warnings"].append("multiple_size_options")
    return result


def extract(row: dict[str, Any]) -> dict[str, Any]:
    """Return bounded unreviewed candidates with exact source substrings.

    Structured claims and text mentions stay distinct. Values are never filled
    from images, seller names or model inference. Conflicting structured claims
    are flagged rather than resolved. Negation is a conservative context flag,
    not a language-understanding guarantee.
    """
    candidates: list[dict[str, Any]] = []

    def add(field, value, source, text, start, end, kind):
        """Append a bounded, source-backed claim with negation and scope flags."""
        if len(candidates) >= 500:
            return
        prefix = text[max(0, start - 35):start]
        negated = field != "care_instruction" and bool(re.search(r"\b(?:no|not|without|non|free of)\s+(?:\w+\s+){0,2}$", prefix, re.I))
        candidates.append({"field": field, "value": value, "kind": kind,
                           "review_status": "unreviewed", "negation_flag": negated,
                           "source": source, "matched_text": text[start:end],
                           "start": start, "end": end,
                           **candidate_scope(field, text, start, end, source),
                           "context": text[max(0, start - 60):min(len(text), end + 100)]})

    details = row.get("details") or {}
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            details = {}
    if isinstance(details, dict):
        for key, value in details.items():
            field = DETAIL_FIELDS.get(key.strip().casefold())
            if field and isinstance(value, str) and value.strip().casefold() not in UNKNOWN:
                start = len(value) - len(value.lstrip())
                add(field, value.strip(), {"field": "details", "key": key}, value,
                    start, len(value.rstrip()), "structured_source_claim")
    for source_field in ("features", "description"):
        values = row.get(source_field)
        if isinstance(values, str):
            values = [(None, values)]
        elif isinstance(values, list):
            values = list(enumerate(values))
        else:
            continue
        for index, text in values:
            if not isinstance(text, str):
                continue
            source = {"field": source_field, "index": index}
            masked = visible_source(text)
            for field, pattern in PATTERNS.items():
                for match in re.finditer(pattern, masked, re.I):
                    add(field, text[match.start():match.end()], source, text,
                        match.start(), match.end(), "text_mention")
            # Only explicit labels qualify as textual color/size/style claims.
            for match in re.finditer(r"\b(Cup Style|Sleeve Style|Color|Colour|Size|Style|Pattern)\s*:\s*([^;\n<>]{1,80})(?=[;\n<>]|$)", masked, re.I):
                value = match.group(2).strip()
                # Do not swallow a subsequent inline labelled field.
                value = re.split(r"\s+\b[A-Za-z]+\s*:", value)[0].strip().rstrip(".")
                if value.casefold() in UNKNOWN:
                    continue
                start = match.start(2) + len(match.group(2)) - len(match.group(2).lstrip())
                if text[start:start + len(value)] != value:
                    continue
                field = "color" if match.group(1).casefold() == "colour" else match.group(1).casefold().replace(" ", "_")
                add(field, value, source, text, start, start + len(value), "labelled_source_claim")
    claims: dict[tuple[str, str, str, str], set[str]] = {}
    def claim_group(candidate):
        """Compare claims only within the same component, variant and size."""
        return (candidate["field"], candidate["applies_to"], candidate["variant_condition"].casefold(), candidate["size_label"])
    for candidate in candidates:
        if candidate["kind"] != "text_mention" and not candidate["negation_flag"]:
            claims.setdefault(claim_group(candidate), set()).add(" ".join(candidate["value"].casefold().split()))
    conflicting_groups = {group for group, values in claims.items() if len(values) > 1}
    conflicts = sorted({group[0] for group in conflicting_groups})
    for candidate in candidates:
        candidate["conflict_flag"] = claim_group(candidate) in conflicting_groups
    return {"extractor_version": VERSION, "candidates": candidates, "conflicting_fields": conflicts,
            "candidate_limit_reached": len(candidates) >= 500}


def enrich(connection: psycopg.Connection, manifest_path: Path) -> dict[str, Any]:
    """Store candidate evidence for matching imported records in one transaction.

    A source ID or record-hash mismatch aborts the operation. Original details,
    raw source, vectors and runtime filters are not modified. Repeated extraction
    overwrites only the versioned extraction document, without duplicating it.
    """
    from .catalog import Manifest, local_asset

    if manifest_path.stat().st_size > 2_000_000:
        raise ValueError("Manifest exceeds 2 MB")
    manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    products = []
    total_bytes = 0
    with local_asset(manifest_path.parent, manifest.records).open("rb") as stream:
        for _ in range(manifest.max_lines):
            line = stream.readline(min(1_000_000, manifest.max_bytes - total_bytes) + 1)
            if not line:
                break
            total_bytes += len(line)
            if total_bytes > manifest.max_bytes or len(line) > 1_000_000:
                raise ValueError("Input exceeds manifest budget")
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("parent_asin"), str):
                raise ValueError("Expected product metadata")
            product_id = manifest.source_id + ":" + row["parent_asin"]
            stored = connection.execute(
                "SELECT attributes,source_record_sha256 FROM products "
                "WHERE id=%s AND dataset_source_id=%s FOR UPDATE", (product_id, manifest.source_id),
            ).fetchone()
            if stored is None or stored[1] != hashlib.sha256(line).hexdigest():
                raise ValueError("Imported source record does not match extraction input")
            evidence = extract(row)
            attributes = dict(stored[0] or {"schema_version": 1, "attributes": {}})
            attributes["extraction"] = evidence
            connection.execute("UPDATE products SET attributes=%s WHERE id=%s", (Jsonb(attributes), product_id))
            products.append({"product_id": row["parent_asin"], **evidence})
            if len(products) >= manifest.max_products:
                break
    fields = Counter()
    for product in products:
        fields.update({candidate["field"] for candidate in product["candidates"]})
    return {"extractor_version": VERSION, "source_id": manifest.source_id, "products": products,
            "assessed": len(products), "with_candidates": sum(bool(p["candidates"]) for p in products),
            "with_conflicts": sum(bool(p["conflicting_fields"]) for p in products),
            "product_coverage_by_field": dict(sorted(fields.items())),
            "limitations": "Unreviewed source claims and mentions, not verified facts. No hard filters enabled."}


def main() -> None:
    """Enrich already imported products and save a private candidate report."""
    from agent.config import settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Manifest for an already imported local catalog")
    args = parser.parse_args()
    with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5) as connection:
        report = enrich(connection, args.manifest)
    (args.manifest.parent / "attribute-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "products"}, indent=2))


if __name__ == "__main__":
    main()
