"""Evidence-oriented curation checks and local catalog quality reporting.

Run ``uv run python -m rag.quality .data/catalog/manifest.json``. This command
reads local records and writes a review report; it never changes the database.
Signals are deterministic review heuristics, not verified product attributes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any


class PlainText(HTMLParser):
    """Extract visible source text without retaining markup or script contents."""

    def __init__(self):
        """Initialize visible text fragments and hidden-element nesting."""
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        """Track hidden elements and separate block-level text fragments."""
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in {"br", "li", "p", "div"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        """Close hidden scope and separate text on element boundaries."""
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        self.parts.append(" ")

    def handle_data(self, data):
        """Collect text only outside script and style elements."""
        if not self.hidden:
            self.parts.append(data)


def clean_text(value: str) -> str:
    """Normalize markup and whitespace while leaving the original source intact."""
    parser = PlainText()
    parser.feed(value)
    return " ".join(unescape("".join(parser.parts)).split())


SIGNALS = {
    "material_mention": r"\b(?:cotton|polyester|spandex|elastane|leather|suede|nylon|wool|rayon|silk|linen|alloy|stainless steel|sterling silver)\b",
    "construction": r"\b(?:zipper|button|buckle|closure|pockets?|straps?|sleeves?|sole|upper|lining|clasp|stitching)\b",
    "fit": r"\b(?:slim fit|regular fit|relaxed fit|athletic fit|stretch|stretchy|adjustable|waist|chest|inseam)\b",
    "care": r"\b(?:machine wash|hand wash|dry clean|do not bleach|line dry|tumble dry)\b",
    "measurement": r"\b\d+(?:\.\d+)?\s*(?:cm|mm|inches|inch|oz)\b",
}
USEFUL_DETAILS = {"material", "fabric type", "fabric", "color", "colour", "size", "style",
                  "pattern", "fit type", "closure type", "sleeve type", "neck style"}
UNKNOWN = {"unknown", "n/a", "na", "none", "not applicable", "not_applicable", "null"}
BOILERPLATE = re.compile(
    r"\b(?:about us|brand story|our mission|reason for being|was founded|is proud of|"
    r"is focused on creating|our line of|we listen to customer feedback|"
    r"our consistent sizing|each piece is put to the test)\b", re.I
)


def assess(row: dict[str, Any], min_words: int = 50) -> dict[str, Any]:
    """Return review evidence and strict-policy failures without inventing facts.

    Strict evidence-v2 requires 50 non-flagged description words, two distinct
    product-signal families in that description, and one useful nonempty details
    field. Features enrich the report but never replace the description gate.
    """
    def texts(value):
        """Normalize a source string or string list without coercing other data."""
        if isinstance(value, str):
            return [value]
        return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []

    description = " ".join(clean_text(part) for part in texts(row.get("description")))
    sentences = re.split(r"(?<=[.!?])\s+|(?i:about us\s*:)", description)
    kept = [part for part in sentences if not BOILERPLATE.search(part)]
    flagged = [part for part in sentences if BOILERPLATE.search(part)]
    # About-us sections are generic company information, not product evidence.
    match = re.search(r"\babout us\s*:", description, re.I)
    if match:
        product_text = description[:match.start()]
        kept = [part for part in re.split(r"(?<=[.!?])\s+", product_text) if not BOILERPLATE.search(part)]
        flagged.append(description[match.start():])
    product_text = " ".join(kept)
    signals = [name for name, pattern in SIGNALS.items() if re.search(pattern, product_text, re.I)]
    details = row.get("details") or {}
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            details = {}
    details = details if isinstance(details, dict) else {}
    useful = [key for key, value in details.items() if key.strip().casefold() in USEFUL_DETAILS
              and isinstance(value, str) and value.strip().casefold() not in UNKNOWN and value.strip()]
    evidence = []
    for index, text in enumerate(texts(row.get("features"))):
        cleaned = clean_text(text)
        for name, pattern in SIGNALS.items():
            match = re.search(pattern, cleaned, re.I)
            if match:
                evidence.append({"signal": name, "source": f"features[{index}]",
                                 "matched_text": match.group(0), "excerpt": cleaned[:240]})
    words = len(re.findall(r"\b\w+\b", product_text))
    failures = []
    if words < min_words:
        failures.append("insufficient_product_description")
    if len(signals) < 2:
        failures.append("weak_description_signals")
    if not useful:
        failures.append("no_useful_structured_attributes")
    return {"product_id": row.get("parent_asin"), "description_words": len(re.findall(r"\b\w+\b", description)),
            "nonflagged_description_words": words, "boilerplate_flagged": bool(flagged),
            "description_signals": signals, "useful_detail_keys": sorted(useful),
            "feature_evidence": evidence, "review_ready": not failures, "failures": failures}


def report_quality(manifest_path: Path) -> dict[str, Any]:
    """Assess a bounded local accepted-file manifest and write a private report."""
    from .source import load_manifest, records

    manifest = load_manifest(manifest_path)
    products = []
    for line in records(manifest_path, manifest):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Expected metadata objects")
        products.append(assess(row, manifest.min_description_words))
        if len(products) >= manifest.max_products:
            break
    failures = Counter(reason for product in products for reason in product["failures"])
    report = {"policy": "evidence-v2", "source_id": manifest.source_id, "assessed": len(products),
              "review_ready": sum(product["review_ready"] for product in products),
              "boilerplate_flagged": sum(product["boilerplate_flagged"] for product in products),
              "with_useful_details": sum(bool(product["useful_detail_keys"]) for product in products),
              "with_feature_evidence": sum(bool(product["feature_evidence"]) for product in products),
              "failure_counts": dict(failures), "products": products,
              "limitations": "Heuristics for review, not human quality labels or verified product attributes. Failure counts overlap."}
    (manifest_path.parent / "quality-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """Write a local report and print aggregate counts without modifying storage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Existing accepted-file manifest")
    args = parser.parse_args()
    report = report_quality(args.manifest)
    print(json.dumps({key: value for key, value in report.items() if key != "products"}, indent=2))


if __name__ == "__main__":
    main()
