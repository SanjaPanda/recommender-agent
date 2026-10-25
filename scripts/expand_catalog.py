"""Extend the active private catalog to 2,000 products, preserving existing IDs.

Run ``uv run python scripts/expand_catalog.py`` from the repository root after
docs/SETUP.md. Optional ``--staged .data/expansions/<run>`` reuses acquisition.
Writes private assets, backups, a recovery journal and PostgreSQL products.
Rerunning finishes any pending promotion before starting new acquisition.
"""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import shutil

import psycopg

from agent.config import settings
from rag.catalog import ROOT, Manifest, ingest
from rag.download_catalog import download_catalog
from rag import CATALOG_WRITE_LOCK

PROMOTED_FILES = ('records.jsonl', 'manifest.json', 'pilot-report.json')


def _write_json(path: Path, value: dict) -> None:
    """Durably prepare JSON and atomically replace its destination."""
    temporary = path.with_name(path.name + '.next')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def promote_catalog(active: Path, stage: Path | None = None) -> dict:
    """Import and publish a verified stage, resuming interrupted promotions.

    Args:
        active: Existing private catalog directory.
        stage: Acquired expansion directory; omitted when recovering a journal.

    Returns:
        The idempotent import report.

    Raises:
        ValueError: If recovery paths or staged metadata changed.
        RuntimeError: If staged import counts disagree with acquisition.
        OSError: If backup/preparation/publication fails. A journal written before
            the database import remains available for a subsequent retry.

    Holds the existing maintenance advisory lock through import/publication.
    Database commits and file replacement are not one atomic operation; the
    journal makes their interruption recoverable without reacquisition.
    """
    with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5,
                         autocommit=True) as connection:
        connection.execute('SELECT pg_advisory_lock(%s)', (CATALOG_WRITE_LOCK,))
        journal = active / '.promotion.json'
        if journal.exists():
            pending = json.loads(journal.read_text(encoding='utf-8'))
            recorded = (ROOT / pending['stage']).resolve()
            if stage is not None and stage.resolve() != recorded:
                raise ValueError('Another promotion is pending; rerun without --staged to recover it.')
            stage = recorded
        else:
            if stage is None:
                raise ValueError('No pending promotion to recover.')
            stage = stage.resolve()
            pending = {'stage': stage.relative_to(ROOT.resolve()).as_posix(), 'hashes': {}}
        stage.relative_to((ROOT / '.data' / 'expansions').resolve())
        hashes = {name: hashlib.sha256((stage / name).read_bytes()).hexdigest() for name in PROMOTED_FILES}
        if journal.exists() and pending['hashes'] != hashes:
            raise ValueError('Pending promotion metadata changed; restore its staged files before retrying.')
        report = json.loads((stage / 'pilot-report.json').read_text(encoding='utf-8'))
        if not journal.exists():
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            backup = ROOT / '.data' / 'catalog-backups' / stamp
            backup.mkdir(parents=True)
            for name in (*PROMOTED_FILES, 'ingestion-report.json'):
                if (active / name).exists():
                    shutil.copy2(active / name, backup / name)
            pending['hashes'] = hashes
        # Verify and copy assets before any database mutation. Existing images
        # stay available; normalize()/ingest() verify staged asset contents.
        for image in (stage / 'images').iterdir():
            target = active / 'images' / image.name
            if not target.exists():
                shutil.copy2(image, target)
        for name in PROMOTED_FILES:
            temporary = active / (name + '.next')
            shutil.copy2(stage / name, temporary)
            with temporary.open('r+b') as stream:
                os.fsync(stream.fileno())
        if not journal.exists():
            _write_json(journal, pending)
        with connection.transaction():
            imported = ingest(connection, stage / 'manifest.json')
            if imported['accepted'] != report['accepted'] or imported['rejected']:
                raise RuntimeError('Staged import rejected products; transaction rolled back')
        for name in PROMOTED_FILES:
            (active / (name + '.next')).replace(active / name)
        _write_json(active / 'ingestion-report.json', imported)
        journal.unlink()
        return imported


def main() -> None:
    """Acquire up to 2,000 products or recover/promote an existing staged run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", type=Path, help="Import a previously acquired .data/expansions directory without downloading again")
    args = parser.parse_args()
    active = ROOT / ".data" / "catalog"
    if (active / '.promotion.json').exists():
        print(json.dumps(promote_catalog(active, args.staged), indent=2), flush=True)
        print('Recovered promotion. Run the existing text/image embedding commands next.', flush=True)
        return
    seed = Manifest.model_validate_json((active / "manifest.json").read_text(encoding="utf-8"))
    if len(seed.images) >= 2000 and args.staged is None:
        print("Catalog already contains 2,000 images; no acquisition needed.")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.staged is not None:
        stage = args.staged.resolve()
        stage.relative_to((ROOT / ".data" / "expansions").resolve())
        report = json.loads((stage / "pilot-report.json").read_text(encoding="utf-8"))
    else:
        stage = ROOT / ".data" / "expansions" / stamp
        plan = seed.model_copy(update={"images": {}, "source_lines": {}, "max_products": 2000})
        print(f"Staging acquisition in {stage}; preserving {len(seed.images)} existing products", flush=True)
        report = download_catalog(plan, stage, seed_root=active)
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2), flush=True)
    if report["accepted"] <= len(seed.images):
        raise RuntimeError("No new products acquired; active catalog unchanged")
    imported = promote_catalog(active, stage)
    print(json.dumps(imported, indent=2), flush=True)
    print("Next: python -m rag.embed_text; python -m rag.embed_images", flush=True)


if __name__ == "__main__":
    main()
