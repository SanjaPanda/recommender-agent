"""Bounded downloader tests use authored records and mocked HTTP only."""
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import psycopg

from agent.config import settings
from rag.catalog import Manifest, ingest
from rag.download_catalog import download_catalog, fetch_image, ordered_images, validate_url, IMAGE_HOSTS
from test_catalog import fixture


class Response(BytesIO):
    """Minimal streamed HTTP response fixture with caller-defined headers."""

    def __init__(self, body, content_type="application/octet-stream"):
        super().__init__(body)
        self.headers = {"Content-Type": content_type, "Content-Length": str(len(body))}


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.row, config = fixture(self.root)
        self.image = (self.root / "shirt.png").read_bytes()
        self.url = "https://m.media-amazon.com/images/I/authored.png"
        self.row["images"] = [{"variant": "MAIN", "large": self.url}]
        config.update(rights_status="permitted", rights_evidence="Authored HTTP test fixture, no remote acquisition",
                      revision="a" * 40, images={}, max_products=1,
                      source_url="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/"
                      + "a" * 40 + "/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl")
        self.plan = Manifest(**config)
        self.output = self.root / "output"
        self.sleep = patch("rag.download_catalog.time.sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def run_acquisition(self, body=None, image_response=None):
        body = body if body is not None else (json.dumps(self.row) + "\n").encode()
        with patch("rag.download_catalog.open_remote", side_effect=[Response(body), image_response or Response(self.image, "image/png")]) as remote:
            report = download_catalog(self.plan, self.output)
        return report, remote

    def test_acquisition_preserves_source_lines_and_reports_quality(self):
        self.plan = self.plan.model_copy(update={"rights_status": "personal_use"})
        short = {**self.row, "description": []}
        body = (json.dumps(short) + "\n" + json.dumps(self.row) + "\n").encode()
        report, remote = self.run_acquisition(body)
        manifest = Manifest.model_validate_json((self.output / "manifest.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(report["rejected"], {"short_description": 1})
        self.assertEqual(manifest.source_lines, {"AUTHORED1": 2})
        self.assertEqual(manifest.rights_status, "personal_use")
        self.assertEqual(report["samples"][0]["attribute_keys"], ["Brand", "Material"])
        self.assertEqual(report["image_bytes"], len(self.image))
        self.assertEqual(remote.call_count, 2)
        self.assertEqual((self.output / "records.jsonl").read_bytes(), (json.dumps(self.row) + "\n").encode())
        with self.assertRaises(FileExistsError):
            download_catalog(self.plan, self.output)

    def test_metadata_budget_stops_without_image_requests(self):
        self.plan = self.plan.model_copy(update={"max_bytes": 10})
        report, remote = self.run_acquisition()
        self.assertEqual(report["metadata_bytes"], 10)
        self.assertEqual(report["accepted"], 0)
        self.assertEqual(report["stop_reason"], "incomplete_line_or_byte_limit")
        self.assertEqual(remote.call_count, 1)

    def test_oversized_record_is_drained_within_budget_and_later_records_are_read(self):
        body = b"x" * 1_000_010 + b"\n" + (json.dumps(self.row) + "\n").encode()
        report, _ = self.run_acquisition(body)
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(report["rejected"], {"oversized_record": 1})
        self.assertEqual(report["metadata_bytes"], len(body))
        manifest = Manifest.model_validate_json((self.output / "manifest.json").read_text())
        self.assertEqual(manifest.source_lines, {"AUTHORED1": 2})

    def test_expansion_reuses_original_records_images_and_source_lines(self):
        self.run_acquisition()
        expanded = self.root / "expanded"
        plan = self.plan.model_copy(update={"max_products": 2})
        # Exhaustion preserves the seed without downloading its image again.
        body = (json.dumps(self.row) + "\n").encode()
        with patch("rag.download_catalog.open_remote", return_value=Response(body)) as remote:
            report = download_catalog(plan, expanded, seed_root=self.output)
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(report["seeded"], 1)
        self.assertEqual(report["image_requests"], 0)
        self.assertEqual(remote.call_count, 1)
        self.assertEqual((expanded / "records.jsonl").read_bytes(), (self.output / "records.jsonl").read_bytes())
        manifest = Manifest.model_validate_json((expanded / "manifest.json").read_text())
        self.assertEqual(manifest.source_lines, {"AUTHORED1": 1})
        self.assertEqual((expanded / manifest.images["AUTHORED1"]["local_path"]).read_bytes(), self.image)

    def test_rate_limit_stops_instead_of_repeated_downloads(self):
        body = (json.dumps(self.row) + "\n").encode() * 2
        with patch("rag.download_catalog.open_remote", side_effect=[Response(body), HTTPError(self.url, 429, "limited", {}, None)]) as remote:
            report = download_catalog(self.plan, self.output)
        self.assertEqual(report["stop_reason"], "rate_limited")
        self.assertEqual(report["image_failures"], {"http_429": 1})
        self.assertEqual(remote.call_count, 2)

    def test_image_limits_and_mime_are_checked_before_body_read(self):
        budget = {"image_bytes": 0, "image_requests": 0}
        for response in (Response(b"html", "text/html"), Response(self.image, "image/png")):
            if response.headers["Content-Type"] == "image/png":
                response.headers["Content-Length"] = "10000001"
            with patch("rag.download_catalog.open_remote", return_value=response):
                with self.assertRaises(ValueError):
                    fetch_image(self.url, budget, float("inf"))
        self.assertEqual(budget["image_bytes"], 0)

    def test_corrupt_image_is_reported_and_not_kept(self):
        report, _ = self.run_acquisition(image_response=Response(b"not an image", "image/png"))
        self.assertEqual(report["accepted"], 0)
        self.assertEqual(report["rejected"], {"no_valid_image": 1})
        self.assertEqual(list((self.output / "images").iterdir()), [])

    def test_unpinned_or_wrong_source_does_not_open_network(self):
        with patch("rag.download_catalog.open_remote") as remote:
            with self.assertRaises(ValueError):
                download_catalog(self.plan.model_copy(update={"revision": "main"}), self.output)
            remote.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_public_https_allowlist_and_dns_are_enforced(self):
        for url in ("http://m.media-amazon.com/image", "https://127.0.0.1/image", "https://evil.example/image",
                    "https://user:password@m.media-amazon.com/image", "https://m.media-amazon.com:444/image"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_url(url, IMAGE_HOSTS)
        with patch("rag.download_catalog.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("127.0.0.1", 443))]):
            with self.assertRaisesRegex(ValueError, "nonpublic_address"):
                validate_url(self.url, IMAGE_HOSTS)
        with patch("rag.download_catalog.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("8.8.8.8", 443))]):
            validate_url(self.url, IMAGE_HOSTS)

    def test_image_candidates_prioritize_main_and_support_transposed_arrays(self):
        secondary = "https://m.media-amazon.com/secondary.jpg"
        self.assertEqual(ordered_images({"variant": ["PT01", "MAIN"], "large": [secondary, self.url]}), [self.url, secondary])

    def test_expired_deadline_never_opens_an_image_connection(self):
        with patch("rag.download_catalog.open_remote") as remote:
            with self.assertRaisesRegex(ValueError, "wall_limit"):
                fetch_image(self.url, {"image_bytes": 0, "image_requests": 0}, 0)
            remote.assert_not_called()

    def test_metadata_html_is_reported_as_failure(self):
        with patch("rag.download_catalog.open_remote", return_value=Response(b"login page", "text/html")):
            with self.assertRaisesRegex(ValueError, "metadata_mime"):
                download_catalog(self.plan, self.output)
        self.assertEqual(json.loads((self.output / "pilot-report.json").read_text())["status"], "failed")

    def test_metadata_failure_leaves_failed_report_without_import_manifest(self):
        with patch("rag.download_catalog.open_remote", side_effect=HTTPError(self.plan.source_url, 503, "unavailable", {}, None)):
            with self.assertRaises(HTTPError):
                download_catalog(self.plan, self.output)
        report = json.loads((self.output / "pilot-report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["http_status"], 503)
        self.assertFalse((self.output / "manifest.json").exists())

    @unittest.skipUnless(os.environ.get("RUN_DATABASE_TESTS") == "1", "Opt-in PostgreSQL integration test")
    def test_downloaded_fixture_import_preserves_original_line(self):
        short = {**self.row, "description": []}
        self.run_acquisition((json.dumps(short) + "\n" + json.dumps(self.row) + "\n").encode())
        with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5) as connection:
            try:
                report = ingest(connection, self.output / "manifest.json")
                self.assertEqual(report["accepted"], 1)
                self.assertEqual(connection.execute("SELECT source_line FROM products WHERE id=%s",
                                 (self.plan.source_id + ":AUTHORED1",)).fetchone()[0], 2)
            finally:
                connection.rollback()
