"""Offline pipeline regressions use authored assets and deterministic encoders."""
from contextlib import contextmanager, nullcontext, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent.tools.embeddings import document_text
from rag import CATALOG_WRITE_LOCK, embed_images, embed_text
from rag.attributes import extract
from rag.catalog import normalize, normalize_metadata
from rag.source import Manifest, load_manifest, records
from test_catalog import fixture


class BatchConnection:
    """Record durable batches and model transaction rollback on a job connection."""

    def __init__(self, products):
        self.products = products
        self.writes = []
        self.queries = []
        self.commits = 0

    @contextmanager
    def transaction(self):
        previous = list(self.writes)
        try:
            yield
        except Exception:
            self.writes = previous
            raise
        else:
            self.commits += 1

    def execute(self, query, params):
        self.queries.append((query, params))
        if query.lstrip().startswith('INSERT'):
            self.writes.append(params)
        return Mock(fetchall=lambda: self.products)


class TextJobTests(unittest.TestCase):
    def run_job(self, products, vectors, *arguments):
        connection = BatchConnection(products)
        self.enterContext(patch.object(embed_text.psycopg, 'connect')).return_value.__enter__.return_value = connection
        self.enterContext(patch.object(embed_text, 'register_vector'))
        self.enterContext(patch.object(embed_text, 'model_digest', return_value='fixed-revision'))
        encoder = self.enterContext(patch.object(embed_text, 'embed_texts', side_effect=vectors))
        self.enterContext(patch('sys.argv', ['embed_text', *arguments]))
        self.enterContext(redirect_stdout(StringIO()))
        return connection, encoder

    def test_batches_preserve_encoder_input_raw_vectors_and_revision(self):
        products = [('a', 'Shirt', 'Cotton', ['Soft']), ('b', 'Dress', 'Silk', []),
                    ('c', 'Hat', 'Wool', ['Warm'])]
        connection, encoder = self.run_job(products, [[[3., 4.], [6., 8.]], [[9., 12.]]],
                                           '--batch-size', '2', '--limit', '3')
        embed_text.main()
        self.assertEqual(encoder.call_args_list[0].args[0],
                         [document_text(*product[1:]) for product in products[:2]])
        self.assertEqual(connection.commits, 2)
        self.assertEqual([row[0] for row in connection.writes], ['a', 'b', 'c'])
        for row, expected in zip(connection.writes, [[3., 4.], [6., 8.], [9., 12.]], strict=True):
            self.assertEqual(row[1:4], ('text', embed_text.MODEL, 'fixed-revision'))
            self.assertEqual(row[4].to_list(), expected)
        self.assertEqual(connection.queries[0][1], (CATALOG_WRITE_LOCK,))
        self.assertEqual(connection.queries[1][1], ['text', embed_text.MODEL, 'fixed-revision', 3])

    def test_force_includes_ready_products_and_no_pending_skips_encoder(self):
        connection, encoder = self.run_job([], [], '--force', '--limit', '1')
        embed_text.main()
        self.assertEqual(connection.queries[1][1], [1])
        self.assertNotIn('NOT EXISTS', connection.queries[1][0])
        encoder.assert_not_called()
        self.assertEqual(connection.commits, 0)

    def test_cardinality_error_rolls_back_batch(self):
        connection, _ = self.run_job([('a', 'Shirt', '', []), ('b', 'Hat', '', [])], [[[1., 2.]]])
        with self.assertRaises(ValueError):
            embed_text.main()
        self.assertEqual(connection.writes, [])
        self.assertEqual(connection.commits, 0)

    def test_inconsistent_dimensions_keep_only_previously_committed_batch(self):
        connection, _ = self.run_job([('a', 'Shirt', '', []), ('b', 'Hat', '', [])],
                                      [[[1., 2.]], [[1., 2., 3.]]], '--batch-size', '1')
        with self.assertRaisesRegex(RuntimeError, 'inconsistent embedding dimensions'):
            embed_text.main()
        self.assertEqual([row[0] for row in connection.writes], ['a'])
        self.assertEqual(connection.commits, 1)


class ImageJobTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        catalog = self.root / '.data' / 'catalog'
        catalog.mkdir(parents=True)
        fixture(catalog)
        (catalog / 'bad.png').write_bytes(b'not an image')
        self.connection = BatchConnection([
            ('valid', {'local_path': 'shirt.png'}),
            ('unsafe', {'local_path': '../escape.png'}),
            ('missing', {}), ('bad', {'local_path': 'bad.png'}),
        ])
        self.enterContext(patch.object(embed_images.psycopg, 'connect')).return_value.__enter__.return_value = self.connection
        self.enterContext(patch.object(embed_images, 'register_vector'))
        self.enterContext(patch.object(embed_images, 'ROOT', self.root))
        self.enterContext(patch('sys.argv', ['embed_images']))
        self.output = StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.model = Mock()
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model
        self.model.config = SimpleNamespace(_commit_hash='image-revision', projection_dim=2)
        self.model.get_image_features.return_value.cpu.return_value.tolist.return_value = [[3., 4.]]
        self.processor = Mock(return_value={})
        transformers = SimpleNamespace(CLIPModel=Mock(from_pretrained=Mock(return_value=self.model)),
                                       CLIPProcessor=Mock(from_pretrained=Mock(return_value=self.processor)))
        self.enterContext(patch.dict('sys.modules', {
            'torch': SimpleNamespace(inference_mode=nullcontext), 'transformers': transformers,
        }))

    def test_normalizes_images_skips_unsafe_assets_and_closes_decoded_images(self):
        embed_images.main()
        self.assertEqual(len(self.connection.writes), 1)
        row = self.connection.writes[0]
        self.assertEqual(row[:4], ('valid', 'image', embed_images.MODEL, 'image-revision'))
        for actual, expected in zip(row[4].to_list(), [.6, .8], strict=True):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(self.connection.commits, 1)
        report = json.loads(self.output.getvalue().splitlines()[-1])
        self.assertEqual((report['embedded'], report['missing_image_files'], report['unreadable_images']), (1, 2, 1))
        decoded = self.processor.call_args.kwargs['images'][0]
        with self.assertRaises(ValueError):
            decoded.getpixel((0, 0))

    def test_encoder_failure_writes_nothing_and_closes_decoded_images(self):
        self.model.get_image_features.side_effect = RuntimeError('encoder failed')
        with self.assertRaisesRegex(RuntimeError, 'encoder failed'):
            embed_images.main()
        self.assertEqual(self.connection.writes, [])
        decoded = self.processor.call_args.kwargs['images'][0]
        with self.assertRaises(ValueError):
            decoded.getpixel((0, 0))

    def test_unversioned_model_is_rejected_before_catalog_access(self):
        self.model.config._commit_hash = None
        with self.assertRaisesRegex(RuntimeError, 'immutable revision'):
            embed_images.main()
        self.assertEqual(self.connection.queries, [])


class SourcePipelineTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        self.row, config = fixture(self.root)
        self.manifest = Manifest(**config)
        self.path = self.root / 'manifest.json'

    def test_import_preserves_extraction_and_acquisition_needs_no_extractor(self):
        with patch('rag.attributes.extract', side_effect=AssertionError('unnecessary extraction')):
            product = normalize_metadata(self.row)
        self.assertEqual(product['attributes']['attributes'], self.row['details'])
        imported = normalize(self.row, self.manifest, self.root)
        self.assertEqual(imported['attributes']['extraction'], extract(self.row))
        self.assertEqual(imported['raw_source']['description'], self.row['description'])

    def test_original_bytes_including_whitespace_and_newlines_are_preserved(self):
        original = json.dumps(self.row, ensure_ascii=False, indent=None).encode() + b'  \r\n'
        (self.root / self.manifest.records).write_bytes(original)
        self.assertEqual(list(records(self.path, self.manifest)), [original])
        # EOF at the exact byte budget is valid, including an unterminated final line.
        original = original.rstrip(b'\r\n')
        (self.root / self.manifest.records).write_bytes(original)
        bounded = self.manifest.model_copy(update={'max_bytes': len(original)})
        self.assertEqual(list(records(self.path, bounded)), [original])

    def test_oversized_record_and_budget_fail_without_partial_yield(self):
        for content, budget in [(b'x' * 1_000_001, 2_000_000), (b'{}\n', 2)]:
            with self.subTest(size=len(content), budget=budget):
                (self.root / self.manifest.records).write_bytes(content)
                with self.assertRaisesRegex(ValueError, 'manifest budget'):
                    list(records(self.path, self.manifest.model_copy(update={'max_bytes': budget})))

    def test_record_limit_and_path_escape_remain_enforced(self):
        (self.root / self.manifest.records).write_bytes(b'{}\n{}\n')
        self.assertEqual(list(records(self.path, self.manifest.model_copy(update={'max_lines': 1}))), [b'{}\n'])
        with self.assertRaisesRegex(ValueError, 'asset_path'):
            list(records(self.path, self.manifest.model_copy(update={'records': '../outside.jsonl'})))

    def test_manifest_size_is_bounded_before_json_parsing(self):
        self.path.write_bytes(b' ' * 2_000_001)
        with self.assertRaisesRegex(ValueError, 'manifest_byte_limit'):
            load_manifest(self.path)
