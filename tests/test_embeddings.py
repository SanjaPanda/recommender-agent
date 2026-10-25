"""Pydantic validation for the local Ollama embedding contract."""
import unittest
from unittest.mock import patch

from rag.embed_text import embed_texts


class EmbeddingContractTests(unittest.TestCase):
    def test_cpu_override_is_explicit_and_provider_default_remains_optional(self):
        response = {'model': 'nomic-embed-text', 'embeddings': [[1., 0.]]}
        for gpu in (None, 0):
            with self.subTest(gpu=gpu), patch('rag.embed_text.settings.embedding_num_gpu', gpu), \
                 patch('rag.embed_text.post_json', return_value=response) as call:
                embed_texts('shirt')
                payload = call.call_args.args[1]
                if gpu is None:
                    self.assertNotIn('options', payload)
                else:
                    self.assertEqual(payload['options'], {'num_gpu': 0})

    def test_single_and_batched_inputs_use_validated_vectors(self):
        with patch('rag.embed_text.post_json', return_value={
            'model': 'nomic-embed-text', 'embeddings': [[1.0, 0.0], [0.0, 1.0]],
        }) as call:
            result = embed_texts(['first product', 'second product'])
        self.assertEqual(result, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(call.call_args.args[1]['input'], ['first product', 'second product'])

    def test_empty_and_malformed_responses_fail_explicitly(self):
        with patch('rag.embed_text.post_json') as call:
            with self.assertRaisesRegex(ValueError, 'non-empty'):
                embed_texts(' ')
            call.assert_not_called()
        with patch('rag.embed_text.post_json', return_value={'embeddings': [[float('nan')]]}):
            with self.assertRaisesRegex(RuntimeError, 'failed validation'):
                embed_texts('blue dress')
        with patch('rag.embed_text.post_json', return_value={
            'model': 'nomic-embed-text', 'embeddings': [[1.0]],
        }):
            with self.assertRaisesRegex(RuntimeError, 'returned 1 vectors for 2 inputs'):
                embed_texts(['dress', 'shirt'])
