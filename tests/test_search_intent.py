"""Open-vocabulary intent extraction and its retrieval/evidence contracts."""
import unittest
import json
from unittest.mock import MagicMock, patch

from agent.llm import ModelError
from agent.models import Plan, SearchIntent
from agent.models import SearchIntentRequest
from agent.search_intent import extract_search_intent
from agent.workflow import _retrieve
from agent.product_evidence import listing_checks
from rag.phrases import phrase_in_title
from rag.retrieval import retrieve_products


class SearchIntentTests(unittest.TestCase):
    def setUp(self):
        for name in ('text_model_version', 'clip_model_version'):
            version = patch('rag.retrieval.' + name, return_value='test-revision')
            version.start()
            self.addCleanup(version.stop)

    def test_extraction_releases_local_model_before_embedding_work(self):
        from agent.llm import complete
        response = MagicMock()
        response.read.return_value = json.dumps({'message': {'content': SearchIntent(product_phrase='dress').model_dump_json()}}).encode()
        with patch('agent.llm.urllib.request.urlopen') as request:
            request.return_value.__enter__.return_value = response
            complete(SearchIntent, 'Extract', SearchIntentRequest(query='dress'))
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload['keep_alive'], 0)

    def test_llm_extracts_compound_and_grounded_attributes(self):
        intent = SearchIntent(product_phrase='bow tie', search_terms=['blue bowtie'],
                              attributes=[dict(name='color', value='blue', evidence='blue')])
        with patch('agent.search_intent.complete', return_value=intent) as model:
            self.assertEqual(extract_search_intent('Find a blue bow tie'), intent)
        self.assertEqual(model.call_args.args[0], SearchIntent)
        self.assertEqual(model.call_args.args[2].query, 'Find a blue bow tie')

    def test_invented_phrase_or_attribute_evidence_is_rejected(self):
        for intent in (SearchIntent(product_phrase='necktie'),
                       SearchIntent(product_phrase='bow tie', attributes=[dict(name='material', value='silk', evidence='silk')]),
                       SearchIntent(product_phrase='bow tie', attributes=[dict(name='color', value='red', evidence='blue')])):
            with self.subTest(intent=intent), patch('agent.search_intent.complete', return_value=intent):
                with self.assertRaises(ModelError):
                    extract_search_intent('blue bow tie')

    def test_request_product_evidence_accepts_regular_inflections(self):
        for query, phrase in (('Dresses', 'dress'), ('dress', 'dresses'),
                              ('blue bow ties', 'bow tie'), ('ankle boots', 'ankle boot'),
                              ('silver ear cuffs', 'ear cuff')):
            with self.subTest(query=query), patch('agent.search_intent.complete',
                                                  return_value=SearchIntent(product_phrase=phrase)):
                self.assertEqual(extract_search_intent(query).product_phrase, phrase)

    def test_inflection_grounding_does_not_allow_missing_words_or_attributes(self):
        for query, intent in (
            ('black cotton dress', SearchIntent(product_phrase='black dress')),
            ('dresser', SearchIntent(product_phrase='dress')),
            ('blue bow ties', SearchIntent(product_phrase='necktie')),
            ('Dresses', SearchIntent(product_phrase='dress', attributes=[
                dict(name='color', value='blue', evidence='blue')])),
        ):
            with self.subTest(query=query), patch('agent.search_intent.complete', return_value=intent):
                with self.assertRaises(ModelError):
                    extract_search_intent(query)

    def test_normalized_plural_intent_reaches_dress_retrieval(self):
        with patch('agent.search_intent.complete', return_value=SearchIntent(product_phrase='dress')):
            intent = extract_search_intent('Dresses')
        candidate = {'id': 'dress', 'title': 'Summer Dress', 'text_similarity': .8}
        with patch('agent.workflow.extract_search_intent', return_value=intent), \
             patch('agent.workflow.retrieve_products', return_value=[candidate]) as retrieve:
            result = _retrieve({'plan': Plan(intent='recommend', query='Dresses').model_dump()})
        self.assertEqual(result['result_kind'], 'matches')
        self.assertEqual(result['candidates'], [candidate])
        self.assertEqual(retrieve.call_args.kwargs['product_phrase'], 'dress')

    def test_unknown_product_phrases_need_no_registry(self):
        for phrase, title in (('sling bag', 'Leather Sling Bags'),
                              ('ear cuff', 'Silver Ear-Cuffs'),
                              ('mary jane shoes', 'Mary Jane Shoes'),
                              ('bow tie', 'Blue Bowtie')):
            with self.subTest(phrase=phrase):
                self.assertTrue(phrase_in_title(title, phrase))
                self.assertFalse(phrase_in_title('Shirt with ' + title, phrase))
                check = listing_checks(title, {}, phrase, {}, phrase)[-1]
                self.assertEqual(check['status'], 'match')
        self.assertFalse(phrase_in_title('Blue Tie', 'bow tie'))
        self.assertFalse(phrase_in_title('Blue Butterfly Print', 'tie'))

    def test_request_interpretation_reaches_search_and_is_reused(self):
        intent = SearchIntent(product_phrase='ear cuff', search_terms=['silver ear cuff'])
        plan = Plan(intent='recommend', query='silver ear cuff')
        candidate = {'id': 'cuff', 'title': 'Silver Ear Cuff', 'text_similarity': .8}
        state = {'plan': plan.model_dump()}
        with patch('agent.workflow.extract_search_intent', return_value=intent) as extract, \
             patch('agent.workflow.retrieve_products', return_value=[candidate]) as retrieve:
            first = _retrieve(state)
            second = _retrieve({**state, **first})
        extract.assert_called_once_with('silver ear cuff')
        self.assertEqual(retrieve.call_args.kwargs['product_phrase'], 'ear cuff')
        self.assertEqual(retrieve.call_args.kwargs['search_terms'], ['silver ear cuff'])
        self.assertEqual(second['candidates'], [candidate])
        self.assertEqual(second['search_intent'], intent.model_dump())

    def test_changed_request_reextracts_and_failure_does_not_search(self):
        state = {'plan': Plan(intent='recommend', query='sling bag').model_dump(),
                 'intent_query': 'ear cuff', 'search_intent': SearchIntent(product_phrase='ear cuff').model_dump()}
        with patch('agent.workflow.extract_search_intent', side_effect=ModelError('offline')) as extract, \
             patch('agent.workflow.retrieve_products') as retrieve:
            with self.assertRaises(ModelError):
                _retrieve(state)
        extract.assert_called_once_with('sling bag')
        retrieve.assert_not_called()

    def test_category_browse_and_image_only_skip_intent_extraction(self):
        with patch('agent.workflow.extract_search_intent') as extract, \
             patch('agent.workflow.retrieve_products', return_value=[]):
            for selected in ([], ['seed']):
                _retrieve({'plan': Plan(intent='recommend').model_dump(), 'selected_product_ids': selected})
        extract.assert_not_called()

    def test_saved_interpretation_is_exposed_in_api_with_phrase_check(self):
        from agent.api import _response
        intent = SearchIntent(product_phrase='ear cuff', search_terms=['silver earcuff'])
        item = {'id': 'cuff', 'title': 'Silver Ear Cuff'}
        with patch('agent.api.psycopg.connect') as connect, patch('agent.api._find_image_path', return_value=None):
            connect.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [('cuff', '', [], {}, {}, 'Silver Ear Cuff', None, 'Apparel & Accessories', None, None, None)]
            response = _response({'recommendations': [item], 'search_intent': intent.model_dump(),
                                  'query_context': 'silver ear cuff'}, 'test', 'Results', create_surface=True)
        self.assertEqual(response['search_intent'], intent.model_dump())
        self.assertEqual(response['products'][0]['match_checks'][-1]['label'], 'Ear cuff')
        self.assertEqual(response['products'][0]['match_checks'][-1]['status'], 'match')

    def test_dynamic_phrase_applies_before_limits_to_all_rankers(self):
        connection = MagicMock()
        connection.execute.return_value.fetchall.return_value = [
            ('wrong', 'Silver Bracelet', 'ear cuff', None, 'Apparel & Accessories', None, .99),
            ('correct', 'Silver Ear Cuffs', '', None, 'Apparel & Accessories', None, .6)]
        with patch('rag.retrieval.psycopg.connect') as connect, patch('rag.retrieval.register_vector'), \
             patch('rag.retrieval._search_bm25', return_value=[]) as bm25, \
             patch('rag.retrieval.embed_texts', return_value=[[1., 0.]]), \
             patch('rag.retrieval.clip_text_embedding', return_value=[1., 0.]):
            connect.return_value.__enter__.return_value = connection
            result = retrieve_products(query_context='silver ear cuff', product_phrase='ear cuff',
                                       search_terms=['silver earcuff'], limit=1)
        self.assertEqual([item['id'] for item in result], ['correct'])
        self.assertIn('silver earcuff', bm25.call_args.kwargs['query'])
        self.assertIn('regexp_split_to_array', bm25.call_args.kwargs['category_clause'])
        for call in connection.execute.call_args_list:
            self.assertIn('regexp_split_to_array', call.args[0])
            self.assertIn('Apparel & Accessories', call.args[1])

    def test_search_terms_cannot_broaden_required_product_phrase(self):
        plan = Plan(intent='recommend', query='blue bow tie')
        intent = SearchIntent(product_phrase='bow tie', search_terms=['blue tie'])
        product = {'id': 'tie', 'title': 'Blue Tie', 'text_similarity': .99, 'image_similarity': .99}
        with patch('agent.workflow.extract_search_intent', return_value=intent), \
             patch('agent.workflow.retrieve_products', return_value=[product]):
            result = _retrieve({'plan': plan.model_dump()})
        self.assertEqual(result['result_kind'], 'none')
        self.assertEqual(result['candidates'], [])
