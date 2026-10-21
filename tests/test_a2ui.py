"""Regression for reset/repeated discovery leaving stale gallery data."""
import unittest
from agent.a2ui import product_gallery_operations


class GalleryLifecycleTests(unittest.TestCase):
    def test_reset_and_new_search_replace_existing_surface_before_updates(self):
        surfaces = {}

        def render(products, create=True):
            operations = product_gallery_operations(products, heading='Products',
                surface_id='session-gallery', create_surface=create)
            # The renderer rejects duplicate creation and stops processing the batch.
            for message in operations:
                if 'deleteSurface' in message:
                    surfaces.pop(message['deleteSurface']['surfaceId'], None)
                elif 'createSurface' in message:
                    sid = message['createSurface']['surfaceId']
                    self.assertNotIn(sid, surfaces, 'Duplicate creation prevents gallery updates')
                    surfaces[sid] = {}
                elif 'updateDataModel' in message:
                    update = message['updateDataModel']
                    self.assertIn(update['surfaceId'], surfaces)
                    surfaces[update['surfaceId']] = update['value']
            return surfaces['session-gallery']['products']

        old = [{'id': str(i), 'title': 'Old product'} for i in range(5)]
        self.assertEqual(render(old), old)
        self.assertEqual(render(old), old)  # Cart/session response on an existing surface.
        self.assertEqual(render([]), [])  # Start a new search.
        fresh = [{'id': 'new', 'title': 'New product'}]
        self.assertEqual(render(fresh), fresh)
        refined = [{'id': 'refined', 'title': 'Refined product'}]
        self.assertEqual(render(refined, create=False), refined)

    def test_refinement_only_updates_the_current_surface(self):
        operations = product_gallery_operations([], heading='Products',
            surface_id='session-gallery', create_surface=False)
        self.assertFalse(any('createSurface' in op or 'deleteSurface' in op for op in operations))
        self.assertEqual(operations[-1]['updateDataModel']['value']['products'], [])
