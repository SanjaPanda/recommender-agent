"""Reject malformed model edits before they reach shopping state."""
import unittest
from pydantic import ValidationError
from agent.models import TurnEdits


class EditContractTests(unittest.TestCase):
    def test_add_and_replace_require_nonblank_values(self):
        for operation in ('add', 'replace'):
            for fields in ({}, {'value': ''}, {'value': '   '}):
                with self.subTest(operation=operation, fields=fields):
                    with self.assertRaises(ValidationError):
                        TurnEdits(intent='recommend', edits=[dict(
                            field='preference', operation=operation,
                            evidence='prefer lightweight', **fields)])

    def test_removal_cannot_supply_a_new_constraint(self):
        with self.assertRaises(ValidationError):
            TurnEdits(intent='recommend', edits=[dict(field='preference',
                operation='remove', target='not blue', value='blue', evidence='blue is fine')])

    def test_valid_mixed_edits_keep_the_existing_wire_format(self):
        turn = TurnEdits(intent='recommend', edits=[
            dict(field='brand', operation='replace', value='Trendsi', evidence='Trendsi'),
            dict(field='preference', operation='remove', target='not blue', evidence='blue is fine'),
            dict(field='preference', operation='add', value='lightweight',
                 strength='soft', evidence='prefer lightweight')])
        self.assertEqual([e.value for e in turn.edits], ['Trendsi', '', 'lightweight'])
        self.assertEqual(turn.edits[-1].strength, 'soft')
        self.assertEqual(TurnEdits.model_validate_json(turn.model_dump_json()), turn)
