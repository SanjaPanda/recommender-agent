"""Handle explicit color/count follow-ups without losing the product need."""
import re
from .models import Plan

COLORS = ('black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple',
          'orange', 'brown', 'grey', 'gray', 'beige', 'silver', 'gold')
COUNTS = {'a': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
          **{str(i): i for i in range(1, 6)}}


def color_refinement(previous: Plan, message: str) -> Plan | None:
    """Recognize only a complete, unambiguous follow-up, never part of a sentence."""
    if previous.intent != 'recommend':
        return None
    text = message.strip().rstrip('.!').strip().casefold()
    match = re.fullmatch(
        r'(?:(?:find|show|get)(?: me)? )?'
        r'(?:(?P<count>[1-5]|a|one|two|three|four|five)(?: ones?)? )?'
        r'in (?P<color>' + '|'.join(COLORS) + r')', text)
    clear = text in {'any color', 'any colour', 'any color is fine', 'any colour is fine'}
    if not match and not clear:
        return None
    plan = previous.model_copy(deep=True)
    plan.intent, plan.message, plan.product_ids = 'recommend', '', []
    # A single product noun can be enforced using title evidence. Leave complex
    # needs to semantic retrieval rather than guessing a noun from arbitrary prose.
    subject = re.fullmatch(r'(?:(?:looking for|show me|find me|find|show) )?'
                          r'(?:a |an |the )?([a-zA-Z]+)(?: in (?:' + '|'.join(COLORS) + r'))?',
                          previous.query.strip().casefold())
    if subject and subject[1] not in COLORS:
        plan.filters.title_product = subject[1]
    plan.filters.title_color = None if clear else match['color']
    if match and match['count']:
        plan.result_limit = COUNTS[match['count']]
    return plan
