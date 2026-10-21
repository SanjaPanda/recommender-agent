"""Apply validated, atomic edits while preserving all untouched context."""
import re

from .models import Plan, Preference, TurnEdits


class AmbiguousEdit(ValueError):
    """A proposed edit cannot safely be applied to this snapshot."""


def normalized(value: str) -> str:
    return ' '.join(value.casefold().split())


def apply_edits(previous: Plan, turn: TurnEdits, message: str) -> Plan:
    """Validate on a copy; any conflict leaves the caller's plan untouched."""
    result = previous.model_copy(deep=True)
    result.intent, result.message, result.product_ids = turn.intent, turn.message, turn.product_ids
    if turn.intent != 'recommend':
        if turn.edits:
            raise AmbiguousEdit('Shopping edits need a recommendation intent.')
        return result
    if not turn.edits:
        raise AmbiguousEdit('No explicit change was identified; ask before repeating the old search.')
    touched = set()
    removed = set()
    for edit in turn.edits:
        evidence = normalized(edit.evidence)
        if not evidence or evidence not in normalized(message):
            raise AmbiguousEdit('An edit lacks evidence in the current message.')
        key = (edit.field, normalized(edit.target or edit.value) if edit.field in {'preference', 'category'} else '')
        if key in touched:
            raise AmbiguousEdit('Multiple edits address the same constraint.')
        touched.add(key)
        value = edit.value.strip()
        if (edit.operation == 'remove' and value) or (edit.operation != 'remove' and not value):
            raise AmbiguousEdit('An edit has an inconsistent value.')
        if edit.field == 'preference':
            if edit.operation != 'remove' and re.search(r'\bis (fine|ok|okay|allowed|acceptable)\b', evidence):
                raise AmbiguousEdit('Permission to include an attribute is not a requirement for it.')
            matches = [i for i, pref in enumerate(result.preferences)
                       if normalized(pref.requirement) == normalized(edit.target)]
            if edit.operation == 'add':
                if edit.target:
                    raise AmbiguousEdit('Adding a preference must not target an existing one.')
            else:
                if len(matches) != 1:
                    raise AmbiguousEdit('The preference to change is not uniquely identified.')
                if edit.operation == 'remove':
                    removed.add(normalized(result.preferences[matches[0]].requirement))
                del result.preferences[matches[0]]
            if edit.operation != 'remove':
                result.preferences.append(Preference(requirement=value, strength=edit.strength, visual=edit.visual))
        elif edit.field == 'category':
            prefixes = result.filters.category_prefixes
            if edit.operation != 'add':
                if edit.target not in prefixes:
                    raise AmbiguousEdit('The category to change is not active.')
                prefixes.remove(edit.target)
            elif edit.target:
                raise AmbiguousEdit('Adding a category cannot replace a target.')
            if edit.operation != 'remove':
                prefixes.append(value)
        else:
            if edit.target:
                raise AmbiguousEdit('Scalar edits do not take a target.')
            current = result.query if edit.field == 'query' else getattr(result.filters, edit.field)
            if edit.operation == 'add' and current not in (None, ''):
                raise AmbiguousEdit('Use replace to change an existing filter.')
            updated = None if edit.operation == 'remove' else value
            if edit.field == 'is_secondhand' and edit.operation != 'remove':
                terms = {'new': r'\b(new|brand.new|unused)\b', 'used': r'\b(used|secondhand|second.hand|pre.owned)\b'}
                if value not in terms or not re.search(terms[value], evidence):
                    raise AmbiguousEdit('Condition needs an explicit new/used request.')
                updated = value == 'used'
            if edit.field == 'brand' and edit.operation != 'remove' and normalized(value) not in evidence:
                raise AmbiguousEdit('The requested brand is not in the supporting quote.')
            if edit.field == 'query':
                result.query = updated or ''
            else:
                setattr(result.filters, edit.field, updated)
    requirements = [normalized(p.requirement) for p in result.preferences]
    if len(result.filters.category_prefixes) != len(set(result.filters.category_prefixes)):
        raise AmbiguousEdit('Duplicate category constraints.')
    if len(requirements) != len(set(requirements)) or removed.intersection(requirements):
        raise AmbiguousEdit('A removed or duplicate preference was added again.')
    for requirement in requirements:
        positive = re.sub(r'^(not |no |without )', '', requirement)
        if positive != requirement and positive in requirements:
            raise AmbiguousEdit('Conflicting positive and negative preferences.')
    # Check a narrow, unambiguous withdrawal form against existing exclusions.
    # This validates the proposed edits; it never invents a replacement edit.
    for preference in previous.preferences:
        old = normalized(preference.requirement)
        positive = re.sub(r'^(not |no |without )', '', old)
        if positive != old and re.search(
                rf'\b{re.escape(positive)}\s+is\s+(fine|ok|okay|allowed|acceptable)\b', normalized(message)):
            if old in requirements or positive in requirements:
                raise AmbiguousEdit('An explicit withdrawal was ignored or turned into a new preference.')
    return Plan.model_validate(result.model_dump())
