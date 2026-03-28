"""Tests for JSON answer extraction from agent responses."""

import pytest

from accelbench.extract import extract_json_answer


class TestFencedJsonBlock:
    def test_simple_fenced_block(self):
        text = 'Here is the answer:\n```json\n{"value": 42}\n```'
        assert extract_json_answer(text) == {"value": 42}

    def test_fenced_block_with_whitespace(self):
        text = '```json\n  {"value": 42}  \n```'
        assert extract_json_answer(text) == {"value": 42}

    def test_multiple_fenced_blocks_uses_last(self):
        text = (
            '```json\n{"value": 1}\n```\n'
            "Some reasoning...\n"
            '```json\n{"value": 2}\n```'
        )
        assert extract_json_answer(text) == {"value": 2}

    def test_nested_object_in_fenced_block(self):
        text = '```json\n{"orbit": {"BPM1": 0.1, "BPM2": -0.2}}\n```'
        result = extract_json_answer(text)
        assert result == {"orbit": {"BPM1": 0.1, "BPM2": -0.2}}

    def test_fenced_block_with_array_value(self):
        text = '```json\n{"devices": ["QF", "QD", "BPM1"]}\n```'
        result = extract_json_answer(text)
        assert result == {"devices": ["QF", "QD", "BPM1"]}

    def test_invalid_json_in_fenced_block_falls_through(self):
        text = '```json\n{invalid json}\n```\nLater: {"value": 99}'
        assert extract_json_answer(text) == {"value": 99}


class TestBraceFallback:
    def test_simple_object_in_text(self):
        text = 'The answer is {"value": 3.14} as computed.'
        assert extract_json_answer(text) == {"value": 3.14}

    def test_last_object_wins(self):
        text = 'First {"a": 1} then {"b": 2}'
        assert extract_json_answer(text) == {"b": 2}

    def test_nested_braces(self):
        text = 'Result: {"orbit_change": {"BPM1": 0.001}}'
        result = extract_json_answer(text)
        assert result == {"orbit_change": {"BPM1": 0.001}}

    def test_non_dict_json_skipped(self):
        # Arrays are not dicts, should be skipped
        text = 'Only an array: [1, 2, 3]'
        assert extract_json_answer(text) is None


class TestNoMatch:
    def test_empty_string(self):
        assert extract_json_answer("") is None

    def test_no_json_at_all(self):
        assert extract_json_answer("Just some plain text with no JSON.") is None

    def test_only_invalid_json(self):
        assert extract_json_answer("{not: valid: json}") is None

    def test_only_non_dict_braces(self):
        text = "Some text with {unclosed brace"
        assert extract_json_answer(text) is None


class TestEdgeCases:
    def test_multiline_fenced_block(self):
        text = '```json\n{\n  "value": 42,\n  "unit": "mm"\n}\n```'
        result = extract_json_answer(text)
        assert result == {"value": 42, "unit": "mm"}

    def test_fenced_block_preferred_over_brace(self):
        text = (
            'Stray {"wrong": true} in text.\n'
            '```json\n{"right": true}\n```'
        )
        assert extract_json_answer(text) == {"right": True}

    def test_numeric_string_values(self):
        text = '```json\n{"count": 5, "name": "QF"}\n```'
        result = extract_json_answer(text)
        assert result == {"count": 5, "name": "QF"}

    def test_boolean_values(self):
        text = '```json\n{"passed": true, "value": null}\n```'
        result = extract_json_answer(text)
        assert result == {"passed": True, "value": None}
