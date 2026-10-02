from src.evaluation import character_error_rate, word_error_rate


def test_error_rates_are_zero_for_matching_text() -> None:
    assert word_error_rate("Board notes", "board notes") == 0
    assert character_error_rate("Board notes", "board notes") == 0


def test_error_rates_count_edits() -> None:
    assert word_error_rate("one two three", "one three") == 1 / 3
    assert character_error_rate("abc", "adc") == 1 / 3


def test_empty_reference_is_handled() -> None:
    assert word_error_rate("", "") == 0
    assert word_error_rate("", "extra") == 1
    assert character_error_rate("", "extra") == 1