from genrl.grading import extract_answer, is_correct, normalize_answer


def test_extract_boxed_nested():
    assert extract_answer(r"so \boxed{\frac{1}{2}}.") == r"\frac{1}{2}"


def test_extract_last_boxed_wins():
    assert extract_answer(r"\boxed{3} wait, \boxed{4}") == "4"


def test_extract_gsm8k_marker():
    assert extract_answer("work...\n#### 1,234\n") == "1,234"


def test_extract_last_number_fallback():
    assert extract_answer("The total is 42 apples, so 45.") == "45"


def test_extract_none():
    assert extract_answer("no idea") is None


def test_normalize_numbers():
    assert normalize_answer("1,000") == "1000"
    assert normalize_answer("5.0") == "5"
    assert normalize_answer("$12$.") == "12"


def test_normalize_latex():
    assert normalize_answer(r"\dfrac{1}{2}") == normalize_answer(r"\frac{1}{2}")
    assert normalize_answer(r"10\text{ cm}") == "10cm"


def test_is_correct():
    assert is_correct("72", "72")
    assert is_correct("1,000", "1000")
    assert not is_correct(None, "3")
    assert not is_correct("4", "3")
