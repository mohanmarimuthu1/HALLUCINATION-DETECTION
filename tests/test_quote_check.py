from halludetect.detect.quote_check import quote_is_grounded


def test_exact_substring_match_is_grounded():
    chunks = {"c1": "The Eiffel Tower was completed in 1889."}
    assert quote_is_grounded("completed in 1889", ["c1"], chunks) is True


def test_missing_quote_is_not_grounded():
    chunks = {"c1": "The Eiffel Tower was completed in 1889."}
    assert quote_is_grounded("completed in 1990", ["c1"], chunks) is False


def test_empty_quote_is_never_grounded():
    chunks = {"c1": "Some evidence text."}
    assert quote_is_grounded("", ["c1"], chunks) is False


def test_no_cited_chunks_is_not_grounded():
    chunks = {"c1": "Some evidence text."}
    assert quote_is_grounded("Some evidence", [], chunks) is False


def test_unknown_chunk_id_is_not_grounded():
    chunks = {"c1": "Some evidence text."}
    assert quote_is_grounded("Some evidence", ["does-not-exist"], chunks) is False


def test_checks_every_cited_chunk_not_just_the_first():
    chunks = {"c1": "Unrelated text.", "c2": "The launch date was March 3rd, 2021."}
    assert quote_is_grounded("March 3rd, 2021", ["c1", "c2"], chunks) is True


def test_whitespace_differences_are_normalized():
    chunks = {"c1": "The   Eiffel   Tower   was completed in 1889."}
    assert quote_is_grounded("The Eiffel Tower was completed in 1889.", ["c1"], chunks) is True


def test_a_quote_with_an_altered_fact_is_not_grounded_even_if_similar():
    """A near-miss quote (right sentence, wrong number) must not be
    accepted just because most of the text matches - see quote_check.py's
    docstring for why fuzzy similarity matching was rejected for this.
    """
    chunks = {"c1": "The bridge was built in 1932."}
    assert quote_is_grounded("The bridge was built in 1999", ["c1"], chunks) is False
