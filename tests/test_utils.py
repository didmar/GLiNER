from gliner.data_processing.utils import pretokenize_text


def test_pretokenize_text():
    assert pretokenize_text("L'Allemagne a un nouveau chancelier", [(2, 11, 'country'), (17, 35, 'title')]) == {
      'tokenized_text': ["L'", "Allemagne", " a un ", "nouveau chancelier"], 'ner': [[1, 1, 'country'], [3, 3, 'title']]
    }
    assert pretokenize_text("L'Allemagne a un nouveau chancelier", []) == {
      'tokenized_text': ["L'Allemagne a un nouveau chancelier"], 'ner': []
    }
    assert pretokenize_text("United States of America", [(0, 24, 'country')]) == {
        'tokenized_text': ["United States of America"], 'ner': [[0, 0, 'country']]
    }
    # Remove spans if they are outside the text
    assert pretokenize_text("Too short", [(-10, 0, 'country'), (9, 12, 'country')]) == {
        'tokenized_text': ["Too short"], 'ner': []
    }