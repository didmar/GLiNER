import random
from gliner.data_processing.collator import DataCollator
from gliner.model import GLiNER


def test_data_processor():
  # Fix random seed for reproducibility.
  # Gliner does some shuffling during data processing, so we need to fix the seed!
  random.seed(0)

  model_name = "urchade/gliner_multi-v2.1"
  model = GLiNER.from_pretrained(model_name)
  data_collator = DataCollator(
    model.config,
    data_processor=model.data_processor,
    prepare_labels=True,
  )

  # Pre-tokenization based on annotated spans
  dataset = [{
    'tokenized_text': ["L'", 'Allemagne', ' a un ', 'nouveau chancelier'],
    'ner': [
      [1, 1, 'country'], # Allemagne
      [3, 3, 'title'], # nouveau chancelier
    ]
  }]
  x = data_collator(dataset)
  tokenizer = model.data_processor.transformer_tokenizer
  trf_token_texts = get_trf_token_texts(x, 0, tokenizer)
  assert trf_token_texts == [
        '[CLS]',
        '<<ENT>>',
        'title',
        '<<ENT>>',
        'country',
        '<<SEP>>',
        'L',
        "'",
        'Allemagne',
        '',
        'a',
        'un',
        '',
        'nouveau',
        'chance',
        'lier',
        '[SEP]',
    ]
  # Check that the spans are correct for each doc idx and class idx
  assert get_spans_for_class(x, 0, 0) == [(3, 3)]
  assert get_spans_for_class(x, 0, 1) == [(1, 1)]

  # Test passing a dataset with no entities or pre-tokens, only a single "word"
  dataset = [{
    'tokenized_text': ["L'Allemagne a un nouveau chancelier"],
    'ner': []
  }]
  x = data_collator(dataset)
  tokenizer = model.data_processor.transformer_tokenizer
  trf_token_texts = get_trf_token_texts(x, 0, tokenizer)
  # Tokenization should be the same as in the previous test
  assert trf_token_texts == [
    '[CLS]',
    '<<SEP>>',
    'L',
    "'",
    'Allemagne',
    '',
    'a',
    'un',
    '',
    'nouveau',
    'chance',
    'lier',
    '[SEP]',
  ]

  # Batch with more than one item
  dataset = [{
    'tokenized_text': ["L'", 'Allemagne', ' a un ', 'nouveau chancelier'],
    'ner': [
      [1, 1, 'country'], # Allemagne
      [3, 3, 'title'], # nouveau chancelier
    ],
  }, {
    'tokenized_text': ["Roi", " de ", "France", " et de ", "Navarre"],
    'ner': [
      [0, 0, 'title'], # Roi
      [2, 2, 'country'], # France
      [4, 4, 'country'], # Navarre
    ],
  }]
  x = data_collator(dataset)
  tokenizer = model.data_processor.transformer_tokenizer
  trf_token_texts = get_trf_token_texts(x, 0, tokenizer)
  assert trf_token_texts == [
    '[CLS]',
    '<<ENT>>',
    'country',
    '<<ENT>>',
    'title',
    '<<SEP>>',
    'L',
    "'",
    'Allemagne',
    '',
    'a',
    'un',
    '',
    'nouveau',
    'chance',
    'lier',
    '[SEP]',
  ]
  assert get_spans_for_class(x, 0, 0) == [(1, 1)]  # Allemagne -> country
  assert get_spans_for_class(x, 0, 1) == [(3, 3)]  # nouveau chancelier -> title
  trf_token_texts = get_trf_token_texts(x, 1, tokenizer)
  assert trf_token_texts == [
    '[CLS]',
    '<<ENT>>',
    'country',
    '<<ENT>>',
    'title',
    '<<SEP>>',
    'Roi',
    'de',
    'France',
    'et',
    'de',
    'Navarr',
    'e',
    '[SEP]',
    '[PAD]',
    '[PAD]',
    '[PAD]',
  ]
  assert get_spans_for_class(x, 1, 0) == [(2, 2), (4, 4)]  # France, Navarre -> country
  assert get_spans_for_class(x, 1, 1) == [(0, 0)]  # Roi -> title


def get_trf_token_texts(x, doc_idx, tokenizer):
  """
  Returns the token texts as decoded by the tokenizer.
  """
  input_ids = x['input_ids'][doc_idx].tolist()  # Taking first batch item
  # full_text = tokenizer.decode(input_ids)
  individual_tokens = [tokenizer.decode([token_id]) for token_id in input_ids]
  return individual_tokens


def get_spans_for_class(x, doc_idx, class_idx):
  if len(x['labels'].shape) == 2:
    l = x['labels'][:,class_idx].tolist()
  elif len(x['labels'].shape) == 3:
    l = x['labels'][doc_idx, :, class_idx].tolist()
  else:
    raise ValueError(f"Unexpected number of dimensions for labels: {len(x['labels'].shape)}")
  coords = [i for i, x in enumerate(l) if x == 1]
  spans = []
  for coord in coords:
    start_idx, end_idx = x['span_idx'][doc_idx][coord].tolist()
    spans.append((start_idx, end_idx))
  return spans