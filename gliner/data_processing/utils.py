from typing import Any, Dict, List, Tuple
import torch

def pad_2d_tensor(key_data):
    """
    Pad a list of 2D tensors to have the same size along both dimensions.
    
    :param key_data: List of 2D tensors to pad.
    :return: Tensor of padded tensors stacked along a new batch dimension.
    """
    if not key_data:
        raise ValueError("The input list 'key_data' should not be empty.")

    # Determine the maximum size along both dimensions
    max_rows = max(tensor.shape[0] for tensor in key_data)
    max_cols = max(tensor.shape[1] for tensor in key_data)
    
    tensors = []

    for tensor in key_data:
        rows, cols = tensor.shape
        row_padding = max_rows - rows
        col_padding = max_cols - cols

        # Pad the tensor along both dimensions
        padded_tensor = torch.nn.functional.pad(tensor, (0, col_padding, 0, row_padding),
                                                                 mode='constant', value=0)
        tensors.append(padded_tensor)

    # Stack the tensors into a single tensor along a new batch dimension
    padded_tensors = torch.stack(tensors)

    return padded_tensors


def pretokenize_text(text: str, char_spans: List[Tuple[int, int, str]]) -> Dict[str, Any]:
    """
    Create a pretokenized version of the text, splitting it based on the spans.

    >>> pretokenize_text("L'Allemagne a un nouveau chancelier", [(2, 11, 'country'), (17, 35, 'title')])
    {'tokenized_text': ["L'", "Allemagne", " a un ", "nouveau chancelier"], 'ner': [[1, 1, 'country'], [3, 3, 'title']]}
    """
    # Filter out spans that are out of bounds
    valid_spans = [
        (start, end, label) for start, end, label in char_spans 
        if start >= 0 and end <= len(text)
    ]
    
    # Sort spans by start position
    sorted_spans = sorted(valid_spans, key=lambda x: x[0])
    
    # Initialize lists
    tokens = []
    ner = []
    current_pos = 0
    token_idx = 0

    # Process each span and text between spans
    for span_start, span_end, label in sorted_spans:
        # Add text before the span
        if span_start > current_pos:
            tokens.append(text[current_pos:span_start])
            token_idx += 1
            
        # Add the span text
        span_text = text[span_start:span_end]
        tokens.append(span_text)
        ner.append([token_idx, token_idx, label])
        token_idx += 1
        current_pos = span_end

    # Add any remaining text after the last span
    if current_pos < len(text):
        tokens.append(text[current_pos:])

    # If no valid spans, return full text as single token
    if not tokens:
        tokens = [text]

    return {
        'tokenized_text': tokens,
        'ner': ner
    }