import torch
import torch.nn.functional as F

def compute_attention(
    input_matrix, 
    query_proj, 
    key_proj, 
    value_proj, 
    dropout, 
    storyteller_mask, 
    scaling_factor=0.3, 
    device=None
):
    """
    Compute the attention mechanism for the input matrix.

    Parameters:
    -----------
    input_matrix : torch.Tensor
        The input matrix containing embeddings.
    query_proj : torch.nn.Module
        Query projection layer.
    key_proj : torch.nn.Module
        Key projection layer.
    value_proj : torch.nn.Module
        Value projection layer.
    dropout : torch.nn.Dropout
        Dropout layer for regularization.
    storyteller_mask : torch.Tensor
        Boolean mask indicating storyteller entries.
    scaling_factor : float
        Scaling factor for storyteller context vector.
    device : torch.device
        The device to use for computations.

    Returns:
    --------
    torch.Tensor
        The combined context vector with scaled storyteller context.
    """
    # Compute query, keys, and values
    keys = key_proj(input_matrix)
    values = value_proj(input_matrix)

    # Apply dropout to keys and values
    keys = dropout(keys)
    values = dropout(values)

    # Use the last entry as the query
    query = query_proj(input_matrix[-1, :].unsqueeze(0))
    query = dropout(query)

    # Compute attention scores over all entries
    attention_scores = torch.matmul(query, keys.transpose(0, 1))

    # Apply attention mask
    attention_mask = torch.ones(attention_scores.size(), device=device)
    attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

    ### Step 1: Compute attention for storyteller prompts
    keys_storyteller = keys[storyteller_mask]
    values_storyteller = values[storyteller_mask]

    # Compute storyteller attention scores
    storyteller_attention_scores = torch.matmul(query, keys_storyteller.transpose(0, 1))

    if storyteller_mask.sum() == 0:
        storyteller_attention_scores = storyteller_attention_scores - 1e9

    storyteller_attention_weights = F.softmax(storyteller_attention_scores, dim=-1)
    storyteller_attention_weights = dropout(storyteller_attention_weights)

    # Compute storyteller context
    storyteller_context = torch.matmul(storyteller_attention_weights, values_storyteller)

    ### Step 2: Adjust response attention scores
    expanded_storyteller_weights = torch.zeros(attention_scores.size(), device=device)
    expanded_storyteller_weights[:, storyteller_mask] = storyteller_attention_weights

    response_attention_scores = attention_scores - expanded_storyteller_weights
    response_attention_scores = response_attention_scores.masked_fill(storyteller_mask.unsqueeze(0), -1e9)

    # Apply softmax to response attention scores
    response_attention_weights = F.softmax(response_attention_scores, dim=-1)
    response_attention_weights = dropout(response_attention_weights)

    # Compute response context
    response_context = torch.matmul(response_attention_weights, values)

    ### Step 3: Combine contexts and scale storyteller context
    combined_context = (storyteller_context * scaling_factor) + response_context
    return combined_context

