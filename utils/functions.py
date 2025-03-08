import torch
import numpy as np

def torch_cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.
    Args:
        vec1, vec2 (torch.Tensor): Vectors to compare.
    Returns:
        float: Cosine similarity value.
    """
    return torch.nn.functional.cosine_similarity(vec1, vec2, dim=-1)

def torch_euclidean_distance(vec1, vec2):
    """
    Calculates the Euclidean distance between two vectors.
    Args:
        vec1, vec2 (torch.Tensor): Vectors to compare.
    Returns:
        float: Euclidean distance between the two vectors.
    """
    return torch.norm(vec1 - vec2, p=2, dim=-1)
  
def np_cosine_similarity(candidates, targets):
    """
    Computes cosine similarity between candidate vectors and target vectors.
    
    Args:
        candidates (np.ndarray): Matrix of shape (n_candidates, embedding_dim).
        targets (np.ndarray): Matrix of shape (n_targets, embedding_dim).
        
    Returns:
        np.ndarray: Array of shape (n_candidates,) with similarity scores.
    """
    # Normalize vectors to unit length
    def normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-10)  # Add small value to avoid division by zero

    candidates = normalize(candidates)
    targets = normalize(targets)

    # Compute cosine similarity (dot product of normalized vectors)
    similarity_scores = np.dot(candidates, targets.T).mean(axis=1)

    return similarity_scores

# TODO check this
    # norm_candidates = np.linalg.norm(candidates, axis=1, keepdims=True)
    # norm_targets= np.linalg.norm(targets, axis=1, keepdims=True)
    
    # dot_product = np.dot(norm_candidates, norm_targets.T)
    
    # return (dot_product / (norm_candidates * norm_targets)).mean(axis=1)