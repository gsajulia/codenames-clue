from utils.functions import  np_cosine_similarity
from sklearn.neighbors import NearestNeighbors

def get_clue_candidates(target_words, target_embeddings, n_neighbors=3, threshold_coef=0.95):
    """
    Finds the best clue candidates for the blue team in Codenames.
    
    Parameters:
        target_words: List of blue team words.
        n_neighbors: Number of nearest neighbors to consider for each blue word.
        threshold_coef: Fraction of best score threshold to filter candidate subsets.
    
    Returns:
        A list cards choosen (clue_candidates_filtered) meeting the threshold.
    """
    num_words = len(target_words)
    
    no_neighbors = num_words <= 1
    if no_neighbors:
        return target_words
    
    # neighbords cannot be greater than target len
    n_neighbors = min(n_neighbors, num_words)
    
    # KNN on the target words only
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine') # neighbors + the element itself
    nbrs.fit(target_embeddings)
    distances, indices = nbrs.kneighbors(target_embeddings)

    # Store neighbors for each target word
    blue_neighbors = {}
    for i, word in enumerate(target_words):
        # Exclude the word itself
        neighbor_idxs = [j for j in indices[i] if j != i and j < len(target_words)]
        neighbor_words = [target_words[j] for j in neighbor_idxs]
        blue_neighbors[word] = neighbor_words
        
    print(blue_neighbors)

    # Generate all card candidates from neighbors
    similarity_candidates = dict()
    clue_candidates_overall = set()
    for word in target_words:
        if word in blue_neighbors:
            clue_candidates_overall.update(blue_neighbors[word])
            
            try:
                word_idx = target_words.index(word)
                word_embedding = target_embeddings[word_idx]
            except ValueError:
                print(f"Error: {word} not found in target_words")
                continue

            
            for neighbor in blue_neighbors[word]:
                # KNN can generate embeddings that are not present in target so we need to check
                if neighbor in target_words: 
                    neighbor_embedding = target_embeddings[target_words.index(neighbor)]
                    similarity = np_cosine_similarity([word_embedding], [neighbor_embedding]).max()
                    similarity_candidates[neighbor] = similarity
                else:
                    print(f"Warning: Word {neighbor} not found in target_words")
            
    print('clue_candidates_overall', clue_candidates_overall)
    max_similarity = max(similarity_candidates.values()) * threshold_coef

    filtered_candidates_score = {}
    clue_candidates_filtered = []
      
    for word, score in similarity_candidates.items():
        scaled_score = score * threshold_coef
        print(f"Word: {word}, Similarity Score: {score:.4f}, Scaled Score: {scaled_score:.4f}")
        
        if scaled_score >= max_similarity * threshold_coef:
            filtered_candidates_score[word] = score
            clue_candidates_filtered.extend(blue_neighbors.get(word, []))
            clue_candidates_filtered.append(word)
            
    clue_candidates_filtered = list(dict.fromkeys(clue_candidates_filtered))
    
    return clue_candidates_filtered