from utils.functions import np_cosine_similarity

import numpy as np
from nltk.stem import WordNetLemmatizer

import os

embedding_file_path = os.path.join(os.path.dirname(__file__), 'embeddings', 'glove.6b', 'glove.6B.300d.txt')

# Download necessary NLTK resources
#nltk.download('wordnet')
#nltk.download('omw-1.4')

class GloveModel:
    def __init__(self):
        """
        Initializes the GloVe model by loading embeddings.
        """
        self.embeddings = self._load_glove_embeddings(embedding_file_path)
        self.lemmatizer = WordNetLemmatizer()
        print("GloVe model is ready.")

    def _load_glove_embeddings(self, file_path):
        """
        Loads GloVe embeddings from the specified file.
        """
        print("Loading GloVe embeddings...")
        embeddings = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
                embeddings[word] = vector
        print("Embeddings loaded!")
        return embeddings

    def lemmatize_word(self, word):
        """
        Lemmatizes a word to its base form (removes derivations).
        """
        return self.lemmatizer.lemmatize(word.lower())

    def get_word_embedding(self, target_words): 
        target_embeddings = []
        valid_words = []
        
        for word in target_words:
            if word in self.embeddings:
                target_embeddings.append(self.embeddings[word])
                valid_words.append(word)  # Only words with embedding are considered
                
        if not target_embeddings:
            raise ValueError("No word with embedding")

        return np.array(target_embeddings)
    
    def get_filtered_vectors(self, target_words, avoid_words):
        """
        Helper function to process target and avoid words, and retrieve their embeddings.
        
        Args:
            target_words (list of str): Words to prioritize.
            avoid_words (list of str): Words to avoid.
        
        Returns:
            np.ndarray: Array of target vectors.
            np.ndarray: Array of candidate vectors.
            list: List of all candidate words.
        """
        # Lemmatizing target and avoid words
        target_words = {self.lemmatize_word(word) for word in target_words}
        avoid_words = {self.lemmatize_word(word) for word in avoid_words}
        excluded_words = target_words.union(avoid_words)

        # Prepare the candidate words and vectors (avoiding excluded words)
        all_words = [
            word for word in self.embeddings.keys()
            if word not in excluded_words and self.lemmatize_word(word) not in excluded_words
        ]
        all_vectors = np.array([self.embeddings[word] for word in all_words])
        
        return all_vectors, all_words

    
    def select_best_hint_from_embeddings_and_neighbors(self, target_words, avoid_words, k=10):
        """
        Finds the best hint word based on GloVe embeddings.
        
        Args:
            target_words (list of str): Words to prioritize.
            avoid_words (list of str): Words to avoid.
            k (int): Number of nearest neighbors to consider for each target word.
        
        Returns:
            dict: Mapping of each target word to its best hint and neighbors.
        """
        all_vectors, all_words = self.get_filtered_vectors(target_words, avoid_words)

        # Store the results
        results = {}
        all_top_k_words = []
        
        target_vectors = []
        for word in target_words:
            if word in self.embeddings:
                target_embeddings = np.array(self.embeddings[word])
                target_vectors.append(target_embeddings)

        target_vectors = np.array(target_vectors) 
            
        target_vectors = np.array([self.embeddings[word] for word in target_words if word in self.embeddings])
        if target_vectors.size == 0:
            raise ValueError("Target or avoid vectors are empty. Check the input words.")

        # Process in batches (similar to what was done with BERT)
        batch_size = 1000
        for batch_start in range(0, len(all_words), batch_size):
            batch_end = batch_start + batch_size
            batch_words = all_words[batch_start:batch_end]
            batch_vectors = all_vectors[batch_start:batch_end]

            # Compute cosine similarities between target words and candidate words in the batch
            for i, target_vector in enumerate(target_vectors):
                # Compute cosine similarity for this target word with all candidates
                similarities = np_cosine_similarity(batch_vectors, target_vector.reshape(1, -1))

                # Get the top-k neighbors
                top_k = np.argsort(similarities[:, i])[::-1][:k]
                top_k_words = [batch_words[idx] for idx in top_k]
                top_k_similarities = [similarities[idx, i] for idx in top_k]

                # Add to the list of all top-k words
                [all_top_k_words.append(batch_words[idx]) for idx in top_k]

                # Best hint is the first nearest neighbor
                best_idx = top_k[0]
                best_hint = batch_words[best_idx]
                best_score = similarities[best_idx, i]

                # Add the result for this target word
                results[target_words[i]] = {
                    "best_hint": best_hint,
                    "best_score": best_score,
                    "top_k_neighbors": list(zip(top_k_words, top_k_similarities))
                }

        # Now, to find the best pair (targets, neighbor) with the highest similarity:
        best_hint = None
        best_score = -1

        for word in all_top_k_words:  # Iterate through the lists of top-k words
            similarity = np_cosine_similarity(self.embeddings[word].reshape(1, -1), np.array([self.embeddings[word] for word in target_words if word in self.embeddings]))

            max_similarity = similarity.max()
            if max_similarity > best_score:
                best_score = similarity
                best_hint = word

        print(f"Best hint: {best_hint} with score {best_score.item()}")
        print('NN method')
        return {"best_hint": best_hint, "best_score": best_score.item()}
        
    def select_best_hint_from_embeddings(self, target_words, avoid_words):
        """
        Finds the best hint word using batches.
        Args:
            target_words (list of str): Words to prioritize.
            avoid_words (list of str): Words to avoid.
        Returns:
            str: The best hint word.
        """
        all_vectors, all_words = self.get_filtered_vectors(target_words, avoid_words)

        # Ensure all target words have embeddings
        target_vectors = np.array([self.embeddings[word] for word in target_words if word in self.embeddings])
        if target_vectors.size == 0:
            raise ValueError("Target or avoid vectors are empty. Check the input words.")
        
        best_hint = None
        best_score = -1
        batch_size = 1000
        # Process candidates in batches
        for batch_start in range(0, len(all_words), batch_size):
            batch_end = batch_start + batch_size
            batch_words = all_words[batch_start:batch_end]
            batch_vectors = all_vectors[batch_start:batch_end]

            # Calculate similarities
            target_similarities = np_cosine_similarity(batch_vectors, target_vectors)
            # Get max similarity
            scores = target_similarities.max(axis=1)

            # Find the best hint in the batch
            batch_best_idx = scores.argmax()
            batch_best_score = scores[batch_best_idx]

            if batch_best_score > best_score:
                best_hint = batch_words[batch_best_idx]
                best_score = batch_best_score
        
        print('Embedding method')
        return {"best_hint": best_hint, "best_score":best_score}