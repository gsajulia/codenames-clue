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

    def cosine_similarity(self, vec1, vec2):
        """
        Computes the cosine similarity between two vectors.
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def similarity_from_distance(self, vec1, vec2):
        """
        Computes the similarity between two vectors based on Euclidean distance.
        """
        # Compute Euclidean distance
        distance = np.linalg.norm(vec1 - vec2)
        return 1 - (distance ** 2 / 2)

    def lemmatize_word(self, word):
        """
        Lemmatizes a word to its base form (removes derivations).
        """
        return self.lemmatizer.lemmatize(word.lower())

    def find_hint(self, target_words, avoid_words):
        """
        Finds the best hint word using batches.
        Args:
            target_words (list of str): Words to prioritize.
            avoid_words (list of str): Words to avoid.
        Returns:
            str: The best hint word.
        """
        target_words = {self.lemmatize_word(word) for word in target_words}
        avoid_words = {self.lemmatize_word(word) for word in avoid_words}
        excluded_words = target_words.union(avoid_words)

        all_words = [
            word for word in self.embeddings.keys()
            if word not in excluded_words and self.lemmatize_word(word) not in excluded_words
        ]
        all_vectors = np.array([self.embeddings[word] for word in all_words])

        best_hint = None
        best_score = float('-inf')

        target_vectors = np.array([self.embeddings[word] for word in target_words if word in self.embeddings])
        # avoid_vectors = np.array([self.embeddings[word] for word in avoid_words if word in self.embeddings])

        if target_vectors.size == 0:
            raise ValueError("Target or avoid vectors are empty. Check the input words.")

        batch_size = 1000
        # Process candidates in batches
        for batch_start in range(0, len(all_words), batch_size):
            batch_end = batch_start + batch_size
            batch_words = all_words[batch_start:batch_end]
            batch_vectors = all_vectors[batch_start:batch_end]

            # Calculate similarities
            target_similarities = np.dot(batch_vectors, target_vectors.T).mean(axis=1)
            #avoid_similarities = np.dot(batch_vectors, avoid_vectors.T).mean(axis=1)
            scores = target_similarities # - avoid_similarities

            # Find the best hint in the batch
            batch_best_idx = scores.argmax()
            batch_best_score = scores[batch_best_idx]

            if batch_best_score > best_score:
                best_hint = batch_words[batch_best_idx]
                best_score = batch_best_score
        
        return {"best_hint": best_hint, "best_score":best_score}