from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
import torch

nltk.download('stopwords')
nltk.download('wordnet')

class BertModel:
    def __init__(self, model_name="bert-base-uncased"):
        """
        Initializes the BERT model and tokenizer.
        Args:
            model_name (str): Name of the pretrained BERT model to load.
        """
        print("Loading BERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.stemmer = PorterStemmer()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print("BERT model loaded!")

    # ... (Outras funções continuam iguais)

    def is_valid_hint(self, word, target_words):
        """
        Determines if a word is valid for use as a hint.
        - Excludes stopwords and words with specific unwanted characteristics.
        """
        stop_words = set(stopwords.words('english'))  # Load a set of common stopwords
        if word in stop_words:
            return False
        if len(word) < 3:  # Exclude words that are too short
            return False
        
        # Check if the word is a simple derivative (plural/singular)
        if word in target_words:
            return False
        for target_word in target_words:
            if self.stemmer.stem(word) == self.stemmer.stem(target_word):
                return False
        
        # Check other technical or unwanted substrings in words
        technical_words = {} # TODO
        if any(substring in word for substring in technical_words):
            return False

        return True
    
    def get_word_embedding(self, word):
        """
        Computes the embedding vector for a given word using BERT.
        Args:
            word (str): The word for which the embedding is required.
        Returns:
            torch.Tensor: The embedding vector.
        """
        inputs = self.tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Average pooling to get a single vector per word
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    def cosine_similarity(self, vec1, vec2):
        """
        Computes cosine similarity between two vectors.
        Args:
            vec1, vec2 (torch.Tensor): Vectors to compare.
        Returns:
            float: Cosine similarity value.
        """
        return torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))
    
    def similarity_from_distance(vec1, vec2):
        """
        Computes the similarity between two vectors based on Euclidean distance.
        """
        # Compute Euclidean distance
        distance = torch.norm(vec1 - vec2)
        return 1 - (distance ** 2 / 2)
    
    def find_hint(self, target_words, avoid_words):
        """
        Finds the best hint word that is most similar to the target words 
        and least similar to the avoid words.
        Args:
            target_words (list of str): Words to prioritize.
            avoid_words (list of str): Words to avoid.
        Returns:
            str: The best hint word.
        """
        # Get embeddings for target and avoid words
        target_vectors = [self.get_word_embedding(word) for word in tqdm(target_words, desc="Processing target words")]
        avoid_vectors = [self.get_word_embedding(word) for word in tqdm(avoid_words, desc="Processing avoid words")]

        best_hint = None
        best_score = float('-inf')

        # Iterate over BERT vocabulary
        for word in tqdm(self.tokenizer.vocab.keys(), desc="Checking words with Bert"):
            # Skip words that are split into subwords or not valid hints
            if len(self.tokenizer.tokenize(word)) > 1:
                continue

            candidate_vec = self.get_word_embedding(word)

            # Exclude target words from being used as hints
            if word not in target_words and self.is_valid_hint(word, target_words):
                # Compute similarity to target and avoid word sets
                target_sim = sum(self.cosine_similarity(candidate_vec, vec) for vec in target_vectors) / len(target_vectors)
                avoid_sim = sum(self.cosine_similarity(candidate_vec, vec) for vec in avoid_vectors) / len(avoid_vectors)

                # Calculate the final score
                score = target_sim - avoid_sim

                #print(f"Candidate: {word}, Target Similarity: {target_sim}, Avoid Similarity: {avoid_sim}, Score: {score}")

                # Update the best hint if the score is higher
                if score > best_score:
                    best_hint = word
                    best_score = score

        print(f"Best hint: {best_hint}, Score: {best_score}")
        return best_hint