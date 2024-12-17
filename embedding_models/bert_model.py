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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.stemmer = PorterStemmer()

        print("BERT model loaded!")


    def is_valid_hint(self, word, target_words):
        """
        Determines if a word is valid for use as a hint.
        - Excludes stopwords and words with specific unwanted characteristics.
        """
        stop_words = set(stopwords.words('english'))
        if word.startswith("##"):
            return False
        if word in stop_words:
            return False
        if len(word) < 3:
            return False
        if word in target_words:
            return False
        for target_word in target_words:
            if self.stemmer.stem(word) == self.stemmer.stem(target_word):
                return False
        technical_words = {}
        if any(substring in word for substring in technical_words):
            return False
        return word.isalpha()
    
    def get_word_embedding(self, words):
        """
        Computes the embedding vector for a given word using BERT.
        Args:
            word (str): The word for which the embedding is required.
        Returns:
            torch.Tensor: The embedding vector.
        """
        inputs = self.tokenizer(words, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

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
        print("Computing embeddings for target and avoid words...")
        target_embeddings = self.get_word_embedding(target_words)
        avoid_embeddings = self.get_word_embedding(avoid_words)

        print("Precomputing embeddings for candidate words...")
        candidate_words = [
            word for word in tqdm(self.tokenizer.vocab.keys(), desc="Checking words with Bert") 
            if self.is_valid_hint(word, target_words)
        ]
        candidate_embeddings = []
        for batch_start in tqdm(range(0, len(candidate_words), 16), desc="Processing candidate words"):
            batch = candidate_words[batch_start:batch_start + 16]
            candidate_embeddings.append(self.get_word_embedding(batch))
        candidate_embeddings = torch.cat(candidate_embeddings, dim=0)

        target_similarities = torch.mm(candidate_embeddings, target_embeddings.T).mean(dim=1)
        avoid_similarities = torch.mm(candidate_embeddings, avoid_embeddings.T).mean(dim=1)
        scores = target_similarities - avoid_similarities

        best_index = scores.argmax().item()
        best_hint = candidate_words[best_index]
        best_score = scores[best_index].item()

        print(f"Best hint: {best_hint}, Score: {best_score}")
        return best_hint