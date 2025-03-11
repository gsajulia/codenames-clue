
from embedding_models.bert_model import BertModel
from embedding_models.glove_model import GloveModel
from utils.generate_board import GenerateBoard
import pandas as pd
import uuid
from algorithms.find_closest_cards import get_clue_candidates  
        
class BatchResults:
    def __init__(self):
      self.total_results = 20
      self.results = []
      self.easy_board_8_words = [
        ["sun", "solar", "sunshine", "energy", "heat", "light", "radiation", "environment"],
        ["ocean", "sea", "water", "wave", "tide", "coast", "marine", "salt"],
        ["mountain", "peak", "summit", "climb", "adventure", "altitude", "hike", "trail"],
        ["computer", "processor", "ram", "technology", "software", "system", "data", "network"],
        ["artist", "painting", "brush", "canvas", "gallery", "sculpture", "exhibition", "creativity"]
      ]
      # batch 20
      self.target_board_8_words = [
        ['ash', 'future', 'wall', 'wanted', 'fishing', 'hotel', 'seal', 'boy'],
        ['water', 'salad', 'note', 'link', 'group', 'mole', 'soccer', 'columbus'],
        ['planets', 'neptune', 'blind', 'root', 'africa', 'life', 'nebula', 'beverage'],
        ['hot', 'wheel', 'kitchen', 'bolt', 'share', 'internet', 'monkey', 'pyramid'],
        ['cannon', 'genius', 'balance', 'baseball', 'port', 'toronto', 'spider', 'jinx'],
        ['magazine', 'duel', 'tuna', 'steel', 'brace', 'comb', 'astronomy', 'spring'],
        ['farm', 'potato', 'ace', 'resume', 'volume', 'plot', 'track', 'card'],
        ['small', 'rage', 'cloak', 'alaska', 'taboo', 'turkey', 'saturn', 'basketball'],
        ['king', 'clown', 'frozen', 'leak', 'fiddle', 'range', 'candle', 'potion'],
        ['diamond', 'junk', 'drain', 'brazil', 'wet', 'bench', 'chief', 'wax'],
        ['philippine', 'vegas', 'treat', 'witch', 'magician', 'mile', 'chance', 'agent'],
        ['cane', 'turn', 'alliance', 'statue', 'date', 'carrot', 'lancelot', 'margin'],
        ['frankenstein', 'application', 'antarctica', 'pendant', 'kick', 'union', 'jeweler', 'disney'],
        ['figure', 'post', 'score', 'shack', 'point', 'cook', 'wine', 'viking'],
        ['dress', 'cinderella','swamp','fat', 'sword', 'sneak', 'comic', 'mushroom']
      ]
      
      self.target_board_8_words.extend(self.easy_board_8_words)
      self.model_glove = GloveModel()
      self.model_bert = BertModel()
      self.board = GenerateBoard()
      self.target_board_2_words = [list(line)[:2] for line in self.target_board_8_words]
      self.target_board_3_words = [list(line)[:3] for line in self.target_board_8_words]

    def save_results(self):
        """Save current results to a CSV file."""
        if self.results:
            df = pd.DataFrame(self.results)
            file_name = f"results/{uuid.uuid4()}_results_simple.csv"
            df.to_csv(file_name, index=False)
            print(f"Results saved in {file_name}")
        else:
            print("Empty results.")
      
    def fixed_batch_results(self, target_board, board_name):
      i = 0
      while i < len(target_board):
        try:
          default_target = target_board[i]
          default_avoid = []
          print("target", default_target)
          print("\n\n avoid", default_avoid)
          
          target_words = default_target
          
          target_embeddings_bert = self.model_bert.get_word_embedding(target_words).cpu().numpy()
          target_embeddings_glove = self.model_glove.get_word_embedding(target_words)
          clue_candidates_target_bert = get_clue_candidates(target_words, target_embeddings_bert)
          clue_candidates_target_glove = get_clue_candidates(target_words, target_embeddings_glove)
          
          if len(clue_candidates_target_bert) > 0:
            target_words =clue_candidates_target_bert

          model_result_bert_embedding_nn = self.model_bert.select_best_hint_from_embeddings_and_neighbors(target_words)
          print(f"Suggested BERT NN hint: {model_result_bert_embedding_nn["best_hint"]}\n")
          
          model_result_bert_embedding = self.model_bert.select_best_hint_from_embeddings(target_words)
          print(f"Suggested BERT embedding hint: {model_result_bert_embedding["best_hint"]}\n")
          
          if len(clue_candidates_target_glove) > 0:
            target_words = clue_candidates_target_glove
          
          model_result_glove_embedding_nn  = self.model_glove.select_best_hint_from_embeddings_and_neighbors(target_words)
          print(f"Suggested Glove NN hint: {model_result_glove_embedding_nn["best_hint"]}\n")
          
          model_result_glove_embedding = self.model_glove.select_best_hint_from_embeddings(target_words)
          print(f"Suggested Glove embedding hint: {model_result_glove_embedding["best_hint"]}\n")

          self.results.append(
              {
                  "Cards target": target_words,
                  "Cards BERT": clue_candidates_target_bert,
                  "Expected word NN BERT": model_result_bert_embedding_nn["best_hint"],
                  "Similarity NN BERT":model_result_bert_embedding_nn["best_score"],
                  "Expected embedding word BERT": model_result_bert_embedding["best_hint"],
                  "Similarity embedding BERT":model_result_bert_embedding["best_score"],
                  "Cards Glove": clue_candidates_target_glove,
                  "Expected word NN GloVe": model_result_glove_embedding_nn["best_hint"],
                  "Similarity NN GloVe": model_result_glove_embedding_nn["best_score"],
                  "Expected word embedding GloVe": model_result_glove_embedding["best_hint"],
                  "Similarity embedding GloVe": model_result_glove_embedding["best_score"],
                  "Board": board_name
                  
              }
          )
          i += 1
        
        except Exception as e:
            self.save_results()
            print(f"Error {i}: {e}")
            return
          
    def get_fixed_batch_results(self):
      self.fixed_batch_results(self.target_board_8_words,"board_8")
      self.fixed_batch_results(self.target_board_2_words,"board_2")
      self.fixed_batch_results(self.target_board_3_words,"board_3")
      
      self.save_results()
      
    def get_batch_results(self):
      i = 0
      while i < self.total_results:
        try:
          board_cards = self.board.get_codenames_board()
          default_target = board_cards["target"]
          default_avoid = board_cards["avoid"]
          print("target", default_target)
          print("\n\n avoid", default_avoid)
          
          target_words = default_target

          target_embeddings_bert = self.model_bert.get_word_embedding(target_words).cpu().numpy()
          target_embeddings_glove = self.model_glove.get_word_embedding(target_words)
          target_words = default_target
          clue_candidates_target_bert = get_clue_candidates(target_words, target_embeddings_bert)
          clue_candidates_target_glove = get_clue_candidates(target_words, target_embeddings_glove)

          avoid_words = default_avoid
          
          print("Choosed words BERT", clue_candidates_target_bert)
          model_result_bert = self.model_bert.find_hint(clue_candidates_target_bert, avoid_words)
          print(f"Suggested BERT hint: {model_result_bert["best_hint"]}\n")
          
          print("Choosed words GloVe", clue_candidates_target_glove)
          model_result_glove = self.model_glove.find_hint(clue_candidates_target_glove, avoid_words)
          print(f"Suggested Glove hint: {model_result_glove["best_hint"]}\n")
          self.results.append(
              {
                  "Cards target": target_words,
                  "Cards BERT": clue_candidates_target_bert,
                  "Expected word BERT": model_result_bert["best_hint"],
                  "Similarity BERT":model_result_bert["best_score"],
                  "Cards Glove": clue_candidates_target_glove,
                  "Expected word GloVe": model_result_glove["best_hint"],
                  "Similarity GloVe": model_result_glove["best_score"],
              }
          )
          i += 1
        
        except Exception as e:
            print(f"Error {i}: {e}")
            break

      if self.results:
          df = pd.DataFrame(self.results)
          file_name = f"results/{uuid.uuid4()}_results_simple.csv"
          df.to_csv(file_name, index=False)
          print(f"Results saved in {file_name}")
      else:
          print("Empty results.")