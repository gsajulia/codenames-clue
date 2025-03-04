
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
      self.model_glove = GloveModel()
      self.model_bert = BertModel()
      self.board = GenerateBoard()
      
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
          print(f"Suggested Glove hint: {model_result_bert["best_hint"]}\n")
          
          print("Choosed words GloVe", clue_candidates_target_glove)
          model_result_glove = self.model_glove.find_hint(clue_candidates_target_glove, avoid_words)
          print(f"Suggested Bert hint: {model_result_glove["best_hint"]}\n")
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