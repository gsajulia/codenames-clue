
from embedding_models.bert_model import BertModel
from embedding_models.glove_model import GloveModel
from utils.generate_board import GenerateBoard
import pandas as pd

class BatchResults:
    def __init__(self):
      self.total_results = 1
      self.results = []
      self.model_glove = GloveModel()
      self.model_bert = BertModel()
      self.board = GenerateBoard()
      
    def get_batch_results(self):
      i = 0
      while i < self.total_results:
        board_cards = self.board.get_codenames_board()
        default_target = board_cards["target"]
        default_avoid = board_cards["avoid"]
        print("target", default_target)
        print("\n\n avoid", default_avoid)

        target_words = default_target
        avoid_words = default_avoid
        model_result_bert = self.model_bert.find_hint(target_words, avoid_words)
        print(f"Suggested Glove hint: {model_result_bert["best_hint"]}\n")
        model_result_glove = self.model_glove.find_hint(target_words, avoid_words)
        print(f"Suggested Bert hint: {model_result_glove["best_hint"]}\n")
        self.results.append(
            {
                "Clue": default_target[0],
                "Expected word BERT": model_result_bert["best_hint"],
                "Similarity BERT":model_result_bert["best_score"],
                "Expected word GloVe": model_result_glove["best_hint"],
                "Similarity GloVe": model_result_glove["best_score"],
            }
        )
        i += 1

      df = pd.DataFrame(self.results)

      file_name = "results/2.0_results_simple.csv"
      df.to_csv(file_name, index=False)