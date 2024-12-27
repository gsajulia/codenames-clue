from embedding_models.glove_model import GloveModel
from embedding_models.bert_model import BertModel
from generate_board import GenerateBoard
import pandas as pd

if __name__ == "__main__":
    board = GenerateBoard()

    try:
        # board_cards = board.get_codenames_board()
        # default_target = board_cards["target"]
        # default_avoid = board_cards["avoid"]

        # # default_target = ['children', 'classroom']
        # # default_avoid = ['bath', 'break', 'thorn', 'caesar', 'date', 'gymnast', 'log', 'sea',]

        # print("target", default_target)
        # print("\n\n avoid", default_avoid)

        # target_words = default_target
        # avoid_words = default_avoid
        # model = GloveModel()
        # model_result = model.find_hint(target_words, avoid_words)
        # print(f"Suggested Glove hint: {model_result.best_hint}\n")
        # model = BertModel()
        # model_result = model.find_hint(target_words, avoid_words)
        # print(f"Suggested Bert hint: {model_result.best_hint}\n")

        # Generate results
        total_results = 20
        results = []
        i = 0
        model_glove = GloveModel()
        model_bert = BertModel()

        while i < total_results:
            board_cards = board.get_codenames_board()
            default_target = board_cards["target"]
            default_avoid = board_cards["avoid"]
            print("target", default_target)
            print("\n\n avoid", default_avoid)

            target_words = default_target
            avoid_words = default_avoid
            model_result_bert = model_bert.find_hint(target_words, avoid_words)
            print(f"Suggested Glove hint: {model_result_bert["best_hint"]}\n")
            model_result_glove = model_glove.find_hint(target_words, avoid_words)
            print(f"Suggested Bert hint: {model_result_glove["best_hint"]}\n")
            results.append(
                {
                    "Clue": default_target[0],
                    "Expected word BERT": model_result_bert["best_hint"],
                    "Similarity BERT":model_result_bert["best_score"],
                    "Expected word GloVe": model_result_glove["best_hint"],
                    "Similarity GloVe": model_result_glove["best_score"],
                }
            )
            i += 1

        df = pd.DataFrame(results)

        file_name = "results/1.0_results_simple.csv"
        df.to_csv(file_name, index=False)

    except Exception as e:
        print(f"An error occurred: {e}\n")
