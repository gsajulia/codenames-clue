import random

class GenerateBoard:
    def __init__(self):
        self.used_words = set() 

    def get_random_words_by_batch(self, batch):
        """
        Open a file and select random words.

        Args:
            file_path (str): file path .txt.
            x (int): Number of words to be selected

        Returns:
            list: Lista contendo x palavras aleatórias.
        """
        with open("example_words.txt", "r") as file:
            words = file.read().splitlines()
            
        words = [word.lower() for word in words]

        available_words = [word for word in words if word not in self.used_words]

        if len(available_words) < batch:
            raise ValueError(
                f"Not enough unique words available. Needed {batch}, but only {len(available_words)} left."
            )

        selected_words = random.sample(available_words, batch)

        self.used_words.update(selected_words)

        return selected_words

    def get_codenames_board(self):
        neutral_size = 7
        assassin_size = 1
        avoid_team_size = 8
        target_team_size = 8

        board_size = [neutral_size, assassin_size, avoid_team_size, target_team_size]
        board_result = []
        for board_element in board_size:
            new_word = self.get_random_words_by_batch(board_element)
            board_result.append(new_word)

        if len(board_result) == 4:
            return {
                "neutral": board_result[0],
                "assassin": board_result[1],
                "avoid": board_result[2],
                "target": board_result[3],
            }
