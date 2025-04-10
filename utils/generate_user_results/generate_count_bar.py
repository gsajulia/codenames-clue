import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# This chart is a count from each word to be selected and the correct and incorrect guesses from gameplays
folder_path = "board_precision"
correct_counts = defaultdict(int)
all_words = set()

for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)

        correct_words_in_file = set()

        for _, row in df.iterrows():
            for word_col, correct_col in [('word_1', 'is_correct_1'), ('word_2', 'is_correct_2')]:
                word = str(row[word_col]).strip().lower()
                all_words.add(word)
                if row[correct_col] == True:
                    correct_words_in_file.add(word)

        for word in correct_words_in_file:
            correct_counts[word] += 1

words_df = pd.DataFrame([
    {
        'word': word,
        'correct': correct_counts.get(word, 0),
        'incorrect': 5 - correct_counts.get(word, 0)
    }
    for word in all_words
])

# Order by correct guesses
words_df = words_df.sort_values(by='correct', ascending=True)

plt.figure(figsize=(12, 10))
plt.barh(words_df['word'], words_df['correct'], color='#50b99b', label='Correct')
plt.barh(words_df['word'], words_df['incorrect'], left=words_df['correct'], color='#ad2831', label='Incorrect')

plt.xlabel('Count')
plt.title('Correct vs Incorrect Guesses per Word')
plt.legend()
plt.xlim([0, 5])
plt.tight_layout()

plt.savefig("board_word_correctness.png", dpi=300)
plt.close()
