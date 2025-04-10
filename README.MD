## Embeddings

### 1. GloVe - Wikipedia - need to download the GloVe embeddings and add in embedding_models/embeddings/glove.6b

### 2. BERT

## Board words

- You can use or modify possible words in example_words.txt

## Usage in Codenames.py

### 1. Defining the Target

You can define your own target words by specifying them in the script.

### 2. Using Pre-filtering for 8-Target Words

Alternatively, you can define a target set with 8 words and ask the model to pre-filter the words using the find_closest_cards function. This function helps identify the most relevant word pairs for the given target set.
In this case, the find_closest_cards function will automatically filter and select words that are most semantically related to your target set.

### 3. Running the Algorithm

Once you’ve defined your target (either specific words or using the pre-filtering approach), you can run the algorithm to generate clues based on the selected words. The script will use the models (such as GloVe or BERT) to provide suggestions that align with the target words.

## Requirements

- If you want to choose between GPU and CPU, you need to install the specific torch version available on torch site

## Commands

### (Optional) Creating a virtual environment

```bash
python -m venv codenames_env
```

### (Optional) Open the virtual environment

```bash
codenames_env\Scripts\activate
```

### Setting Up and Running the Project

```bash
pip install -r requirements.txt
```

```bash
python codenames.py
```
