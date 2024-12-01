## Under Construction 🚧

## Embeddings

### 1. GloVe - Wikipedia
### 2. BERT 

## TODO

- Test with multiple models
- Improve embeddings (embeddings can guess really wrong and fact of generic and difficult clue)
- Include logic for assassin and neutral cards

## Commands

### (Optional) Creating a virtual environment

```bash
python -m venv codenames_env
```

### (Optional) Open the virtual environment

## Windows
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

## Mac (Python 3.12 required)
```bash
source codenames_env/bin/activate
```

### Setting Up and Running the Project

```bash
pip3.12 install -r requirements.txt
```

(Required for improving performance of Apple Silicon (M1, M2))
```bash
pip3.12 install torch torchvision torchaudio
```

```bash
python3.12 mac/codenames_mac.py
```
