# Word Representations from Scratch

Hi! This repo has two small projects I built to teach myself how word embeddings actually work under the hood. Both are written from scratch in PyTorch and NumPy as [marimo](https://marimo.io) notebooks.

The two models are:

1. **Word2Vec (Skip-gram)** — the classic "one fixed vector per word" model
2. **ELMo** — a follow-up that gives each word a different vector depending on the sentence it appears in

The notebooks go in order. The Word2Vec one ends by pointing out why static embeddings aren't enough, and the ELMo one ends by pointing out why even ELMo isn't enough (and why Transformers / BERT came next). So the whole repo is basically a little story about how word representations got better over time.

---

## What's in each notebook

### [word2vec_skipgram.py](word2vec_skipgram.py)

A from-scratch Skip-gram Word2Vec model. I kept it simple and used full softmax instead of negative sampling so the loss function is easy to read.

It walks through:

- Making (center word, context word) training pairs from a tiny corpus using a sliding window
- A `SkipGramWord2Vec` module with two embedding matrices (one for center words, one for context words)
- A normal training loop with Adam
- A PCA plot so you can see the learned vectors in 2D
- The **polysemy problem** — the word "bank" shows up in both financial and river sentences, but Word2Vec has to squeeze both meanings into the same vector
- The famous `king - man + woman ≈ queen` analogy, implemented from scratch

### [elmo.py](elmo.py)

A simplified ELMo built with two independent LSTMs — one reading left-to-right and one reading right-to-left. Their hidden states get concatenated at the end. This is the "shallow bidirectionality" that ELMo is known for.

It walks through:

- A `SimpleELMo` module: embedding layer + forward LSTM + backward LSTM
- A demo where the word "bank" gets **different vectors** in "i went to the bank to deposit money" vs. "he sat on the bank of the river" — finally fixing the problem Word2Vec couldn't
- A small utility function that lets you check how similar a word's representation is across two different sentences
- A closing discussion about why ELMo still isn't ideal (the two directions never really talk to each other, LSTMs are slow, etc.) and how BERT / Transformers fixed that

Both notebooks use the same small toy corpus that was designed on purpose to include polysemous words, so you can clearly see the difference between the two approaches.

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) and Python 3.13.

```bash
git clone https://github.com/AtharvaBhorpe/word_representations_from_scratch.git
cd word_representations_from_scratch
uv sync
```

A small heads-up: `pyproject.toml` is set up to install PyTorch from the CUDA 12.8 wheel index because I'm running this on an RTX 50-series GPU. If you're on a different GPU or just on CPU, you'll want to edit the `[tool.uv.sources]` block to point at whichever PyTorch index you need (like `cpu` or `cu121`).

## Running the notebooks

marimo notebooks are just regular `.py` files, which is nice because they diff cleanly in git.

To open them interactively:

```bash
uv run marimo edit word2vec_skipgram.py
uv run marimo edit elmo.py
```

To just run one as a script:

```bash
uv run python word2vec_skipgram.py
```

---

## Repo layout

```
.
├── word2vec_skipgram.py    # Skip-gram Word2Vec notebook
├── elmo.py                 # ELMo notebook
├── main.py                 # Tiny placeholder
├── pyproject.toml          # uv config + PyTorch CUDA 12.8 index
├── uv.lock                 # Locked dependencies
├── .python-version         # Pins Python 3.13
└── README.md
```

---

## Why I made this

I wanted to actually build these models instead of just reading about them. Writing the code from scratch — even on a tiny corpus — made the ideas click in a way that just reading papers never quite did. If you're learning NLP or trying to brush up on the foundations of word embeddings, I hope working through these notebooks helps you the same way.

By the end you should have a decent feel for:

- Why a single fixed vector per word breaks down on words like "bank" or "play"
- How a biLSTM can produce a different vector for the same word in different sentences
- Why ELMo's bidirectionality is called "shallow", and what self-attention does differently
