import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Building Word2Vec Skip-gram model from scratch
    """)
    return


@app.cell
def _():
    # Setup
    import marimo as mo
    import sys
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter
    import random

    # Check GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("No GPU detected. Some cells may run slowly.")
        print("Go to Runtime -> Change runtime type -> GPU")

    print(f"\nPython {sys.version.split()[0]}")
    print(f"\nPytorch {torch.__version__}")

    # Set random seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"Random seed set to {SEED}")


    # Uncomment the line below if not using Marimo Notebook (ex. uncomment if using Jupyter/Google  Notebook)
    # %matplotlib inline
    return Counter, device, mo, nn, np, optim, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preparing a Small Corpus
    """)
    return


@app.cell
def _(Counter):
    # A small corpus with deliberate polysemy
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat chased the dog",
        "the dog chased the cat",
        "i went to the bank to deposit money",
        "she walked to the bank to withdraw cash",
        "he sat on the bank of the river",
        "the river bank was covered with grass",
        "the cat purred on the mat",
        "the dog barked at the cat",
        "money was deposited at the bank",
        "the river bank had beautiful flowers",
        "the mat was on the floor",
        "the rug was under the dog",
        "cash was withdrawn from the bank",
        "grass grew along the river bank",
    ]

    # Tokenize
    tokenized_corpus = [sentence.split() for sentence in corpus]

    # Build vocabulary
    all_words = [word for sentence in tokenized_corpus for word in sentence]
    word_count = Counter(all_words)
    vocab = sorted(word_count.keys())
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx:word for word, idx in word_to_idx.items()}

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Vocabulary: {vocab}")
    return idx_to_word, tokenized_corpus, vocab, vocab_size, word_to_idx


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Building Skip-gram Training Pairs
    For each word in the corpus, we create (center_word, context_word) pairs using a sliding window approach.
    """)
    return


@app.cell
def _(idx_to_word, tokenized_corpus, word_to_idx):
    def create_skipgram_pairs(tokenized_corpus, word_to_idx, window_size=2):
        """
        Create (center_word, context_word) pairs for Skip-gram training.

        For each word in each sentence, look at 'window_size' words to the
        left and right as context.
        """
        pairs = []
        for sentence in tokenized_corpus:
            for i, center_word in enumerate(sentence):
                # Look at window_size words in each direction
                for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                    if i != j:  # Skip the center word itself
                        context_word = sentence[j]
                        pairs.append((word_to_idx[center_word], word_to_idx[context_word]))
        return pairs

    pairs = create_skipgram_pairs(tokenized_corpus, word_to_idx, window_size=2)
    print(f"Total training pairs: {len(pairs)}")
    print(f"\nFirst 5 pairs:")
    for center_idx, context_idx in pairs[:5]:
        print(f"Center: '{idx_to_word[center_idx]}' ->  Context: '{idx_to_word[context_idx]}'")
    return (pairs,)


@app.cell
def _(Counter, idx_to_word, pairs, plt):
    # Visualization: distribution of training pairs
    center_words = [idx_to_word[p[0]] for p in pairs]
    center_count = Counter(center_words)
    top_words = center_count.most_common(10)

    plt.figure(figsize=(10, 4))
    plt.bar([w for w,c in top_words], [c for w, c in top_words], color='steelblue')
    plt.title("Most Common Center Words in Training Pairs")
    plt.xlabel("Word")
    plt.ylabel("Number of Training Pairs")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Word2Vec Model from Scratch
    """)
    return


@app.cell
def _(nn, torch):
    class SkipGramWord2Vec(nn.Module):
        """
        Skip-gram Word2Vec model.

        Two embedding matrices:
        - center_embeddings: vectors for center words (input)
        - context_embeddings: vectors for context words (output)
        """
        def __init__(self, vocab_size, embedding_dim):
            super().__init__()
            self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

            # Initialize with small random values
            nn.init.uniform_(self.center_embeddings.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
            nn.init.uniform_(self.context_embeddings.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)

        def forward(self, center_ids, context_ids):
            # Get embeddings: (batch_size, embedding_dim)
            center_vecs = self.center_embeddings(center_ids)
            context_vecs = self.context_embeddings(context_ids)

            # Dot product for each pair: (batch_size,)
            scores = torch.sum(center_vecs * context_vecs, dim=1)

            # Compute log-softmax over entire vocabulary
            # For full softmax: score of center with ALL context words
            all_context = self.context_embeddings.weight  # (vocab_size, embedding_dim)
            all_scores = torch.matmul(center_vecs, all_context.T)  # (batch_size, vocab_size)

            log_probs = torch.log_softmax(all_scores, dim=1)

            # Gather the log probabilities for the actual context words
            loss = -log_probs.gather(1, context_ids.unsqueeze(1)).squeeze(1)
            return loss.mean()

        def get_embedding(self, word_idx):
            """Get the learned embedding for a word."""
            return self.center_embeddings.weight[word_idx].detach().cpu().numpy()

    return (SkipGramWord2Vec,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Word2Vec
    """)
    return


@app.cell
def _(SkipGramWord2Vec, device, optim, pairs, torch, vocab_size):
    # Hyperparameters
    EMBEDDING_DIM = 32
    LEARNING_RATE = 0.01
    EPOCHS = 200
    BATCH_SIZE = 64

    model = SkipGramWord2Vec(vocab_size, EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Convert pairs to tensors
    center_ids = torch.tensor([p[0] for p in pairs], dtype=torch.long).to(device)
    context_ids = torch.tensor([p[1] for p in pairs], dtype=torch.long).to(device)

    # Training Loop
    losses = []
    for epoch in range(EPOCHS):
        # Shuffle data
        perm = torch.randperm(len(pairs))
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(pairs), BATCH_SIZE):
            batch_idx = perm[i:i+BATCH_SIZE]
            batch_centers = center_ids[batch_idx]
            batch_contexts = context_ids[batch_idx]

            loss = model(batch_centers, batch_contexts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    return losses, model


@app.cell
def _(losses, plt):
    # 📊 Training curve
    plt.figure(figsize=(8, 4))
    plt.plot(losses, color='steelblue', linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Word2Vec Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing the Embeddings
    """)
    return


@app.cell
def _(model, plt, vocab):
    # Get all word embeddings
    embeddings = model.center_embeddings.weight.detach().cpu().numpy()

    # Use PCA to project to 2D
    from numpy.linalg import svd

    # Center the data
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    U, S, Vt = svd(centered, full_matrices=False)
    projected = centered @ Vt[:2].T  # Project to first 2 principal components

    # Plot
    plt.figure(figsize=(12, 8))
    for index, word in enumerate(vocab):
        x, y = projected[index]
        plt.scatter(x, y, color='steelblue', s=50, zorder=5)
        plt.annotate(word, (x, y), fontsize=9, ha='center', va='bottom',
                     xytext=(0, 5), textcoords='offset points')

    plt.title("Word2Vec Embeddings (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return (projected,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Polysemy Problem - Word2Vec's Fatal Flaw

    Word2Vec (and its Skip-Gram variant) learns a **single, fixed embedding vector** for each word in the vocabulary. Once training is done, the word *"bank"* always maps to the exact same vector — whether it appears in *"river bank"* or *"investment bank."* This is the heart of the problem.

    **Polysemy** — the phenomenon where one word carries multiple distinct meanings — is completely invisible to Word2Vec. The model resolves this ambiguity by essentially averaging all the contexts a word appears in during training, producing one "blended" vector that faithfully represents none of its actual meanings.

    Consider the word *"play"*:

    | Sentence | Intended Meaning |
    |---|---|
    | *"She went to see a play."* | Theatrical performance |
    | *"He made a play for the leadership role."* | Strategic move |
    | *"The kids went out to play."* | Recreational activity |

    Word2Vec assigns all three the **same vector**. Downstream tasks — sentiment analysis, QA, NER — then inherit this confusion.

    The word "bank" appears in two very different contexts in our corpus — financial and river. But Word2Vec gives it one single vector.
    """)
    return


@app.cell
def _(model, np, word_to_idx):
    # Find nearest neighbors using cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Find the embedding for "bank"
    bank_idx = word_to_idx["bank"]
    bank_embedding = model.get_embedding(bank_idx)

    def _():  # Required to wrap this cell inside a function to reuse the variable names like "word"
        print("=== Nearest neighbors to 'bank' ===\n")
        similarities = []
        for word, idx in word_to_idx.items():
            if word != "bank":
                sim = cosine_similarity(bank_embedding, model.get_embedding(idx))
                similarities.append((word, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        for word, sim in similarities[:8]:
            print(f"  {word:12s} → similarity: {sim:.3f}")


    _()
    return bank_embedding, cosine_similarity


@app.cell
def _(idx_to_word, plt, projected, word_to_idx):
    def _():
        # 📊 The polysemy problem visualized
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: financial context sentences
        financial_words = ["bank", "money", "deposit", "cash", "withdrawn"]
        river_words = ["bank", "river", "grass", "flowers", "grew"]

        # Show that bank is the SAME point regardless of context
        financial_indices = [word_to_idx[w] for w in financial_words if w in word_to_idx]
        river_indices = [word_to_idx[w] for w in river_words if w in word_to_idx]

        # Financial context
        ax = axes[0]
        for idx in financial_indices:
            word = idx_to_word[idx]
            x, y = projected[idx]
            color = 'red' if word == 'bank' else 'steelblue'
            size = 150 if word == 'bank' else 80
            ax.scatter(x, y, color=color, s=size, zorder=5)
            ax.annotate(word, (x, y), fontsize=11, ha='center', va='bottom',
                        xytext=(0, 6), textcoords='offset points', fontweight='bold' if word == 'bank' else 'normal')
        ax.set_title("Financial Context Words", fontsize=13)
        ax.grid(alpha=0.3)

        # River context
        ax = axes[1]
        for idx in river_indices:
            word = idx_to_word[idx]
            x, y = projected[idx]
            color = 'red' if word == 'bank' else 'forestgreen'
            size = 150 if word == 'bank' else 80
            ax.scatter(x, y, color=color, s=size, zorder=5)
            ax.annotate(word, (x, y), fontsize=11, ha='center', va='bottom',
                        xytext=(0, 6), textcoords='offset points', fontweight='bold' if word == 'bank' else 'normal')
        ax.set_title("River Context Words", fontsize=13)
        ax.grid(alpha=0.3)

        plt.suptitle("⚠️ 'bank' has ONE vector — it cannot distinguish contexts!", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


    _()
    return


@app.cell
def _(bank_embedding, plt):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Word2Vec: same vector
    ax.bar(range(10), bank_embedding[:10], color='coral', alpha=0.8, label='All contexts')
    ax.set_title("Word2Vec: 'bank' embedding\n(SAME for all contexts)", fontsize=11)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Value")
    ax.legend()
    ax.set_ylim(-2, 2)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Function to compute Word2Vec analogy

    One of the famous properties of Word2Vec is that it captures analogies: king - man + woman ≈ queen

    This analogy function is implemented below
    """)
    return


@app.cell
def _(cosine_similarity):
    def word_analogy(model, word_to_idx, idx_to_word, word_a, word_b, word_c):
        """
        Compute: word_a - word_b + word_c = ???

        For example: king - man + woman = ???

        Args:
            word_a, word_b, word_c: strings
        Returns:
            The word closest to (vec_a - vec_b + vec_c)
        """
        vec_a = model.get_embedding(word_to_idx[word_a])
        vec_b = model.get_embedding(word_to_idx[word_b])
        vec_c = model.get_embedding(word_to_idx[word_c])

        # ==============================
        # Step 1: Compute the analogy vector: vec_a - vec_b + vec_c
        # Step 2: Find the word in vocabulary whose embedding is most similar
        #         (using cosine similarity) to the analogy vector
        # Step 3: Exclude words a, b, c from candidates
        # ==============================

        analogy_vec = vec_a - vec_b + vec_c

        best_word = None
        best_sim = -1

        for word, idx in word_to_idx.items():
            if word in [word_a, word_b, word_c]:
                continue
            # compute similarity and track the best match
            sim = cosine_similarity(analogy_vec, model.get_embedding(idx))
            if sim > best_sim:
                best_sim = sim
                best_word = word

        return best_word

    return (word_analogy,)


@app.cell
def _(idx_to_word, model, word_analogy, word_to_idx):
    # ✅ Verification
    # With our small corpus, exact analogies are unlikely, but the function should work
    # Let's test the mechanics: "cat" - "mat" + "rug" should lean toward "dog"
    result = word_analogy(model, word_to_idx, idx_to_word, "cat", "mat", "rug")
    print(f"cat - mat + rug = {result}")
    assert result is not None, "❌ Function returned None — check your implementation"
    assert isinstance(result, str), "❌ Function should return a string (word)"
    print("✅ Analogy function works correctly!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Beyond polysemy, Word2Vec also suffers from:

    - **Out-of-vocabulary (OOV) words** — any word not seen during training gets no representation at all
    - **No morphological awareness** — *"run"*, *"runs"*, *"running"* are treated as completely unrelated tokens (unless you use FastText)
    - **Context-free by design** — the embedding of a word depends only on its training co-occurrences globally, never on its neighboring words *at inference time*

    ---
    ## From Static to Contextual: The ELMo Approach Fixes This

    ELMo (Embeddings from Language Models), introduced by Peters et al. (2018) at AllenNLP, takes a fundamentally different approach: instead of assigning a word a fixed vector, it **computes the embedding dynamically based on the entire input sentence.**

    The architecture is a **deep, bidirectional LSTM** trained as a language model:

    ```
    Forward LM:   "The river bank ..."   →  predicts next word
    Backward LM:  "... bank river The"  →  predicts previous word
    ```

    Both directions run simultaneously across multiple LSTM layers. The final representation for a word is a **learned weighted combination of all layer outputs**:

    $$\mathbf{ELMo}_k = \gamma \sum_{j=0}^{L} s_j \cdot \mathbf{h}_{k,j}^{LM}$$

    where $s_j$ are softmax-normalized weights, $\gamma$ is a task-specific scalar, and $\mathbf{h}_{k,j}^{LM}$ is the hidden state of token $k$ at layer $j$.

    ### What Each Layer Captures

    | Layer | What It Encodes |
    |---|---|
    | Layer 0 (embedding) | Raw character-level / token features |
    | Lower LSTM layers | Syntax — POS tags, morphology |
    | Higher LSTM layers | Semantics — word sense, context meaning |

    So when ELMo sees *"river bank"*, the higher LSTM layers pick up on the surrounding context (*"river"*) and produce a vector that sits in a completely different region of embedding space compared to *"savings bank"*. The **same token, two different vectors** — polysemy solved.

    ---

    ## Word2Vec vs ELMo — Side by Side

    | Property | Word2Vec (Skip-Gram) | ELMo |
    |---|---|---|
    | Embedding type | Static, context-free | Dynamic, context-sensitive |
    | Polysemy handling | ✗ Blended average vector | ✓ Distinct vector per context |
    | OOV words | ✗ No representation | ✓ Character CNN handles any word |
    | Architecture | Shallow, single lookup | Deep biLSTM |
    | Training objective | Predict neighboring words | Predict next/previous word (LM) |
    | Usage in downstream task | Frozen lookup table | Weighted sum of all layers |

    ---
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
