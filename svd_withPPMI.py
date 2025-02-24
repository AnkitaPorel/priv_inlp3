import numpy as np
import torch
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
from collections import Counter
from nltk.corpus import brown
import nltk
import string
from scipy.stats import spearmanr

nltk.download('brown')

def preprocess(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return sentence

def build_vocab(corpus, min_freq=5):
    word_counts = Counter()
    for sentence in corpus:
        words = sentence.split()
        word_counts.update(words)

    vocab = {'UNK': 0}
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab

def build_co_occurrence_matrix(corpus, vocab, window_size=3):
    vocab_size = len(vocab)
    co_matrix = lil_matrix((vocab_size, vocab_size), dtype=np.float32)
    word_counts = Counter()
    total_pairs = 0

    for sentence in corpus:
        words = sentence.split()
        for i, word in enumerate(words):
            if word not in vocab:
                word = 'UNK'
            word_id = vocab[word]
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            for j in range(start, end):
                if j == i:
                    continue
                context_word = words[j]
                if context_word not in vocab:
                    context_word = 'UNK'
                context_id = vocab[context_word]
                co_matrix[word_id, context_id] += 1
                word_counts[word_id] += 1
                word_counts[context_id] += 1
                total_pairs += 1

    co_matrix = co_matrix.tocoo()
    rows, cols = co_matrix.row, co_matrix.col
    data = co_matrix.data
    ppmi_data = []
    for i, j, count in zip(rows, cols, data):
        p_wc = count / total_pairs
        p_w = word_counts[i] / total_pairs
        p_c = word_counts[j] / total_pairs
        pmi = np.log2(max(p_wc / (p_w * p_c), 1e-10))
        ppmi = max(pmi, 0)
        ppmi_data.append(ppmi)
    
    co_matrix = lil_matrix((vocab_size, vocab_size), dtype=np.float32)
    co_matrix[rows, cols] = ppmi_data
    return co_matrix

def main():
    sentences = brown.sents()
    processed_sentences = [preprocess(" ".join(sentence)) for sentence in sentences]

    vocab = build_vocab(processed_sentences, min_freq=2)

    co_matrix = build_co_occurrence_matrix(processed_sentences, vocab, window_size=5)

    embedding_dim = 100
    U, s, VT = svds(co_matrix, k=embedding_dim)

    word_embeddings = U / np.linalg.norm(U, axis=1, keepdims=True)
    embeddings_dict = {word: torch.tensor(word_embeddings[idx], dtype=torch.float32) for word, idx in vocab.items()}
    torch.save(embeddings_dict, '/content/svd.pt')
    print("Word embeddings saved to 'svd.pt'")
    
if __name__ == "__main__":
    main()