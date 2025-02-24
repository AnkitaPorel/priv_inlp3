import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.corpus import brown
import nltk
import string
from collections import Counter
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import pandas as pd
import sys

nltk.download('brown')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def preprocess(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    words = [word for word in sentence.split()]
    return " ".join(words)

def build_vocab(corpus, min_freq=5):
    word_counts = Counter()
    for sentence in corpus:
        words = sentence.split()
        word_counts.update(words)
    vocab = {'UNK': 0}
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab, word_counts

def generate_training_data(corpus, vocab, window_size=7):
    context_size = 2 * window_size
    training_data = []
    for sentence in corpus:
        words = sentence.split()
        for i, target_word in enumerate(words):
            target_id = vocab.get(target_word, vocab['UNK'])
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            context_words = [words[j] for j in range(start, end) if j != i]
            context_ids = [vocab.get(word, vocab['UNK']) for word in context_words]
            while len(context_ids) < context_size:
                context_ids.append(vocab['UNK'])
            training_data.append(context_ids + [target_id])
    training_data = np.array(training_data, dtype=np.int64)
    return torch.tensor(training_data).to(device)

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.W_embed = nn.Parameter(torch.empty(vocab_size, embedding_dim))
        self.W_out = nn.Parameter(torch.empty(vocab_size, embedding_dim))
        nn.init.xavier_uniform_(self.W_embed)
        nn.init.xavier_uniform_(self.W_out)
    
    def forward(self, context_ids, target_ids, neg_ids):
        context_emb = self.W_embed[context_ids]
        context_mean = torch.mean(context_emb, dim=1)
        target_emb = self.W_out[target_ids]
        neg_emb = self.W_out[neg_ids]

        pos_score = torch.sum(context_mean * target_emb, dim=1)
        pos_loss = -F.logsigmoid(pos_score).mean()
        
        neg_score = torch.bmm(neg_emb, context_mean.unsqueeze(2)).squeeze(2)
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1).mean()
        
        return pos_loss + neg_loss

def train_cbow(model, training_data, noise_dist, epochs=100, batch_size=512, learning_rate=0.0001, num_neg=5):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_samples = training_data.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for epoch in range(epochs):
        print(f"\nStarting Epoch {epoch + 1}/{epochs}")
        indices = torch.randperm(num_samples).to(device)
        total_loss = 0

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:min(i + batch_size, num_samples)]
            batch = training_data[batch_indices]
            context_ids = batch[:, :-1]
            target_ids = batch[:, -1]
            neg_ids = torch.multinomial(noise_dist, target_ids.size(0) * num_neg, replacement=True).view(target_ids.size(0), num_neg).to(device)

            optimizer.zero_grad()
            loss = model(context_ids, target_ids, neg_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_num = i // batch_size + 1
            if batch_num % max(1, num_batches // 10) == 0 or i + batch_size >= num_samples:
                progress = (batch_num / num_batches) * 100
                print(f"  Progress: {progress:.1f}% ({batch_num}/{num_batches} batches), "
                      f"Batch Loss: {loss.item():.4f}, Avg Loss so far: {total_loss / batch_num:.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} completed, Average Loss: {avg_loss:.4f}")

    return model

def cosine_similarity(vec1, vec2):
    if isinstance(vec1, torch.Tensor):
        vec1 = vec1.detach().cpu().numpy()
    if isinstance(vec2, torch.Tensor):
        vec2 = vec2.detach().cpu().numpy()
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def evaluate_word_similarity(embeddings, wordsim_path):
    human_scores = []
    model_scores = []
    word_pairs = []
    try:
        with open(wordsim_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find file '{wordsim_path}'")
        sys.exit(1)
    
    for line in lines:
        try:
            word1, word2, score = line.strip().split(',')
            if word1 in embeddings and word2 in embeddings:
                sim = cosine_similarity(embeddings[word1], embeddings[word2])
                human_scores.append(float(score))
                model_scores.append(sim)
                word_pairs.append((word1, word2))
        except ValueError:
            continue
    if not human_scores:
        print("Warning: No valid word pairs found in the embeddings")
        return 0.0, [], [], []
    correlation, _ = spearmanr(human_scores, model_scores)
    return correlation, human_scores, model_scores, word_pairs

def save_to_csv(word_pairs, human_scores, model_scores, output_path):
    data = {'Word1': [pair[0] for pair in word_pairs], 'Word2': [pair[1] for pair in word_pairs],
            'Human_Score': human_scores, 'Model_Score': model_scores}
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Results saved to '{output_path}'")

def plot_similarity(human_scores, model_scores, word_pairs, correlation_val):
    human_scores_normalized = [score / 10.0 for score in human_scores]
    plt.figure(figsize=(14, 12))
    plt.scatter(human_scores_normalized, model_scores, alpha=0.5, color='blue', label='Word Pairs')
    for i in range(len(word_pairs)):
        plt.annotate(f"{word_pairs[i][0]}-{word_pairs[i][1]}", 
                     (human_scores_normalized[i], model_scores[i]), 
                     fontsize=5, alpha=0.6, xytext=(5, 5), textcoords='offset points')
    plt.xlabel('Normalized Human Mean Similarity (0-1)')
    plt.ylabel('Cosine Similarity (Model, -1 to 1)')
    plt.title(f'Cosine Similarity vs Human Similarity (Spearman: {correlation_val:.3f})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cbow_word_similarity_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'cbow_word_similarity_plot.png'")
    plt.show()

def main():
    sentences = brown.sents()
    processed_sentences = [preprocess(" ".join(sentence)) for sentence in sentences]

    vocab, word_counts = build_vocab(processed_sentences, min_freq=5)
    print(f"Vocabulary size: {len(vocab)}")
    unk_count = sum(1 for s in processed_sentences for w in s.split() if w not in vocab)
    total_words = sum(len(s.split()) for s in processed_sentences)
    print(f"UNK frequency: {unk_count} ({unk_count / total_words * 100:.2f}%)")

    training_data = generate_training_data(processed_sentences, vocab, window_size=6)
    print(f"Training data size: {training_data.size(0)}")

    id_to_word = {idx: word for word, idx in vocab.items()}
    freqs = np.array([word_counts[id_to_word[idx]] if idx in id_to_word else 0 for idx in range(len(vocab))], dtype=np.float32)
    noise_dist = torch.from_numpy(freqs ** 0.75 / (freqs ** 0.75).sum()).to(device)

    embedding_dim = 300
    model = CBOW(vocab_size=len(vocab), embedding_dim=embedding_dim)
    trained_model = train_cbow(model, training_data, noise_dist, epochs=50, batch_size=512, learning_rate=0.0005, num_neg=5)

    embeddings = trained_model.W_embed.data.cpu().numpy()
    embeddings_dict = {word: embeddings[idx] for word, idx in vocab.items()}
    torch.save(embeddings_dict, '/content/cbow.pt')
    print("CBOW embeddings saved to 'cbow.pt'")

if __name__ == "__main__":
    main()