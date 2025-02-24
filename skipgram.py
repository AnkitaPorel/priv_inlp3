import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from nltk.corpus import brown
import nltk
import string
import torch.nn as nn
from collections import Counter
from scipy.stats import spearmanr

nltk.download('brown')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def preprocess(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    words = sentence.split()
    return ' '.join(word for word in words)

def build_vocab(corpus, min_freq=1):
    word_counts = Counter()
    for sentence in corpus:
        words = sentence.split()
        word_counts.update(words)
    vocab = {'UNK': 0}
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def generate_training_data(corpus, vocab, window_size=2):
    training_data = []
    word_counts = Counter()
    for sentence in corpus:
        word_counts.update(sentence.split())
    total_words = sum(word_counts.values())
    
    for sentence in corpus:
        words = sentence.split()
        for i, target_word in enumerate(words):
            freq = word_counts[target_word] / total_words
            p_keep = min(1.0, (1e-4 / freq) ** 0.5 + 1e-4 / freq)
            if np.random.random() > p_keep:
                continue
            target_id = vocab.get(target_word, vocab['UNK'])
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            context_words = [words[j] for j in range(start, end) if j != i]
            context_ids = [vocab.get(word, vocab['UNK']) for word in context_words]
            for context_id in context_ids:
                training_data.append([target_id, context_id])
    
    training_data = np.array(training_data, dtype=np.int64)
    return torch.tensor(training_data).to(device)

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.W_target = nn.Parameter(torch.empty(vocab_size, embedding_dim))
        self.W_context = nn.Parameter(torch.empty(vocab_size, embedding_dim))
        nn.init.xavier_uniform_(self.W_target)
        nn.init.xavier_uniform_(self.W_context)
    
    def forward(self, target_ids, context_ids, neg_ids):
        target_emb = self.W_target[target_ids]
        context_emb = self.W_context[context_ids]
        neg_emb = self.W_context[neg_ids]
        
        pos_score = torch.sum(target_emb * context_emb, dim=1)
        pos_loss = -F.logsigmoid(pos_score).mean()
        
        neg_score = torch.bmm(neg_emb, target_emb.unsqueeze(2)).squeeze(2)
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1).mean()
        
        return pos_loss + neg_loss

def train_skipgram(model, training_data, noise_dist, epochs=15, batch_size=512, learning_rate=0.01, num_neg=5):
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
            target_ids = batch[:, 0]
            context_ids = batch[:, 1]
            neg_ids = torch.multinomial(noise_dist, target_ids.size(0) * num_neg, replacement=True).view(target_ids.size(0), num_neg).to(device)
            
            optimizer.zero_grad()
            loss = model(target_ids, context_ids, neg_ids)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_num = i // batch_size + 1
            if batch_num % max(1, num_batches // 10) == 0 or i + batch_size >= num_samples:
                progress = (batch_num / num_batches) * 100
                print(f"  Progress: {progress:.1f}% ({batch_num}/{num_batches} batches), "
                      f"Batch Loss: {loss.item():.4f}, Avg Loss so far: {total_loss / batch_num:.4f}")
        
        print(f"Epoch {epoch + 1} completed, Average Loss: {total_loss / num_batches:.4f}")
    
    return model

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def evaluate_word_similarity(embeddings, wordsim_path):
    human_scores = []
    model_scores = []
    with open(wordsim_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        word1, word2, score = line.strip().split(',')
        if word1 in embeddings and word2 in embeddings:
            sim = cosine_similarity(embeddings[word1], embeddings[word2])
            human_scores.append(float(score))
            model_scores.append(sim)
    correlation, _ = spearmanr(human_scores, model_scores)
    return correlation

def main():
    sentences = brown.sents()
    processed_sentences = [preprocess(" ".join(sentence)) for sentence in sentences]
    
    vocab = build_vocab(processed_sentences, min_freq=5)
    print(f"Vocabulary size: {len(vocab)}")
    
    training_data = generate_training_data(processed_sentences, vocab, window_size=5)
    print(f"Training data size: {training_data.size(0)}")
    
    word_counts = Counter()
    for sentence in processed_sentences:
        word_counts.update(sentence.split())
    id_to_word = {idx: word for word, idx in vocab.items()}
    freqs = np.array([word_counts[id_to_word[idx]] if idx in id_to_word else 0 for idx in range(len(vocab))], dtype=np.float32)
    noise_dist = torch.from_numpy(freqs ** 0.75 / (freqs ** 0.75).sum()).to(device)
    
    embedding_dim = 100
    model = SkipGram(vocab_size=len(vocab), embedding_dim=embedding_dim).to(device)
    trained_model = train_skipgram(model, training_data, noise_dist, epochs=15, batch_size=512, learning_rate=0.001, num_neg=5)
    
    embeddings = trained_model.W_target.data.cpu().numpy()
    embeddings_dict = {word: embeddings[idx] for word, idx in vocab.items()}
    torch.save(embeddings_dict, '/content/skipgram.pt')
    print("Skip-gram embeddings saved to 'skipgram.pt'")

if __name__ == "__main__":
    main()