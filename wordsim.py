# import matplotlib.pyplot as plt
# import torch
# import numpy as np
# from scipy.stats import spearmanr
# import pandas as pd
# import sys

# def cosine_similarity(vec1, vec2):
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# def evaluate_word_similarity(embeddings, wordsim_path):
#     human_scores = []
#     model_scores = []
#     word_pairs = []
#     with open(wordsim_path, 'r') as f:
#         lines = f.readlines()
#     for line in lines:
#         word1, word2, score = line.strip().split(',')
#         if word1 in embeddings and word2 in embeddings:
#             sim = cosine_similarity(embeddings[word1], embeddings[word2])
#             human_scores.append(float(score))
#             model_scores.append(sim)
#             word_pairs.append((word1, word2))
#     correlation, _ = spearmanr(human_scores, model_scores)
#     return correlation, human_scores, model_scores, word_pairs

# def save_to_csv(word_pairs, human_scores, model_scores, output_path):
#     data = {
#         'Word1': [pair[0] for pair in word_pairs],
#         'Word2': [pair[1] for pair in word_pairs],
#         'Human_Score': human_scores,
#         'Model_Score': model_scores
#     }
#     df = pd.DataFrame(data)
#     df.to_csv(output_path, index=False)
#     print(f"Results saved to '{output_path}'")

# def plot_similarity(human_scores, model_scores, word_pairs, correlation_val):
#     human_scores_normalized = [score / 10.0 for score in human_scores]
#     plt.figure(figsize=(14, 12))
#     plt.scatter(human_scores_normalized, model_scores, alpha=0.5, color='blue', label='Word Pairs')

#     for i in range(len(word_pairs)):
#         plt.annotate(f"{word_pairs[i][0]}-{word_pairs[i][1]}", 
#                      (human_scores_normalized[i], model_scores[i]), 
#                      fontsize=5, alpha=0.6, xytext=(5, 5), textcoords='offset points')

#     plt.xlabel('Normalized Human Mean Similarity (0-1)')
#     plt.ylabel('Cosine Similarity (Model, -1 to 1)')
#     plt.title(f'Cosine Similarity vs Human Similarity (Spearman: {correlation_val:.3f})')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
#     plt.tight_layout()

#     plt.savefig('word_similarity_plot.png', dpi=300, bbox_inches='tight')
#     print("Plot saved as 'word_similarity_plot.png'")
    
#     plt.show()

# def main():
#     if len(sys.argv) != 2:
#         print("Usage: python3 wordsim.py <xyz.pt>")
#         sys.exit(0)

#     wordsim_path = 'wordsim353crowd.csv'
#     embedding_path = sys.argv[1]

#     embeddings = torch.load(embedding_path)

#     correlation_val, human_scores, model_scores, word_pairs = evaluate_word_similarity(embeddings, wordsim_path)
#     print(f"Spearman Correlation: {correlation_val}")
#     plot_similarity(human_scores, model_scores, word_pairs, correlation_val)

#     csv_output_path = 'word_similarity.csv'
#     save_to_csv(word_pairs, human_scores, model_scores, csv_output_path)

# if __name__ == "__main__":
#     main()

import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import sys

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
    data = {
        'Word1': [pair[0] for pair in word_pairs],
        'Word2': [pair[1] for pair in word_pairs],
        'Human_Score': human_scores,
        'Model_Score': model_scores
    }
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

    plt.savefig('word_similarity_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'word_similarity_plot.png'")
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 wordsim.py <trained_model.pt>")
        sys.exit(0)

    wordsim_path = 'wordsim353crowd.csv'
    embedding_path = sys.argv[1]

    try:
        embeddings = torch.load(embedding_path)
        correlation_val, human_scores, model_scores, word_pairs = evaluate_word_similarity(embeddings, wordsim_path)
        print(f"Spearman Correlation: {correlation_val}")
        if human_scores:
            plot_similarity(human_scores, model_scores, word_pairs, correlation_val)
            save_to_csv(word_pairs, human_scores, model_scores, 'word_similarity.csv')
    except FileNotFoundError:
        print(f"Error: Could not find embedding file '{embedding_path}'")
        sys.exit(1)

if __name__ == "__main__":
    main()