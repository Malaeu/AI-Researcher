import nltk
from nltk.corpus import stopwords
import string
import json 
from tqdm import tqdm 
from collections import Counter
import numpy as np
import random
import pandas as pd
import argparse
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

def clean_text(text):
    """Enhanced text cleaning function"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and numbers but keep important punctuation
    text = re.sub(r'[^a-zA-Z\s\.\,\!\?]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def process_text(input_text, tokenize=False):
    """Enhanced text processing with better context preservation"""
    # Clean the text first
    cleaned_text = clean_text(input_text)
    
    # Define the list of stopwords but keep important ones
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'but', 'however', 'although', 'though', 'unless', 'between', 'against'}
    
    # Split into sentences for better context preservation
    sentences = re.split(r'[.!?]+', cleaned_text)
    
    processed_sentences = []
    for sentence in sentences:
        # Tokenize the sentence
        words = sentence.split()
        
        # Remove stopwords while preserving context
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        if filtered_words:  # Only add non-empty sentences
            processed_sentences.append(' '.join(filtered_words))
    
    # Join sentences back together
    return ' . '.join(processed_sentences)

def get_contextual_embeddings(texts, model):
    """Get embeddings with better context handling"""
    # Process texts while preserving context
    processed_texts = [process_text(text) for text in texts]
    
    # Generate embeddings in batches to handle memory efficiently
    batch_size = 32
    embeddings = []
    
    for i in tqdm(range(0, len(processed_texts), batch_size), desc="Generating embeddings"):
        batch = processed_texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

def plot_string_occurrences(strings_list):
    # Count occurrences of each string
    occurrences = Counter(strings_list)
    
    # Count how many strings have each occurrence count
    count_of_occurrences = Counter(occurrences.values())
    
    # Extracting the data for plotting
    x = sorted(count_of_occurrences.keys())
    y = [count_of_occurrences[occ] for occ in x]
    
    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color='skyblue')
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Number of Strings')
    plt.title('Frequency of String Occurrences')
    plt.xticks(x)
    plt.grid(axis='y')
    plt.show()

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def find_representative_paper(cluster, similarity_matrix, labels):
    cluster_indices = [i for i, label in enumerate(labels) if label == cluster]
    cluster_sims = similarity_matrix[cluster_indices][:, cluster_indices]
    avg_sims = cluster_sims.mean(axis=1)
    representative_index = cluster_indices[avg_sims.argmax()]
    return representative_index

def find_top_n_papers(representative_index, similarity_matrix, n=5):
    sims = similarity_matrix[representative_index]
    closest_indices = np.argsort(-sims)[:n]  # Sort in descending order and get top-n
    return closest_indices

def concatenate_idea(idea_k, idea_v):
    output = ""
    output += idea_k + "\n"
    
    if isinstance(idea_v, dict):
        output += "Problem: " + idea_v["Problem"] + "\n"
        output += "Existing Methods: " + idea_v["Existing Methods"] + "\n"
        output += "Motivation: " + idea_v["Motivation"] + "\n"
        output += "Proposed Method: " + idea_v["Proposed Method"] + "\n"
        output += "Experiment Plan: " + idea_v["Experiment Plan"] + "\n"
    else:
        output += str(idea_v) + "\n"

    return output


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, default="bias", help='cache file name')
    parser.add_argument('--cache_name', type=str, default="bias", help='cache file name')
    parser.add_argument("--save_similarity_matrix", action='store_true', help="whether to save the computed similarity matrix")
    parser.add_argument("--load_similarity_matrix", action='store_true', help="whether to load the computed similarity matrix")
    parser.add_argument("--num_ideas", type=int, default=1000, help="top n ideas to consider")
    parser.add_argument("--model_name", type=str, default="all-mpnet-base-v2", help="name of the sentence transformer model to use")
    args = parser.parse_args()

    all_ideas = []
    with open(os.path.join(args.cache_dir, args.cache_name + ".json"), "r") as f:
        ideas_json = json.load(f)
        for ideas_dict in ideas_json["ideas"]:
            for idea_k, idea_v in ideas_dict.items():
                try:
                    all_ideas.append(concatenate_idea(idea_k, idea_v))
                except: 
                    continue
    
    print("#ideas: ", len(all_ideas))
    all_ideas = all_ideas[:args.num_ideas]
    
    # Initialize the model with a more powerful variant
    model = SentenceTransformer(args.model_name)

    if args.load_similarity_matrix:
        similarity_matrix = np.load(os.path.join(args.cache_dir, args.cache_name + "_similarity_matrix.npy"))
    elif args.save_similarity_matrix:
        # Use contextual embeddings
        embeddings = get_contextual_embeddings(all_ideas, model)
        similarity_matrix = cosine_similarity(embeddings)
        # Setting the diagonal to 0
        np.fill_diagonal(similarity_matrix, 0)
        np.save(os.path.join(args.cache_dir, args.cache_name + "_similarity_matrix.npy"), similarity_matrix)

    nn_similarity = []
    nn_similarity_idx = []
    avg_similarity = []
    for i in range(len(all_ideas)):
        nn_similarity.append(np.max(similarity_matrix[i]))
        nn_similarity_idx.append(np.argmax(similarity_matrix[i]))
        avg_similarity.append(np.sum(similarity_matrix[i]) / (len(all_ideas) - 1))

    highest_nn_similarity = np.argmax(nn_similarity)
    print("\nIdea with Highest NN Similarity:\n", all_ideas[highest_nn_similarity])
    print("\nMost Similar Idea:\n", all_ideas[nn_similarity_idx[highest_nn_similarity]])
    print("\nSimilarity Score:", nn_similarity[highest_nn_similarity])

    lowest_nn_similarity = np.argmin(nn_similarity)
    print("\nIdea with Lowest NN Similarity:\n", all_ideas[lowest_nn_similarity])
    print("\nMost Similar Idea:\n", all_ideas[nn_similarity_idx[lowest_nn_similarity]])
    print("\nSimilarity Score:", nn_similarity[lowest_nn_similarity])

    highest_avg_similarity = np.argmax(avg_similarity)
    print("\nIdea with Highest Avg Similarity:\n", all_ideas[highest_avg_similarity])
    print("\nMost Similar Idea:\n", all_ideas[nn_similarity_idx[highest_avg_similarity]])
    print("\nSimilarity Score:", avg_similarity[highest_avg_similarity])

    lowest_avg_similarity = np.argmin(avg_similarity)
    print("\nIdea with Lowest Avg Similarity:\n", all_ideas[lowest_avg_similarity])
    print("\nMost Similar Idea:\n", all_ideas[nn_similarity_idx[lowest_avg_similarity]])
    print("\nSimilarity Score:", avg_similarity[lowest_avg_similarity])

    print("\n\nCorpus level metrics:")
    print("Avg NN Similarity: ", np.mean(nn_similarity))
    print("Avg Avg Similarity: ", np.mean(avg_similarity))

    interval = 0.05
    min_similarity = np.min(nn_similarity)
    max_similarity = np.max(nn_similarity)

    buckets = {}
    current_interval = min_similarity
    while current_interval < max_similarity:
        next_interval = current_interval + interval
        buckets[(current_interval, next_interval)] = []
        current_interval = next_interval

    for i in range(len(all_ideas)):
        for j in range(i + 1, len(all_ideas)):
            similarity = similarity_matrix[i][j]
            for bucket in buckets:
                if bucket[0] <= similarity < bucket[1]:
                    buckets[bucket].append((all_ideas[i], all_ideas[j], similarity))
                    break

    print("\n\nIdea pairs by similarity interval:")
    for bucket in buckets:
        if buckets[bucket]:
            sampled_pair = random.choice(buckets[bucket])
            print(f"\nInterval {bucket[0]:.2f} - {bucket[1]:.2f}:")
            print(f"{sampled_pair[0]}\nand\n{sampled_pair[1]}\n\nSimilarity: {sampled_pair[2]:.2f}\n")

    print("\nSimilarity Statistics:")
    print(f"Average NN Similarity: {np.mean(nn_similarity):.3f}")
    print(f"Median NN Similarity: {np.median(nn_similarity):.3f}")
    print(f"Std Dev NN Similarity: {np.std(nn_similarity):.3f}")
    print(f"Average Overall Similarity: {np.mean(avg_similarity):.3f}")
