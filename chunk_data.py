import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# Load the Whisper dictionary
with open('/Users/ishita/Downloads/CAR0003_whisper.json', 'r') as f:
    whisper_data = json.load(f)

# Load the ground truth text
with open('/Users/ishita/Downloads/CAR0003_ground_truth.txt', 'r') as f:
    ground_truth_text = f.read()


# Function to chunk the data into 30-second segments
def chunk_data_by_time(data, duration=30):
    chunks = []
    current_chunk = []
    current_time = 0

    for item in data:
        if current_time + item['end'] - item['start'] <= duration:
            current_chunk.append(item)
            current_time += item['end'] - item['start']
        else:
            chunks.append(current_chunk)
            current_chunk = [item]
            current_time = item['end'] - item['start']

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Extract the text and timing information from the Whisper dictionary
whisper_words = whisper_data['words']

# Chunk the Whisper words into 30-second segments
whisper_chunks = chunk_data_by_time(whisper_words, duration=30)

# Function to reconstruct text from word chunks
def reconstruct_text_from_chunks(chunks):
    chunk_texts = []
    for chunk in chunks:
        chunk_text = ' '.join([word['word'] for word in chunk])
        chunk_texts.append(chunk_text)
    return chunk_texts

# Reconstruct text from Whisper chunks
whisper_chunk_texts = reconstruct_text_from_chunks(whisper_chunks)

# Function to find potential matching chunks in the ground truth text based on first word match
def find_potential_matches(gt_text, start_word, length):
    words = gt_text.split()
    matches = []
    start_indices = [i for i, word in enumerate(words) if word == start_word]
    
    for start_idx in start_indices:
        end_idx = start_idx
        current_length = 0
        while end_idx < len(words) and current_length < length:
            current_length += len(words[end_idx]) + 1  # +1 for the space
            end_idx += 1
        
        matches.append(' '.join(words[start_idx:end_idx]))
    
    return matches

# Perform similarity match for each Whisper chunk with potential ground truth chunks
vectorizer = TfidfVectorizer()

for idx, whisper_text in enumerate(whisper_chunk_texts):
    whisper_words = whisper_text.split()
    if len(whisper_words) < 2:
        continue
    start_word = whisper_words[0]
    potential_matches = find_potential_matches(ground_truth_text, start_word, len(whisper_text))
    
    if not potential_matches:
        print(f"No matches found for Whisper Chunk {idx + 1}")
        continue
    
    all_texts = [whisper_text] + potential_matches
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    whisper_vector = tfidf_matrix[0:1]
    potential_vectors = tfidf_matrix[1:]
    
    similarities = cosine_similarity(whisper_vector, potential_vectors).flatten()
    most_similar_idx = similarities.argmax()
    
    # Find the original chunk for display
    matched_gt_chunk = potential_matches[most_similar_idx]
    
    print(f"Whisper Chunk {idx + 1} is most similar to Ground Truth Chunk {most_similar_idx + 1}")
    print(f"Cosine Similarity: {similarities[most_similar_idx]:.4f}")
    print(f"Whisper Chunk Text: {whisper_text}")
    print(f"Ground Truth Chunk Text: {matched_gt_chunk}\n")