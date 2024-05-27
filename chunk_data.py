import os
import glob
from multiprocessing import Pool
import json
from Levenshtein import ratio as levenshtein_ratio
import re
import argparse

speaker_attribution_pattern = re.compile(r'\b[A-Za-z]+:\s*')
extra_whitespace_pattern = re.compile(r'\s+')

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

# Function to reconstruct text from word chunks
def reconstruct_text_from_chunks(chunks):
    chunk_texts = []
    for chunk in chunks:
        chunk_text = ' '.join([word['word'] for word in chunk])
        chunk_texts.append(chunk_text)
    return chunk_texts

def find_best_matching_chunk(gt_text, whisper_chunk, start_idx=0):
    word_boundaries = [m.start() for m in re.finditer(r'\b', gt_text)]

    expand_start = max(0, start_idx - len(whisper_chunk))
    expand_end = min(len(gt_text), start_idx + len(whisper_chunk)*2)

    word_boundaries = [idx for idx in word_boundaries if expand_start <= idx <= expand_end]

    max_ratio = float('-inf')
    best_start = 0
    best_end = 0

    for i in word_boundaries:
        for j in word_boundaries:
            if j <= i:
                continue
            cleaned_text = speaker_attribution_pattern.sub('', gt_text[i:j])
            cleaned_text = extra_whitespace_pattern.sub(' ', cleaned_text)
            current_ratio = levenshtein_ratio(cleaned_text, whisper_chunk, score_cutoff=(max_ratio if max_ratio > 0 else 0))

            if current_ratio > max_ratio:
                max_ratio = current_ratio
                best_start = i
                best_end = j

    return best_start, best_end

# Function to find the ground truth chunk with speaker attribution
def get_gt_chunk_with_speaker_attribution(gt_text, whisper_chunk, start_idx):
    best_start, best_end = find_best_matching_chunk(gt_text, whisper_chunk, start_idx)

    # Extract the chunk with speaker attribution
    gt_chunk_with_speaker = gt_text[best_start:best_end]

    # Ensure we include any partial speaker attribution at the start
    speaker_start = best_start
    while speaker_start > 0 and gt_text[speaker_start - 1] != '\n':
        speaker_start -= 1

    # Extract the speaker label
    speaker_label = gt_text[speaker_start:speaker_start+1].strip()

    # Combine the speaker label with the chunk
    if speaker_start < best_start:
        gt_chunk_with_speaker = f"{speaker_label}: {gt_chunk_with_speaker}".strip()

    return gt_chunk_with_speaker, best_end

def process_file(args):
    print(f"Processing file {args[0]}")
    whisper_file, ground_truth_dir, output_dir, duration = args
    base_name = os.path.basename(whisper_file).replace('.json', '')
    ground_truth_file = os.path.join(ground_truth_dir, f"{base_name}.txt")
    output_file = os.path.join(output_dir, f"{base_name}.json")

    # Load the Whisper data
    with open(whisper_file, 'r') as f:
        whisper_data = json.load(f)

    # Load the ground truth text
    with open(ground_truth_file, 'r') as f:
        ground_truth_text = f.read()

    # Extract the text and timing information from the Whisper dictionary
    whisper_words = whisper_data['words']

    # Chunk the Whisper words into segments
    whisper_chunks = chunk_data_by_time(whisper_words, duration=duration)

    whisper_chunk_texts = reconstruct_text_from_chunks(whisper_chunks)

    # Process each Whisper chunk and find the corresponding ground truth chunk
    gt_whisper_chunks = []
    start_idx = 0
    for idx, whisper_chunk in enumerate(whisper_chunk_texts):
        gt_chunk, start_idx = get_gt_chunk_with_speaker_attribution(ground_truth_text, whisper_chunk, start_idx)
        gt_whisper_chunks.append({'whisper_chunk': whisper_chunk, 'gt_chunk': gt_chunk})

    # Save the results to a JSON file
    with open(output_file, 'w') as f:
        json.dump(gt_whisper_chunks, f, indent=4)

def main(whisper_dir, ground_truth_dir, output_dir, duration):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all the Whisper JSON files
    whisper_files = glob.glob(os.path.join(whisper_dir, '*.json'))

    # Prepare the arguments for multiprocessing
    args = []
    for whisper_file in whisper_files:
        args.append((whisper_file, ground_truth_dir, output_dir, duration))

    with Pool(processes=10) as pool:
        pool.map(process_file, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process all Whisper transcripts and ground truth text files.")
    parser.add_argument('whisper_dir', type=str, help="Directory containing the Whisper JSON files.")
    parser.add_argument('ground_truth_dir', type=str, help="Directory containing the ground truth text files.")
    parser.add_argument('output_dir', type=str, help="Directory to save the output JSON files.")
    parser.add_argument('--duration', type=int, default=30, help="Duration for chunking in seconds. Default is 30 seconds.")

    args = parser.parse_args()

    main(args.whisper_dir, args.ground_truth_dir, args.output_dir, args.duration)