import json
import jsonlines
import torchaudio
import os

# Paths to your input directories
json_dir = '/Users/ishita/med_asr_diarization/Data/gt_chunked_data_may_27'
audio_dir = '/Users/ishita/med_asr_diarization/Data/AudioRecordings'
output_file = '/Users/ishita/med_asr_diarization/output_data.json'

# Function to generate the desired output format
def process_data(json_dir, output_file, audio_dir):
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    output_data = []
    
    for json_file in json_files:
        base_name = os.path.splitext(json_file)[0]
        json_path = os.path.join(json_dir, json_file)
        audio_file = os.path.join(audio_dir, f"{base_name}.mp3")
        
        with open(json_path, 'r') as infile:
            data = json.load(infile)
        
        waveform, sr = torchaudio.load(audio_file)
        duration = waveform.shape[1] / sr
        
        start_time = 0
        chunk_duration = 30

        for obj in data:
            whisper_chunk = obj['whisper_chunk']
            gt_chunk = obj['gt_chunk']
            end_time = min(start_time + chunk_duration, duration)
            
            output_data.append({
                "whisper_chunk": base_name,
                "start_time": start_time,
                "end_time": end_time,
                "gt_chunk": gt_chunk
            })
            
            start_time += chunk_duration
            if start_time >= duration:
                break
    with open(output_file, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

process_data(json_dir, output_file, audio_dir)
