import os
import numpy as np
import kagglehub
from load_audio import load_and_clean_audio
from mfcc_features import get_audio_features

def run_preprocessing_pipeline():
    X = []
    labels = []
    actor_ids = []
    
    processed_filenames = set()
    
    emotion_map = {
        "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
        "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
    }
    
    print("Querying kagglehub for the local dataset location...")
    DYNAMIC_DATASET_PATH = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
    print(f"Dataset located at: {DYNAMIC_DATASET_PATH}")
    
    for root, _, files in os.walk(DYNAMIC_DATASET_PATH):
        for file in files:
            if file.endswith(".wav"):
                if file in processed_filenames:
                    continue
                
                filename_without_ext, _ = os.path.splitext(file)
                parts = filename_without_ext.split("-")
                if len(parts) == 7:
                    emotion_code = parts[2]
                    actor_code = parts[6]
                    
                    if emotion_code in emotion_map:
                        full_path = os.path.join(root, file)
                        
                        data, sr = load_and_clean_audio(full_path)
                        features = get_audio_features(data, sr)
                        
                        X.append(features)
                        labels.append(emotion_map[emotion_code])
                        actor_ids.append(int(actor_code))
                        
                        processed_filenames.add(file)
                        
    X = np.array(X)
    y = np.array(labels)
    actors = np.array(actor_ids)
    
    np.save("X_data.npy", X)
    np.save("y_data.npy", y)
    np.save("actors_data.npy", actors)
    print(f"Deduplicated extraction complete! Processed {len(X)} unique files across {len(np.unique(actors))} speakers.")

if __name__ == "__main__":
    run_preprocessing_pipeline()
