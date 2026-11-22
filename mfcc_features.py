import numpy as np
import librosa

def get_audio_features(data, sr):
    mfcc = librosa.feature.mfcc(
        y=data,
        sr=sr,
        n_mfcc=13
    )
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
        
    delta_mean = np.mean(delta_mfcc, axis=1)
    delta_std = np.std(delta_mfcc, axis=1)
    delta2_mean = np.mean(delta2_mfcc, axis=1)
    delta2_std = np.std(delta2_mfcc, axis=1)

    rms = librosa.feature.rms(y = data)
    zcr = librosa.feature.zero_crossing_rate(y = data)

    
    rms_mean = np.mean(rms, axis = 1)
    rms_std = np.std(rms, axis = 1)
    zcr_mean = np.mean(zcr, axis = 1)
    zcr_std = np.std(zcr, axis = 1)

    features = np.hstack(
        (
            mfcc_mean, mfcc_std,
	        delta_mean, delta_std,
	        delta2_mean, delta2_std,
            rms_mean, rms_std,
            zcr_mean, zcr_std
        )
    )
    
    return features     