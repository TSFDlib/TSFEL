import pandas as pd
import numpy as np
#import src.feature_extraction.utils.read_json as rj
from TSFEL.tsfel.utils.read_json import feat_extract

def extract_features(sig, label, cfg, segment=True, window_size=5):
    feat_val = None
    labels = None
    features = []
    row_idx = []
    print("*** Feature extraction started ***")

    #header = np.array(pd.read_csv('TSFEL/tests/input_signal/Signal.txt', delimiter=',', header=None))[0, 1:]
    if segment:
        sig = [sig[i:i + window_size] for i in range(0, len(sig), window_size)]
    for wind_idx, wind_sig in enumerate(sig):
        if len(sig.shape) >= 3:
            for i in range(sig.shape[1]):
                _row_idx, labels = feat_extract(cfg, wind_sig[i], label)
                row_idx.append(_row_idx)
        else:
            row_idx, labels = feat_extract(cfg, wind_sig, label)
        if wind_idx == 0:
            feat_val = row_idx
        else:
            feat_val = np.vstack((feat_val, row_idx))
    feat_val = np.array(feat_val)
    d = {str(lab): feat_val[:,idx] for idx, lab in enumerate(labels)}
    df = pd.DataFrame(data=d)
    df.to_csv('TSFEL/tsfel/utils/Features.csv', sep=',', encoding='utf-8', index_label="Sample")
    print("*** Feature extraction finished ***")

    return df
