import random
import pandas as pd

def fake_model_prediction():
    probs = {
        "Fitzpatrick I": random.uniform(0, 1),
        "Fitzpatrick II": random.uniform(0, 1),
        "Fitzpatrick III": random.uniform(0, 1),
        "Fitzpatrick IV": random.uniform(0, 1),
        "Fitzpatrick V": random.uniform(0, 1),
        "Fitzpatrick VI": random.uniform(0, 1)
    }
    total = sum(probs.values())
    normalized = {k: (100*round(v / total, 3)) for k, v in probs.items()}
    
    # Return as a single-row DataFrame for now
    df = pd.DataFrame([normalized])
    df.index = ['Probability']


    return df