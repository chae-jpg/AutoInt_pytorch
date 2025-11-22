import pandas as pd
import numpy as np
import os

# Scaling for continuous features should be applied after this preprocessing process.

def preprocess():
    source_path = './train_examples.txt'
    output_dir = './'
    
    if not os.path.exists(source_path):
        print(f"Source file not found: {source_path}")
        return

    print("Reading data...")
    # Criteo format: Label, I1-I13, C1-C26
    # We assume tab separation based on the file view
    df = pd.read_csv(source_path, sep='\t', header=None)
    
    # Fill missing values
    # Continuous features (columns 1-13): fill with 0
    df.iloc[:, 1:14] = df.iloc[:, 1:14].fillna(0)
    # Categorical features (columns 14-39): fill with '0' (string)
    df.iloc[:, 14:] = df.iloc[:, 14:].fillna('0')

    # Label is column 0
    label = df.iloc[:, 0]
    
    # Continuous features
    continuous = df.iloc[:, 1:14]
    
    # Categorical features
    categorical = df.iloc[:, 14:]
    
    # Encode categorical features
    print("Encoding categorical features...")
    feature_sizes = []
    # For continuous features, we just put 1 as size (placeholder, as they use index 0)
    # But DeepFM implementation expects feature_sizes to contain sizes for ALL fields?
    # Let's check DeepFM.py. 
    # self.fm_first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, 1) ...])
    # It creates embeddings for ALL features.
    # For continuous features, Xi is 0. So embedding(0) is used. Size can be 1.
    for i in range(13):
        feature_sizes.append(1)
        
    encoded_categorical = []
    for i in range(26):
        col_idx = 14 + i
        col_data = categorical.iloc[:, i].astype(str)
        unique_vals = col_data.unique()
        val_map = {val: idx for idx, val in enumerate(unique_vals)}
        encoded_col = col_data.map(val_map).values
        encoded_categorical.append(encoded_col)
        feature_sizes.append(len(unique_vals))
        
    encoded_categorical = np.stack(encoded_categorical, axis=1)
    
    # Combine: Continuous, Categorical, Label
    # CriteoDataset expects: [Continuous, Categorical, Label]
    # because it does: data.iloc[:, :-1] as features, data.iloc[:, -1] as target
    # and splits features into continuous (first 13) and categorical (rest).
    
    processed_data = np.concatenate([continuous.values, encoded_categorical, label.values.reshape(-1, 1)], axis=1)
    
    # Save all data to train.txt as the original code expects train.txt to contain both train and val sets
    # and uses SubsetRandomSampler to split them.
    print(f"Saving train.txt ({len(processed_data)} rows)...")
    np.savetxt(os.path.join(output_dir, 'train.txt'), processed_data, fmt='%s', delimiter=',')
    
    # Save the last part as test.txt as well, just in case
    train_size = int(len(processed_data) * 0.9)
    test_data = processed_data[train_size:]
    print(f"Saving test.txt ({len(test_data)} rows)...")
    np.savetxt(os.path.join(output_dir, 'test.txt'), test_data, fmt='%s', delimiter=',')
    
    print("Saving feature_sizes.txt...")
    np.savetxt(os.path.join(output_dir, 'feature_sizes.txt'), feature_sizes, fmt='%d', delimiter=',')
    
    print("Done.")

if __name__ == '__main__':
    preprocess()
