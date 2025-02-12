import random
import pandas as pd
from sklearn.model_selection import train_test_split


# Load your combined data (edges with node features)
data = pd.read_csv('your_combined_edge_data.csv')

# Add the target column for existing links
data['target'] = 1  # All existing links are labeled with 1 (because they exist)




# Generate a list of all possible node pairs (sourceId, targetId) combinations
# Assuming nodes range from 1 to 400k, adjust accordingly
all_node_ids = data['sourceId'].append(data['targetId']).unique()

# Create negative samples (pairs of nodes that don't have an edge)
negative_samples = []

# Loop to generate non-existing links (ensure no duplicates)
while len(negative_samples) < len(data):  # Number of negative samples = number of positive samples
    source = random.choice(all_node_ids)
    target = random.choice(all_node_ids)
    if source != target and not ((data['sourceId'] == source) & (data['targetId'] == target)).any():
        negative_samples.append([source, target])

# Convert the negative samples to a DataFrame
negative_samples_df = pd.DataFrame(negative_samples, columns=['sourceId', 'targetId'])

# Add target column to the negative samples (set to 0 because these are non-existing links)
negative_samples_df['target'] = 0




# Combine positive and negative samples
final_data = pd.concat([data[['sourceId', 'targetId', 'target']], negative_samples_df])

# Shuffle the data (optional but recommended)
final_data = final_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the final combined data
print(final_data.head())




from autogluon.tabular import TabularDataset, TabularPredictor

# Load the final dataset
train_data = TabularDataset(final_data)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


# Train the model
predictor = TabularPredictor(label='target').fit(train_data)

# Optionally, evaluate the model (if you have a test set)
# performance = predictor.evaluate(test_data)

new_data = pd.read_csv('new_data.csv')  # Your new data with source and target node features
predictions = predictor.predict(new_data)

