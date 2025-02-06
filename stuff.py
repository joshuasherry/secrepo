import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

# Load your data (replace with your actual file paths)
edges_df = pd.read_csv('edges.csv')  # Contains sourceId, targetId, edgeName, timeStamp
featureList_df = pd.read_csv('featureList.csv')  # Contains nodeID, nodeType, createdTimestamp, updatedTimestamp, etc.

# Merge edge data with node feature data based on sourceId and targetId
edges_df = edges_df.merge(featureList_df[['nodeID', 'nodeType', 'createdTimestamp', 'updatedTimestamp', 'appName', 
                                           'version', 'isClientSecretManaged', 'appStatus', 'isApproved', 'categoryPrefix', 
                                           'targetScheme', 'appProtocol', 'resourceName', 'isActive', 'readTimeout', 
                                           'targetUri', 'serviceContextRoot', 'endpointUri', 'geolocation']], 
                           left_on='sourceId', right_on='nodeID', suffixes=('_source', '_target'), how='left')

edges_df = edges_df.merge(featureList_df[['nodeID', 'nodeType', 'createdTimestamp', 'updatedTimestamp', 'appName', 
                                           'version', 'isClientSecretManaged', 'appStatus', 'isApproved', 'categoryPrefix', 
                                           'targetScheme', 'appProtocol', 'resourceName', 'isActive', 'readTimeout', 
                                           'targetUri', 'serviceContextRoot', 'endpointUri', 'geolocation']], 
                           left_on='targetId', right_on='nodeID', suffixes=('_source', '_target'), how='left')

# Step 1: Feature Engineering

# Calculate time differences between edge timestamp and node creation timestamp
edges_df['time_difference_source'] = edges_df['timeStamp'] - edges_df['createdTimestamp_source']
edges_df['time_difference_target'] = edges_df['timeStamp'] - edges_df['createdTimestamp_target']

# If the time difference is negative, set it to zero (no edge before node creation)
edges_df['time_difference_source'] = edges_df['time_difference_source'].apply(lambda x: max(x, 0))
edges_df['time_difference_target'] = edges_df['time_difference_target'].apply(lambda x: max(x, 0))

app_name_counts = edges_df['appName_source'].value_counts().to_dict()
edges_df['appName_source_freq'] = edges_df['appName_source'].map(app_name_counts)

app_name_counts = edges_df['appName_target'].value_counts().to_dict()
edges_df['appName_target_freq'] = edges_df['appName_target'].map(app_name_counts)
edges_df.drop(['appName_source', 'appName_target'], axis=1, inplace=True)

# Step 2: Create features and labels (assuming you have a label column for link prediction)
# If you don't have labels, you can assume all links are positive (1) or generate negative samples.
features = ['time_difference_source', 'time_difference_target', 'node_type_source', 'node_type_target', 
            'appName_source', 'appName_target', 'version_source', 'version_target', 
            'isClientSecretManaged_source', 'isClientSecretManaged_target']

X = edges_df[features]

# For link prediction, if you have a label column (e.g., 1 for existing links, 0 for non-existent links):
y = edges_df['label']  # Replace with actual label column if you have it, else generate negative samples.

# Step 3: Train-test split based on temporal data (using time-based split)
train_data = edges_df[edges_df['timeStamp'] < 1672531199]  # E.g., before 2023-01-01
test_data = edges_df[edges_df['timeStamp'] >= 1672531199]  # After 2023-01-01

X_train = train_data[features]
y_train = train_data['label']
X_test = test_data[features]
y_test = test_data['label']

# Step 4: Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("ROC AUC Score:", roc_auc)

# Optionally: Hyperparameter tuning or use a more sophisticated model (e.g., Temporal Graph Neural Networks for advanced temporal models)
