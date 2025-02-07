import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
edges_df = pd.read_csv('edges.csv')  # sourceId, targetId, edgeName, timeStamp
featureList_df = pd.read_csv('featureList.csv')  # node features

# Merge edge data with node features for both source and target nodes
edges_df = edges_df.merge(featureList_df, left_on='sourceId', right_on='nodeID', suffixes=('_source', '_drop'))
edges_df = edges_df.merge(featureList_df, left_on='targetId', right_on='nodeID', suffixes=('_source', '_target'))

# Drop redundant columns
edges_df.drop(columns=['nodeID_source', 'nodeID_target'], inplace=True)

### --- Feature Engineering --- ###

# Time Difference Features
edges_df['time_difference_source'] = edges_df['timeStamp'] - edges_df['createdTimestamp_source']
edges_df['time_difference_target'] = edges_df['timeStamp'] - edges_df['createdTimestamp_target']
edges_df['time_difference_source'] = edges_df['time_difference_source'].clip(lower=0)  # No negative values
edges_df['time_difference_target'] = edges_df['time_difference_target'].clip(lower=0)

# Frequency Encoding for High-Cardinality Features
for col in ['appName', 'categoryPrefix', 'resourceName']:
    source_freq = edges_df[f'{col}_source'].value_counts().to_dict()
    target_freq = edges_df[f'{col}_target'].value_counts().to_dict()
    
    edges_df[f'{col}_source_freq'] = edges_df[f'{col}_source'].map(source_freq)
    edges_df[f'{col}_target_freq'] = edges_df[f'{col}_target'].map(target_freq)

# Label Encoding for appProtocol (low-cardinality categorical)
protocol_mapping = {'REST': 0, 'SOAP': 1, 'GRPC': 2, 'GRAPHQL': 3, 'WEBSOCKET': 4}
edges_df['appProtocol_source'] = edges_df['appProtocol_source'].map(protocol_mapping).fillna(-1)
edges_df['appProtocol_target'] = edges_df['appProtocol_target'].map(protocol_mapping).fillna(-1)

# Direct Numeric Encoding for version
edges_df['version_source'] = edges_df['version_source'].replace('*', 11).astype(int)
edges_df['version_target'] = edges_df['version_target'].replace('*', 11).astype(int)

# Binary Encoding for Boolean Features
for col in ['isClientSecretManaged', 'isApproved', 'isActive']:
    edges_df[f'{col}_source'] = edges_df[f'{col}_source'].astype(float).fillna(-1)
    edges_df[f'{col}_target'] = edges_df[f'{col}_target'].astype(float).fillna(-1)

# One-Hot Encoding for appStatus (D, A, NaN)
edges_df['appStatus_source_D'] = (edges_df['appStatus_source'] == 'D').astype(int)
edges_df['appStatus_source_A'] = (edges_df['appStatus_source'] == 'A').astype(int)
edges_df['appStatus_target_D'] = (edges_df['appStatus_target'] == 'D').astype(int)
edges_df['appStatus_target_A'] = (edges_df['appStatus_target'] == 'A').astype(int)

# Handling readTimeout (Numeric with NaN)
edges_df['readTimeout'].fillna(0, inplace=True)
edges_df['readTimeout_isNaN'] = edges_df['readTimeout'].isna().astype(int)

### --- Prepare Training Data --- ###
# Select Features
features = [
    'time_difference_source', 'time_difference_target',
    'appName_source_freq', 'appName_target_freq',
    'categoryPrefix_source_freq', 'categoryPrefix_target_freq',
    'resourceName_source_freq', 'resourceName_target_freq',
    'appProtocol_source', 'appProtocol_target',
    'version_source', 'version_target',
    'isClientSecretManaged_source', 'isClientSecretManaged_target',
    'isApproved_source', 'isApproved_target',
    'isActive_source', 'isActive_target',
    'appStatus_source_D', 'appStatus_source_A',
    'appStatus_target_D', 'appStatus_target_A',
    'readTimeout', 'readTimeout_isNaN'
]

X = edges_df[features]

# Label column (1 for existing links, 0 for non-existent links)
y = edges_df['label']  # Replace with actual label column if available

# Temporal Train-Test Split (based on timeStamp)
split_time = edges_df['timeStamp'].quantile(0.8)  # 80% training, 20% testing
train_data = edges_df[edges_df['timeStamp'] < split_time]
test_data = edges_df[edges_df['timeStamp'] >= split_time]

X_train, y_train = train_data[features], train_data['label']
X_test, y_test = test_data[features], test_data['label']

### --- Train Random Forest Model --- ###
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

### --- Evaluate Model --- ###
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("ROC AUC Score:", roc_auc)
