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

# One-Hot Encoding for all categorical features

# Categorical columns to encode
categorical_columns = [
    'appName', 'categoryPrefix', 'resourceName', 'appProtocol', 'isActive',
    'isClientSecretManaged', 'appStatus', 'version', 'nodeType'
]

# Apply One-Hot Encoding for categorical features and handle NaN as a category (dummy_na=True)
edges_df = pd.get_dummies(edges_df, columns=categorical_columns, dummy_na=True)

# For binary features like isClientSecretManaged (True/False), we can also one-hot encode it
# No further encoding needed since they will be handled by `pd.get_dummies`

# Handling readTimeout - filling NaN with a default value, adding a binary column for NaN status
edges_df['readTimeout'].fillna(0, inplace=True)
edges_df['readTimeout_isNaN'] = edges_df['readTimeout'].isna().astype(int)

### --- Prepare Training Data --- ###
# Select Features
features = [col for col in edges_df.columns if col not in ['label', 'sourceId', 'targetId', 'timeStamp']]

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
model = RandomForestClassifier(n_estimators=100, random
