import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np

# Load dataset
df = pd.read_csv('your_file.csv')

# Ensure Category column is string type
df['Category'] = df['Category'].astype(str)


def split_category(category):
    # Split by hyphens
    words = category.replace("-", " ").split()
    # Split CamelCase words
    split_words = []
    for word in words:
        split_words += re.sub(r'([a-z])([A-Z])', r'\1 \2', word).split()
    return split_words

# Train Word2Vec model on single-word categories (or split multi-word categories)
sentences = [[word] for word in df['Category'].unique()]  # Each category as a "sentence"
w2v_model = Word2Vec(sentences, vector_size=10, window=5, min_count=1, workers=4)



# Generate category vectors
category_vectors = np.array([w2v_model.wv[word] for word in df['Category'].unique() if word in w2v_model.wv])

# Perform KMeans clustering
num_clusters = 50  # Adjust based on dataset size
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df_clusters = pd.DataFrame(df['Category'].unique(), columns=['Category'])
df_clusters['Cluster'] = kmeans.fit_predict(category_vectors)

# Create a mapping from category to cluster
category_to_cluster = dict(zip(df_clusters['Category'], df_clusters['Cluster']))

# Apply clustering to the original dataset
df['Category_Clustered'] = df['Category'].map(category_to_cluster)

# Save to CSV
df.to_csv('clustered_categories.csv', index=False)

print("Clustering complete. Categories have been grouped into", num_clusters, "clusters.")
