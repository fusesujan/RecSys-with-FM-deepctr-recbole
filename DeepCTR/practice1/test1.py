import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, get_feature_names

# Create dummy data
data = {
    'user_id': np.random.randint(1, 100, 1000),
    'item_id': np.random.randint(1, 50, 1000),
    'rating': np.random.randint(1, 6, 1000)  # Simulating user ratings (1 to 5)
}

df = pd.DataFrame(data)
print(df)
user_col = 'user_id'
item_col = 'item_id'
rating_col = 'rating'

# Label encode user and item IDs
df[user_col] = LabelEncoder().fit_transform(df[user_col])
df[item_col] = LabelEncoder().fit_transform(df[item_col])

# Define feature columns
user_feature = SparseFeat(
    user_col, vocabulary_size=df[user_col].nunique(), embedding_dim=16)
item_feature = SparseFeat(
    item_col, vocabulary_size=df[item_col].nunique(), embedding_dim=16)

feature_columns = [user_feature, item_feature]
feature_names = get_feature_names(feature_columns)

# Split data into train and test sets
train, test = train_test_split(df, test_size=0.2, random_state=2020)
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

# Define the recommendation model (DeepFM)
# 'regression' for ratings prediction
model = DeepFM(feature_columns, feature_columns, task='regression')
model.compile("adam", "mse", metrics=['mse'])

# Train the model
history = model.fit(train_model_input, train[rating_col].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2)

# Make predictions on the test set
pred_ratings = model.predict(test_model_input, batch_size=256)

# Evaluate the recommendation system using RMSE for ratings prediction
rmse = np.sqrt(mean_squared_error(test[rating_col], pred_ratings))
print("Root Mean Squared Error (RMSE):", rmse)
