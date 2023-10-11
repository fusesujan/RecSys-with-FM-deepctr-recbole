import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, get_feature_names

if __name__ == "__main__":

    data = pd.read_csv("./movielens_sample.txt")
    print(data.info())
    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip"]
    target = ['rating']
    label_encoders = {}
    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        label_encoders[feat] = lbe  # Store the label encoder for later use
        # print(data[feat])
    # 2.count #unique features for each sparse field
    fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=4)
                              for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    print(train.columns, "-----------------------------")
    train_model_input = {name: train[name].values for name in feature_names}
    test_model_input = {name: test[name].values for name in feature_names}
    print("Here is the test_model_input:", test_model_input)

    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns,
                   dnn_feature_columns, task='regression')
    model.compile("adam", "mse", metrics=['mse'], )
    print(train[target].values)
    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test MSE", round(mean_squared_error(
        test[target].values, pred_ans), 4))

# 380,3481,5,976316283,High Fidelity (2000),Comedy,M,25,2,92024
# 4694,159,3,963602574,Clockers (1995),Drama,M,56,7,40505

    user_data = {
        'movie_id': 3481,
        'user_id': 380,
        'gender': 'M',
        # 'rating': 3,
        # "timestamp": 963602574,
        # "genres": "Drama",
        'age': 25,
        'occupation': 2,
        'zip': '92024'
    }

    user_df = pd.DataFrame([user_data])
    print("/n/n")
    print(user_df)
    # exit()
    print('Before encoding====', user_df[feat])

    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip"]

    print("================", lbe, type(lbe))
    for feat in sparse_features:
        print("================", user_df[feat])
        # lbe = LabelEncoder()
        # user_df[feat] = lbe.transform([user_df[feat]])
        user_df[feat] = label_encoders[feat].transform(user_df[feat])

    fixlen_feature_columns = [SparseFeat(feat, user_df[feat].max() + 1, embedding_dim=4)
                              for feat in sparse_features]

    # print("fixlen_feature_columns", fixlen_feature_columns)
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    user_model_input = {name: user_df[name].values for name in feature_names}
    print("Here is the user_model_input:", user_model_input)

    movie_recommendations = model.predict(user_model_input, batch_size=256)
    print("Here are movie_recommendations:", movie_recommendations)

    top_movie_indices = movie_recommendations.argsort()[0][-10:][::-1]
    print("Here are the top movies indices:", top_movie_indices)

    recommended_movies = data.loc[data['movie_id'].isin(
        top_movie_indices)]['title']

    print("Top Recommended Movies:")
    print(recommended_movies)
