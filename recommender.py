import pandas as pd
import numpy as np
from math import sqrt
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


def load_and_filter(ratings_path, movies_path, min_movie=10, min_user=5):
    ratings = pd.read_csv(ratings_path)
    movies  = pd.read_csv(movies_path)
    mc = ratings.groupby('movieId').size()
    ratings = ratings[ratings['movieId'].isin(mc[mc > min_movie].index)]
    uc = ratings.groupby('userId').size()
    ratings = ratings[ratings['userId'].isin(uc[uc > min_user].index)].copy()
    return ratings, movies


def build_user_item_matrix(ratings):
    return ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)


def build_content_model(movies):
    # TF-IDF on pipe-separated genres, returns similarity matrix and movie id → row index map
    text    = movies['genres'].str.replace('|', ' ', regex=False).fillna('unknown')
    sim     = cosine_similarity(TfidfVectorizer().fit_transform(text))
    idx_map = pd.Series(range(len(movies)), index=movies['movieId'].values)
    return sim, idx_map


def train_cf_models(train_matrix):
    model_item = NearestNeighbors(metric='cosine', algorithm='brute').fit(train_matrix.T)
    model_user = NearestNeighbors(metric='cosine', algorithm='brute').fit(train_matrix)
    return model_item, model_user


def _predict_cf(mode, user_id, movie_id, train_matrix, model, k):
    if user_id not in train_matrix.index or movie_id not in train_matrix.columns:
        return 3.0
    mat = train_matrix.T if mode == 'item' else train_matrix
    tid = movie_id     if mode == 'item' else user_id
    n   = min(k + 1, len(mat))
    vec = mat.loc[tid].values.reshape(1, -1)
    dists, inds = model.kneighbors(vec, n_neighbors=n)
    sims = 1 - dists.flatten()[1:]
    nbrs = mat.index[inds.flatten()[1:]]
    rv   = train_matrix.loc[user_id, nbrs] if mode == 'item' else train_matrix.loc[nbrs, movie_id]
    return float(np.dot(sims, rv) / (np.abs(sims).sum() + 1e-9))


def predict_item(user_id, movie_id, train_matrix, model_item, k=10):
    return _predict_cf('item', user_id, movie_id, train_matrix, model_item, k)


def predict_user(user_id, movie_id, train_matrix, model_user, k=10):
    return _predict_cf('user', user_id, movie_id, train_matrix, model_user, k)


def top_n_cf(user_id, train_matrix, model_item, model_user, movies, n=10, method='item', k=10):
    if user_id not in train_matrix.index:
        return None
    rated   = set(train_matrix.loc[user_id][train_matrix.loc[user_id] > 0].index)
    fn      = predict_item if method == 'item' else predict_user
    model   = model_item   if method == 'item' else model_user
    preds   = [(mid, fn(user_id, mid, train_matrix, model, k))
               for mid in set(train_matrix.columns) - rated]
    preds.sort(key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(preds[:n], columns=['movieId', 'predicted_rating'])
    df = df.merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')
    df.index = range(1, len(df) + 1)
    return df[['title', 'genres', 'predicted_rating']]


def top_n_content(user_ratings, movies, sim, idx_map, n=10):
    scores = {}
    for mid, rating in user_ratings.items():
        if mid not in idx_map.index:
            continue
        for i, s in enumerate(sim[idx_map[mid]]):
            cand = int(movies.iloc[i]['movieId'])
            if cand not in user_ratings:
                scores[cand] = scores.get(cand, 0) + s * rating
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    df  = pd.DataFrame(top, columns=['movieId', 'score'])
    df  = df.merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')
    df.index = range(1, len(df) + 1)
    return df[['title', 'genres', 'score']]


def tune_k(test_data, train_matrix, model_item, model_user, k_values, sample_n=100):
    sample = test_data.sample(min(sample_n, len(test_data)), random_state=42)
    valid  = sample[
        sample['userId'].isin(train_matrix.index) &
        sample['movieId'].isin(train_matrix.columns)
    ]
    item_rmse, user_rmse = [], []
    for k in k_values:
        ip = [predict_item(r.userId, r.movieId, train_matrix, model_item, k) for r in valid.itertuples()]
        up = [predict_user(r.userId, r.movieId, train_matrix, model_user, k) for r in valid.itertuples()]
        a  = valid['rating'].tolist()
        item_rmse.append(sqrt(mean_squared_error(a, ip)))
        user_rmse.append(sqrt(mean_squared_error(a, up)))
    return item_rmse, user_rmse


def evaluate(test_data, train_matrix, model_item, model_user, k_item, k_user, sample_n=200):
    sample = test_data.sample(min(sample_n, len(test_data)), random_state=42)
    rows   = []
    for r in sample.itertuples():
        if r.userId in train_matrix.index and r.movieId in train_matrix.columns:
            pi = predict_item(r.userId, r.movieId, train_matrix, model_item, k_item)
            pu = predict_user(r.userId, r.movieId, train_matrix, model_user, k_user)
            rows.append({
                'userId': r.userId, 'movieId': r.movieId, 'actual': r.rating,
                'pred_item':   round(pi, 3),
                'pred_user':   round(pu, 3),
                'err_item':    round(abs(r.rating - pi), 3),
                'err_user':    round(abs(r.rating - pu), 3),
                'signed_item': round(pi - r.rating, 3),
            })
    df = pd.DataFrame(rows)
    return (
        df['err_item'].mean(),
        sqrt(mean_squared_error(df['actual'], df['pred_item'])),
        df['err_user'].mean(),
        sqrt(mean_squared_error(df['actual'], df['pred_user'])),
        df,
    )
