import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import recommender as rec

os.makedirs('charts', exist_ok=True)
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 120})

TEAL  = '#1a7a6e'
CORAL = '#e07060'
STEEL = '#456b8a'


def save(name):
    plt.tight_layout()
    plt.savefig(f'charts/{name}.png')
    plt.close()


print("=" * 56)
print("  MOVIE RECOMMENDATION SYSTEM")
print("=" * 56)

# --- 1. Load & filter ---
ratings, movies = rec.load_and_filter('dataset/ratings.csv', 'dataset/movies.csv')

n_users  = ratings['userId'].nunique()
n_movies = ratings['movieId'].nunique()
sparsity = 1 - len(ratings) / (n_users * n_movies)

print(f"\nAfter filtering:  {n_users} users · {n_movies} movies · {len(ratings)} ratings")
print(f"Sparsity:         {sparsity:.4f}")

# --- 2. Feature engineering ---
genres_dummies = movies['genres'].str.get_dummies(sep='|')
movies_enc     = movies.copy()
movies_enc     = movies_enc.join(genres_dummies)

ratings['datetime'] = ratings['timestamp'].apply(
    lambda t: __import__('datetime').datetime.utcfromtimestamp(t)
)

# --- 3. User-item matrix + content model ---
user_item  = rec.build_user_item_matrix(ratings)
cosine_sim, movie_idx = rec.build_content_model(movies)

print(f"User-item matrix: {user_item.shape}")

# --- 4. Visualizations ---

# Sparsity heatmap
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(user_item.iloc[:100, :100] > 0, cbar=False, cmap='viridis', ax=ax)
ax.set_title('User-Item Matrix Sparsity (first 100×100)', fontsize=13)
ax.set_xlabel('Movies (first 100)')
ax.set_ylabel('Users (first 100)')
save('sparsity_heatmap')

# User activity
uc_sorted = ratings.groupby('userId').size().sort_values(ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(range(len(uc_sorted)), uc_sorted, alpha=0.35, color=TEAL)
ax.plot(uc_sorted, color=TEAL, linewidth=1.2)
ax.set_title('User Activity Distribution (sorted)')
ax.set_xlabel('Users (ranked by activity)')
ax.set_ylabel('Number of ratings')
save('user_activity')

# Movie popularity
mc_sorted = ratings.groupby('movieId').size().sort_values(ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(range(len(mc_sorted)), mc_sorted, alpha=0.35, color=TEAL)
ax.plot(mc_sorted, color=TEAL, linewidth=1.2)
ax.set_title('Movie Popularity Distribution (sorted)')
ax.set_xlabel('Movies (ranked by popularity)')
ax.set_ylabel('Number of ratings')
save('movie_popularity')

# Rating distribution
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(ratings['rating'], bins=10, color=TEAL, kde=False, ax=ax)
ax.set_title('Rating Distribution')
ax.set_xlabel('Rating')
ax.set_ylabel('Count')
ax.grid(axis='y', alpha=0.3)
save('rating_distribution')

# Genre counts
fig, ax = plt.subplots(figsize=(11, 5))
genres_dummies.sum().sort_values(ascending=False).plot(kind='bar', color=TEAL, alpha=0.75, ax=ax)
ax.set_title('Movies per Genre')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=75)
save('genre_counts')

print("Charts saved →  charts/")

# --- 5. Train-test split ---
train, test = train_test_split(ratings, test_size=0.2, random_state=42)
train_matrix = rec.build_user_item_matrix(train)
print(f"\nTrain: {len(train)} ratings  |  Test: {len(test)} ratings")

# --- 6. Train CF models ---
model_item, model_user = rec.train_cf_models(train_matrix)

# --- 7. Hyperparameter tuning ---
K_VALUES = [5, 10, 15, 20, 25, 30, 40, 50]
item_rmse_list, user_rmse_list = rec.tune_k(
    test, train_matrix, model_item, model_user, K_VALUES, sample_n=100
)

print(f"\n--- HYPERPARAMETER TUNING: k ---")
print(f"{'k':>5} | {'Item RMSE':>10} | {'User RMSE':>10}")
print("-" * 32)
for k, ri, ru in zip(K_VALUES, item_rmse_list, user_rmse_list):
    print(f"{k:>5} | {ri:>10.4f} | {ru:>10.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(K_VALUES, item_rmse_list, marker='o', color=TEAL,  label='Item-Based')
ax.plot(K_VALUES, user_rmse_list, marker='s', color=CORAL, label='User-Based')
ax.set_title('Hyperparameter Tuning: k vs RMSE')
ax.set_xlabel('k (Number of Neighbors)')
ax.set_ylabel('RMSE (lower is better)')
ax.legend()
ax.grid(True, alpha=0.3)
save('k_tuning')

best_k_item = K_VALUES[int(np.argmin(item_rmse_list))]
best_k_user = K_VALUES[int(np.argmin(user_rmse_list))]
print(f"\nBest k — Item-Based: {best_k_item}  |  User-Based: {best_k_user}")

# --- 8. Full evaluation (MAE + RMSE) ---
mae_i, rmse_i, mae_u, rmse_u, error_df = rec.evaluate(
    test, train_matrix, model_item, model_user,
    best_k_item, best_k_user, sample_n=200
)

print(f"\n--- EVALUATION RESULTS ---")
print(f"{'Method':<16} {'MAE':>8} {'RMSE':>8}")
print("-" * 34)
print(f"{'Item-Based CF':<16} {mae_i:>8.4f} {rmse_i:>8.4f}")
print(f"{'User-Based CF':<16} {mae_u:>8.4f} {rmse_u:>8.4f}")
winner = 'Item-Based' if rmse_i < rmse_u else 'User-Based'
print(f"Winner: {winner} Collaborative Filtering")

# Error charts
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(error_df['err_item'], bins=20, color=CORAL, kde=True, ax=ax)
ax.set_title('Absolute Error Distribution (Item-Based CF)')
ax.set_xlabel('|actual − predicted|')
ax.set_ylabel('Count')
ax.grid(axis='y', alpha=0.3)
save('error_abs_dist')

fig, ax = plt.subplots(figsize=(9, 5))
error_df.groupby('actual')['err_item'].mean().plot(
    kind='bar', color=CORAL, alpha=0.8, edgecolor='black', ax=ax
)
ax.set_title('Mean Absolute Error by Actual Rating')
ax.set_xlabel('Actual Rating')
ax.set_ylabel('MAE')
ax.tick_params(axis='x', rotation=0)
ax.grid(axis='y', alpha=0.3)
save('error_by_rating')

fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(error_df['signed_item'], bins=20, color=STEEL, kde=True, ax=ax)
ax.axvline(0, color='red', linestyle='--', linewidth=1.2, label='No error')
ax.set_title('Signed Error Distribution (positive = over-predicted)')
ax.set_xlabel('Predicted − Actual')
ax.set_ylabel('Count')
ax.legend()
ax.grid(axis='y', alpha=0.3)
save('error_signed')

bias = 'over-predicts' if error_df['signed_item'].mean() > 0 else 'under-predicts'
print(f"\nMean signed error: {error_df['signed_item'].mean():.4f}  ({bias} on average)")
print("\nMAE by actual rating:")
print(error_df.groupby('actual')['err_item'].mean().round(3).to_string())
print(f"\nWorst 5 predictions:")
print(error_df.sort_values('err_item', ascending=False).head(5)[
    ['userId', 'actual', 'pred_item', 'err_item']
].to_string(index=False))

# --- 9. Top-N recommendations ---
demo_user = train_matrix.index[0]
n_rated   = int((train_matrix.loc[demo_user] > 0).sum())
print(f"\n--- RECOMMENDATIONS FOR USER {demo_user} ({n_rated} movies rated) ---")

print(f"\n>> Item-Based CF  (k={best_k_item}):")
recs_item = rec.top_n_cf(
    demo_user, train_matrix, model_item, model_user, movies,
    n=10, method='item', k=best_k_item
)
print(recs_item.to_string() if recs_item is not None else "User not found")

print(f"\n>> User-Based CF  (k={best_k_user}):")
recs_user = rec.top_n_cf(
    demo_user, train_matrix, model_item, model_user, movies,
    n=10, method='user', k=best_k_user
)
print(recs_user.to_string() if recs_user is not None else "User not found")

# Content-based demo using the 3 most-rated movies as a fake "user profile"
top3_ids   = ratings.groupby('movieId').size().nlargest(3).index.tolist()
demo_cbf   = {mid: 5.0 for mid in top3_ids}
recs_content = rec.top_n_content(demo_cbf, movies, cosine_sim, movie_idx, n=10)
demo_titles  = movies[movies['movieId'].isin(top3_ids)]['title'].tolist()
print(f"\n>> Content-Based CF  (based on: {', '.join(demo_titles)}):")
print(recs_content.to_string())

# --- 10. Final summary ---
print("\n" + "=" * 56)
print("  FINAL SUMMARY")
print("=" * 56)
print(f"  Dataset:       MovieLens 100k")
print(f"  Users:         {n_users}   Movies: {n_movies}   Sparsity: {sparsity:.3f}")
print("-" * 56)
print(f"  Item-Based CF  k={best_k_item:2d}   MAE={mae_i:.4f}   RMSE={rmse_i:.4f}")
print(f"  User-Based CF  k={best_k_user:2d}   MAE={mae_u:.4f}   RMSE={rmse_u:.4f}")
print(f"  Winner: {winner}")
print("=" * 56)
print(f"\n[READY]  rec.top_n_cf({demo_user}, ...) to generate recommendations.")
print(f"         Charts saved in charts/  — push to GitHub to render in README.")
