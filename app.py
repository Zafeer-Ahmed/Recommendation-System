from flask import Flask, render_template, request
import recommender as rec

app = Flask(__name__)

# Build content model on startup — fast, no training needed
ratings, movies = rec.load_and_filter('dataset/ratings.csv', 'dataset/movies.csv')
cosine_sim, movie_idx = rec.build_content_model(movies)

# Top 30 most-rated movies to show in the UI
popular_ids     = ratings.groupby('movieId').size().sort_values(ascending=False).head(30).index
display_movies  = (
    movies[movies['movieId'].isin(popular_ids)][['movieId', 'title', 'genres']]
    .sort_values('title')
    .to_dict('records')
)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', movies=display_movies)


@app.route('/recommend', methods=['POST'])
def recommend():
    user_ratings = {}
    for movie in display_movies:
        val = request.form.get(f"rating_{movie['movieId']}", '').strip()
        if val:
            user_ratings[int(movie['movieId'])] = float(val)

    if not user_ratings:
        return render_template('index.html', movies=display_movies, no_ratings=True)

    recs_df = rec.top_n_content(user_ratings, movies, cosine_sim, movie_idx, n=10)
    recs    = recs_df[['title', 'genres']].to_dict('records')
    return render_template(
        'index.html',
        movies=display_movies,
        recommendations=recs,
        user_ratings=user_ratings,
    )


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)