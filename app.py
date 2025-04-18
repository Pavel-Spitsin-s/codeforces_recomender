from flask import Flask, request, render_template
import preprocess_data
from fetcher import fetch_codeforces_problems, fetch_user_submissions
import numpy as np
import requests


def recommend_problems(user_handle, problems_df, interactions_df, top_n=10,
                       α=0.6, β=0.2, γ=0.2):
    solved_ids = set(interactions_df['problem_id'])
    cand = problems_df[~problems_df['problem_id'].isin(solved_ids)].copy()
    tag_cols = [c for c in problems_df.columns if c.startswith('tag_')]
    user_vec = problems_df[
        problems_df['problem_id'].isin(solved_ids)
    ][tag_cols].sum(axis=0).values.astype(float)

    if np.linalg.norm(user_vec) == 0:
        sims = np.zeros(len(cand))
    else:
        user_vec /= np.linalg.norm(user_vec)
        mat = cand[tag_cols].values
        norms = np.linalg.norm(mat, axis=1)
        sims = (mat @ user_vec) / np.where(norms == 0, 1, norms)

    info = requests.get(
        f'https://codeforces.com/api/user.info?handles={user_handle}'
    ).json()['result'][0]
    user_rating = info.get('rating', 1500)
    σ = 0.3 * user_rating
    cand['rating_score'] = np.exp(
        -((cand['rating'].fillna(user_rating) - user_rating) ** 2) / (2 * σ ** 2)
    )

    cand['score'] = α * sims + β * cand['pop_norm'] + γ * cand['rating_score']
    top = cand.sort_values('score', ascending=False).head(top_n)
    return top[['contestId', 'index', 'problem_id', 'name', 'rating', 'solvedCount', 'tags']]


app = Flask(__name__)
problems_df = preprocess_data.preprocess()
user_cache = {}


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/recommend')
def recommend():
    handle = request.args.get('handle')
    if not handle:
        return "Handle is required", 400
    offset = int(request.args.get('offset', 0))
    page_size = 10

    if handle not in user_cache:
        user_cache[handle] = fetch_user_submissions(handle)
    recs = recommend_problems(handle, problems_df, user_cache[handle], top_n=offset + page_size)
    more = len(recs) >= offset + page_size

    return render_template(
        'results.html',
        handle=handle,
        recs=recs,
        offset=offset,
        page_size=page_size,
        more=more
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
