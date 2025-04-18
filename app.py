from flask import Flask, request, render_template
from recommender.preprocessor import Preprocessor
from recommender.api import CodeforcesAPI
from recommender.engine import RecommenderEngine

app = Flask(__name__)
processor = Preprocessor()
problem_df = processor.build_problem_df()
engine = RecommenderEngine(problem_df)
api = CodeforcesAPI()
user_cache = {}


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/recommend')
def recommend():
    handle = request.args.get('handle', '').strip()
    if not handle:
        return "Handle is required", 400

    offset = int(request.args.get('offset', 0))
    page_size = 10

    if handle not in user_cache:
        user_cache[handle] = api.fetch_user_submissions(handle)
    interactions = user_cache[handle]

    top_n = offset + page_size + 1
    all_recs = engine.recommend(handle, interactions, top_n=top_n)

    more = len(all_recs) > offset + page_size

    recs = all_recs.iloc[0: offset + page_size]

    return render_template('results.html',
                           handle=handle,
                           recs=recs,
                           offset=offset,
                           page_size=page_size,
                           more=more)


if __name__ == '__main__':
    app.run(debug=True)
