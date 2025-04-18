# Codeforces Recommender

A simple Flask web application that recommends Codeforces problems to users based on their submission history. It uses a preprocessing pipeline, a recommendation engine, and the Codeforces API to fetch user submissions and generate personalized recommendations.

## Features

- Fetches user submission history from Codeforces with caching to reduce redundant API calls.
- Builds a problem dataset and computes recommendations with a custom engine.
- Supports pagination that accumulates all previous recommendations and adds the next batch on each click.
- Minimal dependencies and straightforward Flask setup.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/codeforces-recommender.git
   cd codeforces-recommender
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
.
├── app.py                  # Main Flask application
├── recommender             # Recommendation logic.
│   ├── preprocessor.py     # Builds the problem DataFrame
│   ├── api.py              # Wrapper for Codeforces API
│   └── engine.py           # Recommendation engine
├── templates               # HTML templates
│   ├── base.html           # Base layout
│   ├── home.html           # Homepage with form
│   └── results.html        # Recommendation table and pagination
└── README.md               # Project documentation
```

## Configuration

- No additional configuration is required. The application uses the public Codeforces API.

## Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://localhost:5000`.

3. Enter your Codeforces handle on the homepage and click **Recommend**.

4. View your top 10 recommendations. Click **Recommend More** to load the next batch, which accumulates all previous recommendations and adds the next page of results.

## Pagination Logic

The pagination is implemented without altering `results.html`. The `recommend` view:

- Reads `offset` (default `0`) and fixed `page_size` (default `10`).
- Fetches `offset + page_size + 1` recommendations to detect if more pages exist.
- Sets `more = len(all_recs) > offset + page_size` to control the **Recommend More** button.
- Slices recommendations as `all_recs.iloc[0 : offset + page_size]`, accumulating all previous plus the next page on each request.

## Recommendation Engine Details

The recommendation logic is encapsulated in `RecommenderEngine` (see `recommender/engine.py`). It works as follows:

1. **User Profile Construction** (`_user_profile`):
   - Given a set of solved problem IDs `S`, extract their tag indicator vectors from `problem_df`.
   - Sum these vectors to form a raw profile:
     ```
     p_j = sum_{i in S} x_{ij}
     ```
     where `x_{ij}` is 1 if problem *i* has tag *j*, else 0.
   - Normalize the profile vector:
     ```
     u = p / ||p||
     ```
     ensuring unit length (or zero if no solved problems).

2. **Candidate Selection**:
   - Filter out problems the user has already solved, yielding a candidate set `C`.

3. **Tag-based Similarity**:
   - For each candidate problem *k*, retrieve its tag vector `x_k`.
   - Compute cosine similarity:
     ```
     sim_k = (x_k^T • u) / ||x_k||
     ```
     with a guard to treat zero-norm vectors appropriately.

4. **Difficulty-based Score**:
   - Fetch the user rating `r_u` via the Codeforces API.
   - Define a Gaussian kernel centered at `r_u` with standard deviation `sigma = 0.3 * r_u`:
     ```
     s_k = exp(-((r_k - r_u)^2) / (2 * sigma^2))
     ```
     where `r_k` is the candidate’s rating (or `r_u` if missing).

5. **Popularity Normalization**:
   - Use a precomputed `pop_norm` feature representing normalized solved counts for each problem.

6. **Combined Score**:
   - Aggregate the three components into a final score:
     ```
     score_k = alpha * sim_k + beta * pop_norm_k + gamma * s_k
     ```
     with default weights `alpha = 0.6`, `beta = 0.2`, `gamma = 0.2`.

7. **Recommendation Output**:
   - Select the top `n` candidates by `score_k` using a descending sort (`nlargest`).

Example usage in code:
```python
engine = RecommenderEngine(problem_df)
recs = engine.recommend(handle, interactions, top_n=20)
```

## Caching

User submissions are cached in memory (`user_cache`) for the duration of the application to avoid repeated API calls:

```python
if handle not in user_cache:
    user_cache[handle] = api.fetch_user_submissions(handle)
interactions = user_cache[handle]
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

