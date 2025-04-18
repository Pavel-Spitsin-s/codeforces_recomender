import numpy as np
from fetcher import fetch_codeforces_problems


def preprocess():
    problems_df = fetch_codeforces_problems()

    all_tags = sorted({tag for tags in problems_df['tags'] for tag in tags})

    for tag in all_tags:
        problems_df[f"tag_{tag}"] = problems_df['tags'].apply(lambda lst: tag in lst).astype(int)

    min_solved, max_solved = problems_df['solvedCount'].min(), problems_df['solvedCount'].max()
    problems_df['pop_norm'] = (
            (problems_df['solvedCount'] - min_solved) /
            (max_solved - min_solved)
    )
    return problems_df
