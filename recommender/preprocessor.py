import numpy as np
from .api import CodeforcesAPI


class Preprocessor:
    def __init__(self):
        self.api = CodeforcesAPI()

    def build_problem_df(self) -> np.ndarray:
        df = self.api.fetch_problems()
        all_tags = sorted({t for tags in df.tags for t in tags})
        for tag in all_tags:
            df[f"tag_{tag}"] = df.tags.map(lambda L: tag in L).astype(int)
        mn, mx = df.solvedCount.min(), df.solvedCount.max()
        df['pop_norm'] = (df.solvedCount - mn) / (mx - mn)
        return df
