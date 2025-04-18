import numpy as np
from .api import CodeforcesAPI
import pandas as pd


class RecommenderEngine:
    def __init__(self, problem_df):
        self.df = problem_df.reset_index(drop=True)
        self.tag_cols = [c for c in self.df if c.startswith('tag_')]
        self.api = CodeforcesAPI()

    def _user_profile(self, solved_ids: set) -> np.ndarray:
        vec = self.df[self.df.problem_id.isin(solved_ids)][self.tag_cols]
        profile = vec.sum(axis=0).values.astype(float)
        norm = np.linalg.norm(profile)
        return profile / norm if norm else profile

    def recommend(self, handle: str, interactions: pd.DataFrame,
                  top_n: int = 10, a=0.6, b=0.2, c=0.2) -> pd.DataFrame:
        solved_ids = set(interactions.problem_id)
        candidates = self.df[~self.df.problem_id.isin(solved_ids)].copy()
        user_vec = self._user_profile(solved_ids)

        mat = candidates[self.tag_cols].values
        norms = np.linalg.norm(mat, axis=1)
        sims = (mat @ user_vec) / np.where(norms == 0, 1, norms) if user_vec.any() else np.zeros(len(candidates))

        ur = self.api.fetch_user_rating(handle)
        sigma = 0.3 * ur
        rating_score = np.exp(-((candidates.rating.fillna(ur) - ur) ** 2) / (2 * sigma ** 2))

        candidates['score'] = a * sims + b * candidates.pop_norm + c * rating_score
        top = candidates.nlargest(top_n, 'score')
        return top[['contestId', 'index', 'problem_id', 'name', 'rating', 'solvedCount', 'tags']]
