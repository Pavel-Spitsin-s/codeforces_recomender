import requests
import pandas as pd


class CodeforcesAPI:
    BASE_URL = 'https://codeforces.com/api'

    def fetch_problems(self) -> pd.DataFrame:
        resp = requests.get(f"{self.BASE_URL}/problemset.problems")
        resp.raise_for_status()
        data = resp.json()['result']
        probs = pd.DataFrame(data['problems'])
        stats = pd.DataFrame(data['problemStatistics'])
        df = probs.merge(stats[['contestId', 'index', 'solvedCount']],
                         on=['contestId', 'index'], how='left')
        df['problem_id'] = df['contestId'].astype(str) + df['index']
        return df

    def fetch_user_submissions(self, handle: str, max_count: int = 10000) -> pd.DataFrame:
        resp = requests.get(
            f"{self.BASE_URL}/user.status?handle={handle}&from=1&count={max_count}"
        )
        resp.raise_for_status()
        subs = pd.DataFrame(resp.json()['result'])
        subs = subs[subs.verdict == 'OK']
        subs['problem_id'] = subs.problem.apply(lambda p: f"{p['contestId']}{p['index']}")
        return subs[['problem_id']].drop_duplicates()

    def fetch_user_rating(self, handle: str) -> int:
        resp = requests.get(f"{self.BASE_URL}/user.info?handles={handle}")
        resp.raise_for_status()
        info = resp.json()['result'][0]
        return info.get('rating', 1500)
