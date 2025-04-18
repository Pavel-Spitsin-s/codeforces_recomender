import pandas as pd
import requests


def fetch_codeforces_problems():
    url = 'https://codeforces.com/api/problemset.problems'
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()['result']
    problems = data['problems']
    stats = data['problemStatistics']
    stats_df = pd.DataFrame(stats)[['contestId', 'index', 'solvedCount']]
    meta_df = pd.DataFrame(problems)
    df = meta_df.merge(stats_df, how='left', on=['contestId', 'index'])
    df['problem_id'] = df['contestId'].astype(str) + df['index']
    return df


def fetch_user_submissions(handle, max_count=10000):
    url = f'https://codeforces.com/api/user.status?handle={handle}&from=1&count={max_count}'
    resp = requests.get(url)
    resp.raise_for_status()
    subs = resp.json()['result']
    df = pd.DataFrame(subs)
    df = df[df['verdict'] == 'OK']
    df['problem_id'] = df['problem'].apply(lambda p: str(p['contestId']) + p['index'])
    df['user_handle'] = df['author'].apply(
        lambda a: a['members'][0]['handle'] if isinstance(a, dict) and a.get('members') else None
    )
    return df[['user_handle', 'problem_id']].drop_duplicates()
