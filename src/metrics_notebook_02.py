import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity


def compute_lift(ratings_df, target_users, entity_col, min_target_readers=20, min_global_readers=30):
    if isinstance(target_users, str):
        target_users = ratings_df[ratings_df[entity_col].str.lower() == target_users.lower()]['User-ID'].unique()

    total_target_users = len(set(target_users))
    total_users = ratings_df['User-ID'].nunique()

    target_df = ratings_df[ratings_df['User-ID'].isin(target_users)]
    target_counts = target_df.groupby(entity_col)['User-ID'].nunique().reset_index(name='Target_Readers')
    global_counts = ratings_df.groupby(entity_col)['User-ID'].nunique().reset_index(name='Global_Readers')

    merged = target_counts.merge(global_counts, on=entity_col, how='inner')
    merged['Lift'] = (merged['Target_Readers'] / total_target_users) / (merged['Global_Readers'] / total_users)

    filtered = merged[
        (merged['Target_Readers'] >= min_target_readers) &
        (merged['Global_Readers'] >= min_global_readers)
    ]

    return filtered.sort_values(by='Lift', ascending=False).reset_index(drop=True)

def compute_reader_ratio(ratings_df, target_users, entity_col, min_target_readers=20, min_global_readers=30):
    if isinstance(target_users, str):
        target_users = ratings_df[ratings_df[entity_col].str.lower() == target_users.lower()]['User-ID'].unique()

    target_df = ratings_df[ratings_df['User-ID'].isin(target_users)]
    target_counts = target_df.groupby(entity_col)['User-ID'].nunique().reset_index(name='Target_Readers')
    global_counts = ratings_df.groupby(entity_col)['User-ID'].nunique().reset_index(name='Global_Readers')

    merged = target_counts.merge(global_counts, on=entity_col, how='inner')
    merged['Reader_Ratio'] = merged['Target_Readers'] / merged['Global_Readers']

    filtered = merged[
        (merged['Target_Readers'] >= min_target_readers) &
        (merged['Global_Readers'] >= min_global_readers)
    ]

    return filtered.sort_values(by='Reader_Ratio', ascending=False).reset_index(drop=True)

def compute_lift_modif(ratings_df, target_users, entity_col, min_target_readers=20, min_global_readers=30):
    if isinstance(target_users, str):
        target_users = ratings_df[ratings_df[entity_col].str.lower() == target_users.lower()]['User-ID'].unique()

    target_users = set(target_users)
    non_target_users = set(ratings_df['User-ID'].unique()) - target_users

    total_target_users = len(target_users)
    total_non_target_users = len(non_target_users)

    target_df = ratings_df[ratings_df['User-ID'].isin(target_users)]
    non_target_df = ratings_df[ratings_df['User-ID'].isin(non_target_users)]

    target_counts = target_df.groupby(entity_col)['User-ID'].nunique().reset_index(name='Target_Readers')
    non_target_counts = non_target_df.groupby(entity_col)['User-ID'].nunique().reset_index(name='Non_Target_Readers')

    merged = target_counts.merge(non_target_counts, on=entity_col, how='inner')
    merged['Lift'] = (merged['Target_Readers'] / total_target_users) / (merged['Non_Target_Readers'] / total_non_target_users)

    filtered = merged[
        (merged['Target_Readers'] >= min_target_readers) &
        (merged['Non_Target_Readers'] >= min_global_readers)
    ]

    return filtered.sort_values(by='Lift', ascending=False).reset_index(drop=True)