"""
Data loading and preprocessing for the hybrid recommendation engine.
Loads users, items, and ratings; builds sparse user-item matrix.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def load_data(
    users_path="data/users.csv",
    items_path="data/items.csv",
    ratings_path="data/ratings.csv",
):
    """
    Load all three CSVs and return DataFrames.

    Returns:
        users, items, ratings DataFrames
    """
    users = pd.read_csv(users_path)
    items = pd.read_csv(items_path)
    ratings = pd.read_csv(ratings_path)

    print(f"Users:   {len(users)} rows, columns: {list(users.columns)}")
    print(f"Items:   {len(items)} rows, columns: {list(items.columns)}")
    print(f"Ratings: {len(ratings)} rows, columns: {list(ratings.columns)}")
    print(f"Rating range: {ratings['rating'].min()} - {ratings['rating'].max()}")
    print(f"Unique users with ratings: {ratings['user_id'].nunique()}")
    print(f"Unique items with ratings: {ratings['item_id'].nunique()}")

    return users, items, ratings


def build_user_item_matrix(ratings, n_users=None, n_items=None):
    """
    Build a sparse user-item rating matrix from the ratings DataFrame.

    Args:
        ratings: DataFrame with user_id, item_id, rating columns
        n_users: total number of users (auto-detected if None)
        n_items: total number of items (auto-detected if None)

    Returns:
        sparse_matrix: scipy csr_matrix of shape (n_users, n_items)
        user_id_to_idx: dict mapping user_id to matrix row index
        item_id_to_idx: dict mapping item_id to matrix column index
        idx_to_user_id: dict mapping matrix row index to user_id
        idx_to_item_id: dict mapping matrix column index to item_id
    """
    unique_users = sorted(ratings["user_id"].unique())
    unique_items = sorted(ratings["item_id"].unique())

    user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    item_id_to_idx = {iid: i for i, iid in enumerate(unique_items)}
    idx_to_user_id = {i: uid for uid, i in user_id_to_idx.items()}
    idx_to_item_id = {i: iid for iid, i in item_id_to_idx.items()}

    rows = ratings["user_id"].map(user_id_to_idx).values
    cols = ratings["item_id"].map(item_id_to_idx).values
    vals = ratings["rating"].values.astype(np.float32)

    n_u = n_users if n_users else len(unique_users)
    n_i = n_items if n_items else len(unique_items)

    sparse_matrix = csr_matrix((vals, (rows, cols)), shape=(n_u, n_i))

    sparsity = 1.0 - sparse_matrix.nnz / (n_u * n_i)
    print(f"\nUser-item matrix: {n_u} users x {n_i} items")
    print(f"Non-zero entries: {sparse_matrix.nnz}")
    print(f"Sparsity: {sparsity:.2%}")

    return sparse_matrix, user_id_to_idx, item_id_to_idx, idx_to_user_id, idx_to_item_id


def train_test_split_ratings(ratings, test_size=0.2, random_state=42):
    """
    Split ratings into train and test sets, ensuring each user has at least
    one rating in the training set.

    Returns:
        train_ratings, test_ratings DataFrames
    """
    rng = np.random.RandomState(random_state)

    # For each user, hold out test_size fraction of their ratings
    train_list = []
    test_list = []

    for uid, group in ratings.groupby("user_id"):
        n = len(group)
        if n <= 1:
            train_list.append(group)
            continue
        n_test = max(1, int(n * test_size))
        test_idx = rng.choice(group.index, size=n_test, replace=False)
        test_list.append(group.loc[test_idx])
        train_list.append(group.drop(test_idx))

    train_ratings = pd.concat(train_list).reset_index(drop=True)
    test_ratings = pd.concat(test_list).reset_index(drop=True)

    print(f"\nTrain/test split:")
    print(f"  Train: {len(train_ratings)} ratings")
    print(f"  Test:  {len(test_ratings)} ratings")
    print(f"  Users in train: {train_ratings['user_id'].nunique()}")
    print(f"  Users in test:  {test_ratings['user_id'].nunique()}")

    return train_ratings, test_ratings


if __name__ == "__main__":
    users, items, ratings = load_data()
    matrix, u2i, i2i, i2u, i2item = build_user_item_matrix(ratings)
    train, test = train_test_split_ratings(ratings)
