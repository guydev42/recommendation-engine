"""
Hybrid recommendation engine combining collaborative filtering, content-based
filtering, and matrix factorization (SVD). Includes evaluation metrics and
cold-start handling.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"


def _ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


# =====================================================================
# COLLABORATIVE FILTERING
# =====================================================================

def user_based_cf(train_matrix, user_idx, n_neighbors=20, top_n=10):
    """
    User-based collaborative filtering using cosine similarity.

    Args:
        train_matrix: sparse user-item matrix
        user_idx: target user row index
        n_neighbors: number of similar users to consider
        top_n: number of items to recommend

    Returns:
        list of (item_idx, predicted_score) tuples
    """
    # Cosine similarity between target user and all others
    user_vec = train_matrix[user_idx]
    similarities = cosine_similarity(user_vec, train_matrix).flatten()
    similarities[user_idx] = 0  # exclude self

    # Top-k similar users
    neighbor_idx = np.argsort(similarities)[::-1][:n_neighbors]
    neighbor_sims = similarities[neighbor_idx]

    # Weighted average of neighbor ratings
    if neighbor_sims.sum() == 0:
        return []

    neighbor_ratings = train_matrix[neighbor_idx].toarray()
    weighted_scores = neighbor_sims @ neighbor_ratings / (np.abs(neighbor_sims).sum() + 1e-8)

    # Exclude already-rated items
    rated_items = set(train_matrix[user_idx].nonzero()[1])
    candidates = [
        (i, weighted_scores[i])
        for i in np.argsort(weighted_scores)[::-1]
        if i not in rated_items
    ]

    return candidates[:top_n]


def item_based_cf(train_matrix, user_idx, n_neighbors=20, top_n=10):
    """
    Item-based collaborative filtering using cosine similarity.

    Args:
        train_matrix: sparse user-item matrix
        user_idx: target user row index
        n_neighbors: number of similar items per rated item
        top_n: number of items to recommend

    Returns:
        list of (item_idx, predicted_score) tuples
    """
    # Item-item similarity (transpose matrix)
    item_sim = cosine_similarity(train_matrix.T)

    user_ratings = train_matrix[user_idx].toarray().flatten()
    rated_items = np.where(user_ratings > 0)[0]

    if len(rated_items) == 0:
        return []

    scores = np.zeros(train_matrix.shape[1])
    sim_sums = np.zeros(train_matrix.shape[1])

    for item_idx in rated_items:
        sim_row = item_sim[item_idx]
        scores += sim_row * user_ratings[item_idx]
        sim_sums += np.abs(sim_row)

    # Normalize
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted = np.where(sim_sums > 0, scores / sim_sums, 0)

    # Exclude already-rated items
    rated_set = set(rated_items)
    candidates = [
        (i, predicted[i])
        for i in np.argsort(predicted)[::-1]
        if i not in rated_set
    ]

    return candidates[:top_n]


# =====================================================================
# CONTENT-BASED FILTERING
# =====================================================================

def build_content_similarity(items_df):
    """
    Build item-item similarity matrix from TF-IDF on item descriptions.

    Returns:
        tfidf_matrix: sparse TF-IDF matrix
        content_sim: dense cosine similarity matrix (n_items x n_items)
        tfidf_vectorizer: fitted TfidfVectorizer
    """
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    tfidf_matrix = tfidf.fit_transform(items_df["description"].fillna(""))
    content_sim = cosine_similarity(tfidf_matrix)

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Content similarity matrix shape: {content_sim.shape}")

    return tfidf_matrix, content_sim, tfidf


def content_based_recommend(user_idx, train_matrix, content_sim, top_n=10):
    """
    Content-based recommendations: for items a user liked, find similar items
    based on TF-IDF content similarity.

    Returns:
        list of (item_idx, score) tuples
    """
    user_ratings = train_matrix[user_idx].toarray().flatten()
    rated_items = np.where(user_ratings > 0)[0]

    if len(rated_items) == 0:
        return []

    # Weight content similarity by user ratings
    liked_items = rated_items[user_ratings[rated_items] >= 4]
    if len(liked_items) == 0:
        liked_items = rated_items  # fall back to all rated items

    scores = np.zeros(content_sim.shape[0])
    for item_idx in liked_items:
        scores += content_sim[item_idx] * user_ratings[item_idx]

    # Normalize by number of liked items
    scores /= (len(liked_items) + 1e-8)

    # Exclude already-rated
    rated_set = set(rated_items)
    candidates = [
        (i, scores[i])
        for i in np.argsort(scores)[::-1]
        if i not in rated_set
    ]

    return candidates[:top_n]


# =====================================================================
# MATRIX FACTORIZATION (SVD)
# =====================================================================

def train_svd(train_matrix, n_factors=50):
    """
    Train SVD on the user-item matrix using scipy sparse SVD.

    Returns:
        user_factors: (n_users, n_factors)
        sigma: (n_factors,)
        item_factors: (n_factors, n_items)
        predicted_ratings: full predicted rating matrix
    """
    # Mean-center the matrix
    dense = train_matrix.toarray().astype(np.float64)
    user_means = np.true_divide(dense.sum(axis=1), (dense > 0).sum(axis=1).clip(1))

    # Subtract user mean from non-zero entries
    centered = dense.copy()
    for i in range(centered.shape[0]):
        mask = centered[i] > 0
        centered[i, mask] -= user_means[i]

    # SVD
    k = min(n_factors, min(centered.shape) - 1)
    U, sigma, Vt = svds(csr_matrix(centered), k=k)

    # Sort by singular values (descending)
    idx = np.argsort(-sigma)
    U = U[:, idx]
    sigma = sigma[idx]
    Vt = Vt[idx, :]

    # Predicted ratings = U * diag(sigma) * Vt + user_means
    predicted = U @ np.diag(sigma) @ Vt
    predicted += user_means[:, np.newaxis]
    predicted = np.clip(predicted, 1, 5)

    print(f"SVD: {k} factors, explained variance in top factor: {sigma[0]:.2f}")

    return U, sigma, Vt, predicted, user_means


def svd_recommend(user_idx, predicted_ratings, train_matrix, top_n=10):
    """
    Recommend top-N items for a user based on SVD predicted ratings.

    Returns:
        list of (item_idx, predicted_rating) tuples
    """
    user_preds = predicted_ratings[user_idx]
    rated_items = set(train_matrix[user_idx].nonzero()[1])

    candidates = [
        (i, user_preds[i])
        for i in np.argsort(user_preds)[::-1]
        if i not in rated_items
    ]

    return candidates[:top_n]


# =====================================================================
# HYBRID MODEL
# =====================================================================

def hybrid_recommend(
    user_idx,
    train_matrix,
    predicted_ratings_svd,
    content_sim,
    w_collab=0.4,
    w_content=0.2,
    w_svd=0.4,
    top_n=10,
):
    """
    Hybrid recommendation combining collaborative filtering, content-based,
    and SVD scores with configurable weights.

    Returns:
        list of (item_idx, hybrid_score) tuples
    """
    n_items = train_matrix.shape[1]
    rated_items = set(train_matrix[user_idx].nonzero()[1])

    # Get scores from each method
    collab_recs = user_based_cf(train_matrix, user_idx, top_n=n_items)
    content_recs = content_based_recommend(user_idx, train_matrix, content_sim, top_n=n_items)
    svd_recs = svd_recommend(user_idx, predicted_ratings_svd, train_matrix, top_n=n_items)

    # Build score dictionaries
    collab_scores = {idx: score for idx, score in collab_recs}
    content_scores = {idx: score for idx, score in content_recs}
    svd_scores = {idx: score for idx, score in svd_recs}

    # Normalize each set of scores to [0, 1]
    def normalize(scores_dict):
        if not scores_dict:
            return scores_dict
        vals = np.array(list(scores_dict.values()))
        mn, mx = vals.min(), vals.max()
        if mx - mn == 0:
            return {k: 0.5 for k in scores_dict}
        return {k: (v - mn) / (mx - mn) for k, v in scores_dict.items()}

    collab_norm = normalize(collab_scores)
    content_norm = normalize(content_scores)
    svd_norm = normalize(svd_scores)

    # Combine scores
    all_items = set(collab_norm.keys()) | set(content_norm.keys()) | set(svd_norm.keys())
    hybrid_scores = {}
    for item_idx in all_items:
        if item_idx in rated_items:
            continue
        score = (
            w_collab * collab_norm.get(item_idx, 0)
            + w_content * content_norm.get(item_idx, 0)
            + w_svd * svd_norm.get(item_idx, 0)
        )
        hybrid_scores[item_idx] = score

    # Sort and return top-N
    sorted_items = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_n]


# =====================================================================
# COLD-START HANDLING
# =====================================================================

def cold_start_recommend(items_df, content_sim, category=None, top_n=10):
    """
    Recommend items for a new user (cold start) based on overall popularity
    and optionally filtered by category.

    Returns:
        list of item_ids
    """
    candidates = items_df.copy()
    if category:
        cat_items = candidates[candidates["category"] == category]
        if len(cat_items) > 0:
            candidates = cat_items

    # Score by popularity (num_ratings * avg_rating)
    candidates = candidates.copy()
    candidates["popularity_score"] = candidates["num_ratings"] * candidates["avg_rating"]
    top_items = candidates.nlargest(top_n, "popularity_score")

    return top_items["item_id"].tolist()


# =====================================================================
# EVALUATION METRICS
# =====================================================================

def rmse(actual, predicted):
    """Root mean squared error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mae(actual, predicted):
    """Mean absolute error."""
    return np.mean(np.abs(actual - predicted))


def precision_at_k(recommended, relevant, k):
    """Precision@K: fraction of top-K recommendations that are relevant."""
    if k == 0:
        return 0.0
    rec_at_k = recommended[:k]
    hits = len(set(rec_at_k) & set(relevant))
    return hits / k


def recall_at_k(recommended, relevant, k):
    """Recall@K: fraction of relevant items found in top-K."""
    if len(relevant) == 0:
        return 0.0
    rec_at_k = recommended[:k]
    hits = len(set(rec_at_k) & set(relevant))
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    """Normalized discounted cumulative gain at K."""
    if k == 0 or len(relevant) == 0:
        return 0.0
    rec_at_k = recommended[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(rec_at_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)

    # Ideal DCG
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommendations(
    train_matrix,
    test_ratings,
    predicted_ratings_svd,
    content_sim,
    idx_to_item_id,
    item_id_to_idx,
    k=10,
    n_eval_users=200,
):
    """
    Evaluate all recommendation methods on the test set.

    Returns:
        results_dict with metrics per method
    """
    rng = np.random.RandomState(RANDOM_STATE)

    # Select users that have test ratings
    test_users = test_ratings["user_id"].unique()
    if len(test_users) > n_eval_users:
        test_users = rng.choice(test_users, n_eval_users, replace=False)

    methods = {
        "User-based CF": lambda u: user_based_cf(train_matrix, u, top_n=k),
        "Item-based CF": lambda u: item_based_cf(train_matrix, u, top_n=k),
        "Content-based": lambda u: content_based_recommend(u, train_matrix, content_sim, top_n=k),
        "SVD": lambda u: svd_recommend(u, predicted_ratings_svd, train_matrix, top_n=k),
        "Hybrid": lambda u: hybrid_recommend(
            u, train_matrix, predicted_ratings_svd, content_sim, top_n=k
        ),
    }

    results = {name: {"precision": [], "recall": [], "ndcg": []} for name in methods}

    # Also compute RMSE and MAE for SVD (rating prediction)
    svd_actual = []
    svd_predicted = []

    for uid in test_users:
        if uid not in item_id_to_idx.__class__(
            {uid: 0 for uid in [uid]}
        ):  # always true, just need idx
            pass

        # Get user's test items (relevant = items rated >= 4 in test)
        user_test = test_ratings[test_ratings["user_id"] == uid]
        relevant_item_ids = user_test[user_test["rating"] >= 4]["item_id"].values
        relevant_idx = [item_id_to_idx[iid] for iid in relevant_item_ids if iid in item_id_to_idx]

        if len(relevant_idx) == 0:
            continue

        # User index in matrix
        from src.data_loader import build_user_item_matrix
        # We need user_id_to_idx - pass it externally or compute
        # For simplicity, user_id - 1 = idx (since IDs are 1-indexed sequential)
        user_matrix_idx = uid - 1
        if user_matrix_idx >= train_matrix.shape[0]:
            continue

        # SVD RMSE/MAE
        for _, row in user_test.iterrows():
            iid = row["item_id"]
            if iid in item_id_to_idx:
                item_matrix_idx = item_id_to_idx[iid]
                svd_actual.append(row["rating"])
                svd_predicted.append(predicted_ratings_svd[user_matrix_idx, item_matrix_idx])

        # Evaluate each method
        for name, rec_fn in methods.items():
            try:
                recs = rec_fn(user_matrix_idx)
                rec_item_idx = [r[0] for r in recs]
            except Exception:
                rec_item_idx = []

            results[name]["precision"].append(precision_at_k(rec_item_idx, relevant_idx, k))
            results[name]["recall"].append(recall_at_k(rec_item_idx, relevant_idx, k))
            results[name]["ndcg"].append(ndcg_at_k(rec_item_idx, relevant_idx, k))

    # Aggregate
    summary = {}
    for name, metrics in results.items():
        summary[name] = {
            f"Precision@{k}": np.mean(metrics["precision"]),
            f"Recall@{k}": np.mean(metrics["recall"]),
            f"NDCG@{k}": np.mean(metrics["ndcg"]),
        }

    # Add RMSE and MAE for SVD
    if svd_actual:
        svd_actual = np.array(svd_actual)
        svd_predicted = np.array(svd_predicted)
        summary["SVD"]["RMSE"] = rmse(svd_actual, svd_predicted)
        summary["SVD"]["MAE"] = mae(svd_actual, svd_predicted)

    return summary


def compute_coverage_diversity(recommendations, n_total_items, items_df, item_id_to_idx):
    """
    Compute catalog coverage and recommendation diversity.

    Args:
        recommendations: dict of user_id -> list of recommended item_idx
        n_total_items: total items in catalog
        items_df: items DataFrame
        item_id_to_idx: mapping from item_id to matrix index

    Returns:
        coverage: fraction of catalog recommended to at least one user
        avg_diversity: average pairwise category difference within rec lists
    """
    all_recommended = set()
    diversity_scores = []

    idx_to_item_id = {v: k for k, v in item_id_to_idx.items()}
    item_categories = dict(zip(items_df["item_id"], items_df["category"]))

    for uid, rec_list in recommendations.items():
        all_recommended.update(rec_list)

        # Diversity: fraction of unique categories in recommendation list
        cats = []
        for idx in rec_list:
            iid = idx_to_item_id.get(idx)
            if iid and iid in item_categories:
                cats.append(item_categories[iid])

        if len(cats) > 1:
            unique_cats = len(set(cats))
            diversity_scores.append(unique_cats / len(cats))

    coverage = len(all_recommended) / n_total_items if n_total_items > 0 else 0
    avg_diversity = np.mean(diversity_scores) if diversity_scores else 0

    return coverage, avg_diversity


# =====================================================================
# TRAINING PIPELINE
# =====================================================================

def train_and_evaluate(
    train_matrix,
    test_ratings,
    items_df,
    item_id_to_idx,
    idx_to_item_id,
    n_factors=50,
    k=10,
):
    """
    Full training and evaluation pipeline.

    Returns:
        summary: metrics DataFrame
        predicted_ratings: SVD predicted rating matrix
        content_sim: content similarity matrix
    """
    _ensure_dirs()

    print("=" * 70)
    print("TRAINING RECOMMENDATION MODELS")
    print("=" * 70)

    # 1. Content-based similarity
    print("\n--- Building content similarity ---")
    tfidf_matrix, content_sim, tfidf_vec = build_content_similarity(items_df)

    # 2. SVD
    print("\n--- Training SVD ---")
    U, sigma, Vt, predicted_ratings, user_means = train_svd(train_matrix, n_factors=n_factors)

    # 3. Evaluate all methods
    print("\n--- Evaluating all methods ---")
    summary = evaluate_recommendations(
        train_matrix,
        test_ratings,
        predicted_ratings,
        content_sim,
        idx_to_item_id,
        item_id_to_idx,
        k=k,
    )

    # Print results
    summary_df = pd.DataFrame(summary).T.round(4)
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS (top-{k})")
    print("=" * 70)
    print(summary_df.to_string())

    # Save results
    summary_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"))
    print(f"\nSaved model comparison to {OUTPUTS_DIR}/model_comparison.csv")

    # Save models
    joblib.dump({
        "U": U, "sigma": sigma, "Vt": Vt,
        "user_means": user_means,
        "predicted_ratings": predicted_ratings,
    }, os.path.join(MODELS_DIR, "svd_model.joblib"))

    joblib.dump(content_sim, os.path.join(MODELS_DIR, "content_similarity.joblib"))
    joblib.dump(tfidf_vec, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    print(f"Saved models to {MODELS_DIR}/")

    # Generate plots
    _plot_metrics_comparison(summary_df, k)
    _plot_rating_distribution_svd(predicted_ratings, train_matrix)
    _plot_similarity_heatmap(content_sim, items_df)

    return summary_df, predicted_ratings, content_sim


def _plot_metrics_comparison(summary_df, k):
    """Bar chart comparing recommendation metrics across methods."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metric_cols = [c for c in summary_df.columns if c.startswith(("Precision", "Recall", "NDCG"))]

    for ax, col in zip(axes, metric_cols[:3]):
        vals = summary_df[col].sort_values(ascending=True)
        colors = ["#E8C230" if v == vals.max() else "#3B6FD4" for v in vals.values]
        ax.barh(vals.index, vals.values, color=colors, edgecolor="black", alpha=0.85)
        ax.set_title(col)
        ax.set_xlim(0, max(vals.max() * 1.2, 0.1))
        for i, v in enumerate(vals.values):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

    fig.suptitle(f"Recommendation metrics comparison (top-{k})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved metrics comparison plot.")


def _plot_rating_distribution_svd(predicted_ratings, train_matrix):
    """Compare actual vs SVD predicted rating distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    actual_ratings = train_matrix.toarray().flatten()
    actual_ratings = actual_ratings[actual_ratings > 0]

    axes[0].hist(actual_ratings, bins=5, range=(0.5, 5.5), color="#3B6FD4",
                 edgecolor="black", alpha=0.8, density=True)
    axes[0].set_title("Actual rating distribution")
    axes[0].set_xlabel("Rating")
    axes[0].set_ylabel("Density")

    pred_flat = predicted_ratings.flatten()
    axes[1].hist(pred_flat, bins=50, color="#E8C230", edgecolor="black", alpha=0.8, density=True)
    axes[1].set_title("SVD predicted rating distribution")
    axes[1].set_xlabel("Predicted rating")
    axes[1].set_ylabel("Density")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "rating_distributions.png"), dpi=150)
    plt.close(fig)
    print("Saved rating distributions plot.")


def _plot_similarity_heatmap(content_sim, items_df):
    """Plot a sample of the content similarity matrix grouped by category."""
    # Take first 50 items, sorted by category
    sample_idx = items_df.head(50).sort_values("category").index.tolist()
    sample_sim = content_sim[np.ix_(sample_idx, sample_idx)]
    sample_labels = items_df.iloc[sample_idx]["category"].values

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(sample_sim, cmap="YlOrRd", ax=ax, xticklabels=False, yticklabels=sample_labels,
                vmin=0, vmax=1)
    ax.set_title("Content similarity heatmap (first 50 items by category)")
    ax.set_ylabel("Item category")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "content_similarity_heatmap.png"), dpi=150)
    plt.close(fig)
    print("Saved content similarity heatmap.")


if __name__ == "__main__":
    from data_loader import load_data, build_user_item_matrix, train_test_split_ratings

    users, items, ratings = load_data()
    train_ratings, test_ratings = train_test_split_ratings(ratings)
    train_matrix, u2i, i2i, i2u, i2item = build_user_item_matrix(train_ratings)

    summary, pred_ratings, content_sim = train_and_evaluate(
        train_matrix, test_ratings, items, i2i, i2item
    )
