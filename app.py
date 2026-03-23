"""
Streamlit dashboard for the hybrid recommendation engine.
Pages: User recommendations, Because you liked, Item similarity, Metrics comparison, Coverage stats.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import matplotlib
matplotlib.use("Agg")

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recommendation engine dashboard", layout="wide")

DATA_DIR = "data"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"


@st.cache_data
def load_data():
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    return users, items, ratings


@st.cache_data
def build_matrix(ratings):
    unique_users = sorted(ratings["user_id"].unique())
    unique_items = sorted(ratings["item_id"].unique())
    user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    item_id_to_idx = {iid: i for i, iid in enumerate(unique_items)}
    idx_to_user_id = {i: uid for uid, i in user_id_to_idx.items()}
    idx_to_item_id = {i: iid for iid, i in item_id_to_idx.items()}

    rows = ratings["user_id"].map(user_id_to_idx).values
    cols = ratings["item_id"].map(item_id_to_idx).values
    vals = ratings["rating"].values.astype(np.float32)

    matrix = csr_matrix((vals, (rows, cols)), shape=(len(unique_users), len(unique_items)))
    return matrix, user_id_to_idx, item_id_to_idx, idx_to_user_id, idx_to_item_id


@st.cache_resource
def load_models():
    svd_path = os.path.join(MODELS_DIR, "svd_model.joblib")
    content_path = os.path.join(MODELS_DIR, "content_similarity.joblib")

    svd_model = None
    content_sim = None

    if os.path.exists(svd_path):
        svd_model = joblib.load(svd_path)
    if os.path.exists(content_path):
        content_sim = joblib.load(content_path)

    return svd_model, content_sim


@st.cache_data
def load_comparison():
    path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


def get_hybrid_recs(user_idx, matrix, predicted_ratings, content_sim, w_cf, w_cb, w_svd, top_n=10):
    """Get hybrid recommendations with configurable weights."""
    n_items = matrix.shape[1]
    rated_items = set(matrix[user_idx].nonzero()[1])

    # User-based CF scores
    user_vec = matrix[user_idx]
    sims = cosine_similarity(user_vec, matrix).flatten()
    sims[user_idx] = 0
    top_neighbors = np.argsort(sims)[::-1][:20]
    neighbor_sims = sims[top_neighbors]
    if neighbor_sims.sum() > 0:
        neighbor_ratings = matrix[top_neighbors].toarray()
        cf_scores = neighbor_sims @ neighbor_ratings / (np.abs(neighbor_sims).sum() + 1e-8)
    else:
        cf_scores = np.zeros(n_items)

    # Content-based scores
    user_ratings = matrix[user_idx].toarray().flatten()
    liked_mask = user_ratings >= 4
    if liked_mask.sum() == 0:
        liked_mask = user_ratings > 0
    cb_scores = np.zeros(n_items)
    if content_sim is not None and liked_mask.sum() > 0:
        liked_idx = np.where(liked_mask)[0]
        for i in liked_idx:
            cb_scores += content_sim[i] * user_ratings[i]
        cb_scores /= (len(liked_idx) + 1e-8)

    # SVD scores
    svd_scores = predicted_ratings[user_idx] if predicted_ratings is not None else np.zeros(n_items)

    # Normalize each to [0, 1]
    def norm(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn == 0:
            return np.full_like(arr, 0.5)
        return (arr - mn) / (mx - mn)

    combined = w_cf * norm(cf_scores) + w_cb * norm(cb_scores) + w_svd * norm(svd_scores)

    # Zero out rated items
    for idx in rated_items:
        combined[idx] = -1

    top_idx = np.argsort(combined)[::-1][:top_n]
    return [(idx, combined[idx]) for idx in top_idx]


# ---- Sidebar ----
page = st.sidebar.radio(
    "Navigate",
    ["User recommendations", "Because you liked", "Item similarity", "Metrics comparison", "Coverage and diversity"],
)

users, items, ratings = load_data()
matrix, u2i, i2i, i2u, i2item = build_matrix(ratings)
svd_model, content_sim = load_models()
comparison_df = load_comparison()

predicted_ratings = None
if svd_model is not None:
    predicted_ratings = svd_model.get("predicted_ratings")


# =====================================================================
# PAGE 1: USER RECOMMENDATIONS
# =====================================================================
if page == "User recommendations":
    st.title("Top-N recommendations")
    st.markdown("Select a user and adjust hybrid weights to generate personalized recommendations.")

    col1, col2 = st.columns([1, 2])
    with col1:
        user_id = st.selectbox("Select user", sorted(ratings["user_id"].unique())[:200])
        top_n = st.slider("Number of recommendations", 5, 25, 10)
        st.markdown("**Hybrid weights**")
        w_cf = st.slider("Collaborative filtering", 0.0, 1.0, 0.4, 0.05)
        w_cb = st.slider("Content-based", 0.0, 1.0, 0.2, 0.05)
        w_svd = st.slider("SVD", 0.0, 1.0, 0.4, 0.05)

        # Normalize weights
        total_w = w_cf + w_cb + w_svd
        if total_w > 0:
            w_cf, w_cb, w_svd = w_cf / total_w, w_cb / total_w, w_svd / total_w

    with col2:
        if user_id in u2i:
            user_idx = u2i[user_id]

            # User profile
            user_info = users[users["user_id"] == user_id].iloc[0]
            user_ratings = ratings[ratings["user_id"] == user_id]
            st.markdown(f"**User {user_id}** | {user_info['age_group']} | {user_info['region']} | "
                        f"Signed up {user_info['signup_months_ago']} months ago | "
                        f"{len(user_ratings)} ratings (avg {user_ratings['rating'].mean():.1f})")

            # Get recommendations
            if predicted_ratings is not None and content_sim is not None:
                recs = get_hybrid_recs(user_idx, matrix, predicted_ratings, content_sim,
                                       w_cf, w_cb, w_svd, top_n)
                rec_data = []
                for idx, score in recs:
                    item_id = i2item[idx]
                    item_row = items[items["item_id"] == item_id].iloc[0]
                    rec_data.append({
                        "Rank": len(rec_data) + 1,
                        "Item ID": item_id,
                        "Category": item_row["category"],
                        "Price tier": item_row["price_tier"],
                        "Avg rating": item_row["avg_rating"],
                        "Score": round(score, 3),
                    })

                rec_df = pd.DataFrame(rec_data)
                st.dataframe(rec_df, use_container_width=True, hide_index=True)

                # Category distribution of recs
                fig = px.pie(rec_df, names="Category", title="Recommended item categories",
                             color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Models not trained yet. Run `python -c \"from src.data_loader import ...; ...\"` first.")


# =====================================================================
# PAGE 2: BECAUSE YOU LIKED
# =====================================================================
elif page == "Because you liked":
    st.title("Because you liked...")
    st.markdown("See why specific items were recommended based on your rating history.")

    user_id = st.selectbox("Select user", sorted(ratings["user_id"].unique())[:200])

    if user_id in u2i:
        user_idx = u2i[user_id]
        user_ratings = ratings[ratings["user_id"] == user_id].merge(items, on="item_id")
        top_rated = user_ratings.nlargest(10, "rating")

        st.markdown("### Your top-rated items")
        st.dataframe(
            top_rated[["item_id", "category", "description", "rating", "price_tier"]],
            use_container_width=True, hide_index=True,
        )

        if content_sim is not None:
            st.markdown("### Recommendations explained")
            selected_item = st.selectbox(
                "Pick a liked item to see similar recommendations",
                top_rated["item_id"].tolist(),
                format_func=lambda x: f"Item {x} - {items[items['item_id']==x].iloc[0]['category']} "
                                      f"({items[items['item_id']==x].iloc[0]['description'][:50]}...)",
            )

            if selected_item in i2i:
                item_idx = i2i[selected_item]
                sim_scores = content_sim[item_idx]
                rated_item_ids = set(user_ratings["item_id"])

                # Find similar unrated items
                sim_items = []
                for i in np.argsort(sim_scores)[::-1]:
                    iid = i2item[i]
                    if iid != selected_item and iid not in rated_item_ids:
                        item_row = items[items["item_id"] == iid].iloc[0]
                        sim_items.append({
                            "Item ID": iid,
                            "Category": item_row["category"],
                            "Description": item_row["description"],
                            "Similarity": round(sim_scores[i], 3),
                            "Avg rating": item_row["avg_rating"],
                        })
                    if len(sim_items) >= 10:
                        break

                sim_df = pd.DataFrame(sim_items)
                st.dataframe(sim_df, use_container_width=True, hide_index=True)

                # Similarity bar chart
                fig = px.bar(
                    sim_df, x="Similarity", y="Item ID", orientation="h",
                    title=f"Items similar to Item {selected_item}",
                    color="Similarity",
                    color_continuous_scale=["#3B6FD4", "#E8C230"],
                )
                fig.update_layout(yaxis=dict(type="category"))
                st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# PAGE 3: ITEM SIMILARITY EXPLORER
# =====================================================================
elif page == "Item similarity":
    st.title("Item similarity explorer")
    st.markdown("Explore content-based similarity between items using TF-IDF on descriptions.")

    col1, col2 = st.columns(2)
    with col1:
        category_filter = st.selectbox("Filter by category", ["All"] + sorted(items["category"].unique()))
    with col2:
        n_similar = st.slider("Number of similar items", 5, 20, 10)

    filtered_items = items if category_filter == "All" else items[items["category"] == category_filter]
    selected_item = st.selectbox(
        "Select an item",
        filtered_items["item_id"].tolist(),
        format_func=lambda x: f"Item {x} - {items[items['item_id']==x].iloc[0]['category']}: "
                              f"{items[items['item_id']==x].iloc[0]['description'][:60]}",
    )

    if content_sim is not None and selected_item in i2i:
        item_idx = i2i[selected_item]
        sim_scores = content_sim[item_idx]

        similar = []
        for i in np.argsort(sim_scores)[::-1]:
            iid = i2item[i]
            if iid != selected_item:
                row = items[items["item_id"] == iid].iloc[0]
                similar.append({
                    "Item ID": iid,
                    "Category": row["category"],
                    "Description": row["description"],
                    "Similarity": round(sim_scores[i], 3),
                    "Price tier": row["price_tier"],
                    "Avg rating": row["avg_rating"],
                })
            if len(similar) >= n_similar:
                break

        sim_df = pd.DataFrame(similar)
        st.dataframe(sim_df, use_container_width=True, hide_index=True)

        # Heatmap for top similar items
        top_idx = [i2i[selected_item]] + [i2i[s["Item ID"]] for s in similar if s["Item ID"] in i2i]
        if len(top_idx) > 1:
            sub_sim = content_sim[np.ix_(top_idx, top_idx)]
            labels = [f"Item {i2item[i]}" for i in top_idx]
            fig = px.imshow(
                sub_sim, x=labels, y=labels,
                color_continuous_scale="YlOrRd",
                title="Pairwise similarity heatmap",
            )
            fig.update_layout(width=600, height=500)
            st.plotly_chart(fig, use_container_width=True)
    elif content_sim is None:
        st.warning("Content similarity model not found. Train models first.")


# =====================================================================
# PAGE 4: METRICS COMPARISON
# =====================================================================
elif page == "Metrics comparison":
    st.title("Model metrics comparison")
    st.markdown("Compare Precision@K, Recall@K, and NDCG@K across all recommendation methods.")

    if comparison_df is not None:
        st.dataframe(comparison_df.style.highlight_max(axis=0, color="#E8C230"),
                     use_container_width=True)

        # Bar charts for each metric
        metric_cols = [c for c in comparison_df.columns if any(
            c.startswith(p) for p in ("Precision", "Recall", "NDCG")
        )]

        for col in metric_cols:
            fig = px.bar(
                comparison_df.reset_index().rename(columns={"index": "Method"}),
                x="Method", y=col,
                title=col,
                color=col,
                color_continuous_scale=["#3B6FD4", "#E8C230"],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # RMSE and MAE if available
        if "RMSE" in comparison_df.columns:
            col1, col2 = st.columns(2)
            with col1:
                svd_rmse = comparison_df.loc["SVD", "RMSE"] if "SVD" in comparison_df.index else None
                if svd_rmse is not None:
                    st.metric("SVD RMSE", f"{svd_rmse:.4f}")
            with col2:
                svd_mae = comparison_df.loc["SVD", "MAE"] if "SVD" in comparison_df.index else None
                if svd_mae is not None:
                    st.metric("SVD MAE", f"{svd_mae:.4f}")

        # Radar chart
        if len(metric_cols) >= 3:
            fig = go.Figure()
            for method in comparison_df.index:
                vals = comparison_df.loc[method, metric_cols].values.tolist()
                vals.append(vals[0])  # close the polygon
                fig.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=metric_cols + [metric_cols[0]],
                    fill="toself",
                    name=method,
                    opacity=0.6,
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Method comparison radar chart",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No model comparison data found. Train models first.")


# =====================================================================
# PAGE 5: COVERAGE AND DIVERSITY
# =====================================================================
elif page == "Coverage and diversity":
    st.title("Coverage and diversity statistics")
    st.markdown("Analyze how well recommendations cover the item catalog and how diverse they are.")

    # Basic stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total users", f"{len(users):,}")
    with col2:
        st.metric("Total items", f"{len(items):,}")
    with col3:
        st.metric("Total ratings", f"{len(ratings):,}")
    with col4:
        sparsity = 1 - len(ratings) / (len(users) * len(items))
        st.metric("Matrix sparsity", f"{sparsity:.1%}")

    st.markdown("---")

    # Rating distribution
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(ratings, x="rating", nbins=5, title="Rating distribution",
                          color_discrete_sequence=["#3B6FD4"])
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ratings_per_user = ratings.groupby("user_id").size()
        fig = px.histogram(ratings_per_user, nbins=50, title="Ratings per user distribution",
                          color_discrete_sequence=["#E8C230"])
        fig.update_layout(xaxis_title="Number of ratings", yaxis_title="Number of users")
        st.plotly_chart(fig, use_container_width=True)

    # Items per category
    cat_counts = items["category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    fig = px.bar(cat_counts, x="Category", y="Count", title="Items per category",
                 color="Count", color_continuous_scale=["#3B6FD4", "#E8C230"])
    st.plotly_chart(fig, use_container_width=True)

    # Ratings per category
    ratings_with_cat = ratings.merge(items[["item_id", "category"]], on="item_id")
    cat_rating_counts = ratings_with_cat["category"].value_counts().reset_index()
    cat_rating_counts.columns = ["Category", "Ratings"]
    fig = px.bar(cat_rating_counts, x="Category", y="Ratings",
                 title="Number of ratings per category",
                 color="Ratings", color_continuous_scale=["#3B6FD4", "#E8C230"])
    st.plotly_chart(fig, use_container_width=True)

    # Avg rating by category
    avg_by_cat = ratings_with_cat.groupby("category")["rating"].mean().reset_index()
    avg_by_cat.columns = ["Category", "Average rating"]
    fig = px.bar(avg_by_cat.sort_values("Average rating", ascending=False),
                 x="Category", y="Average rating",
                 title="Average rating by category",
                 color="Average rating", color_continuous_scale=["#3B6FD4", "#E8C230"])
    fig.update_layout(yaxis_range=[0, 5])
    st.plotly_chart(fig, use_container_width=True)

    # Price tier distribution
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(items, names="price_tier", title="Item price tier distribution",
                     color_discrete_sequence=["#3B6FD4", "#E8C230", "#162240"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        region_counts = users["region"].value_counts().reset_index()
        region_counts.columns = ["Region", "Users"]
        fig = px.pie(region_counts, names="Region", values="Users",
                     title="User region distribution",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
