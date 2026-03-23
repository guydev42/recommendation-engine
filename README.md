<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Hybrid%20Recommendation%20Engine&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Combining%20collaborative%20filtering%2C%20content-based%20filtering%2C%20and%20SVD%20for%20personalized%20item%20recommendations&descAlignY=55&descSize=16" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/NDCG%4010-0.78-9558B2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/RMSE-0.91-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Ratings-50K-f59e0b?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge"/>
</p>

<p>
  <a href="#overview">Overview</a> •
  <a href="#key-results">Key results</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#methodology">Methodology</a>
</p>

</div>

---

## Overview

> **A hybrid recommendation engine combining collaborative filtering, content-based filtering, and matrix factorization to achieve 0.78 NDCG@10 across 500 items.**

Recommendation systems are central to e-commerce, streaming, and content platforms. This project builds a hybrid engine that combines three approaches -- user-based/item-based collaborative filtering (cosine similarity on sparse matrices), content-based filtering (TF-IDF on item descriptions), and matrix factorization (truncated SVD) -- into a weighted hybrid that outperforms any single method. The system handles cold-start users through popularity-based fallback and provides "because you liked X" explanations for every recommendation.

```
Problem   →  Users face information overload when choosing from 500+ items
Solution  →  Hybrid engine (CF + content + SVD) ranks items with 0.78 NDCG@10
Impact    →  0.91 RMSE on rating prediction, with explainable recommendations
```

---

## Key results

| Metric | Value |
|--------|-------|
| Best method | Hybrid (CF + content + SVD) |
| NDCG@10 | 0.78 |
| RMSE | 0.91 |
| Users | 2,000 |
| Items | 500 |
| Ratings | ~50,000 |
| Matrix sparsity | ~95% |

**Key findings**

- **The hybrid model outperforms every individual method** on NDCG@10, confirming that combining collaborative and content signals produces better ranking
- **SVD with 50 latent factors provides the best single-method RMSE** at 0.91, capturing the dominant patterns in the rating matrix
- **Optimal hybrid weights are 40% CF + 20% content + 40% SVD** -- collaborative signals dominate but content adds measurable value
- **Cold-start users receive reasonable recommendations** through popularity-based fallback filtered by category preference
- **Item-based CF is more stable than user-based CF** on this dataset, consistent with item profiles changing less frequently than user profiles

---

## Architecture

```
┌─────────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
│  Users (2K)     │───▶│  User-item matrix    │───▶│  Collaborative       │
│  Items (500)    │    │  (sparse, 95%)       │    │  filtering (CF)      │
│  Ratings (50K)  │    └──────────┬──────────┘    └──────────┬───────────┘
└─────────────────┘               │                          │
                                  │                          ▼
                    ┌─────────────┘              ┌──────────────────────┐
                    ▼                            │  Hybrid combiner     │
          ┌──────────────────┐                   │  (weighted scores)   │
          │  SVD (50 factors)│──────────────────▶│                      │
          └──────────────────┘                   └──────────┬───────────┘
                                                            │
┌─────────────────┐    ┌──────────────────┐                 ▼
│  Item            │───▶│  TF-IDF +        │───▶  ┌──────────────────────┐
│  descriptions    │    │  content sim     │      │  Top-N recs +        │
└─────────────────┘    └──────────────────┘      │  explanations        │
                                                  └──────────────────────┘
```

<details>
<summary><b>Project structure</b></summary>

```
project_22_recommendation_engine/
├── data/                  # Users, items, ratings CSVs
│   └── generate_data.py   # Synthetic data generator
├── src/                   # Data loading, model training
│   ├── __init__.py
│   ├── data_loader.py
│   └── model.py
├── models/                # Saved SVD model, content similarity, TF-IDF
├── outputs/               # Plots, comparison tables
├── notebooks/             # EDA, collaborative filtering, SVD, hybrid eval
├── app.py                 # Streamlit dashboard (5 pages)
├── requirements.txt       # Python dependencies
├── index.html             # Project landing page
└── README.md
```

</details>

---

## Quickstart

```bash
# Clone and navigate
git clone https://github.com/guydev42/calgary-data-portfolio.git
cd calgary-data-portfolio/project_22_recommendation_engine

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python data/generate_data.py

# Train models and generate outputs
python -c "
from src.data_loader import load_data, build_user_item_matrix, train_test_split_ratings
from src.model import train_and_evaluate
users, items, ratings = load_data()
train_ratings, test_ratings = train_test_split_ratings(ratings)
train_matrix, u2i, i2i, i2u, i2item = build_user_item_matrix(train_ratings)
train_and_evaluate(train_matrix, test_ratings, items, i2i, i2item)
"

# Launch dashboard
streamlit run app.py
```

---

## Dataset

| Property | Details |
|----------|---------|
| Source | Synthetic user-item interactions |
| Users | 2,000 with age_group, region, signup_months_ago |
| Items | 500 across 10 categories with descriptions, price_tier, avg_rating |
| Ratings | ~50,000 on 1-5 scale |
| Sparsity | ~95% (typical for recommendation datasets) |
| Categories | Electronics, Books, Clothing, Home & Kitchen, Sports, Movies, Music, Games, Tools, Toys |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white"/>
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge"/>
</p>

---

## Methodology

<details>
<summary><b>Collaborative filtering</b></summary>

- User-based CF: cosine similarity between user rating vectors, top-K neighbor weighted average
- Item-based CF: cosine similarity between item rating vectors, weighted by user ratings
- Sparse matrix representation for memory efficiency (CSR format)
- Neighborhood size K=20 selected by validation
</details>

<details>
<summary><b>Content-based filtering</b></summary>

- TF-IDF vectorization of item descriptions (unigrams + bigrams, 5000 features)
- Cosine similarity between TF-IDF vectors
- Recommendations weighted by user's rating of similar items
- Items rated >= 4 are treated as "liked" for content matching
</details>

<details>
<summary><b>Matrix factorization (SVD)</b></summary>

- Mean-centered user-item matrix decomposed via scipy sparse SVD
- 50 latent factors selected by validation RMSE
- Predicted ratings = U * diag(sigma) * Vt + user_means
- Ratings clipped to [1, 5] range
</details>

<details>
<summary><b>Hybrid model</b></summary>

- Weighted combination of normalized CF, content, and SVD scores
- Optimal weights: 40% CF + 20% content + 40% SVD
- Score normalization to [0, 1] before combining
- Already-rated items excluded from recommendations
</details>

<details>
<summary><b>Cold-start handling</b></summary>

- New users: popularity-based ranking (num_ratings * avg_rating)
- Optional category filter for preference-guided cold start
- Transition to hybrid model once user accumulates 5+ ratings
</details>

<details>
<summary><b>Evaluation</b></summary>

- Precision@K, Recall@K, NDCG@K for ranking quality
- RMSE, MAE for rating prediction accuracy
- Catalog coverage and recommendation diversity
- Stratified user-level evaluation (80/20 train/test per user)
</details>

---

## Acknowledgements

Dataset generated synthetically for this project. Built as part of the [Calgary Data Portfolio](https://guydev42.github.io/calgary-data-portfolio/).

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**[Ola K.](https://github.com/guydev42)**
</div>
