"""
Generate synthetic recommendation engine data: users, items, and ratings.
2,000 users, 500 items, ~50K ratings with ~95% sparsity.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

N_USERS = 2000
N_ITEMS = 500
TARGET_RATINGS = 50000

# ---- Users ----
age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
age_probs = [0.15, 0.30, 0.25, 0.18, 0.12]

regions = ["West", "Prairies", "Ontario", "Quebec", "Atlantic"]
region_probs = [0.22, 0.18, 0.30, 0.20, 0.10]

users = pd.DataFrame({
    "user_id": range(1, N_USERS + 1),
    "age_group": np.random.choice(age_groups, N_USERS, p=age_probs),
    "region": np.random.choice(regions, N_USERS, p=region_probs),
    "signup_months_ago": np.random.randint(1, 60, N_USERS),
})

# ---- Items ----
categories = [
    "Electronics", "Books", "Clothing", "Home & Kitchen", "Sports",
    "Movies", "Music", "Games", "Tools", "Toys",
]
cat_probs = [0.15, 0.14, 0.12, 0.12, 0.10, 0.10, 0.08, 0.08, 0.06, 0.05]

price_tiers = ["budget", "mid-range", "premium"]
price_probs = [0.40, 0.40, 0.20]

# Item descriptions for content-based filtering
description_templates = {
    "Electronics": [
        "wireless bluetooth device with long battery life",
        "high performance gadget with fast processor",
        "compact portable electronics with usb charging",
        "smart home device with voice control features",
        "digital display screen with high resolution output",
    ],
    "Books": [
        "fiction novel with mystery and adventure themes",
        "nonfiction guide covering science and technology",
        "self help book on productivity and habits",
        "historical narrative about world events",
        "biography of a notable public figure",
    ],
    "Clothing": [
        "casual cotton shirt comfortable for daily wear",
        "formal business attire with tailored fit",
        "outdoor jacket waterproof and windproof",
        "athletic wear with moisture wicking fabric",
        "winter accessories warm scarf and gloves set",
    ],
    "Home & Kitchen": [
        "stainless steel kitchen appliance easy to clean",
        "decorative home accent modern minimalist design",
        "storage organizer for closet and pantry",
        "nonstick cookware set for everyday cooking",
        "bedding set soft cotton with high thread count",
    ],
    "Sports": [
        "fitness equipment for home gym training",
        "running shoes lightweight with arch support",
        "yoga mat thick and non slip surface",
        "cycling accessories helmet and lights set",
        "camping gear tent and sleeping bag combo",
    ],
    "Movies": [
        "action thriller movie with stunning visual effects",
        "comedy film family friendly and lighthearted",
        "documentary exploring nature and wildlife",
        "drama series with compelling character arcs",
        "animated feature film colorful and imaginative",
    ],
    "Music": [
        "rock album with electric guitar and drums",
        "classical compilation piano and orchestra pieces",
        "pop music collection upbeat and catchy tunes",
        "jazz recordings smooth saxophone and piano",
        "electronic beats ambient and downtempo tracks",
    ],
    "Games": [
        "board game strategy and puzzle solving fun",
        "video game action adventure open world",
        "card game multiplayer competitive gameplay",
        "role playing game with character progression",
        "party game group activity for all ages",
    ],
    "Tools": [
        "power drill cordless with multiple speed settings",
        "hand tool set wrench and screwdriver collection",
        "measuring tape and level precision instruments",
        "garden tool set shovel rake and pruner",
        "workbench organizer with drawer storage",
    ],
    "Toys": [
        "building blocks set creative construction play",
        "plush toy soft and cuddly stuffed animal",
        "remote control car fast and durable design",
        "educational toy stem learning for kids",
        "outdoor play set swing and slide combo",
    ],
}

item_categories = np.random.choice(categories, N_ITEMS, p=cat_probs)
item_price_tiers = np.random.choice(price_tiers, N_ITEMS, p=price_probs)

item_descriptions = []
for cat in item_categories:
    templates = description_templates[cat]
    desc = np.random.choice(templates)
    # Add some random variation
    extras = [
        "great value", "top rated", "best seller", "new arrival",
        "highly reviewed", "popular choice", "durable build",
        "easy to use", "lightweight design", "premium quality",
    ]
    n_extras = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
    if n_extras > 0:
        chosen = np.random.choice(extras, n_extras, replace=False)
        desc = desc + " " + " ".join(chosen)
    item_descriptions.append(desc)

items = pd.DataFrame({
    "item_id": range(1, N_ITEMS + 1),
    "category": item_categories,
    "price_tier": item_price_tiers,
    "description": item_descriptions,
    "avg_rating": np.round(np.random.uniform(2.5, 4.8, N_ITEMS), 2),
    "num_ratings": np.random.randint(5, 500, N_ITEMS),
})

# ---- Ratings ----
# Sparsity: 50K ratings out of 2000*500 = 1M possible = 5% filled, 95% sparse
# Create user-item pairs with realistic patterns

# Some users rate more than others (power-law-ish distribution)
user_activity = np.random.exponential(scale=1.0, size=N_USERS)
user_activity = user_activity / user_activity.sum()

# Some items are more popular
item_popularity = np.random.exponential(scale=1.0, size=N_ITEMS)
item_popularity = item_popularity / item_popularity.sum()

# Category preferences per user (latent factor simulation)
n_latent = len(categories)
user_prefs = np.random.randn(N_USERS, n_latent) * 0.5  # user latent factors
item_factors = np.zeros((N_ITEMS, n_latent))
for i, cat in enumerate(item_categories):
    cat_idx = categories.index(cat)
    item_factors[i, cat_idx] = 1.0
    # Add some noise
    item_factors[i] += np.random.randn(n_latent) * 0.1

ratings_list = []
seen_pairs = set()

while len(ratings_list) < TARGET_RATINGS:
    batch_size = min(TARGET_RATINGS - len(ratings_list), 10000)
    batch_users = np.random.choice(N_USERS, batch_size, p=user_activity)
    batch_items = np.random.choice(N_ITEMS, batch_size, p=item_popularity)

    for u, i in zip(batch_users, batch_items):
        pair = (u, i)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        # Base rating from latent factor dot product
        base_score = np.dot(user_prefs[u], item_factors[i])
        # Add item's average rating as anchor
        anchor = items.iloc[i]["avg_rating"]
        # Combine: anchor + latent signal + noise
        raw_rating = anchor + base_score * 0.4 + np.random.normal(0, 0.6)
        # Clamp to 1-5 and round
        rating = int(np.clip(np.round(raw_rating), 1, 5))

        ratings_list.append({
            "user_id": u + 1,
            "item_id": i + 1,
            "rating": rating,
        })

        if len(ratings_list) >= TARGET_RATINGS:
            break

ratings = pd.DataFrame(ratings_list)

# Shuffle
ratings = ratings.sample(frac=1, random_state=42).reset_index(drop=True)

# ---- Save ----
data_dir = os.path.dirname(os.path.abspath(__file__))
users.to_csv(os.path.join(data_dir, "users.csv"), index=False)
items.to_csv(os.path.join(data_dir, "items.csv"), index=False)
ratings.to_csv(os.path.join(data_dir, "ratings.csv"), index=False)

total_possible = N_USERS * N_ITEMS
sparsity = 1 - len(ratings) / total_possible

print(f"Saved {len(users)} users to data/users.csv")
print(f"Saved {len(items)} items to data/items.csv")
print(f"Saved {len(ratings)} ratings to data/ratings.csv")
print(f"\nSparsity: {sparsity:.2%}")
print(f"Avg ratings per user: {len(ratings) / N_USERS:.1f}")
print(f"Avg ratings per item: {len(ratings) / N_ITEMS:.1f}")
print(f"\nRating distribution:")
print(ratings["rating"].value_counts().sort_index().to_string())
print(f"\nCategory distribution (items):")
print(items["category"].value_counts().to_string())
print(f"\nSample ratings:")
print(ratings.head(10).to_string(index=False))
