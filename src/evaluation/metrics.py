"""Evaluation metrics for recommendation systems."""

from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix


def precision_at_k(
    recommendations: List[int],
    relevant_items: List[int],
    k: int
) -> float:
    """Calculate Precision@K.
    
    Args:
        recommendations: List of recommended item indices.
        relevant_items: List of relevant item indices.
        k: Number of top recommendations to consider.
        
    Returns:
        Precision@K score.
    """
    if k == 0:
        return 0.0
    
    top_k_recs = recommendations[:k]
    relevant_set = set(relevant_items)
    
    if len(top_k_recs) == 0:
        return 0.0
    
    relevant_recommendations = sum(1 for item in top_k_recs if item in relevant_set)
    return relevant_recommendations / len(top_k_recs)


def recall_at_k(
    recommendations: List[int],
    relevant_items: List[int],
    k: int
) -> float:
    """Calculate Recall@K.
    
    Args:
        recommendations: List of recommended item indices.
        relevant_items: List of relevant item indices.
        k: Number of top recommendations to consider.
        
    Returns:
        Recall@K score.
    """
    if len(relevant_items) == 0:
        return 0.0
    
    top_k_recs = recommendations[:k]
    relevant_set = set(relevant_items)
    
    if len(top_k_recs) == 0:
        return 0.0
    
    relevant_recommendations = sum(1 for item in top_k_recs if item in relevant_set)
    return relevant_recommendations / len(relevant_items)


def ndcg_at_k(
    recommendations: List[int],
    relevant_items: List[int],
    k: int
) -> float:
    """Calculate NDCG@K.
    
    Args:
        recommendations: List of recommended item indices.
        relevant_items: List of relevant item indices.
        k: Number of top recommendations to consider.
        
    Returns:
        NDCG@K score.
    """
    if k == 0:
        return 0.0
    
    top_k_recs = recommendations[:k]
    relevant_set = set(relevant_items)
    
    if len(top_k_recs) == 0:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(top_k_recs):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
    
    # Calculate IDCG (ideal DCG)
    idcg = 0.0
    for i in range(min(len(relevant_items), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(
    recommendations: List[int],
    relevant_items: List[int],
    k: int
) -> float:
    """Calculate MAP@K.
    
    Args:
        recommendations: List of recommended item indices.
        relevant_items: List of relevant item indices.
        k: Number of top recommendations to consider.
        
    Returns:
        MAP@K score.
    """
    if len(relevant_items) == 0 or k == 0:
        return 0.0
    
    top_k_recs = recommendations[:k]
    relevant_set = set(relevant_items)
    
    if len(top_k_recs) == 0:
        return 0.0
    
    # Calculate average precision
    precision_sum = 0.0
    relevant_count = 0
    
    for i, item in enumerate(top_k_recs):
        if item in relevant_set:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    
    return precision_sum / len(relevant_items)


def hit_rate_at_k(
    recommendations: List[int],
    relevant_items: List[int],
    k: int
) -> float:
    """Calculate Hit Rate@K.
    
    Args:
        recommendations: List of recommended item indices.
        relevant_items: List of relevant item indices.
        k: Number of top recommendations to consider.
        
    Returns:
        Hit Rate@K score (0 or 1).
    """
    if k == 0:
        return 0.0
    
    top_k_recs = recommendations[:k]
    relevant_set = set(relevant_items)
    
    return 1.0 if any(item in relevant_set for item in top_k_recs) else 0.0


def coverage(
    recommendations: List[List[int]],
    all_items: List[int]
) -> float:
    """Calculate catalog coverage.
    
    Args:
        recommendations: List of recommendation lists for each user.
        all_items: List of all available items.
        
    Returns:
        Coverage score.
    """
    if not recommendations or not all_items:
        return 0.0
    
    recommended_items = set()
    for user_recs in recommendations:
        recommended_items.update(user_recs)
    
    return len(recommended_items) / len(all_items)


def diversity(
    recommendations: List[List[int]],
    item_features: Optional[np.ndarray] = None
) -> float:
    """Calculate recommendation diversity.
    
    Args:
        recommendations: List of recommendation lists for each user.
        item_features: Item feature matrix for computing diversity.
        
    Returns:
        Average pairwise diversity score.
    """
    if not recommendations:
        return 0.0
    
    if item_features is None:
        # Use simple Jaccard diversity
        diversities = []
        for user_recs in recommendations:
            if len(user_recs) < 2:
                diversities.append(0.0)
                continue
            
            # Calculate average pairwise Jaccard distance
            jaccard_sum = 0.0
            pairs = 0
            
            for i in range(len(user_recs)):
                for j in range(i + 1, len(user_recs)):
                    set_i = set([user_recs[i]])
                    set_j = set([user_recs[j]])
                    jaccard = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
                    jaccard_sum += 1 - jaccard  # Convert to distance
                    pairs += 1
            
            diversities.append(jaccard_sum / pairs if pairs > 0 else 0.0)
        
        return np.mean(diversities)
    
    else:
        # Use feature-based diversity
        diversities = []
        for user_recs in recommendations:
            if len(user_recs) < 2:
                diversities.append(0.0)
                continue
            
            # Calculate average pairwise cosine distance
            cosine_sum = 0.0
            pairs = 0
            
            for i in range(len(user_recs)):
                for j in range(i + 1, len(user_recs)):
                    feat_i = item_features[user_recs[i]]
                    feat_j = item_features[user_recs[j]]
                    
                    # Cosine similarity
                    dot_product = np.dot(feat_i, feat_j)
                    norm_i = np.linalg.norm(feat_i)
                    norm_j = np.linalg.norm(feat_j)
                    
                    if norm_i > 0 and norm_j > 0:
                        cosine_sim = dot_product / (norm_i * norm_j)
                        cosine_sum += 1 - cosine_sim  # Convert to distance
                        pairs += 1
            
            diversities.append(cosine_sum / pairs if pairs > 0 else 0.0)
        
        return np.mean(diversities)


def evaluate_model(
    model,
    test_interactions: Union[pd.DataFrame, csr_matrix],
    k_values: List[int] = [5, 10, 20],
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: Optional[str] = None
) -> dict:
    """Evaluate a recommendation model on test data.
    
    Args:
        model: Trained recommendation model.
        test_interactions: Test interaction data.
        k_values: List of k values for evaluation.
        user_col: Name of user column.
        item_col: Name of item column.
        rating_col: Name of rating column.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    # Convert test data to appropriate format
    if isinstance(test_interactions, pd.DataFrame):
        test_matrix, user_ids, item_ids = model._df_to_matrix(
            test_interactions, user_col, item_col, rating_col
        )
    else:
        test_matrix = test_interactions
        user_ids = list(range(test_matrix.shape[0]))
        item_ids = list(range(test_matrix.shape[1]))
    
    if not isinstance(test_matrix, csr_matrix):
        test_matrix = csr_matrix(test_matrix)
    
    results = {}
    
    # Calculate metrics for each k
    for k in k_values:
        precisions = []
        recalls = []
        ndcgs = []
        maps = []
        hit_rates = []
        
        for user_idx in range(test_matrix.shape[0]):
            # Get user's relevant items in test set
            relevant_items = test_matrix[user_idx].indices.tolist()
            
            if len(relevant_items) == 0:
                continue
            
            # Get recommendations
            recommendations = model.recommend(user_idx, n_recommendations=k)
            rec_items = [item_idx for item_idx, _ in recommendations]
            
            # Calculate metrics
            precisions.append(precision_at_k(rec_items, relevant_items, k))
            recalls.append(recall_at_k(rec_items, relevant_items, k))
            ndcgs.append(ndcg_at_k(rec_items, relevant_items, k))
            maps.append(map_at_k(rec_items, relevant_items, k))
            hit_rates.append(hit_rate_at_k(rec_items, relevant_items, k))
        
        results[f"precision@{k}"] = np.mean(precisions)
        results[f"recall@{k}"] = np.mean(recalls)
        results[f"ndcg@{k}"] = np.mean(ndcgs)
        results[f"map@{k}"] = np.mean(maps)
        results[f"hit_rate@{k}"] = np.mean(hit_rates)
    
    return results
