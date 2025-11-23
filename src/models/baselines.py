"""Baseline recommendation models for comparison."""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Popularity-based recommender (most popular items).
    
    Recommends the most popular items to all users.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.item_popularity_: Optional[np.ndarray] = None
        self.logger = logging.getLogger(__name__)
    
    def fit(
        self, 
        interactions: Union[pd.DataFrame, np.ndarray, csr_matrix],
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: Optional[str] = None
    ) -> "PopularityRecommender":
        """Fit the popularity model."""
        if isinstance(interactions, pd.DataFrame):
            matrix, _, _ = self._df_to_matrix(interactions, user_col, item_col, rating_col)
        else:
            matrix = interactions
        
        if isinstance(matrix, csr_matrix):
            self.item_popularity_ = np.array(matrix.sum(axis=0)).flatten()
        else:
            self.item_popularity_ = matrix.sum(axis=0)
        
        return self
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict popularity score for item."""
        if self.item_popularity_ is None:
            raise ValueError("Model must be fitted before making predictions")
        return float(self.item_popularity_[item_idx])
    
    def recommend(
        self, 
        user_idx: int, 
        n_recommendations: int = 10,
        exclude_seen: bool = True,
        interactions: Optional[csr_matrix] = None
    ) -> List[Tuple[int, float]]:
        """Generate popularity-based recommendations."""
        if self.item_popularity_ is None:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get top popular items
        top_items = np.argsort(self.item_popularity_)[::-1][:n_recommendations]
        
        # Exclude seen items if requested
        if exclude_seen and interactions is not None:
            seen_items = set(interactions[user_idx].indices)
            top_items = [item for item in top_items if item not in seen_items][:n_recommendations]
        
        return [(int(item_idx), float(self.item_popularity_[item_idx])) for item_idx in top_items]


class UserKNNRecommender(BaseRecommender):
    """User-based collaborative filtering with k-nearest neighbors."""
    
    def __init__(self, k: int = 50, metric: str = "cosine") -> None:
        super().__init__()
        self.k = k
        self.metric = metric
        self.knn_model_: Optional[NearestNeighbors] = None
        self.interactions_: Optional[csr_matrix] = None
        self.logger = logging.getLogger(__name__)
    
    def fit(
        self, 
        interactions: Union[pd.DataFrame, np.ndarray, csr_matrix],
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: Optional[str] = None
    ) -> "UserKNNRecommender":
        """Fit the user-kNN model."""
        if isinstance(interactions, pd.DataFrame):
            matrix, _, _ = self._df_to_matrix(interactions, user_col, item_col, rating_col)
        else:
            matrix = interactions
        
        if not isinstance(matrix, csr_matrix):
            matrix = csr_matrix(matrix)
        
        self.interactions_ = matrix
        
        # Fit k-NN model
        self.knn_model_ = NearestNeighbors(
            n_neighbors=self.k + 1,  # +1 to exclude self
            metric=self.metric,
            algorithm="brute"
        )
        self.knn_model_.fit(matrix)
        
        return self
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating using user-kNN."""
        if self.knn_model_ is None or self.interactions_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Find similar users
        distances, indices = self.knn_model_.kneighbors(
            self.interactions_[user_idx], n_neighbors=self.k + 1
        )
        
        # Exclude self
        similar_users = indices[0][1:]
        similarities = 1 - distances[0][1:]  # Convert distance to similarity
        
        # Compute weighted average
        ratings = self.interactions_[similar_users, item_idx].toarray().flatten()
        mask = ratings > 0
        
        if not np.any(mask):
            return 0.0
        
        weighted_sum = np.sum(similarities[mask] * ratings[mask])
        similarity_sum = np.sum(similarities[mask])
        
        return float(weighted_sum / similarity_sum) if similarity_sum > 0 else 0.0
    
    def recommend(
        self, 
        user_idx: int, 
        n_recommendations: int = 10,
        exclude_seen: bool = True,
        interactions: Optional[csr_matrix] = None
    ) -> List[Tuple[int, float]]:
        """Generate user-kNN recommendations."""
        if self.knn_model_ is None or self.interactions_ is None:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Find similar users
        distances, indices = self.knn_model_.kneighbors(
            self.interactions_[user_idx], n_neighbors=self.k + 1
        )
        
        similar_users = indices[0][1:]
        similarities = 1 - distances[0][1:]
        
        # Compute scores for all items
        scores = np.zeros(self.interactions_.shape[1])
        
        for i, (user, sim) in enumerate(zip(similar_users, similarities)):
            user_ratings = self.interactions_[user].toarray().flatten()
            scores += sim * user_ratings
        
        # Normalize by similarity sum
        similarity_sum = np.sum(similarities)
        if similarity_sum > 0:
            scores /= similarity_sum
        
        # Exclude seen items if requested
        if exclude_seen:
            seen_items = self.interactions_[user_idx].indices
            scores[seen_items] = -np.inf
        
        # Get top recommendations
        top_items = np.argsort(scores)[::-1][:n_recommendations]
        
        return [(int(item_idx), float(scores[item_idx])) for item_idx in top_items]


class ItemKNNRecommender(BaseRecommender):
    """Item-based collaborative filtering with k-nearest neighbors."""
    
    def __init__(self, k: int = 50, metric: str = "cosine") -> None:
        super().__init__()
        self.k = k
        self.metric = metric
        self.knn_model_: Optional[NearestNeighbors] = None
        self.interactions_: Optional[csr_matrix] = None
        self.logger = logging.getLogger(__name__)
    
    def fit(
        self, 
        interactions: Union[pd.DataFrame, np.ndarray, csr_matrix],
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: Optional[str] = None
    ) -> "ItemKNNRecommender":
        """Fit the item-kNN model."""
        if isinstance(interactions, pd.DataFrame):
            matrix, _, _ = self._df_to_matrix(interactions, user_col, item_col, rating_col)
        else:
            matrix = interactions
        
        if not isinstance(matrix, csr_matrix):
            matrix = csr_matrix(matrix)
        
        self.interactions_ = matrix
        
        # Fit k-NN model on items (transpose the matrix)
        self.knn_model_ = NearestNeighbors(
            n_neighbors=self.k + 1,  # +1 to exclude self
            metric=self.metric,
            algorithm="brute"
        )
        self.knn_model_.fit(matrix.T)
        
        return self
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating using item-kNN."""
        if self.knn_model_ is None or self.interactions_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Find similar items
        distances, indices = self.knn_model_.kneighbors(
            self.interactions_[:, item_idx].T, n_neighbors=self.k + 1
        )
        
        # Exclude self
        similar_items = indices[0][1:]
        similarities = 1 - distances[0][1:]  # Convert distance to similarity
        
        # Compute weighted average
        ratings = self.interactions_[user_idx, similar_items].toarray().flatten()
        mask = ratings > 0
        
        if not np.any(mask):
            return 0.0
        
        weighted_sum = np.sum(similarities[mask] * ratings[mask])
        similarity_sum = np.sum(similarities[mask])
        
        return float(weighted_sum / similarity_sum) if similarity_sum > 0 else 0.0
    
    def recommend(
        self, 
        user_idx: int, 
        n_recommendations: int = 10,
        exclude_seen: bool = True,
        interactions: Optional[csr_matrix] = None
    ) -> List[Tuple[int, float]]:
        """Generate item-kNN recommendations."""
        if self.knn_model_ is None or self.interactions_ is None:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get user's rated items
        user_items = self.interactions_[user_idx].indices
        user_ratings = self.interactions_[user_idx, user_items].toarray().flatten()
        
        # Compute scores for all items
        scores = np.zeros(self.interactions_.shape[1])
        
        for item, rating in zip(user_items, user_ratings):
            # Find similar items
            distances, indices = self.knn_model_.kneighbors(
                self.interactions_[:, item].T, n_neighbors=self.k + 1
            )
            
            similar_items = indices[0][1:]
            similarities = 1 - distances[0][1:]
            
            # Add weighted contributions
            scores[similar_items] += similarities * rating
        
        # Exclude seen items if requested
        if exclude_seen:
            scores[user_items] = -np.inf
        
        # Get top recommendations
        top_items = np.argsort(scores)[::-1][:n_recommendations]
        
        return [(int(item_idx), float(scores[item_idx])) for item_idx in top_items]
