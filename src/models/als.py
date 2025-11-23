"""Modern Alternating Least Squares implementation for collaborative filtering."""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

from .base import BaseRecommender


class ALS(BaseRecommender):
    """Alternating Least Squares for collaborative filtering.
    
    Implements ALS algorithm for matrix factorization with implicit feedback.
    Optimized for sparse user-item interaction matrices.
    
    Args:
        n_factors: Number of latent factors.
        regularization: Regularization parameter.
        iterations: Number of iterations.
        alpha: Confidence weight for implicit feedback.
        random_state: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n_factors: int = 50,
        regularization: float = 0.01,
        iterations: int = 15,
        alpha: float = 40.0,
        random_state: Optional[int] = None
    ) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        
        self.user_factors_: Optional[np.ndarray] = None
        self.item_factors_: Optional[np.ndarray] = None
        self.user_ids_: Optional[List[str]] = None
        self.item_ids_: Optional[List[str]] = None
        
        self.logger = logging.getLogger(__name__)
    
    def _init_factors(self, n_users: int, n_items: int) -> None:
        """Initialize user and item factor matrices."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.user_factors_ = np.random.normal(
            0, 0.1, (n_users, self.n_factors)
        ).astype(np.float32)
        self.item_factors_ = np.random.normal(
            0, 0.1, (n_items, self.n_factors)
        ).astype(np.float32)
    
    def _update_user_factors(
        self, 
        interactions: csr_matrix, 
        confidence: csr_matrix
    ) -> None:
        """Update user factors using ALS."""
        n_users = interactions.shape[0]
        
        for u in range(n_users):
            # Get items interacted by user u
            user_items = interactions[u].indices
            if len(user_items) == 0:
                continue
            
            # Get confidence values for user u's items
            user_confidence = confidence[u, user_items].toarray().flatten()
            
            # Compute Cu - I
            Cu_minus_I = np.diag(user_confidence - 1)
            
            # Compute Cu * Y
            CuY = np.dot(Cu_minus_I + np.eye(len(user_items)), 
                        self.item_factors_[user_items])
            
            # Solve for Xu: (Yt * Cu * Y + λI)^-1 * Yt * Cu * p(u)
            YtCuY = np.dot(self.item_factors_[user_items].T, 
                          np.dot(Cu_minus_I + np.eye(len(user_items)), 
                                self.item_factors_[user_items]))
            YtCuY += self.regularization * np.eye(self.n_factors)
            
            YtCup = np.dot(self.item_factors_[user_items].T, 
                          np.dot(Cu_minus_I + np.eye(len(user_items)), 
                                interactions[u, user_items].toarray().flatten()))
            
            try:
                self.user_factors_[u] = np.linalg.solve(YtCuY, YtCup)
            except np.linalg.LinAlgError:
                # Fallback to least squares if matrix is singular
                self.user_factors_[u] = np.linalg.lstsq(YtCuY, YtCup, rcond=None)[0]
    
    def _update_item_factors(
        self, 
        interactions: csr_matrix, 
        confidence: csr_matrix
    ) -> None:
        """Update item factors using ALS."""
        n_items = interactions.shape[1]
        
        for i in range(n_items):
            # Get users who interacted with item i
            item_users = interactions[:, i].indices
            if len(item_users) == 0:
                continue
            
            # Get confidence values for item i's users
            item_confidence = confidence[item_users, i].toarray().flatten()
            
            # Compute Ci - I
            Ci_minus_I = np.diag(item_confidence - 1)
            
            # Compute Ci * X
            CiX = np.dot(Ci_minus_I + np.eye(len(item_users)), 
                        self.user_factors_[item_users])
            
            # Solve for Yi: (Xt * Ci * X + λI)^-1 * Xt * Ci * p(i)
            XtCiX = np.dot(self.user_factors_[item_users].T, 
                          np.dot(Ci_minus_I + np.eye(len(item_users)), 
                                self.user_factors_[item_users]))
            XtCiX += self.regularization * np.eye(self.n_factors)
            
            XtCip = np.dot(self.user_factors_[item_users].T, 
                          np.dot(Ci_minus_I + np.eye(len(item_users)), 
                                interactions[item_users, i].toarray().flatten()))
            
            try:
                self.item_factors_[i] = np.linalg.solve(XtCiX, XtCip)
            except np.linalg.LinAlgError:
                # Fallback to least squares if matrix is singular
                self.item_factors_[i] = np.linalg.lstsq(XtCiX, XtCip, rcond=None)[0]
    
    def fit(
        self, 
        interactions: Union[pd.DataFrame, np.ndarray, csr_matrix],
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: Optional[str] = None
    ) -> "ALS":
        """Fit the ALS model.
        
        Args:
            interactions: User-item interaction data.
            user_col: Name of user column (if DataFrame).
            item_col: Name of item column (if DataFrame).
            rating_col: Name of rating column (if DataFrame).
            
        Returns:
            Self.
        """
        # Convert to sparse matrix format
        if isinstance(interactions, pd.DataFrame):
            matrix, user_ids, item_ids = self._df_to_matrix(
                interactions, user_col, item_col, rating_col
            )
            self.user_ids_ = user_ids
            self.item_ids_ = item_ids
        else:
            matrix = interactions
            self.user_ids_ = list(range(matrix.shape[0]))
            self.item_ids_ = list(range(matrix.shape[1]))
        
        # Convert to sparse matrix
        if not isinstance(matrix, csr_matrix):
            matrix = csr_matrix(matrix)
        
        n_users, n_items = matrix.shape
        
        # Initialize factors
        self._init_factors(n_users, n_items)
        
        # Create confidence matrix (implicit feedback)
        confidence = matrix.copy()
        confidence.data = confidence.data * self.alpha + 1
        
        # ALS iterations
        for iteration in range(self.iterations):
            self._update_user_factors(matrix, confidence)
            self._update_item_factors(matrix, confidence)
            
            if iteration % 5 == 0:
                loss = self._compute_loss(matrix, confidence)
                self.logger.info(f"Iteration {iteration + 1}/{self.iterations}, Loss: {loss:.4f}")
        
        return self
    
    def _compute_loss(
        self, 
        interactions: csr_matrix, 
        confidence: csr_matrix
    ) -> float:
        """Compute reconstruction loss."""
        predictions = np.dot(self.user_factors_, self.item_factors_.T)
        
        # Compute weighted squared error
        error = interactions.toarray() - predictions
        weighted_error = confidence.toarray() * (error ** 2)
        
        # Add regularization
        reg_loss = (self.regularization * 
                   (np.sum(self.user_factors_ ** 2) + np.sum(self.item_factors_ ** 2)))
        
        return np.sum(weighted_error) + reg_loss
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for user-item pair.
        
        Args:
            user_idx: User index.
            item_idx: Item index.
            
        Returns:
            Predicted rating.
        """
        if self.user_factors_ is None or self.item_factors_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return float(np.dot(self.user_factors_[user_idx], self.item_factors_[item_idx]))
    
    def recommend(
        self, 
        user_idx: int, 
        n_recommendations: int = 10,
        exclude_seen: bool = True,
        interactions: Optional[csr_matrix] = None
    ) -> List[Tuple[int, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_idx: User index.
            n_recommendations: Number of recommendations to return.
            exclude_seen: Whether to exclude already seen items.
            interactions: Interaction matrix for excluding seen items.
            
        Returns:
            List of (item_idx, score) tuples.
        """
        if self.user_factors_ is None or self.item_factors_ is None:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Compute scores for all items
        scores = np.dot(self.user_factors_[user_idx], self.item_factors_.T)
        
        # Exclude seen items if requested
        if exclude_seen and interactions is not None:
            seen_items = interactions[user_idx].indices
            scores[seen_items] = -np.inf
        
        # Get top recommendations
        top_items = np.argsort(scores)[::-1][:n_recommendations]
        
        return [(int(item_idx), float(scores[item_idx])) for item_idx in top_items]
    
    def get_similar_items(
        self, 
        item_idx: int, 
        n_similar: int = 10
    ) -> List[Tuple[int, float]]:
        """Find items similar to the given item.
        
        Args:
            item_idx: Item index.
            n_similar: Number of similar items to return.
            
        Returns:
            List of (item_idx, similarity_score) tuples.
        """
        if self.item_factors_ is None:
            raise ValueError("Model must be fitted before finding similar items")
        
        # Compute cosine similarity with all items
        item_vector = self.item_factors_[item_idx]
        similarities = np.dot(self.item_factors_, item_vector) / (
            np.linalg.norm(self.item_factors_, axis=1) * np.linalg.norm(item_vector)
        )
        
        # Get top similar items (excluding self)
        top_items = np.argsort(similarities)[::-1][1:n_similar + 1]
        
        return [(int(item_idx), float(similarities[item_idx])) for item_idx in top_items]
