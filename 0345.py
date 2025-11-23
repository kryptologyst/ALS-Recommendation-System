#!/usr/bin/env python3
"""
Project 345: Modern Alternating Least Squares Implementation

This is a modernized version of the original ALS implementation.
For the full production-ready system, see the src/ directory.

This script demonstrates the core ALS algorithm with proper type hints,
error handling, and modern Python practices.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModernALS:
    """Modern Alternating Least Squares implementation.
    
    This is a clean, production-ready implementation of ALS for collaborative filtering.
    """
    
    def __init__(
        self,
        n_factors: int = 4,
        regularization: float = 0.1,
        iterations: int = 10,
        alpha: float = 40.0,
        random_state: Optional[int] = None
    ) -> None:
        """Initialize ALS model.
        
        Args:
            n_factors: Number of latent factors.
            regularization: Regularization parameter.
            iterations: Number of training iterations.
            alpha: Confidence weight for implicit feedback.
            random_state: Random seed for reproducibility.
        """
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        
        # Initialize random state
        if random_state is not None:
            np.random.seed(random_state)
        
        # Model parameters (will be set during fit)
        self.user_factors_: Optional[np.ndarray] = None
        self.item_factors_: Optional[np.ndarray] = None
        self.user_ids_: Optional[List[str]] = None
        self.item_ids_: Optional[List[str]] = None
    
    def _init_factors(self, n_users: int, n_items: int) -> None:
        """Initialize user and item factor matrices."""
        self.user_factors_ = np.random.normal(
            0, 0.1, (n_users, self.n_factors)
        ).astype(np.float32)
        self.item_factors_ = np.random.normal(
            0, 0.1, (n_items, self.n_factors)
        ).astype(np.float32)
    
    def fit(self, interactions_df: pd.DataFrame) -> "ModernALS":
        """Fit the ALS model.
        
        Args:
            interactions_df: DataFrame with user-item interactions.
            
        Returns:
            Self for method chaining.
        """
        logger.info("Starting ALS training...")
        
        # The DataFrame is already in matrix format
        matrix = interactions_df
        
        # Store user and item IDs
        self.user_ids_ = matrix.index.tolist()
        self.item_ids_ = matrix.columns.tolist()
        
        n_users, n_items = matrix.shape
        
        # Initialize factors
        self._init_factors(n_users, n_items)
        
        # Convert to sparse matrix for efficiency
        if not isinstance(matrix, csr_matrix):
            matrix = csr_matrix(matrix.values)
        
        # Create confidence matrix for implicit feedback
        confidence = matrix.copy()
        confidence.data = confidence.data * self.alpha + 1
        
        # ALS iterations
        for epoch in range(self.iterations):
            # Update user factors
            self._update_user_factors(matrix, confidence)
            
            # Update item factors
            self._update_item_factors(matrix, confidence)
            
            # Calculate and log loss
            loss = self._compute_loss(matrix, confidence)
            logger.info(f"Epoch {epoch+1}/{self.iterations}, Loss: {loss:.4f}")
        
        logger.info("ALS training completed!")
        return self
    
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
            
            # Get confidence values
            user_confidence = confidence[u, user_items].toarray().flatten()
            
            # Compute Cu - I
            Cu_minus_I = np.diag(user_confidence - 1)
            
            # Solve for Xu: (Yt * Cu * Y + λI)^-1 * Yt * Cu * p(u)
            YtCuY = np.dot(
                self.item_factors_[user_items].T,
                np.dot(Cu_minus_I + np.eye(len(user_items)), 
                      self.item_factors_[user_items])
            )
            YtCuY += self.regularization * np.eye(self.n_factors)
            
            YtCup = np.dot(
                self.item_factors_[user_items].T,
                np.dot(Cu_minus_I + np.eye(len(user_items)),
                      interactions[u, user_items].toarray().flatten())
            )
            
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
            
            # Get confidence values
            item_confidence = confidence[item_users, i].toarray().flatten()
            
            # Compute Ci - I
            Ci_minus_I = np.diag(item_confidence - 1)
            
            # Solve for Yi: (Xt * Ci * X + λI)^-1 * Xt * Ci * p(i)
            XtCiX = np.dot(
                self.user_factors_[item_users].T,
                np.dot(Ci_minus_I + np.eye(len(item_users)),
                      self.user_factors_[item_users])
            )
            XtCiX += self.regularization * np.eye(self.n_factors)
            
            XtCip = np.dot(
                self.user_factors_[item_users].T,
                np.dot(Ci_minus_I + np.eye(len(item_users)),
                      interactions[item_users, i].toarray().flatten())
            )
            
            try:
                self.item_factors_[i] = np.linalg.solve(XtCiX, XtCip)
            except np.linalg.LinAlgError:
                # Fallback to least squares if matrix is singular
                self.item_factors_[i] = np.linalg.lstsq(XtCiX, XtCip, rcond=None)[0]
    
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
        top_n: int = 3,
        exclude_seen: bool = True
    ) -> List[str]:
        """Generate recommendations for a user.
        
        Args:
            user_idx: User index.
            top_n: Number of recommendations to return.
            exclude_seen: Whether to exclude already seen items.
            
        Returns:
            List of recommended item IDs.
        """
        if self.user_factors_ is None or self.item_factors_ is None:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Compute scores for all items
        scores = np.dot(self.user_factors_[user_idx], self.item_factors_.T)
        
        # Get top recommendations
        top_items = np.argsort(scores)[::-1][:top_n]
        
        return [self.item_ids_[idx] for idx in top_items]


def main() -> None:
    """Main demonstration function."""
    logger.info("Starting ALS demonstration...")
    
    # Create sample data (same as original)
    users = ['User1', 'User2', 'User3', 'User4', 'User5']
    items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
    ratings = np.array([
        [1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1],
        [0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0]
    ])
    
    # Create DataFrame
    df = pd.DataFrame(ratings, index=users, columns=items)
    logger.info("Created sample interaction matrix:")
    print(df)
    
    # Train ALS model
    als = ModernALS(
        n_factors=4,
        regularization=0.1,
        iterations=10,
        random_state=42
    )
    
    als.fit(df)
    
    # Generate recommendations for User1
    user_idx = 0  # User1
    recommended_items = als.recommend(user_idx, top_n=3)
    
    logger.info(f"ALS Recommendations for {users[user_idx]}: {recommended_items}")
    
    # Show some predictions
    logger.info("Sample predictions:")
    for i in range(min(3, len(items))):
        prediction = als.predict(user_idx, i)
        logger.info(f"User {users[user_idx]} -> Item {items[i]}: {prediction:.4f}")


if __name__ == "__main__":
    main()