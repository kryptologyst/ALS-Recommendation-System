"""Base recommender class and common interfaces."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class BaseRecommender(ABC):
    """Base class for all recommendation models."""
    
    def _df_to_matrix(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Convert DataFrame to interaction matrix.
        
        Args:
            df: DataFrame with interactions.
            user_col: Name of user column.
            item_col: Name of item column.
            rating_col: Name of rating column.
            
        Returns:
            Tuple of (matrix, user_ids, item_ids).
        """
        if rating_col is None:
            # Binary implicit feedback
            matrix = df.pivot_table(
                index=user_col,
                columns=item_col,
                values=item_col,  # Use item_col as values for counting
                fill_value=0,
                aggfunc=lambda x: 1 if len(x) > 0 else 0
            )
        else:
            # Use actual ratings
            matrix = df.pivot_table(
                index=user_col,
                columns=item_col,
                values=rating_col,
                fill_value=0,
                aggfunc="mean"
            )
        
        return matrix.values, matrix.index.tolist(), matrix.columns.tolist()
    
    @abstractmethod
    def fit(
        self, 
        interactions: Union[pd.DataFrame, np.ndarray, csr_matrix],
        **kwargs: Any
    ) -> "BaseRecommender":
        """Fit the recommendation model.
        
        Args:
            interactions: User-item interaction data.
            **kwargs: Additional arguments.
            
        Returns:
            Self.
        """
        pass
    
    @abstractmethod
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for user-item pair.
        
        Args:
            user_idx: User index.
            item_idx: Item index.
            
        Returns:
            Predicted rating.
        """
        pass
    
    @abstractmethod
    def recommend(
        self, 
        user_idx: int, 
        n_recommendations: int = 10,
        **kwargs: Any
    ) -> List[Tuple[int, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_idx: User index.
            n_recommendations: Number of recommendations.
            **kwargs: Additional arguments.
            
        Returns:
            List of (item_idx, score) tuples.
        """
        pass
