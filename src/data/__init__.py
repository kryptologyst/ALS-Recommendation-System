"""Data loading and preprocessing utilities."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.utils import set_seed


class DataLoader:
    """Data loader for recommendation system datasets."""
    
    def __init__(self, data_dir: Union[str, Path]) -> None:
        """Initialize data loader.
        
        Args:
            data_dir: Directory containing data files.
        """
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
    
    def load_interactions(
        self,
        filename: str = "interactions.csv",
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating",
        timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """Load interaction data from CSV file.
        
        Args:
            filename: Name of the CSV file.
            user_col: Name of user column.
            item_col: Name of item column.
            rating_col: Name of rating column.
            timestamp_col: Name of timestamp column.
            
        Returns:
            DataFrame with interactions.
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_cols = [user_col, item_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert user and item IDs to strings for consistency
        df[user_col] = df[user_col].astype(str)
        df[item_col] = df[item_col].astype(str)
        
        # Add timestamp if not present
        if timestamp_col not in df.columns:
            df[timestamp_col] = range(len(df))
        
        # Add rating if not present (for implicit feedback)
        if rating_col not in df.columns:
            df[rating_col] = 1
        
        self.logger.info(f"Loaded {len(df)} interactions from {filepath}")
        return df
    
    def load_items(
        self,
        filename: str = "items.csv",
        item_col: str = "item_id"
    ) -> pd.DataFrame:
        """Load item metadata from CSV file.
        
        Args:
            filename: Name of the CSV file.
            item_col: Name of item ID column.
            
        Returns:
            DataFrame with item metadata.
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            self.logger.warning(f"Items file not found: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        
        if item_col in df.columns:
            df[item_col] = df[item_col].astype(str)
        
        self.logger.info(f"Loaded {len(df)} items from {filepath}")
        return df
    
    def load_users(
        self,
        filename: str = "users.csv",
        user_col: str = "user_id"
    ) -> pd.DataFrame:
        """Load user metadata from CSV file.
        
        Args:
            filename: Name of the CSV file.
            user_col: Name of user ID column.
            
        Returns:
            DataFrame with user metadata.
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            self.logger.warning(f"Users file not found: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        
        if user_col in df.columns:
            df[user_col] = df[user_col].astype(str)
        
        self.logger.info(f"Loaded {len(df)} users from {filepath}")
        return df


class DataSplitter:
    """Data splitting utilities for recommendation systems."""
    
    def __init__(self, random_state: int = 42) -> None:
        """Initialize data splitter.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        set_seed(random_state)
        self.logger = logging.getLogger(__name__)
    
    def split_chronological(
        self,
        interactions: pd.DataFrame,
        user_col: str = "user_id",
        timestamp_col: str = "timestamp",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data chronologically for each user.
        
        Args:
            interactions: DataFrame with interactions.
            user_col: Name of user column.
            timestamp_col: Name of timestamp column.
            train_ratio: Ratio of data for training.
            val_ratio: Ratio of data for validation.
            test_ratio: Ratio of data for testing.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        train_data = []
        val_data = []
        test_data = []
        
        for user_id in interactions[user_col].unique():
            user_interactions = interactions[
                interactions[user_col] == user_id
            ].sort_values(timestamp_col)
            
            n_interactions = len(user_interactions)
            train_end = int(n_interactions * train_ratio)
            val_end = int(n_interactions * (train_ratio + val_ratio))
            
            train_data.append(user_interactions.iloc[:train_end])
            val_data.append(user_interactions.iloc[train_end:val_end])
            test_data.append(user_interactions.iloc[val_end:])
        
        train_df = pd.concat(train_data, ignore_index=True)
        val_df = pd.concat(val_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        
        self.logger.info(f"Split data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        
        return train_df, val_df, test_df
    
    def split_random(
        self,
        interactions: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data randomly.
        
        Args:
            interactions: DataFrame with interactions.
            train_ratio: Ratio of data for training.
            val_ratio: Ratio of data for validation.
            test_ratio: Ratio of data for testing.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # Shuffle data
        shuffled = interactions.sample(frac=1, random_state=self.random_state)
        
        n_total = len(shuffled)
        train_end = int(n_total * train_ratio)
        val_end = int(n_total * (train_ratio + val_ratio))
        
        train_df = shuffled.iloc[:train_end]
        val_df = shuffled.iloc[train_end:val_end]
        test_df = shuffled.iloc[val_end:]
        
        self.logger.info(f"Split data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        
        return train_df, val_df, test_df


class NegativeSampler:
    """Negative sampling for implicit feedback recommendation."""
    
    def __init__(self, random_state: int = 42) -> None:
        """Initialize negative sampler.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        set_seed(random_state)
        self.logger = logging.getLogger(__name__)
    
    def sample_negatives(
        self,
        interactions: pd.DataFrame,
        n_negatives: int = 1,
        user_col: str = "user_id",
        item_col: str = "item_id"
    ) -> pd.DataFrame:
        """Sample negative interactions.
        
        Args:
            interactions: DataFrame with positive interactions.
            n_negatives: Number of negative samples per positive.
            user_col: Name of user column.
            item_col: Name of item column.
            
        Returns:
            DataFrame with negative interactions.
        """
        # Get all unique users and items
        all_users = set(interactions[user_col].unique())
        all_items = set(interactions[item_col].unique())
        
        # Create user-item interaction set
        user_item_pairs = set(
            zip(interactions[user_col], interactions[item_col])
        )
        
        negative_data = []
        
        for user in all_users:
            # Get items this user has interacted with
            user_items = set(
                interactions[interactions[user_col] == user][item_col]
            )
            
            # Sample negative items
            available_items = all_items - user_items
            n_samples = min(n_negatives * len(user_items), len(available_items))
            
            if n_samples > 0:
                negative_items = np.random.choice(
                    list(available_items), 
                    size=n_samples, 
                    replace=False
                )
                
                for item in negative_items:
                    negative_data.append({
                        user_col: user,
                        item_col: item,
                        "rating": 0,
                        "timestamp": np.random.randint(0, 1000000)
                    })
        
        negative_df = pd.DataFrame(negative_data)
        self.logger.info(f"Sampled {len(negative_df)} negative interactions")
        
        return negative_df
