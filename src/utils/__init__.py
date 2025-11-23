"""Core utilities for the ALS recommendation system."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Loaded configuration object.
    """
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save.
        config_path: Path where to save the configuration.
    """
    OmegaConf.save(config, config_path)


def create_interaction_matrix(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: Optional[str] = None,
    binary: bool = True
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Create user-item interaction matrix from DataFrame.
    
    Args:
        interactions: DataFrame with user-item interactions.
        user_col: Name of the user column.
        item_col: Name of the item column.
        rating_col: Name of the rating column (optional for binary).
        binary: Whether to create binary matrix (1/0) or use ratings.
        
    Returns:
        Tuple of (interaction_matrix, user_ids, item_ids).
    """
    if binary:
        # Create binary matrix (implicit feedback)
        matrix = interactions.pivot_table(
            index=user_col, 
            columns=item_col, 
            values=rating_col or "rating",
            fill_value=0,
            aggfunc=lambda x: 1 if len(x) > 0 else 0
        )
    else:
        # Use actual ratings
        matrix = interactions.pivot_table(
            index=user_col,
            columns=item_col,
            values=rating_col,
            fill_value=0,
            aggfunc="mean"
        )
    
    return matrix.values, matrix.index.tolist(), matrix.columns.tolist()


def split_data_chronological(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    timestamp_col: str = "timestamp",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically for each user.
    
    Args:
        interactions: DataFrame with interactions.
        user_col: Name of the user column.
        timestamp_col: Name of the timestamp column.
        train_ratio: Ratio of data for training.
        val_ratio: Ratio of data for validation.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
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
    
    return (
        pd.concat(train_data, ignore_index=True),
        pd.concat(val_data, ignore_index=True),
        pd.concat(test_data, ignore_index=True)
    )


def generate_synthetic_data(
    n_users: int = 1000,
    n_items: int = 500,
    n_interactions: int = 10000,
    sparsity: float = 0.95,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic interaction data for testing.
    
    Args:
        n_users: Number of users.
        n_items: Number of items.
        n_interactions: Number of interactions to generate.
        sparsity: Desired sparsity level.
        seed: Random seed.
        
    Returns:
        DataFrame with synthetic interactions.
    """
    set_seed(seed)
    
    # Generate user-item pairs
    user_ids = np.random.randint(0, n_users, n_interactions)
    item_ids = np.random.randint(0, n_items, n_interactions)
    
    # Add some structure (popularity bias, user preferences)
    item_popularity = np.random.beta(2, 5, n_items)  # Some items more popular
    user_activity = np.random.beta(2, 5, n_users)    # Some users more active
    
    # Adjust probabilities based on popularity and activity
    item_probs = item_popularity[item_ids]
    user_probs = user_activity[user_ids]
    interaction_probs = item_probs * user_probs
    
    # Sample interactions based on probabilities
    interaction_mask = np.random.random(n_interactions) < interaction_probs
    
    # Create DataFrame
    interactions = pd.DataFrame({
        "user_id": user_ids[interaction_mask],
        "item_id": item_ids[interaction_mask],
        "rating": 1,  # Binary implicit feedback
        "timestamp": np.random.randint(0, 1000000, np.sum(interaction_mask))
    })
    
    # Remove duplicates
    interactions = interactions.drop_duplicates(subset=["user_id", "item_id"])
    
    return interactions
