"""Interactive demo for ALS recommendation system."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from omegaconf import DictConfig, OmegaConf

from src.data import DataLoader
from src.models.als import ALS
from src.models.baselines import (
    ItemKNNRecommender,
    PopularityRecommender,
    UserKNNRecommender,
)
from src.utils import generate_synthetic_data, set_seed


def load_data_and_models() -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load data and trained models."""
    # Load configuration
    config = OmegaConf.load("configs/config.yaml")
    
    # Set random seed
    set_seed(config.training.random_state)
    
    # Load or generate data
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    loader = DataLoader(data_dir)
    
    try:
        interactions = loader.load_interactions()
        st.success("Loaded existing interaction data")
    except FileNotFoundError:
        st.info("Generating synthetic data...")
        interactions = generate_synthetic_data(
            n_users=config.data.synthetic.n_users,
            n_items=config.data.synthetic.n_items,
            n_interactions=config.data.synthetic.n_interactions,
            sparsity=config.data.synthetic.sparsity,
            seed=config.data.synthetic.seed,
        )
        interactions.to_csv(data_dir / "interactions.csv", index=False)
        st.success("Generated and saved synthetic data")
    
    # Initialize and train models
    models = {}
    
    with st.spinner("Training models..."):
        # Popularity model
        models["popularity"] = PopularityRecommender()
        models["popularity"].fit(interactions)
        
        # User-kNN model
        models["user_knn"] = UserKNNRecommender(
            k=config.models.user_knn.k,
            metric=config.models.user_knn.metric,
        )
        models["user_knn"].fit(interactions)
        
        # Item-kNN model
        models["item_knn"] = ItemKNNRecommender(
            k=config.models.item_knn.k,
            metric=config.models.item_knn.metric,
        )
        models["item_knn"].fit(interactions)
        
        # ALS model
        models["als"] = ALS(
            n_factors=config.models.als.n_factors,
            regularization=config.models.als.regularization,
            iterations=config.models.als.iterations,
            alpha=config.models.als.alpha,
            random_state=config.models.als.random_state,
        )
        models["als"].fit(interactions)
    
    st.success("All models trained successfully!")
    
    return interactions, models


def get_user_history(interactions: pd.DataFrame, user_id: str) -> List[str]:
    """Get user's interaction history."""
    user_interactions = interactions[interactions["user_id"] == user_id]
    return user_interactions["item_id"].tolist()


def get_item_info(interactions: pd.DataFrame, item_id: str) -> Dict:
    """Get item information."""
    item_interactions = interactions[interactions["item_id"] == item_id]
    
    return {
        "total_interactions": len(item_interactions),
        "unique_users": item_interactions["user_id"].nunique(),
        "avg_rating": item_interactions["rating"].mean() if "rating" in item_interactions.columns else 1.0,
    }


def main() -> None:
    """Main demo application."""
    st.set_page_config(
        page_title="ALS Recommendation System Demo",
        page_icon="🎯",
        layout="wide",
    )
    
    st.title("🎯 ALS Recommendation System Demo")
    st.markdown("Interactive demo for Alternating Least Squares collaborative filtering")
    
    # Load data and models
    interactions, models = load_data_and_models()
    
    # Get unique users and items
    unique_users = sorted(interactions["user_id"].unique())
    unique_items = sorted(interactions["item_id"].unique())
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model",
        list(models.keys()),
        index=3,  # Default to ALS
    )
    
    # User selection
    selected_user = st.sidebar.selectbox(
        "Select User",
        unique_users,
        index=0,
    )
    
    # Number of recommendations
    n_recommendations = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=50,
        value=10,
    )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("User Recommendations")
        
        # Get user's history
        user_history = get_user_history(interactions, selected_user)
        
        st.subheader(f"User {selected_user}")
        st.write(f"**Interaction History:** {len(user_history)} items")
        
        if user_history:
            st.write("**Recently interacted items:**")
            for item in user_history[-5:]:  # Show last 5 items
                st.write(f"- {item}")
        
        # Get recommendations
        model = models[selected_model]
        
        # Find user index
        user_idx = unique_users.index(selected_user)
        
        # Generate recommendations
        recommendations = model.recommend(
            user_idx, 
            n_recommendations=n_recommendations,
            exclude_seen=True,
        )
        
        st.subheader(f"Top {n_recommendations} Recommendations ({selected_model})")
        
        for i, (item_idx, score) in enumerate(recommendations, 1):
            item_id = unique_items[item_idx]
            item_info = get_item_info(interactions, item_id)
            
            with st.expander(f"{i}. {item_id} (Score: {score:.4f})"):
                st.write(f"**Total Interactions:** {item_info['total_interactions']}")
                st.write(f"**Unique Users:** {item_info['unique_users']}")
                st.write(f"**Average Rating:** {item_info['avg_rating']:.2f}")
    
    with col2:
        st.header("Item Similarity")
        
        # Item selection for similarity
        selected_item = st.selectbox(
            "Select Item for Similarity",
            unique_items,
            index=0,
        )
        
        # Get item info
        item_info = get_item_info(interactions, selected_item)
        
        st.subheader(f"Item {selected_item}")
        st.write(f"**Total Interactions:** {item_info['total_interactions']}")
        st.write(f"**Unique Users:** {item_info['unique_users']}")
        st.write(f"**Average Rating:** {item_info['avg_rating']:.2f}")
        
        # Get similar items (if model supports it)
        if hasattr(model, "get_similar_items"):
            item_idx = unique_items.index(selected_item)
            similar_items = model.get_similar_items(item_idx, n_similar=5)
            
            st.subheader("Similar Items")
            
            for i, (similar_item_idx, similarity) in enumerate(similar_items, 1):
                similar_item_id = unique_items[similar_item_idx]
                similar_item_info = get_item_info(interactions, similar_item_id)
                
                with st.expander(f"{i}. {similar_item_id} (Similarity: {similarity:.4f})"):
                    st.write(f"**Total Interactions:** {similar_item_info['total_interactions']}")
                    st.write(f"**Unique Users:** {similar_item_info['unique_users']}")
                    st.write(f"**Average Rating:** {similar_item_info['avg_rating']:.2f}")
        else:
            st.info("Similarity analysis not available for this model")
    
    # Model comparison
    st.header("Model Comparison")
    
    # Compare recommendations across models
    comparison_data = []
    
    for model_name, model in models.items():
        recommendations = model.recommend(
            user_idx,
            n_recommendations=5,
            exclude_seen=True,
        )
        
        for rank, (item_idx, score) in enumerate(recommendations, 1):
            item_id = unique_items[item_idx]
            comparison_data.append({
                "Model": model_name,
                "Rank": rank,
                "Item": item_id,
                "Score": score,
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Pivot table for comparison
    pivot_df = comparison_df.pivot_table(
        index="Rank",
        columns="Model",
        values="Item",
        aggfunc="first",
    )
    
    st.dataframe(pivot_df)
    
    # Dataset statistics
    st.header("Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", len(unique_users))
    
    with col2:
        st.metric("Total Items", len(unique_items))
    
    with col3:
        st.metric("Total Interactions", len(interactions))
    
    with col4:
        sparsity = 1 - (len(interactions) / (len(unique_users) * len(unique_items)))
        st.metric("Sparsity", f"{sparsity:.2%}")


if __name__ == "__main__":
    main()
