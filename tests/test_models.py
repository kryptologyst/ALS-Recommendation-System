"""Unit tests for the ALS recommendation system."""

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.models.als import ALS
from src.models.baselines import (
    PopularityRecommender,
    UserKNNRecommender,
    ItemKNNRecommender,
)
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    map_at_k,
    hit_rate_at_k,
    coverage,
    diversity,
)
from src.utils import generate_synthetic_data, set_seed


class TestALS:
    """Test cases for ALS model."""
    
    def setup_method(self):
        """Setup test data."""
        set_seed(42)
        self.interactions = generate_synthetic_data(
            n_users=100,
            n_items=50,
            n_interactions=500,
            seed=42
        )
    
    def test_als_initialization(self):
        """Test ALS model initialization."""
        model = ALS(n_factors=10, regularization=0.01, iterations=5)
        assert model.n_factors == 10
        assert model.regularization == 0.01
        assert model.iterations == 5
    
    def test_als_fit(self):
        """Test ALS model fitting."""
        model = ALS(n_factors=10, regularization=0.01, iterations=5)
        model.fit(self.interactions)
        
        assert model.user_factors_ is not None
        assert model.item_factors_ is not None
        assert model.user_factors_.shape[1] == 10
        assert model.item_factors_.shape[1] == 10
    
    def test_als_predict(self):
        """Test ALS prediction."""
        model = ALS(n_factors=10, regularization=0.01, iterations=5)
        model.fit(self.interactions)
        
        prediction = model.predict(user_idx=0, item_idx=0)
        assert isinstance(prediction, float)
        assert not np.isnan(prediction)
    
    def test_als_recommend(self):
        """Test ALS recommendations."""
        model = ALS(n_factors=10, regularization=0.01, iterations=5)
        model.fit(self.interactions)
        
        recommendations = model.recommend(user_idx=0, n_recommendations=5)
        
        assert len(recommendations) <= 5
        assert all(isinstance(item_idx, int) for item_idx, _ in recommendations)
        assert all(isinstance(score, float) for _, score in recommendations)
    
    def test_als_similar_items(self):
        """Test ALS similar items."""
        model = ALS(n_factors=10, regularization=0.01, iterations=5)
        model.fit(self.interactions)
        
        similar_items = model.get_similar_items(item_idx=0, n_similar=5)
        
        assert len(similar_items) <= 5
        assert all(isinstance(item_idx, int) for item_idx, _ in similar_items)
        assert all(isinstance(similarity, float) for _, similarity in similar_items)


class TestBaselineModels:
    """Test cases for baseline models."""
    
    def setup_method(self):
        """Setup test data."""
        set_seed(42)
        self.interactions = generate_synthetic_data(
            n_users=100,
            n_items=50,
            n_interactions=500,
            seed=42
        )
    
    def test_popularity_recommender(self):
        """Test popularity recommender."""
        model = PopularityRecommender()
        model.fit(self.interactions)
        
        recommendations = model.recommend(user_idx=0, n_recommendations=5)
        assert len(recommendations) <= 5
        
        # Check that recommendations are sorted by popularity
        scores = [score for _, score in recommendations]
        assert scores == sorted(scores, reverse=True)
    
    def test_user_knn_recommender(self):
        """Test user-kNN recommender."""
        model = UserKNNRecommender(k=10)
        model.fit(self.interactions)
        
        recommendations = model.recommend(user_idx=0, n_recommendations=5)
        assert len(recommendations) <= 5
        
        prediction = model.predict(user_idx=0, item_idx=0)
        assert isinstance(prediction, float)
    
    def test_item_knn_recommender(self):
        """Test item-kNN recommender."""
        model = ItemKNNRecommender(k=10)
        model.fit(self.interactions)
        
        recommendations = model.recommend(user_idx=0, n_recommendations=5)
        assert len(recommendations) <= 5
        
        prediction = model.predict(user_idx=0, item_idx=0)
        assert isinstance(prediction, float)


class TestEvaluationMetrics:
    """Test cases for evaluation metrics."""
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        recommendations = [1, 2, 3, 4, 5]
        relevant_items = [1, 3, 5, 7, 9]
        
        precision = precision_at_k(recommendations, relevant_items, k=5)
        assert precision == 0.6  # 3 out of 5 recommendations are relevant
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        recommendations = [1, 2, 3, 4, 5]
        relevant_items = [1, 3, 5, 7, 9]
        
        recall = recall_at_k(recommendations, relevant_items, k=5)
        assert recall == 0.6  # 3 out of 5 relevant items are recommended
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        recommendations = [1, 2, 3, 4, 5]
        relevant_items = [1, 3, 5, 7, 9]
        
        ndcg = ndcg_at_k(recommendations, relevant_items, k=5)
        assert 0 <= ndcg <= 1
    
    def test_map_at_k(self):
        """Test MAP@k calculation."""
        recommendations = [1, 2, 3, 4, 5]
        relevant_items = [1, 3, 5, 7, 9]
        
        map_score = map_at_k(recommendations, relevant_items, k=5)
        assert 0 <= map_score <= 1
    
    def test_hit_rate_at_k(self):
        """Test hit rate@k calculation."""
        recommendations = [1, 2, 3, 4, 5]
        relevant_items = [1, 3, 5, 7, 9]
        
        hit_rate = hit_rate_at_k(recommendations, relevant_items, k=5)
        assert hit_rate == 1.0  # At least one relevant item is recommended
        
        # Test case with no hits
        recommendations_no_hit = [2, 4, 6, 8, 10]
        hit_rate_no_hit = hit_rate_at_k(recommendations_no_hit, relevant_items, k=5)
        assert hit_rate_no_hit == 0.0
    
    def test_coverage(self):
        """Test coverage calculation."""
        recommendations = [
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]
        ]
        all_items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        cov = coverage(recommendations, all_items)
        assert cov == 0.5  # 5 out of 10 items are recommended
    
    def test_diversity(self):
        """Test diversity calculation."""
        recommendations = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        div = diversity(recommendations)
        assert 0 <= div <= 1


class TestUtils:
    """Test cases for utility functions."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        data = generate_synthetic_data(
            n_users=100,
            n_items=50,
            n_interactions=500,
            seed=42
        )
        
        assert len(data) <= 500
        assert "user_id" in data.columns
        assert "item_id" in data.columns
        assert "rating" in data.columns
        assert "timestamp" in data.columns
        
        # Check data types
        assert data["user_id"].dtype == "object"
        assert data["item_id"].dtype == "object"
        assert data["rating"].dtype in ["int64", "float64"]
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        rand1 = np.random.random()
        
        set_seed(42)
        rand2 = np.random.random()
        
        assert rand1 == rand2


if __name__ == "__main__":
    pytest.main([__file__])
