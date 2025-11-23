"""Main training script for ALS recommendation system."""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data import DataLoader, DataSplitter, NegativeSampler
from src.evaluation.metrics import evaluate_model
from src.models.als import ALS
from src.models.baselines import (
    ItemKNNRecommender,
    PopularityRecommender,
    UserKNNRecommender,
)
from src.utils import generate_synthetic_data, set_seed


def setup_logging(verbose: bool = True) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_data(config: DictConfig) -> pd.DataFrame:
    """Load or generate data."""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    loader = DataLoader(data_dir)
    
    try:
        # Try to load existing data
        interactions = loader.load_interactions()
        logging.info("Loaded existing interaction data")
    except FileNotFoundError:
        # Generate synthetic data
        logging.info("Generating synthetic data...")
        interactions = generate_synthetic_data(
            n_users=config.data.synthetic.n_users,
            n_items=config.data.synthetic.n_items,
            n_interactions=config.data.synthetic.n_interactions,
            sparsity=config.data.synthetic.sparsity,
            seed=config.data.synthetic.seed,
        )
        
        # Save synthetic data
        interactions.to_csv(data_dir / "interactions.csv", index=False)
        logging.info(f"Saved synthetic data to {data_dir / 'interactions.csv'}")
    
    return interactions


def train_models(
    train_data: pd.DataFrame, config: DictConfig
) -> Dict[str, object]:
    """Train all models."""
    models = {}
    
    # Initialize models
    models["popularity"] = PopularityRecommender()
    models["user_knn"] = UserKNNRecommender(
        k=config.models.user_knn.k,
        metric=config.models.user_knn.metric,
    )
    models["item_knn"] = ItemKNNRecommender(
        k=config.models.item_knn.k,
        metric=config.models.item_knn.metric,
    )
    models["als"] = ALS(
        n_factors=config.models.als.n_factors,
        regularization=config.models.als.regularization,
        iterations=config.models.als.iterations,
        alpha=config.models.als.alpha,
        random_state=config.models.als.random_state,
    )
    
    # Train models
    for name, model in models.items():
        logging.info(f"Training {name} model...")
        model.fit(train_data)
        logging.info(f"Finished training {name} model")
    
    return models


def evaluate_models(
    models: Dict[str, object], test_data: pd.DataFrame, config: DictConfig
) -> Dict[str, Dict[str, float]]:
    """Evaluate all models."""
    results = {}
    
    for name, model in models.items():
        logging.info(f"Evaluating {name} model...")
        results[name] = evaluate_model(
            model, test_data, k_values=config.evaluation.k_values
        )
        logging.info(f"Finished evaluating {name} model")
    
    return results


def print_results(results: Dict[str, Dict[str, float]]) -> None:
    """Print evaluation results in a formatted table."""
    # Get all metric names
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    
    all_metrics = sorted(all_metrics)
    
    # Print header
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    # Print table header
    header = f"{'Model':<15}"
    for metric in all_metrics:
        header += f"{metric:<12}"
    print(header)
    print("-" * len(header))
    
    # Print results for each model
    for model_name, model_results in results.items():
        row = f"{model_name:<15}"
        for metric in all_metrics:
            value = model_results.get(metric, 0.0)
            row += f"{value:<12.4f}"
        print(row)
    
    print("=" * 80)


def main() -> None:
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train ALS recommendation system")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Config file path"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Set random seed
    set_seed(config.training.random_state)
    
    # Load data
    interactions = load_data(config)
    logging.info(f"Loaded {len(interactions)} interactions")
    
    # Split data
    splitter = DataSplitter(random_state=config.training.random_state)
    
    if config.data.split_method == "chronological":
        train_data, val_data, test_data = splitter.split_chronological(
            interactions,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
        )
    else:
        train_data, val_data, test_data = splitter.split_random(
            interactions,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
        )
    
    # Train models
    models = train_models(train_data, config)
    
    # Evaluate models
    results = evaluate_models(models, test_data, config)
    
    # Print results
    print_results(results)
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv("results.csv")
    logging.info("Saved results to results.csv")


if __name__ == "__main__":
    main()
