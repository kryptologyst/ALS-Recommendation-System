# ALS Recommendation System

A production-ready implementation of Alternating Least Squares (ALS) for collaborative filtering recommendation systems. This project provides a comprehensive framework for building, training, and evaluating recommendation models with proper evaluation metrics and an interactive demo.

## Features

- **Modern ALS Implementation**: Optimized ALS algorithm with implicit feedback support
- **Multiple Baseline Models**: Popularity, User-kNN, Item-kNN for comparison
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate, Coverage, Diversity
- **Interactive Demo**: Streamlit-based web interface for exploring recommendations
- **Production Ready**: Type hints, proper logging, configuration management, and testing
- **Synthetic Data Generation**: Built-in data generation for testing and demonstration

## Project Structure

```
├── src/
│   ├── models/           # Recommendation models
│   │   ├── als.py        # ALS implementation
│   │   ├── baselines.py  # Baseline models
│   │   └── base.py       # Base recommender class
│   ├── data/             # Data loading and preprocessing
│   ├── evaluation/       # Evaluation metrics
│   └── utils/            # Utility functions
├── configs/              # Configuration files
├── scripts/              # Training and demo scripts
├── data/                 # Data directory
├── tests/                # Unit tests
└── notebooks/            # Jupyter notebooks
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/ALS-Recommendation-System.git
cd ALS-Recommendation-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install with optional dependencies:
```bash
pip install -e ".[dev,tracking,ann]"
```

### Training Models

Train all models with default configuration:
```bash
python scripts/train.py
```

Train with custom configuration:
```bash
python scripts/train.py --config configs/config.yaml --verbose
```

### Interactive Demo

Launch the Streamlit demo:
```bash
streamlit run scripts/demo.py
```

The demo will be available at `http://localhost:8501`

## Usage

### Basic Usage

```python
import pandas as pd
from src.models.als import ALS
from src.evaluation.metrics import evaluate_model

# Load your data
interactions = pd.read_csv("data/interactions.csv")

# Initialize and train ALS model
model = ALS(n_factors=50, regularization=0.01, iterations=15)
model.fit(interactions)

# Generate recommendations
recommendations = model.recommend(user_idx=0, n_recommendations=10)

# Evaluate model
results = evaluate_model(model, test_interactions, k_values=[5, 10, 20])
print(results)
```

### Data Format

The system expects interaction data in the following format:

**interactions.csv**:
```csv
user_id,item_id,rating,timestamp
user1,item1,1,1234567890
user1,item2,1,1234567891
user2,item1,1,1234567892
...
```

**Optional files**:
- `items.csv`: Item metadata (item_id, title, category, etc.)
- `users.csv`: User metadata (user_id, age, gender, etc.)

### Configuration

Modify `configs/config.yaml` to customize:

- Model hyperparameters
- Data splitting ratios
- Evaluation metrics
- Demo settings

## Models

### ALS (Alternating Least Squares)
- **Type**: Matrix factorization for collaborative filtering
- **Best for**: Implicit feedback, sparse data
- **Parameters**: 
  - `n_factors`: Number of latent factors (default: 50)
  - `regularization`: L2 regularization (default: 0.01)
  - `iterations`: Training iterations (default: 15)
  - `alpha`: Confidence weight for implicit feedback (default: 40.0)

### Baseline Models

1. **Popularity Recommender**: Recommends most popular items
2. **User-kNN**: User-based collaborative filtering
3. **Item-kNN**: Item-based collaborative filtering

## Evaluation Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation
- **Coverage**: Fraction of catalog items that are recommended
- **Diversity**: Average pairwise dissimilarity of recommendations

## Advanced Features

### Synthetic Data Generation

Generate realistic synthetic data for testing:
```python
from src.utils import generate_synthetic_data

interactions = generate_synthetic_data(
    n_users=1000,
    n_items=500,
    n_interactions=10000,
    sparsity=0.95
)
```

### Data Splitting

Chronological splitting (recommended for time-sensitive data):
```python
from src.data import DataSplitter

splitter = DataSplitter()
train, val, test = splitter.split_chronological(
    interactions, 
    train_ratio=0.7, 
    val_ratio=0.15, 
    test_ratio=0.15
)
```

### Negative Sampling

Generate negative samples for implicit feedback:
```python
from src.data import NegativeSampler

sampler = NegativeSampler()
negatives = sampler.sample_negatives(interactions, n_negatives=1)
```

## Development

### Code Quality

The project uses modern Python development practices:

- **Type hints**: Full type annotation support
- **Code formatting**: Black + Ruff for consistent formatting
- **Linting**: Comprehensive linting with Ruff
- **Testing**: pytest for unit tests

Run code quality checks:
```bash
black src/ scripts/
ruff check src/ scripts/
mypy src/
```

### Testing

Run tests:
```bash
pytest tests/
```

### Adding New Models

1. Inherit from `BaseRecommender`
2. Implement required methods: `fit()`, `predict()`, `recommend()`
3. Add to model registry in training script

Example:
```python
from src.models.base import BaseRecommender

class MyModel(BaseRecommender):
    def fit(self, interactions, **kwargs):
        # Training logic
        return self
    
    def predict(self, user_idx, item_idx):
        # Prediction logic
        return score
    
    def recommend(self, user_idx, n_recommendations=10, **kwargs):
        # Recommendation logic
        return recommendations
```

## Performance Tips

1. **Use sparse matrices**: The implementation is optimized for sparse data
2. **Tune hyperparameters**: Adjust `n_factors`, `regularization`, and `alpha`
3. **Chronological splitting**: Use time-aware splits for better evaluation
4. **Batch processing**: For large datasets, consider batch processing

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `n_factors` or use smaller datasets
2. **Slow training**: Reduce `iterations` or use fewer factors
3. **Poor performance**: Check data quality and try different hyperparameters

### Getting Help

- Check the logs for detailed error messages
- Verify data format matches expected schema
- Ensure all dependencies are installed correctly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{als_recommendation_system,
  title={ALS Recommendation System},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/ALS-Recommendation-System}
}
```
# ALS-Recommendation-System
