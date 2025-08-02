# Hierarchical Ensemble Classifier

A scikit-learn compatible implementation of hierarchical ensemble classification where each sub-classifier can use unique features and classification algorithms.

![Beta](https://img.shields.io/badge/status-beta-orange)
![Not Production Ready](https://img.shields.io/badge/production-not%20ready-red)

 **Warning: This is a beta release and has not been thoroughly tested.**

## Overview

The Hierarchical Ensemble Classifier (HEC) is a machine learning method that decomposes complex multi-class classification problems into a hierarchy of simpler binary or multi-class problems. Each node in the hierarchy represents a sub-classifier that can:

- Use a different subset of features
- Employ a different classification algorithm  
- Have custom hyperparameters optimized for its specific task

This approach is particularly useful for:
- **Biological classification tasks** where natural hierarchies exist
- **Complex multi-class problems** that benefit from decomposition
- **Cases where different feature sets** are optimal for different classification decisions
- **Interpretable models** where the decision path can be visualized and understood

## Key Features

- **Scikit-learn compatible API** - Easy integration with existing ML pipelines
- **Flexible architecture** - Each sub-classifier can use different algorithms and features
- **Automatic hierarchy discovery** - Build hierarchies from data patterns using the `HierarchyBuilder` class
- **Comprehensive evaluation** - Cross-validation support for individual sub-classifiers
- **Rich visualization** - Plot hierarchical structure, prediction paths, and performance metrics
- **Integration with skclust** - Enhanced dendrogram visualization using your skclust package
- **Single module design** - All functionality in one importable module

## Installation

```bash
pip install hierarchical-ensemble-classifier
```

For enhanced visualization with skclust:
```bash
pip install hierarchical-ensemble-classifier[skclust]
```

For all optional dependencies:
```bash
pip install hierarchical-ensemble-classifier[all]
```

For development installation:
```bash
git clone https://github.com/jolespin/hierarchical-ensemble-classifier.git
cd hierarchical-ensemble-classifier
pip install -e ".[dev,viz,skclust]"
```

## Quick Start

Here's a simple example using the Iris dataset:

```python
from hierarchical_ensemble_classifier import (
    HierarchicalEnsembleClassifier,
    create_iris_example
)

# Run the iris example
results = create_iris_example()

print(f"Accuracy: {results['accuracy']:.3f}")
print("Cross-validation results:")
print(results['cv_results'])

# Visualize the hierarchy
results['model'].visualize_hierarchy()
```

## Manual Hierarchy Construction

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from hierarchical_ensemble_classifier import HierarchicalEnsembleClassifier

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target_names[iris.target])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Create hierarchical classifier
hec = HierarchicalEnsembleClassifier(name="IrisHEC", verbose=True)

# Add first level sub-classifier (sepal features)
hec.add_sub_classifier(
    name="sepal_classifier",
    estimator=LogisticRegression(random_state=42),
    features=["sepal length (cm)", "sepal width (cm)"]
)

# Add second level sub-classifier (petal features)
hec.add_sub_classifier(
    name="petal_classifier", 
    estimator=DecisionTreeClassifier(random_state=42),
    features=["petal length (cm)", "petal width (cm)"],
    parent="sepal_classifier"
)

# Add terminal classes
for class_name in y.unique():
    hec.add_terminal_class(class_name, parent="petal_classifier")

# Fit the model
hec.fit(X_train, y_train)

# Make predictions
y_pred = hec.predict(X_test)
y_proba = hec.predict_proba(X_test)

# Evaluate sub-classifiers
cv_results = hec.cross_validate_sub_classifiers(X_train, y_train, cv=5)
print(cv_results)

# Visualize the hierarchy
hec.visualize_hierarchy()
```

## Automatic Hierarchy Discovery

```python
from hierarchical_ensemble_classifier import (
    HierarchyBuilder,
    create_hierarchy_builder_example
)

# Run the automatic hierarchy example
results = create_hierarchy_builder_example()

print(f"Discovered paths: {results['paths']}")
print(f"Accuracy: {results['accuracy']:.3f}")

# Visualize the discovered hierarchy
results['hierarchy_builder'].visualize_dendrogram()
```

## Manual Hierarchy Building

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from hierarchical_ensemble_classifier import HierarchyBuilder

# Create synthetic dataset
X, y = make_classification(
    n_samples=500, n_features=20, n_classes=5, random_state=42
)

# Convert to pandas
import pandas as pd
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y = pd.Series([f'class_{i}' for i in y])

# Build hierarchy from data patterns
hierarchy_builder = HierarchyBuilder(linkage_method='ward', distance_metric='euclidean')
hierarchy_builder.build_from_class_profiles(X, y, n_components=10)

# Get hierarchical paths
paths = hierarchy_builder.get_paths(n_clusters=3)
print("Discovered hierarchical paths:")
for terminal_class, path in paths.items():
    print(f"{terminal_class}: {' -> '.join(path)}")

# Visualize the hierarchy (uses skclust if available)
hierarchy_builder.visualize_dendrogram(figsize=(12, 6))
```

## API Reference

### HierarchicalEnsembleClassifier

The main classifier class that implements the hierarchical ensemble approach.

**Key Methods:**
- `add_sub_classifier(name, estimator, features, parent=None)` - Add a sub-classifier to the hierarchy
- `add_terminal_class(class_name, parent)` - Add a terminal classification class
- `fit(X, y)` - Fit the hierarchical ensemble
- `predict(X)` - Make predictions
- `predict_proba(X)` - Get prediction probabilities
- `cross_validate_sub_classifiers(X, y, cv=5)` - Cross-validate individual sub-classifiers
- `visualize_hierarchy()` - Plot the hierarchical structure

### HierarchyBuilder

Utility class for discovering hierarchical structures from data.

**Key Methods:**
- `build_from_class_profiles(X, y, n_components=None)` - Build hierarchy from class profiles
- `get_paths(n_clusters=None)` - Extract hierarchical paths
- `get_target_matrix(y, paths)` - Create target matrix for training
- `visualize_dendrogram()` - Plot the clustering dendrogram (enhanced with skclust)

## Utility Functions

- `create_mixed_feature_pipeline(estimators, features)` - Create pipelines for different feature subsets
- `validate_hierarchy_structure(paths)` - Validate hierarchy structure
- `compute_path_probabilities(predictions, paths)` - Compute cumulative path probabilities
- `plot_hierarchy_with_performance(hec)` - Plot hierarchy with performance metrics
- `plot_feature_usage_heatmap(hec)` - Plot feature usage across sub-classifiers

## Integration with skclust

When `skclust` is installed, the `HierarchyBuilder` class automatically uses enhanced dendrogram visualization:

```python
from hierarchical_ensemble_classifier import HierarchyBuilder

# HierarchyBuilder will automatically use skclust's enhanced plotting
hierarchy_builder = HierarchyBuilder()
hierarchy_builder.build_from_class_profiles(X, y)

# This will use skclust's advanced dendrogram features if available
hierarchy_builder.visualize_dendrogram(
    figsize=(12, 8),
    show_clusters=True,
    show_tracks=True
)
```

## Examples

The package includes built-in example functions:

1. **`create_iris_example()`** - Simple hierarchy with the Iris dataset
2. **`create_hierarchy_builder_example()`** - Automatic hierarchy discovery with synthetic data

## Dependencies

**Required:**
- numpy >= 1.19.0
- pandas >= 1.2.0
- scikit-learn >= 0.24.0
- networkx >= 2.5
- scipy >= 1.6.0

**Optional:**
- matplotlib >= 3.3.0 (for visualization)
- seaborn >= 0.11.0 (for heatmaps)
- skclust >= 2025.7.26 (for enhanced dendrograms)

## Testing

Run the test suite:

```bash
pytest tests/
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## Citation

If you use this package in your research, please cite:

**Original Soothsayer implementation:**
```bibtex
@article{espinoza2021predicting,
  title={Predicting antimicrobial mechanism-of-action from transcriptomes: A generalizable explainable artificial intelligence approach},
  author={Espinoza, Josh L and Dupont, Chris L and O'Rourke, Aubrie and Beyhan, Seherzada and Morales, Paula and others},
  journal={PLOS Computational Biology},
  volume={17},
  number={3},
  pages={e1008857},
  year={2021},
  publisher={Public Library of Science San Francisco, CA USA},
  doi={10.1371/journal.pcbi.1008857},
  url={https://doi.org/10.1371/journal.pcbi.1008857}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [https://hierarchical-ensemble-classifier.readthedocs.io](https://hierarchical-ensemble-classifier.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/jolespin/hierarchical-ensemble-classifier/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jolespin/hierarchical-ensemble-classifier/discussions)
