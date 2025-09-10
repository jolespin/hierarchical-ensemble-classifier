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

## Installation
```bash
pip install hierarchical-ensemble-classifier
```

## Example Usage

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from hierarchical_ensemble_classifier import (
    HierarchyBuilder, 
    HierarchicalEnsembleClassifier,
)

# 1. Load and prepare Iris data using pandas
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target_names[iris.target], name='species').astype(str)
X.columns = X.columns.str.replace(' (cm)', '').str.replace(' ', '_')
```

### Building a Hierarchy

```python
# Automatically build a hierarchy from class eigenprofiles
hb = HierarchyBuilder()
hb.fit(X, y)

# Inspect the generated hierarchy
hb.hierarchy_.edges() # .hierarchy_ is a networkx DiGraph and .Y_ is a pd.DataFrame of subclassifier classes
# OutEdgeView([('y1', 'setosa'), ('y1', 'y2'), ('y2', 'versicolor'), ('y2', 'virginica')])
```

### Training Hierarchical Classifiers

```python
# Specify an estimator for each internal node in the hierarchy
estimators = {
    'y1': {
            'classifier': DecisionTreeClassifier(random_state=42),
            'features': ['sepal_length', 'sepal_width'],
            },
    'y2': {
            'classifier': LogisticRegression(random_state=42),
            'features': ['petal_length', 'petal_width']},
    }

clf = HierarchicalEnsembleClassifier(hierarchy=hb.hierarchy_, estimators=estimators)
clf.fit(X, y)
```

### Making Predictions
```python
y_hat = clf.predict(X)
```

### Evaluating Performance
```python
performance = cross_val_score(clf, X, y, cv=5)
print(f"Cross-validated accuracy: {performance.mean():.4f} ± {performance.std():.4f}")
# Cross-validated accuracy: 0.9200 ± 0.0686
```

### Visualizing the Hierarchy

```python
clf.plot_model()
```

### Visualizing Prediction Path

```python
sample = X.iloc[[0]]
clf.plot_prediction_paths(sample)
```


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

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/jolespin/hierarchical-ensemble-classifier/issues)
