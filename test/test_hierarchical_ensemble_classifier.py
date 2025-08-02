"""
Comprehensive test suite for hierarchical ensemble classifier
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from hierarchical_ensemble_classifier import (
    HierarchicalEnsembleClassifier,
    HierarchyBuilder,
    SubClassifier,
    create_mixed_feature_pipeline,
    validate_hierarchy_structure,
    compute_path_probabilities,
    create_iris_example,
    create_hierarchy_builder_example
)


class TestSubClassifier:
    """Test SubClassifier functionality."""
    
    def test_init(self):
        """Test SubClassifier initialization."""
        estimator = LogisticRegression(random_state=42)
        features = ['feature_0', 'feature_1']
        
        sub_clf = SubClassifier(
            name='test_clf',
            estimator=estimator, 
            features=features
        )
        
        assert sub_clf.name == 'test_clf'
        assert sub_clf.estimator == estimator
        assert sub_clf.features == features
        assert not sub_clf.is_fitted_
        
    def test_fit_predict(self):
        """Test fitting and prediction."""
        # Create simple dataset
        X = pd.DataFrame({
            'feature_0': [1, 2, 3, 4],
            'feature_1': [2, 3, 4, 5],
            'feature_2': [0, 1, 0, 1]
        })
        y = pd.Series([0, 0, 1, 1])
        
        sub_clf = SubClassifier(
            name='test_clf',
            estimator=LogisticRegression(random_state=42),
            features=['feature_0', 'feature_1']
        )
        
        # Fit
        sub_clf.fit(X, y)
        assert sub_clf.is_fitted_
        
        # Predict
        predictions = sub_clf.predict(X)
        assert len(predictions) == len(y)
        
        # Predict proba
        probabilities = sub_clf.predict_proba(X)
        assert probabilities.shape == (len(y), 2)


class TestHierarchicalEnsembleClassifier:
    """Test HierarchicalEnsembleClassifier functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=3,
            n_informative=8,
            n_redundant=2,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series([f'class_{i}' for i in y])
        
        return train_test_split(X, y, test_size=0.3, random_state=42)
        
    def test_init(self):
        """Test HierarchicalEnsembleClassifier initialization."""
        hec = HierarchicalEnsembleClassifier(name='test_hec')
        assert hec.name == 'test_hec'
        assert hec.copy_X is True
        assert hec.verbose is False
        
    def test_add_sub_classifier(self, sample_data):
        """Test adding sub-classifiers."""
        X_train, X_test, y_train, y_test = sample_data
        
        hec = HierarchicalEnsembleClassifier()
        
        # Add first sub-classifier
        hec.add_sub_classifier(
            name='clf1',
            estimator=LogisticRegression(random_state=42),
            features=['feature_0', 'feature_1', 'feature_2']
        )
        
        assert 'clf1' in hec.sub_classifiers_
        assert hec.graph_.has_node('clf1')
        assert hec.graph_.has_edge('input', 'clf1')
        
        # Add second sub-classifier with parent
        hec.add_sub_classifier(
            name='clf2',
            estimator=DecisionTreeClassifier(random_state=42),
            features=['feature_3', 'feature_4', 'feature_5'],
            parent='clf1'
        )
        
        assert 'clf2' in hec.sub_classifiers_
        assert hec.graph_.has_edge('clf1', 'clf2')
        
    def test_add_terminal_class(self):
        """Test adding terminal classes."""
        hec = HierarchicalEnsembleClassifier()
        
        # Add sub-classifier first
        hec.add_sub_classifier(
            name='clf1',
            estimator=LogisticRegression(random_state=42),
            features=['feature_0', 'feature_1']
        )
        
        # Add terminal class
        hec.add_terminal_class('class_A', parent='clf1')
        
        assert hec.graph_.has_node('class_A')
        assert hec.graph_.has_edge('clf1', 'class_A')
        assert hec.graph_.nodes['class_A']['node_type'] == 'terminal'
        
    def test_fit_simple_hierarchy(self, sample_data):
        """Test fitting a simple hierarchy."""
        X_train, X_test, y_train, y_test = sample_data
        
        hec = HierarchicalEnsembleClassifier(verbose=True)
        
        # Build simple hierarchy
        hec.add_sub_classifier(
            name='level1',
            estimator=LogisticRegression(random_state=42),
            features=['feature_0', 'feature_1', 'feature_2', 'feature_3']
        )
        
        # Add terminal classes
        for class_name in y_train.unique():
            hec.add_terminal_class(class_name, parent='level1')
            
        # Fit
        hec.fit(X_train, y_train)
        
        # Check that sub-classifier was fitted
        assert hec.sub_classifiers_['level1'].is_fitted_
        assert hasattr(hec, 'classes_')
        assert len(hec.classes_) == len(y_train.unique())
        
    def test_predict(self, sample_data):
        """Test prediction functionality."""
        X_train, X_test, y_train, y_test = sample_data
        
        hec = HierarchicalEnsembleClassifier()
        
        # Build and fit hierarchy
        hec.add_sub_classifier(
            name='classifier',
            estimator=LogisticRegression(random_state=42),
            features=['feature_0', 'feature_1', 'feature_2', 'feature_3']
        )
        
        for class_name in y_train.unique():
            hec.add_terminal_class(class_name, parent='classifier')
            
        hec.fit(X_train, y_train)
        
        # Make predictions
        predictions = hec.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert all(pred in y_train.unique() for pred in predictions)
        
    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X_train, X_test, y_train, y_test = sample_data
        
        hec = HierarchicalEnsembleClassifier()
        
        # Build and fit hierarchy  
        hec.add_sub_classifier(
            name='classifier',
            estimator=LogisticRegression(random_state=42),
            features=['feature_0', 'feature_1', 'feature_2']
        )
        
        for class_name in y_train.unique():
            hec.add_terminal_class(class_name, parent='classifier')
            
        hec.fit(X_train, y_train)
        
        # Get probabilities
        probabilities = hec.predict_proba(X_test)
        
        assert probabilities.shape == (len(y_test), len(y_train.unique()))
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
        
    def test_cross_validate_sub_classifiers(self, sample_data):
        """Test cross-validation of sub-classifiers."""
        X_train, X_test, y_train, y_test = sample_data
        
        hec = HierarchicalEnsembleClassifier()
        
        # Build hierarchy
        hec.add_sub_classifier(
            name='clf1',
            estimator=LogisticRegression(random_state=42),
            features=['feature_0', 'feature_1', 'feature_2']
        )
        
        hec.add_sub_classifier(
            name='clf2', 
            estimator=DecisionTreeClassifier(random_state=42),
            features=['feature_3', 'feature_4', 'feature_5'],
            parent='clf1'
        )
        
        for class_name in y_train.unique():
            hec.add_terminal_class(class_name, parent='clf2')
            
        hec.fit(X_train, y_train)
        
        # Cross-validate
        cv_results = hec.cross_validate_sub_classifiers(X_train, y_train, cv=3)
        
        assert isinstance(cv_results, pd.DataFrame)
        assert 'mean_score' in cv_results.columns
        assert 'std_score' in cv_results.columns
        assert len(cv_results) <= len(hec.sub_classifiers_)

    def test_get_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = sample_data
        
        hec = HierarchicalEnsembleClassifier()
        
        # Build hierarchy with tree-based classifier
        hec.add_sub_classifier(
            name='tree_clf',
            estimator=DecisionTreeClassifier(random_state=42),
            features=['feature_0', 'feature_1', 'feature_2']
        )
        
        for class_name in y_train.unique():
            hec.add_terminal_class(class_name, parent='tree_clf')
            
        hec.fit(X_train, y_train)
        
        # Get feature importance
        importance_df = hec.get_feature_importance()
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'sub_classifier' in importance_df.columns
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns


class TestHierarchyBuilder:
    """Test HierarchyBuilder functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        X, y = make_classification(
            n_samples=200,
            n_features=15,
            n_classes=5,
            n_informative=10,
            n_redundant=5,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        class_names = [f'class_{i}' for i in range(5)]
        y = pd.Series([class_names[i] for i in y])
        
        return X, y
        
    def test_init(self):
        """Test HierarchyBuilder initialization."""
        hierarchy_builder = HierarchyBuilder(linkage_method='ward', distance_metric='euclidean')
        assert hierarchy_builder.linkage_method == 'ward'
        assert hierarchy_builder.distance_metric == 'euclidean'
        assert hierarchy_builder.linkage_matrix_ is None
        
    def test_build_from_class_profiles(self, sample_data):
        """Test building hierarchy from class profiles."""
        X, y = sample_data
        
        hierarchy_builder = HierarchyBuilder()
        hierarchy_builder.build_from_class_profiles(X, y, n_components=5)
        
        assert hierarchy_builder.linkage_matrix_ is not None
        assert hierarchy_builder.labels_ is not None
        assert len(hierarchy_builder.labels_) == len(y.unique())
        
    def test_get_paths(self, sample_data):
        """Test extracting hierarchical paths."""
        X, y = sample_data
        
        hierarchy_builder = HierarchyBuilder()
        hierarchy_builder.build_from_class_profiles(X, y, n_components=5)
        
        paths = hierarchy_builder.get_paths(n_clusters=3)
        
        assert isinstance(paths, dict)
        assert len(paths) == len(y.unique())
        
        # Check that all paths start with 'input'
        for path in paths.values():
            assert path[0] == 'input'
            assert len(path) >= 2
            
    def test_get_target_matrix(self, sample_data):
        """Test creating target matrix."""
        X, y = sample_data
        
        hierarchy_builder = HierarchyBuilder()
        hierarchy_builder.build_from_class_profiles(X, y, n_components=5)
        paths = hierarchy_builder.get_paths(n_clusters=3)
        
        Y_matrix = hierarchy_builder.get_target_matrix(y, paths)
        
        assert isinstance(Y_matrix, pd.DataFrame)
        assert Y_matrix.index.equals(y.index)
        
        # Check that each class has appropriate entries
        for class_name in y.unique():
            mask = (y == class_name)
            class_entries = Y_matrix.loc[mask]
            
            # Should have at least some non-NaN entries
            assert not class_entries.isna().all().all()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_mixed_feature_pipeline(self):
        """Test creating mixed feature pipelines."""
        estimators = {
            "sepal": LogisticRegression(random_state=42),
            "petal": DecisionTreeClassifier(random_state=42)
        }
        
        features = {
            "sepal": ["sepal_length", "sepal_width"],
            "petal": ["petal_length", "petal_width"]
        }
        
        pipelines = create_mixed_feature_pipeline(estimators, features)
        
        assert len(pipelines) == 2
        assert pipelines[0][0] == "sepal"
        assert pipelines[1][0] == "petal"
        
        # Check that pipelines are properly constructed
        for name, pipeline in pipelines:
            assert hasattr(pipeline, 'fit')
            assert hasattr(pipeline, 'predict')
            
    def test_validate_hierarchy_structure_valid(self):
        """Test validation of valid hierarchy structure."""
        paths = {
            "class_A": ["input", "level1", "class_A"],
            "class_B": ["input", "level1", "class_B"],
            "class_C": ["input", "level2", "class_C"]
        }
        
        # Should not raise exception
        assert validate_hierarchy_structure(paths) is True
        
    def test_validate_hierarchy_structure_invalid_empty(self):
        """Test validation with empty paths."""
        with pytest.raises(ValueError, match="Paths dictionary cannot be empty"):
            validate_hierarchy_structure({})
            
    def test_validate_hierarchy_structure_invalid_roots(self):
        """Test validation with multiple roots."""
        paths = {
            "class_A": ["input", "level1", "class_A"],
            "class_B": ["root2", "level1", "class_B"]
        }
        
        with pytest.raises(ValueError, match="All paths must start with the same root"):
            validate_hierarchy_structure(paths)
            
    def test_validate_hierarchy_structure_invalid_short_path(self):
        """Test validation with too short paths."""
        paths = {
            "class_A": ["input"],
            "class_B": ["input", "class_B"]
        }
        
        with pytest.raises(ValueError, match="Path for class 'class_A' must have at least 2 nodes"):
            validate_hierarchy_structure(paths)

    def test_compute_path_probabilities(self):
        """Test computing path probabilities."""
        # Create mock prediction data
        predictions = pd.DataFrame({
            ('clf1', 'class_A'): [0.7, 0.3],
            ('clf1', 'class_B'): [0.2, 0.5],
            ('clf1', 'class_C'): [0.1, 0.2],
            ('clf2', 'class_A'): [0.8, 0.4],
            ('clf2', 'class_B'): [0.2, 0.6]
        })
        predictions.columns = pd.MultiIndex.from_tuples(predictions.columns)
        
        paths = {
            'class_A': ['input', 'clf1', 'clf2', 'class_A'],
            'class_B': ['input', 'clf1', 'clf2', 'class_B'],
            'class_C': ['input', 'clf1', 'class_C']
        }
        
        result = compute_path_probabilities(predictions, paths)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == predictions.shape[0]
        assert len(result.columns) == len(paths)


class TestExampleFunctions:
    """Test example functions."""
    
    def test_create_iris_example(self):
        """Test iris example function."""
        results = create_iris_example()
        
        assert 'model' in results
        assert 'accuracy' in results
        assert 'cv_results' in results
        assert 'predictions' in results
        
        # Check model is fitted
        assert hasattr(results['model'], 'classes_')
        assert len(results['model'].sub_classifiers_) > 0
        
        # Check accuracy is reasonable
        assert 0.0 <= results['accuracy'] <= 1.0
        
    def test_create_hierarchy_builder_example(self):
        """Test hierarchy builder example function."""
        results = create_hierarchy_builder_example()
        
        assert 'hierarchy_builder' in results
        assert 'paths' in results
        assert 'model' in results
        assert 'accuracy' in results
        
        # Check hierarchy builder is fitted
        assert results['hierarchy_builder'].linkage_matrix_ is not None
        assert results['hierarchy_builder'].labels_ is not None
        
        # Check paths are valid
        paths = results['paths']
        assert len(paths) > 0
        for path in paths.values():
            assert len(path) >= 2
            assert path[0] == 'input'


class TestIntegrationTests:
    """Integration tests for complete workflows."""
    
    def test_iris_workflow(self):
        """Test complete workflow with Iris dataset."""
        # Load iris data
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target_names[iris.target])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Build hierarchy using HierarchyBuilder
        hierarchy_builder = HierarchyBuilder()
        hierarchy_builder.build_from_class_profiles(X_train, y_train)
        paths = hierarchy_builder.get_paths(n_clusters=2)
        
        # Create HEC based on discovered hierarchy
        hec = HierarchicalEnsembleClassifier(name="IrisIntegration")
        
        # Add sub-classifiers based on paths
        intermediate_nodes = set()
        for path in paths.values():
            intermediate_nodes.update(path[1:-1])
            
        for i, node in enumerate(sorted(intermediate_nodes)):
            # Use different feature subsets
            if i == 0:
                features = ["sepal length (cm)", "sepal width (cm)"]
                estimator = LogisticRegression(random_state=42)
            else:
                features = ["petal length (cm)", "petal width (cm)"]
                estimator = DecisionTreeClassifier(random_state=42)
                
            hec.add_sub_classifier(
                name=node,
                estimator=estimator,
                features=features
            )
            
        # Add terminal classes
        for terminal_class in paths.keys():
            path = paths[terminal_class]
            parent = path[-2]
            hec.add_terminal_class(terminal_class, parent=parent)
            
        # Fit and evaluate
        hec.fit(X_train, y_train)
        predictions = hec.predict(X_test)
        probabilities = hec.predict_proba(X_test)
        
        # Basic checks
        assert len(predictions) == len(y_test)
        assert probabilities.shape == (len(y_test), len(y_train.unique()))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        
        # Cross-validate
        cv_results = hec.cross_validate_sub_classifiers(X_train, y_train)
        assert len(cv_results) > 0
        
    def test_synthetic_multiclass_workflow(self):
        """Test workflow with synthetic multi-class data."""
        # Create synthetic dataset
        X, y = make_classification(
            n_samples=300,
            n_features=20,
            n_informative=15,
            n_classes=6,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Convert to pandas
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        class_names = [f'class_{i}' for i in range(6)]
        y = pd.Series([class_names[i] for i in y])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Build hierarchy
        hierarchy_builder = HierarchyBuilder(linkage_method='ward')
        hierarchy_builder.build_from_class_profiles(X_train, y_train, n_components=10)
        paths = hierarchy_builder.get_paths(n_clusters=3)
        
        # Create and fit HEC
        hec = HierarchicalEnsembleClassifier(name="SyntheticTest", verbose=False)
        
        # Add sub-classifiers
        intermediate_nodes = set()
        for path in paths.values():
            intermediate_nodes.update(path[1:-1])
            
        for i, node in enumerate(sorted(intermediate_nodes)):
            start_idx = i * 5
            end_idx = min(start_idx + 8, len(feature_names))
            features = feature_names[start_idx:end_idx]
            
            # Use different algorithms
            if i % 2 == 0:
                estimator = LogisticRegression(random_state=42, max_iter=1000)
            else:
                estimator = RandomForestClassifier(n_estimators=10, random_state=42)
                
            hec.add_sub_classifier(
                name=node,
                estimator=estimator,
                features=features
            )
            
        # Add terminal classes
        for terminal_class in paths.keys():
            path = paths[terminal_class]
            parent = path[-2]
            hec.add_terminal_class(terminal_class, parent=parent)
            
        # Fit and test
        hec.fit(X_train, y_train)
        predictions = hec.predict(X_test)
        
        # Validate results
        assert len(predictions) == len(y_test)
        assert all(pred in y_train.unique() for pred in predictions)
        
        # Test feature importance
        importance_df = hec.get_feature_importance()
        assert len(importance_df) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
