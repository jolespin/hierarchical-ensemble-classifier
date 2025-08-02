"""
Hierarchical Ensemble Classifier (HEC)

A scikit-learn compatible implementation of hierarchical ensemble classification
where each sub-classifier can use unique features and classification algorithms.
"""

__version__ = "0.1.0"
__author__ = "Josh L. Espinoza"

import numpy as np
import pandas as pd
import networkx as nx
from collections import OrderedDict, defaultdict
from copy import deepcopy
import warnings
from typing import Dict, List, Union, Optional, Any, Tuple

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_score, check_cv
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import pdist

# Optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from skclust import HierarchicalClustering
    HAS_SKCLUST = True
except ImportError:
    HAS_SKCLUST = False
    warnings.warn("skclust not available, dendrogram functionality will be limited")


class SubClassifier:
    """
    Container for individual sub-classifiers in the hierarchical ensemble.
    
    Each sub-classifier can have its own feature subset and classification algorithm.
    """
    
    def __init__(self, 
                 name: str,
                 estimator: BaseEstimator,
                 features: List[str],
                 hyperparameters: Optional[Dict] = None):
        """
        Initialize a sub-classifier.
        
        Parameters
        ----------
        name : str
            Unique identifier for this sub-classifier
        estimator : BaseEstimator
            Scikit-learn compatible classifier
        features : List[str]
            Feature names to use for this sub-classifier
        hyperparameters : Dict, optional
            Custom hyperparameters for the estimator
        """
        self.name = name
        self.estimator = estimator
        self.features = features
        self.hyperparameters = hyperparameters or estimator.get_params()
        self.is_fitted_ = False
        self.accuracy_ = None
        self.cv_scores_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SubClassifier':
        """Fit the sub-classifier."""
        X_subset = X.loc[:, self.features]
        self.estimator.fit(X_subset, y)
        self.is_fitted_ = True
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the sub-classifier."""
        check_is_fitted(self.estimator)
        X_subset = X.loc[:, self.features]
        return self.estimator.predict(X_subset)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using the sub-classifier."""
        check_is_fitted(self.estimator)
        X_subset = X.loc[:, self.features]
        return self.estimator.predict_proba(X_subset)
        
    @property
    def classes_(self):
        """Get the classes learned by the estimator."""
        if hasattr(self.estimator, 'classes_'):
            return self.estimator.classes_
        return None


class HierarchicalEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Hierarchical Ensemble Classifier (HEC)
    
    A hierarchical ensemble method where each sub-classifier in the hierarchy
    can use different features and different classification algorithms.
    
    The classifier builds a directed graph where each node represents a 
    sub-classifier and edges represent the flow of predictions through
    the hierarchy.
    
    Parameters
    ----------
    name : str, optional
        Name identifier for the classifier
    copy_X : bool, default=True
        Whether to copy X during fitting
    verbose : bool, default=False
        Whether to print verbose output
    
    Attributes
    ----------
    graph_ : networkx.DiGraph
        The hierarchical structure as a directed graph
    sub_classifiers_ : Dict[str, SubClassifier]
        Dictionary of sub-classifiers by name
    paths_ : Dict[str, List[str]]
        Paths from root to each terminal class
    classes_ : ndarray of shape (n_classes,)
        The classes seen at fit
    """
    
    def __init__(self, 
                 name: Optional[str] = None,
                 copy_X: bool = True,
                 verbose: bool = False):
        self.name = name
        self.copy_X = copy_X
        self.verbose = verbose
        
    def _initialize_graph(self):
        """Initialize the directed graph structure."""
        self.graph_ = nx.DiGraph()
        self.sub_classifiers_ = OrderedDict()
        self.paths_ = OrderedDict()
        self._root_node = "input"
        self.graph_.add_node(self._root_node, node_type="input")
        
    def add_sub_classifier(self,
                          name: str,
                          estimator: BaseEstimator,
                          features: List[str],
                          parent: Optional[str] = None) -> 'HierarchicalEnsembleClassifier':
        """
        Add a sub-classifier to the hierarchy.
        
        Parameters
        ----------
        name : str
            Unique name for the sub-classifier
        estimator : BaseEstimator
            Scikit-learn compatible classifier
        features : List[str]
            Feature names to use for this sub-classifier
        parent : str, optional
            Parent node name. If None, connects to root input node.
            
        Returns
        -------
        self : HierarchicalEnsembleClassifier
        """
        if not hasattr(self, 'graph_'):
            self._initialize_graph()
            
        # Create sub-classifier
        sub_clf = SubClassifier(name, estimator, features)
        self.sub_classifiers_[name] = sub_clf
        
        # Add to graph
        self.graph_.add_node(name, node_type="classifier", sub_classifier=sub_clf)
        
        # Connect to parent
        if parent is None:
            parent = self._root_node
        self.graph_.add_edge(parent, name)
        
        return self
        
    def add_terminal_class(self,
                          class_name: str,
                          parent: str) -> 'HierarchicalEnsembleClassifier':
        """
        Add a terminal classification class to the hierarchy.
        
        Parameters
        ----------
        class_name : str
            Name of the terminal class
        parent : str
            Parent sub-classifier name
            
        Returns
        -------
        self : HierarchicalEnsembleClassifier
        """
        if not hasattr(self, 'graph_'):
            self._initialize_graph()
            
        self.graph_.add_node(class_name, node_type="terminal")
        self.graph_.add_edge(parent, class_name)
        
        return self
        
    def _build_paths(self):
        """Build all paths from root to terminal nodes."""
        self.paths_ = OrderedDict()
        
        # Find all terminal nodes
        terminal_nodes = [node for node in self.graph_.nodes() 
                         if self.graph_.nodes[node].get('node_type') == 'terminal']
        
        # Build path for each terminal node
        for terminal in terminal_nodes:
            try:
                path = nx.shortest_path(self.graph_, self._root_node, terminal)
                self.paths_[terminal] = path
            except nx.NetworkXNoPath:
                raise ValueError(f"No path found from root to terminal class {terminal}")
                
    def _prepare_target_matrix(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Prepare the target matrix for training sub-classifiers.
        
        For each sub-classifier, determine which samples should be used
        for training based on the hierarchical paths.
        """
        # Build target matrix
        Y_matrix = pd.DataFrame(index=X.index)
        
        for class_name, path in self.paths_.items():
            # For each sub-classifier in the path (excluding input and terminal)
            classifier_path = [node for node in path[1:-1] 
                             if node in self.sub_classifiers_]
            
            for i, sub_clf_name in enumerate(classifier_path):
                if sub_clf_name not in Y_matrix.columns:
                    Y_matrix[sub_clf_name] = np.nan
                
                # Samples of this class should be labeled for this sub-classifier
                mask = (y == class_name)
                
                if i == 0:
                    # First sub-classifier: binary classification
                    # (this class vs all others that go through this path)
                    Y_matrix.loc[mask, sub_clf_name] = class_name
                else:
                    # Subsequent sub-classifiers: only samples that should
                    # flow through this path
                    Y_matrix.loc[mask, sub_clf_name] = class_name
                    
        return Y_matrix
        
    def fit(self, X, y):
        """
        Fit the hierarchical ensemble classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : HierarchicalEnsembleClassifier
        """
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # Convert to DataFrame/Series if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index)
            
        # Store classes
        self.classes_ = unique_labels(y)
        
        # Build paths if not already done
        if not hasattr(self, 'paths_') or not self.paths_:
            self._build_paths()
            
        # Prepare target matrix
        Y_matrix = self._prepare_target_matrix(X, y)
        
        # Fit each sub-classifier
        for sub_clf_name, sub_clf in self.sub_classifiers_.items():
            if sub_clf_name in Y_matrix.columns:
                # Get samples for this sub-classifier (non-NaN)
                mask = ~Y_matrix[sub_clf_name].isna()
                if mask.sum() > 0:
                    X_sub = X.loc[mask]
                    y_sub = Y_matrix.loc[mask, sub_clf_name]
                    
                    # Fit the sub-classifier
                    sub_clf.fit(X_sub, y_sub)
                    
                    if self.verbose:
                        print(f"Fitted sub-classifier '{sub_clf_name}' with "
                              f"{len(y_sub)} samples and {len(sub_clf.features)} features")
        
        # Store training data if requested
        if self.copy_X:
            self._X_train = X.copy()
            self._y_train = y.copy()
            
        return self
        
    def _predict_single_sample(self, x_sample: pd.Series) -> str:
        """Predict class for a single sample by traversing the hierarchy."""
        current_node = self._root_node
        
        while True:
            # Get successors of current node
            successors = list(self.graph_.successors(current_node))
            
            if not successors:
                # No successors - this shouldn't happen in a valid hierarchy
                raise ValueError(f"No successors for node {current_node}")
                
            # Check if any successor is a terminal node
            terminal_successors = [s for s in successors 
                                 if self.graph_.nodes[s].get('node_type') == 'terminal']
            
            if terminal_successors:
                # If there are terminal successors, we need a classifier to choose
                classifier_successors = [s for s in successors 
                                       if s in self.sub_classifiers_]
                
                if classifier_successors:
                    # Use the classifier to make prediction
                    classifier_node = classifier_successors[0]
                    sub_clf = self.sub_classifiers_[classifier_node]
                    prediction = sub_clf.predict(x_sample.to_frame().T)[0]
                    
                    # Return the prediction (which should be a terminal class)
                    return prediction
                else:
                    # No classifier, return first terminal
                    return terminal_successors[0]
            else:
                # All successors are classifiers, choose the next one
                # For now, take the first one (could be made more sophisticated)
                current_node = successors[0]
                
    def predict(self, X):
        """
        Predict classes for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted classes
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        predictions = []
        for idx in X.index:
            pred = self._predict_single_sample(X.loc[idx])
            predictions.append(pred)
            
        return np.array(predictions)
        
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        This returns the cumulative probabilities along each path
        in the hierarchy.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_samples, n_classes))
        
        # For each sample, compute path probabilities
        for i, idx in enumerate(X.index):
            x_sample = X.loc[idx]
            
            # Compute probability for each terminal class
            for j, class_name in enumerate(self.classes_):
                if class_name in self.paths_:
                    path = self.paths_[class_name]
                    prob = 1.0
                    
                    # Multiply probabilities along the path
                    for node in path[1:-1]:  # Skip input and terminal
                        if node in self.sub_classifiers_:
                            sub_clf = self.sub_classifiers_[node]
                            node_proba = sub_clf.predict_proba(x_sample.to_frame().T)[0]
                            
                            # Find probability for this class
                            if hasattr(sub_clf.estimator, 'classes_'):
                                class_idx = np.where(sub_clf.estimator.classes_ == class_name)[0]
                                if len(class_idx) > 0:
                                    prob *= node_proba[class_idx[0]]
                                    
                    probabilities[i, j] = prob
                    
        # Normalize probabilities
        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        probabilities = probabilities / row_sums
        
        return probabilities
        
    def cross_validate_sub_classifiers(self, 
                                     X: pd.DataFrame, 
                                     y: pd.Series,
                                     cv: Union[int, Any] = 5,
                                     scoring: str = 'accuracy') -> pd.DataFrame:
        """
        Cross-validate each sub-classifier individually.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data
        y : pd.Series  
            Target values
        cv : int or cross-validation generator, default=5
            Cross-validation strategy
        scoring : str, default='accuracy'
            Scoring metric
            
        Returns
        -------
        results : pd.DataFrame
            Cross-validation results for each sub-classifier
        """
        # Prepare target matrix
        Y_matrix = self._prepare_target_matrix(X, y)
        
        results = {}
        
        for sub_clf_name, sub_clf in self.sub_classifiers_.items():
            if sub_clf_name in Y_matrix.columns:
                # Get samples for this sub-classifier
                mask = ~Y_matrix[sub_clf_name].isna()
                if mask.sum() > 0:
                    X_sub = X.loc[mask]
                    y_sub = Y_matrix.loc[mask, sub_clf_name]
                    
                    # Clone the estimator for cross-validation
                    estimator_clone = clone(sub_clf.estimator)
                    
                    # Perform cross-validation
                    scores = cross_val_score(
                        estimator_clone, 
                        X_sub.loc[:, sub_clf.features], 
                        y_sub,
                        cv=cv, 
                        scoring=scoring
                    )
                    
                    results[sub_clf_name] = {
                        'mean_score': scores.mean(),
                        'std_score': scores.std(),
                        'scores': scores,
                        'n_samples': len(y_sub)
                    }
                    
                    # Store in sub-classifier
                    sub_clf.accuracy_ = scores.mean()
                    sub_clf.cv_scores_ = scores
                    
        return pd.DataFrame(results).T
        
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance for sub-classifiers that support it.
        
        Returns
        -------
        importance_df : pd.DataFrame
            Feature importance by sub-classifier
        """
        importance_data = []
        
        for name, sub_clf in self.sub_classifiers_.items():
            if hasattr(sub_clf.estimator, 'feature_importances_'):
                importances = sub_clf.estimator.feature_importances_
                for feature, importance in zip(sub_clf.features, importances):
                    importance_data.append({
                        'sub_classifier': name,
                        'feature': feature,
                        'importance': importance
                    })
            elif hasattr(sub_clf.estimator, 'coef_'):
                # For linear models, use absolute coefficient values
                coefs = np.abs(sub_clf.estimator.coef_[0])
                for feature, coef in zip(sub_clf.features, coefs):
                    importance_data.append({
                        'sub_classifier': name,
                        'feature': feature,
                        'importance': coef
                    })
                    
        return pd.DataFrame(importance_data)
        
    def visualize_hierarchy(self, figsize=(10, 8), node_size=2000):
        """
        Visualize the hierarchical structure.
        
        Parameters
        ----------
        figsize : tuple, default=(10, 8)
            Figure size
        node_size : int, default=2000
            Size of nodes in the visualization
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for visualization")
            
        plt.figure(figsize=figsize)
        
        # Create layout
        pos = nx.spring_layout(self.graph_, k=2, iterations=50)
        
        # Draw nodes with different colors by type
        input_nodes = [n for n in self.graph_.nodes() 
                      if self.graph_.nodes[n].get('node_type') == 'input']
        classifier_nodes = [n for n in self.graph_.nodes() 
                           if self.graph_.nodes[n].get('node_type') == 'classifier']
        terminal_nodes = [n for n in self.graph_.nodes() 
                         if self.graph_.nodes[n].get('node_type') == 'terminal']
        
        nx.draw_networkx_nodes(self.graph_, pos, nodelist=input_nodes, 
                              node_color='lightblue', node_size=node_size, alpha=0.7)
        nx.draw_networkx_nodes(self.graph_, pos, nodelist=classifier_nodes,
                              node_color='lightgreen', node_size=node_size, alpha=0.7) 
        nx.draw_networkx_nodes(self.graph_, pos, nodelist=terminal_nodes,
                              node_color='lightcoral', node_size=node_size, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph_, pos, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph_, pos, font_size=10)
        
        plt.title("Hierarchical Ensemble Classifier Structure")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class HierarchyBuilder:
    """
    Utility class for generating hierarchical topologies from data.
    
    This class helps automatically determine hierarchical structures
    based on data patterns, particularly useful for biological 
    classification tasks.
    """
    
    def __init__(self, 
                 linkage_method: str = 'ward',
                 distance_metric: str = 'euclidean'):
        """
        Initialize hierarchy builder.
        
        Parameters
        ----------
        linkage_method : str, default='ward'
            Linkage method for hierarchical clustering
        distance_metric : str, default='euclidean'  
            Distance metric for clustering
        """
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        self.linkage_matrix_ = None
        self.labels_ = None
        self._clusterer = None
        
    def build_from_class_profiles(self, 
                                 X: pd.DataFrame, 
                                 y: pd.Series,
                                 n_components: Optional[int] = None) -> 'HierarchyBuilder':
        """
        Build hierarchy from class profiles (eigenprofiles).
        
        This method creates representative profiles for each class
        and then clusters these profiles to determine hierarchy.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Class labels
        n_components : int, optional
            Number of PCA components to use for dimensionality reduction
            
        Returns
        -------
        self : HierarchyBuilder
        """
        # Create class profiles (eigenprofiles)
        class_profiles = []
        self.labels_ = sorted(y.unique())
        
        for class_label in self.labels_:
            mask = (y == class_label)
            class_data = X.loc[mask]
            
            if n_components is not None and n_components < X.shape[1]:
                # Use PCA to reduce dimensionality
                pca = PCA(n_components=n_components)
                profile = pca.fit_transform(class_data).mean(axis=0)
            else:
                # Use mean profile
                profile = class_data.mean(axis=0).values
                
            class_profiles.append(profile)
            
        # Convert to array
        profiles_array = np.array(class_profiles)
        
        # Use skclust for hierarchical clustering if available
        if HAS_SKCLUST:
            self._clusterer = HierarchicalClustering(
                method=self.linkage_method,
                metric=self.distance_metric,
                name="ClassProfileClustering"
            )
            
            # Create DataFrame for skclust
            profiles_df = pd.DataFrame(
                profiles_array, 
                index=self.labels_
            )
            
            self._clusterer.fit(profiles_df)
            self.linkage_matrix_ = self._clusterer.linkage_matrix_
        else:
            # Fallback to scipy clustering
            if self.distance_metric == 'euclidean' and self.linkage_method == 'ward':
                self.linkage_matrix_ = linkage(profiles_array, method=self.linkage_method)
            else:
                distances = pdist(profiles_array, metric=self.distance_metric)
                self.linkage_matrix_ = linkage(distances, method=self.linkage_method)
            
        return self
        
    def get_paths(self, n_clusters: Optional[int] = None) -> Dict[str, List[str]]:
        """
        Extract hierarchical paths from the fitted hierarchy.
        
        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters to cut the dendrogram at.
            If None, uses a heuristic to determine optimal cuts.
            
        Returns
        -------
        paths : Dict[str, List[str]]
            Dictionary mapping terminal classes to their hierarchical paths
        """
        if self.linkage_matrix_ is None:
            raise ValueError("Must call build_from_class_profiles first")
            
        if n_clusters is None:
            # Use heuristic: create intermediate clusters at multiple levels
            n_clusters = max(2, len(self.labels_) // 2)
            
        # Cut dendrogram to get cluster assignments
        cluster_labels = cut_tree(self.linkage_matrix_, n_clusters=n_clusters).flatten()
        
        # Build paths
        paths = {}
        
        # Group classes by cluster
        clusters = {}
        for i, class_label in enumerate(self.labels_):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(class_label)
            
        # Create hierarchical paths
        for class_label in self.labels_:
            # Find which cluster this class belongs to
            cluster_id = cluster_labels[self.labels_.index(class_label)]
            
            # Create path: input -> cluster -> class
            cluster_name = f"cluster_{cluster_id}"
            path = ["input", cluster_name, class_label]
            paths[class_label] = path
            
        return paths
        
    def get_target_matrix(self, 
                         y: pd.Series, 
                         paths: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Create target matrix for training hierarchical classifier.
        
        Parameters
        ----------
        y : pd.Series
            Original class labels
        paths : Dict[str, List[str]]
            Hierarchical paths from get_paths()
            
        Returns
        -------
        Y_matrix : pd.DataFrame
            Target matrix with columns for each sub-classifier
        """
        # Initialize target matrix
        Y_matrix = pd.DataFrame(index=y.index)
        
        # Extract all intermediate nodes (sub-classifiers)
        all_nodes = set()
        for path in paths.values():
            all_nodes.update(path[1:-1])  # Exclude input and terminal nodes
            
        # Initialize columns for each sub-classifier
        for node in all_nodes:
            Y_matrix[node] = np.nan
            
        # Fill target matrix based on paths
        for terminal_class, path in paths.items():
            mask = (y == terminal_class)
            
            # For each intermediate node in the path
            for i, node in enumerate(path[1:-1]):  # Skip input and terminal
                if i == 0:
                    # First level: cluster assignment
                    Y_matrix.loc[mask, node] = path[i+2]  # Terminal class
                else:
                    # Subsequent levels
                    Y_matrix.loc[mask, node] = terminal_class
                    
        return Y_matrix
        
    def visualize_dendrogram(self, figsize=(10, 6), **kwargs):
        """
        Visualize the hierarchical clustering dendrogram.
        
        Parameters
        ----------
        figsize : tuple, default=(10, 6)
            Figure size
        **kwargs
            Additional arguments passed to plot_dendrogram
        """
        if self.linkage_matrix_ is None:
            raise ValueError("Must call build_from_class_profiles first")
            
        if HAS_SKCLUST and self._clusterer is not None:
            # Use skclust's enhanced dendrogram plotting
            self._clusterer.plot_dendrogram(figsize=figsize, **kwargs)
        else:
            # Fallback to basic matplotlib dendrogram
            if not HAS_MATPLOTLIB:
                raise ImportError("matplotlib required for visualization")
                
            from scipy.cluster.hierarchy import dendrogram
            
            plt.figure(figsize=figsize)
            dendrogram(self.linkage_matrix_, labels=self.labels_)
            plt.title("Hierarchical Clustering Dendrogram")
            plt.xlabel("Classes")
            plt.ylabel("Distance")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


def create_mixed_feature_pipeline(estimators: Dict[str, BaseEstimator], 
                                 features: Dict[str, List[str]]) -> List[tuple]:
    """
    Create mixed feature pipeline estimators for stacking.
    
    This function creates pipeline estimators that can be used with
    scikit-learn's StackingClassifier when different estimators
    need different feature subsets.
    
    Parameters
    ----------
    estimators : Dict[str, BaseEstimator]
        Dictionary mapping estimator names to estimator objects
    features : Dict[str, List[str]]
        Dictionary mapping estimator names to their feature lists
        
    Returns
    -------
    pipelines : List[tuple]
        List of (name, pipeline) tuples for use with StackingClassifier
        
    Example
    -------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.ensemble import StackingClassifier
    >>> 
    >>> estimators = {
    ...     "sepal": LogisticRegression(random_state=0),
    ...     "petal": DecisionTreeClassifier(random_state=0)
    ... }
    >>> features = {
    ...     "sepal": ["sepal_length", "sepal_width"],
    ...     "petal": ["petal_length", "petal_width"]  
    ... }
    >>> 
    >>> pipelines = create_mixed_feature_pipeline(estimators, features)
    >>> ensemble = StackingClassifier(
    ...     estimators=pipelines,
    ...     final_estimator=LogisticRegression(random_state=0)
    ... )
    """
    assert set(estimators.keys()) == set(features.keys()), \
        "estimators and features must have the same keys"
        
    pipelines = []
    
    for estimator_name in estimators.keys():
        feature_list = features[estimator_name]
        estimator = estimators[estimator_name]
        
        # Create pipeline with feature selection and estimator
        pipeline = Pipeline([
            ('feature_selector', ColumnTransformer([
                ('selected_features', 'passthrough', feature_list)
            ], remainder='drop')),
            ('classifier', estimator)
        ])
        
        pipelines.append((estimator_name, pipeline))
        
    return pipelines


def validate_hierarchy_structure(paths: Dict[str, List[str]]) -> bool:
    """
    Validate that the hierarchy structure is well-formed.
    
    Parameters
    ----------
    paths : Dict[str, List[str]]
        Dictionary mapping terminal classes to their hierarchical paths
        
    Returns
    -------
    is_valid : bool
        True if hierarchy structure is valid
        
    Raises
    ------
    ValueError
        If hierarchy structure is invalid
    """
    if not paths:
        raise ValueError("Paths dictionary cannot be empty")
        
    # Check that all paths start with the same root
    roots = set(path[0] for path in paths.values() if len(path) > 0)
    if len(roots) != 1:
        raise ValueError(f"All paths must start with the same root node. Found roots: {roots}")
        
    # Check that all paths are at least length 2 (root -> terminal)
    for class_name, path in paths.items():
        if len(path) < 2:
            raise ValueError(f"Path for class '{class_name}' must have at least 2 nodes (root -> terminal)")
            
    # Check that terminal nodes are unique
    terminals = list(paths.keys())
    if len(terminals) != len(set(terminals)):
        raise ValueError("Terminal class names must be unique")
        
    return True


def compute_path_probabilities(predictions: pd.DataFrame, 
                             paths: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Compute cumulative probabilities along hierarchical paths.
    
    Parameters
    ----------
    predictions : pd.DataFrame
        Prediction probabilities from sub-classifiers
        Columns should be (sub_classifier, class) MultiIndex
    paths : Dict[str, List[str]]
        Hierarchical paths for each terminal class
        
    Returns
    -------
    path_probabilities : pd.DataFrame
        Cumulative probabilities for each terminal class
    """
    n_samples = predictions.shape[0]
    terminal_classes = list(paths.keys())
    
    # Initialize result matrix
    result = pd.DataFrame(
        index=predictions.index,
        columns=terminal_classes,
        dtype=float
    )
    
    for terminal_class in terminal_classes:
        path = paths[terminal_class]
        
        # Initialize with probability 1.0
        cumulative_prob = pd.Series(1.0, index=predictions.index)
        
        # Multiply probabilities along the path
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            
            # Look for predictions from current_node
            if current_node in predictions.columns.get_level_values(0):
                # Get probability of transitioning to next_node
                if (current_node, next_node) in predictions.columns:
                    node_prob = predictions[(current_node, next_node)]
                    cumulative_prob *= node_prob
                elif (current_node, terminal_class) in predictions.columns:
                    # Direct prediction to terminal class
                    node_prob = predictions[(current_node, terminal_class)]
                    cumulative_prob *= node_prob
                    
        result[terminal_class] = cumulative_prob
        
    return result


# Enhanced visualization functions
def plot_hierarchy_with_performance(hec, 
                                   figsize: Tuple[int, int] = (12, 8),
                                   node_size: int = 3000,
                                   font_size: int = 10,
                                   edge_width: float = 2.0,
                                   show_accuracy: bool = True) -> None:
    """
    Plot the hierarchical structure with performance metrics.
    
    Parameters
    ----------
    hec : HierarchicalEnsembleClassifier
        Fitted hierarchical ensemble classifier
    figsize : tuple, default=(12, 8)
        Figure size
    node_size : int, default=3000
        Size of nodes
    font_size : int, default=10
        Font size for labels
    edge_width : float, default=2.0
        Width of edges
    show_accuracy : bool, default=True
        Whether to show accuracy scores on nodes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization")
        
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create layout
    pos = nx.spring_layout(hec.graph_, k=3, iterations=100, seed=42)
    
    # Separate nodes by type
    input_nodes = [n for n in hec.graph_.nodes() 
                  if hec.graph_.nodes[n].get('node_type') == 'input']
    classifier_nodes = [n for n in hec.graph_.nodes() 
                       if hec.graph_.nodes[n].get('node_type') == 'classifier']
    terminal_nodes = [n for n in hec.graph_.nodes() 
                     if hec.graph_.nodes[n].get('node_type') == 'terminal']
    
    # Color nodes based on performance if available
    classifier_colors = []
    for node in classifier_nodes:
        if node in hec.sub_classifiers_:
            accuracy = hec.sub_classifiers_[node].accuracy_
            if accuracy is not None:
                # Color based on accuracy (green for high, red for low)
                color_intensity = accuracy
                classifier_colors.append(plt.cm.RdYlGn(color_intensity))
            else:
                classifier_colors.append('lightgreen')
        else:
            classifier_colors.append('lightgreen')
    
    # Draw nodes
    nx.draw_networkx_nodes(hec.graph_, pos, nodelist=input_nodes,
                          node_color='lightblue', node_size=node_size,
                          alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(hec.graph_, pos, nodelist=classifier_nodes,
                          node_color=classifier_colors, node_size=node_size,
                          alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(hec.graph_, pos, nodelist=terminal_nodes,
                          node_color='lightcoral', node_size=node_size,
                          alpha=0.8, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(hec.graph_, pos, alpha=0.6, 
                          width=edge_width, ax=ax,
                          edge_color='gray', arrows=True,
                          arrowsize=20, arrowstyle='->')
    
    # Create labels with accuracy if requested
    labels = {}
    for node in hec.graph_.nodes():
        if show_accuracy and node in hec.sub_classifiers_:
            accuracy = hec.sub_classifiers_[node].accuracy_
            if accuracy is not None:
                labels[node] = f"{node}\n({accuracy:.3f})"
            else:
                labels[node] = node
        else:
            labels[node] = node
            
    nx.draw_networkx_labels(hec.graph_, pos, labels=labels,
                           font_size=font_size, ax=ax)
    
    # Add colorbar for accuracy if showing performance
    if show_accuracy and any(n in hec.sub_classifiers_ and 
                           hec.sub_classifiers_[n].accuracy_ is not None 
                           for n in classifier_nodes):
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                                  norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label('Accuracy', rotation=270, labelpad=20)
    
    plt.title("Hierarchical Ensemble Classifier Structure", 
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_feature_usage_heatmap(hec, figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot heatmap showing which features are used by each sub-classifier.
    
    Parameters
    ----------
    hec : HierarchicalEnsembleClassifier
        Fitted hierarchical ensemble classifier
    figsize : tuple, default=(10, 6)
        Figure size
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib and seaborn required for visualization")
    
    # Get all features used across sub-classifiers
    all_features = set()
    for sub_clf in hec.sub_classifiers_.values():
        all_features.update(sub_clf.features)
    all_features = sorted(list(all_features))
    
    # Create binary matrix
    usage_matrix = pd.DataFrame(
        index=sorted(hec.sub_classifiers_.keys()),
        columns=all_features,
        dtype=int
    )
    usage_matrix.fillna(0, inplace=True)
    
    for name, sub_clf in hec.sub_classifiers_.items():
        for feature in sub_clf.features:
            usage_matrix.loc[name, feature] = 1
            
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(usage_matrix, 
                cmap='Blues',
                cbar_kws={'label': 'Feature Used'},
                xticklabels=True,
                yticklabels=True)
    plt.title('Feature Usage by Sub-Classifier', fontsize=14, fontweight='bold')
    plt.xlabel('Features')
    plt.ylabel('Sub-Classifiers')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# Example usage functions
def create_iris_example():
    """
    Create a simple example using the Iris dataset.
    
    This example demonstrates how to set up a hierarchical ensemble
    classifier with different feature subsets for each sub-classifier.
    
    Returns
    -------
    results : dict
        Dictionary containing the fitted model and evaluation results
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report, accuracy_score
    
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
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Cross-validate sub-classifiers
    cv_results = hec.cross_validate_sub_classifiers(X_train, y_train, cv=5)
    
    return {
        'model': hec,
        'accuracy': accuracy,
        'classification_report': report,
        'cv_results': cv_results,
        'predictions': y_pred,
        'probabilities': y_proba,
        'X_test': X_test,
        'y_test': y_test
    }


def create_hierarchy_builder_example():
    """
    Create an example using automatic hierarchy generation.
    
    This example shows how to use the HierarchyBuilder class to automatically
    determine hierarchical structure from data patterns.
    
    Returns
    -------
    results : dict
        Dictionary containing the hierarchy builder and fitted model
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Create synthetic multi-class dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=6,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert to DataFrame/Series
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    class_names = [f"class_{i}" for i in range(6)]
    y = pd.Series([class_names[i] for i in y])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Build hierarchy from data
    hierarchy_builder = HierarchyBuilder(linkage_method='ward', distance_metric='euclidean')
    hierarchy_builder.build_from_class_profiles(X_train, y_train, n_components=10)
    
    # Get hierarchical paths
    paths = hierarchy_builder.get_paths(n_clusters=3)
    
    # Create hierarchical classifier based on hierarchy
    hec = HierarchicalEnsembleClassifier(name="AutoHierarchyHEC", verbose=True)
    
    # Add sub-classifiers based on the discovered hierarchy
    # Get intermediate nodes
    intermediate_nodes = set()
    for path in paths.values():
        intermediate_nodes.update(path[1:-1])
        
    # Add sub-classifiers for each intermediate node
    for i, node in enumerate(sorted(intermediate_nodes)):
        # Use different feature subsets for each sub-classifier
        start_idx = i * 7
        end_idx = min(start_idx + 10, len(feature_names))
        features = feature_names[start_idx:end_idx]
        
        # Use different algorithms for variety
        if i % 3 == 0:
            estimator = LogisticRegression(random_state=42)
        elif i % 3 == 1:
            estimator = DecisionTreeClassifier(random_state=42)
        else:
            estimator = RandomForestClassifier(n_estimators=10, random_state=42)
            
        hec.add_sub_classifier(
            name=node,
            estimator=estimator,
            features=features
        )
        
    # Add terminal classes
    for terminal_class in paths.keys():
        # Find the parent (last intermediate node in path)
        path = paths[terminal_class]
        parent = path[-2]  # Second to last node
        hec.add_terminal_class(terminal_class, parent=parent)
        
    # Fit the model
    hec.fit(X_train, y_train)
    
    # Evaluate
    y_pred = hec.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'hierarchy_builder': hierarchy_builder,
        'paths': paths,
        'model': hec,
        'accuracy': accuracy,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }


# Export main classes and functions
__all__ = [
    "HierarchicalEnsembleClassifier",
    "HierarchyBuilder", 
    "SubClassifier",
    "create_mixed_feature_pipeline",
    "validate_hierarchy_structure",
    "compute_path_probabilities",
    "plot_hierarchy_with_performance",
    "plot_feature_usage_heatmap",
    "create_iris_example",
    "create_hierarchy_builder_example"
]
