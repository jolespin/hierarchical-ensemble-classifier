# ==============================================================================
# Modules
# ==============================================================================
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, to_tree

__version__ = "2025.9.9"
# ==============================================================================
# Hierarchy Builder
# ==============================================================================

class HierarchyBuilder(BaseEstimator, TransformerMixin):
    """
    Automatically builds a class hierarchy from data using eigenprofile clustering.

    This class takes a feature matrix `X` and a target vector `y` of terminal
    classes. For each class, it computes an "eigenprofile" from the first
    principal component. It then performs hierarchical clustering on these
    profiles to create a dendrogram, which is converted into a directed
    graph representing the classification hierarchy.

    Attributes:
    -----------
    hierarchy_ : nx.DiGraph
        The generated hierarchy as a NetworkX directed graph.

    Y_ : pd.DataFrame
        A DataFrame containing the intermediate target labels for each
        subclassifier in the hierarchy. Samples not passing through a
        subclassifier will have a NaN value for that subclassifier's column.

    linkage_matrix_ : np.ndarray
        The linkage matrix produced by scipy's hierarchical clustering.
    """
    def __init__(self, distance_metric='euclidean', linkage_method='ward', subclassifier_prefix='y'):
        """
        Initializes the HierarchyBuilder.

        Parameters:
        -----------
        linkage_method : str, default='ward'
            The linkage method to use for hierarchical clustering. See
            `scipy.cluster.hierarchy.linkage` for options.

        subclassifier_prefix : str, default='y'
            The prefix to use for naming the internal subclassifier nodes.
        """
        self.distance_metric = distance_metric
        self.linkage_method = linkage_method
        self.subclassifier_prefix = subclassifier_prefix

    def fit(self, X, y):
        """
        Computes eigenprofiles, clusters them, and builds the hierarchy graph.

        Parameters:
        -----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The training input samples.

        y : pd.Series of shape (n_samples,)
            The terminal class labels for each sample.

        Returns:
        --------
        self : object
            Returns the fitted instance.
        """
        X, y = check_X_y(X, y, accept_sparse=False, ensure_all_finite=True, ensure_2d=True, dtype=None)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y.ravel(), index=X.index)

        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]

        # 1. Calculate Eigenprofiles for each class
        eigenprofiles = [
            PCA(n_components=1).fit(X[y == cls]).components_[0]
            if X[y == cls].shape[0] > 1 else X[y == cls].values.flatten()
            for cls in self.classes_
        ]

        # 2. Perform hierarchical clustering
        self.linkage_matrix_ = linkage(np.array(eigenprofiles), metric=self.distance_metric, method=self.linkage_method,  optimal_ordering=True)

        # 3. Convert linkage matrix to a NetworkX graph
        self.hierarchy_ = self._linkage_to_graph(self.linkage_matrix_, [str(c) for c in self.classes_])

        # 4. Create the intermediate target matrix Y_
        self.Y_ = self._create_intermediate_targets(X, y.astype(str), self.hierarchy_)

        return self

    def _linkage_to_graph(self, Z, labels):
        """Converts a scipy linkage matrix to a networkx DiGraph."""
        tree = to_tree(Z)
        G = nx.DiGraph()
        node_counter = 1  # <-- Changed from 0 to 1
        root_name = f"{self.subclassifier_prefix}{node_counter}"
        node_counter += 1
        queue = [(tree, root_name)]
        G.add_node(root_name)

        while queue:
            scipy_node, graph_node_name = queue.pop(0)
            for child_node in [scipy_node.get_left(), scipy_node.get_right()]:
                if child_node.is_leaf():
                    child_name = labels[child_node.id]
                else:
                    child_name = f"{self.subclassifier_prefix}{node_counter}"
                    node_counter += 1
                    queue.append((child_node, child_name))
                G.add_edge(graph_node_name, child_name)
        return G

    def _create_intermediate_targets(self, X, y, G):
        """Creates the multi-output target matrix for all subclassifiers."""
        root = [n for n, d in G.in_degree() if d == 0][0]
        subclassifiers = [n for n in G.nodes() if G.out_degree(n) > 0]
        Y_ = pd.DataFrame(index=X.index, columns=subclassifiers, dtype=object)

        for leaf in y.unique():
            path = nx.shortest_path(G, source=root, target=leaf)
            sample_indices = y[y == leaf].index
            for i in range(len(path) - 1):
                if path[i] in Y_.columns:
                    Y_.loc[sample_indices, path[i]] = path[i+1]
        return Y_

# ==============================================================================
# Hierarchical Classifier
# ==============================================================================

class HierarchicalEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    A hierarchical classifier that uses different models and feature subsets
    at each decision node in a predefined hierarchy.

    This classifier traverses a directed graph where internal nodes are
    subclassifiers and leaf nodes are the final class labels.
    """
    def __init__(self, hierarchy, estimators):
        """
        Initializes the HierarchicalEnsembleClassifier.

        Parameters:
        -----------
        hierarchy : nx.DiGraph
            A NetworkX directed graph representing the class hierarchy. Internal
            nodes must be strings, and leaf nodes are the final class labels.

        estimators : dict
            A nested dictionary defining the classifier and features for each
            internal node. The format is:
            {
                'subclassifier_name': {
                    'classifier': scikit-learn_compatible_classifier,
                    'features': [list_of_feature_names]
                },
                ...
            }
        """
        self.hierarchy = hierarchy
        self.estimators = estimators

    def fit(self, X, y):
        """
        Trains the ensemble of subclassifiers.

        Parameters:
        -----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The training input samples. Column names must match the features
            specified in the `estimators` dict.

        y : pd.Series of shape (n_samples,)
            The target values, corresponding to the terminal nodes in the hierarchy.

        Returns:
        --------
        self : object
            The fitted classifier.
        """
        self._validate_input(X, y)
        self.classes_ = unique_labels(y)
        self.fitted_estimators_ = {}

        for name, params in self.estimators.items():
            y_sub = self._get_sub_targets(name, y)
            valid_idx = y_sub.dropna().index
            if valid_idx.empty:
                continue

            X_sub, y_sub = X.loc[valid_idx, params['features']], y_sub.loc[valid_idx]
            
            clf = clone(params['classifier'])
            clf.fit(X_sub, y_sub)
            self.fitted_estimators_[name] = clf
            
        return self

    # def predict(self, X):
    #     """
    #     Predicts the terminal class for each sample in X.

    #     Parameters:
    #     -----------
    #     X : pd.DataFrame of shape (n_samples, n_features)
    #         The input samples to predict.

    #     Returns:
    #     --------
    #     y_pred : np.ndarray of shape (n_samples,)
    #         The predicted terminal class labels.
    #     """
    #     check_is_fitted(self)
    #     if not isinstance(X, pd.DataFrame):
    #         raise TypeError("X must be a pandas DataFrame.")

    #     predictions = []
    #     for _, sample in X.iterrows():
    #         current_node = self._root
    #         while current_node in self.fitted_estimators_:
    #             clf = self.fitted_estimators_[current_node]
    #             features = self.estimators[current_node]['features']
    #             sample_featured = sample[features].values.reshape(1, -1)
    #             current_node = clf.predict(sample_featured)[0]
    #         predictions.append(current_node)
            
    #     return np.array(predictions)
    
    def predict(self, X):
        """
        Predicts the terminal class for each sample in X.

        Parameters:
        -----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.

        Returns:
        --------
        y_pred : np.ndarray of shape (n_samples,)
            The predicted terminal class labels.
        """
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        predictions = []
        # Iterate by index to easily create DataFrame slices
        for i in range(X.shape[0]):
            sample_df = X.iloc[[i]] # Get the current row as a DataFrame
            current_node = self._root

            while current_node in self.fitted_estimators_:
                clf = self.fitted_estimators_[current_node]
                features = self.estimators[current_node]['features']
                
                # Predict using the DataFrame slice, which keeps feature names
                current_node = clf.predict(sample_df[features])[0]
                
            predictions.append(current_node)
            
        return np.array(predictions)

    def _get_sub_targets(self, subclassifier_name, y_true):
        """Generates the target Series for a single subclassifier."""
        successors = list(self.hierarchy.successors(subclassifier_name))
        reachable_leaves = {
            d for succ in successors
            for d in nx.descendants(self.hierarchy, succ).union({succ})
            if self.hierarchy.out_degree(d) == 0
        }

        y_sub = pd.Series(index=y_true.index, dtype=object)
        for idx, leaf in y_true[y_true.isin(reachable_leaves)].items():
            path = nx.shortest_path(self.hierarchy, source=subclassifier_name, target=leaf)
            y_sub.loc[idx] = path[1]
        return y_sub

    def _validate_input(self, X, y):
        """Validates inputs during fit."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series.")

        self.feature_names_in_ = X.columns.to_numpy()
        
        roots = [n for n, d in self.hierarchy.in_degree() if d == 0]
        if len(roots) != 1:
            raise ValueError(f"Hierarchy must have one root, but found {len(roots)}.")
        self._root = roots[0]

        subclassifier_nodes = {n for n in self.hierarchy.nodes() if self.hierarchy.out_degree(n) > 0}
        if set(self.estimators.keys()) != subclassifier_nodes:
            raise ValueError("Estimator keys must match the hierarchy's subclassifier nodes.")

    def plot_model(self, ax=None, **kwargs):
        """
        Visualizes the classifier's hierarchy structure.
        Requires `pydot` to be installed (`pip install pydot`).
        """
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.pop("figsize", (8, 6)))
        try:
            pos = nx.drawing.nx_agraph.graphviz_layout(self.hierarchy, prog='dot', root=self._root)
        except ImportError:
            pos = nx.spring_layout(self.hierarchy)
            print("Warning: pydot not found. Using spring layout. For a hierarchical layout, please `pip install pydot`.")
        
        nx.draw(self.hierarchy, pos, ax=ax, with_labels=True, node_color='c',
                node_size=3000, font_size=10, arrows=True, **kwargs)
        plt.show()

    def plot_prediction_paths(self, X_sample, ax=None, **kwargs):
        """
        Visualizes the prediction path for a single sample, showing probabilities
        at each step. Requires classifiers to have a `predict_proba` method.
        """
        check_is_fitted(self)
        if not isinstance(X_sample, pd.DataFrame) or X_sample.shape[0] != 1:
            raise ValueError("X_sample must be a pandas DataFrame with a single row.")

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 8)))
        
        try:
            pos = nx.drawing.nx_agraph.graphviz_layout(self.hierarchy, prog='dot', root=self._root)
        except ImportError:
            pos = nx.spring_layout(self.hierarchy)
            print("Warning: pydot not found. Using spring layout. For a hierarchical layout, please `pip install pydot`.")

        path_nodes, path_edges, edge_labels = [self._root], [], {}
        current_node = self._root
        while current_node in self.fitted_estimators_:
            clf = self.fitted_estimators_[current_node]
            features = self.estimators[current_node]['features']
            
            probas = clf.predict_proba(X_sample[features])[0]
            next_node = clf.predict(X_sample[features])[0]
            
            for i, child in enumerate(clf.classes_):
                edge_labels[(current_node, child)] = f"{probas[i]:.2f}"
            
            path_edges.append((current_node, next_node))
            path_nodes.append(next_node)
            current_node = next_node

        nx.draw_networkx_nodes(self.hierarchy, pos, ax=ax, node_color='lightgray', node_size=3000)
        nx.draw_networkx_labels(self.hierarchy, pos, ax=ax)
        nx.draw_networkx_edges(self.hierarchy, pos, ax=ax, edge_color='gray', arrows=True)
        nx.draw_networkx_nodes(self.hierarchy, pos, ax=ax, nodelist=path_nodes, node_color='c', node_size=3000)
        nx.draw_networkx_edges(self.hierarchy, pos, ax=ax, edgelist=path_edges, edge_color='black', width=2.5)
        nx.draw_networkx_edge_labels(self.hierarchy, pos, edge_labels=edge_labels, font_color='red')

        ax.set_title(f"Prediction Path for Sample (Predicted: {path_nodes[-1]})")
        plt.show()