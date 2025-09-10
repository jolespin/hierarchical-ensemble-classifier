import numpy as np
import pandas as pd
import networkx as nx
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, to_tree
import matplotlib.pyplot as plt
from collections import defaultdict

__version__ = "2025.9.9"

class HierarchicalEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Hierarchical classifier with unique features and models per subclassifier."""
    
    def __init__(self, hierarchy, estimators):
        """
        Parameters
        ----------
        hierarchy : nx.DiGraph
            Directed graph representing the decision hierarchy
        estimators : dict
            Nested dict: {subclassifier_name: {"features": [...], "classifier": estimator}}
        """
        self.hierarchy = hierarchy
        self.estimators = estimators
        self.fitted_estimators_ = {}
        self.paths_ = {}
        
    def fit(self, X, y):
        """Fit the hierarchical classifier.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (n_samples, n_features)
        y : pd.Series
            Terminal classifications (must be terminal nodes in hierarchy)
        """
        # Validate inputs
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series")
            
        # Check that all y values are terminal nodes
        terminal_nodes = [n for n in self.hierarchy.nodes() if self.hierarchy.out_degree(n) == 0]
        if not set(y.unique()).issubset(set(terminal_nodes)):
            raise ValueError("All y values must be terminal nodes in the hierarchy")
        
        # Build paths from root to each terminal node
        root = [n for n in self.hierarchy.nodes() if self.hierarchy.in_degree(n) == 0][0]
        self.paths_ = {}
        for terminal in terminal_nodes:
            path = nx.shortest_path(self.hierarchy, root, terminal)
            self.paths_[terminal] = path
        
        # Fit each subclassifier
        for subclassifier_name, config in self.estimators.items():
            # Get samples that pass through this subclassifier
            mask = pd.Series(False, index=X.index)
            for terminal, path in self.paths_.items():
                if subclassifier_name in path:
                    mask |= (y == terminal)
            
            if mask.sum() == 0:
                continue
                
            X_sub = X.loc[mask, config["features"]]
            
            # Determine y_sub based on immediate children
            y_sub = pd.Series(index=X_sub.index, dtype=object)
            children = list(self.hierarchy.successors(subclassifier_name))
            
            for idx in X_sub.index:
                terminal = y[idx]
                path = self.paths_[terminal]
                # Find the immediate child in the path
                sub_idx = path.index(subclassifier_name)
                if sub_idx < len(path) - 1:
                    y_sub[idx] = path[sub_idx + 1]
            
            # Fit the classifier
            clf = clone(config["classifier"])
            clf.fit(X_sub, y_sub)
            self.fitted_estimators_[subclassifier_name] = clf
            
        return self
    
    def predict(self, X):
        """Predict class labels."""
        predictions = []
        root = [n for n in self.hierarchy.nodes() if self.hierarchy.in_degree(n) == 0][0]
        
        for idx in X.index:
            current = root
            while self.hierarchy.out_degree(current) > 0:
                if current not in self.fitted_estimators_:
                    # If no classifier, choose first child
                    current = list(self.hierarchy.successors(current))[0]
                else:
                    config = self.estimators[current]
                    X_row = X.loc[[idx], config["features"]]
                    pred = self.fitted_estimators_[current].predict(X_row)[0]
                    current = pred
            predictions.append(current)
            
        return pd.Series(predictions, index=X.index)
    
    def predict_proba(self, X):
        """Predict probability for each path through the hierarchy."""
        root = [n for n in self.hierarchy.nodes() if self.hierarchy.in_degree(n) == 0][0]
        terminal_nodes = [n for n in self.hierarchy.nodes() if self.hierarchy.out_degree(n) == 0]
        
        proba_dict = defaultdict(list)
        
        for idx in X.index:
            # Calculate probability for each terminal node
            terminal_probas = {}
            for terminal in terminal_nodes:
                path = self.paths_[terminal]
                prob = 1.0
                
                for i in range(len(path) - 1):
                    node = path[i]
                    next_node = path[i + 1]
                    
                    if node in self.fitted_estimators_:
                        config = self.estimators[node]
                        X_row = X.loc[[idx], config["features"]]
                        clf = self.fitted_estimators_[node]
                        
                        # Get probability for the next node
                        probas = clf.predict_proba(X_row)[0]
                        classes = clf.classes_
                        if next_node in classes:
                            node_prob = probas[list(classes).index(next_node)]
                            prob *= node_prob
                        else:
                            prob = 0
                            break
                
                terminal_probas[terminal] = prob
            
            for terminal, prob in terminal_probas.items():
                proba_dict[terminal].append(prob)
        
        return pd.DataFrame(proba_dict, index=X.index)
    
    def plot_model(self, figsize=(10, 8), title="Hierarchical Model", ax=None):
        """Plot the hierarchical model structure."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        pos = nx.nx_agraph.graphviz_layout(self.hierarchy, prog='dot')
        nx.draw(self.hierarchy, pos, ax=ax, with_labels=True, 
                node_color='lightblue', node_size=1500, 
                font_size=10, font_weight='bold',
                arrows=True, arrowsize=20, edge_color='gray')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        return ax.figure, ax
    
    def plot_prediction_paths(self, X, y_true=None, figsize=(10, 8), ax=None):
        """Plot prediction paths with probabilities."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Get predictions and probabilities
        probas = self.predict_proba(X)
        
        # Create weighted graph
        G = self.hierarchy.copy()
        
        # Add edge weights based on average probabilities
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 0
        
        for terminal, path in self.paths_.items():
            avg_prob = probas[terminal].mean()
            for i in range(len(path) - 1):
                current_weight = G[path[i]][path[i+1]].get('weight', 0)
                G[path[i]][path[i+1]]['weight'] = max(current_weight, avg_prob)
        
        # Plot
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        
        # Draw edges with width proportional to weight
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax, width=[w*5 for w in weights], 
                              edge_color=weights, edge_cmap=plt.cm.Blues)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', 
                              node_size=1500)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
        
        # Add edge labels with weights
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" 
                      for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)
        
        ax.set_title("Prediction Paths", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        return ax.figure, ax


class HierarchyBuilder:
    """Build hierarchy using eigenprofiles and hierarchical clustering."""
    
    def __init__(self, linkage='ward', subclassifier_prefix='y'):
        """
        Parameters
        ----------
        linkage : str
            Linkage type for hierarchical clustering
        subclassifier_prefix : str
            Prefix for subclassifier names
        """
        self.linkage = linkage
        self.subclassifier_prefix = subclassifier_prefix
        self.hierarchy_ = None
        self.Y_ = None
        self.eigenprofiles_ = None
        
    def fit(self, X, y):
        """Build hierarchy from eigenprofiles.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Class labels
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series")
        
        # Calculate eigenprofiles (1st PC for each class)
        classes = y.unique()
        eigenprofiles = []
        
        pca = PCA(n_components=1)
        for cls in classes:
            X_cls = X[y == cls]
            pca.fit(X_cls)
            eigenprofiles.append(pca.components_[0])
        
        self.eigenprofiles_ = pd.DataFrame(eigenprofiles, index=classes)
        
        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=0,
            linkage=self.linkage
        )
        clustering.fit(self.eigenprofiles_)
        
        # Convert to hierarchy
        self.hierarchy_ = self._clustering_to_hierarchy(clustering, classes)
        
        # Create Y_ matrix with subclassifier targets
        self.Y_ = self._create_target_matrix(X, y)
        
        return self
    
    def _clustering_to_hierarchy(self, clustering, classes):
        """Convert clustering to NetworkX DiGraph."""
        n_samples = len(classes)
        children = clustering.children_
        
        # Build the graph
        G = nx.DiGraph()
        
        # Map original indices to class names
        node_map = {i: classes[i] for i in range(n_samples)}
        
        # Add internal nodes
        for i, (left, right) in enumerate(children):
            internal_node = f"{self.subclassifier_prefix}{i+1}"
            node_map[n_samples + i] = internal_node
            
            # Add edges from internal node to children
            left_node = node_map[left]
            right_node = node_map[right]
            
            G.add_edge(internal_node, left_node)
            G.add_edge(internal_node, right_node)
        
        # The root is the last internal node
        root = f"{self.subclassifier_prefix}{len(children)}"
        
        # Ensure all nodes are in the graph
        for cls in classes:
            if cls not in G:
                G.add_node(cls)
        
        return G
    
    def _create_target_matrix(self, X, y):
        """Create target matrix for all subclassifiers."""
        Y = pd.DataFrame(index=X.index)
        
        # Get all internal nodes (subclassifiers)
        internal_nodes = [n for n in self.hierarchy_.nodes() 
                         if self.hierarchy_.out_degree(n) > 0]
        
        # Sort by depth (root first)
        root = [n for n in internal_nodes if self.hierarchy_.in_degree(n) == 0][0]
        
        for node in internal_nodes:
            Y[node] = np.nan
            
            # Get all terminal nodes reachable from this node
            reachable = set()
            for terminal in y.unique():
                if nx.has_path(self.hierarchy_, node, terminal):
                    reachable.add(terminal)
            
            # For samples with these terminals, determine immediate child
            mask = y.isin(reachable)
            if mask.sum() > 0:
                children = list(self.hierarchy_.successors(node))
                
                for idx in X.index[mask]:
                    terminal = y[idx]
                    # Find which child leads to this terminal
                    for child in children:
                        if child == terminal or \
                           (self.hierarchy_.out_degree(child) > 0 and 
                            nx.has_path(self.hierarchy_, child, terminal)):
                            Y.loc[idx, node] = child
                            break
        
        return Y
    
    def get_hierarchy(self):
        """Get the built hierarchy."""
        return self.hierarchy_
    
    def get_targets(self):
        """Get the target matrix Y_."""
        return self.Y_