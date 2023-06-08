##
## Functions related to creating and working with a custom tree object
## This custom tree can work with SHAP and ALE plots.
## See: 
## https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Example%20of%20loading%20a%20custom%20tree%20model%20into%20SHAP.html
## https://sklearn-template.readthedocs.io/en/latest/user_guide.html
##

import pandas as pd
import numpy as np
import copy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.tree import plot_tree
from sklearn.utils.validation import check_is_fitted
from sklearn.tree._tree import Tree

class TemplateTree(Tree):
    
    # Override attributes to make writeable
    children_left = None
    children_right = None
    n_node_samples = None
    weighted_n_node_samples = None
    feature = None
    threshold = None
    value = None
    
    def __init__(self, n_features, n_classes, n_outputs, 
                 tree_dict, max_depth=None):
        """ Instantiates a custom tree object for use with TemplateClassifier.
        Allows use of some tree functions from sklearn. 
          Inputs:
            tree_dict: Dictionary containing tree data suitable for SHAP.  
              The dictionary must contain the following keys:
                children_left:  Array containing the indices of the left children
                children_right: Array containing the indices of the right children
                children_default: Array containing the child nodes for missing values
                  (can be set to a copy of children_right)
                features: Indices of the features used to split each node
                thresholds:  Thresholds for the feature split
                values: Values for return at the split (array of arrays)
                node_sample_weight:  Relative number of samples reaching the node.
            max_depth:  Optional.  The depth of the tree.  If it is not
              known, can be set using set_max_depth()
         """
        #super(TemplateTree, self).__init__(self.n_features, self.n_classes, self.n_outputs)
        
        self.max_depth = max_depth
        self.children_left = tree_dict['children_left']
        self.children_right = tree_dict['children_right']
        self.n_node_samples = tree_dict['n_node_samples']
        self.weighted_n_node_samples = self.n_node_samples
        self.feature = tree_dict['feature']
        self.threshold = tree_dict['threshold']
        
        # Handle values - if singular, convert
        if isinstance(tree_dict['value'][0], np.number):
            self.value = np.array([np.array([[v]]) for v in tree_dict['value']])
        else:
            self.value = tree_dict['value']
    
    def __new__(cls,*args,**kwargs):
        return super().__new__(cls, args[0], args[1], args[2])



class TemplateClassifier(BaseEstimator, ClassifierMixin):
    
    @staticmethod
    def convert_tree_dict(tree_dict):
        """ Converts a tree dictionary of the type used by
        sklearn.tree._tree.Tree to the format needed by SHAP.
        The default children object is set to the same as children_right.
          Inputs:  tree_dict: sklearn.tree._tree.Tree dictionary 
            (keys 'chldren_right', 'children_right', 
            'feature', 'threshold', 'n_node_samples', 
            'impurity', 'value')
          Value: tree dict with keys ('children_left', 'children_right',
            'children_default', 'features', thresholds', 'values',
            'node_sample_weight')
        """
        
        new_tree_dict = {}
        new_tree_dict['children_left'] = tree_dict['children_left'].copy()
        new_tree_dict['children_right'] = tree_dict['children_right'].copy()
        new_tree_dict['children_default'] = tree_dict['children_right'].copy()
        new_tree_dict['features'] = tree_dict['feature'].copy()
        new_tree_dict['thresholds'] = tree_dict['threshold'].copy()
        new_tree_dict['node_sample_weight'] = tree_dict['n_node_samples'].copy()
        
        # Values - save as array 
        new_tree_dict['values'] = np.array([t.copy() if isinstance(t, list) else [t] \
            for t in tree_dict['value']])
                                   
        return new_tree_dict
    

    
    def measure_depth(self, node_id = 0, depth = 1):
        """ Traverses a custom tree, finding the max depth.
          Inputs:
            node_id: Starting node for counting.  0 indicates the root.
            depth:  Starting depth of node_id.  The tree depth below 
              this is returned.
          Value:  Number of layers in the tree (int)
        """ 

        # Get children
        this_left = self.tree_dict['children_left'][node_id]
        this_right = self.tree_dict['children_right'][node_id]
    
        # If a leaf node return the value
        if this_left == this_right:
            return depth + 1
        
        # Helper function to pass most features to the next node
        def return_node(node_id):
            return self.measure_depth(node_id, depth + 1)
    
        # Traverse tree both ways
        return np.max([return_node(this_left),
                       return_node(this_right)])
    
    def predict_recurs(self, X, node_id = 0):
        """ Traverses the tree and returns the value for a single observation
          Inputs:
            X: Array of feature values for one observation.
            node_id: Starting node for counting.  0 indicates the root.
            depth:  Starting depth of node_id.  The tree depth below 
              this is returned.
          Value:  Number of layers in the tree (int)
        """ 

        # Get children
        this_left = self.tree_dict['children_left'][node_id]
        this_right = self.tree_dict['children_right'][node_id]
    
        # If a leaf node return the value
        if this_left == this_right:
            return self.tree_dict['value'][node_id]
        
        this_feature_value = X[self.tree_dict['feature'][node_id]]     
        
        # Go right or left depending on threshold
        if this_feature_value <= self.tree_dict['threshold'][node_id]:
            return self.predict_recurs(X, self.tree_dict['children_left'][node_id])
        else:
            return self.predict_recurs(X, self.tree_dict['children_right'][node_id])
    
    def set_max_depth(self):
        self.max_depth = self.measure_depth()
        
    #def __sklearn_is_fitted__(self):
    #    return self.is_fitted_
        
    def __init__(self, tree_dict, max_depth=None, criterion='friedman_mse'):
        """ Instantiates a custom estimator for use with SHAP or PyALE.
          Inputs:
            tree_dict: Dictionary containing tree data suitable for SHAP. These arrays
              are the same as those found in sklearn.tree.tree.  The dictionary must
              contain the following keys:
                children_left:  Array containing the indices of the left children
                children_right: Array containing the indices of the right children
                feature: Indices of the features used to split each node
                threshold:  Thresholds for the feature split
                value: Values for return at the split (array of arrays)
                n_node_samples:  Relative number of samples reaching the node.
            max_depth:  Optional.  The depth of the tree.  If it is not
              known, can be set using set_max_depth()
            criterion: Criterion for the estimator.  Default is friedman_mse
         """
            
        self.max_depth = None
        self.tree_dict = tree_dict
        self.model = None
        self.tree_ = None 
        self.n_features = None
        self.n_classes = None
        self.n_outputs = None
        self.is_fitted_ = False
        self.criterion = criterion

    def fit(self, X, y):
        if self.max_depth is None:
            self.set_max_depth()
        #if 'children_default' not in self.tree_dict.keys():
        #    self.tree_dict['children_default'] = self.tree_dict['children_right'].copy()
            
        self.model = {"trees": [self.convert_tree_dict(self.tree_dict)]}
        self.n_features = len(set([f for f in self.tree_dict['feature'] if f != -2]))
        self.n_classes = np.array([1])
        self.n_outputs = 1
        self.tree_ = TemplateTree(self.n_features, self.n_classes, self.n_outputs,
                                  self.tree_dict, self.max_depth)
        self.is_fitted_ = True
        return self
        
    def predict(self, X):
        # Check fit had been called
        if self.is_fitted_:
            return X.apply(lambda x: self.predict_recurs(x), axis=1)
        else:
            return None
        
        
