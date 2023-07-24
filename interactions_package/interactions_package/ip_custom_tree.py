###########################################################
##
## Functions related to creating and working with a custom 
## estimator containing a customized decision tree.
## This custom estimator can work with SHAP and ALE.
##
## See:
##  https://sklearn-template.readthedocs.io/en/latest/user_guide.html
##  https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
##  https://tinyurl.com/39absrss 
##    ("Example of loading a custom tree model into SHAP")
##
############################################################

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree._tree import Tree

class TemplateTree(Tree):
    """ Customized sklearn.tree.tree object, with writeable
    attributes.  Used in custom classifier, and particularly 
    useful for plotting via sklearn.tree.plot_tree()"""
    
    # Override attributes to make writeable
    children_left = None
    children_right = None
    n_node_samples = None
    weighted_n_node_samples = None
    feature = None
    threshold = None
    value = None
    impurity = None
    max_depth = None
    
    def set_max_depth(self, max_depth):
        """ Set the max depth of the tree """
        self.max_depth = max_depth
        
    def get_max_depth(self):
        """ Get the max depth of the tree """
        return self.max_depth 
        
    def set_n_node_samples(self, n_node_samples):
        """ Set the node sample arrays """
        self.n_node_samples  = n_node_samples 
        self.weighted_n_node_samples = n_node_samples
    
    def __init__(self, n_features, n_classes, n_outputs, 
                 tree_dict, max_depth=None):
        """ Instantiates a custom tree object for use with TemplateClassifier.
        Allows use of some tree functions from sklearn. 
          Inputs:
            n_features:  The number of features in the tree
            n_classes:  The number of classes in the target variable;
              see the docs for sklearn.tree.tree
            n_outputs: Number of outputs per observation returned from fit()
            tree_dict: Dictionary containing tree data suitable for SHAP.  
              The dictionary must contain the following keys:
                children_left:  Array containing the indices of the left children
                children_right: Array containing the indices of the right children
                feature: Indices of the features used to split each node
                threshold:  Thresholds for the feature split
                value: Values for return at the split (array of arrays)
                n_node_samples:  Relative number of samples reaching the node.
            max_depth:  Optional.  The depth of the tree.  If it is not
              known, can be set using set_max_depth()
        """
        self.set_max_depth(max_depth)
        self.children_left = tree_dict['children_left']
        self.children_right = tree_dict['children_right']
        self.feature = tree_dict['feature']
        self.threshold = tree_dict['threshold']
        
        # Impurity is optional, and not used here
        if 'impurity' in tree_dict.keys():
            self.impurity = tree_dict['impurity']
        else:
            self.impurity = np.empty(shape=self.children_left.shape)
        
        # The node samples are optional; they can be set with fit()
        # The weighted node samples are just set to the node samples
        if 'n_node_samples' in tree_dict.keys():
            self.set_n_node_samples(tree_dict['n_node_samples'])
        else:
            self.set_n_node_samples(np.empty(shape=self.children_left.shape))
        
        # Node values - if singular, convert
        if isinstance(tree_dict['value'][0], np.number):
            self.value = np.array([np.array([[v]]) for v in tree_dict['value']])
        else:
            self.value = tree_dict['value']
    
    def __new__(cls,*args,**kwargs):
        """ Initialize the parent class with the n_features, n_classes,
        and n_outputs arguments """
        return super().__new__(cls, args[0], args[1], args[2])

class TemplateClassifier(BaseEstimator, ClassifierMixin):
    """ Estimator containing a custom decision tree, which can be derived
    from an existing tree, or made up by a user.  This object can be
    used in SHAP or ALE, and can also be plotted by sklearn.tree.plot_tree()
    """
    
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
        this_left = self.tree_.children_left[node_id]
        this_right = self.tree_.children_right[node_id]
    
        # If a leaf node return the value
        if this_left == this_right:
            return depth + 1
        
        # Helper function to pass most features to the next node
        def return_node(node_id):
            return self.measure_depth(node_id, depth + 1)
    
        # Traverse tree both ways
        return np.max([return_node(this_left),
                       return_node(this_right)])
    
    def measure_node_weight_X(self, X, node_id = 0, node_ary = None):
        """ Traverses the tree and returns an array of 0's and 1's.
        Visited nodes are marked as 1, unvisited as zeros.  Recursive 
        function.
          Inputs:
            X: Array of feature values for one observation.
            node_id: Starting node for counting.  0 indicates the root.
            node_ary: Array containing 0's at unvisited node indices,
              1's at visited indices, so far in the traversal.  If None,
              assume no nodes have been visited.
          Value:  Numpy array containing 0's at unvisited node indices,
              1's at visited indices.
        """ 
        
        if node_ary is None:
            node_ary = np.zeros(len(self.tree_.children_left))
        
        # Set the indicator we have visited this node
        node_ary[node_id] = 1
        
        # Get children
        this_left = self.tree_.children_left[node_id]
        this_right = self.tree_.children_right[node_id]
        
        # If a leaf node return the array
        if this_left == this_right:
            return node_ary
        
        this_feature_value = X[self.tree_.feature[node_id]]     
        
        # Go right or left depending on threshold
        if this_feature_value <= self.tree_.threshold[node_id]:
            return self.measure_node_weight_X(X, this_left, node_ary)
        else:
            return self.measure_node_weight_X(X, this_right, node_ary)

    def measure_node_weight(self, data):
        """ Traverses the tree and returns an array containing counts of
        the number of samples in the passed=in dataset which visit each node.
          Inputs:
            data: Feature data for a set of observations
          Value:  Numpy array containing the counts of visited nodes. 
        """
        
        # Helper function to apply the single-observation function
        def apply_class_meas(ser):
            return pd.Series(self.measure_node_weight_X(ser))
        
        return data.apply(apply_class_meas, axis=1) \
            .sum(axis=0) \
            .astype(int) \
            .to_numpy()
    
    def predict_recurs(self, X, node_id = 0):
        """ Traverses the tree and returns the value for a single observation
          Inputs:
            X: Array of feature values for one observation.
            node_id: Starting node for counting.  0 indicates the root.
          Value:  Tree model prediction
        """ 

        # Get children
        this_left = self.tree_.children_left[node_id]
        this_right = self.tree_.children_right[node_id]
    
        # If a leaf node return the value
        if this_left == this_right:
            return self.tree_.value[node_id]
        
        this_feature_value = X[self.tree_.feature[node_id]]     
        
        # Go right or left depending on threshold
        if this_feature_value <= self.tree_.threshold[node_id]:
            return self.predict_recurs(X, this_left)
        else:
            return self.predict_recurs(X, this_right)
    
    def fit(self, X=None, y=None):
        """ "Fits" the estimator.  Creates objects needed for SHAP and for
        plotting.  Optionally (if X data is suppled), resets the n_node_samples array 
        to reflect the number of observations in the input data
        reaching each node.  This "fit" doesn't modify the model much, if at all,
        since this custom estimator is user-defined or derived from another tree.  
          Inputs:
            X:  Features data.  Optional.  If supplied, the number of observations reaching
              each node is calculated from this data.
            Y: Target series.  Ignored.
          Value:  "Fitted" estimator.
        """
        if self.tree_.get_max_depth() is None:
            self.max_depth = self.measure_depth()
            self.tree_.set_max_depth(self.max_depth)
        
        # If we have features data, measure the number of times each node is reached.
        if X is not None:
            self.tree_dict['n_node_samples'] = self.measure_node_weight(X)
            self.tree_.set_n_node_samples(self.tree_dict['n_node_samples'])
        
        self.is_fitted_ = True
        return self
    
    def get_shap_model(self):
        if self.is_fitted_:
            new_tree_dict = {}
            new_tree_dict['children_left'] = self.tree_.children_left.copy()
            new_tree_dict['children_right'] = self.tree_.children_right.copy()
            new_tree_dict['children_default'] = self.tree_.children_right.copy()
            new_tree_dict['features'] = self.tree_.feature.copy()
            new_tree_dict['thresholds'] = self.tree_.threshold.copy()
            new_tree_dict['node_sample_weight'] = self.tree_.n_node_samples.copy()
        
            # Values - save as array 
            new_tree_dict['values'] = np.array([t.copy() if isinstance(t, list) else [t] \
                for t in self.tree_.value])
            new_tree_dict['values'] = np.array([t[0].copy() for t in self.tree_.value])
            return {"trees": [new_tree_dict]}
        else:
            return None
    
    def predict(self, X):
        # Check fit had been called
        if self.is_fitted_:
            return X.apply(lambda x: self.predict_recurs(x)[0,0], axis=1)
        else:
            return None

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
        
        # The criterion is required but not used for the custom tree
        self.criterion = criterion
        self.tree_dict = tree_dict
        
        # Single ouptut and binary only
        self.n_features = len(set([f for f in tree_dict['feature'] if f != -2]))
        self.n_classes = np.array([1])
        self.n_outputs = 1
        self.max_depth = None

        self.is_fitted_ = False
        
        # Initialize the tree
        self.tree_ = TemplateTree(self.n_features, self.n_classes, self.n_outputs,
                                  tree_dict, max_depth)
        


        

        
        
