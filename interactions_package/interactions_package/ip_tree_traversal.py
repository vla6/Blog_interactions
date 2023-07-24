##
## Functions related to traversing a sklearn decision tree
## (sklearn.tree._tree.Tree).  Note that this code
## assumes a binary classification tree (1 output, 1 value per node)
##

import pandas as pd
import numpy as np
import sklearn as sk

dummy_feature_index = -2
dummy_threshold = -2
leaf_index = -1

#
# Basic tree info functions
#


# Function to return thresholds in a tree, by feature

def get_thresholds_feature(tree, feature_index=0):
    """ Returns all thresholds for a given feature
      Inputs:
        tree:  The tree to traverse (sklearn.tree._tree.Tree)
        feature_index: Integer index of the variable of interest;
            needed as the trees do not include feature names.
      Value:  Dictionary containing an array of thresholds at
        nodes splitting on the feature of interest, and an array
        containing the number of node samples corresponding to each threshold.
    """
    
    # Get indices of feature of interest
    features_ind = (tree.tree_.feature == feature_index)
    
    # Get the thresholds
    thresh = tree.tree_.threshold[features_ind]
    
    # Get the samples at the nodes
    node_samp = tree.tree_.n_node_samples[features_ind]
    
    return pd.DataFrame({'thresh': thresh,
                        'node_samp': node_samp})


# Function to return all nodes upstream of a given node

def get_upstream_nodes(tree, search_node_id,
                   node_id = 0,
                   info_ary = np.array([])):
    """ Traverses the tree, getting the indices of all nodes upstream
    of a given node id. Recursive function.  Usualy you willl not modify 
    node_id or info_ary; these are used during recursion.
      Inputs:
        tree:  The tree to traverse (sklearn.tree._tree.Tree)
        search_node_id: the ID of the node whose ancestors you want to find
        node_id: Starting node for counting.  0 indicates the root.
        info_ary:  Array containing the upstream nodes, starting from the root node.
          Empty if starting from root.
      Value:  Numpy array containing indices of nodes involved in the path.
    """ 
    
    info_ary = info_ary.copy()
    
    # Return the array if we've found the node
    if node_id == search_node_id:
        return info_ary 
    
    # Get children
    this_left = tree.children_left[node_id]
    this_right = tree.children_right[node_id]
    
    # If a leaf node here, not on path, return empty
    if this_left == this_right:
        return np.array([])
    
    # Otherwise try to continue
    info_ary = np.append(info_ary, [node_id])
        
    # Helper function to pass most features to the next node
    def return_node(node_id):
        return get_upstream_nodes(tree, search_node_id, node_id, info_ary)
    
    # Traverse tree both ways
    return np.append(return_node(this_left),
            return_node(this_right))

# Function to return all nodes downstream of a given node

def get_downstream_nodes(tree, node_id,
                   info_ary = np.array([])):
    """ Traverses the tree, getting the indices of all nodes downstream
    of a given node id. Recursive function.  Usualy you willl not modify 
    node_id or info_ary; these are used during recursion.
      Inputs:
        tree:  The tree to traverse (sklearn.tree._tree.Tree)
        node_id: the ID of the node whose descendents you want to find
        info_ary:  Array containing the upstream nodes, starting from the root node.
          Empty if starting from root.
      Value:  Numpy array containing indices of nodes involved in the path.
    """ 
    
    # Add this node to the array
    info_ary = np.append(info_ary, node_id)
    
    # Get children
    this_left = tree.children_left[node_id]
    this_right = tree.children_right[node_id]
    
    # If a leaf node, return the array
    if this_left == this_right:
        return info_ary
        
    # Helper function to pass most features to the next node
    def return_node(node_id):
        return get_downstream_nodes(tree, node_id, info_ary)
    
    # Traverse tree both ways
    return np.unique(np.append(return_node(this_left),
            return_node(this_right)))

# Gets the depth of each node in a tree

def get_node_depth_ary(tree, node_id = 0, depth = 0):
    """ Traverses the tree, finding node depths.  Returns a 2D array with
    1 row per node, columns are the node id and depth.  
      Inputs:
        tree:  The tree to traverse (sklearn.tree._tree.Tree)
        node_id: starting node for traversal, usually the root
        depth: starting depth (depth of node_id)
      Value:  2D numpy array containing 1 row per tree node, with the
        node id and depth of each.
    """ 

    # Get children
    this_left = tree.children_left[node_id]
    this_right = tree.children_right[node_id]
    
    # If a leaf node, return the depth
    if this_left == this_right:
        return np.array([node_id, depth]).reshape((1,2))
        
    # Helper function to pass most features to the next node
    def return_node(node_id, depth):
        return get_node_depth_ary(tree, node_id, depth)
    
    # Traverse tree both ways
    return np.concatenate((np.array([node_id, depth]).reshape((1,2)),
                          return_node(this_left, depth+1),
                           return_node(this_right, depth+1)), axis=0)


def get_node_depth(tree, node_id = 0, depth = 0):
    """ Traverses the tree, finding node depths.  Returns a 1D array of
    depths for each node.  array with
      Inputs:
        tree:  The tree to traverse (sklearn.tree._tree.Tree)
        node_id: starting node for traversal, usually the root
        depth: starting depth (depth of node_id)
      Value:  Numpy array containing depths of each node.
    """ 
    
    ary_2 = get_node_depth_ary(tree, node_id, depth)
    return ary_2[ary_2[:,0].argsort()][:, 1]


#
# Tree simplification functions, to create
# a smaller tree with a subset of nodes
#

# Helper function to convert array data to tree dictionary
def convert_tree_ary_to_dict(simplified_ary):
    """ Takes an array from a tree pruning function (e.g. simplify_tree_include_ary)
    and converts it to a dictionary.
      Inputs:
        simplified_ary:  Array containing tree data for a simplified tree.
      Value:  Dictionary of numpy arrays.  Dictionary keys are: 
           children_left
           children_right
           feature
           threshold
           n_node_samples
           impurity
           value
           original_node_index
    """ 
    
    # Convert to a dictionary
    dict_keys = ['children_left', 'children_right', 'feature', 'threshold', 'n_node_samples',
                 'impurity', 'value', 'original_node_index']
    this_info_dict = {dict_keys[i]:simplified_ary[i, :] for i in range(len(dict_keys))}
    
    # Convert some value arrays to integer type
    int_ary = ['children_left', 'children_right', 'feature', 'original_node_index']
    this_info_dict = {k:(v.astype('int') if k in int_ary else v) for k, v in this_info_dict.items()}
    
    # Get dictionary for remapping indices
    orig_indices = this_info_dict['original_node_index']
    new_indices = np.arange(len(orig_indices))
    ind_remap_dict = dict(zip(orig_indices, new_indices))
    
    # Convert indices for some cases
    index_convert_ary = ['children_left', 'children_right']
    this_info_dict = {k:(np.array([ind_remap_dict[vi] if vi in ind_remap_dict.keys() else vi for vi in v]) \
                            if k in index_convert_ary else v) for k, v in this_info_dict.items()}
    
    return this_info_dict


# Trims tree, including full paths for only selected nodes,
# returning attributes in 2D numpy array

def simplify_tree_include_ary(tree, include_nodes, node_id = 0):
    """ Traverses the tree, trimming off branches not in the included list.
    Returns a 2D numpy array containing the new tree information, with original node
    identifiers.  Recursive function. 
      If the traversal starts above the level included, this function will
    return an empty array. To trim off the top of a tree, start traversal where the inclusion 
    starts. 
      Inputs:
        tree:  The tree to traverse (sklearn.tree._tree.Tree)
        include_nodes: List of nodes to retain in the tree
        node_id: Starting node for counting.  0 indicates the root.
      Value:  2-D numpy array containing node information for a trimmed tree.
        Node indices are as in the initial tree.  Each row contains
        specific tree information.  The rows correspond to:
           children_left
           children_right
           feature
           threshold
           n_node_samples
           impurity
           value
           original_node_index
    """
    
    # Get children
    this_children_left = tree.children_left[node_id]
    this_children_right = tree.children_right[node_id]
    
    # If no children allowed, don't go on
    if (this_children_right not in include_nodes) and (this_children_left not in include_nodes):
        this_children_right = -1
        this_children_left = -1

    # Add selected node data to the 2D array
    this_info = np.array([this_children_left, this_children_right, tree.feature[node_id],
                              tree.threshold[node_id], tree.n_node_samples[node_id], 
                              tree.impurity[node_id], tree.value[node_id][0,0], node_id]) \
            .reshape(8,1)
    
    
    # Helper function to pass information to children
    def return_node(node_id):
        return simplify_tree_include_ary(tree, include_nodes, node_id)
    
    # Traverse if included
    if (this_children_left == this_children_right):
        return this_info
    else:
        return np.concatenate((this_info, return_node(this_children_left),
               return_node(this_children_right)), axis=1)


# Converts 2D numpy array-format trimmed tree to a dictionary format

def simplify_tree_include_dict(tree, include_nodes, node_id = 0):
    """ Traverses the tree, trimming off branches not in the included list.
    Calls simplify_tree_include_ary to perform recursive tree traversal.
    Then converts the 2D array to a dictionary of arrays, and reindexes
    the nodes so that the information can be used in the usual way.
      Inputs:
        tree:  The tree to traverse (sklearn.tree._tree.Tree)
        include_nodes: List of nodes to retain in the tree
        node_id: Starting node for counting.  0 indicates the root.
      Value:  Dictionary of numpy arrays.  Node indices have been remaped 
        to allow array indexing.  Dictionary keys are: 
           children_left
           children_right
           feature
           threshold
           n_node_samples
           impurity
           value
           original_node_index
    """ 
    
    # If the root isn't in the included nodes, find the top-level included node.
    include_node_ary = np.array(include_nodes, dtype=int)
    depths = get_node_depth(tree, 0)[include_node_ary]
    
    # Get starting point - note we should only have one, so ignore multiples!
    start_node = include_node_ary[np.where(depths == np.min(depths))[0][0]]
    
    # Get trimmed tree as 2D array
    this_info_ary = simplify_tree_include_ary(tree, include_nodes, start_node)
    
    this_info_dict = convert_tree_ary_to_dict(this_info_ary)
    
    return this_info_dict

# Trims tree, making a list of nodes into leaves

def simplify_tree_leaf_ary(tree, leaf_nodes, node_id = 0):
    """ Traverses the tree, trimming child nodes of the input list of
    terminal nodes. Returns a 2D numpy array containing the new tree information, 
    with original node identifiers.
      Inputs:
        tree:  The tree to traverse (sklearn.tree._tree.Tree)
        leaf_nodes: List of nodes to make leaf nodes
        node_id: Starting node for counting.  0 indicates the root.
      Value:  2-D numpy array containing node information for a trimmed tree.
        Node indices are as in the initial tree.  Each row contains
        specific tree information.  The rows correspond to:
           children_left
           children_right
           feature
           threshold
           n_node_samples
           impurity
           value
           original_node_index
    """
    
    include_list = [i for i in range(len(tree.feature)) if i not in leaf_nodes]
    
    return simplify_tree_include_ary(tree, include_list, 0)

def simplify_tree_leaf_dict(tree, leaf_nodes, node_id = 0):
    """ Traverses the tree, trimming off any child nodes of the list of 
    leaf nodes supplied.
    Calls simplify_tree_leaf_ary to perform the removal (via simplify_tree_include_ary)
    Then converts the 2D array to a dictionary of arrays, and reindexes
    the nodes so that the information can be used in the usual way.
      Inputs:
        tree:  The tree to traverse (sklearn.tree._tree.Tree)
        leaf_nodes: List of nodes to make leaf nodes
        node_id: Starting node for counting.  0 indicates the root.
      Value:  Dictionary of numpy arrays.  Node indices have been remaped 
        to allow array indexing.  Dictionary keys are: 
           children_left
           children_right
           feature
           threshold
           n_node_samples
           impurity
           value
           original_node_index
    """ 
    
    # Get trimmed tree as 2D array
    this_info_ary = simplify_tree_leaf_ary(tree, leaf_nodes, node_id)
    
    this_info_dict = convert_tree_ary_to_dict(this_info_ary)
    
    return this_info_dict

    


#
# Functions to compare decision paths with respect to features
#

# Tree traversal, returns basic information about two features

def get_tree_stats_two_features(tree, node_id = 0, 
                   info_ary = np.array([0, 0, 0, 0, 0]),
                   feature_index_1 = 0,
                   feature_index_2 = 1):
    """ Traverses the tree, counting nodes of various types.
    Recursive function.  Usualy you willl not modify node_id or
    info_ary; these are used during recursion.
      Inputs:
        tree:  The tree to traverse (sklearn.tree._tree.Tree)
        node_id: Starting node for counting.  0 indicates the root,
        info_ary:  Start counts of various nodes.  See "Value".
          Leave as arrays of 0 to count items at or below node_id.
        feature_index_1: Integer index of the first variable of interest;
            needed as the trees do not include feature names.
        feature_index_2: Integer index of the second variable of interest;
            needed as the trees do not include feature names.
      Value:  Numpy array containing tree information.  The 5 items are:
        total nodes
        total nodes for feature 1
        total nodes for feature 2
        total nodes for feature 1, where the left value is larger
        total nodes for feature 2, where the right value is larger
    """ 
    
    info_ary = info_ary.copy()
    
    # Count nodes
    info_ary[0] += 1
    
    # Get children
    this_left = tree.tree_.children_left[node_id]
    this_right = tree.tree_.children_right[node_id]
    
    # If a leaf node, return
    if this_left == this_right:
        return info_ary
    
    this_feature = tree.tree_.feature[node_id]
    
    if (this_feature == feature_index_1):
        info_ary[1] += 1
    elif (this_feature == feature_index_2):
        info_ary[2] += 1
        value_left = tree.tree_.value[this_left][0, 0]
        value_right = tree.tree_.value[this_right][0, 0]
        if value_left >= value_right:
            info_ary[3] += 1
        else:
            info_ary[4] += 1
        
    # Helper function to pass most features to the next node
    def return_node(node_id):
        return get_tree_stats(tree, node_id, 
                   info_ary = info_ary,
                   feature_index_1 = feature_index_1,
                   feature_index_2 = feature_index_2)
    
    # Traverse tree both ways
    return return_node(this_left) + \
            return_node(this_right)

# Compare values at a feature split.

def path_split_value_diff(tree, decision_paths,
                     feature_index = 0):
    """ For decision paths in a tree, find the first node involving
    a specified feature, if any.  Then, find the change in value
    for the left and right paths after that split"""
    
    # Flag nodes involving the feature in the tree:
    feature_nodes = (tree.tree_.feature == feature_index)
    
    # Repeat the matrix to match the number of decision paths
    feature_nodes_mat = np.asmatrix(np.repeat(feature_nodes, 
                                              decision_paths.shape[0], 0) \
        .reshape((feature_nodes.shape[0], decision_paths.shape[0])).T)
    
    # Element-wise multiplication to get nodes in the decision paths
    # involving the feature
    decision_paths_mask = decision_paths.multiply(feature_nodes_mat)
    found_ind = sp.find(decision_paths_mask == 1)
    
    # Convert to a data frame
    found_df = pd.DataFrame({'ind':found_ind[0], 'node':found_ind[1]}) \
        .sort_values('ind')
    
    # Get the splits
    found_df['val_left'] = found_df['node'] \
        .apply(lambda x: tree.tree_.value[tree.tree_.children_left[x]][0][0])
    found_df['val_right'] = found_df['node'] \
        .apply(lambda x: tree.tree_.value[tree.tree_.children_right[x]][0][0])
    found_df['val_diff'] = found_df['val_right']  - found_df['val_left']
    
    # Get the number of samples reaching this node
    found_df['node_samp'] = found_df['node'] \
        .apply(lambda x: tree.tree_.n_node_samples[x])
    
    # Get mean per path
    found_agg = found_df.groupby('ind')[['val_diff', 'node_samp']].apply('max')
    
    return found_agg

    