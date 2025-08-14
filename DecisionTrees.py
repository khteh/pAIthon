import pandas as pd
import numpy, math
import matplotlib.pyplot as plt
from utils import DecisionTreeViz

class DecisionTree():
    _X_train = None
    _Y_train = None
    _root_indices = None
    _feature: int = None
    _tree = []
    def __init__(self, root_indices, feature):
        self._root_indices = root_indices
        self._feature = feature
        self.PrepareData()

    def PrepareData(self):
        self._X_train = numpy.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
        self._Y_train = numpy.array([1,1,0,0,1,0,0,1,1,0])        

    def Entropy(self, y):
        """
        Args:
        y (ndarray): Numpy array indicating whether each example at a node is
            edible (`1`) or poisonous (`0`)
        
        Returns:
            entropy (float): Entropy at that node

        H(p1) = -p1log2(p1) - (1 - p1)log2(1 - p1)
            = -p1log2(p1) - p0log2(p0)
        Note:
        (1) Use log2 to make the symmetric bell curve range from 0 to 1
        (2) 0log2(0) = 0 <- log(0) is negative infinity
        """
        p = 0 if not len(y) else sum(y) / len(y)        
        return 0 if p == 0 or p == 1 else -p * numpy.log2(p) - (1 - p) * numpy.log2(1 - p)

    def information_gain(self, node_indices, feature):
        """
        Compute the information of splitting the node on a given feature
        
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
            feature (int):           Index of feature to split on
    
        Returns:
            cost (float):        Cost computed

        Information Gain = H(p1root) - (wl * H(p1l) + wr * H(p1r))
        Here, X has the elements in the node and y is theirs respectives classes
        """
        # Split dataset
        left_indices, right_indices = self.split_dataset(node_indices, feature)
        
        # Some useful variables
        X_node, y_node = self._X_train[node_indices], self._Y_train[node_indices]
        X_left, y_left = self._X_train[left_indices], self._Y_train[left_indices]
        X_right, y_right = self._X_train[right_indices], self._Y_train[right_indices]
        
        node_entropy = self.Entropy(y_node)
        left_entropy = self.Entropy(y_left)
        right_entropy = self.Entropy(y_right)
        w_left = len(X_left) / len(X_node)
        w_right = len(X_right) / len(X_node)
        we = w_left * left_entropy + w_right * right_entropy
        information_gain = node_entropy - we
        print(f"node_entropy: {node_entropy}")
        print(f"left_entropy: {left_entropy}")
        print(f"right_entropy: {right_entropy}")
        print(f"w_left: {w_left}, w_right: {w_right}")
        print(f"weighted_entropy: {we}")
        print(f"information_gain: {information_gain}")
        return information_gain

    def split_dataset(self, node_indices, feature):
        """
        Splits the data at the given node into
        left and right branches
        
        Args:
            X (ndarray):             Data matrix of shape(n_samples, n_features)
            node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
            feature (int):           Index of feature to split on
        
        Returns:
            left_indices (list):     Indices with feature value == 1
            right_indices (list):    Indices with feature value == 0

        Given a dataset and a index feature, return two lists for the two split nodes, the left node has the animals that have 
        that feature = 1 and the right node those that have the feature = 0 
        index feature = 0 => ear shape
        index feature = 1 => face shape
        index feature = 2 => whiskers
        """
        left_indices = []
        right_indices = []
        for i,x in enumerate(self._X_train):
            if i in node_indices:
                if x[feature] == 1:
                    left_indices.append(i)
                else:
                    right_indices.append(i)
        return left_indices, right_indices

    def get_best_split(self, node_indices):
        """
        Returns the optimal feature and threshold value
        to split the node data 
        
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

        Returns:
            best_feature (int):     The index of the best feature to split
        """    
        
        # Some useful variables
        num_features = self._X_train.shape[1]
        
        # You need to return the following variables correctly
        best_feature = -1
        
        max_so_far = 0
        for i in range(num_features):
            tmp = self.information_gain(node_indices, i)
            if tmp > max_so_far:
                best_feature = i
                max_so_far = tmp
        return best_feature
    
    def build_tree_recursive(self, node_indices, branch_name, max_depth, current_depth):
        """
        Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
        This function just prints the tree.
        
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
            branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
            max_depth (int):        Max depth of the resulting tree. 
            current_depth (int):    Current depth. Parameter used during recursive call.
    
        """ 
        # Maximum depth reached - stop splitting
        if current_depth == max_depth:
            formatting = " "*current_depth + "-"*current_depth
            print(formatting, "%s leaf node with indices" % branch_name, node_indices)
            return
    
        # Otherwise, get best split and split the data
        # Get the best feature and threshold at this node
        best_feature = self.get_best_split(node_indices) 
        
        formatting = "-"*current_depth
        print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
        
        # Split the dataset at the best feature
        left_indices, right_indices = self.split_dataset(node_indices, best_feature)
        self._tree.append((left_indices, right_indices, best_feature))
        
        # continue splitting the left and the right child. Increment current depth
        self.build_tree_recursive(left_indices, "Left", max_depth, current_depth+1)
        self.build_tree_recursive(right_indices, "Right", max_depth, current_depth+1)

    def show_tree(self):
        DecisionTreeViz.generate_tree_viz(self._root_indices, self._Y_train, self._tree)

if __name__ == "__main__":
    root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dt = DecisionTree(root_indices, 0)
    feature = dt.get_best_split(root_indices)
    print(f"Best feature to split on: {feature}")
    dt.build_tree_recursive(root_indices, "Root", max_depth=2, current_depth=0)
    dt.show_tree()