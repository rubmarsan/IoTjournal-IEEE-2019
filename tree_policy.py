from policy_interface import PolicyIF
import sklearn
from sklearn import tree
import numpy as np

class treePolicy(PolicyIF):
    def __init__(self, decision_tree_or_policy, max_num_leaves = 20, x_max = None, y_max = None, z_max = None):
        if type(decision_tree_or_policy) == sklearn.tree.tree.DecisionTreeClassifier:
            self.decision_tree = decision_tree_or_policy
        else:
            from itertools import product
            assert x_max is not None and y_max is not None and z_max is not None
            print("Constructing the tree")
            X = np.array(list(product(range(x_max), range(y_max), range(z_max))))
            Y = np.array(decision_tree_or_policy)

            clf = sklearn.tree.DecisionTreeClassifier(max_leaf_nodes=max_num_leaves, criterion="gini") #, class_weight={0: 1, 1: 1, 2: 5})
            clf.fit(X, Y)
            self.decision_tree = clf

    def state_to_action(self, compound_state, x, y, z):
        return self.decision_tree.predict(np.array([[x, y, z]]))
