'''
Tree for DecisionTree binary classifier
Expected:
- X NxD matrix (features) can only contain binary/categorical features with values 0 or 1
- Y Nx1 matrix (target class known value from training data)
'''
import numpy as np
from tree_node import TreeNode

class TreeBinaryClassifier:
    def __init__(self, X, Y):
        self.verbose = False
        #We enforce that we have read only access to X and Y from this class
        self.X = np.asarray(X).view()
        self.Y = np.asarray(Y).view()
        self.X.flags.writeable = False
        self.Y.flags.writeable = False

        self.nodes = []
        self.relations : list[tuple[int, int, int, int]] = []
        self.max_depth = 30
        '''
        Relation will be stored as tuple
        - parent_node_idx (refers to self.nodes)
        - child_node_idx (refers to self.nodes)
        - j (j in range(D) where D equals self.X.shape[1])
        - feature_value (should be 0 or 1 and indicates the branch of the child_node)
        '''

    def initialize_tree(self):
        '''
        We create the root node and calculate its values
        '''
        self.nodes.clear()
        self.relations.clear()
        self.nodes.append(TreeNode(X=self.X, Y=self.Y))
        N = self.X.shape[0]
        self.nodes[0].i = [idx for idx in range(N)]
        self.nodes[0].calculate_node_values()
        self.nodes[0].current_depth = 0

    def help_split(self, pn, cn1, cn2, j):
        for row_idx in pn.i:
            fv = self.X[row_idx, j]
            if fv == 0:
                cn1.i.append(row_idx)
            elif fv == 1:
                cn2.i.append(row_idx)
            else:
                raise Exception(f"fv {fv} expected to be 0 or 1")

    def do_split(self, from_node_idx, selected_j):
        '''
        Before calling this method, we have found the best j for next split
        Then we create two child nodes for branches for values 0 and 1
        Then we fill the row-indices that belong to the child nodes
        Then we create the relations between the parent and each of the new child nodes
        '''
        child_1_idx = len(self.nodes)
        child_2_idx = child_1_idx + 1
        self.nodes.append(TreeNode(X=self.X, Y=self.Y))
        self.nodes.append(TreeNode(X=self.X, Y=self.Y))

        #Now in self.nodes[from_node_idx] we identify which i are mapped to feature value 0 and which to 1
        pn = self.nodes[from_node_idx]
        c1 = self.nodes[child_1_idx]
        c2 = self.nodes[child_2_idx]
        self.help_split(pn=pn, cn1=c1, cn2=c2, j=selected_j)

        c1.calculate_node_values()
        c2.calculate_node_values()
        c1.used_features_current_path = pn.used_features_current_path.copy()
        c2.used_features_current_path = pn.used_features_current_path.copy()
        c1.used_features_current_path.append(selected_j)
        c2.used_features_current_path.append(selected_j)
        c1.current_depth = pn.current_depth + 1
        if c1.current_depth > self.max_depth:
            c1.is_leaf = True
        c2.current_depth = pn.current_depth + 1
        if c2.current_depth > self.max_depth:
            c2.is_leaf = True

        #Now add the relations
        self.relations.append((from_node_idx, child_1_idx, selected_j, 0))
        self.relations.append((from_node_idx, child_2_idx, selected_j, 1))
        if self.verbose == True:
            print(f"parent {from_node_idx} used {pn.used_features_current_path} childs {child_1_idx} {child_2_idx} number of relations {len(self.relations)} depth {pn.current_depth}")

    def evaluate_split_by_feature_in_tree_node(self, tn: TreeNode, j):
        '''
        - tn TreeNode from which we split
        - j index of feature that we evaluate - must be categorical/binary feature 1 or 0
        '''
        next_tn_0 = TreeNode(X=self.X, Y=self.Y)
        next_tn_1 = TreeNode(X=self.X, Y=self.Y)

        self.help_split(pn=tn, cn1=next_tn_0, cn2=next_tn_1, j=j)

        next_tn_0.calculate_node_values()
        next_tn_1.calculate_node_values()
        agg_correctcount = next_tn_0.correctcount + next_tn_1.correctcount
        agg_errorcount = next_tn_0.errorcount + next_tn_1.errorcount
        agg_accuracy = agg_correctcount / (agg_correctcount + agg_errorcount)
        return agg_accuracy, agg_correctcount, agg_errorcount

    def select_feature_for_split_from_tree_node(self, tn: TreeNode):
        '''
        - tn TreeNode from which we split
        '''
        if len(tn.i) == 0:
            tn.best_j_from_node = 0
            return
        best_accuracy_so_far = 0.0
        best_j_so_far = -1
        D = self.X.shape[1]
        for j in range(D):
            if j not in tn.used_features_current_path:
                agg_accuracy, agg_correctcount, agg_errorcount = self.evaluate_split_by_feature_in_tree_node(tn=tn,
                                                                                                    j=j)
                if agg_accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = agg_accuracy
                    best_j_so_far = j
        tn.best_j_from_node = best_j_so_far
        if best_j_so_far == -1:
            tn.is_leaf = True

    def find_next_node_idx(self):
        for idx in range(len(self.nodes)):
            if self.nodes[idx].best_j_from_node == -1 and self.nodes[idx].is_leaf == False:
                return idx
        return -1

    def fit(self):
        self.initialize_tree()

        fit_completed = False
        while fit_completed == False:
            next_node_idx = self.find_next_node_idx()
            if next_node_idx == -1:
                fit_completed = True
            else:
                self.select_feature_for_split_from_tree_node(tn=self.nodes[next_node_idx])
                if self.nodes[next_node_idx].is_leaf == False:
                    self.do_split(from_node_idx=next_node_idx,
                                selected_j=self.nodes[next_node_idx].best_j_from_node)
