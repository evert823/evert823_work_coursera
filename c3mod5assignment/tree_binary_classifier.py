'''
Tree for DecisionTree binary classifier
Expected:
- X NxD matrix (features) can only contain binary/categorical features with values 0 or 1
- Y Nx1 matrix (target class known value from training data)
'''
import numpy as np
from tree_node import TreeNode

class TreeBinaryClassifier:
    def __init__(self):
        self.verbose = False
        #We enforce that we have read only access to X and Y from this class

        self.nodes = []
        self.relations : list[tuple[int, int, int, int]] = []
        self.max_depth = 30
        '''
        Relation will be stored as tuple
        - parent_node_idx (refers to self.nodes)
        - child_node_idx (refers to self.nodes)
        - j (j in range(D) where D equals X.shape[1])
        - feature_value (should be 0 or 1 and indicates the branch of the child_node)
        '''

    def initialize_tree(self, X, Y):
        '''
        We create the root node and calculate its values
        '''
        self.nodes.clear()
        self.relations.clear()
        self.nodes.append(TreeNode())
        N = X.shape[0]
        self.nodes[0].i = [idx for idx in range(N)]
        self.nodes[0].calculate_node_values(Y=Y)
        self.nodes[0].current_depth = 0

    def help_split(self, X, pn, cn1, cn2, j):
        for row_idx in pn.i:
            fv = X[row_idx, j]
            if fv == 0:
                cn1.i.append(row_idx)
            elif fv == 1:
                cn2.i.append(row_idx)
            else:
                raise Exception(f"fv {fv} expected to be 0 or 1")

    def do_split(self, X, Y, from_node_idx, selected_j):
        '''
        Before calling this method, we have found the best j for next split
        Then we create two child nodes for branches for values 0 and 1
        Then we fill the row-indices that belong to the child nodes
        Then we create the relations between the parent and each of the new child nodes
        '''
        child_1_idx = len(self.nodes)
        child_2_idx = child_1_idx + 1
        self.nodes.append(TreeNode())
        self.nodes.append(TreeNode())

        #Now in self.nodes[from_node_idx] we identify which i are mapped to feature value 0 and which to 1
        pn = self.nodes[from_node_idx]
        c1 = self.nodes[child_1_idx]
        c2 = self.nodes[child_2_idx]
        self.help_split(X=X, pn=pn, cn1=c1, cn2=c2, j=selected_j)

        c1.calculate_node_values(Y=Y)
        c2.calculate_node_values(Y=Y)
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

    def evaluate_split_by_feature_in_tree_node(self, X, Y, tn: TreeNode, j):
        '''
        - tn TreeNode from which we split
        - j index of feature that we evaluate - must be categorical/binary feature 1 or 0
        '''
        next_tn_0 = TreeNode()
        next_tn_1 = TreeNode()

        self.help_split(X=X, pn=tn, cn1=next_tn_0, cn2=next_tn_1, j=j)

        next_tn_0.calculate_node_values(Y=Y)
        next_tn_1.calculate_node_values(Y=Y)
        agg_correctcount = next_tn_0.correctcount + next_tn_1.correctcount
        agg_errorcount = next_tn_0.errorcount + next_tn_1.errorcount
        agg_accuracy = agg_correctcount / (agg_correctcount + agg_errorcount)
        return agg_accuracy, agg_correctcount, agg_errorcount

    def select_feature_for_split_from_tree_node(self, X, Y, tn: TreeNode):
        '''
        - tn TreeNode from which we split
        '''
        if len(tn.i) == 0:
            tn.best_j_from_node = 0
            return
        best_accuracy_so_far = 0.0
        best_j_so_far = -1
        D = X.shape[1]
        for j in range(D):
            if j not in tn.used_features_current_path:
                agg_accuracy, agg_correctcount, agg_errorcount = self.evaluate_split_by_feature_in_tree_node(X=X,
                                                Y=Y,
                                                tn=tn,
                                                j=j)
                if agg_accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = agg_accuracy
                    best_j_so_far = j
        tn.best_j_from_node = best_j_so_far
        if best_j_so_far == -1:
            tn.is_leaf = True
        '''
        Here we could apply a threshold:
        if accuracy already reached in this node is not improved significantly then let this node be a leaf
        AND:
        if this node as number of data points below another thresholds then let this node be a leaf
        '''

    def find_next_node_idx(self):
        for idx in range(len(self.nodes)):
            if self.nodes[idx].best_j_from_node == -1 and self.nodes[idx].is_leaf == False:
                return idx
        return -1

    def fit(self, X, Y):
        self.initialize_tree(X=X, Y=Y)

        fit_completed = False
        while fit_completed == False:
            next_node_idx = self.find_next_node_idx()
            if next_node_idx == -1:
                fit_completed = True
            else:
                self.select_feature_for_split_from_tree_node(X=X, Y=Y, tn=self.nodes[next_node_idx])
                if self.nodes[next_node_idx].is_leaf == False:
                    self.do_split(X=X, Y=Y, from_node_idx=next_node_idx,
                                selected_j=self.nodes[next_node_idx].best_j_from_node)

    def predict_at_node(self, X, i, node_idx, preceding_path):
        '''
        For given feature matrix X observation index i and node
        we find and return predicted class and probability for it
        '''
        if self.nodes[node_idx].is_leaf == True:
            outpath = f"{preceding_path}|leaf"
            return self.nodes[node_idx].majority_class, self.nodes[node_idx].majority_probability, outpath
        else:
            for ri in range(len(self.relations)):
                if self.relations[ri][0] == node_idx:
                    child_idx = self.relations[ri][1]
                    j = self.relations[ri][2]
                    fv = self.relations[ri][3]
                    if X[i, j] == fv:
                        cv, prob, recursionpath = self.predict_at_node(X=X, i=i, node_idx=child_idx, preceding_path="")
                        pathpart = f"{j}.{fv}"
                        outpath = f"{preceding_path}|{pathpart}|{recursionpath}"
                        return cv, prob, outpath
        raise Exception(f"Not able to traverse tree based on observation {i}")


    def predict(self, X):
        '''
        Based on input features X we predict output values by traversing the tree
        The tree must already be built using the fit method before calling predict
        Returns an Nx2 matrix where:
        - Column 0: predicted class for each row
        - Column 1: probability of that prediction for each row
        '''
        N = X.shape[0]
        predictions = np.zeros((N, 3), dtype=object)
        root_idx = 0
        for i in range(N):
            cv, prob, path = self.predict_at_node(X=X, i=i, node_idx=root_idx, preceding_path="")
            predictions[i, 0] = cv
            predictions[i, 1] = prob
            predictions[i, 2] = path
        return predictions

    def accuracy_on_dataset(self, X, Y):
        '''
        Here we produce predictions based on X and then compare the predictions to know output values Y
        '''
        N = X.shape[0]
        if N == 0:
            return None
        predictions = self.predict(X=X)
        correctcount = 0
        for i in range(N):
            if predictions[i, 0] == Y[i]:
                correctcount += 1
        return correctcount / N

    def to_graphviz_dot(self):
        '''
        Convert the tree to GraphViz DOT format for visualization
        Returns a string in DOT language that can be used to generate a tree diagram
        '''
        dot_lines = ["digraph DecisionTree {"]
        dot_lines.append("    node [shape=box];")
        
        # Add nodes
        for idx, node in enumerate(self.nodes):
            label = f"Node {idx}\n"
            if node.is_leaf:
                label += f"Class: {node.majority_class}\n"
                if node.majority_probability is not None:
                    label += f"Prob: {node.majority_probability:.2f}"
                else:
                    label += "Prob: N/A"
            else:
                label += f"Best feature j: {node.best_j_from_node}\n"
                label += f"Samples: {len(node.i)}"
            
            dot_lines.append(f'    {idx} [label="{label}"];')
        
        # Add edges
        for parent_idx, child_idx, feature_j, feature_value in self.relations:
            label = f"X[{feature_j}]=={feature_value}"
            dot_lines.append(f'    {parent_idx} -> {child_idx} [label="{label}"];')
        
        dot_lines.append("}")
        return "\n".join(dot_lines)

    def save_tree_as_dot(self, filename="tree.dot"):
        '''
        Save the tree as a GraphViz DOT file
        '''
        dot_content = self.to_graphviz_dot()
        with open(filename, 'w') as f:
            f.write(dot_content)
        print(f"Tree saved to {filename}")
