import numpy as np

class Tree_node:
    """
    Data structure for nodes in the decision-tree
    """
    def __init__(self,):
        self.is_leaf = False # whether or not the current node is a leaf node
        self.feature = None # index of the selected feature (for non-leaf node)
        self.label = -1 # class label (for leaf node)
        self.left_child = None # left child node
        self.right_child = None # right child node

class Decision_tree:
    """
    Decision tree with binary features
    """
    def __init__(self,min_entropy):
        self.min_entropy = min_entropy
        self.root = None

    def fit(self,train_x,train_y):
        # construct the decision-tree with recursion
        # print(train_x[0].shape)
        # print('123')
        self.root = self.generate_tree(train_x,train_y)

    def predict(self,test_x):
        # iterate through all samples
        prediction = np.zeros([len(test_x),]).astype('int') # placeholder
        for i in range(len(test_x)):
            # traverse the decision-tree based on the features of the current sample
            # pass # placeholder
            cur_node = self.root
            
            while cur_node.left_child != None or cur_node.right_child != None: 
                if test_x[i][cur_node.feature] == 0:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child
            
            prediction[i] = cur_node.label
            
        return prediction

    def generate_tree(self,data,label):
        # initialize the current tree node
        
        cur_node = Tree_node()

        # compute the node entropy
        node_entropy = self.compute_node_entropy(label)

        # print(node_entropy)
        # determine if the current node is a leaf node
        # print(self.min_entropy)
        if node_entropy < self.min_entropy:
            # determine the class label for leaf node
            classes, counts = np.unique(label, return_counts =True)
            major_class = np.argmax(counts)
            cur_node.label = classes[major_class]
            
            return cur_node
        
        # print('here 2')
        # select the feature that will best split the current non-leaf node
        selected_feature = self.select_feature(data,label)
        cur_node.feature = selected_feature
        
        # split the data based on the selected feature and start the next level of recursion
        # temporary storage for data after split at selected_feature
        left_child_data, left_child_label = [], []
        right_child_data, right_child_label = [], []
        
        # print(left_child_label)
        
        for i in range(len(data)):
            if data[i][selected_feature] == 0:
                left_child_data.append(data[i])
                left_child_label.append(label[i])
            else:
                right_child_data.append(data[i])
                right_child_label.append(label[i])
        
        left_child_data = np.asarray(left_child_data)
        left_child_label = np.asarray(left_child_label)
        right_child_data = np.asarray(right_child_data)
        right_child_label = np.asarray(right_child_label)
        
        cur_node.left_child = self.generate_tree(left_child_data,left_child_label)
        cur_node.right_child = self.generate_tree(right_child_data, right_child_label)
        # print('123')
        return cur_node

    def select_feature(self,data,label):
        # iterate through all features and compute their corresponding entropy
        best_feat = 0
        smallest_entropy = 100 

        for i in range(len(data[0])):

            # compute the entropy of splitting based on the selected features
            left_y = []
            right_y = []  # store temporary left y and right y lables
            for j in range(len(data)):
                if data[j][i] == 0:
                    left_y.append(label[j])
                else:
                    right_y.append(label[j])
            node_split_entropy = self.compute_split_entropy(left_y,right_y) 
            
            # pass
            
            # select the feature with minimum entropy
            if node_split_entropy < smallest_entropy:
                smallest_entropy = node_split_entropy
                best_feat = i
            
        return best_feat

    def compute_split_entropy(self,left_y,right_y):
        # compute the entropy of a potential split, left_y and right_y are labels for the two branches
        split_entropy = -1 # placeholders
        left_y_count = len(left_y)
        right_y_count = len(right_y)
        total_count = left_y_count + right_y_count
        
        left_entropy = self.compute_node_entropy(left_y)
        right_entropy = self.compute_node_entropy(right_y)
        
        split_entropy = (left_y_count/total_count)*left_entropy + (right_y_count/total_count)*right_entropy
        return split_entropy

    def compute_node_entropy(self,label):
        # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
        node_entropy = -1 # placeholder
        classes, counts = np.unique(label, return_counts =True)
        total_count =  np.sum(counts)
        node_entropy = 0 
        for i in range(len(classes)):
            node_entropy += (-counts[i]/total_count)*np.log2((counts[i]/total_count) + 1e-15)
        
        return node_entropy
