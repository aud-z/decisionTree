# -*- coding: utf-8 -*-
"""
Intro to ML 
HW2
Audrey Zhang
Andrew ID: youyouz
"""

import sys
import numpy as np 

#%%

class Node():
    def __init__(self, columns = None, labels = None, level = 0, _prev_attrs = ()):

        """
        self.left is left child of the node
        self.right is right child of the node 
        self.attr is the attribute that the node splits on 
        self.label is the class label of the node, only populated for leaf nodes
        self.leftkey is the value of the attribute that belongs to the left node 
        self.level is the current depth of the tree at this node 
        self.columns is the column headers belonging to the attributes
        """
        self.left = None
        self.right = None 
        self.attr = None
        self.label = None
        self.leftkey = None
        self.level = level 
        self.prev_attrs = _prev_attrs
        self.columns = columns
        self.labels = labels
        
    def get_label(self, data):
        """
        Returns class labels using majority vote
        """ 
        # get counts of classes in the data
        unique, counts = np.unique(data[:, -1], return_counts = True)

        # if evenly split classes
        # label with the class that comes last in lexicographical order
        if len(counts)>1 and counts[0] == counts[1]:
                unique.sort() 
                leaf_class = unique[-1]  
            
        # otherwise return the majority class label
        else:
            leaf_class = unique[np.argmax(counts)]  
        
        return leaf_class 
    
    def pretty_print(self, data): 
        """
        Helper function to print the current count of labels at each level
        """
        counts = []
        for i in self.labels:
            counts.append(str(np.count_nonzero(data[:, -1] == i)) + ' ' + i)
        print('[' + '/ '.join(counts) + ']')

    def grow_tree(self, train, depth): 
        
        """
        Growing the tree during training phase
        Called recursively to create left and right child nodes 
        """
        
        # first calculate entropy for this dataset 
        entropy = self.calc_entropy(train) 
        
        # print the label counts at the root 
        if self.level == 0:
            self.labels = np.unique(train[:, -1])
            self.pretty_print(train)
        
        # if current entropy is already 0, training stops 
        if entropy == 0: 
            self.label = self.get_label(train) 
            return None
        
        # if all attributes have been used, training stops 
        elif len(self.prev_attrs) == len(train[0])-1:
            self.label = self.get_label(train) 
            return None
        
        # if max depth has been reached, training stops
        elif self.level == depth: 
            self.label = self.get_label(train) 
            return None

        
        # else if entropy is not zero and not all attributes have been used and 
        # we have not yet reached the max depth 
        # calculate information gain and pick highest for attribute to split on 

        else: 
            # note: if two attributes have equal info gain, choose the first attribute to break ties 
            # numpy argmax function returns the first instance of the maximum value 
            infoGain = self.calc_info_gain(train, entropy)
            self.attr = np.argmax(infoGain)
            attrs = self.prev_attrs
            attrs += (self.attr, )
            
            # get unique values associated with this attribute 
            values = np.unique(train[:, self.attr])
            
            # pick arbitrary value from binary class to create a filter 
            v = values[0]  
            filt = train[:, self.attr] == v
            self.leftkey = v
            
            # grow the tree
            # tree depth increases by 1 
            self.level += 1
            
            # left node will be created first 
            print('| ' * self.level, self.columns[self.attr], ' = ', values[0], ': ', end = '')
            self.pretty_print(train[filt==True])

            self.left = Node(self.columns, self.labels, self.level,  _prev_attrs = attrs) 
            self.left.grow_tree(train[filt==True],  depth) 
            
            if len(values)>1: 
                print('| ' * self.level, self.columns[self.attr], ' = ', values[1], ': ', end = '')
                self.pretty_print(train[filt==False])
    
                self.right = Node(self.columns, self.labels, self.level, _prev_attrs = attrs)
                self.right.grow_tree(train[filt==False],  depth) 
        
    def predict(self, data, _indices = None, _predictions = None): 
        
        """
        Predicts labels for a dataset based on the trained decision Tree 
        output:
            array of labels associated with observations in the dataset 
        """
        
        # for the initial call, set up full list of indices 
        # and empty list of predictions 
        # the list of indices will be used to track which labels get updated at each leaf node 
        
        if _indices is None and _predictions is None:
            indices = np.arange(len(data)) 
            predictions = np.empty((len(data), ), dtype = 'object')
        
        else: 
            indices = _indices
            predictions = _predictions
                
        if self.label is not None: 

            predictions[indices] = self.label
            return predictions

        else: 
            if self.left is not None: 
                l_subset = np.where(data[:, self.attr] == self.leftkey)[0]  
                predictions = self.left.predict(data[l_subset], _indices = indices[l_subset], _predictions = predictions) 
            
            if self.right is not None:
                r_subset = np.where(data[:, self.attr] != self.leftkey)[0] 
                predictions = self.right.predict(data[r_subset], _indices = indices[r_subset], _predictions = predictions) 
            
            return predictions 
    
    def calc_info_gain(self, data, entropy): 
        
        """
        Calculates information gain from each of the attributes available in the data
        Returns list of I(Y; X) values corresponding to the indices of the attribute columns
        """ 
        
        infoGain = [0 for r in range(len(data[0])-1)]
    
        for i in range(len(data[0])-1):
            if i not in self.prev_attrs:
                h_Y_x = 0
                values = np.unique(data[:, i]) 
                
                # pick arbitrary value from binary class to create a filter 
                v = values[0] 
                filt = data[:, i] == v
                
                # info gain is H(Y) - H(Y|X) 
                # first calculate entropy: H(Y|X) 
                h_Y_x += len(data[filt==True])/len(data) * self.calc_entropy(data[filt==True])
                h_Y_x += len(data[filt==False])/len(data) * self.calc_entropy(data[filt==False]) 
        
                gain = entropy - h_Y_x 
                infoGain[i] = gain       
                
        return infoGain
                
    def calc_entropy(self, data): 
        """ calculates label entropy at root before any splits """
        classes = np.unique(data[:, -1]) 
        if len(classes) > 1:
            h_Y = 0
            # arbitarily pick a class as positive
            pos = classes[0]
            p_pos = np.count_nonzero(data[:, -1] == pos) / len(data)
            p_neg = 1 - p_pos 
            for p in (p_pos, p_neg): 
                h_Y += -p * np.log2(p)
            return h_Y 
        else: 
            return 0 
        
    
    def calc_error(self, predictions, data):
        """ calculates error rate """ 
        
        err_rate = 1 - (np.count_nonzero(data[:, -1] == predictions) /data[:, -1].shape[0])
        return err_rate

    def train(self, train, train_out, depth):
        
        """
        Trains decision tree by calling the grow_tree function
        output:
            predictions: array of predicted values associated with each observation
            train_error: training error based on trained decision tree
        """
        
        self.grow_tree(train, depth) 
        predictions = self.predict(train)
        train_error = self.calc_error(predictions, train) 
        
        with open(train_out, 'w') as output:
            output.writelines(p + '\n' for p in predictions) 
        
        return predictions, train_error 
    
    def test(self, test, test_out): 
        
        """
        Uses a test dataset to create labels based on trained decision tree
        output:
            predictions: array of predicted values associated with each observation
            test_error: training error based on trained decision tree
        """
        
        predictions = self.predict(test)
        test_error = self.calc_error(predictions, test)
        
        with open(test_out, 'w') as output:
            output.writelines(p + '\n' for p in predictions)        
        
        return predictions, test_error 
    
    def train_predict(self, train_ds, test_ds, depth, train_out, test_out, metrics): 
        
        """
        Combined train_predict function that calls train() and test() in one function call
        outputs:
            train_pred: array of predicted values associated with the training dataset
            train_err: training error 
            test_pred: array of predicted values associated with the test dataset
            test_err: testing error 
        """
        
        # remove first row of both train/test datasets as they contain the attribute names
        self.columns = train_ds[0] 
        
        train_ds = train_ds[1: ]
        test_ds = test_ds[1: ] 
        
        # add unique labels to base 
        self.labels = np.unique(train_ds[:, -1]) 

        train_pred, train_err = self.train(train_ds, train_out, depth) 
        test_pred, test_err = self.test(test_ds, test_out) 
        
        with open(metrics, 'w') as output:
            output.write("error(train): ")
            output.write(str(train_err))
            output.write("\nerror(test): ")
            output.write(str(test_err)) 
        
        return train_pred, train_err, test_pred, test_err
                    

#%%

# sysargs: <train input> <test input> <max depth> <train out> <test out> <metrics out>

if __name__ == '__main__':
    
    train_in = sys.argv[1]
    test_in = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    
    train_ds= [] 
    
    with open(train_in, 'r') as file:
        for f in file:
            train_ds.append(f.strip('\n').split('\t'))
            
    train_ds = np.array(train_ds) 

        
    test_ds= [] 
    
    with open(test_in, 'r') as file:
        for f in file:
            test_ds.append(f.strip('\n').split('\t'))
            
    test_ds = np.array(test_ds) 
    
    DT = Node() 
    DT.train_predict(train_ds, test_ds, max_depth, train_out, test_out, metrics_out)
        
        

    
    
    
    
        
