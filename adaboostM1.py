import pandas as pd
import numpy as np

#calculate the entropy of a dataset
def calcEntropy(dataset):
    classes = set(dataset[:,-1])
    entropy = 0
    for c in classes:
        temp = 0
        for row in range(len(dataset)):
            if dataset[row,-1] == c :
                temp += 1
        entropy -= (temp/len(dataset))*np.log2(temp/len(dataset))
    return entropy

#split the dataset based on a value of a feature and return the new datasets
def dataSplit(dataset,value,column):
    left = dataset[ dataset[:,column] < value ]
    right = dataset[ dataset[:,column] >= value ]
    return left,right

#find the best split for a dataset
def bestSplit(dataset):
    number_of_cols = len(dataset[0,:]) - 1
    value,column,ent = 0,0,999999
    for i in range(number_of_cols):
        values = set(dataset[:,i])
        for v in values:
            #get the left and right dataset on the possible split
            left,right = dataSplit(dataset,v,i)
            #calculate the entropy of the split and check if it is decreased
            new_ent = (len(left)/len(dataset))*calcEntropy(left) + (len(right)/len(dataset))*calcEntropy(right)
            if new_ent < ent :
                value = v
                column = i
                ent = new_ent
    #return the value and the feature that achieved the best split
    return value,column

#create a left and a right child on a node
def splitNode(node):
    dataset = node.getDataset()
    value,column = bestSplit(dataset)
    left,right = dataSplit(dataset,value,column)
    #check if a valid split was found
    if left.size != 0 and right.size != 0:
        node.addLeft(left)
        node.addRight(right)
    #update the node about the value and the feature of the split
    node.setValue(value)
    node.setColumn(column)

class Node:
    def __init__(self,dataset):
        self.dataset = dataset
        self.entropy = calcEntropy(dataset)
        self.right = None
        self.left = None
        self.column = 0
        self.value = 0
        #calculate the dominant class of the node
        c,n = np.unique(dataset[:,-1],return_counts=True)
        index = 0
        max_val = n[0]
        for i in range(1,len(c)):
            if n[i] > max_val :
                max_val = n[i]
                index = i
        #set the dominant class as the class of the node
        self.cl = c[index]

    def addLeft(self,dataset):
        self.left = Node(dataset)

    def addRight(self,dataset):
        self.right = Node(dataset) 
    def setValue(self,value):
        self.value = value
    
    def setColumn(self,column):
        self.column = column

    def setDepth(self,depth):
        self.depth = depth

    def getDepth(self):
        return self.depth

    def getValue(self):
        return self.value
    
    def getColumn(self):
        return self.column

    def getEntropy(self):
        return self.entropy

    def getDataset(self):
        return self.dataset

    def getLeft(self):
        return self.left

    def getRight(self):
        return self.right

    def getClass(self):
        return self.cl

#function to build the decision tree
def buildTree(node,depth,min_samples_leaf):
    root = node
    #check the depth of the tree and the sample size of the node 
    if root.getDepth() < depth and len(root.getDataset()) > min_samples_leaf :
        splitNode(root)
        left = root.getLeft()
        right = root.getRight()
        #check if left and right child exists
        if left != None :
            left.setDepth(root.getDepth() + 1)
            buildTree(left,depth,min_samples_leaf)
        if right != None :
            right.setDepth(root.getDepth() + 1)
            buildTree(right,depth,min_samples_leaf)

    return root

#function that predicts a class for a given instance 
def predict(node,example):
    #if this is a final node return the class of it
    if node.getLeft() == None :
        return node.getClass()
    #if it's not a final node move to the next node based on the value of the feature in the current node 
    if example[node.getColumn()] < node.getValue() :
        return predict(node.getLeft(),example)
    else:
        return predict(node.getRight(),example)

#function for predictions on a dataset
def datasetPredictions(node,dataset):
    predictions = []
    for data in dataset:
        predictions.append(predict(node,data[:-1]))
    return predictions

#function to get the final decision tree with the depth set
def decisionTree(dataset,depth=1,min_samples_leaf=5):
    root = Node(dataset)
    root.setDepth(0)
    return buildTree(root,depth,min_samples_leaf)

#function to split the dataset on train and test set
def train_test_split(dataset,test_size):
    indices = np.random.permutation(len(dataset))
    threshold = int(test_size*len(dataset))
    test_indices = indices[:threshold]
    train_indices = indices[threshold:]
    return dataset[train_indices],dataset[test_indices]

#function for handling the discrete values of features
def transform_discrete_values(dataset):
    cat_attribs = []
    dataset.rename(columns={list(dataset.columns)[-1]:'target'},inplace=True)
    #find all the features with discrete values
    for attrib in dataset.columns:
        if dataset[attrib].dtypes == 'object':
            cat_attribs.append(attrib)
    #convert the non numerical values to numerical
    dataset = pd.get_dummies(dataset,columns=cat_attribs)
    cols = list(dataset.columns)
    value = cols.pop(cols.index('target'))
    cols.append(value)
    dataset = dataset[cols]
    return dataset

#function for weighted boostraping
def weightedBootstraping(dataset,weights):
    size = len(dataset)
    sample = []
    ws = np.array(weights)
    #do the weighted sampling 
    while len(sample) < size:
        #shuffle the dataset to increase randomization in the sample 
        indices = np.random.permutation(size)
        temp_d = dataset[indices].copy()
        temp_w = ws[indices].copy()
        sample.append(temp_d[np.random.choice(indices,p=temp_w)])
    return np.array(sample)

def adaboostM1(dataset,n_estimators):
    #instaces weights
    i_weights = []
    #predictors weights
    p_weights = []
    classifiers = []
    #initialize weights
    for i in range(len(dataset)):
        i_weights.append(1/len(dataset))
    #calculate the predictors
    for i in range(n_estimators):
        sample = weightedBootstraping(dataset,i_weights)
        tree = decisionTree(sample,1)
        pred = datasetPredictions(tree,sample)
        error = 0
        for j in range(len(dataset)):
            #increase the error if the prediction was wrong
            if pred[j] != sample[j,-1]:
                error += i_weights[j]
        #normalize the error
        error /= sum(i_weights)
        #calculate the predictor's weight
        a = np.log((1-error)/error)
        #update the weights of the instances
        for j in range(len(sample)):
            if pred[j] != sample[j,-1]:
                i_weights[j] *= np.exp(a)
        #normalize the new weights
        s = sum(i_weights)
        for j in range(len(i_weights)):
            i_weights[j] /= s
        classifiers.append(tree)
        p_weights.append(a)
    return classifiers,p_weights

#function for adaboost predictions
def ensemblePredict(dataset,classifiers,p_weights):
    predictions = []
    for data in dataset:
        classes = { 0 : 0, 1 : 0 }
        for i in range(len(classifiers)):
            pred = predict(classifiers[i],data)
            #increase the class of the prediction by the weight of the predictor
            if pred == 1:
                classes[1] += p_weights[i]
            if pred == 0:
                classes[0] += p_weights[i]
        #check what class has the biggest
        if classes[0] > classes[1]:
            predictions.append(0)
        if classes[0] <= classes[1]:
            predictions.append(1)
    return np.array(predictions)

#adaboostM1 model function
def main(csv_file,tain_set,n_estimators):
    dataset = pd.read_csv(csv_file)
    dataset = transform_discrete_values(dataset)
    dataset_values = dataset.values
    classifiers,p_weights = adaboostM1(train_set,n_estimators)
    p = ensemblePredict(dataset_values,classifiers,p_weights)
    correct = sum(p == dataset_values[:,-1])
    print(correct/len(p))
    f = open('output.txt','w')
    for i in p:
        f.write(str(i))
    f.close()
    
#example of the algorithm
if __name__ == "__main__" :
    #get the data
    dataset = pd.read_csv('heart.csv')
    #convert the non numerical values to numerical
    dataset = transform_discrete_values(dataset)
    #get a numpy array with the values
    dataset_values = dataset.values
    #split the dataset to train and test set
    train_set,test_set = train_test_split(dataset_values,0.2)
    #train the adaboostM1 model on the training set for 40 estimators
    classifiers,p_weights = adaboostM1(train_set,60)
    #use the model to predict the test set
    p = ensemblePredict(test_set,classifiers,p_weights)
    correct = sum(p == test_set[:,-1])
    print('ratio =',correct/len(p))
    #usage of the main 
    main('heart.csv',dataset_values,130)

