import pylab
from matplotlib import gridspec
from sklearn.datasets import make_classification
import numpy as np 
from ipywidgets import  interact,interactive,fixed
import pickle
import os
import gzip

n=50
X,Y=make_classification(n_samples=n,n_features=2,n_redundant=0,n_informative=2,flip_y=0)
Y=Y*2-1
X=X.astype(np.float32);Y=Y.astype(np.int32);

train_x,text_x=np.split(X,[n*8//10])
train_label,text_label=np.split(Y,[n*8//10])

print("Features :\n ",train_x)
print("Labels :\n",train_label)


def  plot_dataset(suptitle,features,labels):

    fig,ax=pylab.subplots(1,1)
    fig.suptitle(suptitle,fontsize=16)
    ax.set_xlabel("$x_i[0]$-- (feature 1)")
    ax.set_ylabel("$x_i[1]$-- (feature 2)")

    colors=['r' if l>0 else 'b' for l in labels]
    ax.scatter(features[:,0],features[:,1],marker='o',c=colors,s=100,alpha=0.5)
    fig.show()
plot_dataset("Training Dataset",train_x,train_label)
pylab.show()

pos_example=np.array([[t[0],t[1],1] for i,t in enumerate(train_x) if train_label[i]>0])
neg_example=np.array([[t[0],t[1],1] for i,t in enumerate(train_x) if train_label[i]<0])
print(pos_example[0:3])

def train(positive_examples, negative_examples, num_iterations = 100, learning_rate = 0.01):
    num_dims = positive_examples.shape[1]
    
    # Initialize weights. 
    # We initialize with 0 for simplicity, but random initialization is also a good idea
    weights = np.zeros((num_dims,1)) 
    
    pos_count = positive_examples.shape[0]
    neg_count = negative_examples.shape[0]
    
    report_frequency = 10
    
    for i in range(num_iterations):
        # Pick one positive and one negative example
        pos = random.choice(positive_examples)
        neg = random.choice(negative_examples)

        z = np.dot(pos, weights)   
        if z < 0: # positive example was classified as negative
            weights = weights + learning_rate * pos.reshape(weights.shape)

        z  = np.dot(neg, weights)
        if z >= 0: # negative example was classified as positive
            weights = weights - learning_rate * neg.reshape(weights.shape)
            
        # Periodically, print out the current accuracy on all examples 
        if i % report_frequency == 0:             
            pos_out = np.dot(positive_examples, weights)
            neg_out = np.dot(negative_examples, weights)        
            pos_correct = (pos_out >= 0).sum() / float(pos_count)
            neg_correct = (neg_out < 0).sum() / float(neg_count)
            print("Iteration={}, pos correct={}, neg correct={}".format(i,pos_correct,neg_correct))

    return weights
