import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import OrderedDict
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

def plot(features,labels,path):
    items = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
    classes = ['','','','','','','','','','']
    plt.figure(figsize=(10, 10))
    cmap = plt.get_cmap('gist_ncar')
    colors = [cmap(i) for i in np.linspace(0, 1,10)]
    for i,label in enumerate(labels):
        x,y=features[i,:]
        plt.scatter(x,y,color=colors[labels[i]],label=items[labels[i]])
        plt.annotate(classes[label],xy=(x,y),xytext=(5,2),\
        textcoords="offset points",ha='right',va='bottom')
    
    handles,labels=plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    # plt.show()
    plt.savefig(path)

def initPlot(layer="fc2",size=10,name="new"):
    y_test=np.load("./features/labels.npy")
    feat=np.load("./features/"+layer+".npy")
    print(feat.shape)
    if len(feat.shape)>2:
        feat=feat.reshape((feat.shape[1],feat.shape[2]))
    
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    w=tsne.fit_transform(feat[:size,:])
    labels=np.argmax(y_test[:size],1)
    print(w.shape)
    plot(w,labels,"./plots/"+layer+name+".png")
