import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import OrderedDict
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

def plot(features,labels,path,export):
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
    if export:
        print ('path', path)
        plt.savefig(path)

def initPlot(layer="fc2",size=10,name="new"):
    y_test=np.load("./features/labels.npy")
    labels=np.argmax(y_test,1)
    feat=[]
    if name.startswith("vgg"):
        for i in range(0,10000,1000):
            feat.append(np.load("./features/"+layer+"/"+name+str(i)+".npy"))
        feat=np.vstack(feat)
        print(feat.shape)
    else:
        feat.append(np.load('./features/'+layer+'.npy'))
    # if len(feat.shape)>2:
    #     feat=feat.reshape((feat.shape[1],feat.shape[2]))
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    # w=tsne.fit_transform(feat[:size,:])
    # labels=np.argmax(y_test[:size],1)

    if name.startswith("vgg"):
        step=1000
        count=1
        for i in xrange(0,feat.shape[0],step):
            w=tsne.fit_transform(feat[i:i+step])
            if count*step==feat.shape[0]:
                print ("in")
                plot(w,labels[i:i+step],"./plots/"+layer+name+".png",True)
            else:
                plot(w,labels[i:i+step],"./plots/"+layer+name+".png",False)
            print("count: ",count)
            count+=1
    else:
        w=tsne.fit_transform(feat[0])
        print("w",w.shape)
        plot(w,labels,"./plots/"+layer+name+".png",True)
