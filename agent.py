# agent.py 
import os
import pickle
import numpy as np
import pandas as pd
from stellargraph import StellarGraph 
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import FullBatchLinkGenerator
from stellargraph.mapper import GraphSAGELinkGenerator

from tensorflow import keras
from stellargraph import globalvar

class Agent:
    def __init__(self,method):
        
        self.DATA_PATH=os.path.join(os.getcwd(),'DATA/')
        self.SAVE_PATH=os.path.join(os.getcwd(),'saved_model/')
        self.Result_PATH=os.path.join(os.getcwd(),'Result/')
        
        
        self.edges_data=pd.read_csv(self.DATA_PATH+'sample.csv')
        self.node_feature=pd.read_csv(self.DATA_PATH+'sample_feature.csv')
        self.node_features = self.node_feature.set_index(keys=['index'],
                                                         inplace=False,
                                                         drop=True)
        self.G = StellarGraph({"feature": self.node_features}, 
                              {"edge": self.edges_data})
        self.edge_splitter_test = EdgeSplitter(self.G)
        self.G_test,_,_=self.edge_splitter_test.train_test_split(p=0.1,
                                                            method="global",
                                                            keep_connected=True)
        

        if method=='gcn':
            self.method='gcn'
            self.test_gen = FullBatchLinkGenerator(self.G_test,
                                                   method="gcn")
            self.model=keras.models.load_model(self.SAVE_PATH+'gcn.model')

        elif method=='sage':
            self.method='sage'
            self.test_gen = GraphSAGELinkGenerator(self.G_test,
                                                   batch_size=32,
                                                   num_samples=[10,10])
            self.model=keras.models.load_model(self.SAVE_PATH+'sage.model')
        
        elif method=='at2v':
            self.method='at2v'
            with open(self.Result_PATH+'AT2v_embedding2.pkl','rb')as f:
                self.embeddingvec=pickle.load(f)
            with open(self.Result_PATH+'clf_at2v.pkl','rb')as f:
                self.classifier=pickle.load(f)

        else:
            self.test_gen=None
            self.model=None

    def __call__(self,index_list):
        if self.method=='gcn':
            self.indexans=np.random.randint(0,1,index_list.shape[0])
            x = self.test_gen.flow(index_list, self.indexans)
            self.output=self.model.predict(x)

        elif self.method=='sage':
            self.indexans=np.random.randint(0,1,index_list.shape[0])
            x = self.test_gen.flow(index_list, self.indexans)
            self.output=self.model.predict(x)
            self.output=self.output.reshape(-1,1)

        elif self.method=='at2v':
            self.input=(self.embeddingvec[index_list[:,0]-1]-
                        self.embeddingvec[index_list[:,1]-1])**2
            self.output=self.classifier.predict_proba(self.input)[:,1]

        else:
            print('no model')
            
        high_prob=np.max(self.output)
        high_index=index_list[np.argmax(self.output)][1]
        
        return high_prob,high_index #prob, next idx
