import os
import random
import numpy as np
import pandas as pd 

class Recommendation:
    def __init__(self,method):
        
        self.DATA_PATH=os.path.join(os.getcwd(),'DATA/')
        self.features=pd.read_csv(self.DATA_PATH+'feature.csv',index_col=0)
                
        if method=='cf':
            self.idx_data=pd.read_csv(self.DATA_PATH+'cf.csv')
            self.idx_data=self.idx_data.set_index(keys=['index'],
                                                  inplace=False,
                                                  drop=True)
            self.method=method
            #print('your recommedation based on ')
        elif method=='cbf':
            self.idx_data=pd.read_csv(self.DATA_PATH+'cbf.csv')
            self.idx_data=self.idx_data.set_index(keys=['Unnamed: 0'],
                                                  inplace=False,
                                                  drop=True)
            self.idx_data=self.idx_data.rename_axis('index')
            self.method=method
        else:
            self.method='random'
        
                
    def __call__(self,init_idx):
        if self.method=='cf':
            index_li=self.idx_data.loc[init_idx][1:].values.tolist()
        elif self.method=='cbf':
            index_li=self.idx_data.loc[init_idx,:].sort_values(ascending=False)[:100].index.tolist()
        elif self.method=='random':
            all_index=list(range(1,8472))
            index_li=random.choices(all_index,k=100)
            index_li=[int(x)for x in index_li]
        #feature=self.find_movie_feature(index_li)
        
        #index_li convert to array of [init,index]
        array_li=[]
        for ele in index_li:
            array_li.append([int(init_idx),int(ele)])
            
        return np.array(array_li)

    def find_movie_feature(self,index_li):
        feature_li=[]
        for index in index_li:
            index=int(index)
            feature_li.append(self.features.loc[index,:])
        return np.array(feature_li)