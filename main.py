import os
import pickle
import random
import argparse
import pandas as pd
from tqdm import tqdm

from agent import Agent
from recommendation import Recommendation

parser=argparse.ArgumentParser()
parser.add_argument('--recommender',type=str,default='random') 
parser.add_argument('--model',type=str,default='gcn')

args=parser.parse_args()

recom=Recommendation(args.recommender) # .random, .cf, .cbf
agent=Agent(args.model) #gcn at2v sage

DATA_PATH=os.path.join(os.getcwd(),'DATA/')
Result_PATH=os.path.join(os.getcwd(),'Result/')


id_index=pd.read_csv(DATA_PATH+'movieid_index.csv')
movie_info=pd.read_csv(DATA_PATH+'Netflix.csv',index_col=0)



def initial():
    return random.choices(list(range(0,8471)),k=1)[0]

def find_movie_name(index):
    movie_id=id_index[id_index['index']==index]['movie_id'].iloc[0]
    title=movie_info[movie_info['movie_id']==movie_id]['title'].iloc[0]
    return [movie_id,title]

idx=158
init_movie=find_movie_name(idx)[1]

print("initial movie is : "+init_movie)  

file_name=str(args.model)+'_'+str(args.recommender)
predict_score=[]
chosen_movie=[idx]

if __name__ == "__main__":
    iteration=0
    with tqdm(total=100) as pbar:
        while iteration<100: #tqdm
            pbar.update(1)
            movie_pairs=recom(idx)  
            prediction,pair_idx=agent(movie_pairs) #find the highest score
            #save
            predict_score.append(prediction)
            chosen_movie.append(pair_idx)
            #prepare next step
            idx=pair_idx
            iteration+=1
    with open(Result_PATH+'predict_score_'+str(file_name)+'.pkl','wb') as f:
        pickle.dump(predict_score,f)
    with open(Result_PATH+'pair_list_'+str(file_name)+'.pkl','wb') as i:
        pickle.dump(chosen_movie,i)