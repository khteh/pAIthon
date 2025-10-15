import os, numpy, pandas as pd, tensorflow as tf

class SiameseNN():
    _path:str = None

    def __init__(self, path: str):
        self._path = path
        self._PrepareData()
    def _PrepareData(self):
        data = pd.read_csv(self._path)
        N = len(data)
        print('Number of question pairs: ', N)
        data.head()       
         
if __name__ == "__main__":
    siamese = SiameseNN("data/questions.csv")