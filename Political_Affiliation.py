import numpy as np
import pandas as pd
import warnings
import pickle
from gensim.models.doc2vec import Doc2Vec

def prediction(pickleModel, xtest):
    data_pred=[]
    model= Doc2Vec.load(pickleModel)
    data_pred.append(model.infer_vector(xtest))
    lrg_pa = pickle.loads(pickleModel)
    pred_conf=lrg_pa.predict_proba(data_pred)
    #print(pred_conf)
    return pred_conf[0][1]


class Political_Affiliation:
    def __init__(self, model):
        self.model = model
    def predict(self, xtest):
        return prediction(model, xtest)



