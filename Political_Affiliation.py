# coding: utf-8

# ## Biased or SlantedNews detection Factor
# #### This notebook is a subset of Biased_Factor.ipynb

# In[1]:


import numpy as np
import pandas as pd
import warnings
import pickle
from gensim.models.doc2vec import Doc2Vec

from google.colab import drive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
googleDrive = GoogleDrive(gauth)
drive.mount('/content/gdrive')


def prediction(xtest):
    pickleModel = "/content/gdrive/My Drive/Drifters/Models/Politcal_Affiliation_Model.pkl"
    data_pred=[]
    model= Doc2Vec.load(pickleModel)
    data_pred.append(model.infer_vector(xtest))
    lrg_pa = pickle.loads(pickleModel)
    pred_conf=lrg_pa.predict_proba(data_pred)
    #print(pred_conf)
    return pred_conf[0][1]


class PoliticalAffiliationDetection:
    def __init__(self, xtest):
        self.xtest = xtest
    def predict(self):
        return prediction(self.xtest)



