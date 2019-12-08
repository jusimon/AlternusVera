import numpy as np
import pandas as pd
import warnings
import pickle

def prediction(xtest, ytest):
    pickleModel = "/content/gdrive/My Drive/Drifters/Models/Politcal_Affiliation_Model.pkl"
    pickle_in = open(pickleModel, "rb")
    loadData = pickle.load(pickle_in)
    return np.mean(loadData.predict(xtest) == ytest)

class Political_Affiliation:
    def __init__(self, xtest):
        self.x_test = xtest[titlecolname]
        self.y_test = xtest[labelcolname]
        self.x_test = self.x_test.replace(np.nan,'No Job Title', regex=True)
    def predict(self):
        return prediction(self.x_test, self.y_test)
