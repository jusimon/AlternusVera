import numpy as np
import pandas as pd
import warnings
import pickle

labelcolname = 'Encoded_Label'
titlecolname = 'partyAffiliation'
def prediction(xtest, ytest):
    pickleModel = "/content/gdrive/My Drive/Drifters/Models/Politcal_Affiliation_Model.pkl"
    pickle_in = open(pickleModel, "rb")
    loadData = pickle.load(pickle_in)
    return np.mean(loadData.predict(xtest) == ytest)

true_labels = ['original','true','mostly-true','half-true']
false_labels = ['barely-true','false','pants-fire']
def simplify_label(input_label):
    if input_label in true_labels:
        return 1
    else:
        return 0

class Political_Affiliation:
    def __init__(self, xtest):
        self.x_test = xtest[titlecolname]
        self.y_test = xtest.apply(lambda row: simplify_label(row['Label']), axis=1)
        self.x_test = self.x_test.replace(np.nan,'No Job Title', regex=True)
    def predict(self):
        return prediction(self.x_test, self.y_test)
