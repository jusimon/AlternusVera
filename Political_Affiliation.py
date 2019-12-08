import warnings
import pickle

def prediction(xtest):
    xtest = xtest.split(' ')
    pickleModel = "/content/gdrive/My Drive/Drifters/Models/Politcal_Affiliation_Model.pkl"
    pickle_in = open(pickleModel, "rb")
    loadData = pickle.load(pickle_in)   
    dataset = loadData.predict(xtest)
    for i in dataset:
        if(i == 0):
            return 0
    return 1
            
class Political_Affiliation:
    def __init__(self, xtest):
        self.x_test = xtest
    def predict(self):
        return prediction(self.x_test)
