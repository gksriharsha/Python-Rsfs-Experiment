import numpy
from sklearn.model_selection import train_test_split
def getdataset():
    Data = numpy.loadtxt(open(str('Isolet.csv'), "rb"), delimiter=",", skiprows=1);
    labels = Data[:, -1]
    Data = Data[:,:-1]
    train, test, train_labels, test_labels = train_test_split(
                Data, labels, test_size=0.33, random_state=42, stratify=labels)
    print((train_labels == train[:,-1]))
    return train, test,train_labels,test_labels