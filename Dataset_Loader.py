import numpy
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
def getdataset(i):
    if(i == 1):
        Data = numpy.loadtxt(open(str('Isolet.csv'), "rb"), delimiter=",", skiprows=1);
        labels = Data[:, -1]
        Data = Data[:,:-1]
        train, test, train_labels, test_labels = train_test_split(
                    Data, labels, test_size=0.33, random_state=42, stratify=labels)
        print((train_labels == train[:,-1]))
    if(i == 2):
        Data = numpy.loadtxt(open(str('AP_Breast_Lung.csv'), "rb"), delimiter=",", skiprows=1);
        pre_labels = Data[:, -1]
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(pre_labels)
        Data = Data[:,:-1]
        train, test, train_labels, test_labels = train_test_split(
                    Data, labels, test_size=0.33, random_state=42, stratify=labels)
        print((train_labels == train[:,-1]))
    if(i == 3):
        Data = numpy.loadtxt(open(str('Bioresponse.csv'), "rb"), delimiter=",", skiprows=1);
        labels = Data[:, -1]
        Data = Data[:,:-1]
        train, test, train_labels, test_labels = train_test_split(
                    Data, labels, test_size=0.33, random_state=42, stratify=labels)
        print((train_labels == train[:,-1]))
    if(i == 4):
        Data = numpy.loadtxt(open(str('gas-drift.csv'), "rb"), delimiter=",", skiprows=1);
        labels = Data[:, -1]
        Data = Data[:,:-1]
        train, test, train_labels, test_labels = train_test_split(
                    Data, labels, test_size=0.33, random_state=42, stratify=labels)
    return train, test,train_labels,test_labels