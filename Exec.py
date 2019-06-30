import rsfs_py as RSFS
import numpy
from sklearn.model_selection import train_test_split
import random
import csv

Data = numpy.loadtxt(open(str('Isolet.csv'), "rb"), delimiter=",", skiprows=1);
labels = Data[:, -1]
Data = Data[:,:-1]
train, test, train_labels, test_labels = train_test_split(
            Data, labels, test_size=0.33, random_state=42, stratify=labels)
print((train_labels == train[:,-1]))

Feature_train = train
Feature_test = test
label_train = train_labels
label_test = test_labels

Parameters = {
     'Top2' : 0,
    'RSFS' : {
        'K' : 3,
        'Dummy feats' : 100,
        'delta' : 0.05,
        'maxiters' : 300000,
        'fn' : 'sqrt',
        'cutoff' : 0.99,
        'stored' : 0,
        'top' : 2,
        'Threshold' : 1000,
    },
    'Classifier':'KNN'
}

States = {
    'RSFS':{
        'stored': [0,1],
        'K' : list(numpy.arange(3,11,2)),
        'Top2' : [0,1],
        'Dummy feats' : list(numpy.arange(100,numpy.size(Data,1),100)),
        'fn' : ['sqrt','10log'],
        'top' : list(range(1,7)),
        'Threshold' : list(numpy.arange(500,2000,200)),
	    'cutoff': [0.95,0.99,0.997]
    }
}

Combinations = numpy.array([])
limit = [len(States['RSFS']['Dummy feats']), len(States['RSFS']['K']), len(States['RSFS']['Threshold']) , len(States['RSFS']['fn']),
                       len(States['RSFS']['cutoff'])]
n= numpy.prod(limit)
for i in list(numpy.arange(1, len(limit) + 1)):
    li = [];
    for ii in list(numpy.arange(1,limit[i-1]+1)):
        li = list(numpy.append(li,numpy.matlib.repmat(ii , 1, int(n/numpy.prod(limit[0:i])))))
    addend = numpy.matlib.repmat(li,1,int(n/numpy.size(li)))
    Combinations = numpy.vstack((Combinations,addend)) if Combinations.size else addend

Combinations = numpy.transpose(Combinations)
#Combinations = list(Combinations)
print(Combinations)

j = 0
heading = ['S.No','Threshold','K','fn','Dummy feats','Orig_Acc','RSFS_Acc','Orig_time','RSFS_time','Total Features','RSFS features','Iteration']
with open('Performance_RSFS.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(heading)
    wr.writerow("\n")
while True:
    if(j == numpy.size(Combinations,axis=0)):
        break
    else:
        random.seed(0)
        numpy.random.seed(0)


        Parameters['RSFS']['cutoff'] = States['RSFS']['cutoff'][int(Combinations[j, 4])-1]
        Parameters['RSFS']['Threshold'] = States['RSFS']['Threshold'][int(Combinations[j, 2])-1]
        Parameters['RSFS']['fn'] = States['RSFS']['fn'][int(Combinations[j, 3])-1]
        Parameters['RSFS']['K'] = States['RSFS']['K'][int(Combinations[j, 1])-1]
        Parameters['RSFS']['Dummy feats'] = States['RSFS']['Dummy feats'][int(Combinations[j, 0])-1]
        print(Parameters)


        Result = RSFS.Run_Class(Feature_train,Feature_test,label_train,label_test,Parameters)
        line = [j,Parameters['RSFS']['cutoff'],Parameters['RSFS']['Threshold'],Parameters['RSFS']['K'],Parameters['RSFS']['fn'],Parameters['RSFS']['Dummy feats'],Result['Orig']['Accuracy'],Result['RSFS']['Accuracy'],
                Result['Orig']['Time'],Result['RSFS']['Time'],Result['Orig']['Features'],numpy.size(Result['RSFS']['Feats']),Result['RSFS']['Iteration']]

        with open('Performance_RSFS.csv', 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(line)
        print('Combination  :',j)
        j = j+1
