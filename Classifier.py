from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def Classifier_I(Feature_train, Feature_test, label_train, Parameters):
    if (Parameters['Classifier'] == 'KNN'):
        K = 3
        KNN_C = KNeighborsClassifier(n_neighbors=K, weights='distance')
        hypos = KNN_C.fit(Feature_train, label_train).predict(Feature_test)
        return hypos
    elif (Parameters['Classifier'] == 'SVM'):
        KNN_C = SVC(kernel = 'linear')
        hypos = KNN_C.fit(StandardScaler.fit(Feature_train),label_train).predict(StandardScaler.fit(Feature_test))
        return hypos
    else:
        pass


def Classifier_II(Feature_train, Feature_test, label_train, Parameters):
    if (Parameters['Classifier'] == 'KNN'):
        K = Parameters['RSFS']['K']
        KNN_Cl = KNeighborsClassifier(n_neighbors=K, weights='distance')
        hypos = KNN_Cl.fit(Feature_train, label_train).predict(Feature_test)
        return hypos
    elif (Parameters['Classifier'] == 'SVM'):
        KNN_C = SVC(kernel = 'linear')
        hypos = KNN_Cl.fit(StandardScaler.fit(Feature_train),label_train).predict(StandardScaler.fit(Feature_test))
        return hypos
    else:
        pass
