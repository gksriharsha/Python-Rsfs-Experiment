import numpy
import numpy.matlib
import time
import math
import Classifier as clf
from scipy.stats import norm
from sklearn import preprocessing



def RSFS(Feature_train, Feature_test, label_train, label_test, Parameters):
    max_iters = Parameters['RSFS']['maxiters']
    n_dummyfeats = Parameters['RSFS']['Dummy feats']
    max_delta = Parameters['RSFS']['delta']
    k_neighbors = Parameters['RSFS']['K']
    label_test = label_test.astype('int')
    label_train = label_train.astype('int')
    verbose = 1
    N_classes = len(numpy.unique(label_train))
    number_of_features = numpy.size(Feature_train, axis=1)
    relevance = numpy.zeros((number_of_features,))
    dummy_relevance = numpy.zeros((n_dummyfeats,))
    stored =[]

    Feature_train = preprocessing.scale(Feature_train)
    Feature_test = preprocessing.scale(Feature_test)

    if (Parameters['RSFS']['fn'] == 'sqrt'):
        feats_to_take = round(math.sqrt(number_of_features))
        #feats_to_take = feats_to_take.astype('int')
        dummy_feats_to_take = round(math.sqrt(n_dummyfeats))
        #dummy_feats_to_take = dummy_feats_to_take.astype('int')
    if (Parameters['RSFS']['fn'] == '10log'):
        feats_to_take = round(10 * math.log10(number_of_features))
        #feats_to_take = feats_to_take.astype('int')
        dummy_feats_to_take = round(10 * math.log10(n_dummyfeats))
        #dummy_feats_to_take = dummy_feats_to_take.astype('int')

    feat_N = numpy.zeros(max_iters)

    totcorrect = numpy.zeros(N_classes)
    totwrong = numpy.zeros(N_classes)

    iteration = 1
    deltaval = math.inf
    cutoff = Parameters['RSFS']['cutoff']
    Threshold = Parameters['RSFS']['Threshold']
    probs = numpy.zeros(numpy.shape(relevance))
    while (iteration <= max_iters and deltaval > max_delta):
        feature_indices =  numpy.floor(number_of_features * numpy.random.rand(1, feats_to_take))
        feature_indices = feature_indices.astype('int')
        if ('stored' in locals()):
            for i in list(range(0, len(stored))):
                feature_indices = feature_indices(feature_indices != stored(i))

        class_hypos = clf.Classifier_II(Feature_train[:, numpy.resize(feature_indices,(numpy.size(feature_indices),))], Feature_test[:,numpy.resize(feature_indices,(numpy.size(feature_indices),))], label_train,
                                    Parameters)

        correct = numpy.zeros(N_classes)
        wrong = numpy.zeros(N_classes)

        for j in list(numpy.arange(0, numpy.size(label_test))):
            if (label_test[j] == class_hypos[j]):
                correct[label_test[j] - 1] = correct[label_test[j] - 1] + 1
            else:
                wrong[label_test[j] - 1] = wrong[label_test[j] - 1] + 1

        totcorrect = totcorrect + correct
        totwrong = totwrong + wrong

        performance_criterion = numpy.mean(numpy.array(correct) * 100 / (numpy.array(correct) + numpy.array(wrong)))
        expected_criterion_value = numpy.mean(numpy.array(totcorrect) * 100 / (numpy.array(totcorrect) + numpy.array(totwrong)))

        target = performance_criterion - expected_criterion_value
        pos = feature_indices
        relevance[pos] += target

        dummy_indices = numpy.floor(n_dummyfeats * numpy.random.rand(1,dummy_feats_to_take))
        dummy_indices = dummy_indices.astype('int')
        target = dummy_relevance[dummy_indices] + performance_criterion - expected_criterion_value
        pos = dummy_indices
        for x, y in zip(pos, target):
            dummy_relevance[x] = y
        if(iteration>5):
            probs = norm.cdf(relevance, loc=numpy.mean(dummy_relevance), scale=numpy.std(dummy_relevance))


        feat_N[iteration] = numpy.size(numpy.where(probs > cutoff))

        if (iteration % Threshold == 0):
            if (verbose == 1):
                deltaval = numpy.std(feat_N[iteration - (Threshold-1):iteration]) / numpy.mean(feat_N[iteration - (Threshold-1):iteration])
                print('RSFS: ', feat_N[iteration], 'features chosen so far (iteration: ', iteration, '/', max_iters,'). Delta: ', deltaval)

        iteration = iteration + 1

        if (Parameters['RSFS']['stored'] == 1):
            top = Parameters['RSFS']['top']
            Threshold = Parameters['RSFS']['Threshold']
            if (iteration > Threshold):
                S = numpy.where(probs > cutoff)
            W = relevance[S]
            comm = [S, W]
            comm = comm[comm[:, 1].argsort(),]
            if (len(S) >= top):
                stored.extend(comm[0:top-1, 1])
            else:
                stored.extend(comm[0:len(S)-1, 1])
            stored = list(numpy.unique(stored))

    S = numpy.where(probs>cutoff)
    W = relevance[S]
    return {'F_RSFS':S, 'W_RSFS':W , 'Stored':stored, 'iteration': iteration}


def Run_Class(Feature_train, Feature_test, label_train, label_test, Parameters):
    t1 = time.time()
    Res =  RSFS(Feature_train,Feature_test,label_train,label_test,Parameters)
    F_RSFS = Res['F_RSFS']
    W_RSFS = Res['W_RSFS']
    stored = Res['Stored']
    iteration = Res['iteration']
    t2 = time.time()
    RSFS_time = t2-t1

    t1 = time.time()
    hypos_orig =  clf.Classifier_I(Feature_train,Feature_test,label_train,Parameters)
    Original_Accuracy = sum(hypos_orig == label_test) / numpy.size(label_test) * 100
    print('Original :',numpy.size(Feature_train,axis = 1),'features: ',Original_Accuracy,'% correct.')
    t2 = time.time()
    Orig_time = t2-t1

    hypos_RSFS = clf.Classifier_I(Feature_train[:,numpy.resize(F_RSFS,(numpy.size(F_RSFS),))], Feature_test[:,numpy.resize(F_RSFS,(numpy.size(F_RSFS),))], label_train, Parameters)
    RSFS_Accuracy = sum(hypos_RSFS == label_test) / numpy.size(label_test) * 100
    print('RSFS feature set (',numpy.size(F_RSFS),'):', RSFS_Accuracy,' correct')

    if (Parameters['RSFS']['stored'] == 1):
        hypos_stored = clf.Classifier_I(Feature_train[:,numpy.resize(stored,(numpy.size(stored),))], Feature_test[:numpy.resize(stored,(numpy.size(stored),))], label_train, Parameters)
        Stored_Accuracy = sum(hypos_stored == label_test) / numpy.size(label_test) * 100
        print('RSFS feature set using new logic (',len(stored),' features):', Stored_Accuracy, 'correct.' )
        Result ={
            'Stored':{
                'Acc' : Stored_Accuracy,
                'Feats' : stored,
            }
        }


    Result={
        'Orig':{
            'Accuracy': Original_Accuracy,
            'Features': numpy.size(Feature_train,axis=1),
            'Time' : Orig_time
        },
        'RSFS':{
            'Accuracy': RSFS_Accuracy - Original_Accuracy,
            'Time':RSFS_time,
            'Iteration':iteration,
            'Feats' : F_RSFS,
            'Weights': W_RSFS,
         }
    }
    return Result
