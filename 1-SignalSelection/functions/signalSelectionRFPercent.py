from sklearn.ensemble import RandomForestClassifier
from signalsToFeatures import signalsToFeatures
from collections import OrderedDict
import matplotlib.pyplot as plt
from operator import itemgetter
from findKnee import findKnee
import numpy as np

class SignalSelectionRFPercent:
    def __init__(self, signals, Dataset, percToDrop=0.2):
        self.signals = signals
        self.Dataset = Dataset
        self.percToDrop = percToDrop
        self.minOOBError = None
        
        self.signalsSelected = []

               
    def select(self):
        
        features = list(self.Dataset.columns)
        features.remove('Label')
        features.remove('ExpID')
        Xnp = self.Dataset[features]
        Ynp = self.Dataset['Label']
        Id  = self.Dataset['ExpID']
        
        oob_errors = []
        signal_size = []
        d = []
        listsSignals = []
        currentSignals = self.signals
        while(True):
            currentFeatures = signalsToFeatures(currentSignals)        
            Xnp = self.Dataset[currentFeatures]
                        
            sorted_signals, oob_error = self.randomForest(Xnp, Ynp, currentSignals, plot=True)
            
            oob_errors.append(oob_error)
            signal_size.append(len(currentSignals))
            d.append(len(currentSignals))
            listsSignals.append(currentSignals)
            
            numFeaturesToDrop = round(len(sorted_signals.keys())*self.percToDrop) + 1
            #print("Dropping " + str(int(numFeaturesToDrop)) + " signals...")
            currentSignals = list(sorted_signals.keys())[0:len(sorted_signals.keys())-int(numFeaturesToDrop)]

            if (len(sorted_signals) <= 1):
                # Return lists of signals selected
                signalsSelected = []
                minOOBError = None
                for i in range(0, len(d)):
                    if(minOOBError == None): minOOBError = oob_errors[i]
                    if oob_errors[i] <= minOOBError:
                        minOOBError = oob_errors[i]
                        signalsSelected = listsSignals[i] 
                
                self.minOOBError = minOOBError
                self.signalsSelected = signalsSelected
                
                return signalsSelected,oob_errors, signal_size

    def randomForest(self, X, Y, Signals, plot=False):
        
        #grid search
        best_oob      = None
        best_features_importance = None        
        for config in configs:
        d = {}
        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                    max_depth=90, max_features=len(X.columns), max_leaf_nodes=None,
                    min_impurity_decrease=0.0, min_impurity_split=None,
                    min_samples_leaf=1, min_samples_split=2,
                    min_weight_fraction_leaf=0.0,n_jobs=-1,
                    oob_score=True, random_state=0, verbose=0,
                    warm_start=False, n_estimators = 2000)

        rf.fit(X, Y)
        best_oob = 1 - rf.oob_score_
        best_features_importance = rf.feature_importances_     
        
               
        feat_imp = best_features_importance
        subarrays = np.split(feat_imp,len(Signals))

        i = 0 
        for var in Signals:
            importance = sum(subarrays[i])
            d[var] = importance
            i=i+1

        sorted_signals = OrderedDict(sorted(d.items(), key=itemgetter(1), reverse=True))

        if plot==True:
            print("Plot final signals' importance...")
            plt.figure(figsize=(15,10))
            r1=plt.barh(range(len(sorted_signals)), list(sorted_signals.values()), align='center')
            plt.yticks(range(len(sorted_signals)), list(sorted_signals.keys()))
            plt.show()

        print("OOB ERROR: " + str(best_obb))
        return sorted_signals, best_obb