from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from signalsToFeatures import signalsToFeatures
from collections import OrderedDict
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import product
from sklearn import svm
import numpy as np



class SignalSelectionSVMBased:

    def __init__(self, Dataset,signals,percToDrop):
        self.signals = signals
        self.Dataset = Dataset
        self.includeDerivate=True
        
        self.percToDrop = percToDrop
        self.svm_params = {'C': 10**np.linspace(-3,3,8),
        'random_state': [42],
        'kernel': ['linear'],
        'degree': [2]}
    
    def select(self):
        
        print("Start Selection")
        
        n_signals = []
        f1scores = []
        signalsLists = []

        features = list(self.Dataset.columns)
        features.remove('Label')
        features.remove('ExpID')
        
        Xnp = self.Dataset[features]
        Ynp = self.Dataset['Label']
        Id  = self.Dataset['ExpID']
        currentFeatures = features
        currentSignals = self.signals
        
        while(True):  
            best_score = -1
            best_test = None
            
            currentFeatures = signalsToFeatures(currentSignals)        
            Xnp = np.matrix(self.Dataset[currentFeatures])
            
            
            scaler = StandardScaler()
            scaler.fit(Xnp)
            X_train = scaler.transform(Xnp)
            
            for test in [dict(zip(self.svm_params, v)) for v in product(*self.svm_params.values())]:

                clf = svm.SVC(kernel= test["kernel"], C= test["C"], random_state= test["random_state"], degree= test["degree"])

                skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                f1_red = make_scorer(f1_score, average="micro", labels=["red"])
                scoring = {'F1_red': f1_red}               
                actual_stats = cross_validate(clf, X_train, Ynp, scoring = scoring, cv=skf)
                actual_stats= np.mean(actual_stats['test_F1_red'])                
                                
                if(best_score <= actual_stats):
                    best_score = actual_stats
                    best_test = test
                
            n_signals.append(len(currentSignals))
            f1scores.append(best_score)
            signalsLists.append(currentSignals)

            
            clf = svm.SVC(kernel= best_test["kernel"], C= best_test["C"], random_state= best_test["random_state"], degree= best_test["degree"])
            clf = clf.fit(X_train, Ynp)
            w = clf.coef_

            subarrays = np.split(np.transpose(w),len(currentSignals))
            d = {}
            i = 0 
            for var in currentSignals:
                importance = np.linalg.norm(subarrays[i], ord=2) ** 2
                d[var] = importance
                i=i+1


            sorted_d = OrderedDict(sorted(d.items(), key=itemgetter(1), reverse=True))
            first = list(sorted_d.keys())[0]        
            last = list(sorted_d.keys())[-1]        
           
            
            numFeaturesToDrop = round(len(sorted_d.keys())*self.percToDrop) + 1
            currentSignals = list(sorted_d.keys())[0:len(sorted_d.keys())-int(numFeaturesToDrop)]

                    
            if (len(sorted_d) <= 1):
                break

        # Return lists of signals selected
        signalsSelected = []
        max_f1score = -1
        for i in range(0, len(n_signals)):
            if f1scores[i] >= max_f1score:
                max_f1score = f1scores[i]
                signalsSelected = signalsLists[i] 

        self.score = max_f1score
        return signalsSelected,f1scores
