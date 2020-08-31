from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
import matplotlib.pyplot as plt
from operator import itemgetter
from findKnee import findKnee
import numpy as np

class signalSelectionRF:
    def __init__(self, Dataset,signals=[]):
        self.signals = signals
        self.Dataset = Dataset
                
    def select(self):

        features = list(self.Dataset.columns)
        features.remove('Label')
        if('ExpID' in features): features.remove('ExpID')

        Xnp = self.Dataset[features]
        Ynp = self.Dataset['Label']

        d = {}
        first = True
        # Feature importance values from Random Forests
        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                        max_depth=90, max_features=int(len(features)/22), max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0,n_jobs=-1,
                        oob_score=False, random_state=42, verbose=0,
                        warm_start=False, n_estimators=1000)

        rf.fit(Xnp, Ynp)
        feat_imp = rf.feature_importances_
        print("RF - Over")

        subarrays = []
        unit = self.signals
        if(len(self.signals)==0):
            subarrays = np.split(feat_imp,len(features))
            unit = features
        else:
            subarrays = np.split(feat_imp,len(self.signals))

        i = 0 
        for var in unit:
            importance = sum(subarrays[i])
            if first:
                d[var] = importance

            else:
                d[var] += importance
            i=i+1
        first = False
            
        sorted_d = OrderedDict(sorted(d.items(), key=itemgetter(1), reverse=True))

        print("Plot final variables' importance...")
        plt.figure(figsize=(15,10))
        r1=plt.barh(range(len(sorted_d)), list(sorted_d.values()), align='center')
        plt.yticks(range(len(sorted_d)), list(sorted_d.keys()))
        plt.show()
        
        importances_sorted = sorted_d.values()
        cumsum_list = list(importances_sorted)
        d = [n for n in range(1,len(cumsum_list)+1)]
        
        idx, value = findKnee(np.cumsum(cumsum_list))

        plt.figure(figsize=(15, 7))

        plt.plot(d,np.cumsum(cumsum_list), color = 'red',label='Cumulative importance')
        plt.bar(d,[i for i in cumsum_list], label='Signal importance')

        plt.axhline(y = value, color='k', linestyle='--', label = 'knee')
        plt.axvline(x = idx, color='g', linestyle='--', label = 'knee')

        plt.title('Cumulative importance as a Function of the Number of signals selected')
        plt.xticks(d)
        plt.legend(loc='best')
        plt.show()

        FeatureOrder = list(sorted_d.keys())
        print("Knee: ",idx)
        if(len(self.signals)>0): return FeatureOrder[0:idx]
        
        return FeatureOrder