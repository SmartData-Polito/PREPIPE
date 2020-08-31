from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re


## it returns a subset of signal after first phase of data selection (by category, ignored_variables, o2, strong correlated)
class APriori:
    def __init__(self, files_list, fpath, o2_considerated = True, strong_corr_considerated = True):
        self.files_list = files_list
        self.fpath = fpath

        self.o2_considerated = o2_considerated
        self.strong_corr_considerated = strong_corr_considerated

        self.ignoredVariables = ["S09","S11"] #Signals to discard according to domain knowledge
        self.ignoredCategories = [] #Signal categories to discard according to domain knowledge - Must define the category in the extractCategory function!
        self.signalSelected = []
        self.minDistinct = 200
        self.valueHighCorr = 0.995

    def select(self):
        if self.o2_considerated == False:
            self.ignoredVariables.append("S10")

        for file in self.files_list:
            df = pd.read_csv(self.fpath + file+".csv",  encoding ='latin1')
            candidates = df.keys()
            candidates = self.filterVariables(df)
            candidates = self.filterDiscrete(df, candidates)
            for signal in candidates:
                if not signal in self.signalSelected:
                    self.signalSelected.append(signal)
        if self.strong_corr_considerated == False:
            signalToDrop = self.filterCorrSignal()
            for signal in signalToDrop:
                self.signalSelected.remove(signal)

        return self.signalSelected

    def filterVariables(self, df):
        features = []
        # for each variable:
        # - discard it if it is part of the ignoreList
        # - discard it if it does not belong to a 3-letters category
        # - keep it otherwise
        for k in df.keys():
            drop = True
            if k not in self.ignoredVariables:
                categ = self.extractCategory(k) 
                if categ not in self.ignoredCategories:
                    features.append(k)
        return features

    def filterDiscrete (self, df, features):
        candidates = []
        for feature in features:
            distinct = len(set(df[feature]))
            if distinct >= self.minDistinct:
                candidates.append(feature)
        return candidates

    #return signals to drop. these signals have strong correlation with another signal (that we keep in)
    def filterCorrSignal(self):
        df_total = pd.DataFrame();

        for file in self.files_list:

            df_data = pd.read_csv(self.fpath + file+".csv",  encoding ='latin1', usecols=self.signalSelected) 

            frames = [df_total, df_data]
            df_total = pd.concat(frames)

        correlation = df_total.corr()
        plt.figure(figsize=(25,25))
        sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='RdYlGn')
        plt.title('Correlation between different fearures')
        plt.show()

        dropForCorr = []
        for index, row in correlation.iterrows():
            if row.name not in dropForCorr:
                for i in row.index:
                    if(row[i] > self.valueHighCorr):
                        if (row.name != i):
                            print(row.name,i,row[i])
                            dropForCorr.append(i)
        return dropForCorr
    
    def extractCategory(self, key):
        return key[0:2]