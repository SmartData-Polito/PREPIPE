from vest.models.bivariate import BivariateVEST, multivariate_feature_extraction
from vest.config.aggregation_functions \
    import (SUMMARY_OPERATIONS_ALL,
            SUMMARY_OPERATIONS_FAST,
            SUMMARY_OPERATIONS_SMALL)

import tsfel

import pandas as pd
import numpy as np

class MakerDatasetSpec:
    
    def __init__(self, files_list, fpathCameoFiles, minuteWindow=60, includeFiltering = False, n=3):
        self.files_list = files_list
        self.fpathCameoFiles = fpathCameoFiles
        
        # example: minuteWindow = 2 -> 30 window/hour
        self.minuteWindow = minuteWindow
        self.includeFiltering = includeFiltering
        self.n = n #outliers
        self.includeDerivate = True
    
    def makeDataset(self, signals=False,strategy="ad-hoc"):
    
        def adhoc(strategy):
            
            
            self.features = strategy
            if(strategy=="ad-hoc"):
                self.features = self.signalsToFeatures(signals)            

            featuresSpec = [] 
            for var in self.features:
                #print (var)
                x = var.split(" ")
                if len(x) == 2:
                    x.append(False)
                else:
                    x[2] = True
                featuresSpec.append(x)
                signals.append(x[0])            

            for feature in featuresSpec:
                var = feature[0]
                isDerivate = feature[2]
                featureType = feature[1]

                if(isDerivate == True):
                    temp = ""
                    if(self.includeFiltering == True):
                        temp = self.filterOutliers(chunk[var].diff(), self.n)
                    else:
                        temp = chunk[var].diff()                   

                    if featureType == "mean":
                        w.append(np.nanmean(temp))
                    elif featureType == "std":
                        w.append(np.nanstd(temp))
                    else:
                        w.append(np.nanpercentile(temp,int(featureType)))
                else:
                    temp = ""
                    if(self.includeFiltering == True):
                        temp = self.filterOutliers(chunk[var], self.n)
                    else:
                        temp = chunk[var]

                    if featureType == "mean":
                        w.append(np.nanmean(temp))
                    elif featureType == "std":
                        w.append(np.nanstd(temp))
                    else:
                        w.append(np.nanpercentile(temp,int(featureType)))
                            
            return w
        
        
        Id = []
        X = []
        Y = []

        for j, row in self.files_list.iterrows():            
            file = row["cycleName"]
            df = pd.read_csv(self.fpathCameoFiles + file,  encoding ="latin1", usecols = signals)
            
            chunks = self.makeChunks(df, int(df.shape[0]/(self.minuteWindow*60)))
            
            for chunk in chunks:
                chunk = chunk.fillna(method='bfill')
                #return chunk,"a","a"
                w = []
                if(strategy=="ad-hoc" or type(strategy) == list): w = adhoc(strategy)
                elif(strategy=="tsfel-all" or strategy=="tsfel-all-corr"): 
                    cfg = tsfel.get_features_by_domain()
                    w = tsfel.time_series_features_extractor(cfg, chunk)   
                    self.features = list(w.columns)
                elif(strategy=="tsfel-statistical"): 
                    cfg = tsfel.get_features_by_domain('statistical')
                    w = tsfel.time_series_features_extractor(cfg, chunk)  
                    self.features = list(w.columns)
                elif(strategy=="tsfel-temporal"): 
                    cfg = tsfel.get_features_by_domain('temporal')
                    w = tsfel.time_series_features_extractor(cfg, chunk)  
                    self.features = list(w.columns)
                elif(strategy=="vest"): 
                    model = BivariateVEST()
                    features = model.extract_features(df, pairwise_transformations=False, summary_operators=SUMMARY_OPERATIONS_SMALL)
                    w = multivariate_feature_extraction(df, apply_transform_operators=False, summary_operators=SUMMARY_OPERATIONS_SMALL)    
                    self.features = list(w.columns)
                
                X.append(w)
                Id.append(file)
                Y.append(row['label'])        
        
        
        if(strategy=="tsfel-all-corr"):
            dataset = pd.concat(X)
            to_drop = tsfel.correlated_features(dataset)
            dataset = dataset.drop(to_drop,axis=1)
            self.features = dataset.columns
            X = dataset
                
            
        return np.array(X), np.array(Y), np.array(Id)
            
        
    def filterOutliers(self, values, n=3):
        mean = np.nanmean(values)
        std = np.nanstd(values)
        return [v for v in values if v - mean <= n * std]
        #return values
    
    def makeChunks(self, seq, num):
       
        avg = len(seq) / float(num)
        out = []
        last = 0.0
        while last < len(seq) and len(out)<num:
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out
    

    def signalsToFeatures(self,signals):

        Features = []      
        #Generate Features
        for var in signals:
            for i in [10,20,30,40,50,60,70,80,90]:
                Features.append(var + " " + str(i))
            Features.append(var + " mean")
            Features.append(var + " std")

            if self.includeDerivate==True:
                for i in [10,20,30,40,50,60,70,80,90]:
                    Features.append(var + " " + str(i) + " deriv")
                Features.append(var + " mean deriv")
                Features.append(var + " std deriv")
        return Features    
