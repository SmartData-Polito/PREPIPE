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
    
    def makeDataset(self, signals=False,features=False):
        
        if(features==False):
            self.features = self.signalsToFeatures(signals)
        else:
            self.features = features
        
        Id = []
        X = []
        Y = []
        
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
        

        for j, row in self.files_list.iterrows():            
            file = row["cycleName"]
            df = pd.read_csv(self.fpathCameoFiles + file+".csv",  encoding ="latin1", usecols = signals)
            
            chunks = self.makeChunks(df, int(df.shape[0]/(self.minuteWindow*60)))
            
            for chunk in chunks:
                w = []
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
                X.append(w)
                Id.append(file)
                Y.append(row['label'])        
            
            
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
