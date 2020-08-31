from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances

class PFA(object):
    def __init__(self, n_signals, q=None):
        self.q = q
        self.n_signals = n_signals
        self.error_ = 0

    def fit(self, X):
        if not self.q:
            self.q = X.shape[1]

        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA(n_components=self.q).fit(X)
        self.A_q = pca.components_.T

        kmeans = KMeans(n_clusters=self.n_signals, random_state=42).fit(self.A_q)
        self.clusters = kmeans.predict(self.A_q)
        cluster_centers = kmeans.cluster_centers_

        self.inertia = kmeans.inertia_ 
        
        dists = defaultdict(list)
        #print(clusters)
        for i, c in enumerate(self.clusters):
            dist = euclidean_distances([self.A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))
            self.error_ += dist
                
        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]