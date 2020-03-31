import numpy as np

def distance(data):
    """
    Helper function to calculate pair-wise distance in the original data points
    """
    n = len(data)
    dist = [[] for i in range(n)]
    for i in range(n-1):
        for j in range(i+1, n):
            cur = np.linalg.norm(data[i] - data[j])
            dist[i].append([cur, j])
            dist[j].append([cur, i])
    for i in range(n):
        dist[i].sort(key=lambda x : x[0])
            
    return dist

class DBSCAN():
    def __init__(self, eps=5, MinPts=4):
        """Init DBSCAN class
        eps: threshold distance
        MinPts: Minimum required number of neighbours
        """
        self.labels = None
        self.eps = eps
        self.MinPts = MinPts
    
    def fit(self, Data):
        """Fitting with dataset
        #Input
            Data: dataset
        #return
            list of class labels
        """
        self.labels = [None for i in range(len(Data))] 
        # Init cluster id from 0
        C = 0
        for P in range(0, len(Data)):
            # labels[P] not None, which already clustered
            if self.labels[P] is not None: 
                continue
            # return P's NeighborPts
            NeighborPts = self.regionQuery(P, self.eps, Data)
            if len(NeighborPts) < self.MinPts:
                self.labels[P] = -1
            else:
                # Use P to new cluster and expand it
                self.expandCluster(P, NeighborPts, C, Data)
                C += 1
                
        return self.labels
    
    def expandCluster(self, P, NeighborPts, C, data):
        """expand cluster with P
        """
        # Add P to cluster C
        self.labels[P] = C
        # For loop all the NeighborPts of P 
        for Pn in NeighborPts:
            if self.labels[Pn] == None:
                self.labels[Pn] = C
                # return Pn's NeighborPts
                PnNeighborPts = self.regionQuery(Pn, self.eps, data)
                if len(PnNeighborPts) >= self.MinPts:
                    NeighborPts += PnNeighborPts

            if self.labels[Pn] == -1:
                self.labels[Pn] = C
    
    @staticmethod
    def regionQuery(P, eps, data):
        """
        Get list of local neighbor points(including itself)
        """
        P_neighbors = list()
        for Pn in range(len(data)):
            # If distance with the threshold, add to neighbours list
            if np.linalg.norm(data[P] - data[Pn]) < eps:
                P_neighbors.append(Pn)
        return P_neighbors

        