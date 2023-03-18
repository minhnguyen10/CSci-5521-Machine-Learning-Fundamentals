#import libraries
import numpy as np
from numpy.core.arrayprint import printoptions

class Kmeans:
    def __init__(self,k=8): # k is number of clusters
        self.num_cluster = k
        self.center = None
        self.error_history = []

    def run_kmeans(self, X, y):
        # initialize the centers of clutsers as a set of pre-selected samples
        init_idx = [1, 200, 500, 1000, 1001, 1500, 2000, 2005] # indices for the samples
        # store all the initial centers to self.center
        self.center = []
        for j in range(len(init_idx)):
            self.center.append(X[init_idx[j]])


        num_iter = 0 # number of iterations for convergence

        # initialize cluster assignment
        prev_cluster_assignment = np.zeros([len(X),]).astype('int')
        cluster_assignment = np.zeros([len(X),]).astype('int')
        is_converged = False
        # print(X[1].shape)

        # iteratively update the centers of clusters till convergence
        while not is_converged:

            # iterate through the samples and compute their cluster assignment (E step)
            for i in range(len(X)):
                # use euclidean distance to measure the distance between sample and cluster centers
                distances = []
                for j in range(len(self.center)):
                    distances.append(np.linalg.norm(X[i]-self.center[j], axis=0))

                # determine the cluster assignment by selecting the cluster whose center is closest to the sample
                cluster_assignment[i] = np.argmin(distances) 

            # update the centers based on cluster assignment (M step)
            for i in range(len(init_idx)):
                data_each_cluster = []
                for j in range(len(cluster_assignment)):
                    if cluster_assignment[j] == i:
                        data_each_cluster.append(X[j])
                self.center[i] = np.mean(data_each_cluster, axis=0)

            # compute the reconstruction error for the current iteration
            cur_error = self.compute_error(X, cluster_assignment)
            self.error_history.append(cur_error)

            # reach convergence if the assignment does not change anymore
            is_converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1

        # compute the information entropy for different clusters
        entropy = float('inf') # placeholder
        entropy_each = [] #entropy for each cluster
        for i in range(self.num_cluster):
            labels_counts = [0,0,0]
            total = 0 
            for j in range(len(cluster_assignment)): 
                if cluster_assignment[j] == i:            
                    if y[j] == 0:
                        labels_counts[0] += 1
                    elif y[j] == 8:
                        labels_counts[1] += 1
                    else:
                        labels_counts[2] += 1
                    total += 1
            HXk = 0
            for k in range(len(labels_counts)):
                if(labels_counts[k]==0):
                    HXk += 0
                else:
                    PXck = labels_counts[k]/total
                    HXk +=  PXck*np.log2(PXck)
            entropy_each.append(-1*HXk)
            pass
        #compute the avarage entropy
        entropy = np.sum(entropy_each)/self.num_cluster
        return num_iter, self.error_history, entropy

    def compute_error(self,X,cluster_assignment):
        # compute the reconstruction error for given cluster assignment and centers
        error = 0 # placeholder
        for i in range(len(X)):
            error_each = 0
            for j in range(self.num_cluster):
                if cluster_assignment[i] == j: 
                    error_each += np.sum((X[i] - self.center[j])**2)
            error += error_each
        return error

    def params(self):
        return self.center
