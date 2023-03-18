import numpy as np
from numpy.lib import cov

def PCA(X,num_dim=None):
    # X_pca, num_dim = X, len(X[0]) # placeholder
    X_pca = X
    # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)
    centered_X = X - np.mean(X, axis=0)
    cov_X = np.cov(centered_X.T)
    v , w = np.linalg.eigh(cov_X)
    reordered_v = np.flip(v,0)
    reordered_w = np.flip(w,1)


    # select the reduced dimensions that keep >95% of the variance
    if num_dim is None:
        # print('here')
        sum_lamda = sum(reordered_v)
        prob = []
        for i in reordered_v:
            prob.append(i/sum_lamda)
        PoV = 0
        for i in range(len(prob)):
            if PoV < 0.95:
                PoV += prob[i]
            else:
                num_dim = i
                break
        pass

    # project the high-dimensional data to low-dimensional one
    selected_w = (reordered_w.T[:][:num_dim])
    X_pca = centered_X.dot(selected_w.T)
    return X_pca, num_dim
