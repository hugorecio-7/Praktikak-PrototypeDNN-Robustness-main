import sys
import numpy as np
from scipy import stats

def get_pred_entropy(probs):
    assert len(probs.shape)==3, "Expected shape: (n_data, n_inferences, n_classes)"
    avg_probs = np.mean(probs, axis=1) #(n_data, n_classes)
    entrop    = - np.sum(avg_probs * np.log(avg_probs+1e-8), axis=1) #(n_data)
    return entrop

def get_aleatoric_unc(probs):
    assert len(probs.shape)==3, "Expected shape: (n_data, n_inferences, n_classes)"
    plogp_batch   = np.sum(probs * np.log(probs+1e-8), axis=2)  #(n_data, n_infs)
    aleatoric_unc = - np.mean(plogp_batch, axis=1) #(n_data)
    return aleatoric_unc

def get_epistemic_unc(probs):
    assert len(probs.shape)==3, "Expected shape: (n_data, n_inferences, n_classes)"
    return get_pred_entropy(probs) - get_aleatoric_unc(probs)

def riemann_sum(f, a, b, N):
    ''' Riemann sum of a density function f(x) over the interval [a,b] '''
    dx = (b - a)/N
    x = np.linspace(a,b,N+1)
    x_mid = (x[:-1] + x[1:])/2
    return np.sum(f(x_mid)*dx)

def overlap_index_bin(distances, neval, max_margin=10, min_margin=10):
    '''
    Returns the overlap index between two density functions.
    distances: numpy array with 2D or 3D and shape ([n_inputs,] n_classes, n_inferences).
    neval: number of points to be considered for the overlap computation
    '''
    dists = np.copy(distances)
    if len(dists.shape)==2:
        dists = np.expand_dims(dists, 0)
    assert len(dists.shape)==3
        
    outdim = dists.shape[0]
    out    = np.zeros(outdim)
    for i in range(outdim):
        if i%1000==0: print(i)
        kde1 = stats.gaussian_kde(dists[i][0])
        kde2 = stats.gaussian_kde(dists[i][1])
        def kdemin(x):
            return np.min(np.array([kde1(x), kde2(x)]), axis=0)

        xmax = dists[i].max() + max_margin
        xmin = 0.0 - min_margin
        out[i] = riemann_sum(kdemin, xmin, xmax, neval)
    return out


def overlap_index(distances, pred_classes, neval, max_margin=10, min_margin=10):
    '''
    Returns the overlap index between multiple density functions.
    distances: numpy array with 2D or 3D and shape ([n_inputs,] n_classes, n_inferences). 
    pred_classes: 1D numpy array with the most likely class for each input, shape: (n_inputs,)
    neval: number of evaluations of the riemman sum, to estimate the overlap
    '''
    dists = np.copy(distances)
    if len(dists.shape)==2:
        dists = np.expand_dims(dists, 0)
    assert len(dists.shape)==3
        
    num_classes = dists.shape[1]
    outdim = dists.shape[0]
    out    = np.zeros(outdim)
    for i in range(outdim):
        if i%1000==0: print("%d/%d"%(i,outdim))
        cur_ml_class = np.copy(pred_classes[i])
        kde1 = stats.gaussian_kde(dists[i][cur_ml_class])
        opt_overlap  = -1.0
        for j in range(num_classes):
            if j==cur_ml_class:
                continue
            kde2 = stats.gaussian_kde(dists[i][j])
            def kdemin(x):
                return np.min(np.array([kde1(x), kde2(x)]), axis=0)

            xmax = dists[i][j].max() + max_margin
            xmin = 0.0 - min_margin
            cur_overlap = riemann_sum(kdemin, xmin, xmax, neval)
            if cur_overlap>opt_overlap:
                opt_overlap = cur_overlap
        out[i] = opt_overlap
    return out


def explanation_epistemic_unc(train_class_info, dist_info, class_crit, class_info=None):
    '''Computes the explanatory epistemic uncertainty metric'''
    exp_epist_unc = np.zeros(dist_info.shape[0])
    for i in range(dist_info.shape[0]):
        #Get the current distances to each class-prototype
        cur_info  = np.copy(dist_info[i]) #(n_classes)
        
        #Determine which class choose as reference
        if class_crit == "distance":
            cur_min_c = np.argmin(cur_info)  #class with closest mean distance to the prototypes
        elif class_crit == "class":
            cur_min_c = np.argmax(class_info[i])  #class with closest mean distance to the prototypes
        else:
            sys.exit("class_crit options: ['distance', 'class']")
        
        #Compute and store the percentile
        cur_perc = np.sum(train_class_info[cur_min_c]<cur_info[cur_min_c]) / len(train_class_info[cur_min_c])
        exp_epist_unc[i] = np.copy(cur_perc)
    
    return exp_epist_unc