import numpy as NP
from tqdm import tqdm as TQDM
from scipy.spatial.distance import cdist

def create_batchrange( n_data, n_jobs ):
    '''
    return index range [[0,5], [5,10], ... ]
    '''
    size = int(n_data/n_jobs)
    idxs = [size*n for n in range(0,n_jobs)]
    idxs.append(n_data)
    return [[idxs[n], idxs[n+1]] for n in range(n_jobs)]

def get_neighors_brutal(X1, X2, k, distance_type='euclidean', n_jobs=50, tqdm=False):
    '''
    Brutal method to find the k of closest X2 w.r.t X1
    X1 : interest points
    X2 : reference points
    '''
    batches = create_batchrange( len(X1), n_jobs )
    if tqdm:
        batches = TQDM(batches)

    distances = indexes = None
    for i, br in enumerate(batches):
        if tqdm:
            batches.set_description(">> ")
        d = cdist(X1[br[0]:br[1], NP.newaxis], X2, distance_type)
        idx = NP.argsort(d)[:,:k]
        d = d[NP.arange(len(d))[:, NP.newaxis], idx]
        if i == 0:
            distances = d
            indexes = idx
        else:
            distances = NP.concatenate((distances, d), axis=0)
            indexes = NP.concatenate((indexes, idx), axis=0)
    if tqdm: 
        batches.close()
    return distances, indexes
