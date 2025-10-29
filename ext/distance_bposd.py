import numpy as np
from ldpc import bposd_decoder
from random import random

def null2(A):
    rows,n = A.shape
    X = np.identity(n,dtype=int)

    for i in range(rows):
        y = np.dot(A[i,:], X) % 2
        not_y = (y + 1) % 2
        good = X[:,np.nonzero(not_y)]
        good = good[:,0,:]
        bad = X[:, np.nonzero(y)]
        bad = bad[:,0,:]
        if bad.shape[1]>0 :
            bad = np.add(bad,  np.roll(bad, 1, axis=1) ) 
            bad = bad % 2
            bad = np.delete(bad, 0, axis=1)
            X = np.concatenate((good, bad), axis=1)
    # now columns of X span the binary null-space of A
    return np.transpose(X)


def rank2(A):
    return A.shape[1] - null2(A).shape[0]


def distance_upper_bound(gx,gz,trials=1000,max_iter=1024,osd_order=4):
    # Finds Z-type distance. Switch gx and gz for X-type distance.
    n = gz.shape[1]
    assert(n==gx.shape[1])
    gz_perp = null2(gz) # codewords of gz classical code alone (ie null space of check matrix)
    gx_perp = null2(gx)
    d = n
    for iter in range(trials):
        # pick a random logical-X operator and promote it to X-stabilizer
        while 1:
            random_logical_x = (np.random.randint(2,size=gz_perp.shape[0]) @ gz_perp) % 2 # x op wt= linear comb of gz code words (so commute)
            if np.count_nonzero(((gx_perp @ random_logical_x) % 2)>0): # but outside X-stab space HG=0 but here Gx alone != 0 so x not in H.(?)
                break
        augmented_gx = np.vstack([gx,random_logical_x])
        syndrome = np.zeros(augmented_gx.shape[0],dtype=int)
        syndrome[-1]=1 # we seek Z-error with the trivial stabilizer syndrome that anticommutes with random_logical_x
    
        bpd=bposd_decoder(
            augmented_gx,#the parity check matrix
            error_rate=0.01,# dummy error rate
            channel_probs=[None], #assign error_rate to each qubit. This will override "error_rate" input variable
            max_iter=max_iter, #the maximum number of iterations for BP)
            bp_method="ms",
            ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
            osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
            osd_order=osd_order #the osd search depth
            )
    
        bpd.decode(syndrome)
        low_weight_logical_z = bpd.osdw_decoding # Returns the recovery vector from the last round of BP+OSDW decoding.
        syndrome1 = (augmented_gx @ low_weight_logical_z) % 2
        assert(np.array_equal(syndrome,syndrome1))
        wt = np.count_nonzero(low_weight_logical_z)
        if wt>0:
            d = min(d,wt)
    return d