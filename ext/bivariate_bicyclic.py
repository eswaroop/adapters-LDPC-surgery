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

# returns the parity check matrices of a bivariate bicycle code arXiv:2308.07915
# p1 = (a,b,c), p2 = (d,e,f)
# A = x^a + y^b + y^c
# B = y^d + x^e + x^f
def BB_code(ell=12,m=6,p1=(3,2,1),p2=(3,2,1),detailed=False):
    n = 2*ell*m
    a,b,c=p1
    d,e,f=p2
    # define cyclic shift matrices 
    I_ell = np.identity(ell,dtype=int)
    I_m = np.identity(m,dtype=int)
    I = np.identity(ell*m,dtype=int)
    x = np.kron(np.roll(I_ell,1,axis=1),I_m)
    y = np.kron(I_ell,np.roll(I_m,1,axis=1))
    
    # define parity check matrices
    A = (np.linalg.matrix_power(x,a) + np.linalg.matrix_power(y,b) + np.linalg.matrix_power(y,c)) % 2
    B = (np.linalg.matrix_power(y,d) + np.linalg.matrix_power(x,e) + np.linalg.matrix_power(x,f)) % 2

    AT = np.transpose(A)
    BT = np.transpose(B)

    hx = np.hstack((A,B))
    hz = np.hstack((BT,AT))
    
    if detailed:
        return A,B,hx,hz
    return hx,hz

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

# makes a single matrix independent
def make_indep_matrix(h):
    indx = 0
    while ((h[indx]==0).all()):  # incase initial rows are 0
        indx += 1
        
    h_indep = h[indx]
    num_rows = np.shape(h)[0]
    cur_rank = 1
    for i in range(indx+1,num_rows):              # skipping the first
        build_h = np.vstack((h_indep,h[i]))
        if cur_rank < rank2(build_h) :
            h_indep = build_h
            cur_rank = cur_rank + 1
    return h_indep,cur_rank