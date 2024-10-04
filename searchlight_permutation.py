#Implement permtuation / bootsraping statistics based on:
#Stelzer, J., Chen, Y., & Turner, R. (2013). Statistical inference and multiple testing correction in classification-based 
# multi-voxel pattern analysis (MVPA): random permutations and cluster size control. Neuroimage, 65, 69-82.

import numpy as np
import pandas as pd
import nibabel as nib
import nilearn
from scipy.ndimage import label
from nilearn.image import new_img_like

version_info = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/Version_Info.csv")
func_path = '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/decoding/permutation/'

null = np.zeros((len(version_info['sub']), 500, 2473)) #sub by permtation by voxel
num_subj = len(version_info['sub'])

for i, sub in enumerate(version_info['sub']):
    null[i,:,:] = np.load(func_path + "%s_context_decoding_permutation.npy" %sub)

#bootstrap conf interval
bootstrap_null = np.zeros((10000, 2473))
for i in range(10000):
    random_vector = np.random.randint(0, 500, size=num_subj)
    bootstrap_null[i, :] = null[np.arange(num_subj), random_vector, :].mean(axis=0)

#what is the cutoff?
thresh = np.percentile(bootstrap_null, 99.9, axis=0) #two tail test
mask = nib.load("/mnt/nfs/lss/lss_kahwang_hpc/ROIs/Morel_2.5_mask.nii.gz")


### below is parallel verion of the same code. MUCH FASTER!!
from joblib import Parallel, delayed

# Define the function to process each iteration
def process_iteration(i, bootstrap_null, thresh, mask):
    th_vec = bootstrap_null[i, :] > thresh
    null_img = nilearn.masking.unmask(th_vec, mask).get_fdata()
    x, _ = label(null_img)
    try:
        return np.max(np.bincount(x.flatten())[1:]) #skipping 1 because thats "0", everything outside the mask
    except:
        return 0
clust_null = Parallel(n_jobs=-1)(delayed(process_iteration)(i, bootstrap_null, thresh, mask) for i in range(10000))

#cluster size threhold?
cluster_thresh = np.percentile(clust_null, 99.9)

## now cluster the real data
context_decode = np.zeros((len(version_info['sub']), 2473)) 
for i, sub in enumerate(version_info['sub']):
    context_decode[i,:] = np.load(func_path + "%s_context_decoding.npy" %sub)
context_acc = context_decode.mean(axis=0) #the mean accuracy across subjects
clust_mask = context_acc > thresh #a mask of voxles exceeding the .001 threshold from the empirical null
tmp_img = nilearn.masking.unmask(clust_mask, mask).get_fdata() # put the mask back to 3D space
clusts, _ = label(tmp_img) # find connected clusters
sig_clusts = []
for i, c in enumerate(np.bincount(clusts.flatten())[1:]):
    if c > cluster_thresh:
        sig_clusts.append(i+1) #find sig clusters that exceeded the threshold size (corrected cluster size)

th_mask = np.zeros(clusts.shape)
for i in sig_clusts:
    th_mask += clusts==i 

# create the thresholded img, write to nii
context_decode_clustered_nii = new_img_like(mask, nilearn.masking.unmask(context_acc, mask).get_fdata() * th_mask)
context_decode_clustered_nii.to_filename('/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/decoding/context_decode_clustered_nii')

### the end
