import nibabel as nib
from nilearn import image, maskers, masking, plotting
import nilearn.image
import numpy as np
from glob import glob
from scipy import stats, linalg
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import multiprocessing
from scipy.stats import zscore
import pandas as pd
import pickle
from math import sqrt


rois = '/Shared/lss_kahwang_hpc/ROIs/'
thalamus_mask = nib.load(rois+'Morel_2.5_mask.nii.gz')
cortex_mask = nib.load(rois+'Schaefer400_2.5.nii.gz')
cortex_masker = maskers.NiftiLabelsMasker(labels_img=cortex_mask, standardize=False)

#ususable subjects 59 subjects
subjects=['10001', '10002', '10003', '10004', '10005', 
'10008', '10010', '10012', '10013', '10014', 
'10017', '10018', '10019', '10020', '10022', 
'10023', '10024', '10025', '10027', '10028', 
'10031', '10032', '10033', '10034', '10035', 
'10036', '10037', '10038', '10039', '10040', 
'10041', '10042', '10043', '10044', '10054', 
'10057', '10058', '10059', '10060', '10063', 
'10064', '10066', '10068', '10069', '10071',
'10072', '10073', '10074', '10076', '10077',
'10080', '10162', '10169', '10170', '10173',
'10174', '10175', '10176', '10179']

tent_conditions={'EDS':5,
    'IDS':8,
    'Stay':11,
    'All_GLT':20}

def save_object(obj, filename):
	''' Simple function to write out objects into a pickle file
	usage: save_object(obj, filename)
	'''
	with open(filename, 'wb') as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
def read_object(filename):
	''' short hand for reading object because I can never remember pickle syntax'''
	o = pickle.load(open(filename, "rb"))
	return o


'''
extract the evoked response
inputs: zip(subjects, [thalamus_mask]*len(subjects), [cortex_masker]*len(subjects)))
'''

def run_tent_evoke(inputs):

    # three elements expected in input
    s = inputs[0] # subject name
    subcortical_mask = inputs[1]  # subocortical mask
    cortex_masker = inputs[2] # cortex masker

    #thalamus_mask = nib.load(rois+'Morel_2.5_mask.nii.gz')
    #cortex_mask = nib.load(rois+'Schaefer400_2.5.nii.gz')
    #cortex_masker = NiftiLabelsMasker(labels_img=cortex_mask, standardize=False)

    subcortical_mask_size = np.sum(subcortical_mask.get_fdata()>0)
    roi_size = len(np.unique(cortex_masker.labels_img.get_fdata()))-1

    subcortical_evoke_g1 = {}
    subcortical_evoke_g2 = {}
    
    ctx_evoke_g1 = {}
    ctx_evoke_g2 = {}

    for condition in tent_conditions:
        subcortical_evoke_g1[condition] = np.zeros((subcortical_mask_size)) #subject by time by voxel
        ctx_evoke_g1[condition] = np.zeros((roi_size)) #subject by time by cortical ROI
        subcortical_evoke_g2[condition] = np.zeros((subcortical_mask_size)) #subject by time by voxel
        ctx_evoke_g2[condition] = np.zeros((roi_size)) #subject by time by cortical ROI

    # correlation and partial correlation
    #fcmat[:, :] = generate_correlation_mat(thalamus_ts.T, cortex_ts.T) #th by ctx
    #fcmat[:, :] = pcorr_subcortico_cortical_connectivity(thalamus_ts, cortex_ts)[400:, 0:400]
    #fn = nib.load('/data/backed_up/shared/ThalHi_MRI_2020/3dDeconvolve/sub-{}/sub-{}_FIRmodel_MNI_stats_task_switch_REML.nii.gz'.format(s,s)).get_fdata().squeeze()
    fn_g1 = nib.load('/Shared/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/sub-{}/sub-{}_FIRmodel_MNI_stats_task_switch_r1_r4.nii.gz'.format(s,s)).get_fdata().squeeze()
    fir_hrf_g1 = image.new_img_like(subcortical_mask,fn_g1)
    subcortical_betas_g1=masking.apply_mask(fir_hrf_g1, subcortical_mask)
    ctx_betas_g1=cortex_masker.fit_transform(fir_hrf_g1)
    #Extract tha and cortical evoke
    for condition in tent_conditions.keys():
        
        subcortical_evoke_g1[condition][:] = subcortical_betas_g1[tent_conditions[condition],:]  #time by voxel
        ctx_evoke_g1[condition][:] = ctx_betas_g1[tent_conditions[condition],:]  #time by cortical ROI

    fn_g2 = nib.load('/Shared/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/sub-{}/sub-{}_FIRmodel_MNI_stats_task_switch_r5_r8.nii.gz'.format(s,s)).get_fdata().squeeze()
    fir_hrf_g2 = image.new_img_like(subcortical_mask,fn_g2)
    subcortical_betas_g2=masking.apply_mask(fir_hrf_g2, subcortical_mask)
    ctx_betas_g2=cortex_masker.fit_transform(fir_hrf_g2)
    #Extract tha and cortical evoke
    for condition in tent_conditions.keys():
        
        subcortical_evoke_g2[condition][:] = subcortical_betas_g2[tent_conditions[condition],:]  #time by voxel
        ctx_evoke_g2[condition][:] = ctx_betas_g2[tent_conditions[condition],:]  #time by cortical ROI


    return s, subcortical_evoke_g1, subcortical_evoke_g2, ctx_evoke_g1, ctx_evoke_g2

'''
calculate pca FC
inputs: zip(subjects,[thalamus_mask]*len(subjects), [roi_masker]*len(subjects)))
'''

def run_all_pca_fc(inputs):
    s=inputs[0]
    seed_mask=inputs[1]
    cortex_masker=inputs[2]
    print(s)
    fn1=nib.load('/Shared/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/sub-{}/sub-{}_FIRmodel_task_switch_errts_r1_r4.nii.gz'.format(s,s))
    ctx_ts1=masking.apply_mask(fn1,cortex_masker)
    thal_ts1=masking.apply_mask(fn1,seed_mask)

    fn2=nib.load('/Shared/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/sub-{}/sub-{}_FIRmodel_task_switch_errts_r5_r8.nii.gz'.format(s,s))
    ctx_ts2=masking.apply_mask(fn2,cortex_masker)
    thal_ts2=masking.apply_mask(fn2,seed_mask)

    #reduced_mat_g1,fc_mat_g1=cal_pca(ctx_ts1,thal_ts1,'max')
    #reduced_mat_g2,fc_mat_g2=cal_pca(ctx_ts2,thal_ts2,'max')

    # pcorr_mat1=pcorr_subcortico_cortical_connectivity(ctx_ts1,thal_ts1,'kai')
    # pcorr_mat2=pcorr_subcortico_cortical_connectivity(ctx_ts2,thal_ts2,'kai')


    mts = np.mean(ctx_ts1, axis = 1)
    if any(mts==0):
        del_i = np.where(mts==0)
        ctx_ts1 = np.delete(ctx_ts1, del_i, axis = 0)
        thal_ts1 = np.delete(thal_ts1, del_i, axis = 0)

    mts = np.mean(ctx_ts2, axis = 1)
    if any(mts==0):
        del_i = np.where(mts==0)
        ctx_ts2 = np.delete(ctx_ts2, del_i, axis = 0)
        thal_ts2 = np.delete(thal_ts2, del_i, axis = 0)

    ts_len1=ctx_ts1.shape[0]
    ts_len2=ctx_ts2.shape[0]
    ctx_size=ctx_ts1.shape[1]
    thalamus_size=thal_ts1.shape[1]
    fcmat1=np.zeros((thalamus_size,ctx_size))
    fcmat2=np.zeros((thalamus_size,ctx_size))

    n_comps1=np.amin([ts_len1-1, ctx_size-1, thalamus_size-1])
    print(n_comps1)
    n_comps2=np.amin([ts_len2-1, ctx_size-1, thalamus_size-1])
    print(n_comps2)


    pca=PCA(n_components=n_comps1,svd_solver='full')
    reduced_mat1=pca.fit_transform(thal_ts1)

    regrmodel = LinearRegression()
    reg1 = regrmodel.fit(reduced_mat1, ctx_ts1)
    fcmat1[:,:]=pca.inverse_transform(reg1.coef_).T


    pca=PCA(n_components=n_comps2,svd_solver='full')
    reduced_mat2=pca.fit_transform(thal_ts2)

    reg2 = regrmodel.fit(reduced_mat2, ctx_ts2)
    fcmat2[:,:]=pca.inverse_transform(reg2.coef_).T



    return s,reduced_mat1,reduced_mat2,fcmat1,fcmat2

'''
ses1: the first half observed data
ses2: the second half observed data
return: noise ceiling
'''

def cal_nc(ses1,ses2):
    y=np.r_[ses1,ses2]
    x=np.r_[ses2,ses1]

    vU=np.sum((x-np.mean(x))**2)
    vE=np.sum((y-x)**2)

    nc=np.sqrt(vU)/np.sqrt(vU+vE)

    return nc

'''
calculate activity flow analysis
inputs: 
the first&second session of seed evoked response
the first&second session of fc matrices
the first&second session of cortical evoked response
noise ceiling

return:
prediction accuracy of activity flow analysis
normalized by noise ceiling
'''

def cal_af(seed_evrs1,seed_evrs2,fc1,fc2,ctx_evrs1,ctx_evrs2,nc):
    seed_evrs1 = zscore(seed_evrs1)
    seed_evrs2 = zscore(seed_evrs2)

    pre_ctx1 = zscore(np.dot(seed_evrs1, fc1))
    pre_ctx2 = zscore(np.dot(seed_evrs2, fc2))

    af1 = np.corrcoef(ctx_evrs1,pre_ctx2)[0][1]
    af2 = np.corrcoef(ctx_evrs2,pre_ctx1)[0][1]

    af = (af1+af2)/2

    ncaf = af/nc

    return af, ncaf


'''
calcualte the simulated lesioned af

inputs:
pt: percentile
sub_thal_evrs1,sub_thal_evrs2: the first&second thalamic evoekd response
nc: noise ceiling
o_af: original nc af

return:
reduction in prediciton accuracy
'''

def simulated_lesion_af(pt,sub_thal_evrs1,sub_thal_evrs2,nc,o_af):
    #use absolute value
    percent_start1=np.percentile(abs(sub_thal_evrs1),pt)
    percent_end1=np.percentile(abs(sub_thal_evrs1),pt+20)

    percent_start2=np.percentile(abs(sub_thal_evrs2),pt)
    percent_end2=np.percentile(abs(sub_thal_evrs2),pt+20)

    index1=np.where((abs(sub_thal_evrs1)>percent_start1)&(abs(sub_thal_evrs1)<percent_end1))
    index2=np.where((abs(sub_thal_evrs2)>percent_start2)&(abs(sub_thal_evrs2)<percent_end2))


    sub_lesioned_thal1=np.where((abs(sub_thal_evrs1)>percent_start1)&(abs(sub_thal_evrs1)<percent_end1),0,sub_thal_evrs1)
    sub_lesioned_thal2=np.where((abs(sub_thal_evrs2)>percent_start2)&(abs(sub_thal_evrs2)<percent_end2),0,sub_thal_evrs2)

    predicted_lesioned_ctx1=zscore(np.dot(sub_lesioned_thal1,sub_fc1))
    predicted_lesioned_ctx2=zscore(np.dot(sub_lesioned_thal2,sub_fc2))

    corr1=np.corrcoef(predicted_lesioned_ctx1,sub_ctx_evrs2)[0][1]
    corr2=np.corrcoef(predicted_lesioned_ctx2,sub_ctx_evrs1)[0][1]
    
    af=(corr1+corr2)/2

    ncaf=af/nc

    reduct_ncaf=((ncaf-o_af)/o_af)*100

    return reduct_ncaf



##############################permutation test############################
'''
###generate the permuated fc matrix
fc is the input functional connetivity
'''
def permutation_fcpattern(fc,seed=None):
    if seed is not None:
        np.random.seed(seed)

    x=fc.shape[0]
    y=fc.shape[1]
    random_fc=np.zeros((x,y))
    f_fc=fc.flatten()
    np.random.shuffle(f_fc)
    random_fc[:,:]=f_fc.reshape(x,y)

    # return tfc,cfc
    return random_fc


def permutation_shuffle_seed_predicted_by_tsfc(inputs):
    p=inputs[0]

    fc_name1=inputs[1]

    fc_name2=inputs[2]

    fc_shape1=inputs[3]

    fc_shape2=inputs[4]

    fc1_shm=shared_memory.SharedMemory(name=fc_name1)
    fc2_shm=shared_memory.SharedMemory(name=fc_name2)

    fc1_data=np.ndarray(fc_shape1, buffer=fc1_shm.buf)
    fc2_data=np.ndarray(fc_shape2, buffer=fc2_shm.buf)


    ctx_name1=inputs[5]
    ctx_name2=inputs[6]

    ctx_shape1=inputs[7]
    ctx_shape2=inputs[8]

    reg=inputs[9]

    ctx1_shm=shared_memory.SharedMemory(name=ctx_name1)
    ctx2_shm=shared_memory.SharedMemory(name=ctx_name2)

    ctx1_data=np.ndarray(ctx_shape1, buffer=ctx1_shm.buf)
    ctx2_data=np.ndarray(ctx_shape2, buffer=ctx2_shm.buf)

    nc=pd.read_csv('/Shared/lss_kahwang_hpc/data/ThalHi/Activity_Flow/noise_ceiling/400ROIs_voxels/rsa_af/af/{}_59subs_nc_max.csv'.format(reg))

    conditions=['EDS','IDS','Stay','All_GLT']

    ncaf=np.zeros((len(subjects)))

    null_data={}

    for c,con in enumerate(conditions):
        seed_evrs1=np.load('/Shared/lss_kahwang_hpc/data/ThalHi/Activity_Flow/noise_ceiling/400ROIs/evrs/{}_59subs_run14_thal_evrs.npy'.format(con))
        seed_evrs2=np.load('/Shared/lss_kahwang_hpc/data/ThalHi/Activity_Flow/noise_ceiling/400ROIs/evrs/{}_59subs_run58_thal_evrs.npy'.format(con))

        for s in range(len(subjects)):
            #shuffle the thalamic evoked response
            np.random.shuffle(seed_evrs1[s,:])
            sub_seed_evrs1=zscore(seed_evrs1[s,:])

            np.random.shuffle(seed_evrs2[s,:])
            sub_seed_evrs2=zscore(seed_evrs2[s,:])

            ctx_evrs1=ctx1_data[c,s,:]
            ctx_evrs2=ctx2_data[c,s,:]

            subfc1=fc1_data[s]
            subfc2=fc2_data[s]

            sub_nc=nc[con][s]

            pre_evrs1=np.dot(sub_seed_evrs1,subfc1)
            pre_evrs2=np.dot(sub_seed_evrs2,subfc2)

            corr1=np.corrcoef(ctx_evrs1,pre_evrs2)[0][1]
            corr2=np.corrcoef(ctx_evrs2,pre_evrs1)[0][1]

            con_af=(corr1+corr2)/2

            ncaf[s]=con_af/sub_nc

        null_data[con]=(np.mean(ncaf))
    
    return null_data



def permutation_shuffle_tsfc_predicted_by_tsfc(args,seed=None):

    s, ctx_data1, ctx_data2, subfc1, subfc2=args

    nc=pd.read_csv('/Shared/lss_kahwang_hpc/data/ThalHi/Activity_Flow/noise_ceiling/400ROIs_voxels/rsa_af/af/{}_59subs_nc_max.csv'.format(rsa))

    conditions=['EDS','IDS','Stay','All_GLT']

    null_data={}

    for c,con in enumerate(conditions):
        seed_evrs1=np.load('/Shared/lss_kahwang_hpc/data/ThalHi/Activity_Flow/noise_ceiling/400ROIs/evrs/{}_59subs_run14_thal_evrs.npy'.format(con))
        seed_evrs2=np.load('/Shared/lss_kahwang_hpc/data/ThalHi/Activity_Flow/noise_ceiling/400ROIs/evrs/{}_59subs_run58_thal_evrs.npy'.format(con))

        sub_seed_evrs1=zscore(seed_evrs1[s,:])
        sub_seed_evrs2=zscore(seed_evrs2[s,:])

        
        ctx_evrs1=ctx_data1[c,:]
        ctx_evrs2=ctx_data2[c,:]

        #randomly shuffle the fc matrix
        sub_fc1=permutation_fcpattern(subfc1,seed)
        sub_fc2=permutation_fcpattern(subfc2,seed)

        sub_nc=nc[con][s]

        pre_evrs1=np.dot(sub_seed_evrs1,sub_fc1)
        pre_evrs2=np.dot(sub_seed_evrs2,sub_fc2)

        corr1=np.corrcoef(ctx_evrs1,pre_evrs2)[0][1]
        corr2=np.corrcoef(ctx_evrs2,pre_evrs1)[0][1]

        con_af=(corr1+corr2)/2

        null_data[con]=con_af/sub_nc
    
    return null_data

