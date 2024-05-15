"""
Contributors: Stephanie C Leach, Kai Hwang, Xitong Chen, Evan Sorenson
Last Edited: 01/26/2024

How to run:
  python3 compute_rsa_mri.py 

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - -         import/load packages/functions          - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# -- import most common packages
import sys
import os
import argparse
import pickle
import glob
import numpy as np
import datetime
import multiprocessing
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
# -- import mri packages
import nibabel as nib
from nilearn.image import resample_to_img
from nilearn.input_data import NiftiMasker
from nilearn import masking
from nilearn.image.resampling import coord_transform
#from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity
from nilearn.maskers.nifti_spheres_masker import _apply_mask_and_get_affinity
from nilearn._utils import check_niimg_4d
import nilearn.masking
# -- import more rsa specific packages/functions
import rsatoolbox


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - -          set up for argument parsing            - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def init_argparse() -> argparse.ArgumentParser:
    # -- set up parser
    parser = argparse.ArgumentParser(
        description="run RSA on MRI data",
        usage="[project_path] [data_path] [output_path] [OPTIONS] ... ",
    )
    # -- add required, fixed place arguments
    parser.add_argument("project_path", help="absolute path for the project folder ... Example: /Shared/lss_kahwang_hpc/data/ThalHi/ ... should contain a folder named RSA")
    # parser.add_argument("data_path", help="absolute path to input data for RSA ... Example: /Shared/lss_kahwang_hpc/data/ThalHi/RSA/thalamus/")
    # parser.add_argument("output_path", help="absolute path indicating where to save output data from RSA ... Example: /Shared/lss_kahwang_hpc/data/ThalHi/RSA/thalamus/results/ ")
    # -- add required, but not fixed place arguments
    parser.add_argument("--njobs", 
                        help="number of cores to use: int between 1 and 80")
    parser.add_argument("--nbatches", 
                        help="number of batches to use: int between 1 and 5")
    parser.add_argument("--batch_num", 
                        help="which batch number to use: int between 1 and 5")
    # -- add optional true/false arguments for deciding if cortical or subcortical (and what structures if subcortical)
    parser.add_argument("--use_cortical",
                        help="only use cortical voxels in RSA, default is false",
                        default=False, action="store_true")
    parser.add_argument("--use_subcortical",
                        help="only use subcortical voxels in RSA, default is false",
                        default=False, action="store_true")
    parser.add_argument("--thalamus",
                        help="only use thalamus subcortical voxels in RSA, default is false",
                        default=False, action="store_true")
    parser.add_argument("--basal_ganglia",
                        help="only use basal ganglia subcortical voxels in RSA, default is false",
                        default=False, action="store_true")
    # -- add optional true/false arguments for parcellation method ... ONLY CONSIDERS THESE IF --run_cortical == TRUE and --run_subcortical == FALSE
    parser.add_argument("--schaefer_400rois",
                        help="run with schaefer 400 rois, default is false",
                        default=False, action="store_true")
    parser.add_argument("--searchlight",
                        help="run with searchlight, default is false",
                        default=False, action="store_true")
    parser.add_argument("--mult_comp_corr",
                        help="include multiple comparisons correction with searchlight, default is false",
                        default=False, action="store_true")
    parser.add_argument("--gen_roi_plot",
                        help="generate 400 roi brain surface plot, default is false",
                        default=False, action="store_true")
    parser.add_argument("--gen_searchlight_plots",
                        help="generate 400 roi brain surface plot, default is false",
                        default=False, action="store_true")
    
    return parser

# --------  run parser to pull arguments  -------- # 
parser = init_argparse()
args = parser.parse_args(sys.argv[1:])
# -- pull required, fixed place arguments
project_dir = args.project_path
path_folders = project_dir.split("/")
if path_folders[-1] == '':
    project_name = path_folders[-2]
else:
    project_name = path_folders[-1]
# data_dir = args.data_path
# out_dir = args.output_path
# -- pull required, but not fixed place arguments
num_cores = int(args.njobs)
nbatches = int(args.nbatches)
batch_num = int(args.batch_num)
# -- pull optional true/false arguments for deciding if cortical or subcortical 
use_cortical = args.use_cortical
use_subcortical = args.use_subcortical # redundant, but I think it makes it more user friendly
thalamus = args.thalamus
basal_ganglia = args.basal_ganglia
# -- pull optional true/false arguments for parcellation method
use_schaefer_400rois = args.schaefer_400rois
use_searchlight = args.searchlight



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - -        define useful functions (common)         - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - -          define useful functions (MRI)          - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ----- LOADING MRI FILES/DATA ------ #
def load_mask(mask_file):
    print("\n\nPulling mask file from ... ", mask_file)
    mask = nib.load(mask_file)
    mask_data = mask.get_fdata()
    print(mask_data.shape)
    return mask, mask_data

# ----- MASKING FUNCTIONS ------ #
def get_binary_mask(mask_data,voxels_to_exclude,roi):
    print("number of voxels in current ROI mask:",np.where(mask_data==roi,1,0).sum())
    voxels_orginal = np.where(mask_data==roi,1,0).sum()
    mask_binary=np.where(np.logical_and(mask_data==roi,voxels_to_exclude==0),1,0) # make sure to also exclude voxels with all zeros
    print("number of usable voxels in current ROI mask:",mask_binary.sum(),"\n")
    voxels_usable = mask_binary.sum()
    return mask_binary, voxels_orginal, voxels_usable

def apply_mask(data_file,mask_binary):
    #print("data size: ", data_file.get_fdata().shape, "\napplying mask ...")
    mask_binary_nif=nilearn.image.new_img_like(data_file, mask_binary)
    masked_data=nilearn.masking.apply_mask(data_file,mask_binary_nif)
    #masked_data=input_data.NiftiMasker(mask_binary_nif)
    #print("masked data is a numpy array of size: ", masked_data.shape)
    return masked_data


# ----- USABLE/EXLUDE FUNCTIONS ------ #
def get_voxels_to_exclude(resids_file):
    print("loading ... ", resids_file)
    r=nib.load(resids_file)
    print(r.shape)
    # check for zeros
    r_data = r.get_fdata()
    voxels_to_exclude = np.zeros((r_data.shape[0], r_data.shape[1], r_data.shape[2])) # initialize 3D matrix of what voxels to exclude
    for x in range(r_data.shape[0]):
        for y in range(r_data.shape[1]):
            for z in range(r_data.shape[2]):
                if r_data[x,y,z,:].sum()==0:
                    # voxel had 0 for all time points ... exclude from further analysis
                    voxels_to_exclude[x,y,z]=1
    print("A total of",voxels_to_exclude.sum(),"voxels will be EXCLUDED due to 0s for all time points")
    print(voxels_to_exclude.sum(),"voxels exluded out of",(r_data.shape[0]*r_data.shape[1]*r_data.shape[2]),"total voxels")
    #proportion_excluded=voxels_to_exclude.sum()/(r_data.shape[0]*r_data.shape[1]*r_data.shape[2])
    print((voxels_to_exclude.sum()/(r_data.shape[0]*r_data.shape[1]*r_data.shape[2])),"proportion of voxels excluded\n")
    return voxels_to_exclude

def remove_censored_data(resids_file, mask_binary):
    #    CHECK FOR CENSORED PERIODS IN VOXELS and remove censored portions
    #    note, VOXELSxCUES will be 2D [voxels in current ROI x 8 cues] AND this will identify periods of time where voxels are censored (should have some usable times)
    print("loading ... ", resids_file)
    r=nib.load(resids_file)
    print(r.shape)
    #    FIRST - apply mask to resids file
    mask_binary_nif=nilearn.image.new_img_like(r,mask_binary)
    resids_data=nilearn.masking.apply_mask(r,mask_binary_nif) 
    print("masked residual data is a numpy array of size: ", resids_data.shape) # gives you [time by voxels]
    #    SECOND - look through good voxels and find censored times and remove them
    reduced_resids_data=[]
    for tpt in range(resids_data.shape[0]):
        if resids_data[tpt,1]!=0:
            reduced_resids_data.append(resids_data[tpt,:])
    reduced_resids_data=np.array(reduced_resids_data)
    print("masked residual data is NOW a numpy array of size: ", reduced_resids_data.shape)
    return reduced_resids_data

def remove_censored_data_updated(resids_data):
    #    CHECK FOR CENSORED PERIODS IN VOXELS and remove censored portions
    #    note, VOXELSxCUES will be 2D [voxels in current ROI x 8 cues] AND this will identify periods of time where voxels are censored (should have some usable times)
    #print("loading ... ", resids_file)
    #r=nib.load(resids_file)
    #print(r.shape)
    #    FIRST - apply mask to resids file
    #mask_binary_nif=nilearn.image.new_img_like(r,mask_binary)
    #resids_data=nilearn.masking.apply_mask(r,mask_binary_nif) 
    print("masked residual data is a numpy array of size: ", resids_data.shape) # gives you [time by voxels]
    #    SECOND - look through good voxels and find censored times and remove them
    reduced_resids_data=[]
    for tpt in range(resids_data.shape[0]):
        if resids_data[tpt,1]!=0:
            reduced_resids_data.append(resids_data[tpt,:])
    reduced_resids_data=np.array(reduced_resids_data)
    print("masked residual data is NOW a numpy array of size: ", reduced_resids_data.shape)
    return reduced_resids_data
    
def get_voxels_to_exclude_SL_version(r_data):
    voxels_to_exclude = np.zeros((r_data.shape[0], r_data.shape[1], r_data.shape[2])) # initialize 3D matrix of what voxels to exclude
    for x in range(r_data.shape[0]):
        for y in range(r_data.shape[1]):
            for z in range(r_data.shape[2]):
                if r_data[x,y,z,:].sum()==0:
                    # voxel had 0 for all time points ... exclude from further analysis
                    voxels_to_exclude[x,y,z]=1
    print("A total of",voxels_to_exclude.sum(),"voxels will be EXCLUDED due to 0s for all time points")
    print(voxels_to_exclude.sum(),"voxels exluded out of",(r_data.shape[0]*r_data.shape[1]*r_data.shape[2]),"total voxels")
    #proportion_excluded=voxels_to_exclude.sum()/(r_data.shape[0]*r_data.shape[1]*r_data.shape[2])
    print((voxels_to_exclude.sum()/(r_data.shape[0]*r_data.shape[1]*r_data.shape[2])),"proportion of voxels excluded\n")
    return voxels_to_exclude

def remove_censored_data_noise_version(resids_data):
    # resids_data dim order should be [trs, voxels]
    if resids_data.shape[0]>20000:
        print("dimension order incorrect\ttransposing now\tnew shape will be transpose of",resids_data.shape)
        resids_data = resids_data.T
    resid_vec = np.mean(resids_data,axis=1)
    reduced_resids_data = resids_data[resid_vec!=0,:] # vectorize it, faster!
    # reduced_resids_data=[]
    # for tpt in range(resids_data.shape[0]):
    #     if resid_vec[tpt]!=0:
    #         reduced_resids_data.append(resids_data[tpt,:])
    # reduced_resids_data=np.array(reduced_resids_data)
    #print(“masked residual data is NOW a numpy array of size: “, reduced_resids_data.shape)
    # reduced_resids_data dim order should be trs x voxels
    return reduced_resids_data

# ----- FORMATTING/CONVERSION FUNCTIONS ------ #
def make_rsatoolbox_format(CONDxVOXELS, sub, roi, stats_method, cue_list, num_runs):
    #    ---------------------------------------------
    #    descriptors: subject = #
    #                 ROI = #
    #    channel_descriptors: voxels = ['voxel_0' 'voxel_1' 'voxel_2' ... 'voxel_N']
    #    obs_descriptors: measure = ['condition_0' 'condition_1' ... 'condition_N']
    #    noise: 
    #       * will need for mahalanobis or crossnobis options
    #       * can be the residual file from the regression
    #    cv_descriptor: trials = [0 1 2 3 4 0 1 2 3 4 ... 0 1 2 3 4]
    #       * will need for crossnobis option (tells us what trial a value came from)
    #       * will be same length as obs_descriptors and channel_descriptors
    #    ---------------------------------------------
    if (stats_method=="Mahalanobis" or stats_method=="Correlation"):
        des={'subject': sub, 'ROI': roi}
        #print(des)
        chn_des={'voxels': np.array(['voxel_' + str(x) for x in np.arange(CONDxVOXELS.shape[1])])} # voxels or electrodes
        #print(chn_des)
        obs_des={'measure': np.array(['trl_' + str(f"{x:02d}") for x in np.arange(CONDxVOXELS.shape[0])])} # our conditions to be compared
        #print(obs_des)
        #print("input CONDxVOXELS shape: ",CONDxVOXELS.shape)
        rsatoolbox_formatted_data=rsatoolbox.data.Dataset(CONDxVOXELS, descriptors=des, obs_descriptors=obs_des, channel_descriptors=chn_des)
    elif stats_method=="Crossnobis":
        # Note, CONDxVOXELS will still be 2D, but the cue vectors from the runs will be separated (aka the number of conditions will be cues*num_runs)
        des={'subject': sub, 'ROI': roi}
        single_run_chn_des = np.array(['voxel_' + str(x) for x in np.arange(CONDxVOXELS.shape[1])]) # will need to repeat as many times as we have runs
        #print("single_run_chn_des", len(single_run_chn_des))
        single_run_measure = np.array(['cue_' + str(f"{x:02d}") for x in np.arange(int(CONDxVOXELS.shape[0]/num_runs))])
        #print("single_run_measure", len(single_run_measure))
        chn_des={'voxels': single_run_chn_des} # np.tile(single_run_chn_des, num_runs)
        obs_des={'measure': np.tile(single_run_measure, num_runs), 'runs': np.repeat( range(num_runs), int(CONDxVOXELS.shape[0]/num_runs) )} 
        #print("single_run_measure tiled", len(np.tile(single_run_measure, num_runs)))
        #print("runs tiled", len( np.tile( np.arange(int(CONDxVOXELS.shape[0]/num_runs)), num_runs )))
        #print("input CONDxVOXELS shape: ",CONDxVOXELS.shape)
        rsatoolbox_formatted_data=rsatoolbox.data.Dataset(CONDxVOXELS, descriptors=des, obs_descriptors=obs_des, channel_descriptors=chn_des)
    return rsatoolbox_formatted_data

def convert_numpy_to_niftii(numpy_formatted_data, niftii_to_use_as_template):
    nif_format = nilearn.image.new_img_like(niftii_to_use_as_template,numpy_formatted_data)
    return nif_format


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - -          define useful functions (RSA)          - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ---- define function for parallel processing ... will generate input for RSA for subcortical
def parallel_gen_input_matrices(job_id, inputs, stats_method, subject_list, cue_list, out_dir):
    roi_name = inputs[0]
    roi = inputs[1]
    roi_binary_mask = inputs[2]
    print("current roi name: ", roi_name)
    # ---- load mask files
    mask_file = roi_binary_mask[roi-1]
    mask = nib.load(mask_file)
    mask_dims = [mask.affine[0,0], mask.affine[1,1], mask.affine[2,2], mask.affine[3,3]]
    mask_data = mask.get_fdata()
    if os.path.basename(mask_file) == "Schaefer400_2.5.nii.gz":
        if (int(roi) != int(roi_name)):
            print("ERROR!!! roi and roi_name do not match!!", roi, roi_name)
            "c"+2
        else:
            mask_binary = np.where(mask_data==roi,1,0)
    else:
        mask_binary = np.where(mask_data==1,1,0)
    del mask_data
    del mask
    # --------  LOOP through subjects  -------- #
    for idx, sub in enumerate(subject_list):
        sub_list.append(sub) # keep track of subject order
        # ---- load residuals file and run some checks (for zeros)  ...  note, this should identify voxels that were outside of brain and always 0
        resids_file = os.path.join(project_dir, "3dDeconvolve_fdpt4", ("sub-"+sub), ("sub-"+sub+"_FIRmodel_errts_8cues2resp.nii.gz"))
        # voxels_to_exclude = get_voxels_to_exclude(resids_file) # can't run here because then voxel nums will vary across participants
        # -- pull out current roi from mask AND get usable voxels
        #mask_binary, vo, vu = get_binary_mask(mask_data, voxels_to_exclude, 1)
        # -- if first subject, initialize empty matrices to fill in now that we know what voxels to exclude
        if idx==0:
            resids_data_array = np.zeros( (1728, (mask_binary==1).sum() , len(subject_list)) ) # hardcode 1728 time points
            VOXELSxCUESxSUBS=np.zeros( ((mask_binary==1).sum() , len(cue_list), len(subject_list)) ) # will be 3 dimensions [voxels, cues-resp, subjs]
        # also load residuals file for this subject and mask so it's just the voxels
        print("\nloading residuals file for subject ",sub)
        resids_data_nii = nib.load(resids_file)
        resids_masked = apply_mask(resids_data_nii, mask_binary)
        del resids_data_nii
        resids_data_array[:resids_masked.shape[0],:resids_masked.shape[1],idx] = resids_masked
        del resids_masked
        # ---- load cue files for current subject  ...  loop through cues and load and save them in larger matrix
        print("\nLoading cue files for subject ",sub)
        for ind_c, cue in enumerate(cue_list):
            data_file = nib.load(os.path.join(project_dir, "3dDeconvolve_fdpt4", ("sub-"+sub), ("sub-"+sub+"_"+cue+"_FIR_MNI.nii.gz")))
            # -- quick check to make sure dimensions match
            data_dims = [data_file.affine[0,0], data_file.affine[1,1], data_file.affine[2,2], data_file.affine[3,3]]
            if mask_dims != data_dims:
                print("Error!!! Mask and data file dimensions DO NOT MATCH!!!")
                "c"+2
            masked_data = apply_mask(data_file, mask_binary)
            del data_file
            masked_data_avg=np.mean(masked_data, axis=0) # take average over masked data
            del masked_data
            VOXELSxCUESxSUBS[:,(cue_list.index(cue)),idx] = masked_data_avg # add 1D masked data vector (avg) to my numpy matrix of size VOXELSxCUES
            del masked_data_avg
    # ---- now that we've generated the voxels by cues-resp by subjects matrix, save it out in case we want to save time later
    if os.path.basename(mask_file) == "Schaefer400_2.5.nii.gz":
        np.save(os.path.join(out_dir,("statmethod-"+stats_method+"_roi-"+roi_name+"_VOXELSxCUESxSUBS.npy")), VOXELSxCUESxSUBS)
        np.save(os.path.join(out_dir,("statmethod-"+stats_method+"_roi-"+roi_name+"_residuals.npy")), resids_data_array)
    else:
        np.save(os.path.join(out_dir,("statmethod-"+stats_method+"_roi-"+roi_name+"_VOXELSxCUESxSUBS.npy")), VOXELSxCUESxSUBS)
        np.save(os.path.join(out_dir,("statmethod-"+stats_method+"_roi-"+roi_name+"_residuals.npy")), resids_data_array)
    
    # ---- delete files to save space will parallel processing
    del VOXELSxCUESxSUBS
    del resids_data_array
    
    # nothing to return
    
def parallel_RSA_func_cortical(job_id, inputs):
    # input_list = [0:roi_name, 1:roi_num, 2:cur_roi_inds, 3:resids_masked_list, 4:nii_masked_list, 5:stats_method, 6:subject_list, 7:cue_list, 8:out_dir]
    roi_name = inputs[0] # roi string
    roi = inputs[1] # roi number
    cur_roi_inds = inputs[2] # masked roi data
    resids_masked_list = inputs[3]
    nii_masked_list = inputs[4]
    stats_method = inputs[5]
    subject_list = inputs[6]
    cue_list = inputs[7]
    out_dir = inputs[8]
    print("current roi name: ", roi_name)
    print("\nnumber of voxels in current ROI mask:", len(cur_roi_inds))
    
    # -- initialize 
    SUBxCOEFF_mat = np.zeros((len(subject_list), len(cue_list), len(cue_list)))
    # --------  LOOP through subjects  -------- #
    for idx, sub in enumerate(subject_list):
        # load voxels by cues for current subject and current roi
        VOXELSxCUES = nii_masked_list[idx]
        print("voxels in current ROI:", VOXELSxCUES.shape[0], "\tVOXELSxCUES matrix dimensions:", VOXELSxCUES.shape)
        # -- Now that the entire Cue by Voxel matrix has been filled in, calculate correlation coefficients
        if (stats_method=="Pearson"):
            Coeff_mat=np.corrcoef(VOXELSxCUES, rowvar=False) # use pearson correlation
        elif (stats_method=="Spearman"):
            Coeff_mat=stats.spearmanr(VOXELSxCUES,axis=0)[0] # use spearman correlation
        elif (stats_method=="Mahalanobis"):
            data_rsatool_format = make_rsatoolbox_format(VOXELSxCUES.T, sub, roi, stats_method, cue_list, 1) # function expects CUESxVOXELS so we have to transpose 
            reduced_resids_data = remove_censored_data_updated(resids_masked_list[idx])
            #print("reduced_resids_data looks like \n",reduced_resids_data, "\nwith a shape of ", reduced_resids_data.shape, "\nand data_rsatool_format looks like\n", data_rsatool_format)
            # first try to calculate the noise measure
            try:
                noise_pres_res = rsatoolbox.data.noise.prec_from_residuals(reduced_resids_data, method='diag') # get noise information for mahalanobis option
                print("\nnoise_pres_res shape: ",noise_pres_res.shape)
            except Exception as e:
                noise_pres_res = np.identity(VOXELSxCUES.shape[0])
                print("Encountered exception: ", e, "\n\nnoise_pres_res not properly calculated for ROI ", roi_name, " and set as identity matrix instead")
            try: 
                rdm=rsatoolbox.rdm.calc_rdm(data_rsatool_format, method='mahalanobis', descriptor='measure', noise=noise_pres_res) # run rsatoolbox to get coeff matrix
                rdm.sort_by(measure='alpha') # make sure order is okay
                print(rdm)
                Coeff_mat=rdm.get_matrices()[0] # used to do "Coeff_mat=1-(rdm.get_matrices()[0])" BUT then found out upper bound is NOT 1...
                #print("\n\nCoeff_mat as a distance matrix\n\n",(Coeff_mat),"\nwith a shape of ", Coeff_mat.shape, "\n\n Coefficient matrix calculated for ROI", roi_name)
            except Exception as e:
                CUESxVOXELS=VOXELSxCUES.T
                Coeff_mat = np.empty((CUESxVOXELS.shape[0],CUESxVOXELS.shape[0]))
                Coeff_mat[:] = np.nan
                print("\nEncountered exception: ", e, "\n Coefficient matrix NOT calculated for ROI", roi_name, "\nINVERSION ERROR ... coefficient matrix saved as NaN matrix and ROI recorded")
        # -- now add current roi and current subject into the ALL SUBJECTS matrix 
        SUBxCOEFF_mat[idx,:,:] = Coeff_mat
    # -- save results for current ROI
    curROImat = SUBxCOEFF_mat[:,:,:]
    #np.save(os.path.join(project_dir,"RSA","subcortical","results",("statmethod-"+stats_method+"_roi-"+roi_name+"_"+str(len(sub_list))+"subjectsCoeffMatrix.npy")), curROImat)
    np.save(os.path.join(out_dir,("statmethod-"+stats_method+"_roi-"+roi_name+"_"+str(len(subject_list))+"subjectsCoeffMatrix.npy")), curROImat)
    
    # # --------  LOOP through subjects  -------- #
    # resids_data_array = np.zeros( (1728, len(cur_roi_inds) , len(subject_list)) ) # hardcode 1728 time points
    # VOXELSxCUESxSUBS=np.zeros( (len(cur_roi_inds) , len(cue_list), len(subject_list)) ) # will be 3 dimensions [voxels, cues-resp, subjs]
    # for idx, sub in enumerate(subject_list):
    #     resids_data_array[:resids_masked_list[idx].shape[0], :resids_masked_list[idx].shape[1], idx] = resids_masked_list[idx]
    #     VOXELSxCUESxSUBS[:,:,idx] = nii_masked_list[idx] # add 1D masked data vector (avg) to my numpy matrix of size VOXELSxCUES
    # # ---- now that we've generated the voxels by cues-resp by subjects matrix, save it out in case we want to save time later
    # if os.path.basename(mask_file) == "Schaefer400_2.5.nii.gz":
    #     np.save(os.path.join(out_dir,("statmethod-"+stats_method+"_roi-"+roi_name+"_VOXELSxCUESxSUBS.npy")), VOXELSxCUESxSUBS)
    #     np.save(os.path.join(out_dir,("statmethod-"+stats_method+"_roi-"+roi_name+"_residuals.npy")), resids_data_array)
    # else:
    #     np.save(os.path.join(out_dir,("statmethod-"+stats_method+"_roi-"+roi_name+"_VOXELSxCUESxSUBS.npy")), VOXELSxCUESxSUBS)
    #     np.save(os.path.join(out_dir,("statmethod-"+stats_method+"_roi-"+roi_name+"_residuals.npy")), resids_data_array)
    
    # # ---- delete files to save space will parallel processing
    # del VOXELSxCUESxSUBS
    # del resids_data_array
    
    # nothing to return
    
# ~~~~~~~~~~~~~~~~~~~~~~~ SET UP PARALLEL PROCESSING for subcortical RSA ~~~~~~~~~~~~~~~~~~~~~~~
# split a list into evenly sized chunks
def chunks(list_of_VoxelsXcuesXsubs, list_of_Resids, roi_name_list, num_chunks):
    # create an input list within the slice list
    out_list = []
    for roi, roi_name in enumerate(roi_name_list):
        out_list.append([list_of_VoxelsXcuesXsubs[roi], list_of_Resids[roi], roi_name, roi])
    return out_list

# function to run in the parallel code
def parallel_RSA_func(job_id, inputs, stats_method, cue_list, out_dir):
    # re-define these here
    # cue_list = ["r1_dcb", "r1_fcb", "r1_dpb", "r1_fpb", "r1_dcr", "r1_fcr", "r1_dpr", "r1_fpr",
    #             "r2_dcb", "r2_fcb", "r2_dpb", "r2_fpb", "r2_dcr", "r2_fcr", "r2_dpr", "r2_fpr"]
    # roi_name_list = ['thalamus','AN','VL','MD','LP','IL','VA','PuM','PuL','VP']
    # pull out inputs
    VOXELSxCUESxSUBJS = inputs[0]
    resids_data_array = inputs[1]
    roi_name = inputs[2]
    roi = inputs[3]
    # -- initialize 
    SUBxCOEFF_mat = np.zeros((len(sub_list), len(cue_list), len(cue_list)))
    # --------  LOOP through subjects  -------- #
    for idx, sub in enumerate(sub_list):
        # load voxels by cues for current subject and current roi
        VOXELSxCUES=VOXELSxCUESxSUBJS[:,:,idx]
        print("voxels in current ROI:", VOXELSxCUES.shape[0], "\tVOXELSxCUES matrix dimensions:", VOXELSxCUES.shape)
        # -- Now that the entire Cue by Voxel matrix has been filled in, calculate correlation coefficients
        if (stats_method=="Pearson"):
            Coeff_mat=np.corrcoef(VOXELSxCUES, rowvar=False) # use pearson correlation
        elif (stats_method=="Spearman"):
            Coeff_mat=stats.spearmanr(VOXELSxCUES,axis=0)[0] # use spearman correlation
        elif (stats_method=="Mahalanobis"):
            data_rsatool_format = make_rsatoolbox_format(VOXELSxCUES.T, sub, roi, stats_method, cue_list, 1) # function expects CUESxVOXELS so we have to transpose 
            reduced_resids_data = remove_censored_data_updated(resids_data_array[:,:,idx])
            #print("reduced_resids_data looks like \n",reduced_resids_data, "\nwith a shape of ", reduced_resids_data.shape, "\nand data_rsatool_format looks like\n", data_rsatool_format)
            # first try to calculate the noise measure
            try:
                noise_pres_res = rsatoolbox.data.noise.prec_from_residuals(reduced_resids_data, method='diag') # get noise information for mahalanobis option
                print("\nnoise_pres_res shape: ",noise_pres_res.shape)
            except Exception as e:
                noise_pres_res = np.identity(VOXELSxCUES.shape[0])
                print("Encountered exception: ", e, "\n\nnoise_pres_res not properly calculated for ROI ", roi_name, " and set as identity matrix instead")
            try: 
                rdm=rsatoolbox.rdm.calc_rdm(data_rsatool_format, method='mahalanobis', descriptor='measure', noise=noise_pres_res) # run rsatoolbox to get coeff matrix
                rdm.sort_by(measure='alpha') # make sure order is okay
                print(rdm)
                Coeff_mat=rdm.get_matrices()[0] # used to do "Coeff_mat=1-(rdm.get_matrices()[0])" BUT then found out upper bound is NOT 1...
                #print("\n\nCoeff_mat as a distance matrix\n\n",(Coeff_mat),"\nwith a shape of ", Coeff_mat.shape, "\n\n Coefficient matrix calculated for ROI", roi_name)
            except Exception as e:
                CUESxVOXELS=VOXELSxCUES.T
                Coeff_mat = np.empty((CUESxVOXELS.shape[0],CUESxVOXELS.shape[0]))
                Coeff_mat[:] = np.nan
                print("\nEncountered exception: ", e, "\n Coefficient matrix NOT calculated for ROI", roi_name, "\nINVERSION ERROR ... coefficient matrix saved as NaN matrix and ROI recorded")
        # -- now add current roi and current subject into the ALL SUBJECTS matrix 
        SUBxCOEFF_mat[idx,:,:] = Coeff_mat
    # -- save results for current ROI
    curROImat = SUBxCOEFF_mat[:,:,:]
    #np.save(os.path.join(project_dir,"RSA","subcortical","results",("statmethod-"+stats_method+"_roi-"+roi_name+"_"+str(len(sub_list))+"subjectsCoeffMatrix.npy")), curROImat)
    np.save(os.path.join(out_dir,("statmethod-"+stats_method+"_roi-"+roi_name+"_"+str(len(sub_list))+"subjectsCoeffMatrix.npy")), curROImat)
    
    # nothing to return

# data is the input list before chunking, job number is the number of jobs to run in parallel

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def set_up_model_vars(CUES):
    regressor_list=["Identity", "Context", "Color", "Shape", "Task_Performed", "Resp"]
    stat_list=["_Beta","_T-stat","_P-value"]
    #print("\nGenerating column headers based on given regressors and stats to save out ... \n")
    column_headers=[]
    for rr in regressor_list:
        for ss in stat_list:
            column_headers.append((rr+ss))
    #print(column_headers,"\n")
    temp_vec=np.tril(np.random.rand(len(CUES),len(CUES)), k=-1).flatten() # k=-1 should reduce to only entries below the diagonal
    lower_triangle_inds=np.where(temp_vec!=0)[0] # will use this to pull out the lower triangle from the coeff mats generated below
    #print("length of lower triangle inds vector",len(lower_triangle_inds))

    return regressor_list, lower_triangle_inds

def save_regressor_models_as_pngs(output_dir, CUES, model_dict):
    #CUES = ['r1_dcr','r1_dcb','r1_dpr','r1_dpb','r1_fpr','r1_fcr','r1_fcb','r1_fpb',  'r2_dcr','r2_dcb','r2_dpr','r2_dpb','r2_fpr','r2_fcr','r2_fcb','r2_fpb']
    #model dict will be data_Ctxt, dcfs_relF, dcfs_tskR, dsfc_relF, dsfc_tskR, data_Resp 
    # -- set up plot
    x_axis_labels = CUES #['dcr','dcb','dpr','dpb','fpr','fcr','fcb','fpb'] #[1,2,3,4,5,6,7,8]
    y_axis_labels = CUES #['dcr','dcb','dpr','dpb','fpr','fcr','fcb','fpb'] #[1,2,3,4,5,6,7,8]
    title_list = list(model_dict.keys())
    color_list = ["Blues","Purples","PuRd","Purples","PuRd","Greens"]
    # -- loop to generate model
    for ind, cur_key in enumerate(title_list):
        sns.heatmap( (model_dict[cur_key]).T, linewidth=0.5, xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap=color_list[ind], vmin=0, vmax=1)
        plt.title(title_list[ind], fontsize=20)
        plt.plot()
        plt.savefig(os.path.join(output_dir,"RSA","thalamus","figures",("model-"+cur_key+".png"))) # + "/%s_state_model_fit.png" %sub)
        plt.show()
        plt.close()

# Note, RSA models have 1=more similar and 0=less similar
def extract_char(list_of_strings,index_to_pull_from):
    element_list = [e[index_to_pull_from+3] for e in list_of_strings]
    return element_list

def gen_version_models(version_completed,element_order):
    # -- Get relevant feature model
    RelevantFeature_model = np.zeros((len(element_order),len(element_order)))
    rel_feat_vec=[]
    for indx, cur_cue in enumerate(element_order):
        # check if current cue was d | f ...
        if cur_cue[0+3]==version_completed[0]:
            # check what the relevant feature was
            if version_completed[1]=="c":
                rel_feat_vec.append("C") # this cue focused on color
            else:
                rel_feat_vec.append("S") # this cue focused on shape
        elif cur_cue[0+3]==version_completed[2]:
            # check what the relevant feature was
            if version_completed[3]=="c":
                rel_feat_vec.append("C") # this cue focused on color
            else:
                rel_feat_vec.append("S") # this cue focused on shape
    for ccol, col_cue in enumerate(element_order):
        for crow, row_cue in enumerate(element_order):
            # get the currently relevant feature
            rel_feat = rel_feat_vec[ccol]
            if rel_feat == "C":
                # if the color is the same
                if col_cue[2+3]==row_cue[2+3]:
                    # if the context is the same
                    if col_cue[0+3]==row_cue[0+3]:
                        RelevantFeature_model[crow][ccol] = 1  
                    else:
                        # if the context is different AND both features match  
                        if col_cue[1+3]==row_cue[1+3]:
                            RelevantFeature_model[crow][ccol] = 1  
            elif rel_feat == "S":
                # if the shape is the same
                if col_cue[1+3]==row_cue[1+3]:
                    # if the context is the same
                    if col_cue[0+3]==row_cue[0+3]:
                        RelevantFeature_model[crow][ccol] = 1
                    else:
                        # if the context is different AND both features match 
                        if col_cue[2+3]==row_cue[2+3]:
                            RelevantFeature_model[crow][ccol] = 1
    #print("\n",version_completed,"Relevant Feature Model\n",RelevantFeature_model)

    # -- Get task performed model
    TaskPerformed_model = np.zeros((len(element_order),len(element_order)))
    if version_completed=="dcfs":
        task_performed_order = []
        for elmt in element_order:
            task_perf='S'
            # for face_elmt in ['dpr','dcr','fpr','fpb']:
            for face_elmt in ['r1_dpr', 'r1_dcr', 'r1_fpr', 'r1_fpb', 'r2_dpr', 'r2_dcr', 'r2_fpr', 'r2_fpb']:
                if (elmt == face_elmt):
                    task_perf='F'
                    break # break out so F is the value
            task_performed_order.append(task_perf)
    elif version_completed=="dsfc":
        task_performed_order = []
        for elmt in element_order:
            task_perf='S'
            # for face_elmt in ['dpr', 'dpb', 'fpr', 'fcr']:
            for face_elmt in ['r1_dpr', 'r1_dpb', 'r1_fpr', 'r1_fcr', 'r2_dpr', 'r2_dpb', 'r2_fpr', 'r2_fcr']:
                if (elmt == face_elmt):
                    task_perf='F'
                    break # break out so F is the value
            task_performed_order.append(task_perf)
    for ccol, col_task in enumerate(task_performed_order):
        for crow, row_task in enumerate(task_performed_order):
            if col_task == row_task:
                TaskPerformed_model[crow][ccol] = 1          
            elif col_task == row_task:
                TaskPerformed_model[crow][ccol] = 1
    #print("\n",version_completed,"Task Performed Model\n",TaskPerformed_model)

    return RelevantFeature_model, TaskPerformed_model

def gen_RSA_models_new(element_order):
    # This function takes a list that tells us the order of the elements in the RSA coeff matrix
    # Note, this is pretty specific to ThalHi and it's cue list...
    # For ThalHi, the cues have 3 pieces of info
    #   1. d | f ... donut  | filled
    #   2. c | p ... circle | polygon
    #   3. r | b ...  red   |  blue
    #print(extract_char(element_order,0),"\n",extract_char(element_order,1),"\n",extract_char(element_order,2)) 

    # Set up basic models that don't depend on the version (Context & Identity)
    data_Ctxt = np.zeros((len(element_order),len(element_order)))
    for crow, celmnt1 in enumerate(extract_char(element_order,0)):
        for ccol, celmnt2 in enumerate(extract_char(element_order,0)):
            if extract_char(element_order,0)[crow]==extract_char(element_order,0)[ccol]:
                data_Ctxt[crow][ccol] = 1
    #print("\nContext Model \n",data_Ctxt)
    data_color = np.zeros((len(element_order),len(element_order)))
    for crow, celmnt1 in enumerate(extract_char(element_order,2)):
        for ccol, celmnt2 in enumerate(extract_char(element_order,2)):
            if extract_char(element_order,2)[crow]==extract_char(element_order,2)[ccol]:
                data_color[crow][ccol] = 1
    #print("\nColor Model \n",data_color)
    data_shape = np.zeros((len(element_order),len(element_order)))
    for crow, celmnt1 in enumerate(extract_char(element_order,1)):
        for ccol, celmnt2 in enumerate(extract_char(element_order,1)):
            if extract_char(element_order,1)[crow]==extract_char(element_order,1)[ccol]:
                data_shape[crow][ccol] = 1
    #print("\nShape Model \n",data_shape)

    data_Iden = np.identity(len(element_order))
    #print("\nIdentity Model \n",data_Iden)
    
    data_Resp = np.zeros((len(element_order),len(element_order)))
    re_list = [e[1] for e in element_order]
    for crow, celmnt1 in enumerate(re_list):
        for ccol, celmnt2 in enumerate(re_list):
            if re_list[crow]==re_list[ccol]:
                data_Resp[crow][ccol] = 1
    #print("\Resp Model \n",data_Resp)

    # Set up models that depend on version (Relevant_Feature & Task_Performed)
    # dcfs_relF, dcfs_tskR = gen_version_models("dcfs",element_order)
    # dsfc_relF, dsfc_tskR = gen_version_models("dsfc",element_order)
    dcfs_relF, dcfs_tskR = gen_version_models("dcfs",element_order)
    dsfc_relF, dsfc_tskR = gen_version_models("dsfc",element_order)

    # flip 0s and 1s so that it can work with the distance measure RSA
    data_Ctxt = (data_Ctxt*-1)+1
    data_color = (data_color*-1)+1
    data_shape = (data_shape*-1)+1
    data_Iden = (data_Iden*-1)+1
    data_Resp = (data_Resp*-1)+1
    dcfs_relF = (dcfs_relF*-1)+1
    dcfs_tskR = (dcfs_tskR*-1)+1
    dsfc_relF = (dsfc_relF*-1)+1
    dsfc_tskR = (dsfc_tskR*-1)+1

    return data_Ctxt, data_Iden, dcfs_relF, dcfs_tskR, dsfc_relF, dsfc_tskR, data_Resp, data_color, data_shape

def get_start_and_end(ind, len_of_data_to_add):
    start_ind = ind*len(len_of_data_to_add)
    stop_ind = start_ind+len(len_of_data_to_add)
    return start_ind, stop_ind

def create_regressor_dataframe(model_type, regressors_list, vec_dict, sub_list, lower_triangle_inds, roi_name):
    """
    model_type can be: LinearRegressionAllSubjs, LinearRegressionEachSubj, MixedEffects
    regressors_list could look like: ["Intercept","State","Task","Perceptual"]
    vec_dict could be {'y_vec':y_vec, 'state_vec':state_vec, 'task_vec':task_vec, 'percept_vec':percept_vec}
    """
    regressor_list = regressors_list.copy()
    #print("regressor_list: ", regressor_list)
    
    df = pd.DataFrame(vec_dict)
    regressor_list=["Context", "Color", "Shape", "TaskPerformed", "Resp"]
    
    # -- get correlations between regressors
    # ... maybe figure out how to add interactions some day...
    corr_results = {}
    regressor_corrs = np.array(df[regressor_list].corr(method='pearson')) # make correlation matrix
    for rr in range(regressor_corrs.shape[0]):
        for cc in range(regressor_corrs.shape[1]):
            if rr > cc:
                key_to_add = regressor_list[rr] + "_" + regressor_list[cc] # what two regressors are being compared
                corr_results[key_to_add] = regressor_corrs[rr,cc] # grab out the proper correlation from the table
    #print(corr_results)
    
    # ---- set up and calc model
    # -- add subject info to model dict
    sub_list_flattened = np.zeros( len(sub_list)*len(lower_triangle_inds) ) # add subject info to data frame
    for sub_ind, sub in enumerate(sub_list):
        if model_type == 'MixedEffects':
            sub_list_flattened[(sub_ind*len(lower_triangle_inds)):((sub_ind*len(lower_triangle_inds))+len(lower_triangle_inds))] = str(sub)
        else:
            sub_list_flattened[(sub_ind*len(lower_triangle_inds)):((sub_ind*len(lower_triangle_inds))+len(lower_triangle_inds))] = float(sub)
    df['sub'] = sub_list_flattened
    # -- center regressors
    df['Context'] = df['Context'] - df['Context'].mean()
    #df['Relevant_Feature'] = df['Relevant_Feature'] - df['Relevant_Feature'].mean()
    df['Color'] = df['Color'] - df['Color'].mean()
    df['Shape'] = df['Shape'] - df['Shape'].mean()
    df['TaskPerformed'] = df['TaskPerformed'] - df['TaskPerformed'].mean()
    df['Resp'] = df['Resp'] - df['Resp'].mean()
    
    # -- model setup
    if model_type == 'MixedEffects':
        df['sub'] = df['sub'].astype('category')
        #model = smf.mixedlm("Y ~ Context + Color + Shape + TaskPerformed + Resp", data=df[regressor_list], groups=df["sub"]).fit()
        rand_effects = " + (1|sub) + (0+Context|sub) + (0+Color|sub) + (0+Shape|sub) + (0+TaskPerformed|sub) + (0+Resp|sub)"
        main_effects = "Context + Color + Shape + TaskPerformed + Resp"
        interactions_2way = " + Context:Color + Context:Shape + Context:TaskPerformed + Color:Shape + Color:TaskPerformed + Shape:TaskPerformed"
        interactions_3way = " + Context:Color:TaskPerformed + Context:Shape:TaskPerformed"
        model_formula = "Y ~ " + main_effects + interactions_2way + interactions_3way + rand_effects
        
        # -- add interactions to regressor_list
        list_2way = interactions_2way.split(" + ") # [0] will be an empty str ... ""
        [regressor_list.append(tmp_entry.replace(":","_x_")) for tmp_entry in list_2way[1:]]
        list_3way = interactions_3way.split(" + ") # [0] will be an empty str ... ""
        [regressor_list.append(tmp_entry.replace(":","_x_")) for tmp_entry in list_3way[1:]]
        
    elif model_type == 'LinearRegressionPerSubject':
        main_effects = "Context + Color + Shape + TaskPerformed + Resp"
        interactions_2way = " + Context:Color + Context:Shape + Context:TaskPerformed + Color:Shape + Color:TaskPerformed + Shape:TaskPerformed"
        interactions_3way = " + Context:Color:TaskPerformed + Context:Shape:TaskPerformed"
        model_formula = "Y ~ " + main_effects + interactions_2way + interactions_3way #+ interactions_4way + interactions_5way 
        # add interactions to regressor list
        regressor_list.append("CxCO")
        regressor_list.append("CxSH")
        regressor_list.append("CxTP")
        #regressor_list.append("CxR")
        regressor_list.append("COxSH")
        regressor_list.append("COxTP")
        #regressor_list.append("COxR")
        regressor_list.append("SHxTP")
        regressor_list.append("CxCOxTP")
        regressor_list.append("CxSHxTP")
        regressor_list.append("Intercept")
        regressor_txt_dict = {'Context':'Context', 'Color':'Color', 'Shape':'Shape', 'TaskPerformed':'TaskPerformed', 'Resp':'Resp',
                              'CxCO':'Context:Color', 'CxSH':'Context:Shape', 'CxTP':'Context:TaskPerformed', 'COxSH':'Color:Shape', 'COxTP':'Color:TaskPerformed', 'SHxTP':'Shape:TaskPerformed', 
                              'CxCOxTP':'Context:Color:TaskPerformed', 'CxSHxTP':'Context:Shape:TaskPerformed', 'Intercept':'Intercept'}
    
    # ---- save out data frame for current ROI (usefull for double checking data input to model if needed)
    #ROI_dataset_filename = "ROI_" + str(roi) + "_Dataset_" + stats_method + "_argon.csv"
    #df.to_csv(os.path.join(data_dir,"RSA","ROI_Level_Datasets",ROI_dataset_filename)) 
    
    # -- now run the stats and save the results
    if model_type == 'MixedEffects':
        # -- run mixed effects model
        model = Lmer(model_formula, data=df)
        res = model.fit()
        tmp_df = pd.DataFrame(res)
        tmp_df.to_csv(os.path.join(out_dir,("roi-"+roi_name+"__MixedEffectsModelOutput.csv")))
        
        b_list = list(res['Estimate'])
        t_list = list(res['T-stat']) # pull out t-stat for TASK (0==intercept)
        p_list = list(res['P-val'])
        cur_dict = {'Intercept_beta': b_list[0], 'Intercept_tval': t_list[0], 'Intercept_pval': p_list[0]}
        for r_idx, cur_r in enumerate(regressor_list):
            cur_dict[(cur_r+"_beta")] = b_list[(r_idx+1)]
            cur_dict[(cur_r+"_tval")] = t_list[(r_idx+1)]
            cur_dict[(cur_r+"_pval")] = p_list[(r_idx+1)]
        
    elif model_type == 'LinearRegressionPerSubject':
        # -- set up output dictionary
        cur_dict = {}
        for cur_r in regressor_list:
            for cur_s in ["_beta", "_tval", "_pval"]:
                cur_dict[(cur_r+cur_s)] = []
        # -- loop through subjects and run separately on each subject
        for sub_ind, subj in enumerate(sub_list):
            cur_sub_df = df[df['sub'] == float(subj)] # remember this is the data df we defined with the vectors (not the trial df)
            #print(cur_sub_df)
            # check for all nan values (happens rarely)
            nrow = cur_sub_df.shape[0] # 1st dimension is rows
            nrow_wnan = cur_sub_df.isnull().T.any().T.sum()
            # set up and calc model for each subj
            if nrow != nrow_wnan:
                model = smf.ols(formula = model_formula, data=cur_sub_df[["Y", "Intercept", "Context", "Color", "Shape", "TaskPerformed", "Resp"]])
                res = model.fit()
            else:
                print("current subject (", subj, ") only has nan values... cannot run regression... saving out non-sig values for all regressors")
                #print(cur_sub_df)
            # pull out betas, t-vals, and p-vals for each subj
            # Note, order is: [0]=Intercept [1]=Context, [2]=Relevant_Feature, [3]=Task_Performed, [4]=CxRFxTP
            for cur_regressor in regressor_list:
                regressor_txt=regressor_txt_dict[cur_regressor] # should work with interactions based on how they were defined and added to this list
                # workaround for if no rows have values
                if nrow != nrow_wnan:
                    c_beta = res.params[regressor_txt]
                    c_tval = res.tvalues[regressor_txt]
                    c_pval = res.pvalues[regressor_txt]
                else:
                    c_beta = 0.0
                    c_tval = 0.0
                    c_pval = 1.0
                # add values
                cur_dict[(cur_regressor+"_beta")].append(c_beta) # beta
                cur_dict[(cur_regressor+"_tval")].append(c_tval) # t-val
                cur_dict[(cur_regressor+"_pval")].append(c_pval) # p-val
    
    #print("dictionary: ", cur_dict)
    return cur_dict, df, corr_results

def RSA_cues_for_parallel(inputs):
    err_type = 'none'
    any_error = False
    
    X_shm = shared_memory.SharedMemory(name=inputs[0]) # this is to access the shared memory space that is storing X
    searchlight_data =  np.ndarray(inputs[1], buffer=X_shm.buf)[:, inputs[2]] #input 1 is the X_dim, and use the sphere_voxel_inds to index it
    sphere_voxel_inds = inputs[2] # this is A now in list format comping from the loop
    cue_list = inputs[3]
    if len(cue_list) != 16:
        print("ERROR!!! cue_list not of size 16!!!")
    sub_list = inputs[4] 
    if len(sub_list) != 59:
        print("ERROR!!! sub_list not of size 59!!!")
    # now access residuals in shared memory
    resids_data_shm= shared_memory.SharedMemory(name=inputs[5])
    resids_data_list = np.ndarray(inputs[6], buffer=resids_data_shm.buf)[:,:,sphere_voxel_inds] #create residual data array of sub by TR by voxels
    #print("resids_data_list.shape = ", resids_data_list.shape)
    sphere_idx = inputs[7]
    #print("working on sphere ", sphere_idx)
    #do_mult_comp_corr = inputs[8] # check if we want to do multiple comparisons corrections
    do_mult_comp_corr=False
    project_dir = inputs[9]
    #print("project_dir :  ", project_dir)
    
    # ---- initialize lists, arrays, and variables
    regressor_list, lower_triangle_inds = set_up_model_vars(cue_list)
    y_vec=np.zeros((len(sub_list)*len(lower_triangle_inds)))
    context_vec=np.zeros((len(sub_list)*len(lower_triangle_inds)))
    color_vec=np.zeros((len(sub_list)*len(lower_triangle_inds)))
    shape_vec=np.zeros((len(sub_list)*len(lower_triangle_inds)))
    taskPerform_vec=np.zeros((len(sub_list)*len(lower_triangle_inds)))
    resp_vec=np.zeros((len(sub_list)*len(lower_triangle_inds)))

    # ---- initialize model info
    data_Ctxt, data_Iden, dcfs_relF, dcfs_tskR, dsfc_relF, dsfc_tskR, data_Resp, data_color, data_shape = gen_RSA_models_new(cue_list) 
    version_info = pd.read_csv(os.path.join(project_dir, "Version_Info.csv"))

    # ---- loop through subjects, get rsa on sphere, and run model on all subjects
    for ind, subject in enumerate(sub_list):
        #print("currently on subject ", subject)
        # # # # Loop where we reduce to just current subejct to get RSA for each subj
        #print("\nsearchlight data is size",searchlight_data.shape,"\nsphere voxel arg is size",len(sphere_voxel_inds)) # searchlight data = [subject_x_cues , voxels]
        start_ind, stop_ind = get_start_and_end(ind, cue_list)
        #print("start:",start_ind,"\tstop:",stop_ind)
        cur_sub_data = searchlight_data[start_ind:stop_ind,:] # dims = [16  num_voxels]
        # if sphere_idx<10:
        #     print("number of voxels in searchlight sphere:",len(sphere_voxel_inds),"\tsize current subj data:",cur_sub_data.shape)
        
        # -- remove any voxels with zero
        usable_sphere_inds = np.zeros((len(sphere_voxel_inds)))
        usable_vox_list = []
        for ind_v, vox in enumerate(sphere_voxel_inds):
            for ind_c, cue_pt in enumerate(cur_sub_data[:,ind_v]):
                if cue_pt != 0:
                    usable_sphere_inds[ind_v]=1
            if usable_sphere_inds[ind_v]==1:
                usable_vox_list.append(ind_v)
        usable_vox_arr = np.asarray(usable_vox_list).flatten()
        # if sphere_idx<10:
        #     print("number of usable voxels in searchlight sphere:",len(sphere_voxel_inds))
        
        #print("num usable sphere inds:", len(usable_vox_arr), "\tproportion usable:", (sum(usable_sphere_inds)/len(sphere_voxel_inds)), "\n\tresids_size:", resids_data_list.shape)
        #print("searchlight data for current subject is size",cur_sub_data.shape) # cur_sub_data = [cues , voxels]
        #resids_data = resids_data_list[ind,:,usable_vox_arr] # should have already reduced dim3 to just the current sphere BUT now we reduce to usable sphere inds
        #print("resids data shape", resids_data.shape)
        #                     remove_censored_data_noise_version(resids_data)
        # Note, resids_data flips dimension order when we index (IDK WHY) and so we have to transpose it in the line below
        #reduced_resids_data = remove_censored_data_noise_version(resids_data.T)  #already indexed
        #print("resids data shape before transpose:", resids_data.shape, "reduced resids data shape after transpose and removing censored portions:", reduced_resids_data.shape)
        #print("resids data shape:", resids_data.shape, "\treduced resids data shape:", reduced_resids_data.shape)
        #noise_pres_res = rsatoolbox.data.noise.prec_from_residuals(reduced_resids_data, method='diag')
        
        # -- get noise measure for mahalanobis distance
        try:
            resids_data = resids_data_list[ind,:,usable_vox_arr] # should have already reduced dim3 to just the current sphere BUT now we reduce to usable sphere inds
            #print("resids data shape", resids_data.shape) #  [voxels  TRs] according to testing
            resids_data = resids_data.T # Transposing here since I known I want [TRs  voxels]
            #reduced_resids_data = remove_censored_data_noise_version(resids_data.T)  #already indexed
            if resids_data.shape[0]!=1728:
                print("dimension order incorrect\ttransposing now") # want data to be  [TRs  voxels]
                resids_data = resids_data.T
            voxel_vec = np.mean(resids_data,axis=0) # will identify voxels where all TRs were 0 by collapsing across TRs (dim0)
            TRs_vec = np.mean(resids_data,axis=1) # will identify TRs where all voxels were 0
            reduced_resids_data = resids_data[TRs_vec!=0,:] # now grab all non-censored TRs and usable voxels
            # if len(usable_vox_arr) == reduced_resids_data[:,voxel_vec!=0].shape[1]:
            #     reduced_resids_data = reduced_resids_data[:,voxel_vec!=0]
            # else:
            #     # print("issue with usable voxels... mismatch between nii data and resids data... using usable_vox_arr")
            #     # if sphere_idx<10:
            #     #     print("number of usable voxels in resids data:",reduced_resids_data[:,voxel_vec!=0].shape[1])
            #     reduced_resids_data = reduced_resids_data[:,usable_vox_arr]
            reduced_resids_data = reduced_resids_data[:,voxel_vec!=0]
            #print("reduced_resids_data \n", reduced_resids_data)
            noise_pres_res = rsatoolbox.data.noise.prec_from_residuals(reduced_resids_data, method='shrinkage_diag') # method='diag')
            #print("noise_pres_res: ",noise_pres_res)
        except Exception as e:
            print(repr(e))
            print("encountered an exception with creating the noise object on sphere", sphere_idx, "\treduced_resids_data shape: ", reduced_resids_data.shape, "\nreduced_resids_data \n", reduced_resids_data)
            noise_pres_res = np.identity(len(usable_vox_list))
            any_error = True
            err_type = 'noise_creation'
        
        # -- calculate RSA using mahalanobis distance
        #     * RSA will be calculated for each subject... the coeffs will be pulled out and added to a vector
        #     * the stats model will then be applied to the vector of all subject coeffs for this searchlight
        try:
            #                     make_rsatoolbox_format(CONDxVOXELS,                    sub,     roi,        stats_method,  cue_list, num_runs)
            #data_rsatool_format = make_rsatoolbox_format(cur_sub_data[:,usable_vox_arr], subject, int(sphere_idx), "Mahalanobis", cue_list, 1) # entering 1 as roi (b/c we don't need it)
            data_rsatool_format = make_rsatoolbox_format(cur_sub_data[:,voxel_vec!=0], subject, int(sphere_idx), "Mahalanobis", cue_list, 1) # entering 1 as roi (b/c we don't need it)
            #print(data_rsatool_format) 
            try: 
                rdm = rsatoolbox.rdm.calc_rdm(data_rsatool_format, method='mahalanobis', descriptor='measure', noise=noise_pres_res)
                Coeff_mat = rdm.get_matrices()[0] # the distance matrix
                #print(Coeff_mat)
            except Exception as e:
                print(repr(e))
                print("encountered an exception with creating the rdm on sphere", sphere_idx, "and subject ", subject)
                Coeff_mat = np.empty((cur_sub_data.shape[0],cur_sub_data.shape[0]))
                Coeff_mat[:] = np.nan
                any_error = True
                err_type = 'rdm_calc'
                #rdm_inv_err_dict = {'central_voxel': sphere_idx, 'num_nonzero_voxels': cur_sub_data.shape[1], 'timestamp': datetime.datetime.now()}
                #print(rdm_inv_err_dict)
                #pickle.dump(rdm_inv_err_dict, open(os.path.join("/Shared","lss_kahwang_hpc","ThalHi_MRI_2020","RSA","searchlight","ROIs_w_Inversion_Error", (str(sphere_idx)+"_rdm_inversion_error.p")), "wb"), protocol=4)
                #ROIs_with_inversion_error.append(roi)
        except Exception as e:
            print(repr(e))
            print("encountered an exception when putting data in rsatoolbox format on sphere", sphere_idx)
            Coeff_mat = np.empty((cur_sub_data.shape[0],cur_sub_data.shape[0]))
            Coeff_mat[:] = np.nan
            any_error = True
            err_type = 'usable_vox_arr'
        #print("Coeff mat is size:",Coeff_mat.shape,"\nCoeff mat looks like:",Coeff_mat)
        # rdm = rsatoolbox.rdm.calc_rdm(data_rsatool_format, method='mahalanobis', descriptor='measure', noise=noise_pres_res)
        # Coeff_mat = rdm.get_matrices()[0] # the distance matrix
        
        # ---- Pull out lower triangle from coeff mat for this subject
        coeff_vec = np.tril(Coeff_mat, k=-1).flatten()
        start_pt, end_pt = get_start_and_end(ind, lower_triangle_inds)
        #print("start point =",start_pt,"\tend point =",end_pt)
        y_vec[start_pt:end_pt] = coeff_vec[lower_triangle_inds] # add to y vector for model
        #    set up other model vectors based on what version the current sub did
        context_vec[start_pt:end_pt] = data_Ctxt.flatten()[lower_triangle_inds]
        color_vec[start_pt:end_pt] = data_color.flatten()[lower_triangle_inds]
        shape_vec[start_pt:end_pt] = data_shape.flatten()[lower_triangle_inds]
        resp_vec[start_pt:end_pt] = data_Resp.flatten()[lower_triangle_inds]
        if version_info["version"][ind]=="DCFS":
            # not swapped version
            taskPerform_vec[start_pt:end_pt]=dcfs_tskR.flatten()[lower_triangle_inds]
        else:
            taskPerform_vec[start_pt:end_pt]=dsfc_tskR.flatten()[lower_triangle_inds]

    ## need to close shared memory within each job
    X_shm.close()
    resids_data_shm.close()

    model_data_dict = {'Y': y_vec, 'Intercept': np.ones((len(y_vec))), 'Context': context_vec, 'Color': color_vec, 'Shape': shape_vec, 'TaskPerformed': taskPerform_vec, 'Resp': resp_vec}
    #              cur_dict, df, corr_results = create_regressor_dataframe(model_type, regressors_list, vec_dict, sub_list, lower_triangle_inds, roi_name)
    searchlight_stat_output, df, corr_results = create_regressor_dataframe('LinearRegressionPerSubject', regressor_list, model_data_dict, sub_list, lower_triangle_inds, str(sphere_idx))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if do_mult_comp_corr:
        # ---- correct for multiple comparisons
        results_permuted = {'context_tval': [], 'color_tval': [], 'shape_tval': [], 'taskper_tval': [], 'resp_tval': []}
        y_vec_p = y_vec.copy() # make a copy so we still have the original order
        context_vec_p = context_vec.copy() # make a copy so we still have the original order
        color_vec_p = color_vec.copy() # make a copy so we still have the original order
        shape_vec_p = shape_vec.copy() # make a copy so we still have the original order
        taskPerform_vec_p = taskPerform_vec.copy() # make a copy so we still have the original order
        resp_vec_p = resp_vec.copy() # make a copy so we still have the original order
        for cur_it in range(10000):
            #y_vec_p = np.random.permutation(y_vec_p)
            np.random.shuffle(context_vec_p)
            np.random.shuffle(color_vec_p)
            np.random.shuffle(shape_vec_p)
            np.random.shuffle(taskPerform_vec_p)
            np.random.shuffle(resp_vec_p)
            roi_results_dict_permuted, df_permuted = create_regressor_dataframe_new('LinearRegressionPerSubject', regressor_list, 
                                                                                {'y_vec':y_vec_p, 'context_vec':context_vec_p, 'color_vec':color_vec_p, 'shape_vec':shape_vec_p, 'taskPerform_vec':taskPerform_vec_p, 'resp_vec':resp_vec_p},
                                                                                sub_list, lower_triangle_inds, str(sphere_idx))
            # for current roi, run 10000 times and save results from each time
            for cur_key in results_permuted.keys():
                results_permuted[cur_key].append(roi_results_dict_permuted[cur_key])
        # ---- LOOP through PERMUTED roi results dict and add new values to overall results dict
        for cur_key in results_permuted.keys():
            cur_regressor=cur_key.split("_")[:-1] # eg: context
            out_key_name = cur_regressor[0]+"_thresh2tail" # eg: context_thresh2tail
            searchlight_stat_output[out_key_name] = np.percentile(np.asarray(results_permuted[cur_key]), 97.5)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    searchlight_stat_output['sphere_idx'] = sphere_idx
    # output variable is list format ... add inversion error info if applicable
    if any_error:
        searchlight_stat_output['err_type'] = err_type
    
    return searchlight_stat_output ## also need to save sphere idx




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - -     set up for RSA (cortical | subcortical)     - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
code will follow the general outline
I. load a cortical mask and loop through subjects to remove voxels with no data (per subject calculation, but applied to all subjects)
II. apply cortical mask to 400 ROIs mask, residuals, and 3dDeconvolve output
III. use parallel processing to loop through rois
"""
num_trials = 408
num_runs = 8
# -- variables that do not depend on cortical | subcortical call
cue_list = ["r1_dcb", "r1_fcb", "r1_dpb", "r1_fpb", "r1_dcr", "r1_fcr", "r1_dpr", "r1_fpr",
            "r2_dcb", "r2_fcb", "r2_dpb", "r2_fpb", "r2_dcr", "r2_fcr", "r2_dpr", "r2_fpr"]
    
#THAL_HI_DIR = "/Shared/lss_kahwang_hpc/data/ThalHi/"
if os.path.exists("/Shared/lss_kahwang_hpc/"):
    mask_dir = "/Shared/lss_kahwang_hpc/ROIs/"
elif os.path.exists("/mnt/nfs/lss/lss_kahwang_hpc/"):
    mask_dir = "/mnt/nfs/lss/lss_kahwang_hpc/ROIs/"
else:
    print("Error!!!! need to run from thalamege or argon")
#out_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/"
#mnt_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/"
stats_method = "Mahalanobis" # originally tested with multiple options, but ended up deciding to use Mahalanobis for everything
skip_setup = True # False # 

 # -- create subject list based on usable files for ThalHi
unusable_subs = ['10006', '10007', '10009', '10011', '10015', '10016', '10021', '10026', '10029', '10030', '10055', '10061', '10062', '10065', '10172', 'JH']
subjects = sorted(os.listdir(os.path.join(project_dir,"3dDeconvolve_fdpt4"), )) 
subject_list = []
for ii, subj in enumerate(subjects):
    if os.path.isdir(os.path.join(project_dir,"3dDeconvolve_fdpt4",subj)):
        #print(subj.split('-'))
        cur_subj = subj.split('-')[1]
        if (stats_method=="Crossnobis") and (int(cur_subj)==10175):
            continue
        if cur_subj not in unusable_subs:
            subject_list.append(cur_subj)
print("subject list: ", subject_list)

# -- set up Grey Matter mask
if not(os.path.exists(os.path.join(mask_dir,("GreyMatter_Mask_task-"+project_name+".nii.gz")))):
    print("Grey matter mask doesn't exist so I need to create it now")
    cortical_mask = nib.load(os.path.join(mask_dir,"CorticalBinary_rs.nii.gz"))
    #subcortical_mask = nib.load(os.path.join(mask_dir,"subcortex_2.5mm_binary.nii.gz"))
    BG_mask = nib.load("BG_2.5mm.nii.gz")
    thalamus_mask = nib.load("Morel_2.5_mask.nii.gz")
    gm_mask = nilearn.masking.intersect_masks([cortical_mask, BG_mask, thalamus_mask], threshold=0)
    nib.save(gm_mask, os.path.join(mask_dir, ("GreyMatter_Mask_task-"+project_name+".nii.gz")))


if use_cortical:      
    # code
    if use_searchlight:
        gm_mask = nib.load(os.path.join(mask_dir,("GreyMatter_Mask_task-"+project_name+".nii.gz")))
        mask_voxel_num = int(gm_mask.get_fdata().sum()) # 209453 ... 150260
        cortical_masker = NiftiMasker(gm_mask) # used to be cortical_mask
        # #### if searchlight argument entered, run code below
        # cortical_mask = nib.load(os.path.join(mask_dir,"CorticalBinary_rs.nii.gz")) # get mask that will be used as the current searchlight mask
        # # # Load template for getting size and converting to niftii format
        img = nib.load( project_dir + "fmriprep/sub-10001/func/sub-10001_task-ThalHi_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" )
        dims = img.get_fdata().shape
        # # # Load 8 cue by 2 resp (16 total) files and create voxel by cue&resp matrix (stacked by subject)
        # # # Load errors from regressor for each subject too while we are at it
        os.chdir(os.path.join(project_dir, "RSA", "cortical", "searchlight")) # change directory to the searchlight directory
        print("current working directory:",os.getcwd())
        if os.path.exists(os.path.join(project_dir, "RSA", "cortical", "searchlight", "intermediate_files", "voxels3d_by_cues-respXsubjs.npy")):
            #### if the LSS file has already been created, just load that
            if not(args.gen_searchlight_plots):
                if not(os.path.exists(os.path.join(project_dir, "RSA", "cortical", "searchlight", "intermediate_files", "cues_by_subjs.nii.gz"))):
                    voxels3d_by_cuesXsubjs = np.load(os.path.join(project_dir, "RSA", "cortical", "searchlight", "intermediate_files", "voxels3d_by_cues-respXsubjs.npy"))
                    print("shape of voxels3d_by_cues-respXsubjs", voxels3d_by_cuesXsubjs.shape)
                resids_data_array = np.load(os.path.join(project_dir, "RSA", "cortical", "searchlight", "intermediate_files", "searchlight16_resids_array.npy"))
                print("loaded voxels3d_by_cues-respXsubjs (numpy) and resids_data_list files")
                print("shape of resids_data_array", resids_data_array.shape)
            sub_list = subject_list
        else:
            pickle.dump(subject_list, open(os.path.join(project_dir,"RSA", "cortical", "searchlight", "intermediate_files", "searchlight_sub_list.p"), "wb"), protocol=4)
            voxels3d_by_cuesXsubjs=np.zeros( (dims[0],dims[1],dims[2],(len(cue_list)*len(subject_list))) )
            #resids_data_list = [] # make empty list... will be size==number of participants
            resids_data_array = np.zeros( (len(subject_list), 1728, mask_voxel_num) ) # resid_dims[0], resid_dims[1]) ) # sub by TR by voxels
            #voxels_to_exclude_list = []
            sub_list = []
            for ind_s, subname in enumerate(subject_list):
                # change directory to the deconvolve directory
                sub_list.append(subname)
                os.chdir(os.path.join(project_dir, "3dDeconvolve_fdpt4", ("sub-"+subname)) )
                print("\nLoading cue files for subject ",subname)
                # loop through cues and load and save them in larger 4D matrix 
                for ind_c, cue in enumerate(cue_list):
                    print("Loading cue file ", cue)
                    iresp_file = os.path.join(project_dir,"3dDeconvolve_fdpt4",("sub-"+subname),("sub-"+subname+"_"+cue+"_FIR_MNI.nii.gz"))
                    #iresp_file = os.path.join(subject.deconvolve_dir,("sub-"+subname+"_"+cue+"_FIR_MNI.nii.gz"))
                    data_file = nib.load(iresp_file) # load data file
                    fdata=data_file.get_fdata()
                    print("cue file is shape: ",fdata.shape)
                    fdata_avg = np.mean(fdata, axis=3) # take average over masked data (assumes 4th dimension is tents)
                    print("avg cue file is shape: ",fdata_avg.shape)
                    #data=nilearn.image.new_img_like(data_file, fdata_avg)
                    index = ((ind_s*len(cue_list))+ind_c)
                    print(index)
                    voxels3d_by_cuesXsubjs[:,:,:,index] = fdata_avg
                # also load residuals file for this subject and mask so it's just the voxels
                print("\nloading residuals file for subject ",subname)
                resids_file = os.path.join(project_dir,"3dDeconvolve_fdpt4",("sub-"+subname),("sub-"+subname+"_FIRmodel_errts_8cues2resp.nii.gz"))
                resids_data_nii = nib.load(resids_file)
                #r_data = resids_data_nii.get_fdata()
                resids_data = cortical_masker.fit_transform(resids_data_nii)
                #resids_data = nilearn.masking.apply_mask(resids_data_nii,coritcal_masker) 
                print("resids size: ",resids_data.shape,"\n")
                #resids_data_list.append(resids_data)
                resids_data_array[ind_s,:resids_data.shape[0],:resids_data.shape[1]] = resids_data
                # get voxels to exclude
                #voxels_to_exclude = get_voxels_to_exclude_SL_version(r_data)
                #voxels_to_exclude_list.append(voxels_to_exclude)
            # save voxels3d_by_cuesXsubjs (numpy array) and resids_data_list (list type)
            np.save(os.path.join(project_dir,"RSA", "cortical", "searchlight", "intermediate_files", "voxels3d_by_cues-respXsubjs.npy"), voxels3d_by_cuesXsubjs)
            print(voxels3d_by_cuesXsubjs.shape)
            np.save(os.path.join(project_dir,"RSA", "cortical", "searchlight", "intermediate_files", "searchlight16_resids_array.npy"), resids_data_array)
            #pickle.dump(resids_data_list, open(os.path.join("/Shared","lss_kahwang_hpc","ThalHi_MRI_2020","RSA","searchlight", "searchlight_resids_list.p"), "wb"), protocol=4)
            #pickle.dump(voxels_to_exclude, open(os.path.join(project_dir, "RSA", "cortical", "searchlight", "intermediate_files", "searchlight16_voxels_to_exclude_list.p"), "wb"), protocol=4)
            #del voxels_to_exclude # save memory
            #del resids_data_list
        
    
    if use_schaefer_400rois:
        # -- set up roi mask list (and name list)
        start_roi_num = 1
        end_roi_num = 400
        roi_name_list = [str(f"{x:03d}") for x in np.arange(start_roi_num,(end_roi_num+1))] # 3 number code as a string from 001 - 400
        num_rois = len(roi_name_list)
        # -- define output directory
        out_dir = os.path.join(project_dir,"RSA","cortical","intermediate_files")
        # -- load cortical mask
        cortical_mask = nib.load(os.path.join(mask_dir,"CorticalBinary_rs.nii.gz")) # to make things faster, apply a cortcal mask to reduce roi mask to a vector
        cortical_data = cortical_mask.get_fdata()
        # ------ load 400 ROI mask file
        mask_file = os.path.join(mask_dir, "Schaefer400_2.5.nii.gz")
        mask = nib.load(mask_file)
        mask_data = mask.get_fdata()
        mask_dims = [mask.affine[0,0], mask.affine[1,1], mask.affine[2,2], mask.affine[3,3]]
        # ------ load subject errts and set up for rsa and regression
        mask_vec_list = [] # will be 59 subjects long
        resids_2D_list = [] # will be 59 subjects long
        nii_2D_list = []
        for ind_s, subject in enumerate(subject_list): 
            try:
                mask_vec = np.load(os.path.join(project_dir,"RSA","cortical","intermediate_files",("statmethod-"+stats_method+"_sub-"+subject+"_mask400rois.npy")))
                resids_2D = np.load(os.path.join(project_dir,"RSA","cortical","intermediate_files",("statmethod-"+stats_method+"_sub-"+subject+"_residuals.npy")))
                nii_cue_matrix = np.load(os.path.join(project_dir,"RSA","cortical","intermediate_files",("statmethod-"+stats_method+"_sub-"+subject+"_cueniis.npy")))
            except:
                print("\nloading errts for subject ", subject)
                r = nib.load(os.path.join(project_dir, "3dDeconvolve_fdpt4", ("sub-"+subject), ("sub-"+subject+"_FIRmodel_errts_8cues2resp.nii.gz")))
                # check for zeros
                r_data = r.get_fdata()
                voxels_to_exclude = np.zeros((r_data.shape[0], r_data.shape[1], r_data.shape[2])) # initialize 3D matrix of what voxels to exclude
                for x in range(r_data.shape[0]):
                    for y in range(r_data.shape[1]):
                        for z in range(r_data.shape[2]):
                            if r_data[x,y,z,:].sum()==0:
                                # voxel had 0 for all time points ... exclude from further analysis
                                voxels_to_exclude[x,y,z]=1
                # print("A total of",voxels_to_exclude.sum(),"voxels will be EXCLUDED due to 0s for all time points")
                # print(voxels_to_exclude.sum(),"voxels exluded out of",(r_data.shape[0]*r_data.shape[1]*r_data.shape[2]),"total voxels")
                # print((voxels_to_exclude.sum()/(r_data.shape[0]*r_data.shape[1]*r_data.shape[2])),"proportion of voxels excluded\n")
                # -- now modify cortical mask
                cortical_binary=np.where(np.logical_and(cortical_data>0,voxels_to_exclude==0),1,0) # make sure to also exclude voxels with all zeros
                print("number of usable voxels in current ROI mask:", cortical_binary.sum(), "\n")
                cortical_mask_binary=nilearn.image.new_img_like(cortical_mask, cortical_binary)
                
                mask_vec = nilearn.masking.apply_mask(mask, cortical_mask_binary) # will be 1D (voxels)
                print("mask_vec dimensions ", mask_vec.shape)
                #mask_vec_list.append(mask_vec)
                resids_2D = nilearn.masking.apply_mask(r, cortical_mask_binary) # will be [time x voxels]
                print("resids_2D dimensions ", resids_2D.shape)
                #resids_2D_list.append(resids_2D)
                
                if use_trial_level:
                    # ---- load cue files for current subject  ...  loop through cues and load and save them in larger matrix
                    print("\nLoading cue files for subject ", subject)
                    nii_trl_matrix = np.zeros((cortical_binary.sum(), len(cue_list)))
                    print("nii_cue_matrix shape ", nii_cue_matrix.shape)
                else:
                    # ---- load cue files for current subject  ...  loop through cues and load and save them in larger matrix
                    print("\nLoading cue files for subject ", subject)
                    nii_cue_matrix = np.zeros((cortical_binary.sum(), len(cue_list)))
                    print("nii_cue_matrix shape ", nii_cue_matrix.shape)
                    for ind_c, cue in enumerate(cue_list):
                        nii_data = nib.load(os.path.join(project_dir, "3dDeconvolve_fdpt4", ("sub-"+subject), ("sub-"+subject+"_"+cue+"_FIR_MNI.nii.gz")))
                        nii_data_dims = [nii_data.affine[0,0], nii_data.affine[1,1], nii_data.affine[2,2], nii_data.affine[3,3]]
                        if mask_dims != nii_data_dims:
                            print("Error!!! ROI mask and data file dimensions DO NOT MATCH!!!")
                            "c"+2
                        masked_nii_data = nilearn.masking.apply_mask(nii_data, cortical_mask_binary)
                        print("shape of masked_nii_data is ", masked_nii_data.shape)
                        masked_data_avg=np.mean(masked_nii_data, axis=0) # take average over masked data
                        nii_cue_matrix[:,ind_c] = masked_data_avg # add 1D masked data vector (avg) to my numpy matrix of size VOXELSxCUES
                        
                        del masked_nii_data
                        del masked_data_avg
                        del nii_data
                #nii_2D_list.append(nii_cue_matrix)
                
                # -- save
                np.save(os.path.join(project_dir,"RSA","cortical","intermediate_files",("statmethod-"+stats_method+"_sub-"+subject+"_mask400rois.npy")), mask_vec)
                np.save(os.path.join(project_dir,"RSA","cortical","intermediate_files",("statmethod-"+stats_method+"_sub-"+subject+"_residuals.npy")), resids_2D)
                np.save(os.path.join(project_dir,"RSA","cortical","intermediate_files",("statmethod-"+stats_method+"_sub-"+subject+"_cueniis.npy")), nii_cue_matrix)
            
            # -- append
            mask_vec_list.append(mask_vec)
            resids_2D_list.append(resids_2D)
            nii_2D_list.append(nii_cue_matrix)
            
            
        
        # # loop through and load individual matrices to make the larger list
        # for roi, cur_roi in enumerate(roi_name_list):
        #     VOXELSxCUESxSUBS = np.load(os.path.join(project_dir,"RSA","cortical","intermediate_files",("statmethod-"+stats_method+"_roi-"+cur_roi+"_VOXELSxCUESxSUBS.npy")))
        #     resids_data_array = np.load(os.path.join(project_dir,"RSA","cortical","intermediate_files",("statmethod-"+stats_method+"_roi-"+cur_roi+"_residuals.npy")))
        #     list_of_VoxelsXcuesXsubs.append(VOXELSxCUESxSUBS) # add to list to save out
        #     list_of_Resids.append(resids_data_array)# add to list to save ou
        # # now save out lists
        # pickle.dump(list_of_VoxelsXcuesXsubs, open(os.path.join(project_dir,"RSA","cortical", ("statmethod-"+stats_method+"_list_of_VoxelsXcuesXsubs.p")), "wb"), protocol=4)
        # pickle.dump(list_of_Resids, open(os.path.join(project_dir,"RSA","cortical", ("statmethod-"+stats_method+"_list_of_Resids.p")), "wb"), protocol=4)
    
    
elif use_subcortical:
    if basal_ganglia:
        out_dir = os.path.join(project_dir,"RSA","subcortical","basal_ganglia","intermediate_files")
        # -- variables specific to subcortical call
        roi_name_list = ['BasalGanglia','Caudate','Putamen','GlobusPallidus'] # has to match order of roi_binary_mask
        roi_binary_mask=['BG_2.5mm.nii.gz', 'BG_Caudate_2.5mm.nii.gz', 'BG_Putamen_2.5mm.nii.gz', 'BG_Globus_Pallidus_2.5mm.nii.gz']
    elif thalamus:
        out_dir = os.path.join(project_dir,"RSA","subcortical","thalamus","intermediate_files")
        # -- variables specific to subcortical call
        roi_name_list = ['thalamus','AN','VL','MD','LP','IL','VA','PuM','PuL','VP'] # has to match order of roi_binary_mask
        roi_binary_mask=['Morel_2.5_mask.nii.gz',
                        'Morel_2.5_AN.nii.gz', 'Morel_2.5_VL.nii.gz', 'Morel_2.5_MD.nii.gz',
                        'Morel_2.5_LP.nii.gz', 'Morel_2.5_IL.nii.gz', 'Morel_2.5_VA.nii.gz',
                        'Morel_2.5_PuM.nii.gz', 'Morel_2.5_PuL.nii.gz', 'Morel_2.5_VP.nii.gz']
    else:
        print("Error!!! Not sure what subcortical structure to run RSA on")
    num_rois = len(roi_name_list)
    
    # -- add path
    roi_binary_masks = []
    for cur_mask in roi_binary_mask:
        roi_binary_masks.append(os.path.join(mask_dir, cur_mask))
    
    # -- initialize empty lists to fill below
    sub_list = [] # can be deleted
    list_of_VoxelsXcuesXsubs = []
    list_of_Resids = []
    
    if not(skip_setup):
        # --------  set up to loop through ROIs  -------- #
        # for roi, cur_roi in enumerate(roi_name_list):
        jobs = []
        input_lists = []
        for c_ind, c_entry in enumerate(roi_name_list):
            input_lists.append([c_entry, c_ind, roi_binary_masks])
        # --------  use parallel processing to generate inputs for RSA  -------- #
        for i, input_list in enumerate(input_lists):
            j = multiprocessing.Process(target=parallel_gen_input_matrices, args=(i, input_list, stats_method, subject_list, cue_list, out_dir))
            jobs.append(j)
        # now start the jobs
        print("Starting jobs for loading and prepping input matrices")
        for j in jobs:
            j.start() 
        # pause until all jobs finish
        for job in jobs:
            job.join()
    
    # loop through and load individual matrices to make the larger list
    for roi, cur_roi in enumerate(roi_name_list):
        VOXELSxCUESxSUBS = np.load(os.path.join(out_dir,("statmethod-"+stats_method+"_roi-"+cur_roi+"_VOXELSxCUESxSUBS.npy")))
        resids_data_array = np.load(os.path.join(out_dir,("statmethod-"+stats_method+"_roi-"+cur_roi+"_residuals.npy")))
        list_of_VoxelsXcuesXsubs.append(VOXELSxCUESxSUBS) # add to list to save out
        list_of_Resids.append(resids_data_array)# add to list to save ou
    # now save out lists
    pickle.dump(list_of_VoxelsXcuesXsubs, open(os.path.join(out_dir, ("statmethod-"+stats_method+"_list_of_VoxelsXcuesXsubs.p")), "wb"), protocol=4)
    pickle.dump(list_of_Resids, open(os.path.join(out_dir, ("statmethod-"+stats_method+"_list_of_Resids.p")), "wb"), protocol=4)
    

else:
    print("\nError! ... User entered FALSE for use_cortical and FALSE for use_subcortical ... not sure what voxels to use ...\n")
    



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - -      set up for RSA (400rois | searchlight)     - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if use_cortical: 
    if use_searchlight:
        if args.gen_searchlight_plots:
            print("no need to set up for RSA... moving on to next section")
        else:
            # ---- setup for RSA ... using parallel processing again
            gm_mask = nib.load(os.path.join(mask_dir,("GreyMatter_Mask_task-"+project_name+".nii.gz")))
            
            # # # Convert above 4D array to a niftii image
            if os.path.exists(os.path.join(project_dir, "RSA", "cortical", "searchlight", "intermediate_files", "cues_by_subjs.nii.gz")):
                imgs = nib.load(os.path.join(project_dir, "RSA", "cortical", "searchlight", "intermediate_files", ("cues_by_subjs.nii.gz")))
            else:
                imgs = nib.Nifti1Image(voxels3d_by_cuesXsubjs, affine=img.affine, header=img.header)
                nib.save(imgs, os.path.join(project_dir, "RSA", "cortical", "searchlight", "intermediate_files", ("cues_by_subjs.nii.gz")))
            # check if image is 4D
            imgs = check_niimg_4d(imgs)
            # Get the seeds
            process_mask_img = gm_mask # cortical_mask
            # Compute world coordinates of the seeds
            process_mask, process_mask_affine = masking._load_mask_img(
                process_mask_img)
            process_mask_coords = np.where(process_mask != 0)
            process_mask_coords = coord_transform(
                process_mask_coords[0], process_mask_coords[1],
                process_mask_coords[2], process_mask_affine)
            process_mask_coords = np.asarray(process_mask_coords).T
            # get variables we need to create spheres and loop/parallel process
            X, A = _apply_mask_and_get_affinity(
                    process_mask_coords, imgs, 10, True,
                    mask_img=gm_mask) # used cortical_mask originally
            # X will be size 416 (subject by cue) by #_voxels
            # A will be size seeds by #_voxels
            #   each row of A will be a seed and it's nearest neighbors
            num_of_spheres = X.shape[1]
            num_subjects = len(sub_list)
            # use A.rows as indices to pull out current searchlight voxels from X
            #
            # set up for parallel processing
            # make sure to include voxel (or A row index) so output is in order

            ### restructring the A because the way pool works is to run parallel loops for "lists", so we need to restructure our data 
            # into lists, where each item in the list will be distributred to parallel loops
            A_list = []
            for n in np.arange(num_of_spheres):
                A_list.append(A.rows[n])  # turning A from sparse matrix to list
            print("A_list length:", len(A_list))

            ### put X, A, and resid in "shared memory" so the parallel loop can access them
            from multiprocessing import shared_memory
            
            # -- create share memory buffer for X
            X_shm = shared_memory.SharedMemory(create=True, size=X.nbytes)
            print("shared memory created for X data")
            try:
                # put a version of X in share memory
                X_in_shm = np.ndarray(X.shape, buffer=X_shm.buf)
                X_in_shm[:] = X[:]
                del X # save memory
                X_shm_name = X_shm.name # this is the memory space "name" of the share memory space that will feed into the parallel loop
                X_dim = X_in_shm.shape
                print("X data stored in shared memory")
            except:
                X_shm.close()
                X_shm.unlink()

            # -- create share memory buffer for residuals
            resid_shm = shared_memory.SharedMemory(create=True, size=resids_data_array.nbytes)
            print("shared memory created for residual data")
            try:
                # put a version of resids_data_array in share memory
                resid_in_shm = np.ndarray(resids_data_array.shape, buffer=resid_shm.buf)
                resid_in_shm[:] = resids_data_array[:]
                del resids_data_array # save memory
                resid_name = resid_shm.name
                resid_dim = resid_in_shm.shape
                print("residual data stored in shared memory")
            except:
                X_shm.close()
                resid_shm.close()
                resid_shm.unlink()
                X_shm.unlink()
            
            # -- create pool for multiprocessing
            ct = datetime.datetime.now()
            print("pool setup time:-", ct)
            pool = multiprocessing.Pool(int(num_cores))
            try:
                test_num_of_sphere_seeds = len(A_list) # will be total number of spheres
                batch_num_of_sphere_seeds = int(test_num_of_sphere_seeds / nbatches) # will be the number of spheres to run in this batch
                
                start_idx_batch = int((batch_num-1) * batch_num_of_sphere_seeds)
                if (test_num_of_sphere_seeds % nbatches) != 0:
                    # add extra jobs to last batch
                    if nbatches == batch_num:
                        extra_spheres = test_num_of_sphere_seeds % nbatches
                        batch_num_of_sphere_seeds = int(batch_num_of_sphere_seeds + extra_spheres)
                end_idx_batch = int(start_idx_batch + batch_num_of_sphere_seeds)
                print("start: ", start_idx_batch, "\tend: ", end_idx_batch)
                        
                list_of_seeds = list(range(test_num_of_sphere_seeds))
                if args.mult_comp_corr:
                    do_mult_comp_corr = args.mult_comp_corr
                else:
                    do_mult_comp_corr = False
                #print(do_mult_comp_corr)

                # batch method changed A_list and list_of_seeds indexing calls
                input_lists = zip(
                    [X_shm_name]*batch_num_of_sphere_seeds, 
                    [X_dim]*batch_num_of_sphere_seeds,
                    A_list[start_idx_batch:end_idx_batch], 
                    [cue_list]*batch_num_of_sphere_seeds, 
                    [sub_list]*batch_num_of_sphere_seeds, 
                    [resid_name]*batch_num_of_sphere_seeds,
                    [resid_dim]*batch_num_of_sphere_seeds,
                    list_of_seeds[start_idx_batch:end_idx_batch], 
                    [do_mult_comp_corr]*batch_num_of_sphere_seeds,
                    [project_dir]*batch_num_of_sphere_seeds) #, [progress_name]*test_num_of_sphere_seeds, [progress_dim]*test_num_of_sphere_seeds) #this is crearte an iterable object putting all inputs into list of tuples, that will be upacked in the function. The length of this list is the numer of spheres
            except:
                X_shm.close()
                resid_shm.close()
                pool.close()
                resid_shm.unlink()
                X_shm.unlink()
    
    
    if use_schaefer_400rois:
        # ---- set up for parallel (with rois)    
        # -- initialize empty lists to fill below
        list_of_VoxelsXcuesXsubs = []
        list_of_Resids = []
        
        # --------  set up to loop through ROIs  -------- #
        # will parallel process so that each ROI is run as it's own process
        chunk_size = int(num_rois / num_cores) # use njobs argument to determine chunk size
        input_lists = []
        for c_ind, roi_name in enumerate(roi_name_list):
            c_roi_num = int(c_ind+start_roi_num)
            resids_masked_list = [] # will be 59 subjects long
            nii_masked_list = []
            for ind_s, subject in enumerate(subject_list):
                # initialize arrays
                cur_roi_inds = np.where(mask_vec_list[ind_s]==c_roi_num)[0] # use schaefer mask file to get current indices
                resids_masked_list.append(resids_2D_list[ind_s][:,cur_roi_inds]) # resids_2D_list = [time, voxels]
                nii_masked_list.append(nii_2D_list[ind_s][cur_roi_inds,:]) # nii_2D_list = [voxels, cues]
            # input_list = [0:roi_name, 1:roi_num, 2:cur_roi_inds, 3:resids_masked_list, 4:nii_masked_list, 5:stats_method, 6:subject_list, 7:cue_list, 8:out_dir]
            input_lists.append([roi_name, c_roi_num, cur_roi_inds, resids_masked_list, nii_masked_list, stats_method, subject_list, cue_list, out_dir])
            del resids_masked_list
            del nii_masked_list
            del cur_roi_inds
            
        # # --------  use parallel processing to generate inputs for RSA  -------- #
        # print("Starting jobs for loading and prepping input matrices ... split into " + str(int(chunk_size+1)) + "groups of " + str(num_cores) + " to avoid memory errors")
        # print("start time:-", datetime.datetime.now())
        # for batch_idx in range(chunk_size):
        #     print("working on batch number ", str(int(batch_idx+1)), " out of ", str(chunk_size))
        #     start_idx = int(batch_idx*num_cores)
        #     end_idx = int((batch_idx*num_cores)+num_cores)
        #     jobs = [] # reset as empty on each loop
        #     for i, input_list in enumerate(input_lists[start_idx:end_idx]):
        #         j = multiprocessing.Process(target=parallel_RSA_func_cortical, args=(i, input_list))
        #         jobs.append(j)
        #     for j in jobs:
        #         j.start() 
        #     # pause until all jobs finish
        #     for job in jobs:
        #         job.join()
        #     del jobs
        
        # # ---- setup for RSA ... using parallel processing again
        # sub_list = subject_list
        # total = len(roi_name_list) # total number of processes
        # chunk_size = int(total / num_cores) # use njobs argument to determine chunk size
        # input_lists = []
        # for roi, roi_name in enumerate(roi_name_list):
        #     input_lists.append([list_of_VoxelsXcuesXsubs[roi], list_of_Resids[roi], roi_name, roi])
        # print("Starting jobs for loading and prepping input matrices ... split into " + str(chunk_size) + "groups of " + str(num_cores) + " to avoid memory errors")
        # print("start time:-", datetime.datetime.now())
        # for batch_idx in range(chunk_size):
        #     print("starting job batch number " + str(batch_idx) + " out of " + str(chunk_size) + " total job batches\nbatch start time:-", datetime.datetime.now())
        #     start_idx = int(batch_idx*num_cores)
        #     end_idx = int((batch_idx*num_cores)+num_cores)
        #     jobs = [] # reset as empty on each loop
        #     # --------  use parallel processing to run RSA  -------- #
        #     for i, input_list in enumerate(input_lists[start_idx:end_idx]):
        #         j = multiprocessing.Process(target=parallel_RSA_func, args=(i, input_list, stats_method, cue_list, os.path.join(project_dir,"RSA","cortical","intermediate_files")))
        #         jobs.append(j) 
        #     for j in jobs:
        #         j.start() 
        #     # pause until all jobs finish
        #     for job in jobs:
        #         job.join()
        #     print("batch number " + str(batch_idx) + " finish time:-", datetime.datetime.now())
        print("finish time:-", datetime.datetime.now())

elif use_subcortical:
    if basal_ganglia:
        out_dir = os.path.join(project_dir,"RSA","subcortical","basal_ganglia","intermediate_files")
        results_dir = os.path.join(project_dir,"RSA","subcortical","basal_ganglia","results")
    elif thalamus:
        out_dir = os.path.join(project_dir,"RSA","subcortical","thalamus","intermediate_files")
        results_dir = os.path.join(project_dir,"RSA","subcortical","thalamus","results")
    else:
        print("Error!!! Not sure what subcortical structure to run RSA on")
    # ---- setup for RSA ... using parallel processing again
    print("start time:-", datetime.datetime.now())
    sub_list = subject_list
    try:
        total = len(roi_name_list)
        chunk_size = total / len(roi_name_list) # going to be 1
        slice = chunks(list_of_VoxelsXcuesXsubs, list_of_Resids, roi_name_list, chunk_size)
        print("number of jobs: ", len(slice), "\tchunk size: ", chunk_size)
        jobs = []
        for i, s in enumerate(slice):
            j = multiprocessing.Process(target=parallel_RSA_func, args=(i, s, stats_method, cue_list, out_dir))
            jobs.append(j)
        for j in jobs:
            j.start()
            j.join()
        print("finish time:-", datetime.datetime.now())
    except Exception as e:
        #need to close shared_memory
        print("encountered error and need to close script")
        print(e)
        print("crash time:-", datetime.datetime.now())

else:
    print("Error!!!! not sure if running cortical or subcortical")



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - -      Run statistics (400rois | searchlight)     - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
try:
    from scipy.stats import zscore
    from pymer4.models import Lm
    from pymer4.models import Lmer
except:
    print("couldn't load R packages... if Lmer was requested the code will crash")

if use_cortical:
    # code
    
    if use_searchlight:
        if not(args.gen_searchlight_plots):
            # # # # # # # # # # # # # # # # # # # # # # #
            # # # # #   SEARCHLIGHT
            # # # # # # # # # # # # # # # # # # # # # # #
            ct = datetime.datetime.now()
            print("start time:-", ct)
            try:
                results = pool.map(RSA_cues_for_parallel, input_lists)
                ct = datetime.datetime.now()
                print("finish time:-", ct)
                pool.close()
                pool.join()
            except Exception as e:
                print(repr(e))
                print("encountered error and need to close script")
                print("crash time:-", datetime.datetime.now())
                #need to close shared_memory
                X_shm.close()
                resid_shm.close()
                pool.close()
                resid_shm.unlink()
                X_shm.unlink()
            print("saving out results...")
            pickle.dump(results, open(os.path.join(project_dir, "RSA", "cortical", "searchlight", "results", ("searchlight_rsa_results_batch"+str(batch_num)+"of"+str(nbatches)+".p")), "wb"), protocol=4)
            # results will be a list where each entry is the dictionary of results from each sphere... [{'Context_beta':[#.#, ...], ...}, {'Context_beta':[#.#, ...], ...}, ..., {'Context_beta':[#.#, ...], ...}]
        
        try:
            # Model:  Y ~ C + RF + TP + R + C:RF + C:TP + C:R + RF:TP + RF:R +TP:R + C:RF:TP + C:RF:R + C:TP:R + RF:TP:R + C:RF:TP:R
            #        |__ main effects ___| |_____________ all 2way Is ____________| |____________ all 3ways ____________| |__ 4way _|
            def create_nii(stats_mat):
                cortical_masker = NiftiMasker(gm_mask)
                cortical_masker.fit()
                stat_nii = cortical_masker.inverse_transform(stats_mat)
                return stat_nii
            
            def prep_stats_within_subj(results, test_num_of_sphere_seeds, stats, stats2, sub_idx):
                tmp_data = np.zeros(test_num_of_sphere_seeds)
                tmp_data2 = np.zeros(test_num_of_sphere_seeds)
                for i in np.arange(test_num_of_sphere_seeds):
                    vox_idx = results[i]['sphere_idx']
                    tmp_data[vox_idx] = results[i][stats][sub_idx]
                    if not(stats2==[]):
                        if (results[i][stats][sub_idx] > results[i][stats2][sub_idx]):
                            tmp_data2[vox_idx] = 1
                        else:
                            tmp_data2[vox_idx] = 0
                return tmp_data, tmp_data2
            
            def gen_bmaps_and_group_stats(project_dir, sub_list, results, c_beta_key):
                if os.path.exists(os.path.join(project_dir, "RSA", "cortical", "searchlight", "results", "plots", (c_beta_key+"_map.npy"))):
                    print("all subject beta map numpy file aleady created... loading now")
                    c_bmaps = np.load(os.path.join(project_dir, "RSA", "cortical", "searchlight", "results", "plots", (c_beta_key+"_map.npy")))
                    for subj_idx, cur_subj in enumerate(sub_list):
                        # get beta map for current subject
                        c_stat_str = c_beta_key.split("_")[0]
                        c_tmp = c_bmaps[:,subj_idx]
                        c_group_tnii = create_nii(c_tmp)
                        c_group_tnii.to_filename(os.path.join(project_dir, "RSA", "cortical", "searchlight", "results", "plots", "subject_level_beta_maps", ("sub-"+cur_subj+"_"+c_stat_str+"_beta.nii")))
                        print("beta map for subject ", cur_subj, " saved...")
                else:
                    c_bmaps = np.zeros((len(results),len(sub_list)))
                    for subj_idx, cur_subj in enumerate(sub_list):
                        # get beta map for current subject
                        c_tmp, c_sigmask = prep_stats_within_subj(results, len(results), c_beta_key, [], subj_idx) 
                        # add cur sub beta map to matrix
                        c_bmaps[:,subj_idx] = c_tmp
                        if not(os.path.exists(os.path.join(project_dir, "RSA", "cortical", "searchlight", "results", "plots", (c_beta_key+"_map.npy")))):
                            c_group_tnii = create_nii(c_tmp)
                            c_group_tnii.to_filename(os.path.join(project_dir, "RSA", "cortical", "searchlight", "results", "plots", "subject_level_beta_maps", ("sub-"+cur_subj+"_"+c_stat_str+"_beta.nii")))
                    np.save(os.path.join(project_dir, "RSA", "cortical", "searchlight", "results", "plots", (c_beta_key+"_map.npy")), c_bmaps)
                # -- group stats
                c_stat_str = c_beta_key.split("_")[0]
                c_group_stats = np.zeros((len(results),2)) # 2nd dim is tval and pval in that order
                for v_idx in range(len(results)):
                    c_group_stats[v_idx,0], c_group_stats[v_idx,1] = stats.ttest_1samp(np.asarray(c_bmaps[v_idx,:]), popmean=0, nan_policy='omit')
                c_group_tnii = create_nii(c_group_stats[:,0])
                c_group_tnii.to_filename(os.path.join(project_dir, "RSA", "cortical", "searchlight", "results", "plots", ("GroupAnalysis_"+c_stat_str+"_tval.nii")))
            
            # # # # # # # # # # # # # # # # # # # # # # #
            # # # # #   SEARCHLIGHT PLOTS
            # # # # # # # # # # # # # # # # # # # # # # #
            try:
                print(len(results))
            except:
                print("loading results")
                results = pickle.load(open(os.path.join(project_dir, "RSA", "cortical", "searchlight", "results", ("searchlight16_rsa_results.p")), "rb"))
            print("results loaded... generating plots")
            beta_key_list = [ckey for ckey in results[0].keys() if "beta" in ckey]  #[ckey for ckey in results[0].keys() if "beta" in ckey] # get list of dictionary keys with "beta" in the name
            for c_beta_key in beta_key_list:
                print("creating beta maps and group stats for ", c_beta_key)
                gen_bmaps_and_group_stats(project_dir, sub_list, results, c_beta_key)
            
        except Exception as e:
            print("unable to run nii creation for within subj regression results")
         
        
        
    if use_schaefer_400rois:
        # ---- now load the data and create the larger file
        SUBxROIxCOEFF_mat = np.zeros( (len(subject_list), num_rois, len(cue_list), len(cue_list)) ) 
        print("combining roi outputs into the larger file...")
        for roi, roi_name in enumerate(roi_name_list):
            curROImat = np.load(os.path.join(project_dir,"RSA","cortical","intermediate_files",("statmethod-"+stats_method+"_roi-"+roi_name+"_"+str(len(subject_list))+"subjectsCoeffMatrix.npy")))
            SUBxROIxCOEFF_mat[:,roi,:,:] = curROImat
        print("full SUBxROIxCOEFF_mat matrix created!")
        
        # --------  NOW RUN STATISTICS ON THE DATA  -------- #
        results = {} # will be a dict where roi number (e.g., 1) will be the key to access the stats from that roi
        # -- initialize lists, arrays, and variables
        model_opts = {1:'LinearRegression', 2:'MixedEffects', 3:'LinearRegressionPerSubject', 4:'misc'}
        # ... LinearRegression = runs a linear regression across subjects (aka subject included as a regressor)
        # ... MixedEffects = runs a mixed effects model with random intercepts and slopes
        # ... LinearRegressionPerSubject = runs a linear regression for EACH subject and then saves out these subject-level regressions
        model_type = model_opts[3] # options are: 1=linear_regression  |  2=mixed_effects  |  3=linear_regression_per_subj
        regressors_list, lower_triangle_inds = set_up_model_vars(cue_list)
        
        # -- LOOP through ROIs
        if num_rois != SUBxROIxCOEFF_mat.shape[1]:
            print("MAJOR ERROR!!!!! NUMBER OF ROIS DOES NOT MATCH ROI DIMENSION ON SUBxROIxCOEFF_mat")
            print("SUBxROIxCOEFF_mat.shape[1]: ", SUBxROIxCOEFF_mat.shape[1])
        
        for roi_idx, roi_name in enumerate(roi_name_list):
            print("\ncurrently on ROI:", roi_name)
            SUBxCOEFF_mat=SUBxROIxCOEFF_mat[:,roi_idx,:,:] # pull out sub and coeff mat for current roi
            
            # ---- initialize model info
            y_vec, context_vec, relFeat_vec, taskPerform_vec, resp_vec = create_model_vecs(subject_list,lower_triangle_inds) # vectors will by 59*lower_triangle_inds
            data_Ctxt, data_Iden, dcfs_relF, dcfs_tskR, dsfc_relF, dsfc_tskR, data_Resp, data_color, data_shape = gen_RSA_models_new(cue_list) 
            color_vec=np.zeros((len(subject_list)*len(lower_triangle_inds)))
            shape_vec=np.zeros((len(subject_list)*len(lower_triangle_inds)))
            version_info = pd.read_csv(os.path.join(project_dir, "Version_Info.csv"))

            # ---- LOOP through subjects to set up Y and regressors
            for idx, sub in enumerate(subject_list):
                Coeff_mat = SUBxCOEFF_mat[idx,:,:] #ROIxCOEFF_mat[(roi-1),:,:]
                # -- Pull out lower triangle from coeff mat for this subject
                # get start and end points for insertion into vec
                start_pt = idx*len(lower_triangle_inds)
                end_pt = start_pt+len(lower_triangle_inds)
                #print("start point =",start_pt,"\tend point =",end_pt)
                # -- set up y vec as the coefficient matrix values
                coeff_vec = np.tril(Coeff_mat, k=-1).flatten()
                y_vec[start_pt:end_pt]=coeff_vec[lower_triangle_inds] # add to y vector for model
                # -- set up other model vectors based on what version the current sub did
                context_vec[start_pt:end_pt]=np.tril(data_Ctxt).flatten()[lower_triangle_inds]
                color_vec[start_pt:end_pt]=data_color.flatten()[lower_triangle_inds]
                shape_vec[start_pt:end_pt]=data_shape.flatten()[lower_triangle_inds]
                resp_vec[start_pt:end_pt]=np.tril(data_Resp).flatten()[lower_triangle_inds]
                if int(version_info['sub'][idx]) != int(sub):
                    print("ERROR!!! Subject doesn't match version list row!", "\nsub = ", str(int(sub)), "\tsubject from version info row = ", str(int(version_info['sub'][idx])))
                    "c"+2
                if version_info["version"][idx]=="DCFS":
                    #    not swapped version
                    #relFeat_vec[start_pt:end_pt]=np.tril(dcfs_relF).flatten()[lower_triangle_inds]
                    taskPerform_vec[start_pt:end_pt]=np.tril(dcfs_tskR).flatten()[lower_triangle_inds]
                else:
                    #relFeat_vec[start_pt:end_pt]=np.tril(dsfc_relF).flatten()[lower_triangle_inds]
                    taskPerform_vec[start_pt:end_pt]=np.tril(dsfc_tskR).flatten()[lower_triangle_inds]
            
            # ---- now that we have vectors, set up data dictionary
            model_data_dict = {'Y': y_vec, 'Intercept': np.ones((len(y_vec))), 
                                'Context': context_vec, 'Color': color_vec, 'Shape': shape_vec, 
                                'TaskPerformed': taskPerform_vec, 'Resp': resp_vec}

            # ---- now that we have vectors filled in with all subjects... RUN MODEL CODE
            cur_roi_dict, cur_df, corr_results = create_regressor_dataframe(model_type, regressors_list, model_data_dict , subject_list, lower_triangle_inds, roi_name)

            # -- save out data frame for current ROI (usefull for double checking data input to model if needed)
            cur_df.to_csv(os.path.join(project_dir, "RSA", "cortical", "intermediate_files", ("roi-"+roi_name+"_statsmethod-"+stats_method+"_modelmethod-"+model_type+"__dataframe.csv")))
            #print(cur_roi_dict)
            
            if model_type == model_opts[3]:
                # # if first roi, set up results data frame
                if roi_idx == 0:
                    results['roi'] = []
                    results['sub'] = []
                    for key_to_add in cur_roi_dict.keys():
                        results[key_to_add] = [] # initialize as empty list
                # -- add current roi results to overall results dict AND calculate and add permutation results to overall results dict
                #print(results)
                for cur_key in results.keys():
                    print(cur_key)
                    if cur_key == 'roi':
                        for sub_ind, subj in enumerate(subject_list):
                            results['roi'].append(roi_name)
                    elif cur_key == 'sub':
                        for sub_ind, subj in enumerate(subject_list):
                            results['sub'].append(subj)
                    else:
                        for sub_ind, subj in enumerate(subject_list):
                            #print(cur_roi_dict[cur_key][sub_ind])
                            results[cur_key].append(cur_roi_dict[cur_key][sub_ind]) # original result
                #print(results)
            elif model_type == model_opts[2]:
                # # if first roi, set up results data frame
                if roi_idx == 0:
                    results['roi'] = []
                    for key_to_add in cur_roi_dict.keys():
                        results[key_to_add] = [] # initialize as empty list
                # -- add current roi results to overall results dict AND calculate and add permutation results to overall results dict
                #print(results)
                for cur_key in results.keys():
                    print(cur_key)
                    if cur_key == 'roi':
                        results['roi'].append(roi_name)
                    else:
                        results[cur_key].append(cur_roi_dict[cur_key]) # original result
                #print(results)
        
        # ---- now save results dict as a data frame for easy viewing
        # print(results)
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(project_dir, "RSA", "cortical", "results", ("statsmethod-"+stats_method+"_modelmethod-"+model_type+"__schaefer400rois__results.csv")))
        
    
if use_subcortical:
    # ---- now load the data and create the larger file
    SUBxROIxCOEFF_mat = np.zeros( (len(sub_list), num_rois, len(cue_list), len(cue_list)) ) 
    print("combining roi outputs into the larger file...")
    for roi, roi_name in enumerate(roi_name_list):
        curROImat = np.load(os.path.join(out_dir,("statmethod-"+stats_method+"_roi-"+roi_name+"_"+str(len(sub_list))+"subjectsCoeffMatrix.npy")))
        SUBxROIxCOEFF_mat[:,roi,:,:] = curROImat
    print("full SUBxROIxCOEFF_mat matrix created!")

    # --------  NOW RUN STATISTICS ON THE DATA  -------- #
    results = {} # will be a dict where roi number (e.g., 1) will be the key to access the stats from that roi
    corr_results = {}
    # -- initialize lists, arrays, and variables
    model_opts = {1:'LinearRegression', 2:'MixedEffects', 3:'LinearRegressionPerSubject', 4:'misc'}
    # ... LinearRegression = runs a linear regression across subjects (aka subject included as a regressor)
    # ... MixedEffects = runs a mixed effects model with random intercepts and slopes
    # ... LinearRegressionPerSubject = runs a linear regression for EACH subject and then saves out these subject-level regressions
    model_type = model_opts[1] # options are: 1=linear_regression  |  2=mixed_effects  |  3=linear_regression_per_subj
    regressors_list, lower_triangle_inds = set_up_model_vars(cue_list)
    #regressor_list = ["Intercept","Context","RelevantFeature","TaskPerformed","Resp"]
    # -- LOOP through ROIs
    if num_rois > 1:
        if num_rois != SUBxROIxCOEFF_mat.shape[1]:
            print("MAJOR ERROR!!!!! NUMBER OF ROIS DOES NOT MATCH ROI DIMENSION ON SUBxROIxCOEFF_mat")
            print("SUBxROIxCOEFF_mat.shape[1]: ", SUBxROIxCOEFF_mat.shape[1])
    
    for roi_idx, roi_name in enumerate(roi_name_list):
        print("\ncurrently on ROI:", roi_name)
        if num_rois > 1:
            SUBxCOEFF_mat=SUBxROIxCOEFF_mat[:,roi_idx,:,:] # pull out sub and coeff mat for current roi
        else:
            SUBxCOEFF_mat=SUBxROIxCOEFF_mat[:,:,:]
        print("size of SUBxCOEFF_mat is ", SUBxCOEFF_mat.shape)
        
        # ---- initialize model info
        y_vec, context_vec, relFeat_vec, taskPerform_vec, resp_vec = create_model_vecs(subject_list,lower_triangle_inds) # vectors will by 59*lower_triangle_inds
        data_Ctxt, data_Iden, dcfs_relF, dcfs_tskR, dsfc_relF, dsfc_tskR, data_Resp, data_color, data_shape = gen_RSA_models_new(cue_list) 
        color_vec=np.zeros((len(sub_list)*len(lower_triangle_inds)))
        shape_vec=np.zeros((len(sub_list)*len(lower_triangle_inds)))
        version_info = pd.read_csv(os.path.join(project_dir, "Version_Info.csv"))

        # ---- LOOP through subjects to set up Y and regressors
        for idx, sub in enumerate(subject_list):
            Coeff_mat = SUBxCOEFF_mat[idx,:,:] #ROIxCOEFF_mat[(roi-1),:,:]
            # -- Pull out lower triangle from coeff mat for this subject
            # get start and end points for insertion into vec
            start_pt = idx*len(lower_triangle_inds)
            end_pt = start_pt+len(lower_triangle_inds)
            #print("start point =",start_pt,"\tend point =",end_pt)
            # -- set up y vec as the coefficient matrix values
            coeff_vec = np.tril(Coeff_mat, k=-1).flatten()
            y_vec[start_pt:end_pt]=coeff_vec[lower_triangle_inds] # add to y vector for model
            # -- set up other model vectors based on what version the current sub did
            context_vec[start_pt:end_pt]=np.tril(data_Ctxt).flatten()[lower_triangle_inds]
            color_vec[start_pt:end_pt]=data_color.flatten()[lower_triangle_inds]
            shape_vec[start_pt:end_pt]=data_shape.flatten()[lower_triangle_inds]
            resp_vec[start_pt:end_pt]=np.tril(data_Resp).flatten()[lower_triangle_inds]
            if int(version_info['sub'][idx]) != int(sub):
                print("ERROR!!! Subject doesn't match version list row!", "\nsub = ", str(int(sub)), "\tsubject from version info row = ", str(int(version_info['sub'][idx])))
                "c"+2
            if version_info["version"][idx]=="DCFS":
                #    not swapped version
                #relFeat_vec[start_pt:end_pt]=np.tril(dcfs_relF).flatten()[lower_triangle_inds]
                taskPerform_vec[start_pt:end_pt]=np.tril(dcfs_tskR).flatten()[lower_triangle_inds]
            else:
                #relFeat_vec[start_pt:end_pt]=np.tril(dsfc_relF).flatten()[lower_triangle_inds]
                taskPerform_vec[start_pt:end_pt]=np.tril(dsfc_tskR).flatten()[lower_triangle_inds]
        
        # ---- now that we have vectors, set up data dictionary
        model_data_dict = {'Y': y_vec, 
                        'Intercept': np.ones((len(y_vec))), 
                        'Context': context_vec, 
                        'Color': color_vec, 
                        'Shape': shape_vec, 
                        'TaskPerformed': taskPerform_vec,
                        'Resp': resp_vec}

        # ---- now that we have vectors filled in with all subjects... RUN MODEL CODE
        cur_roi_dict, cur_df, corr_dict = create_regressor_dataframe(model_type, regressors_list, model_data_dict , subject_list, lower_triangle_inds, roi_name)

        # -- save out data frame for current ROI (usefull for double checking data input to model if needed)
        cur_df.to_csv(os.path.join(out_dir, ("roi-"+roi_name+"_statsmethod-"+stats_method+"_modelmethod-"+model_type+"__dataframe.csv")))
        #print(cur_roi_dict)
        
        
        # # if first roi, set up results data frame and correlation data frame
        if roi_idx == 0:
            results['roi'] = []
            corr_results['roi'] = []
            if model_type == "LinearRegressionPerSubject":
                results['sub'] = []
                corr_results['sub'] = []
            for key_to_add in cur_roi_dict.keys():
                results[key_to_add] = [] # initialize as empty list
            for key_to_add in corr_dict.keys():
                corr_results[key_to_add] = []
        
        if model_type == "MixedEffects":
            # -- add current roi results to overall results dict AND calculate and add permutation results to overall results dict
            for cur_key in results.keys():
                if cur_key == 'roi':
                    results['roi'].append(roi_name)
                else:
                    results[cur_key].append(cur_roi_dict[cur_key]) # original result
            #print(results)
            for cur_key in corr_results.keys():
                if cur_key == 'roi':
                    corr_results['roi'].append(roi_name)
                else:
                    corr_results[cur_key].append(corr_dict[cur_key]) # original result
        elif model_type == "LinearRegressionPerSubject":
            # -- add current roi results to overall results dict AND calculate and add permutation results to overall results dict
            for cur_key in results.keys():
                if cur_key == 'roi':
                    for sub_ind, subj in enumerate(subject_list):
                        results['roi'].append(roi_name)
                elif cur_key == 'sub':
                    for sub_ind, subj in enumerate(subject_list):
                        results['sub'].append(subj)
                else:
                    for sub_ind, subj in enumerate(subject_list):
                        #print(cur_roi_dict[cur_key][sub_ind])
                        results[cur_key].append(cur_roi_dict[cur_key][sub_ind]) # original result
            #print(results)
    
    # ---- now save results dict as a data frame for easy viewing
    # print(results)
    results_df = pd.DataFrame(results)
    if basal_ganglia:
        results_df.to_csv(os.path.join(results_dir, ("basalganglia"+str(num_rois)+"ROIs_statsmethod-"+stats_method+"_modelmethod-"+model_type+"__results.csv")))
    else:
        results_df.to_csv(os.path.join(results_dir, ("thalamus"+str(num_rois)+"ROIs_statsmethod-"+stats_method+"_modelmethod-"+model_type+"__results.csv")))                  
    
    # -- also save out correlation information 
    corr_df = pd.DataFrame(corr_results)
    corr_df.to_csv(os.path.join(out_dir, ("roi-"+roi_name+"_statsmethod-"+stats_method+"_modelmethod-"+model_type+"__corrRegressors.csv")))
    
    # run t-test ... EDIT THIS CODE FOR NEW MODEL
    # if args.ttest:
    #     stats_dict = {'roi':[], 'Context_tval':[],'Context_pval':[], 'RelevantFeature_tval':[],'RelevantFeature_pval':[], 
    #                                         'TaskPerformed_tval':[],'TaskPerformed_pval':[], 'Resp_tval':[],'Resp_pval':[]}
    #     # results df should already be loaded
    #     for roi_ind, cur_roi in enumerate(roi_name_list):
    #         roi_df = results_df[results_df['roi']==cur_roi]

    #         context_tval, context_pval = stats.ttest_1samp(np.asarray(roi_df['Context_beta']), popmean=0, nan_policy='omit')
    #         relfeat_tval, relfeat_pval = stats.ttest_1samp(np.asarray(roi_df['RelevantFeature_beta']), popmean=0, nan_policy='omit')
    #         taskper_tval, taskper_pval = stats.ttest_1samp(np.asarray(roi_df['TaskPerformed_beta']), popmean=0, nan_policy='omit')
    #         resp_tval, resp_pval = stats.ttest_1samp(np.asarray(roi_df['Resp_beta']), popmean=0, nan_policy='omit')

    #         stats_dict['Context_tval'].append(context_tval)
    #         stats_dict['Context_pval'].append(context_pval)
    #         stats_dict['RelevantFeature_tval'].append(relfeat_tval)
    #         stats_dict['RelevantFeature_pval'].append(relfeat_pval)
    #         stats_dict['TaskPerformed_tval'].append(taskper_tval)
    #         stats_dict['TaskPerformed_pval'].append(taskper_pval)
    #         stats_dict['Resp_tval'].append(resp_tval)
    #         stats_dict['Resp_pval'].append(resp_pval)
    #         stats_dict['roi'].append(cur_roi)

    #     stats_df = pd.DataFrame(stats_dict)
    #     stats_df.to_csv(os.path.join(project_dir, "RSA", "thalamus", "results", ("t-test_results.csv")))
    
    # -- generate a plto just for visuals
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    if model_type == "MixedEffects":
        col_idxs = ['roi']
        for cur_col in results_df.columns:
            if ('_beta' in cur_col):
                colname = cur_col.split("_be")[0]
                col_idxs.append(cur_col)
        df_melt = pd.melt(results_df[col_idxs], id_vars = ['roi'])
    elif model_type == "LinearRegressionPerSubject":
        col_idxs = ['roi','sub']
        for cur_col in results_df.columns:
            if ('_beta' in cur_col):
                col_idxs.append(cur_col)
        df_melt = pd.melt(results_df[col_idxs], id_vars = ['sub', 'roi'])
    
    if basal_ganglia:
        df_melt.to_csv(os.path.join(results_dir, ("statsmethod-"+stats_method+"_modelmethod-"+model_type+"__basalganglia_plot_dataframe.csv")))
        fig, axes = plt.subplots(3,1, figsize=(24,12))
    else:
        df_melt.to_csv(os.path.join(results_dir, ("statsmethod-"+stats_method+"_modelmethod-"+model_type+"__thalamus_plot_dataframe.csv")))  
        fig, axes = plt.subplots(3,3, figsize=(36,12))
    df_melt2 = df_melt[df_melt['variable']!='Intercept_beta']
    
    
    if basal_ganglia:
        for idx, c_roi in enumerate(roi_name_list[1:]):
            sns.barplot(x="variable", y="value", data=df_melt2[df_melt2['roi']==c_roi], ax=axes[idx%3]) #[int(np.floor(idx/3))])
        plt.plot()
        plt.savefig(os.path.join(project_dir,"RSA","subcortical","figures",("statsmethod-"+stats_method+"_modelmethod-"+model_type+"__basalganglia_results_bar_plot.png")))
    else:
        for idx, c_roi in enumerate(roi_name_list[1:]):
            sns.barplot(x="variable", y="value", data=df_melt2[df_melt2['roi']==c_roi], ax=axes[idx%3][int(np.floor(idx/3))])
        plt.plot()
        plt.savefig(os.path.join(project_dir,"RSA","subcortical","figures",("statsmethod-"+stats_method+"_modelmethod-"+model_type+"__thalamus_results_bar_plot.png"))) # + "/%s_state_model_fit.png" %sub)
    plt.close()
    
    
if args.gen_roi_plot:
    # -- set path/variables
    out_dir = os.path.join(project_dir, 'RSA', 'cortical', 'schaefer_400rois', 'results')
    model_method = "LinearRegressionPerSubject" # "MixedEffects" # 
    model_type = "LinearRegressionPerSubject" # "MixedEffects" # 
    # -- load cortical mask
    cortical_mask = nib.load(os.path.join(mask_dir,"CorticalBinary_rs.nii.gz"))
    
    # mask shaefer 400 rois mask
    mask = nib.load(os.path.join(mask_dir, 'Schaefer400_2.5.nii.gz'))
    mask_vec = nilearn.masking.apply_mask(mask, cortical_mask) # will be 1D (voxels)
    print("mask_vec dimensions ", mask_vec.shape)
    
    # -- load data frame with stats results
    results_df = pd.read_csv(os.path.join(out_dir,("statsmethod-"+stats_method+"_modelmethod-"+model_type+"__schaefer400rois__results.csv")))
    if model_method == "MixedEffects":
        melt_df = pd.melt(results_df, id_vars=["roi"])
    else:
        melt_df = pd.melt(results_df, id_vars=["roi", "sub"])
    reg_list = []
    stat_list = []
    for idx,cv in enumerate(melt_df['variable']):
        if model_method == "MixedEffects":
            print(''.join(cv.split('_',1)[:-1]))
            reg_list.append(''.join(cv.split('_',-1)[:-1]))
            stat_list.append(cv.split('_',-1)[-1])
        else:
            reg_list.append(cv.split('_',1)[0])
            stat_list.append(cv.split('_',1)[1])
    melt_df['regressor'] = reg_list
    melt_df['stat'] = stat_list
    #"c"+2
    if model_method == "MixedEffects":
        beta_df = melt_df[melt_df['stat']=='tval']
        for c_stat in beta_df['regressor'].unique():
            subset_df = beta_df[beta_df['regressor']==c_stat] # reduce to the current regressor ... ( context | color | task )
            
            roi_filledin = np.zeros(mask_vec.shape)
            print("numpy array of rois to fill in size: ", roi_filledin.shape)
            for r_idx, roi in enumerate(range(1,401)):
                roi_subset_df = subset_df[subset_df['roi']==roi]
                # get template indices
                cur_roi_inds = np.where(mask_vec==roi)[0]
                print("current roi number, ", str(roi), " has ", str(len(cur_roi_inds)), " voxels to include")
                #print(float(roi_subset_df['value']))
                roi_filledin[cur_roi_inds] = float(roi_subset_df['value'])
            cortical_masker = NiftiMasker(cortical_mask)
            cortical_masker.fit()
            cur_nii = cortical_masker.inverse_transform(roi_filledin)
            cur_nii.to_filename(os.path.join(out_dir,("GroupAnalysis_"+c_stat+"_tval.nii")))
    else:
        beta_df = melt_df[melt_df['stat']=='beta']
        for c_stat in beta_df['regressor'].unique():
            subset_df = beta_df[beta_df['regressor']==c_stat] # reduce to the current regressor ... ( context | color | task )
            #print(subset_df)
            
            roi_filledin = np.zeros(mask_vec.shape)
            print("numpy array of rois to fill in size: ", roi_filledin.shape)
            for r_idx, roi in enumerate(range(1,401)):
                roi_subset_df = subset_df[subset_df['roi']==roi]
                cur_mean = roi_subset_df['value'].mean()
                # get template indices
                cur_roi_inds = np.where(mask_vec==roi)[0]
                print("current roi number, ", str(roi), " has ", str(len(cur_roi_inds)), " voxels to include in this t-test")
                # t-test
                c_tval, c_pval = stats.ttest_1samp(np.asarray(roi_subset_df['value']), popmean=0, nan_policy='omit')
                print("\tcurrent roi number, ", str(roi), ":  t-value = ", str(c_tval), "   p-value = ", str(c_pval))
                # add back
                roi_filledin[cur_roi_inds] = c_tval
            #cur_nii=nilearn.image.new_img_like(mask, roi_filledin)
            cortical_masker = NiftiMasker(cortical_mask)
            cortical_masker.fit()
            cur_nii = cortical_masker.inverse_transform(roi_filledin)
            cur_nii.to_filename(os.path.join(out_dir,("GroupAnalysis_"+c_stat+"_tval.nii")))
            
            # create subject level plots
            for sub_idx, cur_subj in enumerate(subject_list):
                sub_subset_df = subset_df[subset_df['sub']==int(cur_subj)]
                #print(sub_subset_df)
                roi_filledin = np.zeros(mask_vec.shape)
                for r_idx, roi in enumerate(range(1,401)):
                    cur_roi_inds = np.where(mask_vec==roi)[0]
                    roi_subset_df = sub_subset_df[sub_subset_df['roi']==roi]
                    #print(roi_subset_df)
                    roi_filledin[cur_roi_inds] = float(roi_subset_df['value'])
                # create subject plot
                cortical_masker = NiftiMasker(cortical_mask)
                cortical_masker.fit()
                cur_nii = cortical_masker.inverse_transform(roi_filledin)
                cur_nii.to_filename(os.path.join(out_dir,"LinearRegressionPerSubject","subject_level_beta_maps",("sub-"+cur_subj+"_"+c_stat+"_beta.nii")))