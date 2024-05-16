# try searchlight decoding to classify different task representations
# based on https://nilearn.github.io/dev/auto_examples/02_decoding/plot_haxby_searchlight.html#sphx-glr-auto-examples-02-decoding-plot-haxby-searchlight-py

import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.image import new_img_like, load_img, get_data, index_img
import nibabel as nib
from sklearn.model_selection import KFold
import nilearn.decoding
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR, SVC, LinearSVC
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.metrics import explained_variance_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
import nilearn
import datetime

#import matplotlib.pyplot as plt
#plt.ion()

subject = input()
#subject='10176'
func_path = "/Shared/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/sub-%s/" %subject
version_info = pd.read_csv("/Shared/lss_kahwang_hpc/data/ThalHi/Version_Info.csv")
try:
    version = version_info.loc[version_info['sub'] == int(subject), 'version'].values[0]
except:
    version = 'DCFS'

####  classification with SVM
### compile all the lss cue outputs
dcb = nib.load(func_path +"dcb.LSS.nii.gz")
dcr = nib.load(func_path +"dcr.LSS.nii.gz")
dpb = nib.load(func_path +"dpb.LSS.nii.gz")
dpr = nib.load(func_path +"dpr.LSS.nii.gz")
fcb = nib.load(func_path +"fcb.LSS.nii.gz")
fcr = nib.load(func_path +"fcr.LSS.nii.gz")
fpb = nib.load(func_path +"fpb.LSS.nii.gz")
fpr = nib.load(func_path +"fpr.LSS.nii.gz")

cues = ["dcb", "dcr", "dpb", "dpr", "fcb", "fcr", "fpb", "fpr"]
cue_objects = [dcb, dcr, dpb, dpr, fcb, fcr, fpb, fpr]

# comebine lss outputs
comebined_cues = []
for i, cue in enumerate(cues):
    tr_mask = np.tile([True,False], int(cue_objects[i].shape[3]/2))
    fimg = index_img(cue_objects[i], tr_mask)
    comebined_cues.append(fimg)
all_cues_img = nib.funcs.concat_images(comebined_cues, axis=3)

num_of_betas = all_cues_img.shape[3]

try:
    run_record = pd.read_csv(func_path + "compiled_conditions.csv")
    #if num_of_betas == len(run_record):
    runs = []
    for cue in cues:
        runs = runs + run_record.loc[run_record['Cue'] == cue, 'Run'].values.tolist()
    runs = np.array(runs)
except:
    runs = []
    for i, cue in enumerate(cues):
        runs = runs + np.repeat(i+1, int(cue_objects[i].shape[3]/2)).tolist()
    runs = np.array(runs)
    np.random.shuffle(runs)

# create trial record
trial_records = []
for i, cue in enumerate(cues):
    trial_records = trial_records + [cue] * int(cue_objects[i].shape[3]/2)

mask = index_img(all_cues_img,1)
mask = new_img_like(mask, (mask.get_fdata()!=0))

### now make the Y 
context_Y = []
for cue in trial_records:
    if cue in ["dcb", "dcr", "dpb", "dpr", ]:
        context_Y.append('donut')
    if cue in ["fcb", "fcr", "fpb", "fpr"]:
        context_Y.append('fill')

color_Y = []
for cue in trial_records:
    if cue in ["dcb", "fcb", "dpb", "fpb", ]:
        color_Y.append('blue')
    if cue in ["dcr", "dpr", "fcr", "fpr"]:
        color_Y.append('red')

shape_Y = []
for cue in trial_records:
    if cue in ["dcb", "dcr", "fcb", "fcr", ]:
        shape_Y.append('circle')
    if cue in ["dpb", "dpr", "fpb", "fpr"]:
        shape_Y.append('polygon')

# task is more complicated because of swaped design
if version == 'DCFS':
    task_Y = []
    for cue in trial_records:
        if cue in ["fpr", "fpb", "dcr", "dpr", ]:
            task_Y.append('face')
        if cue in ["fcr", "fcb", "dcb", "dpb"]:
            task_Y.append('scene')

if version == 'DSFC':
    task_Y = []
    for cue in trial_records:
        if cue in ["dpr", "dpb", "fcr", "fpr", ]:
            task_Y.append('face')
        if cue in ["dcb", "dcr", "fcb", "fpb"]:
            task_Y.append('scene')

### searchlight
now = datetime.datetime.now()
print(now)

mean_fmri = nilearn.image.mean_img(all_cues_img)
pipeline = make_pipeline(StandardScaler(), LogisticRegression('l1', solver='liblinear')) 

try:
    searchlight_context = nilearn.decoding.SearchLight(mask, process_mask_img=mask, estimator = pipeline, radius=6, n_jobs=24, verbose=0, cv=LeaveOneGroupOut())
    searchlight_context.fit(all_cues_img, context_Y, runs)
    context_score_img = new_img_like(mean_fmri, searchlight_context.scores_)
    context_score_img.to_filename(func_path + "%s_context_svc_score.nii.gz" %subject)
    now = datetime.datetime.now()
    print(now)
except:
    print("context broke")
    pass

try:
    pipeline = make_pipeline(StandardScaler(), LogisticRegression('l1', solver='liblinear')) 
    searchlight_color = nilearn.decoding.SearchLight(mask, process_mask_img=mask, estimator = pipeline, radius=6, n_jobs=24, verbose=0, cv=LeaveOneGroupOut())
    searchlight_color.fit(all_cues_img, color_Y, runs)
    color_score_img = new_img_like(mean_fmri, searchlight_color.scores_)
    color_score_img.to_filename(func_path + "%s_color_svc_score.nii.gz" %subject)
    now = datetime.datetime.now()
    print(now)
except:
    print("color broke")
    pass

try:
    pipeline = make_pipeline(StandardScaler(), LogisticRegression('l1', solver='liblinear')) 
    searchlight_shape = nilearn.decoding.SearchLight(mask, process_mask_img=mask, estimator = pipeline, radius=6, n_jobs=24, verbose=0, cv=LeaveOneGroupOut())
    searchlight_shape.fit(all_cues_img, shape_Y, runs)
    shape_score_img = new_img_like(mean_fmri, searchlight_shape.scores_)
    shape_score_img.to_filename(func_path + "%s_shape_svc_score.nii.gz" %subject)
    now = datetime.datetime.now()
    print(now)
except:
    print("shape broke")
    pass

try:
    pipeline = make_pipeline(StandardScaler(), LogisticRegression('l1', solver='liblinear')) 
    searchlight_task = nilearn.decoding.SearchLight(mask, process_mask_img=mask, estimator = pipeline, radius=6, n_jobs=24, verbose=0, cv=LeaveOneGroupOut())
    searchlight_task.fit(all_cues_img, task_Y, runs)
    task_score_img = new_img_like(mean_fmri, searchlight_task.scores_)
    task_score_img.to_filename(func_path + "%s_task_svc_score.nii.gz" %subject)
    now = datetime.datetime.now()
    print(now)
except:
    print("task broke")
    pass

### end