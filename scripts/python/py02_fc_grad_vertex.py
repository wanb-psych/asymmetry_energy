import nibabel as nib
import numpy as np
from brainspace.gradient import GradientMaps

thre = [0,0.5,0.9]
sub_list=np.loadtxt('../../sub_list.txt',dtype=str)
embedding='dm'

'''
## fsa5
atlas = np.loadtxt('../../src/fsaverage5.LR.mmp.txt')
ts_all = np.zeros((295, len(atlas)))
for sub in range(len(sub_list)):
  cor_l = nib.load('../../micapipe_v0.2.0/sub-'+sub_list[sub]+'/ses-open/func/desc-se_task-rest_bold/surf/sub-'+sub_list[sub]+'_ses-open_hemi-L_surf-fsaverage5.func.gii').agg_data()
  cor_r = nib.load('../../micapipe_v0.2.0/sub-'+sub_list[sub]+'/ses-open/func/desc-se_task-rest_bold/surf/sub-'+sub_list[sub]+'_ses-open_hemi-R_surf-fsaverage5.func.gii').agg_data()
  cor = np.concatenate((cor_l,cor_r),axis=1)
  ts_all =+ cor
  print('loading......' + sub_list[sub])
  
ts_mean = np.nanmean(ts_all, axis=0)
ts_mean = ts_mean[:,atlas!=0]
corr = np.corrcoef(ts_mean.T).round(6)    
fc = np.arctanh(corr)
fc[fc==np.inf] = 1
fc[np.isnan(fc)] = 0
np.savetxt('/data/p_02801/ds004513_fc_mean_fsa5.txt', fc)
print('saved mean FC matrix at fsa5: /data/p_02801/ds004513_fc_mean_fsa5.txt')
print(fc.shape)

### Gradients
for i in range(len(thre)):
  gm = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=500, random_state=0)
  gm.fit(fc, sparsity=thre[i])
  np.savetxt('../../results/grad/vertex/group_grad_sparsity_'+str(thre[i])+'_lambdas_fsa5.txt', gm.lambdas_)
  np.savetxt('../../results/grad/vertex/group_grad_sparsity_'+str(thre[i])+'_fsa5.txt', gm.gradients_)
  print('finish...fsa5...gradients...sparsity_'+str(thre[i]))
'''
## fsLR-5K
l = nib.load('../../src/fsLR-5k.L.indices.shape.gii').agg_data()
r = nib.load('../../src/fsLR-5k.R.indices.shape.gii').agg_data()
mask = np.concatenate((l,r)).astype(int)
ts_all = np.zeros((len(sub_list), 295, len(mask)))
for sub in range(len(sub_list)):
  cor_l = nib.load('../../micapipe_v0.2.0/sub-'+sub_list[sub]+'/ses-open/func/desc-se_task-rest_bold/surf/sub-'+sub_list[sub]+'_ses-open_hemi-L_surf-fsLR-5k.func.gii').agg_data()
  cor_r = nib.load('../../micapipe_v0.2.0/sub-'+sub_list[sub]+'/ses-open/func/desc-se_task-rest_bold/surf/sub-'+sub_list[sub]+'_ses-open_hemi-R_surf-fsLR-5k.func.gii').agg_data()
  cor = np.concatenate((cor_l,cor_r),axis=1)
  ts_all[sub] = cor
  print('loading......' + sub_list[sub])

ts_mean = np.nanmean(ts_all, axis=0)
ts_mean = ts_mean[:,mask!=0]
corr = np.corrcoef(ts_mean.T).round(6)    
fc = np.arctanh(corr)
fc[fc==np.inf] = 1
np.savetxt('/data/p_02801/ds004513_fc_mean_fsLR-5k.txt', fc)
print('saved mean FC matrix at fsLR space as /data/p_02801/ds004513_fc_mean_fsLR-5k.txt')
print(fc.shape)

### Gradients
for i in range(len(thre)):
  gm = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=2000, random_state=0)
  gm.fit(fc, sparsity=thre[i])
  np.savetxt('../../results/grad/vertex/group_grad_sparsity_'+str(thre[i])+'_lambdas_fsLR-5k.txt', gm.lambdas_)
  np.savetxt('../../results/grad/vertex/group_grad_sparsity_'+str(thre[i])+'_fsLR-5k.txt', gm.gradients_)
  print('finish...fsLR...gradients...sparsity_'+str(thre[i]))

# Gradients asymmetry
for i in range(len(thre)):
  # 2 patterns
  ## lh
  gm_ref = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=2000, random_state=0)
  gm_ref.fit(fc[:4428], sparsity=thre[i])
  np.savetxt('../../results/grad/vertex/group_grad_sparsity_'+str(thre[i])+'_lambdas_fsLR-5k_lh.txt', gm_ref.lambdas_)
  np.savetxt('../../results/grad/vertex/group_grad_sparsity_'+str(thre[i])+'_fsLR-5k_lh.txt', gm_ref.gradients_)
  ## rh
  gm = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=2000, random_state=0, alignment='procrustes')
  gm.fit(fc[4428:], reference=gm_ref.gradients_, sparsity=thre[i])
  np.savetxt('../../results/grad/vertex/group_grad_sparsity_'+str(thre[i])+'_fsLR-5k_rh_raw.txt', gm.gradients_)
  np.savetxt('../../results/grad/vertex/group_grad_sparsity_'+str(thre[i])+'_fsLR-5k_rh_aligned.txt', gm.aligned_)
  # 4 patterns
  ## lhlh
  gm_ref = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=2000, random_state=0)
  gm_ref.fit(fc[:4428,:4428], sparsity=thre[i])
  np.savetxt('../../results/grad/vertex/group_grad_sparsity_'+str(thre[i])+'_lambdas_fsLR-5k_lhlh.txt', gm_ref.lambdas_)
  np.savetxt('../../results/grad/vertex/group_grad_sparsity_'+str(thre[i])+'_fsLR-5k_lhlh.txt', gm_ref.gradients_)
  ## rhrh
  gm = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=2000, random_state=0, alignment='procrustes')
  gm.fit(fc[4428:,4428:], reference= gm_ref.gradients_, sparsity=thre[i])
  np.savetxt('../../results/grad/vertex/group_grad_sparsity_'+str(thre[i])+'_fsLR-5k_rhrh_raw.txt', gm.gradients_)
  np.savetxt('../../results/grad/vertex/group_grad_sparsity_'+str(thre[i])+'_fsLR-5k_rhrh_aligned.txt', gm.aligned_)
  ## lhrh
  gm = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=2000, random_state=0, alignment='procrustes')
  gm.fit(fc[:4428,4428:], reference= gm_ref.gradients_, sparsity=thre[i])
  np.savetxt('../../results/grad/vertex/group_grad_sparsity_'+str(thre[i])+'_fsLR-5k_lhrh_raw.txt', gm.gradients_)
  np.savetxt('../../results/grad/vertex/group_grad_sparsity_'+str(thre[i])+'_fsLR-5k_lhrh_aligned.txt', gm.aligned_)
  ## rhlh
  gm = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=2000, random_state=0, alignment='procrustes')
  gm.fit(fc[4428:,:4428], reference= gm_ref.gradients_, sparsity=thre[i])
  np.savetxt('../../results/grad/vertex/group_grad_sparsity_'+str(thre[i])+'_fsLR-5k_rhlh_raw.txt', gm.gradients_)
  np.savetxt('../../results/grad/vertex/group_grad_sparsity_'+str(thre[i])+'_fsLR-5k_rhlh_aligned.txt', gm.aligned_)
  print('finish...gradients...asymmetry...sparsity_'+str(thre[i]))
