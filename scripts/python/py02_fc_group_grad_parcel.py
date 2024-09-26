import nibabel as nib
import numpy as np
from brainspace.gradient import GradientMaps
import sys
import os

# add first flag fas5 or fsLR
space = sys.argv[1]
# add second flag LH or RH
alignHemi = sys.argv[2]

if space=='fsa5':
  atlas = np.loadtxt('../../src/fsaverage5.LR.mmp.txt')
elif space=='fsLR':
  atlas = np.loadtxt('../../src/fs_LR.64k.mmp_360.txt')

sub_list=np.loadtxt('../../sub_list.txt',dtype=str)
data = [None] * len(sub_list)
for sub in range(len(sub_list)):
  if space=='fsa5':
    cor_l = nib.load('../../micapipe_v0.2.0/sub-'+sub_list[sub]+'/ses-open/func/desc-se_task-rest_bold/surf/sub-'+sub_list[sub]+'_ses-open_hemi-L_surf-fsaverage5.func.gii').agg_data()
    cor_r = nib.load('../../micapipe_v0.2.0/sub-'+sub_list[sub]+'/ses-open/func/desc-se_task-rest_bold/surf/sub-'+sub_list[sub]+'_ses-open_hemi-R_surf-fsaverage5.func.gii').agg_data()
  elif space=='fsLR':
    cor_l = nib.load('../../micapipe_v0.2.0/sub-'+sub_list[sub]+'/ses-open/func/desc-se_task-rest_bold/surf/sub-'+sub_list[sub]+'_ses-open_hemi-L_surf-fsLR-32k.func.gii').agg_data()
    cor_r = nib.load('../../micapipe_v0.2.0/sub-'+sub_list[sub]+'/ses-open/func/desc-se_task-rest_bold/surf/sub-'+sub_list[sub]+'_ses-open_hemi-R_surf-fsLR-32k.func.gii').agg_data()
  cor = np.concatenate((cor_l,cor_r),axis=1)
  tmp = np.zeros((cor.shape[0],360))
  for i in range(cor.shape[0]):
    for j in range(360):
      tmp[i,j]=np.nanmean(cor[i][atlas==j+1])
  corr = np.corrcoef(tmp.T).round(6)    
  fc = np.arctanh(corr)
  fc[fc==np.inf]=1
  data[sub] = fc
  np.savetxt('../../results/func/individual/fc_'+sub_list[sub]+'_'+space+'_mmp.txt', fc)
  print(sub_list[sub])

mean = np.array(data).mean(axis=0)
np.savetxt('../../results/func/fc_mean_'+space+'_mmp.txt', mean)

# Gradients
thre = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in range(10):
  gm_group = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0)
  gm_group.fit(mean, sparsity=thre[i])
  np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_lambdas_'+space+'_mmp.txt', gm_group.lambdas_)
  np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp.txt', gm_group.gradients_)
  print('finish...mmp...gradients...sparsity_'+str(thre[i]))

asymmetry='asymmetryRef' + alignHemi

# Gradients asymmetry 
for i in range(3):
  # 2 patterns
  if alignHemi == 'LH':
    gm_ref = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0)
    gm_ref.fit(mean[:180], sparsity=thre[i])
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_lambdas_'+space+'_mmp_lh.txt', gm_ref.lambdas_)
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lh.txt', gm_ref.gradients_)
    ## rh
    gm = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm.fit(mean[180:], reference=gm_ref.gradients_, sparsity=thre[i])
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rh_raw.txt', gm.gradients_)
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rh_aligned.txt', gm.aligned_) 
    # 4 patterns
    ## lhlh
    gm_ref = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0)
    gm_ref.fit(mean[:180,:180], sparsity=thre[i])
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_lambdas_'+space+'_mmp_lhlh.txt', gm_ref.lambdas_)
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lhlh.txt', gm_ref.gradients_)
    ## rhrh
    gm = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm.fit(mean[180:,180:], reference= gm_ref.gradients_, sparsity=thre[i])
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rhrh_raw.txt', gm.gradients_)
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rhrh_aligned.txt', gm.aligned_)
    ## lhrh
    gm = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm.fit(mean[:180,180:], reference= gm_ref.gradients_, sparsity=thre[i])
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lhrh_raw.txt', gm.gradients_)
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lhrh_aligned.txt', gm.aligned_)
    ## rhlh
    gm = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm.fit(mean[180:,:180], reference= gm_ref.gradients_, sparsity=thre[i])
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rhlh_raw.txt', gm.gradients_)
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rhlh_aligned.txt', gm.aligned_)
    print('finish...mmp...gradients...' + asymmetry + '...sparsity_'+str(thre[i]))

  elif  alignHemi == 'RH':
    gm_ref = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0)
    gm_ref.fit(mean[180:], sparsity=thre[i])
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_lambdas_'+space+'_mmp_rh.txt', gm_ref.lambdas_)
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rh.txt', gm_ref.gradients_)
    ## lh
    gm = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm.fit(mean[:180], reference=gm_ref.gradients_, sparsity=thre[i])
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lh_raw.txt', gm.gradients_)
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lh_aligned.txt', gm.aligned_)
    # 4 patterns
    ## rhrh
    gm_ref = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0)
    gm_ref.fit(mean[180:,180:], sparsity=thre[i])
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_lambdas_'+space+'_mmp_rhrh.txt', gm_ref.lambdas_)
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rhrh.txt', gm_ref.gradients_)
    ## lhlh
    gm = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm.fit(mean[:180,:180], reference= gm_ref.gradients_, sparsity=thre[i])
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lhlh_raw.txt', gm.gradients_)
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lhlh_aligned.txt', gm.aligned_)
    ## lhrh
    gm = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm.fit(mean[:180,180:], reference= gm_ref.gradients_, sparsity=thre[i])
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lhrh_raw.txt', gm.gradients_)
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lhrh_aligned.txt', gm.aligned_)
    ## rhlh
    gm = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm.fit(mean[180:,:180], reference= gm_ref.gradients_, sparsity=thre[i])
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rhlh_raw.txt', gm.gradients_)
    np.savetxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rhlh_aligned.txt', gm.aligned_)
    print('finish...mmp...gradients...' + asymmetry + '...sparsity_'+str(thre[i])) 

## individual gradients
for sub in range(len(sub_list)):
  fc = np.loadtxt('../../results/func/individual/fc_'+sub_list[sub]+'_'+space+'_mmp.txt')
  try:
    os.makedirs('../../results/grad/individual/'+sub_list[sub])
    os.makedirs('../../results/grad/' + asymmetry + '/'+sub_list[sub])
  except:
    pass  
  for i in range(10):
    # whole brain
    gm_group=np.loadtxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp.txt')
    gm = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm.fit(fc, reference= gm_group, sparsity=thre[i])
    np.savetxt('../../results/grad/individual/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_raw.txt', gm.gradients_)
    np.savetxt('../../results/grad/individual/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_aligned.txt', gm.aligned_)
    np.savetxt('../../results/grad/individual/'+sub_list[sub]+'/lambdas_sparsity_'+str(thre[i])+'_'+space+'_mmp.txt', gm.lambdas_)
    # asymmetry 2 patterns
    if alignHemi == 'LH':
      gm_ref=np.loadtxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lh.txt')
    elif alignHemi == 'RH':
      gm_ref=np.loadtxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rh.txt')  
    ## lh  
    gm = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm.fit(fc[:180], reference= gm_ref, sparsity=thre[i])
    np.savetxt('../../results/grad/' + asymmetry + '/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lh_raw.txt', gm.gradients_)
    np.savetxt('../../results/grad/' + asymmetry + '/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lh_aligned.txt', gm.aligned_)
    ## rh
    gm = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm.fit(fc[180:], reference= gm_ref, sparsity=thre[i])
    np.savetxt('../../results/grad/' + asymmetry + '/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rh_raw.txt', gm.gradients_)
    np.savetxt('../../results/grad/' + asymmetry + '/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rh_aligned.txt', gm.aligned_)
    ## rh aligned to self lh
    gm_self = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm_self.fit(fc[180:], reference= gm.gradients_, sparsity=thre[i])
    np.savetxt('../../results/grad/' + asymmetry + '/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rh_alignedSelf.txt', gm_self.aligned_)
    # asymmetry 4 patterns
    gm_ref=np.loadtxt('../../results/grad/group_grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lhlh.txt')
    ## lhlh
    gm = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm.fit(fc[:180,:180], reference= gm_ref, sparsity=thre[i])
    np.savetxt('../../results/grad/' + asymmetry + '/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lhlh_raw.txt', gm.gradients_)
    np.savetxt('../../results/grad/' + asymmetry + '/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lhlh_aligned.txt', gm.aligned_)
    ## rhrh
    gm_tmp = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm_tmp.fit(fc[180:,180:], reference= gm_ref, sparsity=thre[i])
    np.savetxt('../../results/grad/' + asymmetry + '/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rhrh_raw.txt', gm_tmp.gradients_)
    np.savetxt('../../results/grad/' + asymmetry + '/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rhrh_aligned.txt', gm_tmp.aligned_)
    ## lhrh
    gm_tmp = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm_tmp.fit(fc[:180,180:], reference= gm_ref, sparsity=thre[i])
    np.savetxt('../../results/grad/' + asymmetry + '/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lhrh_raw.txt', gm_tmp.gradients_)
    np.savetxt('../../results/grad/' + asymmetry + '/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lhrh_aligned.txt', gm_tmp.aligned_)
    ## rhlh
    gm_tmp = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm_tmp.fit(fc[180:,:180], reference= gm_ref, sparsity=thre[i])
    np.savetxt('../../results/grad/' + asymmetry + '/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rhlh_raw.txt', gm_tmp.gradients_)
    np.savetxt('../../results/grad/' + asymmetry + '/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rhlh_aligned.txt', gm_tmp.aligned_)
    print('finish..'+sub_list[sub]+'...mmp...gradients...' + asymmetry + '...sparsity_'+str(thre[i]))

    ## rhrh aligned to self
    gm_self = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm_self.fit(fc[180:,180:], reference= gm.gradients_, sparsity=thre[i])
    np.savetxt('../../results/grad/' + asymmetry + '/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rhrh_alignedSelf.txt', gm.aligned_)
    ## lhrh aligned to self
    gm_self = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm_self.fit(fc[:180,180:], reference= gm.gradients_, sparsity=thre[i])
    np.savetxt('../../results/grad/' + asymmetry + '/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_lhrh_alignedSelf.txt', gm.aligned_)
    ## rhlh aligned to self
    gm_self = GradientMaps(kernel='normalized_angle', approach='dm', n_components=100, random_state=0, alignment='procrustes')
    gm_self.fit(fc[180:,:180], reference= gm.gradients_, sparsity=thre[i])
    np.savetxt('../../results/grad/' + asymmetry + '/'+sub_list[sub]+'/grad_sparsity_'+str(thre[i])+'_'+space+'_mmp_rhlh_alignedSelf.txt', gm.aligned_)
