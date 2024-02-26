# /bin/bash ~

for i in `cat ../../sub_list.txt`;
do mkdir -p ../../pet/sub-${i}; mkdir -p ../../pet/sub-${i}/ses-open ;

wb_command -volume-to-surface-mapping ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_task-rest_space-MNI152NLin6ASym_res-3mm_desc-CMRglc_pet.nii.gz ../../src/fs_LR.32k.L.midthickness.surf.gii ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_fsLR_32k.L.shape.gii -trilinear;
wb_command -volume-to-surface-mapping ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_task-rest_space-MNI152NLin6ASym_res-3mm_desc-CMRglc_pet.nii.gz ../../src/fs_LR.32k.R.midthickness.surf.gii ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_fsLR_32k.R.shape.gii -trilinear;
wb_command -cifti-create-dense-scalar ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_fsLR_64k.dscalar.nii -left-metric ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_fsLR_32k.L.shape.gii  -right-metric ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_fsLR_32k.R.shape.gii;

wb_command -metric-resample \
  ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_fsLR_32k.R.shape.gii \
  ../../src/fsLR-32k.R.sphere.surf.gii \
  ../../src/fsLR-5k.R.sphere.surf.gii \
  ADAP_BARY_AREA \
  ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_fsLR-5k.R.func.gii \
  -area-surfs ../../src/fsLR-32k.R.inflated.surf.gii \
  ../../src/fsLR-5k.R.inflated.surf.gii
wb_command -metric-resample \
  ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_fsLR_32k.L.shape.gii \
  ../../src/fsLR-32k.L.sphere.surf.gii \
  ../../src/fsLR-5k.L.sphere.surf.gii \
  ADAP_BARY_AREA \
  ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_fsLR-5k.L.func.gii \
  -area-surfs ../../src/fsLR-32k.L.inflated.surf.gii \
  ../../src/fsLR-5k.L.inflated.surf.gii
wb_command -cifti-create-dense-scalar ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_fsLR-5k.dscalar.nii -left-metric ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_fsLR-5k.L.func.gii  -right-metric ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_fsLR-5k.R.func.gii

wb_command -metric-resample \
  ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_fsLR_32k.R.shape.gii \
  ../../src/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii \
  ../../src/fsaverage5_std_sphere.R.10k_fsavg_R.surf.gii \
  ADAP_BARY_AREA \
  ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_10k_fsaverage5.R.func.gii \
  -area-metrics ../../src/fs_LR.R.midthickness_va_avg.32k_fs_LR.shape.gii \
  ../../src/fsaverage5.R.midthickness_va_avg.10k_fsavg_R.shape.gii

wb_command -metric-resample \
  ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_fsLR_32k.L.shape.gii \
  ../../src/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii \
  ../../src/fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii \
  ADAP_BARY_AREA \
  ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_10k_fsaverage5.L.func.gii \
  -area-metrics ../../src/fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii \
  ../../src/fsaverage5.L.midthickness_va_avg.10k_fsavg_L.shape.gii

wb_command -cifti-create-dense-scalar ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_fsaverage5_20k.dscalar.nii -left-metric ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_10k_fsaverage5.L.func.gii  -right-metric ../../pet/sub-${i}/ses-open/sub-${i}_ses-open_CMRglc_pet_10k_fsaverage5.R.func.gii
  
echo sub-${i}
done 
