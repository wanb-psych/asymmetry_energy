# /bin/bash ~

wd=/data/pt_02801/OpenNeuro/ds004513/derivatives/energetic-costs/

for i in `cat ../../sub_list.txt`; do

FSL convert_xfm -omat ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_from-func_to-MNI152NLin6ASym_res-3mm_xfm.mat \
	        -concat ${wd}/sub-${i}/ses-open/anat/sub-${i}_ses-open_from-T1w_to-MNI152NLin6ASym_res-3mm_xfm.mat \
	        ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_from-func_to-T1w_xfm.mat
		   
FSL flirt -in ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_task-rest_desc-preproc_bold.nii.gz \
          -ref /data/pt_02657/projects/energy/src/tpl-MNI152NLin6Asym_res-3mm_desc-brain_T1w.nii.gz \
          -applyxfm -init ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_from-func_to-MNI152NLin6ASym_res-3mm_xfm.mat \
          -out ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_func_MNI152NLin6ASym_res-3mm.nii.gz		   
 
wb_command -volume-to-surface-mapping ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_func_MNI152NLin6ASym_res-3mm.nii.gz \
	   ../../src/fs_LR.32k.L.midthickness.surf.gii \
	   ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_fsLR_32k.L.shape.gii -trilinear;
wb_command -volume-to-surface-mapping ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_func_MNI152NLin6ASym_res-3mm.nii.gz \
           ../../src/fs_LR.32k.R.midthickness.surf.gii \
           ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_fsLR_32k.R.shape.gii -trilinear;
wb_command -cifti-create-dense-scalar ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_fsLR_64k.dscalar.nii \
           -left-metric ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_fsLR_32k.L.shape.gii  \
           -right-metric ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_fsLR_32k.R.shape.gii;

wb_command -metric-resample \
  ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_fsLR_32k.R.shape.gii \
  ../../src/fsLR-32k.R.sphere.surf.gii \
  ../../src/fsLR-5k.R.sphere.surf.gii \
  ADAP_BARY_AREA \
  ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_fsLR-5k.R.func.gii \
  -area-surfs ../../src/fsLR-32k.R.inflated.surf.gii \
  ../../src/fsLR-5k.R.inflated.surf.gii
wb_command -metric-resample \
  ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_fsLR_32k.L.shape.gii \
  ../../src/fsLR-32k.L.sphere.surf.gii \
  ../../src/fsLR-5k.L.sphere.surf.gii \
  ADAP_BARY_AREA \
  ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_fsLR-5k.L.func.gii \
  -area-surfs ../../src/fsLR-32k.L.inflated.surf.gii \
  ../../src/fsLR-5k.L.inflated.surf.gii
wb_command -cifti-create-dense-scalar ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_fsLR-5k.dscalar.nii \
           -left-metric ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_fsLR-5k.L.func.gii \
           -right-metric ${wd}/sub-${i}/ses-open/func/sub-${i}_ses-open_fsLR-5k.R.func.gii
  
echo sub-${i}
done 
