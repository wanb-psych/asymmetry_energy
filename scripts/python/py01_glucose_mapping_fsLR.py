import nibabel as nib
import numpy as np
from brainspace.utils.parcellation import map_to_labels
import func_plot as fp

sub_list=np.loadtxt('../../sub_list.txt',dtype=str)

data = [None] * len(sub_list)
for i in range(len(sub_list)):
  data[i] = nib.load('../../pet/sub-'+sub_list[i]+'/ses-open/sub-'+sub_list[i]+'_ses-open_CMRglc_pet_fsLR_64k.dscalar.nii').get_fdata()[0]

mean = np.array(data).mean(axis=0)
np.savetxt('../../results/glucose/mean_fsLR.txt', mean)

mmp_360 = np.loadtxt('../../src/fs_LR.64k.mmp_360.txt')
data_mmp = np.zeros(360)
for i in range(360):
  data_mmp[i] = np.nanmean(mean[mmp_360==i+1])
np.savetxt('../../results/glucose/mean_fsLR_mmp.txt', data_mmp)
print('mean finished')


for sub in range(len(sub_list)):
  indi_mmp = np.zeros(360)
  for i in range(360):
    indi_mmp[i] = np.nanmean(data[sub][mmp_360==i+1])
  np.savetxt('../../results/glucose/individual/'+sub_list[sub]+'_fsLR_mmp.txt', indi_mmp)
  print(sub_list[sub]+'...finished')

data = [None] * len(sub_list)
for i in range(len(sub_list)):
  data[i] = nib.load('../../pet/sub-'+sub_list[i]+'/ses-open/sub-'+sub_list[i]+'_ses-open_CMRglc_pet_fsLR-5k.dscalar.nii').get_fdata()[0]
  np.savetxt('../../results/glucose/individual/'+sub_list[sub]+'_fsLR-5k.txt', data[i])
mean = np.array(data).mean(axis=0)
np.savetxt('../../results/glucose/mean_fsLR-5k.txt', mean)

print('finish fsLR-5k')