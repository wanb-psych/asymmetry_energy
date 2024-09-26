from brainsmash.mapgen.base import Base
import numpy as np

brain_map_file = '../../results/glucose/mean_fsLR_mmp.txt'
grad_file = '../../results/grad/group_grad_sparsity_0.9_fsLR_mmp.txt'
dist_mat_file_L = '../../src/LeftParcelGeodesicDistmat.txt'
dist_mat_file_R = '../../src/RightParcelGeodesicDistmat.txt'

# generate 10000 permutations for CMRglc map
base_L = Base(x=np.loadtxt(brain_map_file)[:180], D=dist_mat_file_L, kernel='gaussian', pv=25, nh=30, seed=0)
base_R = Base(x=np.loadtxt(brain_map_file)[180:], D=dist_mat_file_R, kernel='gaussian', pv=25, nh=30, seed=0)
surrogates_L = base_L(n=1000)
surrogates_R = base_R(n=1000)
surrogates = np.concatenate((surrogates_L, surrogates_R),axis=1)
np.savetxt('../../results/models/CMRglc_mean_variogram_fsLR-mmp.txt', surrogates)

# for gradient
surrogates=np.zeros((100,1000,360))
for i in range(100):
  base_L = Base(x=np.loadtxt(grad_file)[:180,i], D=dist_mat_file_L, kernel='gaussian', pv=25, nh=30, seed=0)
  base_R = Base(x=np.loadtxt(grad_file)[180:,i], D=dist_mat_file_R, kernel='gaussian', pv=25, nh=30, seed=0)
  surrogates_L = base_L(n=1000)
  surrogates_R = base_R(n=1000)
  surrogates[i] = np.concatenate((surrogates_L, surrogates_R),axis=1)
  print('finish...gradient...'+str(i+1))
np.save('../../results/models/grad_group_variogram_fsLR-mmp', surrogates)

# normal permutation
surrogates=np.zeros((1000,360))
for j in range(1000):
  np.random.seed(j)
  rank=np.random.permutation(180)
  surrogates[j,:180]=np.loadtxt(brain_map_file)[:180][rank]
  surrogates[j,180:]=np.loadtxt(brain_map_file)[180:][rank]
np.savetxt('../../results/models/CMRglc_mean_permute_fsLR-mmp.txt', surrogates)

surrogates=np.zeros((100,1000,360))
for i in range(100):
  for j in range(1000):
    np.random.seed(j)
    rank=np.random.permutation(180)
    surrogates[i,j,:180]=np.loadtxt(grad_file)[:180,i][rank]
    surrogates[i,j,180:]=np.loadtxt(grad_file)[180:,i][rank]
  print('finish...gradient...'+str(i+1))
np.save('../../results/models/grad_group_permute_fsLR-mmp', surrogates)
