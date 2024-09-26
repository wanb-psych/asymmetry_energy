import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import func_plot as fp
import scipy.stats as ss
from statsmodels.regression.linear_model import OLS

# LH alinged to RH
mmp_fsLR = np.loadtxt('../../src/fs_LR.64k.mmp_360.txt')
ratio = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
spa = [0,1,2,99]

## load gradient data
max_mode = 100
lhlh = [np.loadtxt('../../results/grad/group_grad_sparsity_'+str(ratio[i])+'_fsLR_mmp_lh_aligned.txt') for i in range(10)]
rhrh = [np.loadtxt('../../results/grad/group_grad_sparsity_'+str(ratio[i])+'_fsLR_mmp_rh.txt') for i in range(10)]
grad_asy = np.array(lhlh) - np.array(rhrh)
grad_asy_plot = np.concatenate((grad_asy, np.zeros((10,180,100))),axis=1)
plot = [None] * 4
for i in range(4):
  plot[i] = mmp_fsLR.copy()
  for node in range(360):
    plot[i][mmp_fsLR==node+1] = grad_asy_plot[9][:,spa[i]][node]
  plot[i][plot[i]==0] = np.nan

covmatrix=np.corrcoef(grad_asy[9].T)
covmatrix[np.logical_and(covmatrix<0.3,covmatrix>-0.3)]=0
sns.set_context("paper", font_scale = 2.5)
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(covmatrix,cmap='RdBu_r', vmax=1,vmin=-1)
ax.set_xticks([0,19,39,59,79,99])
ax.set_xticklabels(['1','20','40','60','80','100'])
ax.set_yticks([0,19,39,59,79,99])
ax.set_yticklabels(['1','20','40','60','80','100'])
fig.tight_layout()
fig.savefig('../../figures/alignment_asy_gradient_corr_0.9_fsLR_mmp.png', transparent=True, dpi=300)

### notice: some gradients flipped like G1 and G2, but doesn't influence modeling 
fp.plot_surface(data = plot, surf='fsLR', color_range=(-0.05,0.05),
                size = (1200, 800), 
                cmap = 'RdBu_r', filename = '../../figures/alignment_asy_grad_fsLR_mmp_sparsity_'+str(ratio[9])+'.png',
                display=False)

## load CMRglc data
glucose_raw = np.loadtxt('../../results/glucose/mean_mmp.txt')
glucose_l = ss.zscore(glucose_raw[:180])
glucose_r = ss.zscore(glucose_raw[180:])
glucose = np.concatenate((glucose_l, glucose_r))
glucose_asy = glucose_l - glucose_r

## trained LH to predict RH
r_ll = np.zeros((10,max_mode))
r_lr = np.zeros((10,max_mode))
r_ll_pred = np.zeros((10,max_mode,180))
r_lr_pred = np.zeros((10,max_mode,180))
for i in range(10):
  for j in range(max_mode):
    glm = OLS(glucose[:180], lhlh[i][:,:j+1]).fit()
    r_ll_pred[i,j] = glm.predict(lhlh[i][:180,:j+1])
    r_ll[i,j]=np.corrcoef(glucose[:180], r_ll_pred[i,j])[0,1]
    r_lr_pred[i,j] = glm.predict(rhrh[i][:,:j+1])
    r_lr[i,j]=np.corrcoef(glucose[180:], r_lr_pred[i,j])[0,1]
np.savetxt('../../results/models/alignRefRH_trainLHtoLH_fsLR-mmp.txt', r_ll)
np.savetxt('../../results/models/alignRefRH_trainLHtoRH_fsLR-mmp.txt', r_lr)

## trained RH to predict LH
r_rl = np.zeros((10,max_mode))
r_rr = np.zeros((10,max_mode))
r_rl_pred = np.zeros((10,max_mode,180))
r_rr_pred = np.zeros((10,max_mode,180))
for i in range(10):
  for j in range(max_mode):
    glm = OLS(glucose[180:], rhrh[i][:,:j+1]).fit()
    r_rr_pred[i,j] = glm.predict(rhrh[i][:,:j+1])
    r_rr[i,j]=np.corrcoef(glucose[180:], r_rr_pred[i,j])[0,1]
    r_rl_pred[i,j] = glm.predict(lhlh[i][:,:j+1])
    r_rl[i,j]=np.corrcoef(glucose[:180], r_rl_pred[i,j])[0,1]
np.savetxt('../../results/models/alignRefRH_trainRHtoLH_fsLR-mmp.txt', r_rl)
np.savetxt('../../results/models/alignRefRH_trainRHtoRH_fsLR-mmp.txt', r_rr)

## plot cross-hemispheric models
sns.set_context("paper", font_scale = 2.5)
fig, ax = plt.subplots(figsize=(10,8))
i=9
ax.plot(range(1,max_mode+1), r_ll[i], label='Training LH', c=fp.red, lw=3)
ax.plot(range(1,max_mode+1), r_lr[i], label='Fitting RH', c=fp.red, ls='--', lw=3)
ax.plot(range(1,max_mode+1), r_rr[i], label='Training RH', c=fp.blue, lw=3)
ax.plot(range(1,max_mode+1), r_rl[i], label='Fitting LH', c=fp.blue, ls='--', lw=3)
ax.set_xlabel('Steps')
ax.set_ylabel('Pearson $\it{r}$')
plt.legend(fontsize=18)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()
fig.savefig('../../figures/alignment_fitting_sparsity_0.'+str(i)+'_fsLR_mmp.png', transparent=True, dpi=300)

sns.set_context("paper", font_scale = 2.5)
fig, ax = plt.subplots(figsize=(6,6))
ax.plot([0,1], [0,1], c='gray', ls='--', lw=2)
ax.plot(r_ll[i], r_lr[i], c=fp.red, label='LH -> RH', lw=3)
ax.plot(r_rr[i], r_rl[i], c=fp.blue, label='RH -> LH', lw=3)
ax.set_xlabel('Training')
ax.set_ylabel('Fitting')
ax.set_xlim(0.3,0.9)
ax.set_ylim(0.3,0.9)
ax.set_xticks([0.3, 0.6, 0.9])
ax.set_yticks([0.3, 0.6, 0.9])
ax.text(0.05,0.6,'Overfitting', transform=ax.transAxes)
ax.text(0.5,0.3,'Underfitting', transform=ax.transAxes)
plt.legend(loc = 'lower right', fontsize=18)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()
fig.savefig('../../figures/alignment_fitting_diag_sparsity_0.'+str(i)+'_fsLR_mmp.png', transparent=True, dpi=300)

## Asymmetry
### model asymmetry pearson r
r_asy = np.zeros((10,max_mode))
corr = np.zeros((10,max_mode))
corr_pred = np.zeros((10,max_mode,180))
for i in range(10):
  for j in range(max_mode):
    glm = OLS(glucose_asy, grad_asy[i][:,:j+1]).fit()
    corr[i,j]=glm.rsquared_adj
    corr_pred[i,j] = glm.predict(grad_asy[i][:,:j+1])
    r_asy[i,j]=np.corrcoef(glucose_asy, corr_pred[i,j])[0,1]
np.savetxt('../../results/models/alignRefRH_AsyFittingR_fsLR-mmp.txt', r_asy)
np.savetxt('../../results/models/alignRefRH_AsyFittingAdjR2_fsLR-mmp.txt', corr)

### standard regression beta, and plot for step 100, sparsity=0.9
std = grad_asy[9][:,:j+1].std(0)
tt = glm.t_test(np.diag(std))

sns.set_context("paper", font_scale = 2.5)
fig, ax = plt.subplots(figsize=(10,6))
for i in range(100):
    ax.plot([i+1,i+1],[tt.summary_frame()['Conf. Int. Low'][i],tt.summary_frame()['Conf. Int. Upp.'][i]], color='gray',alpha=0.5)
ax.axhline(y=0,c='gray',ls='--')
ax.scatter(range(1,101), list(tt.summary_frame().coef),cmap='Spectral',c=range(1,101),zorder=101)
ax.set_yticks([-0.2,0,0.2])
ax.set_ylabel('Std. Coef.')
ax.set_xlabel('Number of gradients')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()
fig.savefig('../../figures/alignment_fitting_asy_sparsity_0.9_beta_fsLR_mmp.png', transparent=True, dpi=300)

### plot r and adjusted r2 for sparsity=0.9
sns.set_context("paper", font_scale = 2.5)
fig, ax = plt.subplots(figsize=(10,6))
i=9
ax.plot(range(1,max_mode+1), corr[i]*100, color='black',zorder=0, alpha=0.5,  lw=3)
ax.scatter(52, corr[i][51]*100, color='black', s=50)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Adj. R-squared %', color='black')
ax.set_yticks([-10,-5,0,5])
ax.text(48,8,'7.6%')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Steps')
fig.tight_layout()
fig.savefig('../../figures/alignment_fitting_asy_adjR2_sparsity_0.'+str(i)+'_fsLR_mmp.png', transparent=True, dpi=300)

### use elastic net and plot
alphas = np.array([[0.05,0.10],[0.15,0.20],[0.25,0.30]])
sns.set_context("paper", font_scale = 2.5)
fig, ax = plt.subplots(3,2,figsize=(10,6))
spa=9
for i in range(3):
 for j in range(2):
  glm = OLS(ss.zscore(glucose_asy), ss.zscore(grad_asy[spa][:,:101])).fit_regularized(alpha=alphas[i,j], L1_wt=0.5)
  y_pred = glm.predict(ss.zscore(grad_asy[spa][:,:101]))
  rp = ss.pearsonr(ss.zscore(glucose_asy), y_pred)
  ax[i,j].scatter(range(1,101), glm.params,cmap='Spectral',c=range(1,101))
  ax[i,0].set_ylabel('Slope')
  ax[i,j].spines['right'].set_visible(False)
  ax[i,j].spines['top'].set_visible(False)
  ax[i,j].spines['bottom'].set_visible(False)
  ax[i,j].set_xticks([])
  ax[i,j].set_yticks([-0.3,0,0.3])
  ax[i,j].text(0.4,0.9,'\u03B1 = '+format(alphas[i,j], '.2f'), fontsize=16, transform=ax[i,j].transAxes)
  ax[i,j].text(0.4,0.05,'$\it{r}$ = '+format(rp[0], '.3f'), fontsize=16, transform=ax[i,j].transAxes)
ax[i,0].spines['bottom'].set_visible(True)
ax[i,0].set_xticks([0,50,100])  
ax[i,0].set_xlabel('Number of gradients')
ax[i,1].spines['bottom'].set_visible(True)
ax[i,1].set_xticks([0,50,100])  
ax[i,1].set_xlabel('Number of gradients')
fig.tight_layout()
fig.savefig('../../figures/alignment_ElasticNet_asy_sparsity_0.'+str(spa)+'_fsLR_mmp.png', transparent=True, dpi=300)

## Competition and lateralization model comparisons
comp=np.zeros(100)
for i in range(100):
  asy = r_ll_pred[9][i] - r_rr_pred[9][i]
  comp[i] = np.cov(glucose_asy,asy)[0,1]
late = np.zeros(100)
for i in range(100):
  asy = corr_pred[9][i]
  late[i] = np.cov(glucose_asy, asy)[0,1]

sns.set_context("paper", font_scale = 2.5)
fig, ax = plt.subplots(figsize=(10,8))
i=9

ax.plot(range(1,101), comp, c='pink', zorder=1, alpha=0.5, lw=3, label='Competition')
ax.scatter(range(1,101), comp, cmap='pink_r', c=comp, s=10)
ax.plot(range(1,101), late, c='slategray', zorder=0, alpha=0.5, lw=3, label='Lateralization')
ax.scatter(range(1,101), late, cmap='bone_r', c=late, s=10)

ax.set_ylabel('Covariance')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Steps')
plt.legend(loc='lower right', fontsize=18)
fig.tight_layout()
fig.savefig('../../figures/alignment_fitting_asy_sparsity_0.'+str(i)+'_fsLR_mmp.png', transparent=True, dpi=300)
