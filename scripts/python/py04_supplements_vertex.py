import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import func_plot as fp
import scipy.stats as ss
from statsmodels.regression.linear_model import OLS
import nibabel as nib

# Vertex level
### here we calculated sparsity=0,0.5,0.9 for fsLR-5k level
l = nib.load('../../src/fsLR-5k.L.indices.shape.gii').agg_data()
r = nib.load('../../src/fsLR-5k.R.indices.shape.gii').agg_data()
mask_5k = np.concatenate((l,r)).astype(float)
ratio = [0,0.5,0.9]
n=len(ratio)
spa = [0,1,2,99]

## load gradients data
gradient = [ss.zscore(np.loadtxt('../../results/grad/vertex/group_grad_sparsity_'+str(ratio[i])+'_fsLR-5k.txt')) for i in range(n)]
lambdas = [np.loadtxt('../../results/grad/vertex/group_grad_sparsity_'+str(ratio[i])+'_lambdas_fsLR-5k.txt') for i in range(n)]
max_mode=100
print('load gradients data')

lambdas_acc = np.zeros((n,max_mode))
for i in range(max_mode):
  for j in range(n):
    lambdas_acc[j,i] = lambdas[j][:i+1].sum() * 100

### plot the FC variance explained by gradients
sns.set_context("paper", font_scale = 2.5)
fig, ax = plt.subplots(figsize=(8,6))
for i in range(n):
  plt.plot(range(1,max_mode+1),lambdas_acc[i], lw=3, 
           label='sparsity='+str(ratio[i]),
           color=np.array([1.3,1.3,1.3]) * i / 3)
plt.legend(fontsize=16,loc='lower right')
ax.set_xlabel('Number of gradients')
ax.set_ylabel('Variance % accumulated')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()
fig.savefig('../../figures/fsLR-5k_lambdas.png', transparent=True, dpi=300)
print('save fig fsLR-5k_lambdas.png')

### plot 4 gradients for sparsity=0.9
plot = [None] * 4
for i in range(4):
  plot[i] = mask_5k.copy()
  plot[i][mask_5k!=0] = gradient[n-1][:,spa[i]]
  plot[i][mask_5k==0] = np.nan

fp.plot_surface(data = plot, surf='fsLR-5k',
                size = (1200, 800), 
                cmap = 'viridis', filename = '../../figures/fsLR-5k_grad_sparsity_'+str(ratio[n-1])+'.png',
                display=False)
print('save fig grad_fsLR-5k_sparsity_'+str(ratio[n-1])+'.png')

## load CMRglc data and plot
glucose_raw = np.loadtxt('../../results/glucose/mean_fsLR-5k.txt')[mask_5k != 0]
glucose_l = ss.zscore(glucose_raw[:4428])
glucose_r = ss.zscore(glucose_raw[4428:])
glucose = ss.zscore(glucose_raw)
glucose_asy = glucose_l - glucose_r

plot = mask_5k.copy()
plot[mask_5k!=0] = glucose.copy()
plot[mask_5k==0] = np.nan
fp.plot_surface(data = plot, surf='fsLR-5k',
                size = (1200, 200), color_range=(-3,3),
                cmap = 'plasma', filename = '../../figures/fsLR-5k_CMRglc_mean.png',
                display=False)
print('save fig fsLR-5k_CMRglc_mean.png')

plot = mask_5k.copy()
plot[:4842][mask_5k[:4842]!=0] = glucose_asy.copy()
plot[mask_5k==0] = np.nan
plot[4842:][mask_5k[4842:]!=0] = np.nan
fp.plot_surface(data = plot, surf='fsLR-5k',
                size = (1200, 200), color_range=(-1,1),
                cmap = 'PiYG_r', filename = '../../figures/fsLR-5k_CMRglc_asymean.png',
                display=True)
print('save fig fsLR-5k_CMRglc_asymean.png')

## adjusted r2 CMRglc-gradient and plot
corr = np.zeros((n,max_mode))
for i in range(n):
  for j in range(max_mode):
    glm = OLS(glucose, gradient[i][:,:j+1]).fit()
    corr[i,j]=glm.rsquared_adj
np.savetxt('../../results/models/adjR2_fitting_fsLR-5k.txt', corr.T)
print('save adjR2_fitting_fsLR-5k.txt')

corr = np.loadtxt('../../results/models/adjR2_fitting_fsLR-5k.txt').T

sns.set_context("paper", font_scale = 2.5)
fig, ax = plt.subplots(figsize=(8,6))
for i in range(n):
  ax.plot(range(1,max_mode+1), corr[i]*100, lw=3, 
          label='sparsity=' + str(ratio[i]), color=np.array([1.3,1.3,1.3]) * i / 3)
  print('finalsparsity'+str(ratio[i])+'_r2 =', corr[i][max_mode-1].round(4))
ax.set_xlabel('Steps')
ax.set_ylabel('Adj. R-squared %')
#ax.set_xticks([0,20,40,60,80,100])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()
fig.savefig('../../figures/fsLR-5k_fitting.png', transparent=True, dpi=300)
print('save fsLR-5k_fitting.png')

## plot 5 predicted CMRglc
g=[5,10,20,50,100]
y_pred=[]
for i in range(5):
    glm = OLS(glucose, gradient[n-1][:,:g[i]]).fit()
    y_pred.append(glm.predict(gradient[n-1][:,:g[i]]))

plot = [None] * 5
for i in range(5):
  plot[i] = mask_5k.copy()
  plot[i][mask_5k==0] = np.nan
  plot[i][mask_5k!=0] = y_pred[i]
fp.plot_surface(data = plot, surf='fsLR-5k',
                size = (1200, 1000), color_range=(-3,3),
                cmap = 'plasma', filename = '../../figures/fsLR-5k_CMRglc_predicted.png',
                display=True)
print('save fsLR-5k_CMRglc_predicted.png')

## plot scatter
sns.set_context("paper", font_scale = 2.5)
fig, ax = plt.subplots(figsize=(6,6))
for i in range(3):
  ax.plot(lambdas_acc[i], corr[i]*100, lw=3, marker='o',
          color=np.array([1.3,1.3,1.3]) * i / 3)

ax.set_yticks([0,50])
ax.set_xticks([0,50])
ax.set_ylim(0,50)
ax.set_xlim(0,50)
ax.plot([0,50], [0,50], c='black', ls='-', lw=2)

ax.set_xlabel('Explanation to FC %')
ax.set_ylabel('Explanation to CMRglc %')

fig.tight_layout()
fig.savefig('../../figures/fsLR-5k_variance_r2_scatter.png', transparent=True, dpi=300) 
print('save fsLR-5k_variance_r2_scatter.png')

## model penalty and visualize 100
alphas = np.array([[0.05,0.10],[0.15,0.20],[0.25,0.3]])
sns.set_context("paper", font_scale = 2.5)
fig, ax = plt.subplots(3,2,figsize=(10,6))
for i in range(3):
 for j in range(2):
  glm = OLS(glucose, gradient[n-1][:,:100]).fit_regularized(alpha=alphas[i,j], L1_wt=0.5)
  y_pred = glm.predict(gradient[n-1][:,:100])
  rp = ss.pearsonr(glucose, y_pred)
  ax[i,j].scatter(range(1,101), glm.params,cmap='Spectral',c=range(1,101))
  ax[i,0].set_ylabel('Slope')
  ax[i,j].spines['right'].set_visible(False)
  ax[i,j].spines['top'].set_visible(False)
  ax[i,j].spines['bottom'].set_visible(False)
  ax[i,j].set_xticks([])
  ax[i,j].set_yticks([-0.5,0,0.5])
  ax[i,j].text(0.4,0.9,'\u03B1 = '+format(alphas[i,j], '.2f'), fontsize=16, transform=ax[i,j].transAxes)
  ax[i,j].text(0.4,0.05,'$\it{r}$ = '+format(rp[0], '.3f'), fontsize=16, transform=ax[i,j].transAxes)
ax[i,0].spines['bottom'].set_visible(True)
ax[i,0].set_xticks([0,50,100])  
ax[i,0].set_xlabel('Number of gradients')
ax[i,1].spines['bottom'].set_visible(True)
ax[i,1].set_xticks([0,50,100])  
ax[i,1].set_xlabel('Number of gradients')
fig.tight_layout()
fig.savefig('../../figures/fsLR-5k_ElasticNet_sparsity_0.9.png', transparent=True, dpi=300)
print('save fsLR-5k_ElasticNet_sparsity_0.9.png')


## load hemispheric data and plot
lhlh = [np.loadtxt('../../results/grad/vertex/group_grad_sparsity_'+str(ratio[i])+'_fsLR-5k_lh.txt') for i in range(n)]
rhrh = [np.loadtxt('../../results/grad/vertex/group_grad_sparsity_'+str(ratio[i])+'_fsLR-5k_rh_aligned.txt') for i in range(n)]
grad_asy = np.array(lhlh) - np.array(rhrh) 
grad_asy_plot = np.concatenate((grad_asy, np.zeros(grad_asy.shape)),axis=1)
plot = [None] * 4
for i in range(4):
  plot[i] = mask_5k.copy()
  plot[i][mask_5k!=0] = grad_asy_plot[n-1][:,spa[i]]
  plot[i][plot[i]==0] = np.nan

fp.plot_surface(data = plot, surf='fsLR-5k', color_range=(-0.05,0.05),
                size = (1200, 800), 
                cmap = 'RdBu_r', filename = '../../figures/fsLR-5k_asy_grad__sparsity_0.9.png',
                display=False)
print('save asy_grad_fsLR-5k_sparsity_0.9.png')


## train LH to predict RH
r_ll = np.zeros((n,max_mode))
r_lr = np.zeros((n,max_mode))
r_ll_pred = np.zeros((n,max_mode,4428))
r_lr_pred = np.zeros((n,max_mode,4428))
for i in range(n):
  for j in range(max_mode):
    glm = OLS(glucose[:4428], lhlh[i][:,:j+1]).fit()
    r_ll_pred[i,j] = glm.predict(lhlh[i][:4428,:j+1])
    r_ll[i,j]=np.corrcoef(glucose[:4428], r_ll_pred[i,j])[0,1]
    r_lr_pred[i,j] = glm.predict(rhrh[i][:,:j+1])
    r_lr[i,j]=np.corrcoef(glucose[4428:], r_lr_pred[i,j])[0,1]
np.savetxt('../../results/models/trainLHtoLH_fsLR-5k.txt', r_ll)
np.savetxt('../../results/models/trainLHtoRH_fsLR-5k.txt', r_lr)
print('save trainLHtoRH_fsLR-5k.txt')

## train RH to predict LH
r_rl = np.zeros((n,max_mode))
r_rr = np.zeros((n,max_mode))
r_rl_pred = np.zeros((n,max_mode,4428))
r_rr_pred = np.zeros((n,max_mode,4428))
for i in range(n):
  for j in range(max_mode):
    glm = OLS(glucose[4428:], rhrh[i][:,:j+1]).fit()
    r_rr_pred[i,j] = glm.predict(rhrh[i][:,:j+1])
    r_rr[i,j]=np.corrcoef(glucose[4428:], r_rr_pred[i,j])[0,1]
    r_rl_pred[i,j] = glm.predict(lhlh[i][:,:j+1])
    r_rl[i,j]=np.corrcoef(glucose[:4428], r_rl_pred[i,j])[0,1]
np.savetxt('../../results/models/trainRHtoLH_fsLR-5k.txt', r_rl)
np.savetxt('../../results/models/trainRHtoRH_fsLR-5k.txt', r_rr)
print('save trainRHtoRH_fsLR-5k.txt')

### plot sparsity=0.9 cross-hemispheric models
sns.set_context("paper", font_scale = 2.5)
fig, ax = plt.subplots(figsize=(10,6))
i=2
ax.plot(range(1,max_mode+1), r_ll[i], label='Training LH', c=fp.red, lw=3)
ax.plot(range(1,max_mode+1), r_lr[i], label='Fitting RH', c=fp.red, ls='--', lw=3)
ax.plot(range(1,max_mode+1), r_rr[i], label='Training RH', c=fp.blue, lw=3)
ax.plot(range(1,max_mode+1), r_rl[i], label='Fitting LH', c=fp.blue, ls='--', lw=3)
ax.set_xlabel('Steps')
ax.set_ylabel('Pearson $\it{r}$')
ax.set_xticks(g)
plt.legend(fontsize=18)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()
fig.savefig('../../figures/fsLR-5k_fitting_sparsity_'+str(ratio[i])+'.png', transparent=True, dpi=300)
print('save fsLR-5k_fitting_sparsity_'+str(ratio[i])+'.png')

sns.set_context("paper", font_scale = 2.5)
fig, ax = plt.subplots(figsize=(6,6))
i=2
ax.plot([0,1], [0,1], c='gray', ls='--', lw=2)
ax.plot(r_ll[i], r_lr[i], c=fp.red, label='LH -> RH', lw=3)
ax.plot(r_rr[i], r_rl[i], c=fp.blue, label='RH -> LH', lw=3)
ax.set_xlabel('Training')
ax.set_ylabel('Fitting')
ax.set_xlim(0.3,0.7)
ax.set_ylim(0.3,0.7)
ax.set_xticks([0.3, 0.5, 0.7])
ax.set_yticks([0.3, 0.5, 0.7])
ax.text(0.05,0.6,'Overfitting', transform=ax.transAxes)
ax.text(0.5,0.3,'Underfitting', transform=ax.transAxes)
plt.legend(loc = 'lower right', fontsize=18)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()
fig.savefig('../../figures/fsLR-5k_fitting_diag_sparsity_'+str(ratio[i])+'.png', transparent=True, dpi=300)
print('save fsLR-5k_fitting_diag_sparsity_'+str(ratio[i])+'.png')


## Asymmetry
### model asymmetry pearonr and adjusted r2
corr = np.zeros((n,max_mode))
r_asy = np.zeros((n,max_mode))
corr_pred = np.zeros((n,max_mode,4428))
for i in range(n):
  for j in range(max_mode):
    glm = OLS(glucose_asy, grad_asy[i][:,:j+1]).fit()
    corr[i,j]=glm.rsquared_adj
    corr_pred[i,j] = glm.predict(grad_asy[i][:,:j+1])
    r_asy[i,j]=np.corrcoef(glucose_asy, corr_pred[i,j])[0,1]
    print('finish model ' + str(j+1))
np.savetxt('../../results/models/AsyFittingR_fsLR-5k.txt', r_asy.T)   
np.savetxt('../../results/models/AsyFittingAdjR2_fsLR-5k.txt', corr.T)
print('save AsyFittingR_fsLR-5k.txt and AsyFittingAdjR2_fsLR-5k.txt')

### plot adjusted r2 for sparsity=0.9
sns.set_context("paper", font_scale = 2.5)
fig, ax = plt.subplots(figsize=(10,4))
select=2
ax.plot(range(1,max_mode+1), corr[i]*100, color='black',zorder=0, alpha=0.5,  lw=3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Adj. R-squared %', color='black')
ax.set_yticks([0,10])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Steps')
fig.tight_layout()
fig.savefig('../../figures/fitting_asy_adjR2_sparsity_'+str(ratio[select])+'_fsLR_mmp.png', transparent=True, dpi=300)
fig.savefig('../../figures/fsLR-5k_fitting_asy_sparsity_'+str(ratio[select])+'.png', transparent=True, dpi=300)
print('save fsLR-5k_fitting_asy_sparsity_'+str(ratio[select])+'.png')

### use elastic net and plot
alphas = np.array([[0.05,0.10],[0.15,0.20],[0.25,0.30]])
sns.set_context("paper", font_scale = 2.5)
fig, ax = plt.subplots(3,2,figsize=(10,6))
for i in range(3):
 for j in range(2):
  glm = OLS(ss.zscore(glucose_asy), ss.zscore(grad_asy[select][:,:max_mode])).fit_regularized(alpha=alphas[i,j], L1_wt=0.5)
  y_pred = glm.predict(ss.zscore(grad_asy[select][:,:max_mode]))
  rp = ss.pearsonr(ss.zscore(glucose_asy), y_pred)
  ax[i,j].scatter(range(1,max_mode+1), glm.params,cmap='Spectral',c=range(max_mode))
  ax[i,0].set_ylabel('Slope')
  ax[i,j].spines['right'].set_visible(False)
  ax[i,j].spines['top'].set_visible(False)
  ax[i,j].spines['bottom'].set_visible(False)
  ax[i,j].set_xticks([])
  ax[i,j].set_yticks([-0.3,0,0.3])
  ax[i,j].text(0.4,0.9,'\u03B1 = '+format(alphas[i,j], '.2f'), fontsize=16, transform=ax[i,j].transAxes)
  ax[i,j].text(0.4,0.05,'$\it{r}$ = '+format(rp[0], '.3f'), fontsize=16, transform=ax[i,j].transAxes)
ax[i,0].spines['bottom'].set_visible(True)
ax[i,0].set_xticks([0,max_mode/2,max_mode])  
ax[i,0].set_xlabel('Number of gradients')
ax[i,1].spines['bottom'].set_visible(True)
ax[i,1].set_xticks([0,max_mode/2,max_mode])  
ax[i,1].set_xlabel('Number of gradients')
fig.tight_layout()
fig.savefig('../../figures/fsLR-5k_ElasticNet_asy_sparsity_'+str(ratio[select])+'.png', transparent=True, dpi=300)
print('save fsLR-5k_ElasticNet_asy_sparsity_'+str(ratio[select])+'.png')

## plot competetion and lateralization
comp=np.zeros((n,max_mode))
late = np.zeros((n,max_mode))
for i in range(n):
  for j in range(max_mode):
    asy = r_ll_pred[i][j] - r_rr_pred[i][j]
    comp[i,j] = np.cov(glucose_asy, asy)[0,1]
    asy = corr_pred[i][j]
    late[i,j] = np.cov(glucose_asy, asy)[0,1]

sns.set_context("paper", font_scale = 2.5)
fig, ax = plt.subplots(figsize=(10,6))
i=2

ax.plot(range(1,101), comp[i], c='pink', zorder=1, alpha=0.5, lw=3, label='Competition')
ax.scatter(range(1,101), comp[i], cmap='pink_r', c=range(100), s=10)
ax.plot(range(1,101), late[i], c='slategray', zorder=0, alpha=0.5, lw=3, label='Lateralization')
ax.scatter(range(1,101), late[i], cmap='bone_r', c=range(100), s=10)

ax.set_ylabel('Covariance')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Steps')
plt.legend(loc='lower right', fontsize=18)
fig.tight_layout()
fig.savefig('../../figures/fsLR-5k_fitting_asy_sparsity_0.9.png', transparent=True, dpi=300)