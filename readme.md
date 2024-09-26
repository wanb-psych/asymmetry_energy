# Cortical energy cost of hemispheric functional organization

**Bin Wan, Valentin Riedl,  Gabriel Castrillon, Matthias Kirschner, Sofie L. Valk**.  
![img](summary.png) 

## Source data

Group-level and individual data have been processed by the scripts below. See directory "./reseults"

- ./reseults/func -> functional connectivity matrix

- ./reseults/glucose -> glucose metabolism data

- ./reseults/grad -> functional connectivity gradients using diffusion map embedding

Then above results can be used to model the fitting in the paper.
- ./reseults/models -> gradient-energy models, null models, and asymmetry models


## Step 1: transform indiviudal glucose metabolism map from volume to surface

```
bash ./bash/CMRglu3mm_2_surface.sh 
```

## Step 2: functional connectome (FC) gradients at group level
**Outputs also contain individual FC matrix but not gradients**

```
# parcellate CMRglc map into glasser360 and save fsLR-5k individuall  
python ./python/py01_glucose_mapping_fsLR.py

# calculate the FC gradients for hemispheres at fsLR-5k group level
python ./python/py02_fc_group_grad_vertex.py 

# calculate the FC gradients for hemispheres at glasser group level
python ./python/py02_fc_group_grad_parcel.py fsLR

# Prepare surrogate maps for gradients or CMRglc
python ./python/py03_null.py

# When comparing asymmetry alignment between left and right
python ./python/py04_supplements_alignLH2RH.py

# visualization for vertex level
python ./python/py04_supplements_vertex.py

```

## Step 3: Modeling between CMRglc and gradient maps
ipython notebooks to visualize the results

`./python/vis01_fitting_group_level_mmp.ipynb ` visulizes all the group-level result figures. 

`./python/vis02_null.ipynb ` visulizes null model figures. 

`./python/vis03_fitting_invidual_level_mmp.ipynb ` visulizes all the invidual-level result figures.


## Main dependencies based on Python 3.8
- BrainSpace
- BrainSMASH
- Scikit-learn
- Statsmodel

## Acknowdgements
- OpenNeuro