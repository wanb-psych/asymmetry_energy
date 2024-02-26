# Cortical energy cost of hemispheric functional organization

## Step 1: transform indiviudal glucose metabolism map from volume to surface

```
bash ./bash/CMRglu3mm_2_surface.sh 
```

## Step 2: functional connectome (FC) gradients at group level
**Outputs also contain individual FC matrix but not gradients**
```
conda activate pet

# parcellate CMRglc map into glasser360 and save fsLR-5k individuall  
python ./python/py01_glucose_mapping_fsLR.py

# calculate the FC gradients for hemispheres at fsLR-5k group level
python ./python/py02_fc_group_grad_vertex.py
# calculate the FC gradients for hemispheres at glasser group level
python ./python/py02_fc_group_grad_parcel.py fsLR 
```
## Step 3: Modeling between CMRglc and gradient maps