import nibabel as nib
import numpy as np
from brainspace.plotting import plot_hemispheres
from brainspace import mesh

rh_fsLR = mesh.mesh_io.read_surface('../../src/fs_LR.32k.R.inflated.surf.gii')
lh_fsLR = mesh.mesh_io.read_surface('../../src/fs_LR.32k.L.inflated.surf.gii')

rh_fsa5 = mesh.mesh_io.read_surface('../../src/fsaverage5.rh.inflated.surf.gii')
lh_fsa5 = mesh.mesh_io.read_surface('../../src/fsaverage5.lh.inflated.surf.gii')

rh_fsLR_5k = mesh.mesh_io.read_surface('../../src/fsLR-5k.R.inflated.surf.gii')
lh_fsLR_5k = mesh.mesh_io.read_surface('../../src/fsLR-5k.L.inflated.surf.gii')

def plot_surface(data, 
                 size,
                 cmap,
                 filename,
                 surf, nan_color = (0,0,0,1),
                 display=False, **kwargs):
  if surf=='fsa5':
    lh = lh_fsa5
    rh = rh_fsa5
  elif surf=='fsLR':
    lh = lh_fsLR
    rh = rh_fsLR
  elif surf=='fsLR-5k':
    lh = lh_fsLR_5k
    rh = rh_fsLR_5k  
                   
  if display==False:
    plot_hemispheres(surf_lh=lh, surf_rh=rh, array_name = data,size=size,
                     cmap = cmap, color_bar = True, nan_color = nan_color,
                     interactive = False, zoom = 1.5, embed_nb = True, transparent_bg=True,
                     cb__labelTextProperty={"fontSize": 24}, screenshot=True, filename=filename, **kwargs)
  else: 
    plot_hemispheres(surf_lh=lh, surf_rh=rh, array_name = data,size=size,
                     cmap = cmap, color_bar = True, nan_color = nan_color,
                     interactive = False, zoom = 1.5, embed_nb = True, transparent_bg=True,
                     cb__labelTextProperty={"fontSize": 24}, screenshot=True, filename=filename, **kwargs)
    fig = plot_hemispheres(surf_lh=lh, surf_rh=rh, array_name = data, nan_color = nan_color,size = size,
                           cmap = cmap, color_bar = True,
                           cb__labelTextProperty={"fontSize": 24}, interactive = False, zoom = 1.5, embed_nb = True, **kwargs)
    return fig

# colors I like
red=[244/256, 81/256, 30/256]
blue=[37/256, 150/256, 190/256]