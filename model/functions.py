import numpy as np
import torch
import itertools


# Auxiliary functions to create & undo patches
def Patch(
    X:torch.Tensor,
    patch_size:int,
    ):
    if len(X.size())==5:
        X = torch.squeeze(X, dim=1)
    h, w = X.shape[-2], X.shape[-1]
    assert h%patch_size==0, f"Patch size must divide images height"
    assert w%patch_size==0, f"Patch size must divide images width"
    patches = X.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patch_list = []
    for row, col in itertools.product(range(h//patch_size), range(w//patch_size)):
        patch_list.append(patches[:,:,row,col,:,:])
    patches = torch.stack(patch_list, dim = 1)
    return patches

def Unflatten(
    flattened:torch.Tensor,
    ):
    bs, n, p = flattened.size()
    unflattened = torch.reshape(flattened, (bs, n, 3, int(np.sqrt(p/3)), int(np.sqrt(p/3))))
    return unflattened

def Unpatch(
    patches:torch.Tensor,
    ):
    if len(patches.size()) < 5:
        batch_size, num_patches, ch, h, w = Unflatten(patches).size()
    else:
        batch_size, num_patches, ch, h, w = patches.size()
    elem_per_axis = int(np.sqrt(num_patches))
    patches_middle = torch.cat([patch for patch in patches.reshape(batch_size,elem_per_axis,elem_per_axis,ch,h,w)[0]], dim = -2)
    restored_image = torch.cat([patch for patch in patches_middle], dim = -1).reshape(batch_size,1,ch,h*elem_per_axis,w*elem_per_axis)
    return restored_image


# Auxiliary methods to downsampling & upsampling
def DownSampling(encoded_patches):
    _, _, embeddings = encoded_patches.size()
    ch, h, w = 3, int(np.sqrt(embeddings/3)), int(np.sqrt(embeddings/3))
    original_image = Unpatch(Unflatten(encoded_patches))
    new_patches = Patch(original_image, patch_size = h//2)
    new_patches_flattened = torch.nn.Flatten(start_dim = -3, end_dim = -1).forward(new_patches)
    return new_patches_flattened

def UpSampling(encoded_patches):
    _, _, embeddings = encoded_patches.size()
    _, h, _ = 3, int(np.sqrt(embeddings/3)), int(np.sqrt(embeddings/3))
    original_image = Unpatch(Unflatten(encoded_patches))
    new_patches = Patch(original_image, patch_size = h*2)
    new_patches_flattened = torch.flatten(new_patches, start_dim = -3, end_dim = -1)
    return new_patches_flattened