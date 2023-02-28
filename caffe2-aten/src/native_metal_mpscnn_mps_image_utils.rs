crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/metal/mpscnn/MPSImageUtils.h]

#[inline] pub fn image_from_tensor(tensor: &Tensor) -> *mut MPSImage {
    
    todo!();
        /*
            TORCH_CHECK(tensor.is_metal());
      using MetalTensorImplStorage = native::metal::MetalTensorImplStorage;
      using MetalTensorImpl = MetalTensorImpl<MetalTensorImplStorage>;
      MetalTensorImpl* impl = (MetalTensorImpl*)tensor.unsafeGetTensorImpl();
      MetalTensorImplStorage& implStorage = impl->unsafe_opaque_handle();
      return implStorage.texture()->image();
        */
}

/**
  | MPSImage carries a IntList shape which
  | is identical to the shape of the CPU tensor
  | it’s converted from.
  | 
  | 1) 1D tensors (W,) are always stored
  | as MPSImage(N=1, C=1, H=1, W=W).
  | 
  | 2) 2D tensors (H, W) are always stored
  | as MPSImage(N=1, C=1, H=H, W=W).
  | 
  | 3) 3D tensors (C, H, W) are always stored
  | as MPSImage(N=1, C=C, H=H, W=W).
  | 
  | 4) 4D tensors (N, C, H, W) are always
  | stored as MPSImage(N=N, C=C, H=H, W=W).
  | 
  | 5) 5D tensors (T, N, C, H, W) are always
  | stored as MPSImage(N=T*N, C=C, H=H,
  | W=W). 6) ...
  |
  */
#[inline] pub fn compute_image_size(sizes: &[i32]) -> Vec<i64> {
    
    todo!();
        /*
            vector<i64> imageSize(4, 1);
      i64 index = 3;
      i64 batch = 1;
      for (int i = sizes.size() - 1; i >= 0; i--) {
        if (index != 0) {
            imageSize[index] = sizes[i];
          index--;
          continue;
        }
        // For higher dimensional tensors,
        // multiply rest of dims into imageSize[0]
        batch *= sizes[i];
      }
      imageSize[0] = batch;
      return imageSize;
        */
}
