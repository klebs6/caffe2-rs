crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cudnn/Utils.h]

/**
  | cuDNN has a buggy check for tensor being
  | contiguous (that is, it does not ignore stride
  | for dimension that is equal to 0).
  |
  | This function makes tensors which have zero
  | stride contiguous, by setting the strides to
  | 1 as cuDNN likes.
  */
#[inline] pub fn contiguous_if_zero_in_strides(t: &Tensor) -> Tensor {
    
    todo!();
        /*
            for (auto s : t.strides()) {
        if (s == 0) return t.contiguous();
      }
      return t;
        */
}
