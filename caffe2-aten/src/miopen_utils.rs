crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/miopen/Utils.h]

/**
  | This function makes tensors which have
  | zero stride contiguous, by setting
  | the strides to 1.
  |
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
