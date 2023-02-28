crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/InitialTensorOptions.h]

/**
  | Represents the initial TensorOptions, before
  | the "defaults" are ever changed.
  |
  | This is designed to be used in library code,
  | where the explicit devices, dtypes, etc. are
  | known.
  |
  | NOTE: this is not a stable API.
  */
#[inline] pub fn initial_tensor_options() -> TensorOptions {
    
    todo!();
        /*
            return TensorOptions(kCPU).dtype(kFloat).layout(kStrided)
                                .requires_grad(false);
        */
}
