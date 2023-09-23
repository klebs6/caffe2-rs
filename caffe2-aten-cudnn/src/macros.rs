crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cudnn/Macros.h]

/**
  | Note: The version below should not actually be
  | 8000. Instead, it should be whatever version of
  | cuDNN that v8 API work with PyTorch correctly.
  |
  | The version is set to 8000 today for
  | convenience of debugging.
  */
#[cfg(all(USE_EXPERIMENTAL_CUDNN_V8_API,CUDNN_VERSION,CUDNN_VERSION_GTE_8000))]
pub const HAS_CUDNN_V8: bool = true;

#[cfg(not(all(USE_EXPERIMENTAL_CUDNN_V8_API,CUDNN_VERSION,CUDNN_VERSION_GTE_8000)))]
pub const HAS_CUDNN_V8: bool = false;
