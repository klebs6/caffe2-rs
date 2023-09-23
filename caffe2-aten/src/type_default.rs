crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/TypeDefault.h]

/**
  | This is temporary typedef to enable Quantizer
  | in aten native function API we'll remove them
  | when we are actually exposing Quantizer class
  | to frontend
  |
  */
pub type ConstQuantizerPtr = IntrusivePtr<Quantizer>;
