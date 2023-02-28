crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/QScheme.h]

/**
  | QScheme is an enum that specifies the
  | type of quantization. This has a one
  | to one correspondence with Quantizer
  | 
  | Please refer to ATen/quantized/Quantizer.h
  | to see the Quantizers classes.
  | 
  | Keep this file in sync with torch/nn/_qscheme.py
  |
  */
#[repr(u8)]
pub enum QScheme {
    PER_TENSOR_AFFINE                = 0,
    PER_CHANNEL_AFFINE               = 1,
    PER_TENSOR_SYMMETRIC             = 2,
    PER_CHANNEL_SYMMETRIC            = 3,
    PER_CHANNEL_AFFINE_FLOAT_QPARAMS = 4,
    COMPILE_TIME_NUM_QSCHEMES        = 5,
}

pub const K_PER_TENSOR_AFFINE:                 QScheme = QScheme::PER_TENSOR_AFFINE;
pub const K_PER_CHANNEL_AFFINE:                QScheme = QScheme::PER_CHANNEL_AFFINE;
pub const K_PER_TENSOR_SYMMETRIC:              QScheme = QScheme::PER_TENSOR_SYMMETRIC;
pub const K_PER_CHANNEL_SYMMETRIC:             QScheme = QScheme::PER_CHANNEL_SYMMETRIC;
pub const K_PER_CHANNEL_AFFINE_FLOAT_QPARAMS:  QScheme = QScheme::PER_CHANNEL_AFFINE_FLOAT_QPARAMS;
pub const COMPILE_TIME_NUM_QSCHEMES:           QScheme = QScheme::COMPILE_TIME_NUM_QSCHEMES;

#[inline] pub fn to_string(qscheme: QScheme) -> String {
    
    todo!();
        /*
            switch (qscheme) {
        case kPerTensorAffine:
          return "per_tensor_affine";
        case kPerChannelAffine:
          return "per_channel_affine";
        case kPerTensorSymmetric:
          return "per_tensor_symmetric";
        case kPerChannelSymmetric:
          return "per_channel_symmetric";
        case kPerChannelAffineFloatQParams:
          return "per_channel_affine_float_qparams";
        default:
          TORCH_CHECK(false, "Unrecognized qscheme: ", static_cast<int>(qscheme));
      }
        */
}
