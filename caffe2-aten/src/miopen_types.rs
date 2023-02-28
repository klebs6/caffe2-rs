crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/miopen/Types.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/miopen/Types.cpp]

pub fn get_miopen_data_type(tensor: &Tensor) -> miopen::DataType {
    
    todo!();
        /*
            if (tensor.scalar_type() == kFloat) {
        return miopenFloat;
      } else if (tensor.scalar_type() == kHalf) {
        return miopenHalf;
      }  else if (tensor.scalar_type() == kBFloat16) {
        return miopenBFloat16;
      }
      string msg("getMiopenDataType() not supported for ");
      msg += toString(tensor.scalar_type());
      throw runtime_error(msg);
        */
}

pub fn miopen_version() -> i64 {
    
    todo!();
        /*
            return (MIOPEN_VERSION_MAJOR<<8) + (MIOPEN_VERSION_MINOR<<4) + MIOPEN_VERSION_PATCH;
        */
}
