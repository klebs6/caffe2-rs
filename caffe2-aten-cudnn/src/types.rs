crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cudnn/Types.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cudnn/Types.cpp]

pub fn get_cudnn_data_type_from_scalar_type(dtype: ScalarType) -> CudnnDataType {
    
    todo!();
        /*
            if (dtype == kFloat) {
        return CUDNN_DATA_FLOAT;
      } else if (dtype == kDouble) {
        return CUDNN_DATA_DOUBLE;
      } else if (dtype == kHalf) {
        return CUDNN_DATA_HALF;
      }
      string msg("getCudnnDataTypeFromScalarType() not supported for ");
      msg += toString(dtype);
      throw runtime_error(msg);
        */
}

pub fn get_cudnn_data_type(tensor: &Tensor) -> CudnnDataType {
    
    todo!();
        /*
            return getCudnnDataTypeFromScalarType(tensor.scalar_type());
        */
}

pub fn cudnn_version() -> i64 {
    
    todo!();
        /*
            return CUDNN_VERSION;
        */
}
