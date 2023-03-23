crate::ix!();

/**
  | cudnnTypeWrapper is a wrapper class
  | that allows us to refer to the cudnn type
  | in a template function. The class is
  | specialized explicitly for different
  | data types below.
  |
  */
pub trait CudnnTypeWrapper {
    type CudnnDataType;
    type ScalingParamType;
    type BNParamType;
    fn kOne()  -> *const ScalingParamType;
    fn kZero() -> *const ScalingParamType;
}

#[macro_export] macro_rules! impl_cudnn_type_wrapper {
    () => {

    };
    ($ty:ty, $CudnnDataType:tt) => {

        impl CudnnTypeWrapper for $ty {

            type CudnnDataType    = $CudnnDataType;
            type ScalingParamType = $ty;
            type BNParamType      = $ty;

            fn kOne()  -> *const ScalingParamType {
                todo!();
                /*
                   static ScalingParamType v = $ty::one();
                   return &v;
                */
            }
            fn kZero() -> *const ScalingParamType {
                todo!();
                /*
                   static ScalingParamType v = $ty::zero();
                   return &v;
                */
            }
        }
    }
}

impl_cudnn_type_wrapper![/*f32, CUDNN_DATA_FLOAT*/];
impl_cudnn_type_wrapper![/*f64, CUDNN_DATA_DOUBLE*/];
impl_cudnn_type_wrapper![/*f16, CUDNN_DATA_HALF*/];

#[cfg(cudnn_version_min = "6")]
impl_cudnn_type_wrapper![/*i32, cudnnDataType_t::CUDNN_DATA_INT32*/];
