crate::ix!();

/**
  | TODO
  | 
  | cudnn_sys::cudnnGetVersion();
  |
  */
pub const CUDNN_VERSION: usize = 5000; 

/**
  |Caffe2 requires cudnn version 5.0 or above.
  |
  |CUDNN version under 6.0 is supported at best
  |effort.
  |
  |We strongly encourage you to move to 6.0 and
  |above.
  |
  |This message is intended to annoy you enough to
  |update.
  */
const_assert!{ CUDNN_VERSION >= 5000 } 

#[macro_export] macro_rules! cudnn_version_min {
    ($major:ident, 
     $minor:ident, 
     $patch:ident) => {
        todo!();
        /*
        CUDNN_VERSION >= ((major) * 1000 + (minor) * 100 + (patch))
        */
    }
}

/**
  | A helper function to obtain cudnn error
  | strings.
  |
  */
#[inline] pub fn cudnn_get_error_string(status: CudnnStatus) -> *const u8 {
    
    todo!();
    /*
        switch (status) {
        case CUDNN_STATUS_SUCCESS:
          return "CUDNN_STATUS_SUCCESS";
        case CUDNN_STATUS_NOT_INITIALIZED:
          return "CUDNN_STATUS_NOT_INITIALIZED";
        case CUDNN_STATUS_ALLOC_FAILED:
          return "CUDNN_STATUS_ALLOC_FAILED";
        case CUDNN_STATUS_BAD_PARAM:
          return "CUDNN_STATUS_BAD_PARAM";
        case CUDNN_STATUS_INTERNAL_ERROR:
          return "CUDNN_STATUS_INTERNAL_ERROR";
        case CUDNN_STATUS_INVALID_VALUE:
          return "CUDNN_STATUS_INVALID_VALUE";
        case CUDNN_STATUS_ARCH_MISMATCH:
          return "CUDNN_STATUS_ARCH_MISMATCH";
        case CUDNN_STATUS_MAPPING_ERROR:
          return "CUDNN_STATUS_MAPPING_ERROR";
        case CUDNN_STATUS_EXECUTION_FAILED:
          return "CUDNN_STATUS_EXECUTION_FAILED";
        case CUDNN_STATUS_NOT_SUPPORTED:
          return "CUDNN_STATUS_NOT_SUPPORTED";
        case CUDNN_STATUS_LICENSE_ERROR:
          return "CUDNN_STATUS_LICENSE_ERROR";
        default:
          return "Unknown cudnn error number";
      }
    */
}

/**
  | A macro that wraps around a cudnn statement
  | so we can check if the cudnn execution
  | finishes or not.
  |
  */
#[macro_export] macro_rules! cudnn_enforce {
    ($condition:expr) => {
        todo!();
        /*
        cudnnStatus_t status = condition;                     
        CAFFE_ENFORCE_EQ(                                     
            status,                                           
            CUDNN_STATUS_SUCCESS,                             
            ", Error at: ",                                   
            __FILE__,                                         
            ":",                                              
            __LINE__,                                         
            ": ",                                             
            ::caffe2::internal::cudnnGetErrorString(status)); 
        */
    }
}

#[macro_export] macro_rules! cudnn_check {
    ($condition:expr) => {
        todo!();
        /*
        cudnnStatus_t status = condition;                       
        CHECK(status == CUDNN_STATUS_SUCCESS)                   
            << ::caffe2::internal::cudnnGetErrorString(status); 
        */
    }
}

/// report the version of cuDNN Caffe2 was compiled with
#[inline] pub fn cudnn_compiled_version() -> usize {
    
    todo!();
    /*
        return CUDNN_VERSION;
    */
}

/**
  | report the runtime version of cuDNN
  |
  */
#[inline] pub fn cudnn_runtime_version() -> usize {
    
    todo!();
    /*
        return cudnnGetVersion();
    */
}

/**
  | Check compatibility of compiled and
  | runtime cuDNN versions
  |
  */
#[inline] pub fn check_cudnn_versions()  {
    
    todo!();
    /*
        // Version format is major*1000 + minor*100 + patch
      // If compiled with version < 7, major, minor and patch must all match
      // If compiled with version >= 7, then either
      //    runtime_version > compiled_version
      //    major and minor match
      bool version_match = cudnnCompiledVersion() == cudnnRuntimeVersion();
      bool compiled_with_7 = cudnnCompiledVersion() >= 7000;
      bool backwards_compatible_7 = compiled_with_7 && cudnnRuntimeVersion() >= cudnnCompiledVersion();
      bool patch_compatible = compiled_with_7 && (cudnnRuntimeVersion() / 100) == (cudnnCompiledVersion() / 100);
      CAFFE_ENFORCE(version_match || backwards_compatible_7 || patch_compatible,
                    "cuDNN compiled (", cudnnCompiledVersion(), ") and "
                    "runtime (", cudnnRuntimeVersion(), ") versions mismatch");
    */
}

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

/**
  | A wrapper function to convert the Caffe
  | storage order to cudnn storage order
  | enum values.
  |
  */
#[inline] pub fn get_cudnn_tensor_format(order: &StorageOrder) -> CudnnTensorFormat {
    
    todo!();
    /*
        switch (order) {
        case StorageOrder::NHWC:
          return CUDNN_TENSOR_NHWC;
        case StorageOrder::NCHW:
          return CUDNN_TENSOR_NCHW;
        default:
          LOG(FATAL) << "Unknown cudnn equivalent for order: " << order;
      }
      // Just to suppress compiler warnings
      return CUDNN_TENSOR_NCHW;
    */
}

/**
  | cudnnTensorDescWrapper is the placeholder
  | that wraps around a cudnnTensorDescriptor_t,
  | allowing us to do descriptor change
  | as-needed during runtime.
  |
  */
pub struct CudnnTensorDescWrapper {
    desc:   CudnnTensorDescriptor,
    format: CudnnTensorFormat,
    ty:     CudnnDataType,
    dims:   Vec::<i32>
}

impl Default for CudnnTensorDescWrapper {
    
    fn default() -> Self {
        todo!();
        /*
            CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&desc_))
        */
    }
}

impl Drop for CudnnTensorDescWrapper {
    fn drop(&mut self) {
        todo!();
        /*
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc_));
        */
    }
}

impl CudnnTensorDescWrapper {
    
    #[inline] pub fn descriptor(&mut self, 
        format:  CudnnTensorFormat,
        ty:      CudnnDataType,
        dims:    &Vec<i32>,
        changed: *mut bool) -> CudnnTensorDescriptor
    {
        todo!();
        /*
            if (type_ == type && format_ == format && dims_ == dims) {
          // if not changed, simply return the current descriptor.
          if (changed)
            *changed = false;
          return desc_;
        }
        CAFFE_ENFORCE_EQ(
            dims.size(), 4U, "Currently only 4-dimensional descriptor supported.");
        format_ = format;
        type_ = type;
        dims_ = dims;
        CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
            desc_,
            format,
            type,
            dims_[0],
            (format == CUDNN_TENSOR_NCHW ? dims_[1] : dims_[3]),
            (format == CUDNN_TENSOR_NCHW ? dims_[2] : dims_[1]),
            (format == CUDNN_TENSOR_NCHW ? dims_[3] : dims_[2])));
        if (changed)
          *changed = true;
        return desc_;
        */
    }

    #[inline] pub fn descriptor_create<T>(
        &mut self, 
        order: &StorageOrder,
        dims: &Vec<i32>) -> CudnnTensorDescriptor 
    {
        todo!();
        /*
            return Descriptor(
                GetCudnnTensorFormat(order), cudnnTypeWrapper<T>::type, dims, nullptr);
        */
    }
}

pub struct CudnnFilterDescWrapper {
    desc:  CudnnFilterDescriptor,
    order: StorageOrder,
    ty:    CudnnDataType,
    dims:  Vec<i32>,
}

impl Default for CudnnFilterDescWrapper {
    
    fn default() -> Self {
        todo!();
        /*
            CUDNN_ENFORCE(cudnnCreateFilterDescriptor(&desc_))
        */
    }
}

impl Drop for CudnnFilterDescWrapper {
    fn drop(&mut self) {
        todo!();
        //CUDNN_CHECK(cudnnDestroyFilterDescriptor(desc_));
    }
}

impl CudnnFilterDescWrapper {
    
    #[inline] pub fn descriptor(&mut self, 
        order:   &StorageOrder,
        ty:      CudnnDataType,
        dims:    &Vec<i32>,
        changed: *mut bool) {
        
        todo!();
        /*
            if (type_ == type && order_ == order && dims_ == dims) {
          // if not changed, simply return the current descriptor.
          if (changed)
            *changed = false;
          return desc_;
        }
        CAFFE_ENFORCE_EQ(
            dims.size(), 4U, "Currently only 4-dimensional descriptor supported.");
        order_ = order;
        type_ = type;
        dims_ = dims;
        CUDNN_ENFORCE(cudnnSetFilter4dDescriptor(
            desc_,
            type,
            GetCudnnTensorFormat(order),
            dims_[0],
            // TODO - confirm that this is correct for NHWC
            (order == StorageOrder::NCHW ? dims_[1] : dims_[3]),
            (order == StorageOrder::NCHW ? dims_[2] : dims_[1]),
            (order == StorageOrder::NCHW ? dims_[3] : dims_[2])));
        if (changed)
          *changed = true;
        return desc_;
        */
    }

    #[inline] pub fn descriptor_create<T>(
        &mut self,
        order: &StorageOrder,
        dims: &Vec<i32>) -> CudnnFilterDescriptor 
    {
        todo!();
        /*
            return Descriptor(order, cudnnTypeWrapper<T>::type, dims, nullptr);
        */
    }

}

#[inline] pub fn print_cudnn_info() -> bool {
    
    todo!();
    /*
        VLOG(1) << "Caffe2 is built with Cudnn version " << CUDNN_VERSION;
      return true;
    */
}

register_caffe2_init_function!{
   print_cudnn_info, 
   print_cudnn_info, 
   "Print Cudnn Info."
}
