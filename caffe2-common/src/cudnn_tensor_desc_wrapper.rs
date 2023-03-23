crate::ix!();

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
