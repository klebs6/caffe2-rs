crate::ix!();

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
