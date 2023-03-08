crate::ix!();

pub struct TensorDescriptors<T> {
    descs:  Vec<CudnnTensorDescriptor>,
    phantom: PhantomData<T>,
}

impl<T> TensorDescriptors<T> {

    #[inline] pub fn descs(&self) -> *const CudnnTensorDescriptor {
        
        todo!();
        /*
            return descs_.data();
        */
    }

    pub fn new(
        n:      usize,
        dim:    &Vec<i32>,
        stride: &Vec<i32>) -> Self {
    
        todo!();
        /*
            descs_.resize(n);
      CAFFE_ENFORCE_EQ(dim.size(), stride.size());
      for (auto i = 0; i < n; ++i) {
        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&descs_[i]));
        CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
            descs_[i],
            cudnnTypeWrapper<T>::type,
            dim.size(),
            dim.data(),
            stride.data()));
      }
        */
    }
}

impl<T> Drop for TensorDescriptors<T> {

    fn drop(&mut self) {
        todo!();
        /* 
          for (auto desc : descs_) {
            cudnnDestroyTensorDescriptor(desc);
          }
         */
    }
}
