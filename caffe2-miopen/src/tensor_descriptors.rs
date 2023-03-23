crate::ix!();

pub struct TensorDescriptors<T> {
    descs:   Vec<miopenTensorDescriptor_t>,
    phantom: PhantomData<T>,
}

impl<T> TensorDescriptors<T> {

    /*
      TensorDescriptors(
          size_t n,
          // dim and stride are not declared as const as opposed to cuDNN
          // since miopenSetTensorDescriptor doesn't take const arguments
          std::vector<int>& dim,
          std::vector<int>& stride);
    */
    
    #[inline] pub fn descs(&self) -> *const miopenTensorDescriptor_t {
        
        todo!();
        /*
            return descs_.data();
        */
    }
}
