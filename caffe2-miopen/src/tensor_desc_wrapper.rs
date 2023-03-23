crate::ix!();

/**
  | miopenTensorDescWrapper is the placeholder
  | that wraps around a miopenTensorDescriptor_t,
  | allowing us to do descriptor change
  | as-needed during runtime.
  |
  */
pub struct miopenTensorDescWrapper 
{
    desc: miopenTensorDescriptor_t,
    ty:   miopenDataType_t,
    dims: Vec<i32>,
}

impl Default for miopenTensorDescWrapper {
    
    fn default() -> Self {
        todo!();
        /*
            MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&desc_))
        */
    }
}

impl Drop for miopenTensorDescWrapper {
    fn drop(&mut self) {
        todo!();
        /*      MIOPEN_CHECK(miopenDestroyTensorDescriptor(desc_));  */
    }
}

impl miopenTensorDescWrapper {
    
    #[inline] pub fn descriptor(&mut self, 
        type_:   miopenDataType_t,
        dims:    &Vec<i32>,
        changed: *mut bool) -> miopenTensorDescriptor_t {

        todo!();
        /*
            if(type_ == type && dims_ == dims)
            {
                // if not changed, simply return the current descriptor.
                if(changed)
                    *changed = false;
                return desc_;
            }
            CAFFE_ENFORCE_EQ(
                dims.size(), 4, "MIOPEN currently only support 4-dimensional tensor descriptor");

            type_ = type;
            dims_ = dims;
            MIOPEN_ENFORCE(
                miopenSet4dTensorDescriptor(desc_, type, dims_[0], dims_[1], dims_[2], dims_[3]));
            if(changed)
                *changed = true;
            return desc_;
        */
    }
    
    #[inline] pub fn descriptor_from_order_and_dims<T>(
        &mut self, 
        order: &StorageOrder, 
        dims:  &Vec<i32>) -> miopenTensorDescriptor_t 
    {
        todo!();
        /*
            return Descriptor(miopenTypeWrapper<T>::type, dims, nullptr);
        */
    }
}
