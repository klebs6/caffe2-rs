crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/mkl/Descriptors.h]

pub struct DftiDescriptorDeleter {

}

impl DftiDescriptorDeleter {

    pub fn invoke(&mut self, desc: *mut DFTI_DESCRIPTOR)  {
        
        todo!();
        /*
            if (desc != nullptr) {
          MKL_DFTI_CHECK(DftiFreeDescriptor(&desc));
        }
        */
    }
}


pub struct DftiDescriptor {
    desc: Box<DFTI_DESCRIPTOR,DftiDescriptorDeleter>,
}

impl DftiDescriptor {

    pub fn init(&mut self, 
        precision:   DFTI_CONFIG_VALUE,
        signal_type: DFTI_CONFIG_VALUE,
        signal_ndim: MKL_LONG,
        sizes:       *mut MKL_LONG)  {
        
        todo!();
        /*
            if (desc_ != nullptr) {
          throw runtime_error("DFTI DESCRIPTOR can only be initialized once");
        }
        DFTI_DESCRIPTOR *raw_desc;
        if (signal_ndim == 1) {
          MKL_DFTI_CHECK(DftiCreateDescriptor(&raw_desc, precision, signal_type, 1, sizes[0]));
        } else {
          MKL_DFTI_CHECK(DftiCreateDescriptor(&raw_desc, precision, signal_type, signal_ndim, sizes));
        }
        desc_.reset(raw_desc);
        */
    }
    
    pub fn get(&self) -> *mut DFTI_DESCRIPTOR {
        
        todo!();
        /*
            if (desc_ == nullptr) {
          throw runtime_error("DFTI DESCRIPTOR has not been initialized");
        }
        return desc_.get();
        */
    }
}
