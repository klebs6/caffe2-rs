crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/templates/TensorMethods.cpp]

impl Tensor {
    
    pub fn cpu(&self) -> Tensor {
        
        todo!();
        /*
            return to(options().device(DeviceType_CPU), /*non_blocking*/ false, /*copy*/ false);
        */
    }

    /// TODO: The Python version also accepts
    /// arguments
    ///
    pub fn cuda(&self) -> Tensor {
        
        todo!();
        /*
            return to(options().device(DeviceType::Cuda), /*non_blocking*/ false, /*copy*/ false);
        */
    }
    
    pub fn hip(&self) -> Tensor {
        
        todo!();
        /*
            return to(options().device(DeviceType_HIP), /*non_blocking*/ false, /*copy*/ false);
        */
    }
    
    pub fn vulkan(&self) -> Tensor {
        
        todo!();
        /*
            return to(options().device(DeviceType_Vulkan), /*non_blocking*/ false, /*copy*/ false);
        */
    }
    
    pub fn metal(&self) -> Tensor {
        
        todo!();
        /*
            return to(options().device(DeviceType_Metal), /*non_blocking*/ false, /*copy*/ false);
        */
    }
    
    pub fn to_type(&self, t: ScalarType) -> Tensor {
        
        todo!();
        /*
            return to(options().dtype(t), /*non_blocking*/ false, /*copy*/ false);
        */
    }

    /// TODO: Deprecate me
    pub fn to_backend(&self, b: Backend) -> Tensor {
        
        todo!();
        /*
            return to(options().device(backendToDeviceType(b)).layout(layout_from_backend(b)), /*non_blocking*/ false, /*copy*/ false);
        */
    }
    
    pub fn options(&self) -> TensorOptions {
        
        todo!();
        /*
            return TensorOptions().dtype(dtype())
                            .device(device())
                            .layout(layout());
        */
    }
}

#[macro_export] macro_rules! define_cast {
    ($T:ident, $name:ident) => {
        /*
        
          template <>                                                       
           T* Tensor::data_ptr() const {                           
            TORCH_CHECK(                                                    
                scalar_type() == ScalarType::name,                          
                "expected scalar type "                                     
                #name                                                       
                " but found ",                                              
                scalar_type());                                             
            return this->unsafeGetTensorImpl()->data_ptr_impl<T>();         
          }
        */
    }
}

lazy_static!{
    /*
    at_forall_scalar_types_with_complex_except_complex_half!{define_cast}
    at_forall_qint_types!{define_cast}
    */
}

#[macro_export] macro_rules! define_item {
    ($T:ident, $name:ident) => {
        /*
        
          template <>                     
           T Tensor::item() const { 
            return item().to##name();     
          }
        */
    }
}

lazy_static!{
    /*
    at_forall_scalar_types_with_complex_except_complex_half!{define_item}
    */
}

