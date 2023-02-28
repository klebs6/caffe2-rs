crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/Generator.cpp]

impl Generator {
    
    pub fn set_state(&mut self, new_state: &Tensor)  {
        
        todo!();
        /*
            TORCH_CHECK(new_state.defined(), "Undefined tensor is not allowed");
      this->impl_->set_state(*new_state.unsafeGetTensorImpl());
        */
    }
    
    pub fn get_state(&self) -> Tensor {
        
        todo!();
        /*
            return Tensor::wrap_tensor_impl(this->impl_->get_state());
        */
    }
}
