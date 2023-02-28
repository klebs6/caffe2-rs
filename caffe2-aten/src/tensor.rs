crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/Tensor.cpp]

impl Tensor {
    
    pub fn enforce_invariants(&mut self)  {
        
        todo!();
        /*
            if (impl_.get() == nullptr) {
        throw runtime_error("TensorImpl with nullptr is not supported");
      }
      // Following line throws if the method is not a POD data type or is not
      // supported by ATen
      scalar_type();
      if (defined()) {
        TORCH_INTERNAL_ASSERT(
            impl_->dtype_initialized(),
            "Partially-initialized tensor not supported by Tensor");
        TORCH_INTERNAL_ASSERT(
            !impl_->is_sparse(),
            "Sparse Tensors are supported by Tensor, but invariant checking isn't implemented.  Please file a bug.");
        TORCH_INTERNAL_ASSERT(
            impl_->storage_initialized(),
            "Partially-initialized tensor not supported by Tensor");
      }
        */
    }
    
    pub fn print(&self)  {
        
        todo!();
        /*
            if (defined()) {
        cerr << "[" << toString() << " " << sizes() << "]" << endl;
      } else {
        cerr << "[UndefinedTensor]" << endl;
      }
        */
    }
    
    pub fn to_string(&self) -> String {
        
        todo!();
        /*
            string base_str;
      if (scalar_type() == ScalarType::Undefined) {
        base_str = "UndefinedType";
      } else {
        base_str = string(toString(options().computeDispatchKey())) + toString(scalar_type()) + "Type";
      }
      return base_str;
        */
    }
    
    pub fn variable_data(&self) -> Tensor {
        
        todo!();
        /*
            return GetVariableHooks()->variable_data(*this);
        */
    }
    
    pub fn tensor_data(&self) -> Tensor {
        
        todo!();
        /*
            return GetVariableHooks()->tensor_data(*this);
        */
    }
    
    pub fn is_leaf(&self) -> bool {
        
        todo!();
        /*
            return GetVariableHooks()->is_leaf(*this);
        */
    }
    
    pub fn output_nr(&self) -> i64 {
        
        todo!();
        /*
            return GetVariableHooks()->output_nr(*this);
        */
    }
    
    pub fn set_data(&self, new_data: &Tensor)  {
        
        todo!();
        /*
            GetVariableHooks()->set_data(*this, new_data);
        */
    }
    
    pub fn data(&self) -> Tensor {
        
        todo!();
        /*
            return GetVariableHooks()->data(*this);
        */
    }
    
    pub fn version(&self) -> i64 {
        
        todo!();
        /*
            return GetVariableHooks()->_version(*this);
        */
    }
    
    pub fn retain_grad(&self)  {
        
        todo!();
        /*
            GetVariableHooks()->retain_grad(*this);
        */
    }
    
    pub fn retains_grad(&self) -> bool {
        
        todo!();
        /*
            return GetVariableHooks()->retains_grad(*this);
        */
    }
    
    pub fn backward(&self, 
        inputs:       TensorList,
        gradient:     &Option<Tensor>,
        keep_graph:   Option<bool>,
        create_graph: bool)  {
        
        todo!();
        /*
            return GetVariableHooks()->_backward(*this, inputs, gradient, keep_graph, create_graph);
        */
    }
    
    pub fn requires_grad(&self, requires_grad: bool) -> &Tensor {
        
        todo!();
        /*
            GetVariableHooks()->requires_grad_(*this, _requires_grad);
      return *this;
        */
    }

    // View Variables
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    pub fn is_view(&self) -> bool {
        
        todo!();
        /*
            return GetVariableHooks()->is_view(*this);
        */
    }
    
    pub fn base(&self) -> &Tensor {
        
        todo!();
        /*
            return GetVariableHooks()->base(*this);
        */
    }
    
    pub fn name(&self) -> &String {
        
        todo!();
        /*
            return GetVariableHooks()->name(*this);
        */
    }
    
    pub fn grad_fn(&self) -> &Arc<TorchautogradNode> {
        
        todo!();
        /*
            return GetVariableHooks()->grad_fn(*this);
        */
    }
    
    pub fn remove_hook(&self, pos: u32)  {
        
        todo!();
        /*
            GetVariableHooks()->remove_hook(*this, pos);
        */
    }
    
    pub fn register_hook(&self, hook: fn(_0: &Tensor) -> Tensor) -> u32 {
        
        todo!();
        /*
            return GetVariableHooks()->_register_hook(*this, move(hook));
        */
    }
}
