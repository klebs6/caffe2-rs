crate::ix!();

/**
  | Returns a new tensor with boolean elements
  | representing if each element is NaN
  | or not.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct IsNanOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{IsNaN, IsNanOp<CPUContext>}

num_inputs!{IsNaN, 1}

num_outputs!{IsNaN, 1}

inputs!{IsNaN, 
    0 => ("tensor", "Tensor to check for nan")
}

outputs!{IsNaN, 
    0 => ("output", "Tensor containing a 1 at each location of NaN elements.")
}

impl<Context> IsNanOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self, ) -> bool {
        todo!();
        /*
            auto& X = Input(0);
        auto* Y = Output(0, X.sizes(), at::dtype<uint8_t>());
        const auto* X_data = X.template data<T>();
        uint8_t* Y_data = Y->template mutable_data<uint8_t>();
        for (size_t i = 0; i < X.numel(); i++) {
          Y_data[i] = (uint8_t)(std::isnan(X_data[i]));
        }
        return true;
        */
    }
}
