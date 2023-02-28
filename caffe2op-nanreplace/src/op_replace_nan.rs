crate::ix!();

use crate::{
    OperatorStorage,
    CPUContext,
};

/**
  | Replace the NaN (not a number) element
  | in the input tensor with argument `value`
  |
  */
pub struct ReplaceNaNOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{ReplaceNaN, ReplaceNaNOp<CPUContext>}

num_inputs!{ReplaceNaN, 1}

num_outputs!{ReplaceNaN, 1}

inputs!{ReplaceNaN, 
    0 => ("input", "Input tensor"),
    1 => ("output", "Output tensor")
}

args!{ReplaceNaN, 
    0 => ("value (optional)", "the value to replace NaN, the default is 0")
}

identical_type_and_shape!{ReplaceNaN}

allow_inplace!{ReplaceNaN, vec![(0, 0)]}

should_not_do_gradient!{ReplaceNaN}

impl<Context> ReplaceNaNOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            T value = this->template GetSingleArgument<T>("value", 0);

        auto& input = Input(0);

        auto* output = Output(0, input.sizes(), at::dtype<T>());

        const T* input_data = input.template data<T>();
        T* output_data = output->template mutable_data<T>();

        ReplaceNaN<T>(value, input.numel(), input_data, output_data);

        return true;
        */
    }
}

impl ReplaceNaNOp<CPUContext> {

    #[inline] pub fn replace_nan<T>(&mut self, 
        value: &T,
        size:  i64,
        x:     *const T,
        y:     *mut T)  {
    
        todo!();
        /*
            for (int64_t i = 0; i < size; i++) {
        if (std::isnan(X[i])) {
          Y[i] = value;
        } else {
          Y[i] = X[i];
        }
      }
        */
    }
}
