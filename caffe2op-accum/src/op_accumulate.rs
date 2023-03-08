crate::ix!();

use crate::{
    Operator,
    OperatorStorage,
};

/**
  | Accumulate operator accumulates the
  | input tensor to the output tensor. If
  | the output tensor already has the right
  | size, we add to it; otherwise, we first
  | initialize the output tensor to all
  | zeros, and then do accumulation. Any
  | further calls to the operator, given
  | that no one else fiddles with the output
  | in the interim, will do simple accumulations.
  | 
  | Accumulation is done using Axpby operation
  | as shown:
  | 
  | Y = 1*X + gamma*Y
  | 
  | where X is the input tensor, Y is the output
  | tensor and gamma is the multiplier argument.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AccumulateOp<T, Context> {


    storage: OperatorStorage,
    context: Context,
    gamma:   T,
}

impl<T,Context> AccumulateOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            gamma_(static_cast<T>(
                this->template GetSingleArgument<float>("gamma", 1.0)))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);

        // TODO: the operator depends on output being set to 0 before the run
        auto* output = Output(0, input.sizes(), at::dtype<T>());
        math::Axpby<T, T, Context>(
            input.numel(),
            static_cast<T>(1),
            input.template data<T>(),
            gamma_,
            output->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}

register_cpu_operator!{
    Accumulate, 
    AccumulateOp<f32, CPUContext>
}

num_inputs!{Accumulate, 1}

num_outputs!{Accumulate, 1}

inputs!{Accumulate, 
    0 => ("input", 
        "The input tensor that has to be accumulated to the output tensor. 
        If the output size is not the same as input size, the output tensor 
        is first reshaped and initialized to zero, and only then, accumulation is done.")
}

outputs!{Accumulate, 
    0 => ("output", "Accumulated output tensor")
}

args!{Accumulate, 
    0 => ("gamma", "(float, default 1.0) Accumulation multiplier")
}

identical_type_and_shape!{Accumulate}

should_not_do_gradient!{Accumulate}
