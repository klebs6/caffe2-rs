crate::ix!();

/**
  | Return a 1D tensor of type *int64* that
  | contains the number of elements of the
  | input tensor.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc
  | 
  | Return the size of a tensor
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SizeOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{Size, 1}

num_outputs!{Size, 1}

inputs!{Size, 
    0 => ("X", "*(type: Tensor)* Input tensor to calculate number of elements.")
}

outputs!{Size, 
    0 => ("size", "*(type: Tensor)* 1D tensor of type int64 that contains the number of elements in the input tensor *X*.")
}

register_cpu_operator!{Size, SizeOp<CPUContext>}

no_gradient!{Size}

impl<Context> SizeOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);

        auto* output = Output(0, vector<int64_t>(), at::dtype<int64_t>());
        auto* output_data = output->template mutable_data<int64_t>();

        auto size = input.numel();
        math::Set<int64_t, Context>(
            1, static_cast<int64_t>(size), output_data, &context_);

        return true;
        */
    }
}

