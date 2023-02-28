crate::ix!();

use crate::{
    OperatorStorage,
    Tensor
};

/**
  | ScaleBlobs takes one or more input data
  | (Tensor) and produces one or more output
  | data (Tensor) whose value is the input
  | data tensor scaled element-wise.
  |
  */
pub struct ScaleBlobsOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage:          OperatorStorage,
    context:          Context,
    scale:            f32,
    blob_sizes:       Tensor,
    inputs:           Tensor,
    outputs:          Tensor,
    host_blob_sizes:  Tensor,
    host_inputs:      Tensor,
    host_outputs:     Tensor,
}

register_cpu_operator!{ScaleBlobs, ScaleBlobsOp<CPUContext>}

num_inputs!{ScaleBlobs, (1,INT_MAX)}

num_outputs!{ScaleBlobs, (1,INT_MAX)}

args!{ScaleBlobs, 
    0 => ("scale", "(float, default 1.0) the scale to apply.")
}

identical_type_and_shape!{ScaleBlobs}

allow_inplace!{ScaleBlobs, 
    |x: i32, y: i32| {
        true
    }
}

impl<Context> ScaleBlobsOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(float, "scale", scale_, 1.0f)
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            int batchSize = InputSize();

        for (int i = 0; i < batchSize; ++i) {
          const auto& X = Input(i);
          auto* Y = Output(i, X.sizes(), at::dtype<T>());
          math::Scale<float, T, Context>(
              X.numel(),
              scale_,
              X.template data<T>(),
              Y->template mutable_data<T>(),
              &context_);
        }
        return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            for (int i = 0; i < InputSize(); ++i) {
          auto& input = this->template Input<Tensor>(i, CPU);
          auto* output = this->template Output<Tensor>(i, CPU);
          output->ResizeLike(input);
        }
        return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
        */
    }
}
