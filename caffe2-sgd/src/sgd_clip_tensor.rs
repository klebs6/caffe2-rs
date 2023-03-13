crate::ix!();


/**
  Clips the input tensor by scaling based on the input value and the threshold.
  The value is usually the (pre-computed) norm of the tensor. If the value is
  larger than the threshold, scaling would be performed in this way:

  tensor *= (threshold / value).

  An optional input called additional_threshold can be provided which
  will scale the original threshold before it is used. That is,
  the final threshold will become threshold * additional_threshold.
  This op could be used for gradient clipping.
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ClipTensorByScalingOp<Context> {
    context: Context,
    threshold:  f32,
}

impl<Context> Operator for ClipTensorByScalingOp<Context> {

}

register_cpu_operator!{ClipTensorByScaling, ClipTensorByScalingOp<CPUContext>}

num_inputs!{ClipTensorByScaling, (2,3)}

num_outputs!{ClipTensorByScaling, 1}

inputs!{ClipTensorByScaling, 
    0 => ("input_tensor", "Tensor of floats to be clipped."),
    1 => ("val", "Value to be compared against the threshold"),
    2 => ("additional_threshold", "An optional additional threshold to scale the original threshold")
}

outputs!{ClipTensorByScaling, 
    0 => ("clipped", "Tensor of floats, which is the same size as the input tensor, representing the clipped tensor.")
}

args!{ClipTensorByScaling, 
    0 => ("threshold", "Threshold to determine whether to scale down the tensor")
}

allow_inplace!{ClipTensorByScaling, vec![(0, 0)]}

should_not_do_gradient!{ClipTensorByScaling}

impl<Context> ClipTensorByScalingOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator(operator_def, ws) 

        threshold_ = this->template GetSingleArgument<float>("threshold", 0.0);
        CAFFE_ENFORCE_GT(threshold_, 0, "Threshold must be greater than 0");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& input_tensor = Input(0);
        CAFFE_ENFORCE_GT(input_tensor.numel(), 0);
        const auto& val = Input(1);
        CAFFE_ENFORCE_EQ(val.numel(), 1);

        const auto* input_tensor_data = input_tensor.template data<float>();
        const auto* val_data = val.template data<float>();

        auto* clipped = Output(0, input_tensor.sizes(), at::dtype<float>());
        float* clipped_tensor_data = clipped->template mutable_data<float>();

        if (InputSize() > 2) {
          const auto& additional_threshold = Input(2);
          CAFFE_ENFORCE_EQ(additional_threshold.numel(), 1);

          threshold_ *= *(additional_threshold.template data<float>());
        }

        if (*val_data > threshold_) {
          float ratio = threshold_ / *val_data;

          math::Scale<float, float, Context>(
              clipped->numel(),
              ratio,
              input_tensor_data,
              clipped_tensor_data,
              &context_);
        } else {
          if (input_tensor_data != clipped_tensor_data) {
            clipped->CopyFrom(input_tensor, /*async*/ true);
          }
        }

        return true;
        */
    }
}
