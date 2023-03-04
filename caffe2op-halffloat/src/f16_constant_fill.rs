crate::ix!();

///-----------------
#[USE_OPERATOR_FUNCTIONS("CPUContext")]
pub struct Float16ConstantFillOp {
    storage: OperatorStorage,
    context: CPUContext,
    shape:   Vec<i64>,
}

register_cpu_operator!{Float16ConstantFill, Float16ConstantFillOp}

num_inputs!{Float16ConstantFill, 0}

num_outputs!{Float16ConstantFill, 1}

outputs!{Float16ConstantFill, 
    0 => ("output", "Output tensor of constant values specified by 'value'")
}

args!{Float16ConstantFill, 
    0 => ("value", "The value for the elements of the output tensor."),
    1 => ("shape", "The shape of the output tensor.")
}

tensor_inference_function!{Float16ConstantFill, Float16FillerTensorInference }

impl Float16ConstantFillOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            shape_(this->template GetRepeatedArgument<int64_t>("shape"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto* output = Output(0, shape_, at::dtype<at::Half>());
      const float givenValue =
          this->template GetSingleArgument<float>("value", 0.0f);
      at::Half givenFp16Value = givenValue;

      if (output->numel()) {
        at::Half* out = output->template mutable_data<at::Half>();
        std::fill(out, out + output->numel(), givenFp16Value);
      }
      return true;
        */
    }
}
