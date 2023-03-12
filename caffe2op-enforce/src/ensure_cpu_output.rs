crate::ix!();

/**
  | This Op always create TensorCPU output,
  | and may involves cross-device MemCpy.
  | 
  | Under CPU Context, this Op takes TensorCPU
  | as input. Under the CUDA Context, this
  | Op accepts either CUDA or CPU Tensor
  | input.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct EnsureCPUOutputOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

impl<Context> EnsureCPUOutputOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (this->InputIsTensorType(0, CPU)) {
          return CopyWithContext<CPUContext>();
        } else if (this->InputIsTensorType(0, Context::GetDeviceType())) {
          // CUDA Context will go this branch
          return CopyWithContext<Context>();
        } else {
          CAFFE_THROW(
              "Unexpected Input Blob: ",
              OperatorStorage::Inputs().at(0)->meta().name());
        }
        return true;
        */
    }

    #[inline] pub fn copy_with_context<InputContext>(&mut self) -> bool {
        todo!();
        /*
            // Output is always on CPU
            auto* output = this->template Output<Tensor>(0, CPU);
            auto& input =
                this->template Input<Tensor>(0, InputContext::GetDeviceType());
            output->ResizeLike(input);
            context_.CopyItemsToCPU(
                input.dtype(),
                input.numel(),
                input.raw_data(),
                output->raw_mutable_data(input.dtype()));
            context_.FinishDeviceComputation();
            return true;
        */
    }
}

/// From CPU Context, the op takes CPU tensor as input, and produces
/// TensorCPU
register_cpu_operator!{
    EnsureCPUOutput, 
    EnsureCPUOutputOp<CPUContext>
}

num_inputs!{EnsureCPUOutput, 1}

num_outputs!{EnsureCPUOutput, 1}

inputs!{EnsureCPUOutput, 
    0 => ("input", "The input CUDA or CPU tensor.")
}

outputs!{EnsureCPUOutput, 
    0 => ("output", "TensorCPU that is a copy of the input.")
}

identical_type_and_shape!{EnsureCPUOutput}

inputs_can_cross_devices!{EnsureCPUOutput}

device_inference_function!{EnsureCPUOutput, 
    |def: &OperatorDef| {
        todo!();
        /*
          auto op_device = def.has_device_option() ? def.device_option() : DeviceOption();
          auto cpu_option = DeviceOption();
          vector<DeviceOption> in_dev(def.input_size(), op_device);
          vector<DeviceOption> out_dev(def.output_size(), cpu_option);
          return std::make_pair(in_dev, out_dev);
        */
    }
}

no_gradient!{EnsureCPUOutput}
