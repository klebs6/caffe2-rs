crate::ix!();

/**
  | Take a CPU input tensor and copy it to
  | an output in the current
  | 
  | Context (GPU or CPU). This may involves
  | cross-device MemCpy.
  |
  */
pub type CopyFromCPUInputOp = CopyOp<CPUContext, CPUContext, CPUContext>;

num_inputs!{CopyFromCPUInput, 1}

num_outputs!{CopyFromCPUInput, 1}

inputs!{CopyFromCPUInput, 
    0 => ("input", "The input CPU tensor.")
}

outputs!{CopyFromCPUInput, 
    0 => ("output", "either a TensorCUDA or a TensorCPU")
}

identical_type_and_shape!{CopyFromCPUInput}

inputs_can_cross_devices!{CopyFromCPUInput}

device_inference_function!{CopyFromCPUInput, 
    |def: &OperatorDef| {
        /*
          auto op_device = def.has_device_option() ? def.device_option() : DeviceOption();
          auto cpu_option = DeviceOption();
          vector<DeviceOption> in_dev(def.input_size(), cpu_option);
          vector<DeviceOption> out_dev(def.output_size(), op_device);
          return std::make_pair(in_dev, out_dev);
        */
    }
}

// From CPU, copy it to whatever the current context
register_cpu_operator!{
    CopyFromCPUInput,
    CopyOp<CPUContext, CPUContext, CPUContext>
}

