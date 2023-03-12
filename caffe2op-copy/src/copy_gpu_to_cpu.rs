crate::ix!();

/**
  | Copy tensor for GPU to CPU context. Must
  | be run under GPU device option.
  |
  */
pub type CopyGPUToCPUOp = CopyOp<CPUContext, GPUContext, CPUContext>;

num_inputs!{CopyGPUToCPU, 1}

num_outputs!{CopyGPUToCPU, 1}

inputs!{CopyGPUToCPU, 
    0 => ("input", "The input tensor.")
}

outputs!{CopyGPUToCPU, 
    0 => ("output", "Tensor that will contain a copy of the input.")
}

identical_type_and_shape!{CopyGPUToCPU}

inputs_can_cross_devices!{CopyGPUToCPU}

device_inference_function!{CopyGPUToCPU, 
    |def: &OperatorDef| {
        todo!();
        /*
          CAFFE_ENFORCE( def.has_device_option(), "CopyGPUToCPU op should have cuda device option.");
          auto& cuda_option = def.device_option();
          auto cpu_option = DeviceOption();
          vector<DeviceOption> in_dev(def.input_size(), cuda_option);
          vector<DeviceOption> out_dev(def.output_size(), cpu_option);
          return std::make_pair(in_dev, out_dev);
        */
    }
}
