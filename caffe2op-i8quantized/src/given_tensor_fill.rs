crate::ix!();

/**
  | Creates quantized tensor of type char(byte)
  | with scale and zero point info.
  |
  */
pub struct Int8GivenTensorFillOp {
    storage:    OperatorStorage,
    context:    CPUContext,
    scale:      f32,
    zero_point: i32,
    shape:      Vec<i64>,
    values:     Tensor,
}

num_inputs!{Int8GivenTensorFill, 0}

num_outputs!{Int8GivenTensorFill, 1}

outputs!{Int8GivenTensorFill, 
    0 => ("Tensor", "An Int8TensorCPU with scale and zero point info")
}

args!{Int8GivenTensorFill, 
    0 => ("values", "Input array of type char(byte)"),
    1 => ("shape", "Input tensor shape"),
    2 => ("Y_scale", "Output tensor quantization scale"),
    3 => ("Y_zero_point", "Output tensor quantization offset")
}

tensor_inference_function!{Int8GivenTensorFill, FillerTensorInference}

impl Int8GivenTensorFillOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            scale_(this->template GetSingleArgument<float>("Y_scale", 1.0)),
            zero_point_( this->template GetSingleArgument<int32_t>("Y_zero_point", 0)),
            shape_(this->template GetRepeatedArgument<int64_t>("shape")) 

        ExtractValues();
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto* output = Outputs()[0]->template GetMutable<Int8TensorCPU>();
        ReinitializeTensor(&output->t, shape_, at::dtype<uint8_t>().device(CPU));
        output->scale = scale_;
        output->zero_point = zero_point_;
        return Fill(output);
        */
    }
    
    #[inline] pub fn extract_values(&mut self)  {
        
        todo!();
        /*
            auto source_values = this->template GetSingleArgument<string>("values", "");
        ReinitializeTensor(
            &values_,
            {static_cast<int64_t>(source_values.size())},
            at::dtype<uint8_t>().device(CPU));
        uint8_t* values_data = values_.template mutable_data<uint8_t>();
        for (int i = 0; i < source_values.size(); i++) {
          values_data[i] = static_cast<uint8_t>(source_values[i]);
        }
        */
    }
    
    #[inline] pub fn fill(&mut self, output: *mut Int8TensorCPU) -> bool {
        
        todo!();
        /*
            DCHECK_EQ(output->t.numel(), values_.numel())
            << "output size: " << output->t.numel()
            << " given size: " << values_.numel();
        auto* data = output->t.template mutable_data<uint8_t>();
        const uint8_t* values_data = values_.template data<uint8_t>();
        if (output->t.numel()) {
          context_.template CopySameDevice<uint8_t>(
              output->t.numel(), values_data, data);
        }
        return true;
        */
    }
}

/**
  | Creates quantized tensor of type int32
  | with scale and zero point info.
  |
  */
pub struct Int8GivenIntTensorFillOp {
    storage: OperatorStorage,
    context: CPUContext,
    scale:      f32,
    zero_point: i32,
    shape:      Vec<i64>,
    values:     Tensor,
}

num_inputs!{Int8GivenIntTensorFill, 0}

num_outputs!{Int8GivenIntTensorFill, 1}

outputs!{Int8GivenIntTensorFill, 
    0 => ("Tensor", "An Int8TensorCPU with scale and zero point info")
}

args!{Int8GivenIntTensorFill, 
    0 => ("values", "Input array of type int32"),
    1 => ("shape", "Input tensor shape"),
    2 => ("Y_scale", "Output tensor quantization scale"),
    3 => ("Y_zero_point", "Output tensor quantization offset")
}

tensor_inference_function!{Int8GivenIntTensorFill, FillerTensorInference}

impl Int8GivenIntTensorFillOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            scale_(this->template GetSingleArgument<float>("Y_scale", 1.0)),
            zero_point_( this->template GetSingleArgument<int32_t>("Y_zero_point", 0)),
            shape_(this->template GetRepeatedArgument<int64_t>("shape")) 

        ExtractValues();
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto* output = Outputs()[0]->template GetMutable<Int8TensorCPU>();
        output->t.Resize(shape_);
        output->scale = scale_;
        output->zero_point = zero_point_;
        return Fill(output);
        */
    }
    
    #[inline] pub fn extract_values(&mut self)  {
        
        todo!();
        /*
            auto source_values = this->template GetRepeatedArgument<int32_t>("values");
        ReinitializeTensor(
            &values_,
            {static_cast<int64_t>(source_values.size())},
            at::dtype<int32_t>().device(CPU));
        auto* values_data = values_.template mutable_data<int32_t>();
        for (int i = 0; i < source_values.size(); i++) {
          values_data[i] = static_cast<int32_t>(source_values[i]);
        }
        */
    }
    
    #[inline] pub fn fill(&mut self, output: *mut Int8TensorCPU) -> bool {
        
        todo!();
        /*
            DCHECK_EQ(output->t.numel(), values_.numel())
            << "output size: " << output->t.numel()
            << " given size: " << values_.numel();
        auto* data = output->t.template mutable_data<int32_t>();
        const auto* values_data = values_.template data<int32_t>();
        if (output->t.numel()) {
          context_.template CopySameDevice<int32_t>(
              output->t.numel(), values_data, data);
        }
        return true;
        */
    }
}

register_cpu_operator!{Int8GivenTensorFill,    int8::Int8GivenTensorFillOp}

register_cpu_operator!{Int8GivenIntTensorFill, int8::Int8GivenIntTensorFillOp}
