crate::ix!();

/**
  | Split a tensor into a list of tensors,
  | given a lengths input, along the specified
  | 'axis'. If `K` outputs are provided,
  | the op assumes `len(lengths) % K == 0`.
  | 
  | The `input` will be split into `K` parts.
  | Each part of length `sum(lengths[i*k:i*k+k))`
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SplitByLengthsOp<Context> {
    context:                      Context,
    axis:                         i32,
    scaling:                      bool,
    inclusive_scan_buffer:        Tensor, //{Context::GetDeviceType()};
    inclusive_scan_length_buffer: Tensor, //{Context::GetDeviceType()};

    /**
      | Input: X, optionally split
      | 
      | The split tensor is stored in CPU.
      |
      */
    lengths_host: Tensor, ////{CPU};
}

num_inputs!{SplitByLengths, 2}

num_outputs!{SplitByLengths, (1,INT_MAX)}

inputs!{SplitByLengths, 
    0 => ("input", "The tensor to split"),
    1 => ("legnths", "The tensor `l_i` indicates the logic block of input.")
}

args!{SplitByLengths, 
    0 => ("axis", "Which axis to split on"),
    1 => ("order", "Either NHWC or NCWH, will split on C axis, defaults to NCHW"),
    2 => ("use_scaling_lengths", "(*bool*): Enables automatic scaling of the lengths values. When enabled will automatically find a value K >= 1, such that sum(lengths) * K == len(input).")
}

device_inference_function!{SplitByLengths, 
    |def: &OperatorDef| {
        todo!();
        /*
          auto op_device = def.has_device_option() ? def.device_option() : DeviceOption();
          vector<DeviceOption> in_dev(def.input_size(), op_device);
          vector<DeviceOption> out_dev(def.output_size(), op_device);
          // lengths input should be on CPU
          in_dev[1] = DeviceOption();
          return std::make_pair(in_dev, out_dev);
        */
    }
}

impl<Context> SplitByLengthsOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        CAFFE_ENFORCE(
            !(OperatorStorage::HasArgument("axis") &&
              OperatorStorage::HasArgument("order")),
            "You shouldn't specify both the dim to split, and the order "
            "in the case of 4-D images.");
        if (OperatorStorage::HasArgument("axis")) {
          axis_ = this->template GetSingleArgument<int>("axis", 0);
        } else {
          axis_ = GetDimFromOrderString(
              this->template GetSingleArgument<string>("order", "NCHW"));
        }
         scaling_ = this->template GetSingleArgument<bool>("use_scaling_lengths", false);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
      auto lengths_length = Input(1).dim(0);
      int32_t* length_data;

      if (this->InputIsTensorType(1, CPU)) {
          length_data = Input(1).template data<int32_t>();
        } else {
          // Length input in CUDA context
          auto& input_length = Input(1);
          lengths_host_ = TensorCPU(input_length, CPU);
          length_data = lengths_host_.template data<int32_t>();
      }

      CAFFE_ENFORCE_EQ(
          lengths_length % OutputSize(),
          0,
          "len(Lengths) ", lengths_length, "should be divisible by OutputSize() ", OutputSize(), ".");
      int canonical_axis = input.canonical_axis_index(axis_);
      CAFFE_ENFORCE_LT(
          canonical_axis, input.dim(), "Axis not in input ndim range.");
      const int input_channels = input.dim32(canonical_axis);
      const auto* axis_data = length_data;

      auto sum_lengths = std::accumulate(axis_data, axis_data + lengths_length, 0);

      if (scaling_) {
        CAFFE_ENFORCE_EQ(
            input_channels % (sum_lengths ? sum_lengths : 1),
            0,
            "Input channels ", input_channels, " should be divisible by ",
            sum_lengths);
      } else {
        CAFFE_ENFORCE_EQ(
            sum_lengths,
            input_channels,
            "Input channels should be equal to split dimensions sum, ",
            input_channels, " vs ", sum_lengths
            );
      }
      vector<int64_t> output_dims(input.sizes().vec());
      int before = input.size_to_dim(canonical_axis);
      int after = input.size_from_dim(canonical_axis + 1);
      size_t input_offset = 0;
      auto dim_multiplier = sum_lengths ? (input_channels / sum_lengths): 1;

      if (!scaling_) {
        dim_multiplier = 1;
      }

      for (int i = 0; i < OutputSize(); ++i) {
        auto* output = Output(i);
        const auto* axis_offset = axis_data + lengths_length / OutputSize() * i;
        auto axis_dim = dim_multiplier * std::accumulate(
            axis_offset, axis_offset + lengths_length / OutputSize(), 0);
        output_dims[canonical_axis] = axis_dim;
        output->Resize(output_dims);
        math::CopyMatrix<Context>(
            input.itemsize(),
            before,
            axis_dim * after,
            static_cast<const char*>(input.raw_data()) + input_offset,
            input.dim32(canonical_axis) * after,
            output->raw_mutable_data(input.dtype()),
            axis_dim * after,
            &context_,
            input.dtype().copy());
        input_offset += axis_dim * after * input.itemsize();
      }
      return true;
        */
    }
}
