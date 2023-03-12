crate::ix!();

/**
  | The operator fills the diagonal elements
  | of the output tensor (>= 2D) with a constant
  | value specified by the 'value' argument,
  | and others 0. If number of dimensions
  | of the output tensor is greater than
  | 2, all dimensions must be equal.
  | 
  | The data type is specified by the 'dtype'
  | argument. The 'dtype' argument must
  | be one of the data types specified in
  | the 'DataType' enum field in the TensorProto
  | message. If the 'dtype' argument is
  | not provided, the data type of 'value'
  | is used.
  | 
  | The output tensor shape is specified
  | by the 'shape' argument. If the number
  | of input is 1, the shape will be identical
  | to that of the input at run time with optional
  | additional dimensions appended at
  | the end as specified by 'extra_shape'
  | argument. In that case the 'shape' argument
  | should not be set.
  | 
  | If input_as_shape is set to true, then
  | the input should be a 1D tensor containing
  | the desired output shape (the dimensions
  | specified in extra_shape will also
  | be appended)
  | 
  | -----------
  | @note
  | 
  | Currently, it supports data type of
  | float, int32, int64, and bool.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct DiagonalFillOp<Context> {

    base: FillerOp<Context>,
    body: fn(output: *mut Tensor) -> bool,
}

num_inputs!{DiagonalFill, (0,1)}

num_outputs!{DiagonalFill, 1}

inputs!{DiagonalFill, 
    0 => ("input", "Input tensor (optional) to provide shape information.")
}

outputs!{DiagonalFill, 
    0 => ("output", "Output tensor argument and its type is specified by the 'dtype' argument")
}

args!{DiagonalFill, 
    0 => ("value", "The value for the elements of the output tensor."),
    1 => ("dtype", "The data type for the elements of the output tensor. Strictly must be one of the types from DataType enum in TensorProto."),
    2 => ("shape", "The shape of the output tensor. Cannot set the shape argument and pass in an input at the same time."),
    3 => ("extra_shape", "The additional dimensions appended at the end of the shape indicated by the input blob. Cannot set the extra_shape argument when there is no input blob."),
    4 => ("input_as_shape", "1D tensor containing the desired output shape")
}

allow_inplace!{DiagonalFill, vec![(0, 0)]}

tensor_inference_function!{DiagonalFill, FillerTensorInference}

impl<Context> DiagonalFillOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : FillerOp<Context>(std::forward<Args>(args)...) 

        TensorProto_DataType dtype =
            static_cast<TensorProto_DataType>(this->template GetSingleArgument<int>(
                "dtype", TensorProto_DataType_FLOAT));

        if (!OperatorStorage::HasArgument("dtype") &&
            OperatorStorage::HasArgument("value")) {
          // If 'dtype' is not provided, infer type based on the type of 'value'
          // Currently, single argument contains either float, int64 or bytes
          if (this->template HasSingleArgumentOfType<float>("value")) {
            dtype = TensorProto_DataType_FLOAT;
          } else if (this->template HasSingleArgumentOfType<int64_t>("value")) {
            dtype = TensorProto_DataType_INT64;
          } else {
            CAFFE_THROW("Argument 'value' is of unexpected type");
          }
          VLOG(1) << "Argument 'dtype' is not provided. Assume the data type is "
                  << "the same as that of argument 'value': " << dtype;
        }

        switch (dtype) {
          case TensorProto_DataType_FLOAT:
            body_ = &DiagonalFillOp::FillWithType<float>;
            break;
          case TensorProto_DataType_DOUBLE:
            body_ = &DiagonalFillOp::FillWithType<double>;
            break;
          case TensorProto_DataType_BOOL:
            body_ = &DiagonalFillOp::FillWithType<bool>;
            break;
          case TensorProto_DataType_INT8:
            body_ = &DiagonalFillOp::FillWithType<int8_t>;
            break;
          case TensorProto_DataType_INT16:
            body_ = &DiagonalFillOp::FillWithType<int16_t>;
            break;
          case TensorProto_DataType_INT32:
            body_ = &DiagonalFillOp::FillWithType<int>;
            break;
          case TensorProto_DataType_INT64:
            body_ = &DiagonalFillOp::FillWithType<int64_t>;
            break;
          case TensorProto_DataType_UINT8:
            body_ = &DiagonalFillOp::FillWithType<uint8_t>;
            break;
          case TensorProto_DataType_UINT16:
            body_ = &DiagonalFillOp::FillWithType<uint16_t>;
            break;
          case TensorProto_DataType_UNDEFINED:
            CAFFE_THROW("Cannot have undefined 'dtype' argument");
          default:
            CAFFE_THROW("Unexpected 'dtype' argument value: ", dtype);
        }
        */
    }
    
    #[inline] pub fn fill(&mut self, output: *mut Tensor) -> bool {
        
        todo!();
        /*
            return (this->*body_)(output);
        */
    }
    
    #[inline] pub fn verify_output_shape(&mut self, output: *mut Tensor)  {
        
        todo!();
        /*
            CAFFE_ENFORCE(output->dim() >= 2, "Input shape must be >= 2D");
        */
    }
    
    #[inline] pub fn get_step_size(&mut self, output: *mut Tensor) -> i64 {
        
        todo!();
        /*
            int64_t step;
        if (output->dim() == 2) {
          step = output->size(1) + 1;
        } else {
          int64_t prev_i = output->size(0);
          for (auto i : output->sizes()) {
            if (i != prev_i) {
              CAFFE_THROW("All dimensions of input must be of equal length");
            }
          }
          vector<int64_t> cumprod(output->dim());
          auto dims = output->sizes();
          std::partial_sum(
              dims.begin(),
              dims.end() - 1,
              cumprod.begin(),
              std::multiplies<int64_t>());
          step = 1 +
              std::accumulate(
                     cumprod.begin(), cumprod.end(), static_cast<int64_t>(0));
          VLOG(0) << step;
        }
        return step;
        */
    }
}

impl DiagonalFillOp<CPUContext> {

    #[inline] pub fn fill_with_type<T>(&mut self, output: *mut Tensor) -> bool {
        todo!();
        /*
            VerifyOutputShape(output);
          T value = OperatorStorage::GetSingleArgument<T>("value", 0);
          auto* data = output->template mutable_data<T>();
          // first fill everything with 0
          math::Set<T, CPUContext>(output->numel(), T(0), data, &context_);
          // then calculate step size for diagonal
          auto step = GetStepSize(output);
          for (int64_t i = 0; i < output->numel(); i += step) {
            math::Set<T, CPUContext>(1, value, data, &context_);
            data += step;
          }
          return true;
        */
    }
}
