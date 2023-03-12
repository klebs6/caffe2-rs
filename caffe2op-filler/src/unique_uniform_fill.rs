crate::ix!();

/**
  | Fill the output tensor with uniform
  | samples between min and max (inclusive).
  | 
  | If the second input is given, its elements
  | will be excluded from uniform sampling.
  | Using the second input will require
  | you to provide shape via the first input.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct UniqueUniformFillOp<Context> {
    base: FillerOp<Context>,
    body: fn(output: *mut Tensor) -> bool,
}

num_inputs!{UniqueUniformFill, (0,2)}

num_outputs!{UniqueUniformFill, 1}

inputs!{UniqueUniformFill, 
    0 => ("input", "Input tensor to provide shape information"),
    1 => ("avoid", "(optional) Avoid elements in this tensor. Elements must be unique.")
}

outputs!{UniqueUniformFill, 
    0 => ("output", "Output tensor of unique uniform samples")
}

args!{UniqueUniformFill, 
    0 => ("min",             "Minimum value, inclusive"),
    1 => ("max",             "Maximum value, inclusive"),
    2 => ("dtype",           "The data type for the elements of the output tensor. Strictly must be one of the types from DataType enum in TensorProto. This only supports INT32 and INT64 now. If not set, assume INT32"),
    3 => ("shape",           "The shape of the output tensor. Cannot set the shape argument and pass in an input at the same time."),
    4 => ("extra_shape",     "The additional dimensions appended at the end of the shape indicated by the input blob. Cannot set the extra_shape argument when there is no input blob."),
    5 => ("input_as_shape",  "1D tensor containing the desired output shape. First input must be in CPU context.")
}

allow_inplace!{UniqueUniformFill, vec![(0, 0)]}

tensor_inference_function!{UniqueUniformFill, FillerTensorInference}

impl<Context> UniqueUniformFillOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : FillerOp<Context>(std::forward<Args>(args)...) 

        TensorProto_DataType dtype =
            static_cast<TensorProto_DataType>(this->template GetSingleArgument<int>(
                "dtype", TensorProto_DataType_INT32));

        switch (dtype) {
          case TensorProto_DataType_INT32:
            CheckRange<int>();
            body_ = &UniqueUniformFillOp::FillWithType<int>;
            break;
          case TensorProto_DataType_INT64:
            CheckRange<int64_t>();
            body_ = &UniqueUniformFillOp::FillWithType<int64_t>;
            break;
          case TensorProto_DataType_UNDEFINED:
            CAFFE_THROW(
                "UniqueUniformFill op cannot have undefined 'dtype' argument");
          // break;
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

    #[inline] pub fn check_range<T>(&mut self) {
        todo!();
        /*
            CAFFE_ENFORCE(this->template HasSingleArgumentOfType<T>("min"));
            CAFFE_ENFORCE(this->template HasSingleArgumentOfType<T>("max"));
            CAFFE_ENFORCE_LT(
                this->template GetSingleArgument<T>("min", 0),
                this->template GetSingleArgument<T>("max", 0),
                "Max value should be bigger than min value.");
        */
    }

    #[inline] pub fn fill_with_type<T>(&mut self, output: *mut Tensor) -> bool {
        todo!();
        /*
            T min = this->template GetSingleArgument<T>("min", 0);
            T max = this->template GetSingleArgument<T>("max", 0);

            const T* avoid_data = nullptr;
            size_t avoid_size = 0;
            if (InputSize() >= 2) {
              auto& avoid = Input(1);
              avoid_data = avoid.template data<T>();
              avoid_size = avoid.numel();
            }
            math::RandUniformUnique<T, Context>(
                output->numel(),
                min,
                max,
                output->template mutable_data<T>(),
                avoid_size,
                avoid_data,
                &context_);
            return true;
        */
    }
}
