crate::ix!();

/**
 | This operator fills the elements of the output
 | tensor with a constant value specified by the
 | `value` argument.
 |
 | - The data type is specified by the `dtype`
 | argument
 |
 | - Currently, the data types supported are *f32*,
 | *int32*, *int64*, and *bool*
 |
 | - If the `dtype` argument is not provided, the
 | data type of `value` is used
 |
 | - The output tensor shape is either specified by
 | the `shape` argument or will match the shape of
 | the input tensor if one is provided (if an input
 | tensor is provided, a shape argument should not be
 | set)
 |
 | - Optional additional dimensions can be appended
 | at the end as specified by `extra_shape` argument
 |
 | - If `input_as_shape` is set to True, the input
 | should be a 1D tensor containing the desired
 | output shape (the dimensions specified in
 | `extra_shape` will also be appended)
 |
 | - If a second input V is passed, fill the output
 | with the first element of V
 |
 | When specifying `dtype` argument, use the integer
 | keys from the *DataType* enum in TensorProto:
 |
 | ```
 | message TensorProto {
 |   ...
 |   enum DataType {
 |     UNDEFINED = 0;
 |     FLOAT = 1;  // float
 |     INT32 = 2;  // int
 |     BYTE = 3;  // BYTE, when deserialized, is going to be restored as uint8.
 |     STRING = 4;  // string
 |     BOOL = 5;  // bool
 |     UINT8 = 6;  // uint8_t
 |     INT8 = 7;  // int8_t
 |     UINT16 = 8;  // uint16_t
 |     INT16 = 9;  // int16_t
 |     INT64 = 10;  // int64_t
 |     FLOAT16 = 12;  // at::Half
 |     DOUBLE = 13;  // double
 |   }
 | ```
 |
 | Github Links:
 |
 | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.cc
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ConstantFillOp<Context> {

    base: FillerOp<Context>,
    body: fn(output: *mut Tensor) -> bool,
}

num_inputs!{ConstantFill, (0,2)}

num_outputs!{ConstantFill, 1}

inputs!{ConstantFill, 
    0 => ("X", "*(type: Tensor)* [OPTIONAL] Input tensor to provide shape information.")
}

outputs!{ConstantFill, 
    0 => ("Y", "*(type: Tensor)* Output tensor of constant values.")
}

args!{ConstantFill, 
    0 => ("value", "*(type: primitive; default: 0.0f) value to populate output tensor with."),
    1 => ("dtype", "*(type: int)* The data type for the elements of the output tensor. Strictly must be one of the types from *DataType* enum in TensorProto."),
    2 => ("shape", "*(type: int | Tuple(int))* Shape of the output tensor. Cannot pass an input blob and this arg at the same time."),
    3 => ("extra_shape", "*(type: int | Tuple(int))* Additional dimensions appended at the end of the shape indicated by the input blob. Cannot set this argument when there is no input blob."),
    4 => ("input_as_shape", "*(type: int | Tuple(int))* 1D tensor containing the desired output shape. First input must be in CPU context.")
}

allow_inplace!{ConstantFill, vec![(0, 0)]}

tensor_inference_function!{ConstantFill, FillerTensorInference}

impl<Context> ConstantFillOp<Context> {
    
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
            body_ = &ConstantFillOp::FillWithType<float>;
            break;
          case TensorProto_DataType_DOUBLE:
            body_ = &ConstantFillOp::FillWithType<double>;
            break;
          case TensorProto_DataType_BOOL:
            body_ = &ConstantFillOp::FillWithType<bool>;
            break;
          case TensorProto_DataType_INT8:
            body_ = &ConstantFillOp::FillWithType<int8_t>;
            break;
          case TensorProto_DataType_INT16:
            body_ = &ConstantFillOp::FillWithType<int16_t>;
            break;
          case TensorProto_DataType_INT32:
            body_ = &ConstantFillOp::FillWithType<int>;
            break;
          case TensorProto_DataType_INT64:
            body_ = &ConstantFillOp::FillWithType<int64_t>;
            break;
          case TensorProto_DataType_UINT8:
            body_ = &ConstantFillOp::FillWithType<uint8_t>;
            break;
          case TensorProto_DataType_UINT16:
            body_ = &ConstantFillOp::FillWithType<uint16_t>;
            break;
          case TensorProto_DataType_STRING:
            body_ = &ConstantFillOp::FillWithString;
            break;
          case TensorProto_DataType_UNDEFINED:
            CAFFE_THROW("ConstantFill op cannot have undefined 'dtype' argument");
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

    #[inline] pub fn fill_with_type<T>(&mut self, output: *mut Tensor) -> bool {
        todo!();
        /*
            T value = this->template GetSingleArgument<T>("value", 0);
            if (InputSize() == 2) {
              auto& value_vec = Input(1);
              if (value_vec) {
                CAFFE_ENFORCE_EQ(
                    value_vec.size(), 1, "value vector must have 1 element");
                value = value_vec.template data<T>()[0];
              }
            }

            auto* data = output->template mutable_data<T>();
            if (output->numel()) {
              math::Set<T, Context>(output->numel(), value, data, &context_);
            }
            return true;
        */
    }
    
    #[inline] pub fn fill_with_string(&mut self, output: *mut Tensor) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE_LT(
            InputSize(), 2, "constant fill string from tensor is not supported");
        auto value = this->template GetSingleArgument<std::string>("value", "");
        auto* data = output->template mutable_data<std::string>();
        for (int i = 0; i < output->numel(); ++i) {
          data[i] = value;
        }
        return true;
        */
    }
}
