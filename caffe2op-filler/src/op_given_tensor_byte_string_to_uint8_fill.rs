crate::ix!();

#[test] fn given_tensor_byte_string_to_uint8_fill_example() {

    /*
    workspace.ResetWorkspace()

    val = np.array([1, 2, 3], dtype=np.uint8)
    op = core.CreateOperator(
        "GivenTensorByteStringToUInt8Fill",
        [],
        ["out"],
        values=[val.tobytes()],
        shape=val.shape,
    )

    workspace.RunOperatorOnce(op)
    print("Out:\n", workspace.FetchBlob("out"))

    Out:
     [1 2 3]

    */
}

/**
  | This op fills a uint8 output tensor with
  | the data specified by the *value* argument.
  | The data must previously be serialized
  | as a byte string. The output tensor shape
  | is specified by the *shape* argument.
  | Beware, when using this argument *value*
  | should have a value for every element
  | of the output*, as missing values will
  | not be initialized automatically.
  | If *input_as_shape* is set to *true*,
  | then the *input* should be a 1D tensor
  | containing the desired output shape
  | (the dimensions specified in *extra_shape*
  | will also be appended). In this case,
  | the *shape* argument should **not**
  | be set.
  | 
  | This op allows us to write uint8 tensors
  | to
  | 
  | Protobuf as byte strings and read them
  | back as uint8 tensors in order to avoid
  | the Protobuf uint32_t varint encoding
  | size penalty.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GivenTensorByteStringToUInt8FillOp<Context> {
    base: FillerOp<Context>,

    values: Tensor,
}

num_inputs!{GivenTensorByteStringToUInt8Fill, (0,1)}

num_outputs!{GivenTensorByteStringToUInt8Fill, 1}

args!{GivenTensorByteStringToUInt8Fill, 
    0 => ("values",         "The value for the elements of the output tensor. true /* required */"),
    1 => ("shape",          "The shape of the output tensor. Cannot set the shape argument and pass in an input at the same time."),
    2 => ("extra_shape",    "The additional dimensions appended at the end of the shape indicated by the input blob. Cannot set the extra_shape argument when there is no input blob."),
    3 => ("input_as_shape", "1D tensor containing the desired output shape. First input must be in CPU context.")
}

tensor_inference_function!{GivenTensorByteStringToUInt8Fill, 
    FillerTensorInference::<TensorProto_DataType_STRING>
}

allow_inplace!{GivenTensorByteStringToUInt8Fill, vec![(0, 0)]}

impl<Context> GivenTensorByteStringToUInt8FillOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : FillerOp<Context>(operator_def, ws) 

        const ArgumentHelper helper(operator_def);
        if (!helper.HasArgument("dtype")) {
          Extract();
        } else {
          auto dtype = cast::GetCastDataType(helper, "dtype");
          switch (dtype) {
            case TensorProto_DataType_STRING:
              Extract();
              break;
            case TensorProto_DataType_UNDEFINED:
              CAFFE_THROW("Cannot have undefined 'dtype' argument");
            default:
              CAFFE_THROW("Unexpected 'dtype' argument value: ", dtype);
          }
        }
        */
    }
    
    #[inline] pub fn fill(&mut self, output: *mut Tensor) -> bool {
        
        todo!();
        /*
            DCHECK_EQ(output->numel(), values_.numel())
            << "output size: " << output->numel()
            << " given size: " << values_.numel();
        auto* data = output->template mutable_data<uint8_t>();
        const uint8_t* values_data = values_.template data<uint8_t>();
        if (output->numel()) {
          context_.template CopySameDevice<uint8_t>(
              output->numel(), values_data, data);
        }
        return true;
        */
    }
    
    #[inline] pub fn extract(&mut self)  {
        
        todo!();
        /*
            auto source_values = this->template GetRepeatedArgument<string>("values");
        DCHECK_EQ(source_values.size(), 1)
            << "expected size: 1 "
            << " given size: " << source_values.size();

        auto str = source_values[0];
        ReinitializeTensor(
            &values_,
            {static_cast<int64_t>(str.size())},
            at::dtype<uint8_t>().device(CPU));
        uint8_t* values_data = values_.template mutable_data<uint8_t>();
        for (int i = 0; i < str.size(); i++) {
          values_data[i] = static_cast<uint8_t>(str[i]);
        }
        */
    }
}

register_cpu_operator!{
    GivenTensorByteStringToUInt8Fill,
    GivenTensorByteStringToUInt8FillOp<CPUContext>
}

no_gradient!{GivenTensorByteStringToUInt8Fill}
