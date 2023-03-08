crate::ix!();

/**
  | This op fills an output tensor with the
  | data specified by the *value* and *dtype*
  | arguments.
  | 
  | The output tensor shape is specified
  | by the shape* argument. Beware, when
  | using this argument value* should have
  | a value for every element of the *output*,
  | as missing values will not be initialized
  | automatically. If *input_as_shape*
  | is set to *true*, then the *input* should
  | be a 1D tensor containing the desired
  | output shape (the dimensions specified
  | in *extra_shape* will also be appended).
  | In this case, the *shape* argument should
  | **not** be set.
  | 
  | -----------
  | @note
  | 
  | Do not set the shape argument and pass
  | in an input at the same time.*
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/given_tensor_fill_op.h
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/given_tensor_fill_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GivenTensorFillOp<T, Context> {

    base:    FillerOp<Context>,

    body:    fn(output: *mut Tensor) -> bool,
    values:  Tensor,

    phantom: PhantomData<T>,
}

num_inputs!{GivenTensorFill, (0,1)}

num_outputs!{GivenTensorFill, 1}

inputs!{GivenTensorFill, 
    0 => ("input",            "(Optional) 1D tensor specifying the shape of the output. Must be used with *input_as_shape=True*")
}

outputs!{GivenTensorFill, 
    0 => ("output",           "Output tensor with desired dimension filled with specified data. If the shape argument is set, this is the shape specified, and if the *input* exists and *input_as_shape=True*, it is the shape specified by the *input* tensor.")
}

args!{GivenTensorFill, 
    0 => ("values",           "*(type depends on dtype, Required=True)* The value of the elements to go in the *output* tensor. true /* required */"),
    1 => ("dtype",            "The data type for the elements of the output tensor. Strictly must be one of the types from DataType enum in TensorProto."),
    2 => ("shape",            "*(type: [int])* Desired shape of the *output* tensor."),
    3 => ("extra_shape",      "*(type: [int])* The additional dimensions appended at the end of the *shape* indicated by the input blob. Cannot set the *extra_shape* argument when there is no input blob."),
    4 => ("input_as_shape",   "*(type: bool; default: False)* set to *True* to use the *input* as shape. First, input must be in CPU context.")
}

tensor_inference_function!{GivenTensorFill, FillerTensorInference}

allow_inplace!{GivenTensorFill, vec![(0, 0)]}

impl<T, Context> GivenTensorFillOp<T, Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : FillerOp<Context>(operator_def, ws) 

        const ArgumentHelper helper(operator_def);
        // GivenTensorFillOp can be provided with a "dtype" arg if float is
        // is specified as T. Otherwise, "dtype" is ignored.
        // In the ideal world, we would get rid of templating of T at all, but we
        // need to provide backwards compatibility.
        if (!std::is_same<T, float>::value || !helper.HasArgument("dtype")) {
          ExtractValues<T>();
        } else {
          auto dtype = cast::GetCastDataType(helper, "dtype");
          switch (dtype) {
            case TensorProto_DataType_FLOAT:
              ExtractValues<float>();
              break;
            case TensorProto_DataType_DOUBLE:
              ExtractValues<double>();
              break;
            case TensorProto_DataType_BOOL:
              ExtractValues<bool>();
              break;
            case TensorProto_DataType_INT16:
              ExtractValues<int16_t>();
              break;
            case TensorProto_DataType_INT32:
              ExtractValues<int>();
              break;
            case TensorProto_DataType_INT64:
              ExtractValues<int64_t>();
              break;
            case TensorProto_DataType_STRING:
              ExtractValues<std::string>();
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
            return (this->*body_)(output);
        */
    }
    
    #[inline] pub fn extract_values<Type>(&mut self, ) {
        todo!();
        /*
            auto source_values = this->template GetRepeatedArgument<Type>("values");
        ReinitializeTensor(
            &values_,
            {static_cast<int64_t>(source_values.size())},
            at::dtype<Type>().device(CPU));
        Type* values_data = values_.template mutable_data<Type>();
        for (int i = 0; i < source_values.size(); i++) {
          values_data[i] = static_cast<Type>(source_values[i]);
        }
        body_ = &GivenTensorFillOp::FillWithType<Type>;
        */
    }
    
    #[inline] pub fn fill_with_type<Type>(&mut self, output: *mut Tensor) -> bool {
        todo!();
        /*
            CAFFE_ENFORCE_EQ(output->numel(), values_.numel());
        auto* data = output->template mutable_data<Type>();
        const Type* values_data = values_.template data<Type>();
        if (output->numel()) {
          context_.CopyItemsFromCPU(
              TypeMeta::Make<Type>(), output->numel(), values_data, data);
        }
        return true;
        */
    }
}

register_cpu_operator!{GivenTensorFill,         GivenTensorFillOp<f32,    CPUContext>}
register_cpu_operator!{GivenTensorDoubleFill,   GivenTensorFillOp<f64,    CPUContext>}
register_cpu_operator!{GivenTensorBoolFill,     GivenTensorFillOp<bool,   CPUContext>}
register_cpu_operator!{GivenTensorInt16Fill,    GivenTensorFillOp<i16,    CPUContext>}
register_cpu_operator!{GivenTensorIntFill,      GivenTensorFillOp<i32,    CPUContext>}
register_cpu_operator!{GivenTensorInt64Fill,    GivenTensorFillOp<i64,    CPUContext>}
register_cpu_operator!{GivenTensorStringFill,   GivenTensorFillOp<String, CPUContext>}

no_gradient!{GivenTensorFill}
no_gradient!{GivenTensorDoubleFill}
no_gradient!{GivenTensorBoolFill}
no_gradient!{GivenTensorInt16Fill}
no_gradient!{GivenTensorIntFill}
no_gradient!{GivenTensorInt64Fill}
no_gradient!{GivenTensorStringFill}

///-------------------------------------

#[test] fn given_tensor_fill_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "GivenTensorFill",
        [],
        ["out"],
        values=[1., 2., 3.],
        shape=[3],
    )

    workspace.RunOperatorOnce(op)
    print("Out:\n", workspace.FetchBlob("out"))

    Out:
     [1. 2. 3.]

    */
}
///---------------------------------
num_inputs!{GivenTensorDoubleFill, (0,1)}

num_outputs!{GivenTensorDoubleFill, 1}

args!{GivenTensorDoubleFill, 
    0 => ("values",            "The value for the elements of the output tensor. true /* required */"),
    1 => ("shape",             "The shape of the output tensor. Cannot set the shape argument and pass in an input at the same time."),
    2 => ("extra_shape",       "The additional dimensions appended at the end of the shape indicated by the input blob. Cannot set the extra_shape argument when there is no input blob."),
    3 => ("input_as_shape",    "1D tensor containing the desired output shape. First input must be in CPU context.")
}

tensor_inference_function!{GivenTensorDoubleFill, 
    FillerTensorInference::<TensorProto_DataType_DOUBLE> 
}

allow_inplace!{GivenTensorDoubleFill, vec![(0, 0)]}

///--------------------------------

num_inputs!{GivenTensorBoolFill, (0,1)}

num_outputs!{GivenTensorBoolFill, 1}

args!{GivenTensorBoolFill, 
    0 => ("values",           "The value for the elements of the output tensor. true /* required */"),
    1 => ("shape",            "The shape of the output tensor. Cannot set the shape argument and pass in an input at the same time."),
    2 => ("extra_shape",      "The additional dimensions appended at the end of the shape indicated by the input blob. Cannot set the extra_shape argument when there is no input blob."),
    3 => ("input_as_shape",   "1D tensor containing the desired output shape. First input must be in CPU context.")
}

tensor_inference_function!{GivenTensorBoolFill, 
    FillerTensorInference::<TensorProto_DataType_BOOL> 
}

allow_inplace!{GivenTensorBoolFill, vec![(0, 0)]}

///----------------------------

num_inputs!{GivenTensorInt16Fill, (0,1)}

num_outputs!{GivenTensorInt16Fill, 1}

args!{GivenTensorInt16Fill, 
    0 => ("values",           "The value for the elements of the output tensor. true /* required */"),
    1 => ("shape",            "The shape of the output tensor. Cannot set the shape argument and pass in an input at the same time."),
    2 => ("extra_shape",      "The additional dimensions appended at the end of the shape indicated by the input blob. Cannot set the extra_shape argument when there is no input blob."),
    3 => ("input_as_shape",   "1D tensor containing the desired output shape. First input must be in CPU context.")
}

tensor_inference_function!{GivenTensorInt16Fill, 
   FillerTensorInference::<TensorProto_DataType_INT16> 
}

allow_inplace!{GivenTensorInt16Fill, vec![(0, 0)]}

///----------------------------

num_inputs!{GivenTensorIntFill, (0,1)}

num_outputs!{GivenTensorIntFill, 1}

args!{GivenTensorIntFill, 
    0 => ("values",           "The value for the elements of the output tensor. true /* required */"),
    1 => ("shape",            "The shape of the output tensor. Cannot set the shape argument and pass in an input at the same time."),
    2 => ("extra_shape",      "The additional dimensions appended at the end of the shape indicated by the input blob. Cannot set the extra_shape argument when there is no input blob."),
    3 => ("input_as_shape",   "1D tensor containing the desired output shape. First input must be in CPU context.")
}

tensor_inference_function!{GivenTensorIntFill, 
    FillerTensorInference::<TensorProto_DataType_INT32> 
}

allow_inplace!{GivenTensorIntFill, vec![(0, 0)]}

///-----------------------------
num_inputs!{GivenTensorInt64Fill, (0,1)}

num_outputs!{GivenTensorInt64Fill, 1}

args!{GivenTensorInt64Fill, 
    0 => ("values",           "The value for the elements of the output tensor. true /* required */"),
    1 => ("shape",            "The shape of the output tensor. Cannot set the shape argument and pass in an input at the same time."),
    2 => ("extra_shape",      "The additional dimensions appended at the end of the shape indicated by the input blob. Cannot set the extra_shape argument when there is no input blob."),
    3 => ("input_as_shape",   "1D tensor containing the desired output shape. First input must be in CPU context.")
}

tensor_inference_function!{GivenTensorInt64Fill, 
    FillerTensorInference::<TensorProto_DataType_INT64> 
}

allow_inplace!{GivenTensorInt64Fill, vec![(0, 0)]}

///-----------------------
num_inputs!{GivenTensorStringFill, (0,1)}

num_outputs!{GivenTensorStringFill, 1}

args!{GivenTensorStringFill, 
    0 => ("values",           "The value for the elements of the output tensor. true /* required */"),
    1 => ("shape",            "The shape of the output tensor. Cannot set the shape argument and pass in an input at the same time."),
    2 => ("extra_shape",      "The additional dimensions appended at the end of the shape indicated by the input blob. Cannot set the extra_shape argument when there is no input blob."),
    3 => ("input_as_shape",   "1D tensor containing the desired output shape. First input must be in CPU context.")
}

tensor_inference_function!{
    GivenTensorStringFill, 
    FillerTensorInference::<TensorProto_DataType_STRING>
}

allow_inplace!{GivenTensorStringFill, vec![(0, 0)]}

