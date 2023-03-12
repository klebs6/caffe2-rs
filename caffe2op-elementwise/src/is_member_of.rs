crate::ix!();

/**
  | The *IsMemberOf* op takes an input tensor
  | *X* and a list of values as argument,
  | and produces one output data tensor
  | *Y*.
  | 
  | The output tensor is the same shape as
  | *X* and contains booleans. The output
  | is calculated as the function *f(x)
  | = x in value* and is applied to *X* elementwise.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_logical_ops.cc
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_logical_ops.h
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_DISPATCH_HELPER]
pub struct IsMemberOfOp<Context> {

    storage: OperatorStorage,
    context: Context,

    values: IsMemberOfValueHolder,

    // Input: X, output: Y
}

num_inputs!{IsMemberOf, 1}

num_outputs!{IsMemberOf, 1}

inputs!{IsMemberOf, 
    0 => ("X", "Input tensor of any shape")
}

outputs!{IsMemberOf, 
    0 => ("Y", "Output tensor (same size as X containing booleans)")
}

args!{IsMemberOf, 
    0 => ("value", "*(type: []; default: -)* List of values to check for membership."),
    1 => ("dtype", "*(type: TensorProto_DataType; default: -)* The data type for the elements of the output tensor. Strictly must be one of the types from DataType enum in TensorProto.")
}

tensor_inference_function!{IsMemberOf, /*[](const OperatorDef&, const vector<TensorShape>& input_types) {
          vector<TensorShape> out(1);
          out[0] = input_types[0];
          out[0].set_data_type(TensorProto_DataType::TensorProto_DataType_BOOL);
          return out;
        }*/
}

type TestableTypes = TensorTypes<(i32, i64, bool, String)>;

impl<Context> IsMemberOfOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        auto dtype =
            static_cast<TensorProto_DataType>(this->template GetSingleArgument<int>(
                "dtype", TensorProto_DataType_UNDEFINED));
        switch (dtype) {
          case TensorProto_DataType_INT32:
            values_.set(this->template GetRepeatedArgument<int32_t>(VALUE_TAG));
            break;
          case TensorProto_DataType_INT64:
            values_.set(this->template GetRepeatedArgument<int64_t>(VALUE_TAG));
            break;
          case TensorProto_DataType_BOOL:
            values_.set(this->template GetRepeatedArgument<bool>(VALUE_TAG));
            break;
          case TensorProto_DataType_STRING:
            values_.set(this->template GetRepeatedArgument<std::string>(VALUE_TAG));
            break;
          case TensorProto_DataType_UNDEFINED:
            // If dtype is not provided, values_ will be filled the first time that
            // DoRunWithType is called.
            break;
          default:
            CAFFE_THROW("Unexpected 'dtype' argument value: ", dtype);
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<
            TensorTypes<int32_t, int64_t, bool, std::string>>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>() -> bool {
        todo!();
        /*
            auto& input = Input(0);

            auto* output = Output(0, input.sizes(), at::dtype<bool>());

            if (!values_.has_values()) {
              values_.set(this->template GetRepeatedArgument<T>(VALUE_TAG));
            }
            const auto& values = values_.get<T>();

            const T* input_data = input.template data<T>();
            bool* output_data = output->template mutable_data<bool>();
            for (int i = 0; i < input.numel(); ++i) {
              output_data[i] = values.find(input_data[i]) != values.end();
            }
            return true;
        */
    }
}

register_cpu_operator!{Where, WhereOp<CPUContext>}

should_not_do_gradient!{Where}

register_cpu_operator!{IsMemberOf, IsMemberOfOp<CPUContext>}

should_not_do_gradient!{IsMemberOf}
