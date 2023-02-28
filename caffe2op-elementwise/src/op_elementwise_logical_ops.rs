crate::ix!();

use crate::{
    OperatorStorage,
    TensorTypes,
};

/**
  | Operator Where takes three input data
  | (Tensor, Tensor, Tensor) and produces one output
  | data (Tensor) where z = c ? x : y is applied
  | elementwise.
  |
  */
pub struct WhereOp<Context> {

    //USE_OPERATOR_FUNCTIONS(Context);
    //USE_DISPATCH_HELPER;
    storage: OperatorStorage,
    context: Context,

    enable_broadcast: bool,

    /*
      | Input: C, X, Y,
      | 
      | output: Z
      |
      */
}

num_inputs!{Where, 3}

num_outputs!{Where, 1}

inputs!{Where, 
    0 => ("C", "input tensor containing booleans"),
    1 => ("X", "input tensor"),
    2 => ("Y", "input tensor")
}

outputs!{Where, 
    0 => ("Z", "output tensor")
}

identical_type_and_shape_of_input!{Where, 1}

allow_inplace!{Where, vec![(1, 2)]}

impl<Context> WhereOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(bool, "broadcast_on_rows", enable_broadcast_, 0)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<
            TensorTypes<float, double, int, long, std::string, bool>>::
            call(this, Input(1));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& select = Input(0);
            auto& left = Input(1);
            auto& right = Input(2);

            if (enable_broadcast_) {
              CAFFE_ENFORCE_EQ(select.dim(), 1);
              CAFFE_ENFORCE_EQ(select.size(0), right.size(0));
              CAFFE_ENFORCE_EQ(left.sizes(), right.sizes());
            } else {
              CAFFE_ENFORCE_EQ(select.sizes(), left.sizes());
              CAFFE_ENFORCE_EQ(select.sizes(), right.sizes());
            }
            auto* output = Output(0, left.sizes(), at::dtype<T>());

            const bool* select_data = select.template data<bool>();
            const T* left_data = left.template data<T>();
            const T* right_data = right.template data<T>();
            T* output_data = output->template mutable_data<T>();

            if (enable_broadcast_) {
              size_t block_size = left.size_from_dim(1);
              for (int i = 0; i < select.numel(); i++) {
                size_t offset = i * block_size;
                if (select_data[i]) {
                  context_.CopyItemsSameDevice(
                      output->dtype(),
                      block_size,
                      left_data + offset,
                      output_data + offset);
                } else {
                  context_.CopyItemsSameDevice(
                      output->dtype(),
                      block_size,
                      right_data + offset,
                      output_data + offset);
                }
              }
            } else {
              for (int i = 0; i < select.numel(); ++i) {
                output_data[i] = select_data[i] ? left_data[i] : right_data[i];
              }
            }
            return true;
        */
    }
}
///---------------------------------
pub struct IsMemberOfValueHolder {
    int32_values:  HashSet<i32>,
    int64_values:  HashSet<i64>,
    bool_values:   HashSet<bool>,
    string_values: HashSet<String>,
    has_values:    bool,
}

impl IsMemberOfValueHolder {

    #[inline] pub fn set<T>(&mut self, args: &Vec<T>) {
        todo!();
        /*
            has_values_ = true;
            auto& values = get<T>();
            values.insert(args.begin(), args.end());
        */
    }
    
    #[inline] pub fn has_values(&self) -> bool {
        
        todo!();
        /*
            return has_values_;
        */
    }

    #[inline] pub fn get_i32<'a>(&'a mut self) -> &'a HashSet<i32> { 
        &self.int32_values 
    }

    #[inline] pub fn get_i64<'a>(&'a mut self) -> &'a HashSet<i64> { 
        &self.int64_values 
    }
    #[inline] pub fn get_bool<'a>(&'a mut self) -> &'a HashSet<bool> { 
        &self.bool_values 
    }

    #[inline] pub fn get_string<'a>(&'a mut self) -> &'a HashSet<String> { 
        &self.string_values 
    }

    #[inline] pub fn get<'a, T>(&'a mut self) -> &'a HashSet<T> { 
        todo!();
        //dispatch
    }
}

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
pub struct IsMemberOfOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    //USE_DISPATCH_HELPER;
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

#[test] fn is_member_of_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "IsMemberOf",
        ["X"],
        ["Y"],
        value=[0,2,4,6,8],
    )

    // Use a not-empty tensor
    workspace.FeedBlob("X", np.array([0,1,2,3,4,5,6,7,8]).astype(np.int32))
    print("X:\n", workspace.FetchBlob("X"))

    workspace.RunOperatorOnce(op)
    print("Y: \n", workspace.FetchBlob("Y"))

    **Result**

    // value=[0,2,4,6,8]

    X:
     [0 1 2 3 4 5 6 7 8]
    Y:
     [ True False  True False  True False  True False  True]
    */
}

const IS_MEMBER_OF_OP_VALUE_TAG: &'static str = "value";

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
