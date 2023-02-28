crate::ix!();

use crate::{
    OperatorStorage,
    CPUContext,
    Tensor,
    OperatorDef,
    TensorShape
};

/**
  | FillerOp takes in either zero or one
  | input.
  | 
  | If the number of input is 1, the shape
  | will be identical to that of the input
  | at run time with optional additional
  | dimensions appended at the end as specified
  | by "extra_shape" argument. In that
  | case the "shape" parameter should not
  | be set.
  | 
  | If the number of inputs is 0, the full
  | shape must be provided via "shape" argument
  |
  */
pub struct FillerOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;

    storage:        OperatorStorage,
    context:        Context,
    shape:          Vec<i64>,
    extra_shape:    Vec<i64>,
    input_as_shape: bool,
}

impl<Context> FillerOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            shape_(this->template GetRepeatedArgument<int64_t>("shape")),
            extra_shape_(ToVectorint64_t(
                this->template GetRepeatedArgument<int>("extra_shape"))),
            input_as_shape_(
                this->template GetSingleArgument<bool>("input_as_shape", false)) 

        if (InputSize()) {
          if (shape_.size() != 0) {
            CAFFE_THROW(
                "Cannot set the shape argument and pass in an input at "
                "the same time");
          }
        } else {
          if (!extra_shape_.empty()) {
            CAFFE_THROW("Cannot set extra_shape when there is no input");
          }
          if (input_as_shape_) {
            CAFFE_THROW("An input must be given if input_as_shape is true");
          }
          if (shape_.size() == 0 &&
              this->template HasSingleArgumentOfType<int>("shape")) {
            CAFFE_THROW("Fill 'shape' argument was a scalar, list expected");
          }
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto* output = Operator<Context>::Output(0);
        if (InputSize()) {
          auto shape = vector<int64_t>{};
          if (input_as_shape_) {
            if (this->InputIsTensorType(0, CPU)) {
              // originally, shape input must be in CPU context
              auto& input = this->template Input<Tensor>(0, CPU);
              CAFFE_ENFORCE_EQ(
                  input.dim(),
                  1,
                  "When input_as_shape is true, the input must be a 1D tensor of "
                  "data type int64_t");
              CAFFE_ENFORCE(input.numel() > 0);
              auto* shape_data = input.template data<int64_t>();
              shape.insert(shape.end(), shape_data, shape_data + input.dim32(0));
            } else {
              // in ONNX case, we allow shape to be in CUDA context
              auto& input = Input(0);
              CAFFE_ENFORCE_EQ(
                  input.dim(),
                  1,
                  "When input_as_shape is true, the input must be a 1D tensor of "
                  "data type int64_t");
              CAFFE_ENFORCE(input.numel() > 0);
              auto* shape_data = input.template data<int64_t>();
              std::unique_ptr<int64_t[]> shape_data_copy =
                  std::make_unique<int64_t[]>(input.dim32(0));
              context_.template CopyToCPU<int64_t>(
                  input.dim32(0), shape_data, shape_data_copy.get());
              shape.insert(
                  shape.end(),
                  shape_data_copy.get(),
                  shape_data_copy.get() + input.dim32(0));
            }
          } else {
            auto& input = Input(0);
            shape.insert(shape.end(), input.sizes().begin(), input.sizes().end());
          }
          shape.insert(shape.end(), extra_shape_.begin(), extra_shape_.end());
          output->Resize(shape);
          shape_ = shape;
        } else {
          output->Resize(shape_);
        }
        return Fill(output);
        */
    }
}

///--------------------------------
#[test] fn uniform_fill_example_int() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op_1 = core.CreateOperator(
        "UniformIntFill",
        [],
        ["output"],
        min=5,
        max=10,
        shape=(3,3)
    )

    op_2 = core.CreateOperator(
        "UniformIntFill",
        ["shape", "min", "max"],
        ["output"],
        input_as_shape=1
    )

    // Test arg-based op
    workspace.RunOperatorOnce(op_1)
    print("output (op_1):\n", workspace.FetchBlob("output"))

    // Test input-based op
    workspace.ResetWorkspace()
    workspace.FeedBlob("shape", np.array([5,5]))
    workspace.FeedBlob("min", np.array(13, dtype=np.int32))
    workspace.FeedBlob("max", np.array(19, dtype=np.int32))
    workspace.RunOperatorOnce(op_2)
    print("output (op_2):\n", workspace.FetchBlob("output"))

    output (op_1):
     [[ 6 10  7]
     [ 5 10  6]
     [ 7  5 10]]
    output (op_2):
     [[19 13 15 13 13]
     [14 17 14 15 15]
     [17 14 19 13 13]
     [17 18 16 13 18]
     [14 15 16 18 16]]
    */
}

#[test] fn uniform_fill_example_float() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op_1 = core.CreateOperator(
        "UniformFill",
        [],
        ["output"],
        min=5.5,
        max=10.5,
        shape=(3,3)
    )

    op_2 = core.CreateOperator(
        "UniformFill",
        ["shape", "min", "max"],
        ["output"],
        input_as_shape=1
    )

    // Test arg-based op
    workspace.RunOperatorOnce(op_1)
    print("output (op_1):\n", workspace.FetchBlob("output"))

    // Test input-based op
    workspace.ResetWorkspace()
    workspace.FeedBlob("shape", np.array([5,5]))
    workspace.FeedBlob("min", np.array(13.8, dtype=np.float32))
    workspace.FeedBlob("max", np.array(19.3, dtype=np.float32))
    workspace.RunOperatorOnce(op_2)
    print("output (op_2):\n", workspace.FetchBlob("output"))

    output (op_1):
     [[8.894862  8.225005  6.7890406]
     [9.588293  7.1072135 7.7234955]
     [8.210596  6.0202913 9.665462 ]]
    output (op_2):
     [[18.965155 15.603871 15.038921 17.14872  18.134571]
     [18.84237  17.845276 19.214737 16.970337 15.494069]
     [18.754795 16.724329 15.311974 16.962536 18.60965 ]
     [15.186268 15.264773 18.73341  19.077969 14.237255]
     [15.917589 15.844325 16.248466 17.006554 17.502048]]

    */
}

/**
 | Fill the output tensor with float samples from
 | uniform distribution [`min`, `max`].
 |
 | - The range can be defined either by arguments or
 |   input blobs. `min` and `max` are inclusive.
 |
 |     - If the range is given by input blobs, you
 |       also need to give the shape as input.
 |
 |     - When the range is given as arguments, this
 |       operator enforces min <= max. When the range is
 |       given as inputs, the constraint is not enforced.
 |
 |     - When the range is given as inputs and max
 |       < min, the first dimension of the output is set to
 |       0. This behavior is allowed so that dynamically
 |       sampling indices into a dynamically sized tensor
 |       is possible.
 |
 | - The shape of the output can be given as argument
 |   or input.
 |
 | Github Links:
 | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h
 | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc
 |
 */
pub struct UniformFillOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    base: FillerOp<Context>,

    min: T,
    max: T,
}

num_inputs!{UniformFill, vec![0, 1, 3]}

num_outputs!{UniformFill, 1}

inputs!{UniformFill, 
    0 => ("shape", "(*Tensor`<int>`*): 1-D tensor of the shape of the output, must be used with `input_as_shape` argument"),
    1 => ("min", "(*Tensor`<T>`*): scalar tensor containing minimum value, inclusive"),
    2 => ("max", "(*Tensor`<T>`*): scalar tensor containing maximum value, inclusive")
}

outputs!{UniformFill, 
    0 => ("output", "(*Tensor`<T>`*): filled output tensor")
}

args!{UniformFill, 
    0 => ("min", "(*T*): minimum value, inclusive"),
    1 => ("max", "(*T*): maximum value, inclusive"),
    2 => ("shape", "(*Tuple(int)*): shape of the output, do not set when `input_as_shape`=1"),
    3 => ("input_as_shape", "(*int*): set to 1 to use the first input as shape; 
        `shape` input must be in CPU context")
}

allow_inplace!{UniformFill, vec![(0, 0)]}

tensor_inference_function!{UniformFill, FillerTensorInference }

impl<T, Context> UniformFillOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : FillerOp<Context>(std::forward<Args>(args)...),
            min_(this->template GetSingleArgument<T>("min", 0)),
            max_(this->template GetSingleArgument<T>("max", 1)) 

        if (InputSize() == 3) {
          CAFFE_ENFORCE(
              !this->template HasSingleArgumentOfType<T>("min"),
              "Cannot set both min arg and min input blob");
          CAFFE_ENFORCE(
              !this->template HasSingleArgumentOfType<T>("max"),
              "Cannot set both max arg and max input blob");
        } else {
          CAFFE_ENFORCE_LT(
              min_, max_, "Max value should be bigger than min value.");
        }
        */
    }
    
    #[inline] pub fn fill(&mut self, output: *mut Tensor) -> bool {
        
        todo!();
        /*
            T min = min_;
        T max = max_;
        if (InputSize() == 3) {
          CAFFE_ENFORCE_EQ(1, Input(1).numel(), "min blob must be scalar");
          CAFFE_ENFORCE_EQ(1, Input(2).numel(), "max blob must be scalar");
          min = *Input(1).template data<T>();
          max = *Input(2).template data<T>();
          if (min > max) {
            auto shape = output->sizes().vec();
            shape[0] = 0;
            output->Resize(shape);
            output->template mutable_data<T>();
            return true;
          }
        }
        math::RandUniform<T, Context>(
            output->numel(),
            min,
            max,
            output->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}

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
pub struct UniqueUniformFillOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
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

///-----------------------------------------------
#[test] fn constant_fill_example1() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ConstantFill",
        [],
        ["Y"],
        shape=(1,5,5)
    )

    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    **Result**

    Y: [[[0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0.]]]
    */
}

#[test] fn constant_fill_example2() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ConstantFill",
        ["X"],
        ["Y"],
        value=4.0,
        dtype=1,
        extra_shape=(1,2)
    )

    workspace.FeedBlob("X", (np.random.randint(100, size=(3,3))).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X: [[86. 30. 84.]
     [34. 51.  9.]
     [29. 86. 59.]]
    Y: [[[[4. 4.]]

      [[4. 4.]]

      [[4. 4.]]]


     [[[4. 4.]]

      [[4. 4.]]

      [[4. 4.]]]


     [[[4. 4.]]

      [[4. 4.]]

      [[4. 4.]]]]

    */
}

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
pub struct ConstantFillOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
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
pub struct DiagonalFillOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
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

///--------------------------------------------------
#[test] fn gaussian_fill_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "GaussianFill",
        [],
        ["out"],
        shape=[3,3],
        mean=2.0,
        std=1.1
    )

    workspace.RunOperatorOnce(op)
    print("Out:\n", workspace.FetchBlob("out"))

    Out:
     [[1.2084167  2.3336504  2.827349  ]
     [2.7108908  0.9374752  1.7173369 ]
     [0.03320992 2.1775863  1.0894578 ]]
    */
}

/**
 | This op fills an output tensor with samples drawn
 | from a normal distribution specified by the mean
 | and standard deviation arguments. The output
 | tensor shape is specified by the *shape* argument.
 |
 | However, if *input_as_shape* is set to *true*,
 | then the *input* should be a 1D tensor containing
 | the desired output shape (the dimensions specified
 | in *extra_shape* will also be appended). In this
 | case, the *shape* argument should **not** be set.
 |
 | *Note: cannot set the shape argument and pass in
 | an input at the same time.*
 |
 | Github Links:
 | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h
 | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc
 */
pub struct GaussianFillOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    base: FillerOp<Context>,

    mean: T,
    std:  T,
}

num_inputs!{GaussianFill, (0,1)}

num_outputs!{GaussianFill, 1}

inputs!{GaussianFill, 
    0 => ("input", "(Optional) 1D tensor specifying the shape of the output. Must be used with *input_as_shape=True*")
}

outputs!{GaussianFill, 
    0 => ("output", "Output tensor of random values drawn from a normal distribution. If the shape argument is set, this is the shape specified, and if the *input* exists and *input_as_shape=True*, it is the shape specified by the *input* tensor.")
}

args!{GaussianFill, 
    0 => ("mean",           "*(type: float; default: 0.)* Mean of the distribution to draw from."),
    1 => ("std",            "*(type: float; default: 1.)* Standard deviation of the distribution to draw from."),
    2 => ("shape",          "*(type: [int])* Desired shape of the *output* tensor."),
    3 => ("extra_shape",    "*(type: [int])* The additional dimensions appended at the end of the *shape* indicated by the input blob. Cannot set the *extra_shape* argument when there is no input blob."),
    4 => ("input_as_shape", "*(type: bool; default: False)* set to *True* to use the *input* as shape. First, input must be in CPU context.")
}

allow_inplace!{GaussianFill, vec![(0, 0)]}

tensor_inference_function!{GaussianFill, FillerTensorInference}

impl<T, Context> GaussianFillOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : FillerOp<Context>(std::forward<Args>(args)...),
            mean_(this->template GetSingleArgument<float>("mean", 0)),
            std_(this->template GetSingleArgument<float>("std", 1)) 

        DCHECK_GT(std_, 0) << "Standard deviation should be nonnegative.";
        */
    }
    
    #[inline] pub fn fill(&mut self, output: *mut Tensor) -> bool {
        
        todo!();
        /*
            math::RandGaussian<T, Context>(
            output->numel(),
            mean_,
            std_,
            output->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}

///-----------------------------------
#[test] fn xavier_fill_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "XavierFill",
        [],
        ["out"],
        shape=[3,3],
    )

    workspace.RunOperatorOnce(op)
    print("Out:\n", workspace.FetchBlob("out"))

    Out:
     [[-0.8412168   0.33207083 -0.88418937]
     [ 0.43059897 -0.8340702   0.07781601]
     [ 0.93261135 -0.24542928 -0.3980782 ]]
    */
}

/**
  | This op fills an output tensor with values
  | sampled from a uniform distribution
  | with the range determined by the desired
  | shape of the output.
  | 
  | Rather, than specifying the range of
  | values manually, the novelty of Xavier
  | Fill is that it automatically scales
  | the range of the distribution it draws
  | from based on the size of the desired
  | output tensor.
  | 
  | For more information check out the paper
  | [Understanding the difficulty of training
  | deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).
  | The output tensor shape is specified
  | by the *shape* argument.
  | 
  | However, if *input_as_shape* is set
  | to *true*, then the *input* should be
  | a 1D tensor containing the desired output
  | shape (the dimensions specified in
  | *extra_shape* will also be appended).
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
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc
  |
  */
pub struct XavierFillOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    base:    FillerOp<Context>,
    phantom: PhantomData<T>,
}

num_inputs!{XavierFill, (0,1)}

num_outputs!{XavierFill, 1}

inputs!{XavierFill, 
    0 => ("input", "(Optional) 1D tensor specifying the shape of the output. Must be used with *input_as_shape=True*")
}

outputs!{XavierFill, 
    0 => ("output", "Output tensor of random values drawn from an automatically scaled uniform distribution, based on the size of the output tensor. If the shape argument is set, this is the shape specified by the shape argument, and if the *input* exists and *input_as_shape=True*, it is the shape specified by the *input* tensor.")
}

args!{XavierFill, 
    0 => ("shape", "*(type: [int])* Desired shape of the *output* tensor."),
    1 => ("extra_shape", "*(type: [int])* The additional dimensions appended at the end of the *shape* indicated by the input blob. Cannot set the *extra_shape* argument when there is no input blob."),
    2 => ("input_as_shape", "*(type: bool; default: False)* set to *True* to use the *input* as shape. First, input must be in CPU context.")
}

allow_inplace!{XavierFill, vec![(0, 0)]}

tensor_inference_function!{XavierFill, FillerTensorInference}

impl<T,Context> XavierFillOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : FillerOp<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn fill(&mut self, output: *mut Tensor) -> bool {
        
        todo!();
        /*
            const int fan_in = output->numel() / output->dim32(0);
        T scale = std::sqrt(T(3) / fan_in);
        math::RandUniform<T, Context>(
            output->numel(),
            -scale,
            scale,
            output->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}

///--------------------------------------------

pub struct MSRAFillOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    base:    FillerOp<Context>,
    phantom: PhantomData<T>,
}

num_inputs!{MSRAFill, (0,1)}

num_outputs!{MSRAFill, 1}

allow_inplace!{MSRAFill, vec![(0, 0)]}

tensor_inference_function!{MSRAFill, FillerTensorInference}

impl<T, Context> MSRAFillOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : FillerOp<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn fill(&mut self, output: *mut Tensor) -> bool {
        
        todo!();
        /*
            const int fan_out = output->numel() / output->dim32(1);
        T scale = std::sqrt(T(2) / fan_out);
        math::RandGaussian<T, Context>(
            output->numel(),
            0.0,
            scale,
            output->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}

/**
  | This is mostly used just as a debugging
  | purpose stuff: it fills a tensor sequentially
  | with values 0, 1, 2..., which can then be used
  | to check e.g. reshape operations by allowing
  | one to read the indices more easily.
  */
pub struct RangeFillOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    base:    FillerOp<Context>,
    phantom: PhantomData<T>,
}

num_inputs!{RangeFill, (0,1)}

num_outputs!{RangeFill, 1}

allow_inplace!{RangeFill, vec![(0, 0)]}

tensor_inference_function!{RangeFill, FillerTensorInference }

impl<T, Context> RangeFillOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : FillerOp<Context>(std::forward<Args>(args)...)
        */
    }
}

impl RangeFillOp<f32, CPUContext> {

    #[inline] pub fn fill(&mut self, output: *mut Tensor) -> bool {
        
        todo!();
        /*
            float* data = output->template mutable_data<float>();
      for (int i = 0; i < output->numel(); ++i) {
        data[i] = i;
      }
      return true;
        */
    }
}

///------------------------------------
#[test] fn lengths_range_fill_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LengthsRangeFill",
        ["lengths"],
        ["range_sequence"],
    )

    workspace.FeedBlob("lengths", np.array([2,4,1]).astype(np.int32))
    print("lengths:\n", workspace.FetchBlob("lengths"))

    workspace.RunOperatorOnce(op)
    print("range_sequence: \n", workspace.FetchBlob("range_sequence"))

    lengths:
     [2 4 1]
    range_sequence:
     [0 1 0 1 2 3 0]
    */
}

/**
  | The *LengthsRangeFill* op takes a single
  | input lengths* and outputs a single
  | tensor range_sequence*. For each element
  | of *lengths*, the op appends the range(0,lengths)
  | vector to the end of *range_sequence*.
  | For example, if input=[2,4,1], the
  | output would be [0,1,0,1,2,3,0].
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc
  |
  */
pub struct LengthsRangeFillOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{LengthsRangeFill, 1}

num_outputs!{LengthsRangeFill, 1}

inputs!{LengthsRangeFill, 
    0 => ("lengths", "1D tensor of int32 or int64 segment lengths.")
}

outputs!{LengthsRangeFill, 
    0 => ("range_sequence", "1D tensor whose size is the sum of *lengths*")
}

impl<Context> LengthsRangeFillOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);

        auto* input_data = input.template data<int32_t>();

        CAFFE_ENFORCE_EQ(input.dim(), 1, "Input must be a vector.");

        auto len_sum = std::accumulate(input_data, input_data + input.numel(), 0);

        auto* output = Output(0, {len_sum}, at::dtype<int32_t>());
        auto* output_data = output->template mutable_data<int32_t>();

        int32_t offset = 0;
        for (int i = 0; i < input.numel(); ++i) {
          auto len = input_data[i];
          auto start = output_data + offset;
          std::iota(
              start,
              start + len,
              0); // make the third argument the arg of this operator
          offset += len;
        }
        return true;
        */
    }
}

//template <int VALUE_TYPE = TensorProto_DataType_FLOAT>
#[inline] pub fn filler_tensor_inference<const VALUE_TYPE: i32>(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        vector<TensorShape> out(1);
      ArgumentHelper helper(def);
      out[0].set_data_type(static_cast<TensorProto_DataType>(
          helper.GetSingleArgument<int>("dtype", VALUE_TYPE)));

      if (in.size()) {
        // TODO
        bool input_as_shape =
            helper.GetSingleArgument<bool>("input_as_shape", false);
        if (input_as_shape) {
          out[0].set_unknown_shape(true);
          return out;
        }
        for (auto d : in[0].dims()) {
          out[0].add_dims(d);
        }
      } else {
        auto shape = helper.GetRepeatedArgument<int64_t>("shape");
        for (auto d : shape) {
          out[0].add_dims(d);
        }
      }
      return out;
    */
}

register_cpu_operator!{UniformFill,         UniformFillOp<f32, CPUContext>}
register_cpu_operator!{UniformIntFill,      UniformFillOp<i32, CPUContext>}
register_cpu_operator!{UniqueUniformFill,   UniqueUniformFillOp<CPUContext>}
register_cpu_operator!{ConstantFill,        ConstantFillOp<CPUContext>}
register_cpu_operator!{DiagonalFill,        DiagonalFillOp<CPUContext>}
register_cpu_operator!{GaussianFill,        GaussianFillOp<f32, CPUContext>}
register_cpu_operator!{XavierFill,          XavierFillOp<f32, CPUContext>}
register_cpu_operator!{MSRAFill,            MSRAFillOp<f32, CPUContext>}
register_cpu_operator!{RangeFill,           RangeFillOp<f32, CPUContext>}
register_cpu_operator!{LengthsRangeFill,    LengthsRangeFillOp<CPUContext>}

no_gradient!{UniformFill}
no_gradient!{UniformIntFill}
no_gradient!{UniqueUniformFill}
no_gradient!{ConstantFill}
no_gradient!{DiagonalFill}
no_gradient!{GaussianFill}
no_gradient!{XavierFill}
no_gradient!{MSRAFill}
no_gradient!{RangeFill}
no_gradient!{LengthsRangeFill}
