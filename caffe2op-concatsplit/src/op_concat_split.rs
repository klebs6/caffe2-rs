crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorStorage,
    Tensor,
    OpSchemaCost,
    DeviceOption,
    TensorShape,
    OperatorDef
};

#[test] fn split_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Split",
        ["input"],
        ["output_0","output_1","output_2"],
        split=(3,2,4),
        axis=0
    )

    workspace.FeedBlob("input", np.random.randint(10, size=(9)))
    print("input:", workspace.FetchBlob("input"))
    workspace.RunOperatorOnce(op)
    print("output_0:", workspace.FetchBlob("output_0"))
    print("output_1:", workspace.FetchBlob("output_1"))
    print("output_2:", workspace.FetchBlob("output_2"))



    input: [2 2 6 6 6 0 5 7 4]
    output_0: [2 2 6]
    output_1: [6 6]
    output_2: [0 5 7 4]

    */
}

pub const kSplitOpInputSize: i32 = 2;

/**
  | Split an `input` tensor into a list of
  | tensors, along the axis specified by
  | the `axis` dimension. The lengths of
  | the split can be specified using argument
  | `split` or optional second input blob
  | to the operator. Otherwise, the tensor
  | is split to equal sized parts.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SplitOp<Context> {
    context: Context,

    axis:     i32,
    add_axis: i32,
    split:    Vec<i32>,

    /*
      | Input: X, optionally split
      | The split tensor is stored in CPU.
      |
      */
}

num_inputs!{Split, (1,2)}

num_outputs!{Split, (1,INT_MAX)}

inputs!{Split, 
    0 => ("input", "(*Tensor*): tensor to split"),
    1 => ("split", "(*Tensor`<int>`*): [OPTIONAL] list of output lengths (see also arg `split`)")
}

outputs!{Split, 
    0 => ("[output_0, output_1, ...]", "(*Tensor*): output tensor")
}

args!{Split, 
    0 => ("axis", "(*int*): axis to split on"),
    1 => ("add_axis", "*(type: int)* Pass non-zero integer to remove the axis specified in `axis` to all input tensors."),
    2 => ("split", "(*Tuple(int)*): length of each output"),
    3 => ("order", "(*string*): order of dimensions of input and output blobs; either NCHW or NHWC")
}

inherit_onnx_schema!{Split}

tensor_inference_function!{Split, TensorInferenceForSplit}

device_inference_function!{Split, splitOpDevInfer}

impl<Context> SplitOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            split_(this->template GetRepeatedArgument<int>("split")) 

        CAFFE_ENFORCE(
            !(OperatorStorage::HasArgument("axis") &&
              OperatorStorage::HasArgument("order")),
            "You shouldn't specify both the dim to split, and the order "
            "in the case of 4-D images.");
        if (OperatorStorage::HasArgument("axis")) {
          axis_ = this->template GetSingleArgument<int>("axis", -1);
          // only exists for computing the gradient of a Concat with 'add_axis'
          add_axis_ = this->template GetSingleArgument<int>("add_axis", 0);
        } else {
          axis_ = GetDimFromOrderString(
              this->template GetSingleArgument<string>("order", "NCHW"));
          add_axis_ = 0;
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
      int canonical_axis = input.canonical_axis_index(axis_);
      CAFFE_ENFORCE_LT(
          canonical_axis, input.dim(), "Axis not in input ndim range.");
      const int input_channels = input.dim32(canonical_axis);
      const int* axis_data;
      vector<int> equal_split;
      if (InputSize() == kSplitOpInputSize) {
        // We obtain split from the input tensor.
        CAFFE_ENFORCE_EQ(
            split_.size(),
            0,
            "If you set split with an input blob, do not pass in "
            "split in the argument.");
        auto& split_tensor = this->template Input<Tensor>(1, CPU);
        CAFFE_ENFORCE_EQ(split_tensor.numel(), OutputSize());
        axis_data = split_tensor.template data<int>();
      } else if (split_.size() == 0) {
        CAFFE_ENFORCE_EQ(
            input_channels % OutputSize(),
            0,
            "If you did not specify split explicitly, the number of "
            "input channels should be divisible by the output size.");
        equal_split.resize(OutputSize(), input_channels / OutputSize());
        axis_data = equal_split.data();
      } else {
        // We obtain split from the parameters.
        CAFFE_ENFORCE_EQ(
            split_.size(),
            OutputSize(),
            "The number of splits specified should be equal to the "
            "number of outputs.");
        axis_data = split_.data();
      }

      CAFFE_ENFORCE_EQ(
          add_axis_ ? OutputSize()
                    : std::accumulate(axis_data, axis_data + OutputSize(), 0),
          input_channels,
          "Sum of split dimensions do not match: should be ",
          input_channels);
      vector<int64_t> output_dims(input.sizes().vec());
      int before = 1, after = 1;
      for (int i = 0; i < canonical_axis; ++i) {
        before *= input.dim32(i);
      }
      for (int i = canonical_axis + 1; i < input.dim(); ++i) {
        after *= input.dim32(i);
      }
      if (add_axis_) {
        output_dims.erase(output_dims.begin() + canonical_axis);
      }
      size_t input_offset = 0;
      for (int i = 0; i < OutputSize(); ++i) {
        auto* output = Output(i);
        auto axis_dim = add_axis_ ? 1 : axis_data[i];
        if (!add_axis_) {
          output_dims[canonical_axis] = axis_data[i];
        }
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

///------------------------------------------------------------

#[test] fn split_by_lengths_op_example1() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "SplitByLengths",
        ["input", "lengths"],
        ["output_0","output_1","output_2"],
        axis=0
    )

    workspace.FeedBlob("input", np.random.randint(10, size=(9)))
    workspace.FeedBlob("lengths", np.array([3,2,4], dtype=np.int32))
    print("input:", workspace.FetchBlob("input"))
    print("lengths:", workspace.FetchBlob("lengths"))
    workspace.RunOperatorOnce(op)
    print("output_0:", workspace.FetchBlob("output_0"))
    print("output_1:", workspace.FetchBlob("output_1"))
    print("output_2:", workspace.FetchBlob("output_2"))

    input: [2 2 6 6 6 0 5 7 4]
    lengths: [3 2 4]
    output_0: [2 2 6]
    output_1: [6 6]
    output_2: [0 5 7 4]

    */
}

#[test] fn split_by_lengths_op_example2() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "SplitByLengths",
        ["input", "lengths"],
        ["output_0","output_1","output_2"],
        axis=0,
        use_scaling_lengths=true,
    )

    workspace.FeedBlob("input", np.random.randint(10, size=(9)))
    workspace.FeedBlob("lengths", np.array([1,1,1], dtype=np.int32))
    print("input:", workspace.FetchBlob("input"))
    print("lengths:", workspace.FetchBlob("lengths"))
    print("output_0:", workspace.FetchBlob("output_0"))
    print("output_1:", workspace.FetchBlob("output_1"))
    print("output_2:", workspace.FetchBlob("output_2"))

    input: [2 2 6 6 6 0 5 7 4]
    lengths: [1 1 1]
    output_0: [2 2 6]
    output_1: [6 6 6]
    output_2: [5 7 4]
    */
}

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

    context: Context,

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

///-------------------------------------------------------
#[test] fn concat_op_example1() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Concat",
        ["X1",  "X2"],
        ["Y", "split_info"],
        axis=0
    )

    workspace.FeedBlob("X1", np.array([[1,2],[3,4]]))
    workspace.FeedBlob("X2", np.array([[5,6]]))
    print("X1:", workspace.FetchBlob("X1"))
    print("X2:", workspace.FetchBlob("X2"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))
    print("split_info:", workspace.FetchBlob("split_info"))


    X1: [[1 2]
     [3 4]]
    X2: [[5 6]]
    Y: [[1 2]
     [3 4]
     [5 6]]
    split_info: [2 1]
    */
}

#[test] fn concat_op_example2() {

    todo!();
    /*

    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Concat",
        ["X1",  "X2"],
        ["Y", "split_info"],
        add_axis=1,
        axis=3
    )

    workspace.FeedBlob("X1", np.random.randint(10, size=(1, 1, 5, 5))) // NCHW
    workspace.FeedBlob("X2", np.random.randint(10, size=(1, 1, 5, 5))) // NCHW
    print("X1:", workspace.FetchBlob("X1"))
    print("X2:", workspace.FetchBlob("X2"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))
    print("split_info:", workspace.FetchBlob("split_info"))

    X1: [[[[1 8 3 9 0]
       [6 4 6 5 6]
       [3 9 1 9 9]
       [5 1 0 7 7]
       [9 4 0 0 9]]]]
    X2: [[[[7 0 2 6 1]
       [3 9 4 0 3]
       [5 3 8 9 4]
       [3 4 2 1 0]
       [0 8 8 8 1]]]]
    Y: [[[[[1 8 3 9 0]
        [7 0 2 6 1]]

       [[6 4 6 5 6]
        [3 9 4 0 3]]

       [[3 9 1 9 9]
        [5 3 8 9 4]]

       [[5 1 0 7 7]
        [3 4 2 1 0]]

       [[9 4 0 0 9]
        [0 8 8 8 1]]]]]
    split_info: [1 1]

    */
}

/**
  | Concatenate a list of tensors into a
  | single tensor. Similar functionality
  | to
  | 
  | Numpy's [concatenate](https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html)
  | function. The `axis` argument specifies
  | what axis along which the arrays will
  | be concatenated.
  | 
  | When set to non-zero (default=0), the
  | `add_axis` argument adds the axis specified
  | in `axis` to all input tensors.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.cc
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.h
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ConcatOp<Context> {

    context: Context,

    axis:     i32,
    add_axis: i32,

    /*
      | Input: a number of tensors. Output:
      | Y, split
      | 
      | The split are stored in CPU.
      |
      */
}

num_inputs!{Concat, (1,INT_MAX)}

num_outputs!{Concat, 2}

inputs!{Concat, 
    0 => ("X1, X2, ...", "*(type: Tensor`<float>`)* List of input tensors.")
}

outputs!{Concat, 
    0 => ("concat_result", "*(type: Tensor`<float>`)* Concatenated tensor."),
    1 => ("split_info", "*(type: Tensor`<int>`)* The dimensions of the inputs.")
}

args!{Concat, 
    0 => ("axis", "*(type: int; default: -1)* Axis to concatenate on."),
    1 => ("order", "*(type: string; default='NCHW')* Order of blob dimensions. Concats on the C dimension."),
    2 => ("add_axis", "*(type: int)* Pass non-zero integer to add the axis specified in `axis` to all input tensors.")
}

inherit_onnx_schema!{Concat}

tensor_inference_function!{Concat, OpSchema::NeedsAllInputShapes(TensorInferenceForConcat)}

cost_inference_function!{Concat, CostInferenceForConcat}

device_inference_function!{Concat, concatOpDevInfer}

impl<Context> ConcatOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        CAFFE_ENFORCE(
            !(OperatorStorage::HasArgument("axis") &&
              OperatorStorage::HasArgument("order")),
            "You shouldn't specify both the dim to concat, and the order "
            "in the case of 4-D images.");
        if (OperatorStorage::HasArgument("axis")) {
          axis_ = this->template GetSingleArgument<int>("axis", -1);
          add_axis_ = this->template GetSingleArgument<int>("add_axis", 0);
        } else {
          axis_ = GetDimFromOrderString(
              this->template GetSingleArgument<string>("order", "NCHW"));
          add_axis_ = 0;
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto* output = Output(0);

      // We can override default options(Context::GetDeviceType())
      // by explicitly passing in device type we want
      Tensor* split = Output(
          1, std::vector<int64_t>(1, InputSize()), at::dtype<int>().device(CPU));
      int* axis_data = split->template mutable_data<int>();
      auto& input_zero = Input(0);
      int adj_size = input_zero.dim() + (add_axis_ ? 1 : 0);
      int canonical_axis = canonical_axis_index_(axis_, adj_size);
      CAFFE_ENFORCE_LT(canonical_axis, adj_size, "Axis not in input ndim range.");
      for (int i = 1; i < InputSize(); ++i) {
        CAFFE_ENFORCE_EQ(
            Input(i).dtype(),
            input_zero.dtype(),
            "All inputs must have the same type, expected: ",
            input_zero.dtype().name(),
            " but got: ",
            Input(i).dtype().name(),
            " for input: ",
            i);
      }

      int before = 1, after = 1;
      vector<int64_t> output_dims(input_zero.sizes().vec());
      for (int i = 0; i < input_zero.dim(); ++i) {
        if (i == canonical_axis && !add_axis_) {
          continue;
        }
        int dim = input_zero.dim32(i);
        if (i < canonical_axis) {
          before *= dim;
        } else { // i > canonical_axis || i == canonical_axis && add_axis_
          after *= dim;
        }
        // check the input dims are compatible.
        for (int j = 1; j < InputSize(); ++j) {
          int dim_j = Input(j).dim32(i);
          CAFFE_ENFORCE_EQ(
              dim,
              dim_j,
              "Expect dimension = ",
              dim,
              " got ",
              dim_j,
              " at axis = ",
              i,
              " for input: ",
              j,
              ". The input tensors can only have different dimensions "
              "when arg 'add_axis' = 0 and along the axis = ",
              canonical_axis,
              " <",
              Input(0).sizes(),
              "> vs <",
              Input(j).sizes(),
              ">.");
        }
      }

      int output_channels = 0;
      for (int i = 0; i < InputSize(); ++i) {
        axis_data[i] = add_axis_ ? 1 : Input(i).dim32(canonical_axis);
        output_channels += axis_data[i];
      }
      if (add_axis_) {
        output_dims.insert(output_dims.begin() + canonical_axis, output_channels);
      } else {
        output_dims[canonical_axis] = output_channels;
      }
      output->Resize(output_dims);
      size_t output_offset = 0;
      for (int i = 0; i < InputSize(); ++i) {
        auto& input = Input(i);
        auto axis_dim = add_axis_ ? 1 : input.dim32(canonical_axis);
        math::CopyMatrix<Context>(
            input.itemsize(),
            before,
            axis_dim * after,
            input.raw_data(),
            axis_dim * after,
            static_cast<char*>(output->raw_mutable_data(input_zero.dtype())) +
                output_offset,
            output_channels * after,
            &context_,
            input_zero.dtype().copy());
        output_offset += axis_dim * after * input.itemsize();
      }
      return true;
        */
    }
}

#[inline] pub fn split_op_dev_infer(def: &OperatorDef) -> (Vec<DeviceOption>, Vec<DeviceOption>) {
    
    todo!();
    /*
        auto op_device =
          def.has_device_option() ? def.device_option() : DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), op_device);
      vector<DeviceOption> out_dev(def.output_size(), op_device);

      // If we obtain split from input tensor, then 2nd input's type is always CPU.
      if (def.input_size() == SplitOp<CPUContext>::kSplitOpInputSize) {
        CAFFE_ENFORCE_GT(in_dev.size(), 1);
        in_dev[1] = DeviceOption();
      }
      return std::make_pair(in_dev, out_dev);
    */
}

#[inline] pub fn tensor_inference_for_split(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        auto ret_invalid_shape = [&def]() {
        vector<TensorShape> out(def.output().size());
        for (auto& out_ts : out) {
          out_ts.set_unknown_shape(true);
        }
        return out;
      };
      // We only support shape inference of Split with 1 input
      if (def.input_size() != 1 || in.empty() || in.front().unknown_shape()) {
        return ret_invalid_shape();
      } else if (def.output_size() == 0) {
        return vector<TensorShape>();
      }
      ArgumentHelper helper(def);
      const int axis = helper.HasArgument("axis")
          ? helper.GetSingleArgument<int>("axis", -1)
          : GetDimFromOrderString(
                helper.GetSingleArgument<string>("order", "NCHW"));
      const int add_axis = helper.HasArgument("axis")
          ? helper.GetSingleArgument<int>("add_axis", 0)
          : 0;
      const auto& input = in[0];
      const int canonical_axis = canonical_axis_index_(axis, input.dims_size());
      const int input_channels = input.dims(canonical_axis);
      auto split = helper.GetRepeatedArgument<int>("split");
      // Equally split the input into outputs
      const int output_size = def.output_size();
      if (def.input_size() == caffe2::SplitOp<CPUContext>::kSplitOpInputSize) {
        if (!split.empty()) {
          LOG(WARNING) << "If you set split with an input blob, do not pass in "
                          "split in the argument.";
        }
        // We cannot infer output shape until we see the value of split input
        return ret_invalid_shape();
      } else if (split.empty()) {
        if (input_channels % output_size != 0) {
          LOG(WARNING) << "Input channels (" << input_channels
                       << ") should be divisible by number of outputs ("
                       << output_size << ")";
          return ret_invalid_shape();
        }
        split.resize(output_size, input_channels / output_size);
      } else if (split.size() != output_size) {
        LOG(WARNING) << "`split` size (" << split.size()
                     << ") should be equal to output size (" << output_size << ")";
        return ret_invalid_shape();
      }

      // Check validity of the split
      const int total_channels = add_axis
          ? def.output_size()
          : std::accumulate(split.begin(), split.begin() + output_size, 0);
      if (total_channels != input_channels) {
        LOG(WARNING) << "Input channels (" << input_channels
                     << ") is not equal to total output channels ("
                     << total_channels << ")";
        return ret_invalid_shape();
      }

      vector<int> output_dims(input.dims().begin(), input.dims().end());
      if (add_axis) {
        output_dims.erase(output_dims.begin() + canonical_axis);
      }
      vector<TensorShape> output_shapes;
      for (int i = 0; i < output_size; ++i) {
        if (!add_axis) {
          output_dims[canonical_axis] = split[i];
        }
        output_shapes.emplace_back(
            CreateTensorShape(output_dims, input.data_type()));
      }
      return output_shapes;
    */
}

register_cpu_operator!{
    Split, 
    SplitOp<CPUContext>
}

register_cpu_operator!{
    SplitByLengths, 
    SplitByLengthsOp<CPUContext>
}

#[inline] pub fn cost_inference_for_concat(
    def:   &OperatorDef, 
    input: &Vec<TensorShape>) -> OpSchemaCost 
{
    
    todo!();
    /*
        ArgumentHelper helper(def);
      const int axis = helper.HasArgument("axis")
          ? helper.GetSingleArgument<int>("axis", -1)
          : GetDimFromOrderString(
                helper.GetSingleArgument<string>("order", "NCHW"));
      bool add_axis = helper.GetSingleArgument<int>("add_axis", 0) != 0;
      int adj_size = in[0].dims_size() + (add_axis ? 1 : 0);
      const int canonical_axis = canonical_axis_index_(axis, adj_size);
      CAFFE_ENFORCE_LT(canonical_axis, adj_size, "Axis not in input ndim range.");
      CAFFE_ENFORCE_GT(in.size(), 0);
      vector<int> out_shape(in[0].dims().begin(), in[0].dims().end());
      if (add_axis) {
        out_shape.insert(out_shape.begin() + canonical_axis, in.size());
      } else {
        for (size_t i = 1; i < in.size(); ++i) {
          out_shape[canonical_axis] += in[i].dims(canonical_axis);
        }
      }
      uint64_t nElemRead = 1;
      for (int i = 0; i < in.size(); ++i) {
        nElemRead += nElemFromDim(in[i]);
      }
      int size = 1;
      for (auto& s : out_shape) {
        size *= s;
      }

      struct OpSchema::Cost cost;
      cost.flops = 0;
      cost.bytes_read = nElemRead * sizeof(in[0].data_type());
      cost.bytes_written = size * sizeof(in[0].data_type());
      cost.params_bytes = 0;
      return cost;
    */
}

#[inline] pub fn concat_op_dev_infer(def: &OperatorDef) -> (Vec<DeviceOption>,Vec<DeviceOption>) {
    
    todo!();
    /*
        auto op_device =
          def.has_device_option() ? def.device_option() : DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), op_device);
      vector<DeviceOption> out_dev(def.output_size(), op_device);

      // 2nd output's type is always CPU irrespective of op's device option.
      CAFFE_ENFORCE_GT(out_dev.size(), 1);
      out_dev[1] = DeviceOption();
      return std::make_pair(in_dev, out_dev);
    */
}

#[inline] pub fn tensor_inference_for_concat(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        ArgumentHelper helper(def);
      const int axis = helper.HasArgument("axis")
          ? helper.GetSingleArgument<int>("axis", -1)
          : GetDimFromOrderString(
                helper.GetSingleArgument<string>("order", "NCHW"));
      bool add_axis = helper.GetSingleArgument<int>("add_axis", 0) != 0;
      int adj_size = in[0].dims_size() + (add_axis ? 1 : 0);
      const int canonical_axis = canonical_axis_index_(axis, adj_size);
      CAFFE_ENFORCE_LT(canonical_axis, adj_size, "Axis not in input ndim range.");
      CAFFE_ENFORCE_GT(in.size(), 0);
      vector<int> split_shape(1, in.size());
      vector<int> out_shape(in[0].dims().begin(), in[0].dims().end());
      if (add_axis) {
        for (int i = 1; i < in.size(); ++i) {
          CAFFE_ENFORCE_EQ(
              in[0].dims().size(),
              in[i].dims().size(),
              "All inputs of Concat should have same dims when add_axis = 1. "
              "Got different sizes for inputs 0 and ",
              i);
          for (int j = 0; j < in[0].dims().size(); ++j) {
            CAFFE_ENFORCE_EQ(
                in[0].dims(j),
                in[i].dims(j),
                "All inputs of Concat should have same dims when add_axis = 1. "
                "Got different dims for inputs 0 and ",
                i,
                ". At dim: ",
                j);
          }
        }
        out_shape.insert(out_shape.begin() + canonical_axis, in.size());
      } else {
        for (int i = 1; i < in.size(); ++i) {
          CAFFE_ENFORCE(
              in[0].dims_size() == in[i].dims_size() ||
                  (canonical_axis == in[0].dims_size() - 1 &&
                   in[0].dims_size() == in[i].dims_size() + 1),
              "All inputs of Concat should have same dims except "
              "canonical_axis dim that is equal to ",
              canonical_axis,
              "Got different sizes for inputs 0 and ",
              i);
          for (int j = 0; j < in[0].dims_size(); ++j) {
            if (j == canonical_axis) {
              continue;
            }
            CAFFE_ENFORCE_EQ(
                in[0].dims(j),
                in[i].dims(j),
                "All inputs of Concat should have same dims except "
                "canonical_axis dim that is equal to ",
                canonical_axis,
                "Got different dims for inputs 0 and ",
                i,
                ". At dim: ",
                j);
          }
        }

        for (int i = 1; i < in.size(); ++i) {
          out_shape[canonical_axis] += in[i].dims(canonical_axis);
        }
      }
      if (def.output_size() == 1) {
        return vector<TensorShape>{CreateTensorShape(out_shape, in[0].data_type())};
      }
      return vector<TensorShape>{
          CreateTensorShape(out_shape, in[0].data_type()),
          CreateTensorShape(split_shape, TensorProto::INT32)};
    */
}

register_cpu_operator!{
    Concat, 
    ConcatOp<CPUContext>
}

pub struct GetSplitGradient;

impl GetGradientDefs for GetSplitGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> output_grads;
        for (int i = 0; i < def_.output_size(); ++i) {
          if (!GradOut(i).IsEmpty()) {
            output_grads.push_back(GO(i));
          }
        }
        if (output_grads.empty()) {
          return {};
        }
        return SingleGradientDef(
            "Concat",
            "",
            output_grads,
            vector<string>{GI(0), "_" + GI(0) + "_dims"});
        */
    }
}

register_gradient!{Split, GetSplitGradient}
register_gradient!{DepthSplit, GetSplitGradient}
register_gradient!{SplitByLengths, GetSplitGradient}

pub struct GetConcatGradient;

impl GetGradientDefs for GetConcatGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            if (GradOut(0).IsEmpty()) {
          return {};
        }
        vector<string> grads;
        for (int i = 0; i < def_.input_size(); ++i) {
          grads.push_back(GI(i));
        }
        return SingleGradientDef("Split", "", vector<string>{GO(0), O(1)}, grads);
        */
    }
}

register_gradient!{Concat, GetConcatGradient}
register_gradient!{DepthConcat, GetConcatGradient}

register_cuda_operator!{Split, SplitOp<CUDAContext>}
register_cuda_operator!{Concat, ConcatOp<CUDAContext>}
register_cuda_operator!{SplitByLengths, SplitByLengthsOp<CUDAContext>}
