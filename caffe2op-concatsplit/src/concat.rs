crate::ix!();

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
