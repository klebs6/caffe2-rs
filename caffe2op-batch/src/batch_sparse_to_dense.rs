crate::ix!();

#[test] fn batch_sparse_to_dense_op_example() {
    todo!();
    /*
    For example, with input:

      lengths = [2, 3, 1]
      indices = [0, 1, 2, 3, 4, 5]
      values =  [6, 7, 8, 9, 10, 11]
      dense_dim = 6
      default_value = 0

    The output is:

      output = [[6, 7, 0, 0, 0,  0],
                [0, 0, 8, 9, 10, 0],
                [0, 0, 0, 0, 0, 11]]

    after running this operator.
    */
}

/**
  | Convert sparse matrix representation
  | into dense matrix.
  | 
  | A sparse matrix is represented by `lengths`
  | vector, `indices` vector, and `values`
  | vector.
  | 
  | Each element in `lengths` vector (lengths[`i`])
  | represents the number of indices in
  | this batch (batch `i`).
  | 
  | With in each batch, `indices` should
  | not have duplicate number.
  |
  */
pub struct BatchSparseToDenseOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    dense_last_dim: i64,
    default_value:  T,

    // len_prefix_sum_ and len_prefix_tmp_ are buffers on the GPU. It is not used
    // in the CPUContext implementation.

    len_prefix_sum: Tensor, //{Context::GetDeviceType()};
    len_prefix_tmp: Tensor, //{Context::GetDeviceType()};
}

num_inputs!{BatchSparseToDense, (3,4)}

num_outputs!{BatchSparseToDense, 1}

disallow_input_filler!{BatchSparseToDense}

inputs!{BatchSparseToDense, 
    0 => ("lengths", "Flatten tensor, used to break down indices and values into per batch indices and values."),
    1 => ("indices", "Flatten tensor of total size = \\sum lengths, containing the indices "),
    2 => ("values", "Data tensor, dimension has to match `indices`"),
    3 => ("output_shape_inference", "Optional, a dense tensor whose shape define the output shape")
}

outputs!{BatchSparseToDense, 
    0 => ("dense", "2-D dense tensor, with 1st dim = len(lengths), 
        2nd dim = dense_last_dim in the arg list, the tensor is 
        of the same data type as `values`.  Missing values are filled with default_value")
}

args!{BatchSparseToDense, 
    0 => ("dense_last_dim", "Optional, output dense last dimension.  
        If both this argument and output_shape_inference are set, it should be 
        consistent with output_shape_inference's last dim"),
    1 => ("default_value", "Optional, missing values are filled with this value.  
        default_value = 0 when not set")
}

input_tags!{
    BatchSparseToDenseOp {
        Lengths,
        Indices,
        Values
    }
}

tensor_inference_function!{BatchSparseToDense, /*[](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      vector<long> output_dims;
      if (in.size() == 4) {
        const auto& inference_dims = GetDimsVector(in[3]);
        output_dims.insert(output_dims.end(), inference_dims.begin(), inference_dims.end());
        const int dense_last_dim = helper.GetSingleArgument<int>("dense_last_dim", 0);
        if(dense_last_dim > 0) {
          CAFFE_ENFORCE(
              output_dims.back() == dense_last_dim,
              "The last dim of output_shape_inference should be consistent with dense_last_dim");
        }
      } else {
        const int dense_last_dim = helper.GetSingleArgument<int>("dense_last_dim", 0);
        CAFFE_ENFORCE(
          dense_last_dim > 0,
          "dense_last_dim must be set when output shape inference is unavailable");
        const auto& lens_dims = GetDimsVector(in[0]);
        output_dims.insert(output_dims.end(), lens_dims[0]);
        output_dims.insert(output_dims.end(), dense_last_dim);
      }
      vector<TensorShape> out(1);
      out[0] = CreateTensorShape(output_dims, in[2].data_type());
      return out;
    }*/
}

impl<T,Context> BatchSparseToDenseOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int64_t, "dense_last_dim", dense_last_dim_, -1),
            OP_SINGLE_ARG(T, "default_value", default_value_, static_cast<T>(0))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& lengths = Input(LENGTHS);
        auto& indices = Input(INDICES);
        auto& values = Input(VALUES);

        CAFFE_ENFORCE_EQ(indices.numel(), values.numel());
        CAFFE_ENFORCE_EQ(lengths.dim(), 1);
        CAFFE_ENFORCE_EQ(indices.dim(), 1);

        const int64_t* lengths_data = lengths.template data<int64_t>();
        const int64_t* indices_data = indices.template data<int64_t>();
        const T* values_data = values.template data<T>();
        int64_t batch_size = lengths.numel();

        vector<int64_t> output_shape = {batch_size};
        if (InputSize() == 4) {
          auto& shaper = Input(3);
          CAFFE_ENFORCE_EQ(shaper.dim(), 2);
          if (dense_last_dim_ == -1) {
            dense_last_dim_ = shaper.size(1);
          } else {
            CAFFE_ENFORCE(
                dense_last_dim_ == shaper.size(1),
                "The last dim argument is not aligned with the shape input last dim");
          }
        } else {
          CAFFE_ENFORCE(dense_last_dim_ >= 1, "The last dim of dense must be >= 1");
        }
        output_shape.push_back(dense_last_dim_);
        auto* output = Output(0, output_shape, at::dtype<T>());
        T* output_data = output->template mutable_data<T>();
        math::Set(
            output->numel(),
            static_cast<T>(default_value_),
            output_data,
            &context_);

        FillInDenseValues(
            batch_size,
            indices.numel(),
            lengths_data,
            indices_data,
            values_data,
            output_data,
            &context_);

        return true;
        */
    }
}
