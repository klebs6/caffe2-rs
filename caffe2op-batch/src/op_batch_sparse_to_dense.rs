crate::ix!();

use crate::{
    OperatorDef,
    CPUContext,
    Tensor,
    OperatorStorage,
    GradientMakerBase
};

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

input_tags!{
    BatchSparseToDenseOp {
        Lengths,
        Indices,
        Values
    }
}

///---------------------------------------------------------

#[test] fn batch_dense_to_sparse_op_example() {

    todo!();

    /*
    For example, with input:

      lengths = [2, 3, 1]
      indices = [0, 1, 2, 3, 4, 5]
      output = [[6, 7, 0, 0, 0,  0],
                [0, 0, 8, 9, 10, 0],
                [0, 0, 0, 0, 0, 11]]

    The output is:

      values = [6, 7, 8, 9, 10, 11]

    after running this operator.
    */
}

/**
 | This Op is a inverse of BatchSparseToDenseOp.
 |
 | Basically, given a `lengths` vector, a `indices`
 | vector, and a dense matrix `dense`, output `value`
 | vector so that, along with `lengths` vector and
 | `indices` vector, forms a sparse representation of
 | the dense matrix.
 |
 | A sparse matrix is represented by `lengths`
 | vector, `indices` vector, and `values` vector. 
 |
 | Each element in `lengths` vector (lengths[`i`])
 | represents the number of indices in this batch
 | (batch `i`).
 |
 | With in each batch, `indices` should not have
 | duplicate number.
 |
 */
pub struct BatchDenseToSparseOp<T, Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    dense_last_dim: i64,

    // len_prefix_sum_ and len_prefix_tmp_ are buffers on the GPU. It is not used
    // in the CPUContext implementation.

    len_prefix_sum: Tensor, //{Context::GetDeviceType()};
    len_prefix_tmp: Tensor, //{Context::GetDeviceType()};
    phantom: PhantomData<T>,
}

register_cpu_operator!{BatchDenseToSparse, BatchDenseToSparseOp<f32, CPUContext>}

num_inputs!{BatchDenseToSparse, 3}

num_outputs!{BatchDenseToSparse, 1}

inputs!{BatchDenseToSparse, 
    0 => ("lengths", "Flatten lengths, Used to break down indices into per batch indices"),
    1 => ("indices", "Flatten indices, tensor of total size = \\sum lengths, containing the indices "),
    2 => ("dense", "dense 2-D tensor, first dim = len(lengths), last dim > Any(indices)")
}

outputs!{BatchDenseToSparse, 
    0 => ("values", "Values, tensor of the same size as `indices` and same data type as dense tensor.")
}

impl<T,Context> BatchDenseToSparseOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& lengths = Input(LENGTHS);
        auto& indices = Input(INDICES);
        auto& dense = Input(DENSE);

        CAFFE_ENFORCE_EQ(lengths.dim(), 1);
        CAFFE_ENFORCE_EQ(indices.dim(), 1);
        CAFFE_ENFORCE_EQ(dense.dim(), 2);
        const int64_t* lengths_data = lengths.template data<int64_t>();
        const int64_t* indices_data = indices.template data<int64_t>();
        const T* dense_data = dense.template data<T>();

        int64_t batch_size = lengths.numel();

        CAFFE_ENFORCE_EQ(batch_size, dense.size(0));
        dense_last_dim_ = dense.size(1);
        vector<int64_t> output_shape = indices.sizes().vec();
        auto* output = Output(0, output_shape, at::dtype<T>());
        T* output_data = output->template mutable_data<T>();

        FillInSparseValues(
            batch_size,
            indices.numel(),
            lengths_data,
            indices_data,
            dense_data,
            output_data,
            &context_);

        return true;
        */
    }
}

input_tags!{
    BatchDenseToSparseOp {
        Lengths,
        Indices,
        Dense
    }
}

impl BatchSparseToDenseOp<f32, CPUContext> {

    #[inline] pub fn fill_in_dense_values(
        &mut self, 
        batch_size:      i64,
        indice_lengths:  i64,
        lengths_data:    *const i64,
        indices_data:    *const i64,
        values_data:     *const f32,
        output_data:     *mut f32,
        context:         *mut CPUContext)  
    {
        
        todo!();
        /*
            int64_t lengths_sum = 0;
      math::Sum<int64_t, CPUContext>(
          batch_size, lengths_data, &lengths_sum, &context_);
      CAFFE_ENFORCE_EQ(lengths_sum, indice_lengths);

      int64_t k = 0;
      for (int64_t i = 0; i < batch_size; ++i) {
        for (int64_t j = 0; j < lengths_data[i]; ++j) {
          CAFFE_ENFORCE(
              indices_data[k] < dense_last_dim_,
              "An indice (",
              indices_data[k],
              ") is larger then last dim of dense (",
              dense_last_dim_,
              ").");
          output_data[i * dense_last_dim_ + indices_data[k]] = values_data[k];
          k += 1;
        }
      }
        */
    }
}

impl BatchDenseToSparseOp<f32, CPUContext> {

    #[inline] pub fn fill_in_sparse_values(
        &mut self, 
        batch_size:         i64,
        indice_lengths:     i64,
        lengths_data:       *const i64,
        indices_data:       *const i64,
        dense_data:         *const f32,
        output_data:        *mut f32,
        context:            *mut CPUContext)  
    {
        todo!();
        /*
            int64_t lengths_sum = 0;
      math::Sum<int64_t, CPUContext>(
          batch_size, lengths_data, &lengths_sum, &context_);
      CAFFE_ENFORCE_EQ(lengths_sum, indice_lengths);

      int64_t k = 0;
      for (int64_t i = 0; i < batch_size; ++i) {
        for (int64_t j = 0; j < lengths_data[i]; ++j) {
          CAFFE_ENFORCE(
              indices_data[k] < dense_last_dim_,
              "An indice (",
              indices_data[k],
              ") is larger then last dim of dense (",
              dense_last_dim_,
              ").");
          output_data[k] = dense_data[i * dense_last_dim_ + indices_data[k]];
          k += 1;
        }
      }
        */
    }
}

register_cpu_operator!{BatchSparseToDense, BatchSparseToDenseOp<f32, CPUContext>}

pub struct GetBatchSparseToDenseGradient;

impl GetGradientDefs for GetBatchSparseToDenseGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "BatchDenseToSparse",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(2)});
        */
    }
}

pub struct GetBatchDenseToSparseGradient;

impl GetGradientDefs for GetBatchDenseToSparseGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "BatchSparseToDense",
            "",
            vector<string>{I(0), I(1), GO(0), I(2)},
            vector<string>{GI(2)});
        */
    }
}

register_gradient!{BatchSparseToDense, GetBatchSparseToDenseGradient}

register_gradient!{BatchDenseToSparse, GetBatchDenseToSparseGradient}
