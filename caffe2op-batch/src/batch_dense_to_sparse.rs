crate::ix!();

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

input_tags!{
    BatchDenseToSparseOp {
        Lengths,
        Indices,
        Dense
    }
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

