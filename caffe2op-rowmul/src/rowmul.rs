/*!
  | Given a matrix A and column vector w,
  | the output is the multiplication of
  | row i of A and element i of w, e.g. C[i][j]
  | = A[i][j] * w[i].
  | 
  | This operator should be deprecated
  | when the gradient operator of Mul with
  | broadcast is implemented.
  |
  */

crate::ix!();

register_cpu_operator!{
    RowMul, 
    RowMulOp<f32, CPUContext>
}

/**
  | A hacky version of Mul with broadcast
  | 
  | RowMul([mat, w], [output])
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_SIMPLE_CTOR_DTOR("RowMulOp")]
pub struct RowMulOp<T,Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{RowMul, (2,2)}

num_outputs!{RowMul, 1}

inputs!{RowMul, 
    0 => ("mat", "The matrix"),
    1 => ("w", "The column vector")
}

outputs!{RowMul, 
    0 => ("output", "Output")
}

impl<T,Context> RowMulOp<T,Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& mat = Input(0);
        auto& w = Input(1);

        auto* output = Output(0, mat.sizes(), at::dtype<T>());
        T* output_data = output->template mutable_data<T>();
        const T* mat_data = mat.template data<T>();
        const T* w_data = w.template data<T>();

        // Dimension checking
        CAFFE_ENFORCE_EQ(
            w.numel(),
            mat.dim32(0),
            "Length of w should be equal to the first dim of mat");

        auto block_size = mat.size_from_dim(1);
        for (int i = 0; i < w.numel(); i++) {
          size_t offset = i * block_size;
          for (int j = 0; j < block_size; j++) {
            output_data[offset + j] = mat_data[offset + j] * w_data[i];
          }
        }

        return true;
        */
    }
}
