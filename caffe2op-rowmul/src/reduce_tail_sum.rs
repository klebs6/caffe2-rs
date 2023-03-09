crate::ix!();

/**
  | A hacky version
  | 
  | Reduce the tailing dimensions
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_SIMPLE_CTOR_DTOR("ReduceTailSumOp")]
pub struct ReduceTailSumOp<T,Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{ReduceTailSum, (1,1)}

num_outputs!{ReduceTailSum, 1}

inputs!{ReduceTailSum, 
    0 => ("mat", "The matrix")
}

outputs!{ReduceTailSum, 
    0 => ("output", "Output")
}

impl<T,Context> ReduceTailSumOp<T,Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& mat = Input(0);

        int N = mat.dim32(0);
        int block_size = mat.size_from_dim(1);

        auto* output = Output(0, {N}, at::dtype<T>());
        T* output_data = output->template mutable_data<T>();
        const T* mat_data = mat.template data<T>();

        for (int i = 0; i < N; i++) {
          output_data[i] = 0;
          size_t offset = i * block_size;
          for (int j = 0; j < block_size; j++) {
            output_data[i] += mat_data[offset + j];
          }
        }
        return true;
        */
    }
}

register_cpu_operator!{
    ReduceTailSum, 
    ReduceTailSumOp<f32, CPUContext>
}
