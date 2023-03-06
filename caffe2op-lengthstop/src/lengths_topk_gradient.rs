crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LengthsTopKGradientOp<T,Context> {
    storage: OperatorStorage,
    context: Context,
    k:       i32,
    phantom: PhantomData<T>,
}

register_cpu_operator!{LengthsTopKGradient, LengthsTopKGradientOp<float, CPUContext>}

num_inputs!{LengthsTopKGradient, 3}

num_outputs!{LengthsTopKGradient, 1}

input_tags!{
    LengthsTopKGradientOp {
        LengthIn,
        IndicesIn,
        DerTopkIn
    }
}

output_tags!{
    LengthsTopKGradientOp {
        DerXOut
    }
}

impl<T,Context> LengthsTopKGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "k", k_, -1) 

        CAFFE_ENFORCE_GE(k_, 1, "k argument must be >= 1");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input_len = Input(LENGTH_IN);
      int N = input_len.numel();
      auto& input_indices = Input(INDICES_IN);
      CAFFE_ENFORCE_GE(input_indices.dim(), 2, "input dim must be >= 2");
      CAFFE_ENFORCE_EQ(
          input_indices.numel(), N * k_, "input_indices shape is not correct");
      auto& input_topk = Input(DER_TOPK_IN);
      CAFFE_ENFORCE_EQ(
          input_topk.numel(), N * k_, "input_topk shape is not correct");

      const int* input_len_data = input_len.template data<int>();
      const int* input_indices_data = input_indices.template data<int>();
      const T* input_topk_data = input_topk.template data<T>();

      int num_indices = 0;
      for (int i = 0; i < N; i++) {
        num_indices += input_len_data[i];
      }
      auto* X_out = Output(DER_X_OUT, {num_indices}, at::dtype<T>());
      T* X_out_data = X_out->template mutable_data<T>();
      math::Set<T, Context>(num_indices, 0.0, X_out_data, &context_);

      int index_offset = 0;
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < std::min(input_len_data[i], k_); j++) {
          int cur_index = index_offset + input_indices_data[i * k_ + j];
          CAFFE_ENFORCE_LT(
              cur_index, num_indices, "cur_index should be less than num_indices");
          X_out_data[cur_index] = input_topk_data[i * k_ + j];
        }
        index_offset += input_len_data[i];
      }

      return true;
        */
    }
}
