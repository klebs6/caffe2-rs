crate::ix!();

use crate::{
    DNNLowPOp,
    CopyOp,
    Workspace,
    OperatorDef,
    CPUContext,
};

// FIXME
pub type BatchPermutationFP32Op = CopyOp<CPUContext, CPUContext, CPUContext>;

pub struct BatchPermutationDNNLowPOp<T> {
    //USE_OPERATOR_FUNCTIONS(CPUContext);
    //USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, BatchPermutationFP32Op);
    base: DNNLowPOp<T, BatchPermutationFP32Op>,
}

register_cpu_operator_with_engine!{
    BatchPermutation,
    DNNLOWP,
    BatchPermutationDNNLowPOp<u8>
}

register_cpu_operator_with_engine!{
    Int8BatchPermutation,
    DNNLOWP,
    BatchPermutationDNNLowPOp<u8>
}

num_inputs!{Int8BatchPermutation, 2}

num_outputs!{Int8BatchPermutation, 1}

impl<T> BatchPermutationDNNLowPOp<T> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : BaseType(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            using namespace dnnlowp;

      this->ParseDNNLowPOperatorArguments_();

      // Choose quantization params
      in_qparams_[INPUT] =
          GetInputTensorQuantizationParamsOf(this, INPUT, qfactory_.get());

      const auto& X = InputTensorCPU_(INPUT);
      const auto& indices = Input(INDICES);
      auto* Y = OutputTensorCPU_(OUTPUT);

      CAFFE_ENFORCE(indices.ndim() == 1, "indices must be 1-d");
      CAFFE_ENFORCE(
          X.dim32(0) == indices.dim32(0),
          "X.dim32(0) must be equal to indices.dim32(0)",
          "(",
          X.dim32(0),
          " vs. ",
          indices.dim32(0),
          ")");
      CAFFE_ENFORCE_GT(X.dim32(0), 0);

      Y->ResizeLike(X);
      const T* X_data = X.template data<T>();
      const int* indices_data = indices.template data<int>();
      T* Y_data = Y->template mutable_data<T>();

      int N = X.dim32(0);
      int K = X.numel() / N;

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
      for (int i = 0; i < N; ++i) {
        int origIdx = i * K;
        int permuteIdx = indices_data[i] * K;
        std::memcpy(Y_data + origIdx, X_data + permuteIdx, K * sizeof(T));
      }

      // Even if there is a pre-chosen quantization parameters for the output,
      // it is ignored because batch permutation output quantization should be same
      // as the input.
      PropagateOutputTensorQuantizationParams(this, 0, in_qparams_[INPUT]);

      return true;
        */
    }
}

input_tags!{
    BatchPermutationDNNLowPOp
    {
        Input,
        Indices
    }
}

output_tags!{
    BatchPermutationDNNLowPOp
    {
        Output
    }
}
