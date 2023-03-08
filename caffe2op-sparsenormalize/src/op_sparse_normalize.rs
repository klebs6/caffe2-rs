crate::ix!();

use crate::{
    OperatorStorage,
    CPUContext,
};

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SparseNormalizeOp<T,Context> {

    storage:       OperatorStorage,
    context:       Context,

    use_max_norm:  bool,
    norm:          f32,
    phantom:       PhantomData<T>,
}

/**
  | Given a sparse matrix, apply max_norm
  | or constant_norm sparse regularization.
  |
  */
register_cpu_operator!{
    SparseNormalize, 
    SparseNormalizeOp<float, CPUContext>
}

num_inputs!{SparseNormalize, (2,3)}

num_outputs!{SparseNormalize, 1}

inputs!{SparseNormalize, 
    0 => ("param",   "Parameters to be normalized"),
    1 => ("indices", "Sparse indices"),
    2 => ("grad",    "Gradient computed (optional - not used, this argument is for backwards compatibility)")
}

outputs!{SparseNormalize, 
    0 => ("output_param", "Normalized parameters")
}

args!{SparseNormalize, 
    0 => ("use_max_norm", "A bool variable to control whether to use max norm or constant norm. When use_max_norm = false, constant norm is used so that all the embedding vectors are scaled to have a L2 norm equals to A (see blow argument norm=A). If use_max_norm = true, max norm is used so that embedding is scaled so that its l2 norm is no larger than A. If an embedding's norm is less than A originally, the embedding is left unchanged.The default is True."),
    1 => ("norm",         "L2 norm of the embedding. The default is 1.0.")
}

enforce_one_to_one_inplace!{SparseNormalize}

should_not_do_gradient!{SparseNormalize}

/**
  | Given a sparse matrix, apply max_norm
  | or constant_norm sparse regularization.
  |
  */
register_cpu_operator!{Float16SparseNormalize, SparseNormalizeOp<c10::Half, CPUContext>}

num_inputs!{Float16SparseNormalize, (2,3)}

num_outputs!{Float16SparseNormalize, 1}

inputs!{Float16SparseNormalize, 
    0 => ("param",    "Parameters to be normalized"),
    1 => ("indices",  "Sparse indices"),
    2 => ("grad",     "Gradient computed (optional - not used, this argument is for backwards compatibility)")
}

outputs!{Float16SparseNormalize, 
    0 => ("output_param", "Normalized parameters")
}

args!{Float16SparseNormalize, 
    0 => ("use_max_norm", "A bool variable to control whether to use max norm or constant norm. When use_max_norm = false, constant norm is used so that all the embedding vectors are scaled to have a L2 norm equals to A (see blow argument norm=A). If use_max_norm = true, max norm is used so that embedding is scaled so that its l2 norm is no larger than A. If an embedding's norm is less than A originally, the embedding is left unchanged. The default is True."),
    1 => ("norm",         "L2 norm of the embedding. The default is 1.0.")
}

enforce_one_to_one_inplace!{Float16SparseNormalize}

should_not_do_gradient!{Float16SparseNormalize}

input_tags!{
    SparseNormalizeOp {
        Param,
        Indices
    }
}

output_tags!{
    SparseNormalizeOp {
        OutputParam
    }
}


impl<T,Context> SparseNormalizeOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            use_max_norm_( this->template GetSingleArgument<bool>("use_max_norm", true)),
            norm_(this->template GetSingleArgument<float>("norm", 1.0)) 

        CAFFE_ENFORCE_GE(norm_, 0, "norm should be bigger than 0");
        */
    }
}

impl SparseNormalizeOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
          this, Input(INDICES));
        */
    }
    
    #[inline] pub fn do_run_with_type<SIndex>(&mut self) -> bool {
    
        todo!();
        /*
            const auto* indices = Input(INDICES).template data<SIndex>();
      const auto* paramIn = Input(PARAM).template data<float>();
      auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<float>();
      const float kEps = 1e-12f;

      // n: number of sparse embeddings to be normalized
      auto n = Input(INDICES).numel();
      if (n == 0) {
        return true;
      }

      // embedding length, e.g. 32, 64, 128
      auto block_size = Input(PARAM).size_from_dim(1);
      for (int i = 0; i < n; ++i) {
        auto idx = indices[i];
        auto offsetIdx = idx * block_size;
        ConstEigenVectorMap<float> xVec(paramIn + offsetIdx, block_size);
        auto norm = xVec.template lpNorm<2>();

        if (use_max_norm_ && norm <= norm_) {
          continue;
        }

        math::Scale(
            block_size,
            norm_ / (norm + kEps),
            paramOut + offsetIdx,
            paramOut + offsetIdx,
            &context_);
      }
      return true;
        */
    }
}


impl SparseNormalizeOp<f16, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
          this, Input(INDICES));
        */
    }
}

#[inline] pub fn float_16to_float_ref(
    input:  *const f16,
    out:    *mut f32,
    n:      usize)  {
    
    todo!();
    /*
        for (size_t i = 0; i < N; ++i) {
        out[i] = in[i];
      }
    */
}

impl SparseNormalizeOp<f16, CPUContext> {

    #[inline] pub fn do_run_with_type<SIndex>(&mut self) -> bool {
    
        todo!();
        /*
            const auto* indices = Input(INDICES).template data<SIndex>();
      const auto* paramIn = Input(PARAM).template data<c10::Half>();
      auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<c10::Half>();
      const float kEps = 1e-12f;

      // n: number of sparse embeddings to be normalized
      auto n = Input(INDICES).numel();
      if (n == 0) {
        return true;
      }
      // embedding length, e.g. 32, 64, 128
      auto block_size = Input(PARAM).size_from_dim(1);
      vector<float> row_vec_fp32(block_size);
      auto out_data = row_vec_fp32.data();
      for (int i = 0; i < n; ++i) {
        auto idx = indices[i];
        auto offsetIdx = idx * block_size;
    #ifdef USE_FBGEMM
        if (GetCpuId().avx2()) {
          fbgemm::Float16ToFloat_avx2(
              reinterpret_cast<const fbgemm::float16*>(paramIn + offsetIdx),
              out_data,
              block_size);
        } else {
          Float16ToFloat_ref(paramIn + offsetIdx, out_data, block_size);
        }
    #else
        Float16ToFloat_ref(paramIn + offsetIdx, out_data, block_size);
    #endif
        ConstEigenVectorMap<float> xVec_fp32(row_vec_fp32.data(), block_size);
        float norm = xVec_fp32.template lpNorm<2>();
        if (use_max_norm_ && norm <= norm_) {
          continue;
        }
        auto Y = paramOut + offsetIdx;
        EigenVectorArrayMap<c10::Half>(Y, block_size) *=
            static_cast<float>(norm_ / (norm + kEps));
      }
      return true;
        */
    }
}
