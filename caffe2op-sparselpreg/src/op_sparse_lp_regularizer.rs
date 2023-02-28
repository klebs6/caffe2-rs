crate::ix!();

use crate::{
    OperatorStorage,
    CPUContext,
};

/**
  | Given a sparse matrix, apply Lp regularization.
  | 
  | Currently only L1 and L2 are implemented.
  |
  */
pub struct SparseLpRegularizerOp<T,Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage:     OperatorStorage,
    context:     Context,

    p:           f32,
    reg_lambda:  f32,
    phantom:     PhantomData<T>,
}

input_tags!{
    SparseLpRegularizerOp {
        Param,
        Indices
    }
}

output_tags!{
    SparseLpRegularizerOp {
        OutputParam
    }
}

register_cpu_operator!{
    SparseLpRegularizer,
    SparseLpRegularizerOp<f32, CPUContext>}

num_inputs!{SparseLpRegularizer, (2,3)}

num_outputs!{SparseLpRegularizer, 1}

inputs!{SparseLpRegularizer, 
    0 => ("param",   "Parameters to be regularized"),
    1 => ("indices", "Sparse indices"),
    2 => ("grad",    "Gradient computed (optional - not used, this argument is for backwards compatibility)")
}

outputs!{SparseLpRegularizer, 
    0 => ("output_param", "Regularized parameters")
}

args!{SparseLpRegularizer, 
    0 => ("p",          "Value of p in the Lp regularization to use. The default is 2.0."),
    1 => ("reg_lambda", "Value of lambda (multiplier for the regularization term). The default is 1e-5.")
}

enforce_one_to_one_inplace!{SparseLpRegularizer}

should_not_do_gradient!{SparseLpNorm}

impl<T,Context> SparseLpRegularizerOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            p_(this->template GetSingleArgument<float>("p", 2.0)),
            reg_lambda_( this->template GetSingleArgument<float>("reg_lambda", 1e-5)) 

        CAFFE_ENFORCE(
            p_ == 1.0 || p_ == 2.0,
            "Sparse Lp regularizer only implemented for p=1 or p=2.");
        CAFFE_ENFORCE_GT(
            reg_lambda_,
            0.0,
            "Lambda for sparse Lp regularizer must be greater than 0.");
        CAFFE_ENFORCE_LT(
            reg_lambda_,
            1.0,
            "Lambda for sparse Lp regularizer must be less than 1.");
        */
    }
}

impl SparseLpRegularizerOp<f32, CPUContext> {

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
      auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<float>();

      // n: number of sparse embeddings to be normalized
      auto n = Input(INDICES).numel();
      if (n == 0) {
        return true;
      }

      // embedding length, e.g. 32, 64, 128
      auto block_size = Input(PARAM).size_from_dim(1);

      if (p_ == 2.0) { // L2 regularization
    #ifdef LOG_FIRST_N
        LOG_FIRST_N(INFO, 3)
            << "Applying sparse L2 regularization with reg_lambda = "
            << reg_lambda_;
        LOG_FIRST_N(INFO, 3) << "L2 regularization input "
                             << paramOut[indices[0] * block_size];
    #endif // LOG_FIRST_N
        for (int i = 0; i < n; ++i) {
          auto idx = indices[i];
          auto offsetIdx = idx * block_size;
          // Should probably be rewritten using Eigen.
          for (int j = 0; j < block_size; j++) {
            paramOut[offsetIdx + j] = paramOut[offsetIdx + j] * (1 - reg_lambda_);
          }
        }
    #ifdef LOG_FIRST_N
        LOG_FIRST_N(INFO, 3) << "L2 regularization output "
                             << paramOut[indices[0] * block_size];
    #endif // LOG_FIRST_N
      } else if (p_ == 1.0) { // L1 regularization
    #ifdef LOG_FIRST_N
        LOG_FIRST_N(INFO, 3)
            << "Applying sparse L1 regularization with reg_lambda = "
            << reg_lambda_;
        LOG_FIRST_N(INFO, 3) << "L1 regularization input "
                             << paramOut[indices[0] * block_size];
    #endif // LOG_FIRST_N
        for (int i = 0; i < n; ++i) {
          auto idx = indices[i];
          auto offsetIdx = idx * block_size;

          for (int j = 0; j < block_size; j++) {
            // I assume this can be sped up significantly.
            if (paramOut[offsetIdx + j] < -reg_lambda_) {
              paramOut[offsetIdx + j] += reg_lambda_;
            } else if (paramOut[offsetIdx + j] > reg_lambda_) {
              paramOut[offsetIdx + j] -= reg_lambda_;
            } else {
              paramOut[offsetIdx + j] = 0.0;
            }
          }
        }
    #ifdef LOG_FIRST_N
        LOG_FIRST_N(INFO, 3) << "L1 regularization output "
                             << paramOut[indices[0] * block_size];
    #endif // LOG_FIRST_N
      } else { // Currently only handling L1 and L2 regularization.
        return false;
      }
      return true;
        */
    }
}
