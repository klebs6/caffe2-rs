crate::ix!();

use crate::{
    OperatorStorage,
    Operator,
    OperatorDef,
    Workspace
};

/**
 | Computes the STORM
 | (https://arxiv.org/abs/1905.10018) update for an
 | input gradient and accumulated history of
 | gradients. Concretely, given inputs (param,
 | moment, grad_sq_sum, grad, lr), computes:
 |
 |     new_grad_sq_sum = grad_sq_sum + norm(grad)^2
 |     effective_lr = lr / (beta + new_grad_sq_sum)^1/3
 |     alpha = momentum * square(effective_lr)
 |     new_moment = grad + (1 - alpha) * (moment - grad)
 |     new_param = param + effective_lr * new_moment
 |
 | and returns (new_param, new_moment,
 | new_grad_sq_sum).
 |
 | Note that due to caffe2 limitation, it is
 | difficult to re-calculate gradient in the previous
 | iteration using the current example. We simplied
 | calculation for new_moment by using the gradient
 | from the current iteration.
 */
pub struct StormOp<Context> {

    storage: OperatorStorage,
    context: Context,

    momentum: f32,
    beta:     f32,
}

impl<Context> Operator for StormOp<Context> {

}

register_cpu_operator!{Storm, StormOp<CPUContext>}

should_not_do_gradient!{Storm}

num_inputs!{Storm, 5}

num_outputs!{Storm, 3}

inputs!{Storm, 
    0 => ("param",                "Parameters to be updated."),
    1 => ("moment",               "Moment history."),
    2 => ("grad_sq_sum",          "Sum of observed squared gradients."),
    3 => ("grad",                 "Gradients computed."),
    4 => ("lr",                   "Learning rate, k in the original paper.")
}

outputs!{Storm, 
    0 => ("output_param",         "Updated parameters."),
    1 => ("output_moment",        "Updated moment."),
    2 => ("output_grad_sq_sum",   "Updated sum of squared gradients.")
}

args!{Storm, 
    0 => ("momentum",             "Momentum hyperparameter, c in the original paper."),
    1 => ("beta",                 "denominator in adaptive learning rate, w in the original paper.")
}

allow_inplace!{Storm, vec![(0, 0), (1, 1), (2, 2)]}

input_tags!{
    StormOp {
        Param,
        Moment,
        Gradsqsum,
        Grad,
        Lr
    }
}

output_tags!{
    StormOp {
        OutputParam,
        OutputMoment,
        OutputGragsqsum
    }
}

impl<Context> StormOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator(operator_def, ws),
            OP_SINGLE_ARG(float, "momentum", momentum_, 10.0),
            OP_SINGLE_ARG(float, "beta", beta_, 0.1)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Enforce shapes
        CAFFE_ENFORCE_EQ(Input(GRAD).numel(), Input(PARAM).numel());
        CAFFE_ENFORCE_EQ(Input(GRAD).numel(), Input(MOMENT).numel());
        CAFFE_ENFORCE_EQ(Input(GRADSQSUM).numel(), 1);
        CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);

        // Resize [potentially] out-of-place blobs
        Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
        Output(OUTPUT_MOMENT)->ResizeLike(Input(MOMENT));
        Output(OUTPUT_GRAGSQSUM)->ResizeLike(Input(GRADSQSUM));

        storm_update<Context>(
            Input(GRAD).numel(),
            Input(PARAM).template data<float>(),
            Input(MOMENT).template data<float>(),
            Input(GRADSQSUM).template data<float>(),
            Input(GRAD).template data<float>(),
            Input(LR).template data<float>(),
            Output(OUTPUT_PARAM)->template mutable_data<float>(),
            Output(OUTPUT_MOMENT)->template mutable_data<float>(),
            Output(OUTPUT_GRAGSQSUM)->template mutable_data<float>(),
            momentum_,
            beta_,
            &context_);
        return true;
        */
    }
}

/**
  | This operator implement the STORM (https://arxiv.org/abs/1905.10018)
  | optimization algorithm.
  | 
  | Given inputs (param, moment, grad_sq_sum,
  | grad, indices, lr), computes the dense
  | STORM update on (param, moment[indices],
  | grad_sq_sum, grad, lr), and returns
  | (new_param, new_moment, new_grad_sq_sum)
  | as in the dense case.
  |
  */
pub struct SparseStormOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    momentum: f32,
    beta:     f32,
}

num_inputs!{SparseStorm, 6}

num_outputs!{SparseStorm, 3}

inputs!{SparseStorm, 
    0 => ("param",                "Parameters to be updated."),
    1 => ("moment",               "Moment history."),
    2 => ("grad_sq_sum",          "Sum of observed squared gradients."),
    3 => ("grad",                 "Gradients computed."),
    4 => ("indices",              "Sparse indices."),
    5 => ("lr",                   "Learning rate, k in the original paper.")
}

outputs!{SparseStorm, 
    0 => ("output_param",        "Updated parameters."),
    1 => ("output_moment",       "Updated moment."),
    2 => ("output_grad_sq_sum",  "Updated sum of squared gradients.")
}

args!{SparseStorm, 
    0 => ("momentum",            "Momentum hyperparameter, c in the original paper."),
    1 => ("beta",                "denominator in adaptive learning rate, w in the original paper.")
}

enforce_one_to_one_inplace!{SparseStorm}

register_cpu_operator!{SparseStorm, SparseStormOp<CPUContext>}

should_not_do_gradient!{SparseStorm}

input_tags!{
    SparseStormOp {
        Param,
        Moment,
        Gradsqsum,
        Grad,
        Indices,
        Lr
    }
}

output_tags!{
    SparseStormOp {
        OutputParam,
        OutputMoment,
        OutputGragsqsum
    }
}

impl<Context> SparseStormOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator(operator_def, ws),
            OP_SINGLE_ARG(float, "momentum", momentum_, 10.0),
            OP_SINGLE_ARG(float, "beta", beta_, 0.1)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Enforce shapes
        CAFFE_ENFORCE_EQ(Input(PARAM).numel(), Input(MOMENT).numel());
        CAFFE_ENFORCE_EQ(Input(GRADSQSUM).numel(), 1);
        CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
        CAFFE_ENFORCE_EQ(
            Input(PARAM).size_from_dim(1),
            Input(GRAD).size_from_dim(Input(INDICES).dim()));

        return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }
    
    #[inline] pub fn do_run_with_type<SIndex>(&mut self, ) -> bool {
        todo!();
        /*
            const auto* paramIn = Input(PARAM).template data<float>();
        const auto* momentIn = Input(MOMENT).template data<float>();
        const auto* gradSqSumIn = Input(GRADSQSUM).template data<float>();
        const auto* gradIn = Input(GRAD).template data<float>();
        const auto* indices = Input(INDICES).template data<SIndex>();
        const auto* lr = Input(LR).template data<float>();
        auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<float>();
        auto* momentOut = Output(OUTPUT_MOMENT)->template mutable_data<float>();
        auto* gradSqSumOut =
            Output(OUTPUT_GRAGSQSUM)->template mutable_data<float>();

        auto n = Input(INDICES).numel();
        if (n == 0) {
          return true;
        }

        float gradSqSumTmp = 0.0;
        for (auto i = 0; i < Input(GRAD).numel(); ++i) {
          const float gi = gradIn[i];
          gradSqSumTmp += gi * gi;
        }
        gradSqSumOut[0] = gradSqSumIn[0] + gradSqSumTmp;

        const float nlr = lr[0] * std::pow(beta_ + gradSqSumOut[0], -1.0 / 3.0);
        const float alpha = momentum_ * nlr * nlr;
        const auto block_size = Input(GRAD).numel() / n;

        for (auto i = 0; i < n; ++i) {
          auto idx = indices[i];
          if (block_size == 1) {
            const float gi = gradIn[i];
            const float mi = momentIn[idx];
            float new_mi = momentOut[idx] = gi + (1.0 - alpha) * (mi - gi);
            paramOut[idx] = paramIn[idx] + nlr * new_mi;
          } else {
            auto offsetI = i * block_size;
            auto offsetIdx = idx * block_size;

    #ifndef NDEBUG
            CAFFE_ENFORCE_GE(
                Input(PARAM).numel(),
                block_size + offsetIdx,
                this->debug_def().input(PARAM),
                ", out of bound,  idx:",
                idx,
                " for input i:",
                i,
                " and block size:",
                block_size);
            CAFFE_ENFORCE_GE(
                Input(GRAD).numel(),
                block_size + offsetI,
                this->debug_def().input(GRAD),
                ", out of bound idx, idx:",
                idx,
                " for input i:",
                i);
    #endif

            for (auto j = 0; j < block_size; ++j) {
              const float gi = gradIn[offsetI + j];
              const float mi = momentIn[offsetIdx + j];
              float new_mi = momentOut[offsetIdx + j] =
                  gi + (1.0 - alpha) * (mi - gi);
              paramOut[offsetIdx + j] = paramIn[offsetIdx + j] + nlr * new_mi;
            }
          }
        }

        return true;
        */
    }
}

#[inline] pub fn storm_update<Context>(
    n:                 i32,
    param_in:          *const f32,
    moment_in:         *const f32,
    grad_sq_sum_in:    *const f32,
    grad_in:           *const f32,
    lr:                *const f32,
    param_out:         *mut f32,
    moment_out:        *mut f32,
    grad_sq_sum_out:   *mut f32,
    momentum:          f32,
    beta:              f32,
    context:           *mut Context) 
{
    todo!();
    /*
        float gradSqSumTmp = 0.0;
      for (auto i = 0; i < N; ++i) {
        const float gi = gradIn[i];
        gradSqSumTmp += gi * gi;
      }
      gradSqSumOut[0] = gradSqSumIn[0] + gradSqSumTmp;

      const float nlr = lr[0] * std::pow(beta + gradSqSumOut[0], -1.0 / 3.0);
      const float alpha = momentum * nlr * nlr;
      for (auto i = 0; i < N; ++i) {
        const float gi = gradIn[i];
        const float mi = momentIn[i];
        float new_mi = momentOut[i] = gi + (1.0 - alpha) * (mi - gi);
        paramOut[i] = paramIn[i] + nlr * new_mi;
      }
    */
}
