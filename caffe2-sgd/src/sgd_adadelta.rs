crate::ix!();

use crate::{
    OperatorStorage,
    RunOnDevice,
    Workspace,
    Operator,
    OperatorDef
};

/**
 | Computes the AdaDelta update
 | (https://arxiv.org/abs/1212.5701) for an input
 | gradient and accumulated history of squared
 | gradients. Concretely, given inputs (param,
 | moment, moment_delta, grad, learning_rate),
 | computes:
 |
 |     new_moment = moment * decay + square(grad) * (1 - decay)
 |     new_grad = sqrt(moment_delta + epsilon) / sqrt(new_moment + epsilon) * grad
 |     new_param = param + learning_rate * new_grad
 |     new_moment_delta = moment_delta * decay + square(new_grad) * (1 - decay)
 |
 | and returns (new_param, new_moment,
 | new_moment_delta).  
 */
pub struct AdadeltaOp<Context> {
    storage:  OperatorStorage,
    context:  Context,
    epsilon_: f32,
    decay_:   f32,
}

impl<Context> AdadeltaOp<Context> {
    fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
        Self {
            base: Operator::<Context>::new(operator_def, ws),
            epsilon_: 1e-5,
            decay_:   0.95,
        }
        */
    }
}

register_cpu_operator!{Adadelta, AdadeltaOp<CPUContext>}

num_inputs!{Adadelta,  5}

num_outputs!{Adadelta, 3}

inputs!{Adadelta,
    0 => ("param",        "parameters to be updated"),
    1 => ("moment",       "Average of squared gradients"),
    2 => ("moment_delta", "Average of squared parameter updates"),
    3 => ("grad",         "Gradient computed"),
    4 => ("lr",           "Learning rate")
}

outputs!{Adadelta,
    0 => ("output_param",        "Updated parameters"),
    1 => ("output_moment",       "Updated average squared gradient"),
    2 => ("output_moment_delta", "Updated average of squared parameter updates")
}

args!{Adadelta,
    0 => ("epsilon", "default 1e-5"),
    1 => ("decay",   "default 0.95, the squared gradient sum is decayed by this factor")
}

allow_inplace!{Adadelta, vec![(0, 0), (1, 1), (2, 2)]}

should_not_do_gradient!{Adadelta}

input_tags!{
    AdadeltaOp {
        Param,
        MomentGrad,
        MomentDelta,
        Grad,
        LR
    }
}

output_tags!{
    AdadeltaOp {
        OutputParam,
        OutputMomentGrad,
        OutputMomentDelta
    }
}

impl<Context> RunOnDevice for AdadeltaOp<Context> {

    fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
        CAFFE_ENFORCE(Input(GRAD).numel() == Input(MOMENT_GRAD).numel());
        CAFFE_ENFORCE(Input(GRAD).numel() == Input(MOMENT_DELTA).numel());
        CAFFE_ENFORCE(Input(GRAD).numel() == Input(PARAM).numel());
        CAFFE_ENFORCE_GE(epsilon_, 0.0f);
        CAFFE_ENFORCE_GT(decay_, 0.0f);
        CAFFE_ENFORCE_LT(decay_, 1.0f);

        Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
        Output(OUTPUT_MOMENT_GRAD)->ResizeLike(Input(MOMENT_GRAD));
        Output(OUTPUT_MOMENT_DELTA)->ResizeLike(Input(MOMENT_DELTA));
        AdadeltaUpdate<Context>(
            Input(GRAD).numel(),
            Input(PARAM).template data<float>(),
            Input(GRAD).template data<float>(),
            Input(MOMENT_GRAD).template data<float>(),
            Input(MOMENT_DELTA).template data<float>(),
            epsilon_,
            decay_,
            Input(LR).template data<float>(),
            Output(OUTPUT_PARAM)->template mutable_data<float>(),
            Output(OUTPUT_MOMENT_GRAD)->template mutable_data<float>(),
            Output(OUTPUT_MOMENT_DELTA)->template mutable_data<float>(),
            &context_);
        return true;
        */
    }
}

pub fn adadelta_update<Context>(
    n:       i32,
    w:       *const f32,
    g:       *const f32,
    h:       *const f32,
    d:       *const f32,
    epsilon: f32,
    decay:   f32,
    lr:      *const f32,
    nw:      *mut f32,
    nh:      *mut f32,
    nd:      *mut f32,
    context: *mut Context)
{
    todo!();
    /*
        for (int i = 0; i < N; ++i) {
        float gi = g[i];
        float di = d[i];
        float hi = nh[i] = decay * h[i] + (1.0f - decay) * gi * gi;
        float ng = (std::sqrt(di + epsilon) / std::sqrt(hi + epsilon)) * gi;
        nw[i] = w[i] + lr[0] * ng;
        nd[i] = decay * di + (1.0f - decay) * ng * ng;
      }
    */
}

/**
  | Given inputs (param, moment, moment_delta,
  | indices, grad, lr), runs the dense AdaDelta
  | update on (param, grad, moment[indices],
  | moment_delta[indices], lr), and returns
  | (new_param, new_moment, new_moment_delta)
  | as in the dense case.
  |
  */
pub struct SparseAdadeltaOp {
  epsilon_: f32,
  decay_:   f32,
}

impl SparseAdadeltaOp {
    fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
        Self {
            base: Operator::<Context>::new(operator_def, ws),
            epsilon_: 1e-5,
            decay_:   0.95,
        }
        */
    }
}

input_tags!{
    SparseAdadeltaOpInputs {
        Param,
        MomentGrad,
        MomentDelta,
        Indices,
        Grad,
        LR
    }
}

output_tags!{
    SparseAdadeltaOpOutputs {
        OutputParam,
        OutputMomentGrad,
        OutputMomentDelta
    }
}

impl SparseAdadeltaOp {

    pub fn do_run_with_type<SIndex>(&mut self) -> bool {
        todo!();
        /*
            const auto* lr = Input(LR).template data<float>();
            const auto* indices = Input(INDICES).template data<SIndex>();
            const auto* gradIn = Input(GRAD).template data<float>();
            const auto* paramIn = Input(PARAM).template data<float>();
            const auto* momentIn = Input(MOMENT_GRAD).template data<float>();
            const auto* momentDeltaIn = Input(MOMENT_DELTA).template data<float>();
            auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<float>();
            auto* momentOut =
                Output(OUTPUT_MOMENT_GRAD)->template mutable_data<float>();
            auto* momentDeltaOut =
                Output(OUTPUT_MOMENT_DELTA)->template mutable_data<float>();

            auto n = Input(INDICES).numel();
            if (n == 0) {
              return true;
            }

            auto block_size = Input(GRAD).numel() / n;
            for (int i = 0; i < n; ++i) {
              auto idx = indices[i];
              if (block_size == 1) {
                float gi = gradIn[i];
                float di = momentDeltaIn[idx];
                float hi = momentOut[idx] =
                    decay_ * momentIn[idx] + (1.0f - decay_) * gi * gi;
                float ng = (std::sqrt(di + epsilon_) / std::sqrt(hi + epsilon_)) * gi;
                paramOut[idx] = paramIn[idx] + lr[0] * ng;
                momentDeltaOut[idx] = decay_ * di + (1.0f - decay_) * ng * ng;
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
                AdadeltaUpdate(
                    block_size,
                    paramIn + offsetIdx,
                    gradIn + offsetI,
                    momentIn + offsetIdx,
                    momentDeltaIn + offsetIdx,
                    epsilon_,
                    decay_,
                    lr,
                    paramOut + offsetIdx,
                    momentOut + offsetIdx,
                    momentDeltaOut + offsetIdx,
                    &context_);
              }
            }
            return true;
        */
    }
}

impl RunOnDevice for SparseAdadeltaOp {
    fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Enforce shapes
        CAFFE_ENFORCE_EQ(Input(PARAM).numel(), Input(MOMENT_GRAD).numel());
        CAFFE_ENFORCE_EQ(Input(PARAM).numel(), Input(MOMENT_DELTA).numel());
        CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
        CAFFE_ENFORCE_EQ(
            Input(PARAM).size_from_dim(1),
            Input(GRAD).size_from_dim(Input(INDICES).dim()));

        // Enforce domain constraints for attributes
        CAFFE_ENFORCE_GE(epsilon_, 0.0f);
        CAFFE_ENFORCE_GT(decay_, 0.0f);
        CAFFE_ENFORCE_LT(decay_, 1.0f);

        return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }
}

register_cpu_operator!{SparseAdadelta, SparseAdadeltaOp<CPUContext>}
num_inputs!{SparseAdadelta, 6}
num_outputs!{SparseAdadelta, 3}
enforce_one_to_one_inplace!{SparseAdadelta}
inputs!{SparseAdadelta,
    0 =>  ("param",        "Parameters to be updated"),
    1 =>  ("moment",       "Average of squared gradients"),
    2 =>  ("moment_delta", "Average of squared parameter updates"),
    3 =>  ("indices",      "Sparse indices"),
    4 =>  ("grad",         "Gradient computed"),
    5 =>  ("lr",           "learning rate")
}
outputs!{SparseAdadelta,
    0 => ("output_param",        "Updated parameters"),
    1 => ("output_moment",       "Updated average squared gradient"),
    2 => ("output_moment_delta", "Updated average of squared parameter updates")
}
args!{SparseAdadelta,
    0 => ("epsilon", "Default 1e-5"),
    1 => ( "decay", "Default 0.95, the squared gradient sum is decayed by this factor.")
}
should_not_do_gradient!{SparseAdadelta}
