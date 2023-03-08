crate::ix!();

use crate::{
    CPUContext,
    OperatorStorage,
    Workspace,
    Operator,
    OperatorDef
};

#[inline] pub fn wngrad_update<Context>(
    n:        i32,
    w:        *const f32,
    g:        *const f32,
    h:        *const f32,
    nw:       *mut f32,
    nh:       *mut f32,
    epsilon:  f32,
    lr:       *const f32,
    context:  *mut Context)
{
    todo!();
    /*
        for (auto i = 0; i < N; ++i) {
        float gi = g[i];
        nw[i] = w[i] + lr[0] * gi / (h[0] + epsilon);
      }
      float nhTmp = 0.0;
      for (auto i = 0; i < N; ++i) {
        float gi = g[i];
        nhTmp += gi * gi;
      }
      nhTmp /= (h[0] + epsilon);
      nh[0] = h[0] + nhTmp;
    */
}

#[inline] pub fn wngrad_update_output_effective_lr<Context>(
    n:                i32,
    param_in:         *const f32,
    grad_in:          *const f32,
    seq_bin:          *const f32,
    param_out:        *mut f32,
    seq_bout:         *mut f32,
    effective_lrout:  *mut f32,
    epsilon:          f32,
    lr:               *const f32,
    context:          *mut Context) 
{
    todo!();
    /*
        effectiveLROut[0] = lr[0] / (seqBIn[0] + epsilon);
      float seqBTmp = 0.0;
      for (auto i = 0; i < N; ++i) {
        float gi = gradIn[i];
        seqBTmp += gi * gi;
      }
      seqBTmp /= (seqBIn[0] + epsilon);
      seqBOut[0] = seqBIn[0] + seqBTmp;
      for (auto i = 0; i < N; ++i) {
        float grad = gradIn[i];
        paramOut[i] = paramIn[i] + effectiveLROut[0] * grad;
      }
    */
}

#[inline] pub fn wngrad_update_output_effective_lr_and_update<Context>(
    n:                i32,
    param_in:         *const f32,
    grad_in:          *const f32,
    seq_bin:          *const f32,
    param_out:        *mut f32,
    seq_bout:         *mut f32,
    effective_lrout:  *mut f32,
    update_out:       *mut f32,
    epsilon:          f32,
    lr:               *const f32,
    context:          *mut Context) 
{
    todo!();
    /*
        effectiveLROut[0] = lr[0] / (seqBIn[0] + epsilon);
      float seqBTmp = 0.0;
      for (auto i = 0; i < N; ++i) {
        float gi = gradIn[i];
        seqBTmp += gi * gi;
      }
      seqBTmp /= (seqBIn[0] + epsilon);
      seqBOut[0] = seqBIn[0] + seqBTmp;

      for (auto i = 0; i < N; ++i) {
        float grad = gradIn[i];
        float update = updateOut[i] = effectiveLROut[0] * grad;
        paramOut[i] = paramIn[i] + update;
      }
    */
}

/**
 | Computes the WnGrad update for an input gradient
 | and accumulated history. This operator implement
 | the optimization algorithm in
 | https://arxiv.org/abs/1803.02865 by Wu, Ward and
 | Bottou.
 |
 | Concretely, given inputs (param, grad, seq_b,
 | learning_rate), computes
 |
 |     new_seq_b = seq_b + 1 / seq_b * norm(grad)^2
 |     effective_lr = learning_rate / (new_seq_b + epsilon)
 |     update = learning_rate * grad / (new_seq_b + epsilon)
 |     new_param = param + update
 | and returns (new_param, new_seq_b).
 |
 | Optionally returns effective_lr and update as well.
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct WngradOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    epsilon_: T,
}

input_tags!{
    WngradOpInputs {
        Param, 
        SeqB, 
        Grad, 
        Lr
    }
}

output_tags!{
    WngradOpOutputs {
        OutputParam, 
        OutputSeqB, 
        OutputEffectiveLR, 
        OutputUpdate
    }
}

impl<T,Context> Operator for WngradOp<T, Context> {
}

register_cpu_operator!{
    Wngrad, 
    WngradOp::<f32, CPUContext>
}

num_inputs!{Wngrad, 4}

num_outputs!{Wngrad, (2, 4)}

allow_inplace!{Wngrad, vec![(0, 0), (1, 1)]}

inputs!{Wngrad,
    0 => ("param", "Parameters to be updated"),
    1 => ("seq_b", "Seq_b history"),
    2 => ("grad", "Gradient computed"),
    3 => ("lr", "learning rate")
}

outputs!{Wngrad,
    0 => ("output_param",        "Updated parameters"),
    1 => ("output_seq_b",        "Updated seq_b"),
    2 => ("output_effective_lr", "(optional) Effective learning rate"),
    3 => ("output_update",       "(optional) Actual update that is applied.")
}

args!{Wngrad,
    0 => ("epsilon", "Default 1e-5")
}

should_not_do_gradient!{Wngrad}

impl<T,Context> WngradOp<T,Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator(operator_def, ws),
            epsilon_(this->template GetSingleArgument<T>("epsilon", 1e-5f))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(
            Input(GRAD).numel(),
            Input(PARAM).numel(),
            "PARAM size: ",
            Input(PARAM).numel(),
            ", GRAD size: ",
            Input(GRAD).numel(),
            ", SEQ_B size: ",
            Input(SEQ_B).numel(),
            ", LR size: ",
            Input(LR).numel());

        Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
        Output(OUTPUT_SEQ_B)->ResizeLike(Input(SEQ_B));
        if (OutputSize() == 2) {
          wngrad_update<Context>(
              Input(GRAD).numel(),
              Input(PARAM).template data<T>(),
              Input(GRAD).template data<T>(),
              Input(SEQ_B).template data<T>(),
              Output(OUTPUT_PARAM)->template mutable_data<T>(),
              Output(OUTPUT_SEQ_B)->template mutable_data<T>(),
              epsilon_,
              Input(LR).template data<T>(),
              &context_);
        } else if (OutputSize() == 3) {
          Output(OUTPUT_EFFECTIVE_LR)->ResizeLike(Input(SEQ_B));
          wngrad_update_output_effective_lr<Context>(
              Input(GRAD).numel(),
              Input(PARAM).template data<T>(),
              Input(GRAD).template data<T>(),
              Input(SEQ_B).template data<T>(),
              Output(OUTPUT_PARAM)->template mutable_data<T>(),
              Output(OUTPUT_SEQ_B)->template mutable_data<T>(),
              Output(OUTPUT_EFFECTIVE_LR)->template mutable_data<T>(),
              epsilon_,
              Input(LR).template data<T>(),
              &context_);
        } else {
          Output(OUTPUT_EFFECTIVE_LR)->ResizeLike(Input(SEQ_B));
          Output(OUTPUT_UPDATE)->ResizeLike(Input(GRAD));
          wngrad_update_output_effective_lr_and_update<Context>(
              Input(GRAD).numel(),
              Input(PARAM).template data<T>(),
              Input(GRAD).template data<T>(),
              Input(SEQ_B).template data<T>(),
              Output(OUTPUT_PARAM)->template mutable_data<T>(),
              Output(OUTPUT_SEQ_B)->template mutable_data<T>(),
              Output(OUTPUT_EFFECTIVE_LR)->template mutable_data<T>(),
              Output(OUTPUT_UPDATE)->template mutable_data<T>(),
              epsilon_,
              Input(LR).template data<T>(),
              &context_);
        }

        return true;
        */
    }
}

/**
  | This operator implement the optimization
  | algorithm in https://arxiv.org/abs/1803.02865
  | by Wu, Ward and Bottou.
  | 
  | Given inputs (param, seq_b, indices,
  | grad, lr), runs the dense WnGrad update
  | on (param, grad, seq_b, lr), and returns
  | (new_param, new_seq_b) as in the dense
  | case.
  |
  */
pub struct SparseWngradOp<T> {
    storage: OperatorStorage,
    context: CPUContext,
    epsilon_: T,
}

input_tags!{
    SparseWngradOpInputs {
        PARAM, 
        SEQ_B, 
        INDICES, 
        GRAD, 
        LR
    }
}

output_tags!{
    SparseWngradOpOutputs {
        OUTPUT_PARAM, 
        OUTPUT_SEQ_B
    }
}

num_inputs!{SparseWngrad,  5}

num_outputs!{SparseWngrad, 2}

inputs!{SparseWngrad,
    0 => ("param",   "Parameters to be updated"),
    1 => ("seq_b",   "seq_b history"),
    2 => ("indices", "Sparse indices"),
    3 => ("grad",    "Gradient computed"),
    4 => ("lr",      "learning rate")
}

outputs!{SparseWngrad,
    0 => ("output_param", "Updated parameters"),
    1 => ("output_seq_b", "Updated seq_b")
}

args!{SparseWngrad,
    0 => ("epsilon", "Default 1e-5")
}

enforce_one_to_one_inplace!{SparseWngrad}

should_not_do_gradient!{SparseWngrad}

register_cpu_operator!{
    SparseWngrad, 
    SparseWngradOp::<f32, CPUContext>
}

impl<T> SparseWngradOp<T> {
    
    pub fn new(
        operator_def: &OperatorDef,
        ws: *mut Workspace) -> Self 
    {
        todo!();
        /*
            : Operator(operator_def, ws),
            epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Enforce shapes
        CAFFE_ENFORCE_EQ(Input(SEQ_B).numel(), 1);
        CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
        CAFFE_ENFORCE_EQ(
            Input(PARAM).size_from_dim(1),
            Input(GRAD).size_from_dim(Input(INDICES).dim()));

        return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }

    #[inline] pub fn do_run_with_type<SIndex>(&mut self) -> bool {
        todo!();
        /*
            const auto* lr = Input(LR).template data<T>();
            const auto* indices = Input(INDICES).template data<SIndex>();
            const auto* gradIn = Input(GRAD).template data<T>();
            const auto* paramIn = Input(PARAM).template data<T>();
            const auto* seqBIn = Input(SEQ_B).template data<T>();
            auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<T>();
            auto* seqBOut = Output(OUTPUT_SEQ_B)->template mutable_data<T>();

            auto n = Input(INDICES).numel();
            if (n == 0) {
              return true;
            }

            auto block_size = Input(GRAD).numel() / n;

            for (auto i = 0; i < n; ++i) {
              auto idx = indices[i];
              if (block_size == 1) {
                float gi = gradIn[i];
                paramOut[idx] = paramIn[idx] + lr[0] * gi / (seqBIn[0] + epsilon_);
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
                  float gi = gradIn[offsetI + j];
                  paramOut[offsetIdx + j] =
                      paramIn[offsetIdx + j] + lr[0] * gi / (seqBIn[0] + epsilon_);
                }
              }
            }
            float seqBTmp = 0.0;
            for (auto i = 0; i < Input(GRAD).numel(); ++i) {
              float gi = gradIn[i];
              seqBTmp += gi * gi;
            }
            seqBTmp /= seqBIn[0];
            seqBOut[0] = seqBTmp + seqBIn[0];
            return true;
        */
    }
}
