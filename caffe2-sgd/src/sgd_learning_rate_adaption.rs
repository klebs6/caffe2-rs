crate::ix!();

use crate::{
    Workspace,
    Operator,
    OperatorDef
};

#[inline] pub fn lr_update<Context>(
    n:                      i32,
    grad:                   *const f32,
    effgrad:                *const f32,
    lr:                     *const f32,
    nlr:                    *mut f32,
    lr_alpha:               f32,
    normalized_lr_adaption: bool,
    context:                *mut Context)  {
    todo!();
    /*
        float x = 0;
      float y = 0, z = 0;
      const float kEps = 1e-12f;
      for (auto i = 0; i < n; i++) {
        x += grad[i] * effgrad[i];
        if (normalized_lr_adaption) {
          y += grad[i] * grad[i];
          z += effgrad[i] * effgrad[i];
        }
      }
      if (normalized_lr_adaption) {
        y = fmax(std::sqrt(y), kEps);
        z = fmax(std::sqrt(z), kEps);
        nlr[0] = lr[0] * (1 - lr_alpha * x / (y * z));
      } else {
        nlr[0] = lr[0] - lr_alpha * x;
      }
    */
}

/**
  | Learning Rate Adaption is an operation
  | that perform one iteration of gradient
  | descent based on learning rate:
  | 
  | lr(k) = lr(k-1) - lr_alpha * df(k-1)/dlr,
  | 
  | where df(k-1)/dlr is the gradient of
  | objective function f on lr, and lr_alpha
  | is a learning rate hyperparameter.
  | 
  | It can be prove that df(k-1)/dlr equals
  | 
  | INNERPRODUCT(grad(k-1), -grad(k-2)),
  | where grad(k-1) is the grad of f(k-1)
  | on parameters.
  | 
  | When the argument "normalized_lr_adaption"
  | is false, we simply perform the following
  | update:
  | 
  | lr(k) = lr(k-1) - lr_alpha
  | 
  | INNERPRODUCT(grad(k-1), grad(k-2)).
  | 
  | If we set "normalized_lr_adaption"
  | to be true, we do not directly apply INNERPRODUCT(grad(k-1),
  | 
  | -grad(k-2)) as the grad.
  | 
  | Instead, we perform the following update:
  | 
  | lr(k) = lr(k-1) + lr_alpha cosineSimilarity(grad(k-1),
  | grad(k-2)).
  |
  */
pub struct LearningRateAdaptionOp<T,Context> {
    context:                Context,
    lr_alpha:               T, /// {1e-2};
    normalized_lr_adaption: bool, // default = true
}

impl<T,Context> Operator for LearningRateAdaptionOp<T,Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
}

num_inputs!{LearningRateAdaption, 3}

num_outputs!{LearningRateAdaption, 1}

inputs!{LearningRateAdaption, 
    0 => ("lr", "Learning rate"),
    1 => ("grad", "Gradient computed"),
    2 => ("effgrad", "The effective grad")
}

outputs!{LearningRateAdaption, 
    0 => ("output_lr", "Updated learning rate")
}

args!{LearningRateAdaption, 
    0 => ("lr_alpha", "the learning rate for performing gradient descent on learning rate lr"),
    1 => ("normalized_lr_adaption", "whether to apply normalized lr adaption or not")
}

allow_inplace!{LearningRateAdaption, vec![(0, 0)]}

input_tags!{
    LearningRateAdaptionOp
    {
        Lr,
        Grad,
        Effgrad
    }
}

output_tags!{
    LearningRateAdaptionOp
    {
        OutputLr
    }
}

impl<T,Context> LearningRateAdaptionOp<T,Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator(operator_def, ws),
            lr_alpha_(this->template GetSingleArgument<float>("lr_alpha", 0.01f)),
            normalized_lr_adaption_(this->template GetSingleArgument<bool>(
                "normalized_lr_adaption",
                true))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(Input(LR).numel() == 1);
        CAFFE_ENFORCE(Input(GRAD).numel() == Input(EFFGRAD).numel());
        Output(OUTPUT_LR)->ResizeLike(Input(LR));
        lr_update<Context>(
            Input(GRAD).numel(),
            Input(GRAD).template data<T>(),
            Input(EFFGRAD).template data<T>(),
            Input(LR).template data<T>(),
            Output(OUTPUT_LR)->template mutable_data<T>(),
            lr_alpha_,
            normalized_lr_adaption_,
            &context_);
        return true;
        */
    }
}

register_cpu_operator!{
    LearningRateAdaption,
    LearningRateAdaptionOp<f32, CPUContext>
}

no_gradient!{LearningRateAdaption}
