crate::ix!();

use crate::{
    Operator,
    Tensor,
    OperatorDef,
    CPUContext,
    Workspace
};

/**
  | Implement Layer-wise Adaptive Rate
  | Scaling (LARS) with clipping. Before
  | adding weight decay, given a parameter
  | tensor X and its gradient dX, the local
  | learning rate for X will be
  | 
  | local_lr = trust * norm(X) / ( norm(dX)
  | + wd * norm(X) + offset * norm(X) )
  | 
  | = trust / ( norm(dX) / norm(X) + wd + offset),
  | 
  | where offset is a preset hyper-parameter
  | to avoid numerical issue and trust indicates
  | how much we trust the layer to change
  | its parameters during one update.
  | 
  | In this implementation, we uses l2 norm
  | and the computed local learning rate
  | is clipped based on the upper bound lr_max
  | and the lower bound lr_min:
  | 
  | local_lr = min(local_lr, lr_max) and
  | local_lr = max(local_lr, lr_min)
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LarsOp<T,Context> {
    context:         Context,
    offset:          T,
    lr_min:          T,
    x_norm_tensor:   Tensor,
    dX_norm_tensor:  Tensor,
}

impl<T,Context> Operator for LarsOp<T,Context> {

}

num_inputs!{Lars, 5}

num_outputs!{Lars, 1}

inputs!{Lars, 
    0 => ("X", "Parameter tensor"),
    1 => ("dX", "Gradient tensor"),
    2 => ("wd", "Weight decay"),
    3 => ("trust", "Trust"),
    4 => ("lr_max", "Upper bound of learning rate")
}

outputs!{Lars, 
    0 => ("lr_rescaled", "Rescaled local learning rate")
}

args!{Lars, 
    0 => ("offset", "rescaling offset parameter"),
    1 => ("lr_min", "minimum learning rate for clipping")
}

impl<T,Context> LarsOp<T,Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator(operator_def, ws),
            offset_(this->template GetSingleArgument<float>("offset", 0.5)),
            lr_min_(this->template GetSingleArgument<float>("lr_min", 0.02))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
        auto& dX = Input(1);
        CAFFE_ENFORCE(
            dX.numel() == X.numel(), "Gradient size doesn't match parameter size.");
        CAFFE_ENFORCE_GE(offset_, 0);
        CAFFE_ENFORCE_GE(lr_min_, 0);

        auto& wd = Input(2);
        auto& trust = Input(3);
        auto& lr_max = Input(4);

        auto* lr_rescaled = Output(0, vector<int64_t>{1}, at::dtype<T>());

        ReinitializeTensor(&X_norm_tensor_, {1}, at::dtype<T>().device(Context::GetDeviceType()));
        T* X_norm_ = X_norm_tensor_.template mutable_data<T>();

        ReinitializeTensor(&dX_norm_tensor_, {1}, at::dtype<T>().device(Context::GetDeviceType()));
        T* dX_norm_ = dX_norm_tensor_.template mutable_data<T>();

        ComputeNorms(
            dX.numel(),
            X.template data<T>(),
            dX.template data<T>(),
            X_norm_,
            dX_norm_);

        ComputeLearningRate(
            wd.template data<T>(),
            trust.template data<T>(),
            lr_max.template data<T>(),
            offset_,
            lr_min_,
            X_norm_,
            dX_norm_,
            lr_rescaled->template mutable_data<T>());

        return true;
        */
    }

    /// Compute the l2 norm of X_data and dX_data
    #[inline] pub fn compute_norms(&mut self, 
        n:       i64,
        x_data:  *const T,
        dX_data: *const T,
        x_norm:  *mut T,
        dX_norm: *mut T)  {

        todo!();
        /*
            math::SumSqr(N, X_data, X_norm, &context_);
        math::Sqrt(1, X_norm, X_norm, &context_);
        math::SumSqr(N, dX_data, dX_norm, &context_);
        math::Sqrt(1, dX_norm, dX_norm, &context_);
        */
    }

    /// Compute the learning rate and apply clipping
    #[inline] pub fn compute_learning_rate(&mut self, 
        wd:          *const T,
        trust:       *const T,
        lr_max:      *const T,
        offset:      T,
        lr_min:      T,
        x_norm:      *mut T,
        dX_norm:     *mut T,
        lr_rescaled: *mut T)  {

        todo!();
        /*
        
        */
    }

    #[inline] pub fn compute_learning_rate_f32_cpucontext(&mut self, 
        wd:          *const f32,
        trust:       *const f32,
        lr_max:      *const f32,
        offset:      f32,
        lr_min:      f32,
        x_norm:      *mut f32,
        dX_norm:     *mut f32,
        lr_rescaled: *mut f32)  {

        todo!();
        /*
            float val = 1.0;

      if (*X_norm > 0) {
        val = (*trust) / (*dX_norm / *X_norm + (*wd) + offset);
      }
      *lr_rescaled = fmaxf(fminf(val, *lr_max), lr_min);
        */
    }
}

register_cpu_operator!{Lars, LarsOp<float, CPUContext>}

should_not_do_gradient!{Lars}
