crate::ix!();

use crate::{
    OperatorStorage,
    Operator,
    Tensor,
    OperatorDef,
    Workspace
};

/**
  | YellowFin: An automatic tuner for momentum SGD
  | (https://arxiv.org/abs/1706.03471)
  |
  | The YellowFinOp tunes learning rate and
  | momentum and performs momentum SGD steps.
  |
  | The learning rate and momentum are separate
  | for any matrix of parameters.
  */

/**
  | Computes the YellowFin update (https://arxiv.org/abs/1706.03471)
  | and performs momentum SGD optimization
  | step. lr and mu are not being shared between
  | parameters.
  | 
  | curv_win, g_avg, g2_avg and scalars_memory
  | are just auxiliary memory for computing
  | moving averages (see the publication).
  | Takes arguments beta: coefficient
  | for moving averages,
  | 
  | curv_win_width: timeframe when average
  | squared gradient is being stored,
  | 
  | epsilon: for numerical purposes,
  | 
  | nesterov and zero_debias for debias
  | of moving average.
  |
  */
pub struct YellowFinOp<T, Context> {

    storage: OperatorStorage,
    context: Context,

    curv_win_width:             i32,
    nesterov:                   bool,
    zero_debias:                bool,
    epsilon:                    T,
    beta:                       T,
    debias_factor:              T,
    d:                          i32,

    /**
      | Temporary memory on device, listed
      | all variables used in calculations
      |
      */
    aux_vector_tensor:          Tensor,
    aux_vector:                 *mut T,
    g_deb_tensor:               Tensor,
    g_deb:                      *mut T,
    g2_deb_tensor:              Tensor,
    g2_deb:                     *mut T,
    g_deb2_tensor:              Tensor,
    g_deb2:                     *mut T,
    aux_scalar_tensor:          Tensor,
    aux_scalar:                 *mut T,
    distance_tensor:            Tensor,
    distance:                   *mut T,
    distance_deb_tensor:        Tensor,
    distance_deb:               *mut T,
    g_norm_tensor:              Tensor,
    g_norm:                     *mut T,
    g_norm_deb_tensor:          Tensor,
    g_norm_deb:                 *mut T,
    g_norm2_tensor:             Tensor,
    g_norm2:                    *mut T,
    g_norm2_deb_tensor:         Tensor,
    g_norm2_deb:                *mut T,
    g_norm2_max_tensor:         Tensor,
    g_norm2_max:                *mut T,
    g_norm2_max_deb_tensor:     Tensor,
    g_norm2_max_deb:            *mut T,
    g_norm2_min_tensor:         Tensor,
    g_norm2_min:                *mut T,
    g_norm2_min_deb_tensor:     Tensor,
    g_norm2_min_deb:            *mut T,
    lr_tensor:                  Tensor,
    lr:                         *mut T,
    lr_deb_tensor:              Tensor,
    lr_deb:                     *mut T,
    mu_tensor:                  Tensor,
    mu:                         *mut T,
    mu_deb_tensor:              Tensor,
    mu_deb:                     *mut T,
    variance_tensor:            Tensor,
    variance:                   *mut T,

    scratch_tensor:             Tensor, //{Context::GetDeviceType()};

    // Input tensors' data
    param:                      *const T,
    moment:                     *const T,
    lr_avg:                     *const T,
    mu_avg:                     *const T,
    curv_win:                   *const T,
    g_avg:                      *const T,
    g2_avg:                     *const T,
    scalars_memory:             *const T,
    grad:                       *const T,
    iter:                       i32,

    /**
      | Scalar data from scalars_memory_ input
      | tensor
      |
      */
    g_norm_avg:                 *const T,
    g_norm2_avg:                *const T,
    g_norm2_min_avg:            *const T,
    g_norm2_max_avg:            *const T,
    distance_avg:               *const T,

    // Output tensors' data
    param_out:                  *mut T,
    moment_out:                 *mut T,
    lr_avg_out:                 *mut T,
    mu_avg_out:                 *mut T,
    curv_win_out:               *mut T,
    g_avg_out:                  *mut T,
    g2_avg_out:                 *mut T,
    scalars_memory_out:         *mut T,

    /**
      | Scalar data from scalars_memory_ output
      | tensor
      |
      */
    g_norm_avg_out:             *mut T,
    g_norm2_avg_out:            *mut T,
    g_norm2_min_avg_out:        *mut T,
    g_norm2_max_avg_out:        *mut T,
    distance_avg_out:           *mut T,
}

input_tags!{
    YellowFinOpInputs {
        Param,
        Moment,
        LrAvg,
        MuAvg,
        CurvWin,
        GAvg,
        G2Avg,
        ScalarsMemory,
        Grad,
        Iter
    }
}

output_tags!{
    YellowFinOpOutputs {
        OutputParam,
        OutputMoment,
        OutputLrAvg,
        OutputMuAvg,
        OutputCurvWin,
        OutputGAvg,
        OutputG2Avg,
        OutputScalarsMemory
    }
}

impl<T, Context> YellowFinOp<T, Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            curv_win_width_(
                this->template GetSingleArgument<int>("curv_win_width", 20)),
            nesterov_(this->template GetSingleArgument<int>("nesterov", false)),
            zero_debias_(
                this->template GetSingleArgument<bool>("zero_debias", true)),
            epsilon_(this->template GetSingleArgument<T>("epsilon", 1e-6f)),
            beta_(this->template GetSingleArgument<T>("beta", 0.999f))
        */
    }
    
    #[inline] pub fn after_apply(&mut self)  {
        
        todo!();
        /*
            // g
        MovingAverage(D_, grad_, g_avg_, g_avg_out_, g_deb_);
        // g2
        math::Mul(D_, grad_, grad_, aux_vector_, &context_);
        MovingAverage(D_, aux_vector_, g2_avg_, g2_avg_out_, g2_deb_);
        // g_norm2
        math::Dot(D_, grad_, grad_, g_norm2_, &context_);
        math::Maximum(1, epsilon_, g_norm2_, g_norm2_, &context_);
        MovingAverage(1, g_norm2_, g_norm2_avg_, g_norm2_avg_out_, g_norm2_deb_);
        // g_norm
        math::Sqrt(1, g_norm2_, g_norm_, &context_);
        MovingAverage(1, g_norm_, g_norm_avg_, g_norm_avg_out_, g_norm_deb_);
        math::Maximum(1, epsilon_, g_norm_deb_, g_norm_deb_, &context_);
        // Curvature range: g_norm2_min, g_norm2_max
        math::CopyVector(curv_win_width_, curv_win_, curv_win_out_, &context_);
        T* curv_win_cell = curv_win_out_ + (iter_ - 1) % curv_win_width_;
        math::Log(1, g_norm2_, curv_win_cell, &context_);
        int valid_end = std::min(curv_win_width_, iter_);
        math::ReduceMin(
            valid_end, curv_win_out_, g_norm2_min_, &scratch_tensor_, &context_);
        math::ReduceMax(
            valid_end, curv_win_out_, g_norm2_max_, &scratch_tensor_, &context_);
        MovingAverage(
            1,
            g_norm2_min_,
            g_norm2_min_avg_,
            g_norm2_min_avg_out_,
            g_norm2_min_deb_);
        MovingAverage(
            1,
            g_norm2_max_,
            g_norm2_max_avg_,
            g_norm2_max_avg_out_,
            g_norm2_max_deb_);
        math::Exp(1, g_norm2_min_deb_, g_norm2_min_deb_, &context_);
        math::Exp(1, g_norm2_max_deb_, g_norm2_max_deb_, &context_);
        math::Maximum(1, epsilon_, g_norm2_min_deb_, g_norm2_min_deb_, &context_);
        math::Maximum(1, epsilon_, g_norm2_max_deb_, g_norm2_max_deb_, &context_);
        // Gradient variance
        math::Dot(D_, g_deb_, g_deb_, aux_scalar_, &context_);

        math::Sub(1, g_norm2_deb_, aux_scalar_, variance_, &context_);
        math::Maximum(1, epsilon_, variance_, variance_, &context_);
        // Distance to opt
        math::Div(1, g_norm_avg_out_, g_norm2_avg_out_, distance_, &context_);
        MovingAverage(
            1, distance_, distance_avg_, distance_avg_out_, distance_deb_);
        if (iter_ > 1) {
          GetLrMu();
        }
        */
    }
    
    #[inline] pub fn moving_average(&mut self, 
        n:          i32,
        elt:        *const T,
        avg:        *const T,
        new_avg:    *mut T,
        debias_avg: *mut T)  
    {
        todo!();
        /*
            const T one = 1;
        math::Scale(N, beta_, avg, new_avg, &context_);
        math::Axpy(N, one - beta_, elt, new_avg, &context_);
        math::Scale(N, debias_factor_, new_avg, debias_avg, &context_);
        */
    }
    
    #[inline] pub fn zero_debias_factor(&mut self) -> T {
        
        todo!();
        /*
            if (zero_debias_) {
          const T one = 1;
          return one / (one - std::pow(beta_, iter_));
        } else {
          return 1;
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Iter live on the CPU

    #define CAFFE2_YF_READ_INPUT(INPUT_NAME, VAR_NAME)   \
      const auto& VAR_NAME##_tensor = Input(INPUT_NAME); \
      VAR_NAME##_ = VAR_NAME##_tensor.template data<T>();

    CAFFE2_YF_READ_INPUT(PARAM, param)
    CAFFE2_YF_READ_INPUT(MOMENT, moment)
    CAFFE2_YF_READ_INPUT(LR_AVG, lr_avg)
    CAFFE2_YF_READ_INPUT(MU_AVG, mu_avg)
    CAFFE2_YF_READ_INPUT(CURV_WIN, curv_win)
    CAFFE2_YF_READ_INPUT(G_AVG, g_avg)
    CAFFE2_YF_READ_INPUT(G2_AVG, g2_avg)
    CAFFE2_YF_READ_INPUT(SCALARS_MEMORY, scalars_memory)
    CAFFE2_YF_READ_INPUT(GRAD, grad)
    #undef CAFFE2_YF_READ_OUTPUT

    CAFFE_ENFORCE(OperatorStorage::InputIsTensorType(ITER, CPU));
    CAFFE_ENFORCE_EQ(lr_avg_tensor.numel(), 1);
    CAFFE_ENFORCE_EQ(mu_avg_tensor.numel(), 1);
    CAFFE_ENFORCE_EQ(param_tensor.dim(), moment_tensor.dim());
    CAFFE_ENFORCE_EQ(param_tensor.dim(), g_avg_tensor.dim());
    CAFFE_ENFORCE_EQ(param_tensor.dim(), g2_avg_tensor.dim());
    CAFFE_ENFORCE_EQ(param_tensor.dim(), grad_tensor.dim());
    for (int i = 0; i < param_tensor.dim(); ++i) {
      CAFFE_ENFORCE_EQ(param_tensor.dim32(i), moment_tensor.dim32(i));
      CAFFE_ENFORCE_EQ(param_tensor.dim32(i), g_avg_tensor.dim32(i));
      CAFFE_ENFORCE_EQ(param_tensor.dim32(i), g2_avg_tensor.dim32(i));
      CAFFE_ENFORCE_EQ(param_tensor.dim32(i), grad_tensor.dim32(i));
    }

        iter_ = OperatorStorage::Input<Tensor>(ITER, CPU).template data<int64_t>()[0];

        D_ = param_tensor.numel();

        // Input data - persistent memory for internal scalars
        // Note: Memory for these scalars is being allocated during initialization
        //       of the network. If you want to add / remove a scalar, make a
        //       suitable change of memory size in the initialization.
        const T* memory_it = scalars_memory_ - 1;
        g_norm_avg_ = ++memory_it;
        g_norm2_avg_ = ++memory_it;
        g_norm2_min_avg_ = ++memory_it;
        g_norm2_max_avg_ = ++memory_it;
        distance_avg_ = ++memory_it;

    // Output data

    #define CAFFE2_YF_READ_OUTPUT(OUTPUT_NAME, VAR_NAME)                           \
      auto VAR_NAME##_out_tensor =                                                 \
          Output(OUTPUT_##OUTPUT_NAME, VAR_NAME##_tensor.sizes(), at::dtype<T>()); \
      VAR_NAME##_out_ = VAR_NAME##_out_tensor->template mutable_data<T>();

        CAFFE2_YF_READ_OUTPUT(PARAM, param)
        CAFFE2_YF_READ_OUTPUT(MOMENT, moment)
        CAFFE2_YF_READ_OUTPUT(LR_AVG, lr_avg)
        CAFFE2_YF_READ_OUTPUT(MU_AVG, mu_avg)
        CAFFE2_YF_READ_OUTPUT(CURV_WIN, curv_win)
        CAFFE2_YF_READ_OUTPUT(G_AVG, g_avg)
        CAFFE2_YF_READ_OUTPUT(G2_AVG, g2_avg)
        CAFFE2_YF_READ_OUTPUT(SCALARS_MEMORY, scalars_memory)
    #undef CAFFE2_YF_READ_OUTPUT

        T* out_memory_it = scalars_memory_out_ - 1;
        g_norm_avg_out_ = ++out_memory_it;
        g_norm2_avg_out_ = ++out_memory_it;
        g_norm2_min_avg_out_ = ++out_memory_it;
        g_norm2_max_avg_out_ = ++out_memory_it;
        distance_avg_out_ = ++out_memory_it;

    #define CAFFE2_YF_INIT_VECTOR(NAME) \
        ReinitializeTensor(&NAME##_tensor_, {D_}, at::dtype<T>().device(Context::GetDeviceType())); \
        NAME##_ = NAME##_tensor_.template mutable_data<T>();

        CAFFE2_YF_INIT_VECTOR(aux_vector)
        CAFFE2_YF_INIT_VECTOR(g_deb)
        CAFFE2_YF_INIT_VECTOR(g2_deb)
        CAFFE2_YF_INIT_VECTOR(g_deb2)
    #undef CAFFE2_YF_INIT_VECTOR

    #define CAFFE2_YF_INIT_SCALAR(NAME) \
          ReinitializeTensor(&NAME##_tensor_, {1}, at::dtype<T>().device(Context::GetDeviceType())); \
          NAME##_ = NAME##_tensor_.template mutable_data<T>();

        CAFFE2_YF_INIT_SCALAR(aux_scalar)
        CAFFE2_YF_INIT_SCALAR(distance)
        CAFFE2_YF_INIT_SCALAR(distance_deb)
        CAFFE2_YF_INIT_SCALAR(g_norm)
        CAFFE2_YF_INIT_SCALAR(g_norm_deb)
        CAFFE2_YF_INIT_SCALAR(g_norm2)
        CAFFE2_YF_INIT_SCALAR(g_norm2_max)
        CAFFE2_YF_INIT_SCALAR(g_norm2_max_deb)
        CAFFE2_YF_INIT_SCALAR(g_norm2_min)
        CAFFE2_YF_INIT_SCALAR(g_norm2_min_deb)
        CAFFE2_YF_INIT_SCALAR(g_norm2_deb)
        CAFFE2_YF_INIT_SCALAR(lr)
        CAFFE2_YF_INIT_SCALAR(lr_deb)
        CAFFE2_YF_INIT_SCALAR(mu_deb)
        CAFFE2_YF_INIT_SCALAR(mu)
        CAFFE2_YF_INIT_SCALAR(variance)
    #undef CAFFE2_YF_INIT_SCALAR

        debias_factor_ = ZeroDebiasFactor();
        MomentumSgdUpdate();
        AfterApply();
        return true;
        */
    }
}

impl<T,CpuContext> YellowFinOp<T,CpuContext> {
    
    /**
      | GetLrMu and MomentumSgdUpdate have
      | different implementations for GPU
      | and CPU. All other methods are generic.
      |
      */
    #[inline] pub fn get_lr_mu(&mut self)  {

        todo!();
        /*
            const T curv_ratio = std::sqrt(*g_norm2_max_deb_ / *g_norm2_min_deb_);  \
            const T mu_limit = (curv_ratio - 1.0f) / (curv_ratio + 1.0f);           \
            const T pre_p = *distance_deb_ * *g_norm2_min_deb_;                     \
            const T p = (pre_p * pre_p) / (2.0f * *variance_);                      \
            const T w3 = (-std::sqrt(p * p + 4.0f / 27.0f * p * p * p) - p) / 2.0f; \
            const T w3_sign = w3 > 0.0f ? 1.0f : -1.0f;                             \
            const T w = w3_sign * std::pow(std::abs(w3), 1.0f / 3.0f);              \
            const T y = w - p / 3.0f / w;                                           \
            const T root = y + 1.0f;                                                \
            *mu_ = std::max(root * root, mu_limit * mu_limit);                      \
            *lr_ = std::pow(1.0f - std::sqrt(*mu_), 2) / *g_norm2_min_deb_;         \
            MovingAverage(1, mu_, mu_avg_, mu_avg_out_, mu_deb_);                   \
            MovingAverage(1, lr_, lr_avg_, lr_avg_out_, lr_deb_);                   \
        */
    }
    
    /**
      | Usually moment_ == moment_out_ && param_
      | == param_out_
      |
      */
    #[inline] pub fn momentum_sgd_update(&mut self)  {
        
        todo!();
        /*
            const T mu = *mu_avg_out_;                                                 \
            const T lr = *lr_avg_out_;                                                 \
            if (!nesterov_) {                                                          \
              for (int i = 0; i < D_; ++i) {                                           \
                moment_out_[i] = mu * moment_[i] + lr * grad_[i];                      \
                param_out_[i] = param_[i] - moment_out_[i];                            \
              }                                                                        \
            } else {                                                                   \
              for (int i = 0; i < D_; ++i) {                                           \
                const T moment_i = moment_[i];                                         \
                moment_out_[i] = mu * moment_i + lr * grad_[i];                        \
                param_out_[i] = param_[i] - (1 + mu) * moment_out_[i] + mu * moment_i; \
              }                                                                        \
            }                                                                          \
        */
    }
}

impl<T,Context> Operator for YellowFinOp<T,Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;

}

register_cpu_operator!{
    YellowFin, 
    YellowFinOp<f32, CPUContext>
}

num_inputs!{YellowFin, 10}

num_outputs!{YellowFin, 8}

allow_inplace!{YellowFin, vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]}

inputs!{YellowFin,
    0 => ("param",                  "Parameters to be updated"),
    1 => ("moment",                 "Momentum"),
    2 => ("lr",                     "Learning rate"),
    3 => ("mu",                     "Momentum coefficient"),
    4 => ("curv_win",               "Memory for latest curvature ranges"),
    5 => ("g_avg",                  "Moving average of gradient"),
    6 => ("g2_avg",                 "Moving average of squared gradient"),
    7 => ("scalars_memory",         "Memory for stateful scalars"),
    8 => ("grad",                   "Gradient computed"),
    9 => ("iter",                   "Iteration number")
}

outputs!{YellowFin,
    0 => ("output_param",           "Parameters to be updated"),
    1 => ("output_moment",          "Momentum"),
    2 => ("output_lr",              "Output learning rate"),
    3 => ("output_mu",              "Output momentum coefficient"),
    4 => ("output_curv_win",        "Output memory for latest curvature ranges"),
    5 => ("output_g_avg",           "Output moving average of gradient"),
    6 => ("output_g2_avg",          "Output moving average of squared gradient"),
    7 => ("output_scalars_memory",  "Output memory for stateful scalars")
}

args!{YellowFin,
    0 => ("beta",                   "Default 0.999"),
    1 => ("curv_win_width",         "Default 20"),
    2 => ("epsilon",                "Default 1e-6"),
    3 => ("nesterov",               "Default false"),
    4 => ("zero_debias",            "Default true")
}

should_not_do_gradient!{YellowFin}
