crate::ix!();


/// Adam
#[inline] pub fn adam_update<Context>(
    n:          i32,
    g:          *const f32,
    m:          *const f32,
    v:          *const f32,
    ng:         *mut f32,
    nm:         *mut f32,
    nv:         *mut f32,
    beta1:      f32,
    beta2:      f32,
    eps_hat:    f32,
    correction: f32,
    lr:         *const f32,
    context:    *mut Context) 
{
    todo!();
    /*
        for (auto i = 0; i < N; ++i) {
        float gi = g[i];
        float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
        float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
        ng[i] = lr[0] * correction * mi / (std::sqrt(vi) + eps_hat);
      }
    */
}

#[inline] pub fn adam_compute<Context>(
    n:          i32,
    w:          *const f32,
    g:          *const f32,
    m:          *const f32,
    v:          *const f32,
    nw:         *mut f32,
    nm:         *mut f32,
    nv:         *mut f32,
    beta1:      f32,
    beta2:      f32,
    eps_hat:    f32,
    correction: f32,
    lr:         *const f32,
    context:    *mut Context) {
    todo!();
    /*
        for (auto i = 0; i < N; ++i) {
        float gi = g[i];
        float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
        float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
        nw[i] = w[i] + lr[0] * correction * mi / (std::sqrt(vi) + eps_hat);
      }
    */
}

#[inline] pub fn adam_compute_output_grad<Context>(
    n:          i32,
    w:          *const f32,
    g:          *const f32,
    m:          *const f32,
    v:          *const f32,
    nw:         *mut f32,
    nm:         *mut f32,
    nv:         *mut f32,
    ng:         *mut f32,
    beta1:      f32,
    beta2:      f32,
    eps_hat:    f32,
    correction: f32,
    lr:         *const f32,
    context:    *mut Context) {
    todo!();
    /*
        for (auto i = 0; i < N; ++i) {
        float gi = g[i];
        float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
        float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
        float ngi = ng[i] = correction * mi / (std::sqrt(vi) + eps_hat);
        nw[i] = w[i] + lr[0] * ngi;
      }
    */
}

/// RAdam
#[inline] pub fn radam_update<Context>(
    n:                 i32,
    g:                 *const f32,
    m:                 *const f32,
    v:                 *const f32,
    ng:                *mut f32,
    nm:                *mut f32,
    nv:                *mut f32,
    beta1:             f32,
    beta2:             f32,
    eps_hat:           f32,
    beta1_correction:  f32,
    correction:        f32,
    rho_t:             f32,
    r_correction:      f32,
    lr:                *const f32,
    context:           *mut Context) 
{
    todo!();
    /*
        for (auto i = 0; i < N; ++i) {
        float gi = g[i];
        float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
        float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);

        if (rho_t >= 5.) {
          float r_t =
              std::sqrt(((rho_t - 4.) * (rho_t - 2.)) / rho_t) * r_correction;
          ng[i] = lr[0] * r_t * correction * mi / (std::sqrt(vi) + eps_hat);
        } else {
          ng[i] = lr[0] * beta1_correction * mi;
        }
      }
    */
}

#[inline] pub fn radam_compute<Context>(
    n:                i32,
    w:                *const f32,
    g:                *const f32,
    m:                *const f32,
    v:                *const f32,
    nw:               *mut f32,
    nm:               *mut f32,
    nv:               *mut f32,
    beta1:            f32,
    beta2:            f32,
    eps_hat:          f32,
    beta1_correction: f32,
    correction:       f32,
    rho_t:            f32,
    r_correction:     f32,
    lr:               *const f32,
    context:          *mut Context) 
{
    todo!();
    /*
        for (auto i = 0; i < N; ++i) {
        float gi = g[i];
        float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
        float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);

        if (rho_t >= 5.) {
          float r_t =
              std::sqrt(((rho_t - 4.) * (rho_t - 2.)) / rho_t) * r_correction;
          nw[i] = w[i] + lr[0] * r_t * correction * mi / (std::sqrt(vi) + eps_hat);
        } else {
          nw[i] = w[i] + lr[0] * beta1_correction * mi;
        }
      }
    */
}

#[inline] pub fn radam_compute_output_grad<Context>(
    n:                  i32,
    w:                  *const f32,
    g:                  *const f32,
    m:                  *const f32,
    v:                  *const f32,
    nw:                 *mut f32,
    nm:                 *mut f32,
    nv:                 *mut f32,
    ng:                 *mut f32,
    beta1:              f32,
    beta2:              f32,
    eps_hat:            f32,
    beta1_correction:   f32,
    correction:         f32,
    rho_t:              f32,
    r_correction:       f32,
    lr:                 *const f32,
    context:            *mut Context) 
{
    todo!();
    /*
        for (auto i = 0; i < N; ++i) {
        float gi = g[i];
        float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
        float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
        float ngi;

        if (rho_t >= 5.) {
          float r_t =
              std::sqrt(((rho_t - 4.) * (rho_t - 2.)) / rho_t) * r_correction;
          ngi = ng[i] = r_t * correction * mi / (std::sqrt(vi) + eps_hat);
        } else {
          ngi = ng[i] = beta1_correction * mi;
        }
        nw[i] = w[i] + lr[0] * ngi;
      }
    */
}

/**
 | Computes the Adam update
 | (https://arxiv.org/abs/1412.6980) for an input
 | gradient and momentum parameters. Concretely,
 | given inputs (param, m1, m2, grad, lr, iters),
 |
 |     t = iters + 1
 |     correction_multiplier = sqrt(1 - power(beta2, t)) /
 |       (1 - power(beta1, t))
 |     m1_o = (beta1 * m1) + (1 - beta1) * grad
 |     m2_o = (beta2 * m2) + (1 - beta2) * np.square(grad)
 |     grad_o = correction_multiplier * m1_o / \
 |         (sqrt(m2_o) + epsilon)
 |     param_o = param + lr * grad_o
 |
 | and returns (param_o, m1_o, m2_o, grad_o), in
 | which grad_o is an optional output
 */
pub struct AdamOp<T, Context> {
    context: Context,
    beta1:   T,
    beta2:   T,
    epsilon: T,
}

impl<T,Context> Operator for AdamOp<T,Context> {

}

register_cpu_operator!{Adam, AdamOp<f32, CPUContext>}
num_inputs!{Adam, 6}
num_outputs!{Adam, (3, 4)}

allow_inplace!{Adam, vec![(0, 0), (1, 1), (2, 2)]}

device_inference_function!{
    /*
    Adam,
    [](const OperatorDef& def) {
      auto op_device =
          def.has_device_option() ? def.device_option() : DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), op_device);
      vector<DeviceOption> out_dev(def.output_size(), op_device);
      // ITER input lives on CPU
      in_dev[5] = DeviceOption();
      return std::make_pair(in_dev, out_dev);
    }
    */
}

inputs!{Adam,
    0 => ("param",            "Parameters to be updated"),
    1 => ("moment_1",         "First moment history"),
    2 => ("moment_2",         "Second moment history"),
    3 => ("grad",             "Gradient computed"),
    4 => ("lr",               "learning rate"),
    5 => ("iter",             "iteration number")
}

outputs!{Adam,
    0 => ("output_param",     "Updated parameters"),
    1 => ("output_moment_1",  "Updated first moment"),
    2 => ("output_moment_2",  "Updated second moment"),
    3 => ("output_grad",      "Optional Effective gradient")
}

args!{Adam,
    0 => ("beta1",            "Default 0.9"),
    1 => ("beta2",            "Default 0.999"),
    2 => ("epsilon",          "Default 1e-5")
}

should_not_do_gradient!{Adam}

impl<T, Context> Default for AdamOp<T, Context> {
    fn default() -> Self {
        todo!();
        /*
        Self {
            context: Operator::default(),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
        */
    }
}

impl<T, Context> AdamOp<T, Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator(operator_def, ws),
            beta1_(this->template GetSingleArgument<float>("beta1", 0.9f)),
            beta2_(this->template GetSingleArgument<float>("beta2", 0.999f)),
            epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Iter live on the CPU
        CAFFE_ENFORCE(OperatorStorage::InputIsTensorType(ITER, CPU));
        CAFFE_ENFORCE(Input(LR).numel() == 1);
        CAFFE_ENFORCE(Input(GRAD).numel() == Input(PARAM).numel());
        CAFFE_ENFORCE(Input(GRAD).numel() == Input(MOMENT_1).numel());
        CAFFE_ENFORCE(Input(GRAD).numel() == Input(MOMENT_2).numel());
        Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
        Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));
        Output(OUTPUT_MOMENT_2)->ResizeLike(Input(MOMENT_2));

        const auto iter =
            OperatorStorage::Input<Tensor>(ITER, CPU).template data<int64_t>()[0];

        const auto t = iter + 1;
        const auto correction =
            std::sqrt(T(1.) - std::pow(beta2_, t)) / (T(1.) - std::pow(beta1_, t));
        if (OutputSize() == 3) {
          adam_compute<Context>(
              Input(GRAD).numel(),
              Input(PARAM).template data<T>(),
              Input(GRAD).template data<T>(),
              Input(MOMENT_1).template data<T>(),
              Input(MOMENT_2).template data<T>(),
              Output(OUTPUT_PARAM)->template mutable_data<T>(),
              Output(OUTPUT_MOMENT_1)->template mutable_data<T>(),
              Output(OUTPUT_MOMENT_2)->template mutable_data<T>(),
              beta1_,
              beta2_,
              epsilon_,
              correction,
              Input(LR).template data<T>(),
              &context_);
        } else {
          Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
          adam_compute_output_grad<Context>(
              Input(GRAD).numel(),
              Input(PARAM).template data<T>(),
              Input(GRAD).template data<T>(),
              Input(MOMENT_1).template data<T>(),
              Input(MOMENT_2).template data<T>(),
              Output(OUTPUT_PARAM)->template mutable_data<T>(),
              Output(OUTPUT_MOMENT_1)->template mutable_data<T>(),
              Output(OUTPUT_MOMENT_2)->template mutable_data<T>(),
              Output(OUTPUT_GRAD)->template mutable_data<T>(),
              beta1_,
              beta2_,
              epsilon_,
              correction,
              Input(LR).template data<T>(),
              &context_);
        }

        return true;
        */
    }
}

input_tags!{
    AdamOp {
        Param,
        Moment1,
        Moment2,
        Grad,
        Lr,
        Iter
    }
}

output_tags!{
    AdamOp {
        OutputParam,
        OutputMoment1,
        OutputMoment2,
        OutputGrad
    }
}

/**
  | Computes the Adam Update for the sparse
  | case.
  | 
  | Given inputs (param, moment1, moment2,
  | indices, grad, lr, iter), runs the dense
  | Adam on (param, moment1[indices],
  | momemnt2[indices], lr, iter) and returns
  | (new_param, new_moment1, new_moment2)
  | as in dense case.
  | 
  | Adam can be customized as Rectified
  | Adam (RAdam) by setting enableRAdam
  | = true.
  |
  */
pub struct SparseAdamOp<T, Context> {
    context: Context,
    beta1:        T,
    beta2:        T,
    epsilon:      T,
    enable_radam: T,
}

impl<T,Context> Operator for SparseAdamOp<T,Context> {

}

register_cpu_operator!{SparseAdam, SparseAdamOp<f32, CPUContext>}

num_inputs!{SparseAdam, 7}

num_outputs!{SparseAdam, (3, 4)}

enforce_inplace!{SparseAdam, vec![(0, 0), (1, 1), (2, 2)]}

device_inference_function!{
    /* 
    SparseAdam, 
     [](const OperatorDef& def) {
      auto op_device =
          def.has_device_option() ? def.device_option() : DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), op_device);
      vector<DeviceOption> out_dev(def.output_size(), op_device);
      // ITER input lives on CPU
      in_dev[6] = DeviceOption();
      return std::make_pair(in_dev, out_dev);
    }
*/
}

inputs!{SparseAdam,
    0 => ("param",               "Parameters to be updated"),
    1 => ("moment_1",            "First moment history"),
    2 => ("moment_2",            "Second moment history"),
    3 => ("indices",             "Sparse indices"),
    4 => ("grad",                "Gradient computed"),
    5 => ("lr",                  "learning rate"),
    6 => ("iter",                "iteration number")
}

outputs!{SparseAdam,
    0 => ("output_param",        "Updated parameters"),
    1 => ("output_moment_1",     "Updated first moment"),
    2 => ("output_moment_2",     "Updated second moment"),
    3 => ("output_grad",         "Optional Effective gradient")
}

args!{SparseAdam,
    0 => ("beta1",               "Default 0.9"),
    1 => ("beta2",               "Default 0.999"),
    2 => ("epsilon",             "Default 1e-5"),
    3 => ("enableRAdam",         "Default false")
}

should_not_do_gradient!{SparseAdam}

impl<T,Context> SparseAdamOp<T,Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator(operator_def, ws),
            beta1_(this->template GetSingleArgument<float>("beta1", 0.9f)),
            beta2_(this->template GetSingleArgument<float>("beta2", 0.999f)),
            epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)),
            enableRAdam_( this->template GetSingleArgument<bool>("enableRAdam", false))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Enforce shapes
        CAFFE_ENFORCE_EQ(Input(PARAM).numel(), Input(MOMENT_1).numel());
        CAFFE_ENFORCE_EQ(Input(PARAM).numel(), Input(MOMENT_2).numel());
        CAFFE_ENFORCE_EQ(
            Input(PARAM).size_from_dim(1),
            Input(GRAD).size_from_dim(Input(INDICES).dim()));
        CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);

        return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }

    #[inline] pub fn do_run_with_type<SIndex>() -> bool {
        todo!();
        /*
            const auto* lr = Input(LR).template data<T>();
            const auto iter =
                OperatorStorage::Input<Tensor>(ITER, CPU).template data<int64_t>()[0];

            const auto t = iter + 1;
            const auto beta1_correction = T(1.) / (T(1.) - std::pow(beta1_, t));
            const auto beta2_correction =
                T(1.) / std::sqrt(T(1.) - std::pow(beta2_, t));
            const auto correction = beta1_correction / beta2_correction;
            const auto rho_inf = T(2.) / (T(1.) - beta2_) - T(1.);
            const auto rho_t = rho_inf -
                T(2.) * t * std::pow(beta2_, t) / (T(1.) - std::pow(beta2_, t));
            const T r_correction = enableRAdam_
                ? std::sqrt(rho_inf / ((rho_inf - T(4.)) * (rho_inf - T(2.))))
                : 0;

            auto block_size = Input(PARAM).numel() / Input(PARAM).size(0);
            auto n = Input(GRAD).numel() / block_size;

            const auto* paramIn = Input(PARAM).template data<T>();
            const auto* indices = Input(INDICES).template data<SIndex>();
            const auto* gradIn = Input(GRAD).template data<T>();
            const auto* moment1In = Input(MOMENT_1).template data<T>();
            const auto* moment2In = Input(MOMENT_2).template data<T>();
            auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<T>();
            auto* moment1Out = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();
            auto* moment2Out = Output(OUTPUT_MOMENT_2)->template mutable_data<T>();

            if (OutputSize() == 3) {
              for (auto i = 0; i < n; ++i) {
                auto idx = indices[i];

                if (block_size == 1) {
                  float gi = gradIn[i];
                  float mi = moment1Out[idx] =
                      moment1In[idx] * beta1_ + gi * (1 - beta1_);
                  float vi = moment2Out[idx] =
                      moment2In[idx] * beta2_ + gi * gi * (1 - beta2_);

                  if (!enableRAdam_) {
                    paramOut[idx] = paramIn[idx] +
                        lr[0] * correction * mi / (std::sqrt(vi) + epsilon_);
                  } else {
                    // the SMA condition follows author's implementation
                    // 5 is more conservative since it's an approximated value
                    if (rho_t >= T(5.)) {
                      float r_t =
                          std::sqrt(((rho_t - T(4.)) * (rho_t - T(2.))) / rho_t) *
                          r_correction;
                      // epsilon_ is not included in paper, but it is added in author's
                      // implementation:
                      // https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py#L85
                      paramOut[idx] = paramIn[idx] +
                          lr[0] * r_t * correction * mi / (std::sqrt(vi) + epsilon_);
                    } else {
                      paramOut[idx] = paramIn[idx] + lr[0] * beta1_correction * mi;
                    }
                  }
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
                  if (!enableRAdam_) {
                    adam_compute(
                        block_size,
                        paramIn + offsetIdx,
                        gradIn + offsetI,
                        moment1In + offsetIdx,
                        moment2In + offsetIdx,
                        paramOut + offsetIdx,
                        moment1Out + offsetIdx,
                        moment2Out + offsetIdx,
                        beta1_,
                        beta2_,
                        epsilon_,
                        correction,
                        lr,
                        &context_);
                  } else {
                    radam_compute(
                        block_size,
                        paramIn + offsetIdx,
                        gradIn + offsetI,
                        moment1In + offsetIdx,
                        moment2In + offsetIdx,
                        paramOut + offsetIdx,
                        moment1Out + offsetIdx,
                        moment2Out + offsetIdx,
                        beta1_,
                        beta2_,
                        epsilon_,
                        beta1_correction,
                        correction,
                        rho_t,
                        r_correction,
                        lr,
                        &context_);
                  }
                }
              }
            } else {
              Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
              auto* gradOut = Output(OUTPUT_GRAD)->template mutable_data<T>();
              for (auto i = 0; i < n; ++i) {
                auto idx = indices[i];

                if (block_size == 1) {
                  float gi = gradIn[i];
                  float mi = moment1Out[idx] =
                      moment1In[idx] * beta1_ + gi * (1 - beta1_);
                  float vi = moment2Out[idx] =
                      moment2In[idx] * beta2_ + gi * gi * (1 - beta2_);
                  float ngi;

                  if (!enableRAdam_) {
                    ngi = gradOut[i] = correction * mi / (std::sqrt(vi) + epsilon_);
                  } else {
                    if (rho_t >= T(5.)) {
                      float r_t =
                          std::sqrt(((rho_t - T(4.)) * (rho_t - T(2.))) / rho_t) *
                          r_correction;
                      ngi = gradOut[i] =
                          r_t * correction * mi / (std::sqrt(vi) + epsilon_);
                    } else {
                      ngi = gradOut[i] = beta1_correction * mi;
                    }
                  }

                  paramOut[idx] = paramIn[idx] + lr[0] * ngi;
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
                  if (!enableRAdam_) {
                    adam_compute_output_grad(
                        block_size,
                        paramIn + offsetIdx,
                        gradIn + offsetI,
                        moment1In + offsetIdx,
                        moment2In + offsetIdx,
                        paramOut + offsetIdx,
                        moment1Out + offsetIdx,
                        moment2Out + offsetIdx,
                        gradOut + offsetI,
                        beta1_,
                        beta2_,
                        epsilon_,
                        correction,
                        lr,
                        &context_);
                  } else {
                    radam_compute_output_grad(
                        block_size,
                        paramIn + offsetIdx,
                        gradIn + offsetI,
                        moment1In + offsetIdx,
                        moment2In + offsetIdx,
                        paramOut + offsetIdx,
                        moment1Out + offsetIdx,
                        moment2Out + offsetIdx,
                        gradOut + offsetI,
                        beta1_,
                        beta2_,
                        epsilon_,
                        beta1_correction,
                        correction,
                        rho_t,
                        r_correction,
                        lr,
                        &context_);
                  }
                }
              }
            }
            return true;
        */
    }
}

input_tags!{
    SparseAdamOp {
        Param,
        Moment1,
        Moment2,
        Indices,
        Grad,
        Lr,
        Iter
    }
}

output_tags!{
    SparseAdamOp {
        OutputParam,
        OutputMoment1,
        OutputMoment2,
        OutputGrad
    }
}

/**
  | Computes a modified Adam Update for
  | the sparse case.
  | 
  | Given inputs (param, moment1, moment2,
  | indices, grad, lr, iter), runs the Adam
  | update on (param, moment1[indices],
  | moment2[indices], lr, iter) and returns
  | (new_param, new_moment1, new_moment2),
  | where moment2 is a 1D tensor with length
  | equal to the number of rows in param:
  | 
  | shape(moment2) == shape(param)[0].
  | Each element of moment2 is applied to
  | an entire row of param, and the new moment2
  | values are calculated by averaging
  | across the row.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RowWiseSparseAdamOp<T, Context> {
    context: Context,
    beta1:   T,
    beta2:   T,
    epsilon: T,
}

impl<T,Context> RowWiseSparseAdamOp<T,Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator(operator_def, ws),
            beta1_(this->template GetSingleArgument<float>("beta1", 0.9f)),
            beta2_(this->template GetSingleArgument<float>("beta2", 0.999f)),
            epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Enforce shapes
        CAFFE_ENFORCE_EQ(Input(PARAM).numel(), Input(MOMENT_1).numel());
        CAFFE_ENFORCE_EQ(Input(PARAM).sizes()[0], Input(MOMENT_2).numel());
        CAFFE_ENFORCE_EQ(
            Input(PARAM).size_from_dim(1),
            Input(GRAD).size_from_dim(Input(INDICES).dim()));
        CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);

        return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }

    #[inline] pub fn do_run_with_type<SIndex>() -> bool {
        todo!();
        /*
            const auto* lr = Input(LR).template data<T>();
            const auto iter =
                OperatorStorage::Input<Tensor>(ITER, CPU).template data<int64_t>()[0];

            const auto t = iter + 1;
            const auto correction =
                std::sqrt(T(1.) - std::pow(beta2_, t)) / (T(1.) - std::pow(beta1_, t));

            auto block_size = Input(PARAM).numel() / Input(PARAM).size(0);
            auto n = Input(GRAD).numel() / block_size;

            const auto* paramIn = Input(PARAM).template data<T>();
            const auto* indices = Input(INDICES).template data<SIndex>();
            const auto* gradIn = Input(GRAD).template data<T>();
            const auto* moment1In = Input(MOMENT_1).template data<T>();
            const auto* moment2In = Input(MOMENT_2).template data<T>();
            auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<T>();
            auto* moment1Out = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();
            auto* moment2Out = Output(OUTPUT_MOMENT_2)->template mutable_data<T>();

            if (OutputSize() == 3) {
              for (auto i = 0; i < n; ++i) {
                auto idx = indices[i];

                if (block_size == 1) {
                  float gi = gradIn[i];
                  float mi = moment1Out[idx] =
                      moment1In[idx] * beta1_ + gi * (1 - beta1_);
                  float vi = moment2Out[idx] =
                      moment2In[idx] * beta2_ + gi * gi * (1 - beta2_);
                  paramOut[idx] = paramIn[idx] +
                      lr[0] * correction * mi / (std::sqrt(vi) + epsilon_);

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

                  const float* w = paramIn + offsetIdx;
                  const float* g = gradIn + offsetI;
                  const float* m1 = moment1In + offsetIdx;
                  const float* m2 = moment2In + idx;
                  float* nw = paramOut + offsetIdx;
                  float* nm1 = moment1Out + offsetIdx;
                  float* nm2 = moment2Out + idx;

                  float m2_sum = 0.;
                  for (auto j = 0; j < block_size; ++j) {
                    float gj = g[j];
                    m2_sum += gj * gj;
                  }
                  float vi = nm2[0] =
                      m2[0] * beta2_ + (m2_sum / block_size) * (1 - beta2_);
                  for (auto j = 0; j < block_size; ++j) {
                    float mi = nm1[j] = m1[j] * beta1_ + g[j] * (1 - beta1_);
                    nw[j] = w[j] + lr[0] * correction * mi / (std::sqrt(vi) + epsilon_);
                  }
                }
              }
            } else {
              Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
              auto* gradOut = Output(OUTPUT_GRAD)->template mutable_data<T>();
              for (auto i = 0; i < n; ++i) {
                auto idx = indices[i];

                if (block_size == 1) {
                  float gi = gradIn[i];
                  float mi = moment1Out[idx] =
                      moment1In[idx] * beta1_ + gi * (1 - beta1_);
                  float vi = moment2Out[idx] =
                      moment2In[idx] * beta2_ + gi * gi * (1 - beta2_);
                  float ngi = gradOut[i] = correction * mi / (std::sqrt(vi) + epsilon_);
                  paramOut[idx] = paramIn[idx] + lr[0] * ngi;

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

                  const float* w = paramIn + offsetIdx;
                  const float* g = gradIn + offsetI;
                  const float* m1 = moment1In + offsetIdx;
                  const float* m2 = moment2In + idx;
                  float* nw = paramOut + offsetIdx;
                  float* nm1 = moment1Out + offsetIdx;
                  float* nm2 = moment2Out + idx;
                  float* ng = gradOut + offsetI;

                  float m2_sum = 0.;
                  for (auto j = 0; j < block_size; ++j) {
                    float gj = g[j];
                    m2_sum += gj * gj;
                  }
                  float vi = nm2[0] =
                      m2[0] * beta2_ + (m2_sum / block_size) * (1 - beta2_);
                  for (auto j = 0; j < block_size; ++j) {
                    float mi = nm1[j] = m1[j] * beta1_ + g[j] * (1 - beta1_);
                    float ngi = ng[j] = correction * mi / (std::sqrt(vi) + epsilon_);
                    nw[j] = w[j] + lr[0] * ngi;
                  }
                }
              }
            }
            return true;
        */
    }
}

input_tags!{
    RowWiseSparseAdamOp {
        Param,
        Moment1,
        Moment2,
        Indices,
        Grad,
        Lr,
        Iter
    }
}

output_tags!{
    RowWiseSparseAdamOp {
        OutputParam,
        OutputMoment1,
        OutputMoment2,
        OutputGrad
    }
}

register_cpu_operator!{ 
    RowWiseSparseAdam, 
    RowWiseSparseAdamOp<float, CPUContext>
}

num_inputs!{RowWiseSparseAdamOp, 7}

num_outputs!{RowWiseSparseAdamOp, (3, 4)}

enforce_inplace!{RowWiseSparseAdamOp, vec![(0, 0), (1, 1), (2, 2)]}

device_inference_function!{
/*
    RowWiseSparseAdamOp,
[](const OperatorDef& def) {
      auto op_device =
          def.has_device_option() ? def.device_option() : DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), op_device);
      vector<DeviceOption> out_dev(def.output_size(), op_device);
      // ITER input lives on CPU
      in_dev[6] = DeviceOption();
      return std::make_pair(in_dev, out_dev);
    }
*/
}

inputs!{RowWiseSparseAdamOp,
    0 => ("param",            "Parameters to be updated"),
    1 => ("moment_1",         "First moment history"),
    2 => ("moment_2",         "Second moment history"),
    3 => ("indices",          "Sparse indices"),
    4 => ("grad",             "Gradient computed"),
    5 => ("lr",               "learning rate"),
    6 => ("iter",             "iteration number")
}

outputs!{RowWiseSparseAdamOp,
    0 => ("output_param",     "Updated parameters"),
    1 => ("output_moment_1",  "Updated first moment"),
    2 => ("output_moment_2",  "Updated second moment"),
    3 => ("output_grad",      "Optional Effective gradient")
}

args!{RowWiseSparseAdamOp,
    0 => ("beta1",            "Default 0.9"),
    1 => ("beta2",            "Default 0.999"),
    2 => ("epsilon",          "Default 1e-5")
}

should_not_do_gradient!{RowWiseSparseAdam}
