crate::ix!();


#[inline] pub fn adagrad_update<Context>(
    n:            i32,
    w:            *const f32,
    g:            *const f32,
    h:            *const f32,
    nw:           *mut f32,
    nh:           *mut f32,
    epsilon:      f32,
    decay:        f32,
    lr:           *const f32,
    context:      *mut Context,
    weight_decay: f32) 
{
    todo!();
    /*
        return adagrad_update(
          N, w, g, h, nw, nh, epsilon, decay, lr[0], weight_decay);
    */
}

#[inline] pub fn adagrad_update_output_effective_lr<Context>(
    n:                i32,
    param_in:         *const f32,
    grad_in:          *const f32,
    moment_in:        *const f32,
    param_out:        *mut f32,
    moment_out:       *mut f32,
    effective_lrout:  *mut f32,
    epsilon:          f32,
    decay:            f32,
    lr:               *const f32,
    context:          *mut Context,
    weight_decay:     f32)
{
    todo!();
    /*
        for (auto i = 0; i < N; ++i) {
        float grad = std::fma(weight_decay, paramIn[i], gradIn[i]);
        float moment = momentOut[i] = decay * momentIn[i] + grad * grad;
        float effective_lr = effectiveLROut[i] =
            lr[0] / (std::sqrt(moment) + epsilon);
        paramOut[i] = paramIn[i] + effective_lr * grad;
      }
    */
}

#[inline] pub fn adagrad_update_output_effective_lr_and_update<Context>(
    n:                i32,
    param_in:         *const f32,
    grad_in:          *const f32,
    moment_in:        *const f32,
    param_out:        *mut f32,
    moment_out:       *mut f32,
    effective_lrout:  *mut f32,
    update_out:       *mut f32,
    epsilon:          f32,
    decay:            f32,
    lr:               *const f32,
    context:          *mut Context,
    weight_decay:     f32) 
{
    todo!();
    /*
        for (auto i = 0; i < N; ++i) {
        float grad = std::fma(weight_decay, paramIn[i], gradIn[i]);
        float moment = momentOut[i] = decay * momentIn[i] + grad * grad;
        float effective_lr = effectiveLROut[i] =
            lr[0] / (std::sqrt(moment) + epsilon);
        float update = updateOut[i] = effective_lr * grad;
        paramOut[i] = paramIn[i] + update;
      }
    */
}

/**
 | Computes the AdaGrad update for an input gradient
 | and accumulated history. Concretely, given inputs
 | (param, grad, moment, learning_rate), computes
 |
 |     new_moment = moment + square(grad)
 |     effective_lr = learning_rate / (sqrt(new_moment) + epsilon)
 |     update = learning_rate * grad / (sqrt(new_moment) + epsilon)
 |     new_param = param + update
 | and returns (new_param, new_moment).
 |
 | Optionally returns effective_lr and update as
 | well. 
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AdagradOp<Context> {
    storage:      OperatorStorage,
    context:      Context,
    epsilon:      f32,
    decay:        f32,
    weight_decay: f32,
}

should_not_do_gradient!{Adagrad}

register_cpu_operator!{Adagrad, AdagradOp<CPUContext>}

// For backward compatibility
register_cpu_operator_with_engine!{
    Adagrad, 
    SIMD, 
    AdagradOp::<CPUContext>
}

num_inputs!{Adagrad, 4}

num_outputs!{Adagrad, (2,4)}

inputs!{Adagrad, 
    0 => ("param", "Parameters to be updated"),
    1 => ("moment", "Moment history"),
    2 => ("grad", "Gradient computed"),
    3 => ("lr", "learning rate")
}

outputs!{Adagrad, 
    0 => ("output_param", "Updated parameters"),
    1 => ("output_moment", "Updated moment"),
    2 => ("output_effective_lr", "(optional) Effective learning rate"),
    3 => ("output_update", "(optional) Actual update that is applied.")
}

args!{Adagrad, 
    0 => ("epsilon", "Default 1e-5"),
    1 => ("decay", "Default 1. If it is in (0, 1), the gradient square sum is decayed by this factor.")
}

cost_inference_function!{Adagrad, /* (
        OpSchema::CostInferenceFunctionType(CostInferenceForAdagrad)) */ }

allow_inplace!{Adagrad, vec![(0, 0), (1, 1)]}

impl<Context> AdagradOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(
            Input(GRAD).numel(),
            Input(MOMENT_1).numel(),
            "PARAM size: ",
            Input(PARAM).numel(),
            ", GRAD size: ",
            Input(GRAD).numel(),
            ", MOMENT_1 size: ",
            Input(MOMENT_1).numel(),
            ", LR size: ",
            Input(LR).numel());

        CAFFE_ENFORCE_EQ(Input(GRAD).numel(), Input(PARAM).numel());
        Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
        Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));
        if (OutputSize() == 2) {
          adagrad_update<Context>(
              Input(GRAD).numel(),
              Input(PARAM).template data<float>(),
              Input(GRAD).template data<float>(),
              Input(MOMENT_1).template data<float>(),
              Output(OUTPUT_PARAM)->template mutable_data<float>(),
              Output(OUTPUT_MOMENT_1)->template mutable_data<float>(),
              epsilon_,
              decay_,
              Input(LR).template data<float>(),
              &context_,
              weight_decay_);
        } else if (OutputSize() == 3) {
          Output(OUTPUT_EFFECTIVE_LR)->ResizeLike(Input(GRAD));
          adagrad_update_output_effective_lr<Context>(
              Input(GRAD).numel(),
              Input(PARAM).template data<float>(),
              Input(GRAD).template data<float>(),
              Input(MOMENT_1).template data<float>(),
              Output(OUTPUT_PARAM)->template mutable_data<float>(),
              Output(OUTPUT_MOMENT_1)->template mutable_data<float>(),
              Output(OUTPUT_EFFECTIVE_LR)->template mutable_data<float>(),
              epsilon_,
              decay_,
              Input(LR).template data<float>(),
              &context_,
              weight_decay_);
        } else {
          Output(OUTPUT_EFFECTIVE_LR)->ResizeLike(Input(GRAD));
          Output(OUTPUT_UPDATE)->ResizeLike(Input(GRAD));
          adagrad_update_output_effective_lr_and_update<Context>(
              Input(GRAD).numel(),
              Input(PARAM).template data<float>(),
              Input(GRAD).template data<float>(),
              Input(MOMENT_1).template data<float>(),
              Output(OUTPUT_PARAM)->template mutable_data<float>(),
              Output(OUTPUT_MOMENT_1)->template mutable_data<float>(),
              Output(OUTPUT_EFFECTIVE_LR)->template mutable_data<float>(),
              Output(OUTPUT_UPDATE)->template mutable_data<float>(),
              epsilon_,
              decay_,
              Input(LR).template data<float>(),
              &context_,
              weight_decay_);
        }

        return true;
        */
    }
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator(operator_def, ws),
            epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)),
            decay_(this->template GetSingleArgument<float>("decay", 1.0f)),
            weight_decay_( this->template GetSingleArgument<float>("weight_decay", 0.f)) 

        VLOG(1) << "gradient optimization operator in use: "
                << "AdagradOp"
                << " weight_decay_=" << weight_decay_;
        */
    }
}

input_tags!{
    AdagradOpInputs {
        Param,
        Moment1,
        Grad,
        LR
    }
}

output_tags!{
    AdagradOpOutputs {
        OutputParam,
        OutputMoment1,
        OutputEffectiveLR,
        OutputUpdate
    }
}

/**
  | Given inputs (param, moment, indices,
  | grad, lr), runs the dense AdaGrad update
  | on (param, grad, moment[indices],
  | lr), and returns (new_param, new_moment)
  | as in the dense case.
  |
  */
pub struct SparseAdagradOp<Context> {
    context:      Context,
    /*
  float epsilon_;
  float weight_decay_;
#if defined(USE_FBGEMM) && !defined(__NVCC__)
  fbgemm::SparseAdaGradSignature<std::int32_t>::Type kernel_i32_;
  fbgemm::SparseAdaGradSignature<std::int64_t>::Type kernel_i64_;
  std::int64_t last_block_size_{-1};
#endif
    */
}

impl<Context> Operator for SparseAdagradOp<Context> {

}

should_not_do_gradient!{SparseAdagrad}

register_cpu_operator!{SparseAdagrad, SparseAdagradOp}

// For backward compatibility
register_cpu_operator_with_engine!{SparseAdagrad, SIMD, SparseAdagradOp}

num_inputs!{SparseAdagrad, 5}

num_outputs!{SparseAdagrad, 2}

inputs!{SparseAdagrad, 
    0 => ("param", "Parameters to be updated"),
    1 => ("moment", "Moment history"),
    2 => ("indices", "Sparse indices"),
    3 => ("grad", "Gradient computed"),
    4 => ("lr", "learning rate")
}

outputs!{SparseAdagrad, 
    0 => ("output_param",    "Updated parameters"),
    1 => ("output_moment_1", "Updated moment")
}

args!{SparseAdagrad, 
    0 => ("epsilon", "Default 1e-5")
}

enforce_one_to_one_inplace!{SparseAdagrad}

cost_inference_function!{SparseAdagrad, /* (
        OpSchema::CostInferenceFunctionType(CostInferenceForSparseAdagrad)) */ }

impl<Context> RunOnDevice for SparseAdagradOp<Context> {

    #[inline] fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Enforce shapes
        // input(embedding/momentum) == outputs(embedding/momentum)
        CAFFE_ENFORCE_EQ(
            Input(PARAM).numel(),
            Input(MOMENT_1).numel(),
            "Input Param size: ",
            Input(PARAM).numel(),
            " Input Moment size: ",
            Input(MOMENT_1).numel());
        CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
        CAFFE_ENFORCE_EQ(
            Input(PARAM).size_from_dim(1),
            Input(GRAD).size_from_dim(Input(INDICES).dim()));

        return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }
}

impl<Context> SparseAdagradOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws),
            epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)),
            weight_decay_( this->template GetSingleArgument<float>("weight_decay", 0.f)) 

            VLOG(1) << "gradient optimization operator in use: "
                << "SparseAdagradOp"
                << " weight_decay_=" << weight_decay_;
            const float decay = this->template GetSingleArgument<float>("decay", 1.0);
            CAFFE_ENFORCE_EQ(
                decay, 1.0, "Decay is not supported for SparseSimdAdagradOp");
        */
    }

    #[inline] pub fn do_run_with_type<SIndex>() -> bool {
        todo!();
        /*
            const auto* lr = Input(LR).template data<float>();

            auto n = Input(INDICES).numel();

            const auto* indices = Input(INDICES).template data<SIndex>();
            const auto* gradIn = Input(GRAD).template data<float>();
            auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<float>();
            auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<float>();

            if (n == 0) {
              return true;
            }
            auto block_size = Input(GRAD).numel() / n;

            // input(grad) is compatible with size of indexes
            CAFFE_ENFORCE_EQ(
                Input(GRAD).numel() % n,
                0,
                "Incorrect gradient size:",
                Input(GRAD).numel(),
                " size of indexes:",
                n);

        #if defined(USE_FBGEMM) && !defined(__NVCC__)
            VLOG(1) << "using fbgemm::GenerateSparseAdaGrad in SparseAdagradOp";

            if (block_size != last_block_size_) {
              last_block_size_ = block_size;
              if (std::is_same<SIndex, std::int32_t>::value) {
                kernel_i32_ = fbgemm::GenerateSparseAdaGrad<std::int32_t>(
                    block_size,
                    /*rowwise=*/false,
                    /*prefetch=*/16,
                    weight_decay_ != 0.0f);
              } else {
                CAFFE_ENFORCE((std::is_same<SIndex, std::int64_t>::value));
                kernel_i64_ = fbgemm::GenerateSparseAdaGrad<std::int64_t>(
                    block_size,
                    /*rowwise=*/false,
                    /*prefetch=*/16,
                    weight_decay_ != 0.0f);
              }
            }

            int num_rows_processed;
            if (std::is_same<SIndex, std::int32_t>::value) {
              num_rows_processed = kernel_i32_(
                  n,
                  Input(PARAM).numel(),
                  paramOut,
                  gradIn,
                  momentOut,
                  reinterpret_cast<const std::int32_t*>(indices),
                  epsilon_,
                  lr[0],
                  weight_decay_,
                  /*counter=*/nullptr,
                  /*counter_halflife=*/0);
            } else {
              num_rows_processed = kernel_i64_(
                  n,
                  Input(PARAM).numel(),
                  paramOut,
                  gradIn,
                  momentOut,
                  reinterpret_cast<const std::int64_t*>(indices),
                  epsilon_,
                  lr[0],
                  weight_decay_,
                  /*counter=*/nullptr,
                  /*counter_halflife=*/0);
            }
            if (num_rows_processed < n) {
              CAFFE_ENFORCE_GE(
                  Input(PARAM).numel(),
                  (indices[num_rows_processed] + 1) * block_size,
                  this->debug_def().input(PARAM),
                  ", out of bound,  idx:",
                  indices[num_rows_processed],
                  " for input i:",
                  num_rows_processed,
                  " and block_size:",
                  block_size,
                  " max size:",
                  Input(PARAM).numel());
              return false;
            } else {
              return true;
            }
        #endif

            VLOG(1)
                << "using internal::adagrad_update_prefetch_inlined in SparseAdagradOp";

            const auto* paramIn = Input(PARAM).template data<float>();
            const auto* momentIn = Input(MOMENT_1).template data<float>();

            std::vector<float> grad(block_size);
            for (auto i = 0; i < n; ++i) {
              auto idx = indices[i];
              auto offsetI = i * block_size;
              auto offsetIdx = idx * block_size;

              // Enforce:
              // access within range
              // gradient access within range
              CAFFE_ENFORCE_GE(
                  Input(PARAM).numel(),
                  block_size + offsetIdx,
                  this->debug_def().input(PARAM),
                  ", out of bound,  idx:",
                  idx,
                  " for input i:",
                  i,
                  " and block size:",
                  block_size,
                  " max size:",
                  Input(PARAM).numel());

              if (block_size == 1) {
                float gi = std::fma(weight_decay_, paramIn[idx], gradIn[i]);
                float hi = momentOut[idx] = momentIn[idx] + gi * gi;
                paramOut[idx] = paramIn[idx] + lr[0] * gi / (std::sqrt(hi) + epsilon_);
              } else {
                // prefetching
                const int prefdist_T0 = 16;
                int i_pref = (i < n - prefdist_T0) ? i + prefdist_T0 : i;
                std::size_t idx_pref = indices[i_pref];

                internal::adagrad_update_prefetch_inlined(
                    block_size,
                    paramIn + offsetIdx,
                    &paramIn[idx_pref * block_size],
                    gradIn + offsetI,
                    momentIn + offsetIdx,
                    &momentIn[idx_pref * block_size],
                    paramOut + offsetIdx,
                    &paramOut[idx_pref * block_size],
                    momentOut + offsetIdx,
                    &momentOut[idx_pref * block_size],
                    epsilon_,
                    lr[0],
                    weight_decay_);
              }
            }
            return true;
        */
    }
}

input_tags!{
    SparseAdagradOpInputs {
        Param,
        Moment1,
        Indices,
        Grad,
        Lr
    }
}

output_tags!{
    SparseAdagradOpOutputs {
        OutputParam,
        OutputMoment1
    }
}

/**
  | Given inputs (param, moment, indices,
  | grad, lr), runs a modified sparse Adagrad
  | update on (param, grad, moment[indices],
  | lr), and returns (new_param, new_momwnr),
  | where moment is a 1D tensor with length
  | equal to the number of rows in param:
  | shape(moment) == shape(param)[0].
  | 
  | Each element of moment is applied to
  | an entire row of param, and the new moment
  | is calculated by adding the average
  | squared sum of gradients across each
  | row.
  | 
  | -----------
  | @note
  | 
  | indices must also be a 1D tensor indexing
  | into the rows of param.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RowWiseSparseAdagradOp<Context> {

    context:      Context,

    epsilon:       f32,
    weight_decay:  f32,

  /*
#if defined(USE_FBGEMM) && !defined(__NVCC__)
  fbgemm::SparseAdaGradSignature<std::int32_t>::Type kernel_i32_;
  fbgemm::SparseAdaGradSignature<std::int64_t>::Type kernel_i64_;
  std::int64_t last_block_size_{-1};
#endif
  */

}

should_not_do_gradient!{RowWiseSparseAdagrad}

num_inputs!{RowWiseSparseAdagrad, 5}

num_outputs!{RowWiseSparseAdagrad, 2}

inputs!{RowWiseSparseAdagrad, 
    0 => ("param", "Parameters to be updated"),
    1 => ("moment", "Moment history"),
    2 => ("indices", "Sparse indices"),
    3 => ("grad", "Gradient computed"),
    4 => ("lr", "learning rate")
}

outputs!{RowWiseSparseAdagrad, 
    0 => ("output_param",    "Updated parameters"),
    1 => ("output_moment_1", "Updated moment")
}

args!{RowWiseSparseAdagrad, 
    0 => ("epsilon", "Default 1e-5")
}

enforce_one_to_one_inplace!{RowWiseSparseAdagrad}

cost_inference_function!{RowWiseSparseAdagrad, /* (
        OpSchema::CostInferenceFunctionType(CostInferenceForRowWiseSparseAdagrad)) */ }

register_cpu_operator!{RowWiseSparseAdagrad, RowWiseSparseAdagradOp<CPUContext>}

/// For backward compatibility
register_cpu_operator_with_engine!{
    RowWiseSparseAdagrad,
    SIMD,
    RowWiseSparseAdagradOp::<CPUContext>
}

input_tags!{
    RowWiseSparseAdagradOp
    {
        Param,
        Moment1,
        Indices,
        Grad,
        Lr
    }
}

output_tags!{
    RowWiseSparseAdagradOp
    {
        OutputParam,
        OutputMoment1
    }
}

impl<Context> RowWiseSparseAdagradOp<Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator(operator_def, ws),
            epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)),
            weight_decay_( this->template GetSingleArgument<float>("weight_decay", 0.f)) 

        VLOG(1) << "gradient optimization operator in use: "
                << "RowWiseSparseAdagradOp"
                << " weight_decay_=" << weight_decay_;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Enforce shapes
        CAFFE_ENFORCE_EQ(Input(PARAM).sizes()[0], Input(MOMENT_1).numel());
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
            const auto* lr = Input(LR).template data<float>();
        auto* param = Output(OUTPUT_PARAM)->template mutable_data<float>();
        auto* moment = Output(OUTPUT_MOMENT_1)->template mutable_data<float>();

        const auto* indices = Input(INDICES).template data<SIndex>();
        const auto* gradIn = Input(GRAD).template data<float>();

        auto n = Input(INDICES).numel();
        if (n == 0) {
          return true;
        }

        auto block_size = Input(GRAD).numel() / n;

        // Enforce:
        // Input(embedding/momentum) == outputs(embedding/momentum)
        CAFFE_ENFORCE_EQ(
            Input(PARAM).numel() / block_size,
            Input(MOMENT_1).numel(),
            "Input Param size: ",
            Input(PARAM).numel(),
            " Block size: ",
            block_size,
            " Input Moment size: ",
            Input(MOMENT_1).numel());

        // input(grad) is compatible with size of indexes
        CAFFE_ENFORCE_EQ(
            Input(GRAD).numel() % n,
            0,
            "Incorrect gradient size:",
            Input(GRAD).numel(),
            " size of indexes:",
            n);

    #if defined(USE_FBGEMM) && !defined(__NVCC__)
        VLOG(1) << "using fbgemm::GenerateSparseAdaGrad in RowWiseSparseAdagradOp";

        if (block_size != last_block_size_) {
          last_block_size_ = block_size;
          if (std::is_same<SIndex, std::int32_t>::value) {
            kernel_i32_ = fbgemm::GenerateSparseAdaGrad<std::int32_t>(
                block_size,
                /*rowwise=*/true,
                /*prefetch=*/16,
                weight_decay_ != 0.0f);
          } else {
            CAFFE_ENFORCE((std::is_same<SIndex, std::int64_t>::value));
            kernel_i64_ = fbgemm::GenerateSparseAdaGrad<std::int64_t>(
                block_size,
                /*rowwise=*/true,
                /*prefetch=*/16,
                weight_decay_ != 0.0f);
          }
        }

        int num_rows_processed;
        if (std::is_same<SIndex, std::int32_t>::value) {
          num_rows_processed = kernel_i32_(
              n,
              Input(PARAM).numel(),
              param,
              gradIn,
              moment,
              reinterpret_cast<const std::int32_t*>(indices),
              epsilon_,
              lr[0],
              weight_decay_,
              /*counter=*/nullptr,
              /*counter_halflife=*/0);
        } else {
          num_rows_processed = kernel_i64_(
              n,
              Input(PARAM).numel(),
              param,
              gradIn,
              moment,
              reinterpret_cast<const std::int64_t*>(indices),
              epsilon_,
              lr[0],
              weight_decay_,
              /*counter=*/nullptr,
              /*counter_halflife=*/0);
        }

        if (num_rows_processed < n) {
          // Enforce:
          // access within range
          CAFFE_ENFORCE_GE(
              Input(PARAM).numel(),
              (indices[num_rows_processed] + 1) * block_size,
              this->debug_def().input(PARAM),
              ", out of bound,  idx:",
              indices[num_rows_processed],
              " for input i:",
              num_rows_processed,
              " and block size:",
              block_size,
              " max size:",
              Input(PARAM).numel());
          return false;
        } else {
          return true;
        }
    #else
        VLOG(1) << "using plain adagrad updates in RowWiseSparseAdagradOp";

        for (auto i = 0; i < n; ++i) {
          auto idx = indices[i];
          if (block_size == 1) {
            float gi = std::fma(weight_decay_, param[idx], gradIn[i]);
            float hi = moment[idx] = moment[idx] + gi * gi;
            param[idx] = param[idx] + lr[0] * gi / (std::sqrt(hi) + epsilon_);
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

            float* w = param + offsetIdx;
            const float* g = gradIn + offsetI;
            float* h = moment + idx;
            float hs = 0.;
            for (auto j = 0; j < block_size; ++j) {
              float gj = std::fma(weight_decay_, w[j], g[j]);
              hs += gj * gj;
            }
            float hi = h[0] = h[0] + hs / block_size;
            float step = lr[0] / (std::sqrt(hi) + epsilon_);
            for (auto j = 0; j < block_size; ++j) {
              float gj = std::fma(weight_decay_, w[j], g[j]);
              w[j] = w[j] + gj * step;
            }
          }
        }
        return true;
    #endif // !USE_FBGEMM
        */
    }
}

#[inline] pub fn cost_inference_for_adagrad(def: &OperatorDef, inputs: &Vec<TensorShape>) -> OpSchemaCost {
    
    todo!();
    /*
        CAFFE_ENFORCE_GE(inputs.size(), 4, "Adagrad requires at least 4 inputs");

      const TensorShape param = inputs[0];
      const TensorShape moment = inputs[1];
      const TensorShape grad = inputs[2];
      const TensorShape lr = inputs[3];

      uint64_t grad_size = nElemFromDim(grad);
      int output_size = def.output_size();

      OpSchema::Cost c;
      // +2: applying weight decay and add to grads
      // +3: updading moments
      // +3: updating effective lr (including 1 sqrt)
      // +2: updating params
      c.flops = grad_size * 10;

      uint64_t bytes_written =
          grad_size * (sizeof(param.data_type()) + sizeof(moment.data_type()));

      if (output_size == 3) {
        // also need to output effective learning rate in this case
        // assume it's the same data type as lr
        bytes_written += grad_size * sizeof(lr.data_type());
      } else if (output_size == 4) {
        // also need to output effective learning rate and updates in this case
        // assume update is the same data type as param
        bytes_written +=
            grad_size * (sizeof(lr.data_type()) + sizeof(param.data_type()));
      }
      c.bytes_written = bytes_written;
      c.bytes_read = c.bytes_written +
          grad_size * (sizeof(grad.data_type()) + sizeof(lr.data_type()));

      return c;
    */
}

#[inline] pub fn cost_inference_for_sparse_adagrad(unused: &OperatorDef, inputs: &Vec<TensorShape>) -> OpSchemaCost {
    
    todo!();
    /*
        CAFFE_ENFORCE_GE(
          inputs.size(), 4, "SparseAdagrad requires at least 4 inputs");

      const TensorShape param = inputs[0];
      const TensorShape moment = inputs[1];
      const TensorShape indices = inputs[2];
      const TensorShape grad = inputs[3];

      uint64_t n = nElemFromDim(indices);
      uint64_t grad_size = nElemFromDim(grad);

      OpSchema::Cost c;
      // See adagrad_op.h (note that decay is 1 for SparseAdagrad).
      // 2 multiplications, 3 additions, 1 division, and 1 sqrt
      // (optimistically count sqrt as one flop).
      c.flops = grad_size * 7;
      c.bytes_written =
          grad_size * (sizeof(param.data_type()) + sizeof(moment.data_type()));
      c.bytes_read = c.bytes_written + grad_size * sizeof(grad.data_type()) +
          n * sizeof(indices.data_type());

      return c;
    */
}

#[inline] pub fn cost_inference_for_row_wise_sparse_adagrad(unused: &OperatorDef, inputs: &Vec<TensorShape>) -> OpSchemaCost {
    
    todo!();
    /*
        CAFFE_ENFORCE_GE(
          inputs.size(), 5, "RowWiseSparseAdagrad requires at least 4 inputs");

      const TensorShape param = inputs[0];
      const TensorShape moment = inputs[1];
      const TensorShape indices = inputs[2];
      const TensorShape grad = inputs[3];
      const TensorShape lr = inputs[4];

      uint64_t n = nElemFromDim(indices);
      uint64_t grad_size = nElemFromDim(grad);
      OpSchema::Cost c;

      if (n > 0) {
        auto block_size = grad_size / n;
        if (block_size == 1) {
          // +2: applying weight decay and add to grads
          // +2: updading moments
          // +5: updating params
          c.flops = n * 9;
          c.bytes_written =
              n * (sizeof(param.data_type()) + sizeof(moment.data_type()));
          c.bytes_read = c.bytes_written +
              n *
                  (sizeof(grad.data_type()) + sizeof(indices.data_type()) +
                   sizeof(lr.data_type()));
        } else {
          // 5 per block (not counting index transforms)
          // 8 for each value of a block
          c.flops = n * (5 + (block_size * 8));
          c.bytes_written =
              n * sizeof(moment.data_type()) + n * block_size * (param.data_type());

          c.bytes_read = c.bytes_written + n * (sizeof(lr.data_type())) +
              2 * n * block_size *
                  (sizeof(grad.data_type()) + sizeof(param.data_type()));
        }
      }
      return c;
    */
}
