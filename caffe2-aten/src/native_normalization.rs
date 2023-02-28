crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Normalization.h]

pub type RenormScaleFactorFn = fn(iter: &mut TensorIteratorBase, maxnorm: f64) -> c_void;

declare_dispatch!{renorm_scale_factor_fn, renorm_scale_factor_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Normalization.cpp]

pub const MIOPEN_DIM_MAX: i32 = 5;

lazy_static!{
    /*
    TORCH_META_FUNC(renorm)(const Tensor& self, const Scalar& p, i64 dim, const Scalar& maxnorm) {
      TORCH_CHECK(!p.isComplex(), "renorm: p must be real-valued");
      TORCH_CHECK(p.toDouble() > 0.0, "renorm: non-positive-norm not supported");
      TORCH_CHECK(!maxnorm.isComplex(), "renorm: maxnorm must be real-valued");
      TORCH_CHECK(maxnorm.toDouble() >= 0.0,
                  "renorm: expected maxnorm to be >= 0 but got ", maxnorm.toDouble());
      const auto ndim = self.dim();
      TORCH_CHECK(ndim > 1, "renorm: input needs at least 2 dimensions, got ", ndim, "dimensions");
      set_output(self.sizes(), self.options());
    }
    */
}

define_dispatch!{batch_norm_cpu_stub}
define_dispatch!{batch_norm_cpu_collect_stats_stub}
define_dispatch!{batch_norm_cpu_backward_stub}
define_dispatch!{renorm_scale_factor_stub}

pub fn check_dims_match_num_input_features(
        arg_name: *const u8,
        expected: i64,
        actual:   i64)  {
    
    todo!();
        /*
            TORCH_CHECK(actual == expected,
                 arg_name, " should contain ", expected, " elements not ", actual);
        */
}

#[inline] pub fn repeat_if_defined(
        t:      &Tensor,
        repeat: i64) -> Tensor {
    
    todo!();
        /*
            if (t.defined()) {
          return t.repeat(repeat);
        }
        return t;
        */
}

pub struct InvStd<T> {

}

impl InvStd<T> {

    pub fn invoke(&self, 
        var:     T,
        epsilon: f64) -> T {
        
        todo!();
        /*
            T invstd = 0;
        if (var != static_cast<T>(0) || epsilon != static_cast<T>(0)) {
          invstd = static_cast<T>(1) / sqrt(var + epsilon);
        }
        return invstd;
        */
    }
}

pub struct Var<T> {

}

impl Var<T> {
    
    pub fn invoke(&self, 
        var:     T,
        epsilon: f64) -> T {
        
        todo!();
        /*
            return var;
        */
    }
}

#[inline] pub fn is_contiguous(t: &Tensor) -> bool {
    
    todo!();
        /*
            return t.is_contiguous() || t.is_contiguous(MemoryFormat::ChannelsLast);
        */
}

pub fn batch_norm_cpu_transform_input_template<Scalar>(
        input:        &Tensor,
        weight:       &Tensor,
        bias:         &Tensor,
        save_mean:    &Tensor,
        save_invstd:  &Tensor,
        running_mean: &Tensor,
        running_var:  &Tensor,
        train:        bool,
        eps:          f64) -> (Tensor,Tensor,Tensor) {

    todo!();
        /*
            bool all_contiguous = is_contiguous(input)
          && (!weight.defined() || weight.is_contiguous())
          && (!bias.defined() || bias.is_contiguous())
          && running_mean.is_contiguous()
          && running_var.is_contiguous();

      Tensor output = empty_like(input, input.suggest_memory_format());

      // inference contiguous path
      if (all_contiguous) {
        batch_norm_cpu_stub(kCPU, output, input, weight, bias,
            save_mean, save_invstd, running_mean, running_var, train, eps);
        return make_tuple(output, save_mean, save_invstd);
      }

      const i64 ndim = input.dim();
      // Helper to convert 1d tensors to an nd tensor that broadcasts with input
      // All elements go into the channel dimension
      DimVector sizes(ndim, 1), strides(ndim, 0);
      auto as_nd = [&](const Tensor& t) {
        TORCH_INTERNAL_ASSERT(t.defined() && t.dim() == 1);
        sizes[1] = t.sizes()[0];
        strides[1] = t.strides()[0];
        return t.as_strided(sizes, strides);
      };

      auto mean = as_nd(train ? save_mean : running_mean);
      auto invstd = as_nd([&]{
        if (train) {
          return save_invstd;
        } else {
          return 1 / sqrt(running_var + eps);
        }
      }());
      auto w = weight.defined() ? as_nd(weight) :
          scalar_tensor_static(1, input.scalar_type(), kCPU);
      auto b = bias.defined() ? as_nd(bias) :
          scalar_tensor_static(0, input.scalar_type(), kCPU);

      auto iter = TensorIteratorConfig()
        .add_output(output)
        .add_input(input)
        .add_input(mean)
        .add_input(invstd)
        .add_input(w)
        .add_input(b)
        .build();

      cpu_kernel(iter, [=](Scalar input, Scalar mean, Scalar invstd, Scalar weight, Scalar bias) {
        return ((input - mean) * invstd) * weight + bias;
      });
      return make_tuple(output, save_mean, save_invstd);
        */
}


pub fn batch_norm_cpu_update_stats_template<Scalar, VarTransform>(
        input:        &Tensor,
        running_mean: &Tensor,
        running_var:  &Tensor,
        momentum:     f64,
        eps:          f64) -> (Tensor,Tensor) {

    todo!();
        /*
            using accscalar_t = acc_type<Scalar, false>;

      i64 n_input = input.size(1);
      i64 n = input.numel() / n_input;
      const i64 ndim = input.dim();

      // Reduce all dimensions except dim=1
      DimVector reduce_dims(ndim - 1);
      reduce_dims[0] = 0;
      for (i64 i = 2; i < ndim; ++i) {
        reduce_dims[i - 1] = i;
      }

      Tensor save_mean = mean(input, /*dims=*/reduce_dims);
      Tensor save_var_transform = empty({n_input}, input.options());
      auto save_mean_a = save_mean.accessor<Scalar, 1>();
      auto save_var_transform_a = save_var_transform.accessor<Scalar, 1>();

      auto running_mean_a = conditional_accessor_1d<Scalar>(running_mean);
      auto running_var_a = conditional_accessor_1d<Scalar>(running_var);

      bool all_contiguous = is_contiguous(input);
      if (all_contiguous) {
        auto _mean = empty({n_input}, input.options());
        auto _var_sum = empty({n_input}, input.options());
        auto _mean_a = _mean.accessor<Scalar, 1>();
        auto _var_sum_a = _var_sum.accessor<Scalar, 1>();

        batch_norm_cpu_collect_stats_stub(kCPU, _mean, _var_sum, input);

        parallel_for(0, n_input, 1, [&](i64 b_begin, i64 b_end) {
          for (i64 f = b_begin; f < b_end; ++f) {
            save_mean_a[f] = _mean_a[f];
            save_var_transform_a[f] = VarTransform<accscalar_t>{}(_var_sum_a[f] / n, eps);

            if (running_mean.defined()) {
              running_mean_a[f] = momentum * _mean_a[f] + (1 - momentum) * running_mean_a[f];
            }
            if (running_var.defined()) {
               accscalar_t unbiased_var = _var_sum_a[f] / (n - 1);
               running_var_a[f] = momentum * unbiased_var + (1 - momentum) * running_var_a[f];
            }
          }
        });

        return make_tuple(save_mean, save_var_transform);
      }

      // non-contiguous path
      parallel_for(0, n_input, 1, [&](i64 b_begin, i64 b_end) {
        for (i64 f = b_begin; f < b_end; ++f) {
          Tensor in = input.select(1, f);

          // compute variance per input
          auto iter = TensorIteratorConfig()
            .add_input(in)
            .build();
          accscalar_t var_sum = 0;
          auto mean = static_cast<accscalar_t>(save_mean_a[f]);
          cpu_serial_kernel(iter, [&](const Scalar i) -> void {
            var_sum += (i - mean) * (i - mean);
          });
          save_var_transform_a[f] = VarTransform<accscalar_t>{}(var_sum / n, eps);

          // update running averages
          if (running_mean.defined()) {
            running_mean_a[f] = momentum * mean + (1 - momentum) * running_mean_a[f];
          }
          if (running_var.defined()) {
            accscalar_t unbiased_var = var_sum / (n - 1);
            running_var_a[f] = momentum * unbiased_var + (1 - momentum) * running_var_a[f];
          }
        }
      });
      return make_tuple(save_mean, save_var_transform);
        */
}

pub fn batch_norm_backward_cpu_template<Scalar>(
        grad_out:        &Tensor,
        input:           &Tensor,
        weight:          &Tensor,
        running_mean:    &Tensor,
        running_var:     &Tensor,
        save_mean:       &Tensor,
        save_invstd:     &Tensor,
        train:           bool,
        eps:             f64,
        grad_input_mask: [bool; 3]) -> (Tensor,Tensor,Tensor) {

    todo!();
        /*
            using accscalar_t = acc_type<Scalar, false>;

      Tensor grad_input;
      Tensor grad_weight;
      Tensor grad_bias;
      if (grad_input_mask[0]) {
        grad_input = empty_like(input, input.suggest_memory_format());
      }
      if (grad_input_mask[1]) {
        grad_weight = empty_like(weight, MemoryFormat::Contiguous);
      }
      if (grad_input_mask[2]) {
        grad_bias = empty_like(weight, MemoryFormat::Contiguous);
      }

      // since we are directly manipulating pointers in contiguous path,
      // need to make sure input and grad_out have the same memory format.
      bool all_contiguous = is_contiguous(input)
          && is_contiguous(grad_out_)
          && input.suggest_memory_format() == grad_out_.suggest_memory_format();

      if (all_contiguous) {
        batch_norm_cpu_backward_stub(kCPU, grad_input, grad_weight, grad_bias,
            grad_out_, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
        return make_tuple(grad_input, grad_weight, grad_bias);
      }

      auto weight_a = conditional_accessor_1d<Scalar>(weight);
      auto grad_weight_a = conditional_accessor_1d<Scalar>(grad_weight);
      auto grad_bias_a = conditional_accessor_1d<Scalar>(grad_bias);

      i64 n_input = input.size(1);
      i64 n = input.numel() / n_input;

      auto save_mean_a = conditional_accessor_1d<Scalar>(save_mean);
      auto save_invstd_a = conditional_accessor_1d<Scalar>(save_invstd);

      auto running_mean_a = conditional_accessor_1d<Scalar>(running_mean);
      auto running_var_a = conditional_accessor_1d<Scalar>(running_var);

      const i64 ndim = input.dim();

      // Reduce all dimensions except dim=1
      DimVector reduce_dims(ndim - 1);
      reduce_dims[0] = 0;
      for (i64 i = 2; i < ndim; ++i) {
        reduce_dims[i - 1] = i;
      }

      auto sum = sum(grad_out_, /*dims=*/reduce_dims);
      auto sum_a = sum.accessor<Scalar, 1>();

      parallel_for(0, n_input, 1, [&](i64 b_begin, i64 b_end) {
          for (i64 f = b_begin; f < b_end; ++f) {
            Tensor in = input.select(1, f);
            Tensor grad_out = grad_out_.select(1, f);

            Scalar w = weight.defined() ? weight_a[f] : 1;

            Scalar mean, invstd;
            if (train) {
              mean = save_mean_a[f];
              invstd = save_invstd_a[f];
            } else {
              mean = running_mean_a[f];
              invstd = 1 / sqrt(running_var_a[f] + eps);
            }

            // dot product of the Q(X) and gradOuput
            accscalar_t dotp = 0;
            auto iter = TensorIteratorConfig()
              .add_input(in)
              .add_input(grad_out)
              .build();
            cpu_serial_kernel(iter, [&](const Scalar i, const Scalar go) -> void {
              dotp += (i - mean) * go;
            });

            if (grad_input_mask[0]) {
              Tensor grad_in = grad_input.select(1, f);
              if (train) {
                // when in training mode
                // Q(X) = X - E[x] ; i.e. input centered to zero mean
                // Y = Q(X) / sigma    ; i.e. BN output before weight and bias
                // dL/dX = (Q(dL/dY) - dot(Y, dL/dY) * Y) / sigma * w

                // projection of gradOutput on to output scaled by std
                Scalar k = (Scalar) dotp * invstd * invstd / n;
                {
                  auto iter = TensorIterator::unary_op(grad_in, in);
                  cpu_serial_kernel(iter, [&](const Scalar i) -> Scalar {
                    return (i - mean) * k;
                  });
                }

                Scalar grad_mean = sum_a[f] / n;
                {
                  auto iter = TensorIterator::borrowing_binary_op(grad_in, grad_in, grad_out);
                  cpu_serial_kernel(iter, [&](Scalar gi, Scalar go) -> Scalar {
                    return (go - grad_mean - gi) * invstd * w;
                  });
                }
              } else {
                // when in evaluation mode
                // Q(X) = X - running_mean  ; i.e. input centered to zero mean
                // Y = Q(X) / running_std    ; i.e. BN output before weight and bias
                // dL/dX = w / running_std
                {
                  auto iter = TensorIterator::unary_op(grad_in, grad_out);
                  cpu_serial_kernel(iter, [&](const Scalar i) -> Scalar {
                    return i * invstd * w;
                  });
                }
              }
            }
            if (grad_input_mask[1]) {
              grad_weight_a[f] = dotp * invstd;
            }

            if (grad_input_mask[2]) {
              grad_bias_a[f] = sum_a[f];
            }
          }
        });
      return make_tuple(grad_input, grad_weight, grad_bias);
        */
}

/**
  | _batch_norm_impl_index(_backward) are used in
  | the JIT be able to keep the run-time selection
  | of backends, while enabling it to keep the
  | information about the used backend, so that it
  | can use its corresponding backward
  | implementation.
  |
  | XXX: The indices of backends need to be kept
  | synchronized between this function and its
  | _backward.
  */
pub fn batch_norm_impl_index(
        input:            &Tensor,
        weight_opt:       &Option<Tensor>,
        bias_opt:         &Option<Tensor>,
        running_mean_opt: &Option<Tensor>,
        running_var_opt:  &Option<Tensor>,
        training:         bool,
        momentum:         f64,
        eps:              f64,
        cudnn_enabled:    bool) -> (Tensor,Tensor,Tensor,Tensor,i64) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;
      const Tensor& bias = value_or_else(bias_opt, [] {return Tensor();});
      const Tensor& running_mean = value_or_else(running_mean_opt, [] {return Tensor();});
      const Tensor& running_var = value_or_else(running_var_opt, [] {return Tensor();});

      auto num_features = input.sizes()[1];
      if (running_mean.defined()) {
        check_dims_match_num_input_features("running_mean", num_features, running_mean.numel());
      } else if (!training) {
        AT_ERROR("running_mean must be defined in evaluation mode");
      }
      if (running_var.defined()) {
        check_dims_match_num_input_features("running_var", num_features, running_var.numel());
      } else if (!training) {
        AT_ERROR("running_var must be defined in evaluation mode");
      }
      if (weight.defined()) {
        check_dims_match_num_input_features("weight", num_features, weight.numel());
      }
      if (bias.defined()) {
        check_dims_match_num_input_features("bias", num_features, bias.numel());
      }

      const bool use_cudnn = (
          input.is_cuda()
          && input.scalar_type() != kBFloat16 && weight.scalar_type() != kBFloat16
          && (input.scalar_type() != kHalf
            || weight.scalar_type() == kFloat)
          && weight.defined() && bias.defined()
          && ((running_mean.defined() && running_var.defined())
            || (!running_mean.defined() && !running_var.defined() && training))
          && (input.dim() >= 3)
          && ((input.size(0) <= 880801 && training) // spatial, training
              ||(input.size(0) <= 65535 && !training)) //spatial, eval
          && getCUDAHooks().compiledWithCuDNN()
          && eps >= getCUDAHooks().batchnormMinEpsilonCuDNN()
          && cudnn_enabled && getCUDAHooks().versionCuDNN() >= 5110L);

      if (use_cudnn) {
        auto input_c = input.contiguous(input.suggest_memory_format());
        auto weight_c = weight.contiguous();
        auto bias_c = bias.contiguous();
        auto rmean_c = running_mean.defined() ? running_mean.contiguous() : running_mean;
        auto rvar_c = running_var.defined() ? running_var.contiguous() : running_var;

        Tensor output, save_mean, save_var, reserve;
        tie(output, save_mean, save_var, reserve) =
            cudnn_batch_norm(input_c, weight_c, bias_c, rmean_c, rvar_c,
                                 training, momentum, eps);

        return tuple<Tensor, Tensor, Tensor, Tensor, i64>(
            output, save_mean, save_var, reserve, 1);
      }

      Tensor reserve = empty({0}, input.options().dtype(kByte));

      bool use_miopen = (input.is_cuda()
                   && input.dim() <= MIOPEN_DIM_MAX
                   && input.scalar_type() != kDouble
                   && input.scalar_type() != kBFloat16
                   && (weight.scalar_type() != kHalf)
                   && weight.defined() && bias.defined()
                   && ((running_mean.defined() && running_var.defined())
                     || (!running_mean.defined() && !running_var.defined() && training))
                   && getCUDAHooks().compiledWithMIOpen()
                   && cudnn_enabled
                   );

      if (use_miopen) {
        return tuple_cat(
                 miopen_batch_norm(
                   input.contiguous(), weight.contiguous(), bias.contiguous(),
                   running_mean.defined() ? running_mean.contiguous() : running_mean,
                   running_var.defined() ? running_var.contiguous() : running_var,
                   training, momentum, eps),
                 tuple<Tensor>(reserve),
                 make_tuple(2));
      }

      return tuple_cat(
               native_batch_norm(
                 input, weight, bias, running_mean, running_var, training, momentum, eps),
               tuple<Tensor>(reserve),
               make_tuple(0));
        */
}

pub fn batch_norm_impl_index_backward(
        impl_index:             i64,
        input:                  &Tensor,
        grad_output:            &Tensor,
        weight_opt:             &Option<Tensor>,
        running_mean_opt:       &Option<Tensor>,
        running_var_opt:        &Option<Tensor>,
        save_mean_opt:          &Option<Tensor>,
        save_var_transform_opt: &Option<Tensor>,
        train:                  bool,
        epsilon:                f64,
        output_mask:            [bool; 3],
        reserved_space:         &Tensor) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;
      const Tensor& running_mean = value_or_else(running_mean_opt, [] {return Tensor();});
      const Tensor& running_var = value_or_else(running_var_opt, [] {return Tensor();});
      const Tensor& save_mean = value_or_else(save_mean_opt, [] {return Tensor();});
      const Tensor& save_var_transform = value_or_else(save_var_transform_opt, [] {return Tensor();});

      if (impl_index == 0) {
        return native_batch_norm_backward(grad_output, input, weight, running_mean, running_var, save_mean, save_var_transform, train, epsilon, output_mask);
      } else if (impl_index == 1) {
        // TODO: _batch_norm_impl_index_backward is only used in JIT. cudnn NHWC
        // format conversion is done inside cudnn_batch_norm_backward instead
        return cudnn_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, epsilon, reservedSpace);
      } else if (impl_index == 2) {
        return miopen_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, epsilon);
      }
      TORCH_INTERNAL_ASSERT(false, "Unsupported impl_index in _batch_norm_impl_index_backward: ", impl_index);
        */
}


pub fn batch_norm(
        input:            &Tensor,
        weight_opt:       &Option<Tensor>,
        bias_opt:         &Option<Tensor>,
        running_mean_opt: &Option<Tensor>,
        running_var_opt:  &Option<Tensor>,
        training:         bool,
        momentum:         f64,
        eps:              f64,
        cudnn_enabled:    bool) -> Tensor {
    
    todo!();
        /*
            const Tensor& weight = value_or_else(weight_opt, [] {return Tensor();});
      const Tensor& bias = value_or_else(bias_opt, [] {return Tensor();});
      const Tensor& running_mean = value_or_else(running_mean_opt, [] {return Tensor();});
      const Tensor& running_var = value_or_else(running_var_opt, [] {return Tensor();});
      if (input.numel()==0){
        //don't return view of input, don't return empty tensor because it will break gradient chain
        auto out = input.clone();
        if (weight.defined()) out = out * weight[0];
        if (bias.defined()) out = out + bias[0];
        return out;
      }
      return get<0>(_batch_norm_impl_index(input, weight, bias, running_mean, running_var,
                                                    training, momentum, eps, cudnn_enabled));
        */
}

pub fn instance_norm(
        input:            &Tensor,
        weight_opt:       &Option<Tensor>,
        bias_opt:         &Option<Tensor>,
        running_mean_opt: &Option<Tensor>,
        running_var_opt:  &Option<Tensor>,
        use_input_stats:  bool,
        momentum:         f64,
        eps:              f64,
        cudnn_enabled:    bool) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;
      const Tensor& bias = value_or_else(bias_opt, [] {return Tensor();});
      const Tensor& running_mean = value_or_else(running_mean_opt, [] {return Tensor();});
      const Tensor& running_var = value_or_else(running_var_opt, [] {return Tensor();});

      TORCH_CHECK(use_input_stats || (running_mean.defined() && running_var.defined()),
               "Expected running_mean and running_var to be defined when use_input_stats is false");
      vector<i64> shape = input.sizes().vec();
      i64 b = input.size(0);
      i64 c = input.size(1);
      shape[1] = b * c;
      shape[0] = 1;

      Tensor weight_ = repeat_if_defined(weight, b);
      Tensor bias_ = repeat_if_defined(bias, b);
      Tensor running_mean_ = repeat_if_defined(running_mean, b);
      Tensor running_var_ = repeat_if_defined(running_var, b);

      auto input_reshaped = input.contiguous().view(shape);
      auto out = batch_norm(input_reshaped, weight_, bias_, running_mean_, running_var_,
                                use_input_stats, momentum, eps, cudnn_enabled);

      // we alias running_mean and running_var because they are const but we want to modify their data
      if (running_mean.defined()) {
        alias(running_mean).copy_(running_mean_.view({ b, c }).mean(0, false));
      }
      if (running_var.defined()) {
        alias(running_var).copy_(running_var_.view({ b, c }).mean(0, false));
      }

      return out.view(input.sizes());
        */
}

pub fn batch_norm_update_stats_cpu(
        self_:            &Tensor,
        running_mean_opt: &Option<Tensor>,
        running_var_opt:  &Option<Tensor>,
        momentum:         f64) -> (Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> running_mean_maybe_owned = borrow_from_optional_tensor(running_mean_opt);
      const Tensor& running_mean = *running_mean_maybe_owned;
      const Tensor& running_var = value_or_else(running_var_opt, [] {return Tensor();});

      return AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "batch_norm_update_stats_cpu", [&] {
          return batch_norm_cpu_update_stats_template<Scalar, Var>(self, running_mean, running_var, momentum, 0);
        });
        */
}

pub fn batch_norm_cpu(
        self_:            &Tensor,
        weight_opt:       &Option<Tensor>,
        bias_opt:         &Option<Tensor>,
        running_mean_opt: &Option<Tensor>,
        running_var_opt:  &Option<Tensor>,
        train:            bool,
        momentum:         f64,
        eps:              f64) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;
      const Tensor& bias = value_or_else(bias_opt, [] {return Tensor();});
      const Tensor& running_mean = value_or_else(running_mean_opt, [] {return Tensor();});
      const Tensor& running_var = value_or_else(running_var_opt, [] {return Tensor();});

      checkBackend("batch_norm_cpu", {self, weight, bias, running_mean, running_var}, Backend::CPU);

      return AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "batch_norm", [&] {
          if (!train) {
            return batch_norm_cpu_transform_input_template<Scalar>(self, weight, bias, {}, {}, running_mean, running_var, train, eps);
          } else {
            auto save_stats = batch_norm_cpu_update_stats_template<Scalar, InvStd>(self, running_mean, running_var, momentum, eps);
            return batch_norm_cpu_transform_input_template<Scalar>(self, weight, bias, get<0>(save_stats), get<1>(save_stats), running_mean, running_var, train, eps);
          }
        });
        */
}

pub fn batch_norm_backward_cpu(
        grad_out:         &Tensor,
        self_:            &Tensor,
        weight_opt:       &Option<Tensor>,
        running_mean_opt: &Option<Tensor>,
        running_var_opt:  &Option<Tensor>,
        save_mean_opt:    &Option<Tensor>,
        save_invstd_opt:  &Option<Tensor>,
        train:            bool,
        eps:              f64,
        grad_input_mask:  [bool; 3]) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;
      const Tensor& running_mean = value_or_else(running_mean_opt, [] {return Tensor();});
      const Tensor& running_var = value_or_else(running_var_opt, [] {return Tensor();});
      const Tensor& save_mean = value_or_else(save_mean_opt, [] {return Tensor();});
      const Tensor& save_invstd = value_or_else(save_invstd_opt, [] {return Tensor();});

      return AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "batch_norm_backward_cpu", [&] {
          return batch_norm_backward_cpu_template<Scalar>(grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, eps, grad_input_mask);
        });
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(renorm_out)(const Tensor& self, const Scalar& p, i64 dim,
                                const Scalar& maxnorm, const Tensor& out) {
      auto self_sizes = self.sizes();
      dim = maybe_wrap_dim(dim, self_sizes.size());

      DimVector reduce_dims(self_sizes.size());
      iota(reduce_dims.begin(), reduce_dims.end(), 0);
      reduce_dims.erase(reduce_dims.begin() + dim);

      // For cuda half, calculate norm in float precision then cast
      // normalization factor to half
      auto dtype = self.scalar_type();
      auto acc_type = toAccumulateType(dtype, /*is_cuda=*/true);
      Tensor norm;
      if (acc_type != dtype) {
        norm = linalg_vector_norm(self, p.toDouble(), reduce_dims,
                                      /*keepdim=*/true, /*dtype=*/acc_type);
      } else {
        norm = linalg_vector_norm(self, p.toDouble(), reduce_dims,
                                      /*keepdim=*/true);
      }

      auto factor = (acc_type == toValueType(dtype)) ?
          norm : empty(norm.sizes(), self.options());
      auto iter = TensorIteratorConfig()
          .add_output(factor)
          .add_input(norm)
          .set_check_mem_overlap(false)
          .cast_common_dtype_to_outputs(true)
          .build();

      renorm_scale_factor_stub(iter.device_type(), iter, maxnorm.toDouble());
      mul_outf(self, factor, const_cast<Tensor&>(out));
    }
    */
}
