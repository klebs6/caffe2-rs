crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/group_norm.h]

pub type ForwardFn = fn(
        X:     &Tensor,
        gamma: &Tensor,
        beta:  &Tensor,
        N:     i64,
        C:     i64,
        hxw:   i64,
        group: i64,
        eps:   f64,
        Y:     &mut Tensor,
        mean:  &mut Tensor,
        rstd:  &mut Tensor
) -> c_void;

pub type BackwardFn = fn(
        dy:     &Tensor,
        X:      &Tensor,
        mean:   &Tensor,
        rstd:   &Tensor,
        gamma:  &Tensor,
        N:      i64,
        C:      i64,
        hxw:    i64,
        group:  i64,
        dx:     &mut Tensor,
        dgamma: &mut Tensor,
        dbeta:  &mut Tensor
) -> c_void;

declare_dispatch!{forward_fn, GroupNormKernel}
declare_dispatch!{backward_fn, GroupNormBackwardKernel}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/group_norm.cpp]

pub fn native_group_norm(
        X:         &Tensor,
        gamma_opt: &Option<Tensor>,
        beta_opt:  &Option<Tensor>,
        N:         i64,
        C:         i64,
        hxw:       i64,
        group:     i64,
        eps:       f64) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> gamma_maybe_owned =
          borrow_from_optional_tensor(gamma_opt);
      const Tensor& gamma = *gamma_maybe_owned;
      const Tensor& beta = value_or_else(beta_opt, [] { return Tensor(); });

      TORCH_CHECK(X.is_contiguous());

      Tensor Y = native::empty_like(
          X,
          nullopt /* dtype */,
          nullopt /* layout */,
          nullopt /* device */,
          nullopt /* pin_memory */,
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      Tensor mean = empty({N, group}, X.options());
      Tensor rstd = empty({N, group}, X.options());
      GroupNormKernel(
          X.device().type(), X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
      return make_tuple(Y, mean, rstd);
        */
}

pub fn native_group_norm_backward(
    dy:              &Tensor,
    X:               &Tensor,
    mean:            &Tensor,
    rstd:            &Tensor,
    gamma_opt:       &Option<Tensor>,
    N:               i64,
    C:               i64,
    hxw:             i64,
    group:           i64,
    grad_input_mask: [bool; 3]) -> (Tensor,Tensor,Tensor) {

    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> gamma_maybe_owned =
          borrow_from_optional_tensor(gamma_opt);
      const Tensor& gamma = *gamma_maybe_owned;

      Tensor dX;
      Tensor dgamma;
      Tensor dbeta;
      if (grad_input_mask[0]) {
        dX = native::empty_like(
            X,
            nullopt /* dtype */,
            nullopt /* layout */,
            nullopt /* device */,
            nullopt /* pin_memory */,
            LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      }
      if (grad_input_mask[1]) {
        dgamma = native::empty_like(
            gamma,
            nullopt /* dtype */,
            nullopt /* layout */,
            nullopt /* device */,
            nullopt /* pin_memory */,
            LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      }
      if (grad_input_mask[2]) {
        dbeta = native::empty_like(
            gamma,
            nullopt /* dtype */,
            nullopt /* layout */,
            nullopt /* device */,
            nullopt /* pin_memory */,
            LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      }
      GroupNormBackwardKernel(
          X.device().type(),
          dY,
          X,
          mean,
          rstd,
          gamma,
          N,
          C,
          HxW,
          group,
          dX,
          dgamma,
          dbeta);
      return make_tuple(dX, dgamma, dbeta);
        */
}

pub fn group_norm(
    input:      &Tensor,
    num_groups: i64,
    weight_opt: &Option<Tensor>,
    bias_opt:   &Option<Tensor>,
    eps:        f64,
    /* cudnn_enabled, deprecated */
    _5:         bool) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned =
          borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;
      const Tensor& bias = value_or_else(bias_opt, [] { return Tensor(); });

      const i64 N = input.size(0);
      const i64 C = input.size(1);
      TORCH_CHECK(
          C % num_groups == 0,
          "Expected number of channels in input to be divisible by ",
          "num_groups, but got input of shape ",
          input.sizes(),
          " and "
          "num_groups=",
          num_groups);
      TORCH_CHECK(
          !weight.defined() || (weight.dim() == 1 && weight.numel() == C),
          "Expected weight to be a vector of size equal to the number of ",
          "channels in input, but got weight of shape ",
          weight.sizes(),
          " and input of shape ",
          input.sizes());
      TORCH_CHECK(
          !bias.defined() || (bias.dim() == 1 && bias.numel() == C),
          "Expected bias to be a vector of size equal to the number of ",
          "channels in input, but got bias of shape ",
          weight.sizes(),
          " and input of shape ",
          input.sizes());

      const auto input_shape = input.sizes();
      const i64 HxW =
          multiply_integers(input_shape.cbegin() + 2, input_shape.cend());

      const Tensor kEmpty;
      const auto& X = input.is_contiguous() ? input : input.contiguous();
      const auto& gamma = weight.defined() ? weight.contiguous() : kEmpty;
      const auto& beta = bias.defined() ? bias.contiguous() : kEmpty;
      TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
      TORCH_CHECK(!beta.defined() || beta.numel() == C);
      return get<0>(
          native_group_norm(X, gamma, beta, N, C, HxW, num_groups, eps));
        */
}

define_dispatch!{GroupNormKernel}
define_dispatch!{GroupNormBackwardKernel}

/**
  | Ported from pytorch/xla repo
  |
  */
pub fn math_group_norm(
        input:      &Tensor,
        weight_opt: &Option<Tensor>,
        bias_opt:   &Option<Tensor>,
        N:          i64,
        C:          i64,
        hxw:        i64,
        group:      i64,
        eps:        f64) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned =
          borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;
      const Tensor& bias = value_or_else(bias_opt, [] { return Tensor(); });

      auto input_shape = input.sizes();
      Tensor input_reshaped = input.view({1, N * group, N ? -1 : 1});
      auto outputs = native_batch_norm(
          input_reshaped,
          /*weight=*/{},
          /*bias=*/{},
          /*running_mean=*/{},
          /*running_var=*/{},
          /*training=*/true,
          /*momentum=*/0,
          eps);
      Tensor out = get<0>(outputs);
      out = out.view(input_shape);
      vector<i64> affine_param_shape(input.dim(), 1);
      affine_param_shape[1] = C;
      if (weight.defined() && bias.defined()) {
        out = bias.view(affine_param_shape)
                  .addcmul(out, weight.view(affine_param_shape), 1);
      } else if (weight.defined()) {
        out = out.mul(weight.view(affine_param_shape));
      } else if (bias.defined()) {
        out = out.add(bias.view(affine_param_shape));
      }
      Tensor mean = get<1>(outputs).view({N, group});
      Tensor rstd = get<2>(outputs).view({N, group});
      return make_tuple(out, mean, rstd);
        */
}
