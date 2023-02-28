crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/layer_norm.h]

#[inline(always)]
pub fn check_layer_norm_inputs(
        input:            &Tensor,
        normalized_shape: &[i32],
        weight:           &Tensor,
        bias:             &Tensor) -> (i64,i64) {
    
    todo!();
        /*
            const int normalized_ndim = normalized_shape.size();
      TORCH_CHECK(
          normalized_ndim >= 1,
          "Expected normalized_shape to be at least 1-dimensional, i.e., ",
          "containing at least one element, but got normalized_shape = ",
          normalized_shape);
      TORCH_CHECK(
          !weight.defined() || weight.sizes().equals(normalized_shape),
          "Expected weight to be of same shape as normalized_shape, but got ",
          "weight of shape ",
          weight.sizes(),
          " and normalized_shape = ",
          normalized_shape);
      TORCH_CHECK(
          !bias.defined() || bias.sizes().equals(normalized_shape),
          "Expected bias to be of same shape as normalized_shape, but got ",
          "bias of shape ",
          bias.sizes(),
          " and normalized_shape = ",
          normalized_shape);

      const auto input_shape = input.sizes();
      const auto input_ndim = input.dim();

      if (input_ndim < normalized_ndim ||
          !input_shape.slice(input_ndim - normalized_ndim)
               .equals(normalized_shape)) {
        stringstream ss;
        ss << "Given normalized_shape=" << normalized_shape
           << ", expected input with shape [*";
        for (auto size : normalized_shape) {
          ss << ", " << size;
        }
        ss << "], but got input of size" << input_shape;
        AT_ERROR(ss.str());
      }

      const int axis = input_ndim - normalized_ndim;
      const i64 M =
          multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);
      const i64 N =
          multiply_integers(input_shape.cbegin() + axis, input_shape.cend());

      return make_pair(M, N);
        */
}

pub fn layer_norm_cpu_out_a(
    out:              &mut Tensor,
    mean:             &mut Tensor,
    rstd:             &mut Tensor,
    input:            &Tensor,
    normalized_shape: &[i32],
    gamma:            &Tensor,
    beta:             &Tensor,
    eps:              f64,
    M:                i64,
    N:                i64)  {
    
    todo!();
        /*
        
        */
}

lazy_static!{
    /*
    using forward_fn = void (*)(
        const Tensor& /* X */,
        const Tensor& /* gamma */,
        const Tensor& /* beta */,
        i64 /* M */,
        i64 /* N */,
        double /* eps */,
        Tensor* /* Y */,
        Tensor* /* mean */,
        Tensor* /* rstd */);

    using backward_fn = void (*)(
        const Tensor& /* dY */,
        const Tensor& /* X */,
        const Tensor& /* mean */,
        const Tensor& /* rstd */,
        const Tensor& /* gamma */,
        i64 /* M */,
        i64 /* N */,
        Tensor* /* dX */,
        Tensor* /* dgamma */,
        Tensor* /* dbeta */);
    */
}

declare_dispatch!{forward_fn, LayerNormKernel}
declare_dispatch!{backward_fn, LayerNormBackwardKernel}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/layer_norm.cpp]

pub fn layer_norm_cpu_out_b(
    out:              &mut Tensor,
    mean:             &mut Tensor,
    rstd:             &mut Tensor,
    input:            &Tensor,
    normalized_shape: &[i32],
    gamma:            &Tensor,
    beta:             &Tensor,
    eps:              f64,
    M:                i64,
    N:                i64)  {

    todo!();
        /*
            if (M <= 0) {
        return;
      }

      LayerNormKernel(kCPU, input, gamma, beta, M, N, eps, &out, &mean, &rstd);
      const auto input_shape = input.sizes();
      const usize axis = input.dim() - normalized_shape.size();

      DimVector stat_shape;
      for (usize idx = 0; idx < axis; ++idx) {
        stat_shape.emplace_back(input_shape[idx]);
      }
      for (usize idx = axis; idx < input.dim(); ++idx) {
        stat_shape.emplace_back(1);
      }

      mean = mean.view(stat_shape);
      rstd = rstd.view(stat_shape);
        */
}

pub fn layer_norm_cpu(
        input:            &Tensor,
        normalized_shape: &[i32],
        weight_opt:       &Option<Tensor>,
        bias_opt:         &Option<Tensor>,
        eps:              f64) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
      auto M = M_N.first;
      auto N = M_N.second;
      auto X = input.expect_contiguous();
      auto gamma = weight.expect_contiguous();
      auto beta = bias.expect_contiguous();

      Tensor Y = native::empty_like(
          *X,
          nullopt /* dtype */,
          nullopt /* layout */,
          nullopt /* device */,
          nullopt /* pin_memory */,
          MemoryFormat::Contiguous);
      Tensor mean = empty({M}, X->options());
      Tensor rstd = empty({M}, X->options());

      layer_norm_cpu_out(Y, mean, rstd, *X, normalized_shape, *gamma, *beta, eps, M, N);
      return make_tuple(move(Y), move(mean), move(rstd));
        */
}

pub fn layer_norm_backward_cpu(
        dy:               &Tensor,
        input:            &Tensor,
        normalized_shape: &[i32],
        mean:             &Tensor,
        rstd:             &Tensor,
        weight_opt:       &Option<Tensor>,
        bias_opt:         &Option<Tensor>,
        grad_input_mask:  [bool; 3]) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned =
          borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;
      MaybeOwned<Tensor> bias_maybe_owned =
          borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
      auto M = M_N.first;
      auto N = M_N.second;
      auto X = input.expect_contiguous();
      auto gamma = weight.expect_contiguous();
      auto beta = bias.expect_contiguous();

      Tensor dX;
      Tensor dgamma;
      Tensor dbeta;
      if (grad_input_mask[0]) {
        dX = native::empty_like(
            *X,
            nullopt /* dtype */,
            nullopt /* layout */,
            nullopt /* device */,
            nullopt /* pin_memory */,
            MemoryFormat::Contiguous);
      }
      if (grad_input_mask[1]) {
        dgamma = M > 0 ? native::empty_like(
                             *gamma,
                             nullopt /* dtype */,
                             nullopt /* layout */,
                             nullopt /* device */,
                             nullopt /* pin_memory */,
                             MemoryFormat::Contiguous)
                       : native::zeros_like(
                             *gamma,
                             nullopt /* dtype */,
                             nullopt /* layout */,
                             nullopt /* device */,
                             nullopt /* pin_memory */,
                             MemoryFormat::Contiguous);
      }
      if (grad_input_mask[2]) {
        dbeta = M > 0 ? native::empty_like(
                            *beta,
                            nullopt /* dtype */,
                            nullopt /* layout */,
                            nullopt /* device */,
                            nullopt /* pin_memory */,
                            MemoryFormat::Contiguous)
                      : native::zeros_like(
                            *beta,
                            nullopt /* dtype */,
                            nullopt /* layout */,
                            nullopt /* device */,
                            nullopt /* pin_memory */,
                            MemoryFormat::Contiguous);
      }
      if (M > 0) {
        LayerNormBackwardKernel(
            kCPU, dY, *X, mean, rstd, *gamma, M, N, &dX, &dgamma, &dbeta);
      }
      return make_tuple(move(dX), move(dgamma), move(dbeta));
        */
}

pub fn layer_norm(
        input:            &Tensor,
        normalized_shape: &[i32],
        weight_opt:       &Option<Tensor>,
        bias_opt:         &Option<Tensor>,
        eps:              f64,

        /* cudnn_enable, deprecated */
        _5:               bool) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      return get<0>(native_layer_norm(input, normalized_shape, weight, bias, eps));
        */
}

define_dispatch!{LayerNormKernel}
define_dispatch!{LayerNormBackwardKernel}

/**
  | Ported from pytorch/xla repo
  |
  */
pub fn math_native_layer_norm(
        input:            &Tensor,
        normalized_shape: &[i32],
        weight_opt:       &Option<Tensor>,
        bias_opt:         &Option<Tensor>,
        eps:              f64) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
      auto M = M_N.first;
      auto X = input.expect_contiguous();
      auto gamma = weight.expect_contiguous();

      auto input_shape = input.sizes();
      const auto input_ndim = input.dim();
      const int normalized_ndim = normalized_shape.size();
      const int axis = input_ndim - normalized_ndim;
      Tensor input_reshaped = input.view({1, M, -1});
      // Unlike Batch Normalization, which applies scalar scale and bias for each
      // entire channel/plane with the affine option, Layer Normalization applies
      // per-element scale and bias. E.g. For input {N, C, H, W}, weight for
      // batchnorm has shape {C} while weight for layernorm has shape {H, W} or {W}.
      auto outputs = native_batch_norm(
          input_reshaped, /*weight=*/{}, /*bias=*/{}, /*running_mean=*/{},
          /*running_var=*/{}, /*training=*/true, /*momentum=*/0, eps);
      Tensor out = get<0>(outputs);
      out = out.view(input_shape);
      if (weight.defined() && bias.defined()) {
        out = bias.addcmul(out, weight, 1);
      } else if (weight.defined()) {
        out = out.mul(weight);
      } else if (bias.defined()) {
        out = out.add(bias);
      }
      Tensor mean = get<1>(outputs);
      Tensor rstd = get<2>(outputs);
      vector<i64> stat_shape;
      for (usize idx = 0; idx < axis; ++idx) {
        stat_shape.push_back(input_shape[idx]);
      }
      for (usize idx = axis; idx < input.dim(); ++idx) {
        stat_shape.push_back(1);
      }
      mean = mean.view(stat_shape);
      rstd = rstd.view(stat_shape);
      return make_tuple(out, mean, rstd);
        */
}
