crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Activation.h]

pub type StructuredActivationFn = fn(_0: &mut TensorIteratorBase) -> c_void;

pub type StructuredActivationBackwardFn = fn(_0: &mut TensorIteratorBase) -> c_void;

pub type ActivationFn = fn(_0: &mut TensorIterator) -> c_void;

pub type ActivationBackwardFn = fn(_0: &mut TensorIterator) -> c_void;

pub type SoftplusFn = fn(
    _0: &mut TensorIteratorBase,
    _1: &Scalar,
    _2: &Scalar
) -> c_void;

pub type SoftplusBackwardFn = fn(
    _0: &mut TensorIteratorBase,
    _1: &Scalar,
    _2: &Scalar
) -> c_void;

pub type ThresholdFn = fn(
    _0: &mut TensorIteratorBase,
    _1: &Scalar,
    _2: &Scalar
) -> c_void;

pub type HardtanhBackwardFn = fn(
    _0: &mut TensorIterator,
    _1: &Scalar,
    _2: &Scalar
) -> c_void;

pub type HardsigmoidFn = fn(_0: &mut TensorIteratorBase) -> ();

pub type HardsigmoidBackwardFn = fn(_0: &mut TensorIteratorBase) -> ();

pub type HardswishFn = fn(_0: &mut TensorIterator) -> ();

pub type HardswishBackwardFn = fn(_0: &mut TensorIterator) -> ();

pub type ShrinkFn = fn(_0: &mut TensorIteratorBase, _1: &Scalar) -> c_void;

pub type SoftshrinkFn = fn(_0: &mut TensorIteratorBase, _1: &Scalar) -> c_void;

pub type ShrinkBackwardFn = fn(_0: &mut TensorIteratorBase, _1: &Scalar) -> c_void;

pub type EluFn = fn(
    _0: &mut TensorIteratorBase,
    _1: &Scalar,
    _2: &Scalar,
    _3: &Scalar
) -> c_void;

pub type EluBackwardFn = fn(
    _0: &mut TensorIteratorBase,
    _1: &Scalar,
    _2: &Scalar,
    _3: &Scalar,
    _4: bool
) -> c_void;

pub type LeakyReluFn = fn(_0: &mut TensorIteratorBase, _1: &Scalar) -> c_void;

pub type LeakyReluBackwardFn = fn(_0: &mut TensorIteratorBase, _1: &Scalar) -> c_void;

pub type LogSigmoidCpuFn = fn(
    _0: &mut Tensor,
    _1: &mut Tensor,
    _2: &Tensor
) -> c_void;

declare_dispatch!{elu_fn, elu_stub}
declare_dispatch!{elu_backward_fn, elu_backward_stub}
declare_dispatch!{softplus_fn, softplus_stub}
declare_dispatch!{softplus_backward_fn, softplus_backward_stub}
declare_dispatch!{log_sigmoid_cpu_fn, log_sigmoid_cpu_stub}
declare_dispatch!{activation_backward_fn, log_sigmoid_backward_cpu_stub}
declare_dispatch!{threshold_fn, threshold_stub}
declare_dispatch!{structured_activation_fn, GeluKernel}
declare_dispatch!{structured_activation_backward_fn, GeluBackwardKernel}
declare_dispatch!{hardtanh_backward_fn, hardtanh_backward_stub}
declare_dispatch!{hardsigmoid_fn, hardsigmoid_stub}
declare_dispatch!{hardsigmoid_backward_fn, hardsigmoid_backward_stub}
declare_dispatch!{hardswish_fn, hardswish_stub}
declare_dispatch!{hardswish_backward_fn, hardswish_backward_stub}
declare_dispatch!{shrink_fn, hardshrink_stub}
declare_dispatch!{softshrink_fn, softshrink_stub}
declare_dispatch!{shrink_backward_fn, shrink_backward_stub}
declare_dispatch!{leaky_relu_fn, leaky_relu_stub}
declare_dispatch!{leaky_relu_backward_fn, leaky_relu_backward_stub}
declare_dispatch!{activation_fn, glu_stub}
declare_dispatch!{activation_backward_fn, glu_backward_stub}
declare_dispatch!{structured_activation_fn, silu_stub}
declare_dispatch!{activation_backward_fn, silu_backward_stub}
declare_dispatch!{structured_activation_fn, mish_stub}
declare_dispatch!{activation_backward_fn, mish_backward_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Activation.cpp]

/**
  | computes `result = self <= threshold ? value : other`
  |
  | other is `self` in threshold() and `grad` in
  | threshold_backward()
  |
  */
lazy_static!{
    /*
    TORCH_META_FUNC(threshold)(const Tensor& self, const Scalar& threshold, const Scalar& value) {
      const Tensor& result = maybe_get_output();
      build(TensorIteratorConfig()
        .set_check_mem_overlap(false)  // threshold is idempotent, so overlap is okay
        .add_output(result)
        .add_input(self)
        .add_input(self) // other
        .allow_cpu_scalars(true)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .enforce_safe_casting_to_output(true));
    }
    */
}

/**
  | computes `result = self <= threshold ? value : other`
  |
  | other is `self` in threshold() and `grad` in
  | threshold_backward()
  |
  */
lazy_static!{
    /*
    TORCH_META_FUNC(threshold_backward)(const Tensor& grad, const Tensor& self, const Scalar& threshold) {
      const Tensor& gradInput = maybe_get_output();
      build(TensorIteratorConfig()
        .set_check_mem_overlap(false)  // threshold is idempotent, so overlap is okay
        .add_output(gradInput)
        .add_input(self)
        .add_input(grad)  // other
        .allow_cpu_scalars(true)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .enforce_safe_casting_to_output(true));
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(elu) (
      const Tensor& self, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale
    ) {
      build_unary_op(maybe_get_output(), self);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(elu_backward) (
      const Tensor& grad_output,
      const Scalar& alpha,
      const Scalar& scale,
      const Scalar& input_scale,
      bool is_result,
      const Tensor& self_or_result
    ) {
      TORCH_CHECK(
        !is_result || alpha.to<double>() >= 0.0,
        "In-place elu backward calculation is triggered with a negative slope which is not supported. "
        "This is caused by calling in-place forward function with a negative slope, "
        "please call out-of-place version instead.");

      build_borrowing_binary_op(maybe_get_output(), grad_output, self_or_result);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(silu) (const Tensor& self) {
      build_unary_op(maybe_get_output(), self);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(mish) (const Tensor& self) {
      build_unary_op(maybe_get_output(), self);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(softplus) (
      const Tensor& self, const Scalar& beta, const Scalar& threshold
    ) {
      build_unary_op(maybe_get_output(), self);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(softplus_backward) (
      const Tensor& grad_output,
      const Tensor& self,
      const Scalar& beta,
      const Scalar& threshold,
      const Tensor& output
    ) {
      build_borrowing_binary_op(maybe_get_output(), grad_output, self);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(leaky_relu) (
      const Tensor& self, const Scalar& negval
    ) {
      build_unary_op(maybe_get_output(), self);
    }
    */
}

/**
  | Note: leakyReLu backward calculation doesn't
  | support in-place call with negative slope.
  |
  | The reason is that for in-place forward call,
  | the forward result will be saved into autograd
  | node instead of the input itself, when
  | calculating backward gradient, there is no way
  | to know whether the original input for current
  | node is positive or not if the input slope is
  | negative.
  |
  | eg. forward is 2, slope is -0.2, the original
  | input for this node could be either 2, or -10,
  | so no way to get a correct backward gradient in
  | this case.
  |
  */
lazy_static!{
    /*
    TORCH_META_FUNC(leaky_relu_backward) (
      const Tensor& grad_output,
      const Tensor& self_or_result,
      const Scalar& negval,
      bool is_result
    ) {
      TORCH_CHECK(
        !is_result || negval.to<double>() >= 0.0,
        "In-place leakyReLu backward calculation is triggered with a negative slope which is not supported. "
        "This is caused by calling in-place forward function with a negative slope, "
        "please call out-of-place version instead. File an issue at https://github.com/pytorch/pytorch if you do "
        "require supporting in-place leakRelu backward calculation with negative slope");

      build_borrowing_binary_op(maybe_get_output(), self_or_result, grad_output);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(hardsigmoid) (const Tensor& self) {
      build_unary_op(maybe_get_output(), self);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(hardsigmoid_backward) (const Tensor& grad_output, const Tensor& self) {
      build_borrowing_binary_op(maybe_get_output(), grad_output, self);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(hardshrink) (const Tensor & self, const Scalar& lambd) {
      build_unary_op(maybe_get_output(), self);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(hardshrink_backward) (
      const Tensor & grad, const Tensor & self, const Scalar& lambd
    ) {
      build_borrowing_binary_op(maybe_get_output(), grad, self);
    }
    */
}

#[inline] pub fn softshrink_check(lambd: &Scalar)  {
    
    todo!();
        /*
            double lamb = lambd.to<double>();
      TORCH_CHECK(lamb >= 0, "lambda must be greater or equal to 0, but found to be ", lamb, ".");
        */
}

lazy_static!{
    /*
    TORCH_META_FUNC(softshrink) (
      const Tensor & self, const Scalar& lambd
    ) {
      softshrink_check(lambd);
      build_unary_op(maybe_get_output(), self);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(softshrink_backward) (
      const Tensor & grad, const Tensor & self, const Scalar& lambd
    ) {
      build_borrowing_binary_op(maybe_get_output(), grad, self);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(gelu) (const Tensor & self) {
      build_unary_op(maybe_get_output(), self);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(gelu_backward) (
      const Tensor& grad, const Tensor& self
    ) {
      build_borrowing_binary_op(maybe_get_output(), grad, self);
    }
    */
}

pub const SELU_ALPHA: f64 = 1.6732632423543772848170429916717;
pub const SELU_SCALE: f64 = 1.0507009873554804934193349852946;

define_dispatch!(elu_stub);
define_dispatch!(elu_backward_stub);
define_dispatch!(softplus_stub);
define_dispatch!(softplus_backward_stub);
define_dispatch!(log_sigmoid_cpu_stub);
define_dispatch!(log_sigmoid_backward_cpu_stub);
define_dispatch!(threshold_stub);
define_dispatch!(hardtanh_backward_stub);
define_dispatch!(hardsigmoid_stub);
define_dispatch!(hardsigmoid_backward_stub);
define_dispatch!(hardswish_stub);
define_dispatch!(hardswish_backward_stub);
define_dispatch!(hardshrink_stub);
define_dispatch!(softshrink_stub);
define_dispatch!(shrink_backward_stub);
define_dispatch!(leaky_relu_stub);
define_dispatch!(leaky_relu_backward_stub);
define_dispatch!(silu_stub);
define_dispatch!(silu_backward_stub);
define_dispatch!(mish_stub);
define_dispatch!(mish_backward_stub);

lazy_static!{
    /*
    TORCH_IMPL_FUNC(elu_out) (
      const Tensor& self, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale, const Tensor& result
    ) {
      elu_stub(device_type(), *this, alpha, scale, input_scale);
    }
    */
}


lazy_static!{
    /*
    TORCH_IMPL_FUNC(elu_backward_out) (
      const Tensor& grad_output,
      const Scalar& alpha,
      const Scalar& scale,
      const Scalar& input_scale,
      bool is_result,
      const Tensor& self_or_result,
      const Tensor& grad_input
    ) {
      elu_backward_stub(device_type(), *this, alpha, scale, input_scale, is_result);
    }
    */
}


lazy_static!{
    /*
    TORCH_IMPL_FUNC(silu_out) (
      const Tensor& self, const Tensor& result
    ) {
      silu_stub(device_type(), *this);
    }
    */
}


lazy_static!{
    /*
    TORCH_IMPL_FUNC(mish_out) (
      const Tensor& self, const Tensor& result
    ) {
      mish_stub(device_type(), *this);
    }
    */
}


lazy_static!{
    /*
    TORCH_IMPL_FUNC(softplus_out) (
      const Tensor& self, const Scalar& beta, const Scalar& threshold, const Tensor& result
    ) {
      softplus_stub(device_type(), *this, beta, threshold);
    }
    */
}


lazy_static!{
    /*
    TORCH_IMPL_FUNC(softplus_backward_out) (
      const Tensor& grad_output,
      const Tensor& self,
      const Scalar& beta,
      const Scalar& threshold,
      const Tensor& output,
      const Tensor& grad_input
    ) {
      softplus_backward_stub(device_type(), *this, beta, threshold);
    }
    */
}


lazy_static!{
    /*
    TORCH_IMPL_FUNC(leaky_relu_out) (
      const Tensor& self, const Scalar& negval, const Tensor& result
    ) {
      leaky_relu_stub(device_type(), *this, negval);
    }
    */
}


lazy_static!{
    /*
    TORCH_IMPL_FUNC(leaky_relu_backward_out) (
      const Tensor& grad_output,
      const Tensor& self_or_result,
      const Scalar& negval,
      bool is_result,
      const Tensor& grad_input
    ) {
      leaky_relu_backward_stub(device_type(), *this, negval);
    }
    */
}


lazy_static!{
    /*
    TORCH_IMPL_FUNC(hardsigmoid_out) (
      const Tensor& self, const Tensor& result
    ) {
      hardsigmoid_stub(device_type(), *this);
    }
    */
}


lazy_static!{
    /*
    TORCH_IMPL_FUNC(hardsigmoid_backward_out) (
      const Tensor& grad_output, const Tensor& self, const Tensor& grad_input
    ) {
      hardsigmoid_backward_stub(device_type(), *this);
    }
    */
}


lazy_static!{
    /*
    TORCH_IMPL_FUNC(hardshrink_out) (
      const Tensor & self, const Scalar& lambd, const Tensor& result
    ) {
      hardshrink_stub(device_type(), *this, lambd);
    }
    */
}


lazy_static!{
    /*
    TORCH_IMPL_FUNC(hardshrink_backward_out) (
      const Tensor & grad, const Tensor & self, const Scalar& lambd, const Tensor& grad_input
    ) {
      shrink_backward_stub(device_type(), *this, lambd);
    }
    */
}


lazy_static!{
    /*
    TORCH_IMPL_FUNC(softshrink_out) (
      const Tensor & self, const Scalar& lambd, const Tensor& result
    ) {
      softshrink_stub(device_type(), *this, lambd);
    }
    */
}


lazy_static!{
    /*
    TORCH_IMPL_FUNC(softshrink_backward_out) (
      const Tensor & grad, const Tensor & self, const Scalar& lambd, const Tensor& grad_input
    ) {
      shrink_backward_stub(device_type(), *this, lambd);
    }
    */
}


lazy_static!{
    /*
    TORCH_IMPL_FUNC(gelu_out_cpu) (
      const Tensor& self, const Tensor& result
    ) {
      GeluKernel(kCPU, *this);
    }
    */
}


lazy_static!{
    /*
    TORCH_IMPL_FUNC(gelu_backward_out_cpu) (
      const Tensor& grad, const Tensor& self, const Tensor& grad_input
    ) {
      GeluBackwardKernel(kCPU, *this);
    }
    */
}


pub fn hardtanh(
        self_: &Tensor,
        min:   &Scalar,
        max:   &Scalar) -> Tensor {
    
    todo!();
        /*
            return at::clamp(self, min, max);
        */
}


pub fn hardtanh_out<'a>(
        self_:  &Tensor,
        min:    &Scalar,
        max:    &Scalar,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return at::clamp_out(result, self, min, max);
        */
}


pub fn hardtanh_mut<'a>(
        self_: &mut Tensor,
        min:   &Scalar,
        max:   &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            return at::clamp_(self, min, max);
        */
}

pub fn hardtanh_backward_out<'a>(
    grad_output: &Tensor,
    self_:       &Tensor,
    min:         &Scalar,
    max:         &Scalar,
    grad_input:  &mut Tensor) -> &'a mut Tensor {

    todo!();
        /*
            auto iter = TensorIterator::borrowing_binary_op(grad_input, grad_output, self);
      hardtanh_backward_stub(iter.device_type(), iter, min, max);
      return grad_input;
        */
}

pub fn hardtanh_backward(
    grad_output: &Tensor,
    self_:       &Tensor,
    min:         &Scalar,
    max:         &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor result;
      auto iter = TensorIterator::borrowing_binary_op(result, grad_output, self);
      hardtanh_backward_stub(iter.device_type(), iter, min, max);
      return iter.output();
        */
}

pub fn hardswish(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            #if defined(C10_MOBILE) && defined(USE_XNNPACK)
      if (xnnpack::use_hardswish(self)) {
        return xnnpack::hardswish(self);
      }
      #endif
      Tensor result;
      auto iter = TensorIterator::unary_op(result, self);
      hardswish_stub(iter.device_type(), iter);
      return iter.output();
        */
}


pub fn hardswish_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            auto iter = TensorIterator::unary_op(result, self);
      hardswish_stub(iter.device_type(), iter);
      return result;
        */
}


pub fn hardswish_mut(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            #if defined(C10_MOBILE) && defined(USE_XNNPACK)
      if (xnnpack::use_hardswish(self)) {
        xnnpack::hardswish_(self);
        return self;
      }
      #endif
      auto iter = TensorIterator::unary_op(self, self);
      hardswish_stub(iter.device_type(), iter);
      return self;
        */
}


pub fn hardswish_backward(
        grad_output: &Tensor,
        self_:       &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor grad_input;
      auto iter = TensorIterator::borrowing_binary_op(grad_input, grad_output, self);
      hardswish_backward_stub(iter.device_type(), iter);
      return iter.output();
        */
}


pub fn relu<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return at::clamp_min(self, 0);
        */
}


pub fn relu_mut(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return at::clamp_min_(self, 0);
        */
}


pub fn selu(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return at::elu(self, SELU_ALPHA, SELU_SCALE);
        */
}


pub fn relu6<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return at::hardtanh(self, /*min_val=*/0, /*max_val=*/6);
        */
}


pub fn selu_mut<'a>(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return at::elu_(self, SELU_ALPHA, SELU_SCALE);
        */
}


pub fn relu6_mut(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return at::hardtanh_(self, /*min_val=*/0, /*max_val=*/6);
        */
}


pub fn celu(
        self_: &Tensor,
        alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(alpha.to<double>() != 0,
          "ZeroDivisionError: alpha cannot be 0 for CELU");
      double inv_alpha = 1. / alpha.to<double>();
      return at::elu(self, alpha, Scalar(1.0), Scalar(inv_alpha));
        */
}

pub fn celu_mut<'a>(
    self_: &mut Tensor,
    alpha: &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(alpha.to<double>() != 0,
          "ZeroDivisionError: alpha cannot be 0 for CELU");
      double inv_alpha = 1. / alpha.to<double>();
      return at::elu_(self, alpha, Scalar(1.0), Scalar(inv_alpha));
        */
}


pub fn silu_backward(
        grad_output: &Tensor,
        input:       &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor grad_input = at::empty({0}, input.options());
      auto iter = TensorIterator::borrowing_binary_op(grad_input, grad_output, input);
      silu_backward_stub(iter.device_type(), iter);
      return grad_input;
        */
}


pub fn math_silu_backward(
        grad_output: &Tensor,
        input:       &Tensor) -> Tensor {
    
    todo!();
        /*
            auto input_sigmoid = at::sigmoid(input);
      return grad_output * (input_sigmoid * (1 + input * (1 - input_sigmoid)));
        */
}

pub fn mish_backward(
    grad_output: &Tensor,
    input:       &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor grad_input = at::empty({0}, input.options());
      auto iter = TensorIterator::binary_op(grad_input, grad_output, input);
      mish_backward_stub(iter.device_type(), iter);
      return grad_input;
        */
}

pub fn math_mish_backward(
    grad_output: &Tensor,
    input:       &Tensor) -> Tensor {

    todo!();
        /*
            auto input_tanh_softplus = at::tanh(at::softplus(input));
      auto input_sigmoid = at::sigmoid(input);
      return grad_output * (input_tanh_softplus + (input * input_sigmoid * (1 - input_tanh_softplus * input_tanh_softplus)));
        */
}

#[inline] pub fn rrelu_with_noise_train<Scalar>(
    output:    &mut Tensor,
    input:     &Tensor,
    noise:     &Tensor,
    lower:     &Scalar,
    upper:     &Scalar,
    generator: Option<Generator>)  {

    todo!();
    /*
            Scalar lower = lower_.to<Scalar>();
      Scalar upper = upper_.to<Scalar>();
      Tensor tmp_tensor = output.contiguous();
      Scalar* output_data = tmp_tensor.data_ptr<Scalar>();
      Scalar* input_data = input.data_ptr<Scalar>();
      Scalar* noise_data = noise.data_ptr<Scalar>();
      auto gen  = at::get_generator_or_default<CPUGeneratorImpl>(generator, detail::getDefaultCPUGenerator());
      std::lock_guard<std::mutex> lock(gen->mutex_);
      for (i64 i = 0; i < input.numel(); i++) {
        if (input_data[i] <= 0) {
          at::uniform_real_distribution<double> uniform(lower, upper);
          const Scalar r = (Scalar)uniform(gen);
          output_data[i] = input_data[i] * r;
          noise_data[i] = r;
        } else {
          noise_data[i] = 1;
          output_data[i] = input_data[i];
        }
      }
      if (!output.is_contiguous()) {
        output.copy_(tmp_tensor);
      }
        */
}


pub fn rrelu_with_noise_out_cpu<'a>(
        self_:     &Tensor,
        noise:     &Tensor,
        lower:     &Scalar,
        upper:     &Scalar,
        training:  bool,
        generator: Option<Generator>,
        output:    &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            if (training) {
        AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "rrelu_with_noise_out_cpu", [&] {
          _rrelu_with_noise_train<Scalar>(output, self.contiguous(), noise, lower, upper, generator);
        });
        return output;
      } else {
        auto lower_tensor = scalar_to_tensor(lower);
        auto upper_tensor = scalar_to_tensor(upper);
        auto negative = (lower_tensor + upper_tensor) / 2;
        Scalar negative_slope = negative.item();
        return at::leaky_relu_out(output, self, negative_slope);
      }
        */
}


pub fn rrelu_with_noise_cpu(
        self_:     &Tensor,
        noise:     &Tensor,
        lower:     &Scalar,
        upper:     &Scalar,
        training:  bool,
        generator: Option<Generator>) -> Tensor {
    
    todo!();
        /*
            auto output = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      return at::native::rrelu_with_noise_out_cpu(
          self, noise, lower, upper, training, generator, output);
        */
}


pub fn rrelu_with_noise_cpu_mut<'a>(
        self_:     &mut Tensor,
        noise:     &Tensor,
        lower:     &Scalar,
        upper:     &Scalar,
        training:  bool,
        generator: Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            return at::native::rrelu_with_noise_out_cpu(
          self, noise, lower, upper, training, generator, self);
        */
}


pub fn rrelu_with_noise_backward(
        grad_output:    &Tensor,
        self_or_result: &Tensor,
        noise:          &Tensor,
        lower:          &Scalar,
        upper:          &Scalar,
        training:       bool,
        is_result:      bool) -> Tensor {
    
    todo!();
        /*
            auto lower_tensor = scalar_to_tensor(lower);
      auto upper_tensor = scalar_to_tensor(upper);
      if (training && (upper_tensor - lower_tensor).item().to<float>() > 1E-6) {
        return grad_output.mul(noise);
      } else {
        auto negative = (lower_tensor + upper_tensor) / 2;
        Scalar negative_slope = negative.item();
        return at::leaky_relu_backward(grad_output, self_or_result, negative_slope, is_result);
      }
        */
}


pub fn rrelu(
        self_:     &Tensor,
        lower:     &Scalar,
        upper:     &Scalar,
        training:  bool,
        generator: Option<Generator>) -> Tensor {
    
    todo!();
        /*
            return at::rrelu_with_noise(self, at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT), lower, upper, training, generator);
        */
}

pub fn rrelu_mut<'a>(
        self_:     &mut Tensor,
        lower:     &Scalar,
        upper:     &Scalar,
        training:  bool,
        generator: Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            return at::rrelu_with_noise_(self, at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT), lower, upper, training, generator);
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(threshold_out)(const Tensor& self, const Scalar& threshold, const Scalar& value, const Tensor& result) {
      threshold_stub(device_type(), *this, threshold, value);
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(threshold_backward_out)(const Tensor& grad, const Tensor& self, const Scalar& threshold, const Tensor& gradInput) {
      threshold_stub(device_type(), *this, threshold, 0);
    }
    */
}

// -----------------------------------
// prelu forward
// -----------------------------------
#[inline] pub fn prelu_cpu_kernel_share_weights<Scalar>(
        result: &mut Tensor,
        input:  &Tensor,
        weight: &Tensor)  {

    todo!();
        /*
            i64 input_numel = input.numel();
      auto result_data = result.data_ptr<Scalar>();
      auto input_data = input.data_ptr<Scalar>();
      auto weight_val = weight.data_ptr<Scalar>()[0];

      at::parallel_for(0, input_numel, 1000, [&](i64 start, i64 end) {
        for (auto i = start; i < end; i++) {
          Scalar input_data_val = input_data[i];
          // to allow for compiler optimization, here splitting into two lines:
          Scalar r = (input_data_val > 0) ? Scalar(1) : weight_val;
          result_data[i] = r * input_data_val;
        }
      });
        */
}

#[inline] pub fn prelu_cpu_kernel_multi_weights<Scalar>(
    result:          &mut Tensor,
    input:           &Tensor,
    weight:          &Tensor,
    input_dim0_size: i64,
    channel_size:    i64,
    input_stride0:   i64,
    input_stride1:   i64)  {

    todo!();
        /*
            Scalar* result_data = result.data_ptr<Scalar>();
      Scalar* input_data = input.data_ptr<Scalar>();
      Scalar* weight_data = weight.data_ptr<Scalar>();

      auto loop = [&](i64 start, i64 end) {
        for (const auto i : c10::irange(start, end)) {
          i64 offset = i * channel_size * input_stride1;
          Scalar* n_input_data = input_data + offset;
          Scalar* n_result_data = result_data + offset;
          for (const auto j : c10::irange(channel_size)) {
            for (const auto k : c10::irange(input_stride1)) {
              // to allow for compiler optimization, here splitting into two lines:
              Scalar w = (n_input_data[k] > 0) ? Scalar(1) : weight_data[j];
              n_result_data[k] = w * n_input_data[k];
            }
            n_input_data += input_stride1;
            n_result_data += input_stride1;
          }
        }
      };
      if (input.numel() > 1000) {
        at::parallel_for(0, input_dim0_size, 0, loop);
      } else {
        loop(0, input_dim0_size);
      }
        */
}

pub fn prelu_cpu(
    self_:  &Tensor,
    weight: &Tensor) -> Tensor {
    
    todo!();
        /*
            auto input = self.contiguous();
      auto weight = weight_.contiguous();

      TORCH_CHECK(input.is_contiguous());
      TORCH_CHECK(weight.is_contiguous());

      i64 weight_num = weight.numel();
      Tensor result = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto strides = input.strides();

      // case1: shared weight for all channels
      if (weight_num == 1) {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "prelu_cpu", [&] {
          prelu_cpu_kernel_share_weights<Scalar>(result, input, weight);
        });
      }
      else { // case2: multiple weights, one for each channel
        i64 input_ndim = input.dim();
        TORCH_CHECK(input_ndim > 0, "Not allow zero-dim input tensor.");

        i64 channel_size = 1; // channel_size default to 1
        i64 input_dim0_size = 1, input_stride0 = 1, input_stride1 = 1;

        if (input_ndim > 1) {
          channel_size = input.size(1); // channel is the 2nd dim of input
          input_dim0_size = input.size(0);
          input_stride0 = strides[0];
          input_stride1 = strides[1];
        }
        TORCH_CHECK(channel_size == weight_num,
          "Mismatch of parameter numbers and input channel size. Found parameter numbers = ", weight_num,
          " and channel size = ", channel_size, ".");

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "prelu_cpu", [&] {
          prelu_cpu_kernel_multi_weights<Scalar>(
            result,
            input,
            weight,
            input_dim0_size,
            channel_size,
            input_stride0,
            input_stride1);
        });
      }
      return result;
        */
}

// -----------------------------------
// prelu backward
// -----------------------------------
#[inline] pub fn prelu_cpu_backward_kernel_share_weights<Scalar>(
    input:       &Tensor,
    weight:      &Tensor,
    grad_out:    &Tensor,
    input_grad:  &mut Tensor,
    weight_grad: &mut Tensor)  {

    todo!();
        /*
            i64 input_numel = input.numel();
      auto input_data = input.data_ptr<Scalar>();
      auto weight_val = weight.data_ptr<Scalar>()[0];
      auto grad_out_data = grad_out.data_ptr<Scalar>();
      auto input_grad_data = input_grad.data_ptr<Scalar>();
      auto weight_grad_data = weight_grad.data_ptr<Scalar>();

      Scalar sum = at::parallel_reduce(0, input_numel, 1000, Scalar(0),
          [&](i64 start, i64 end, Scalar ident) -> Scalar {
        Scalar partial_sum = ident;
        for (auto i = start; i < end; i++) {
          Scalar input_data_val = input_data[i];
          Scalar grad_out_data_val = grad_out_data[i];
          // to allow for compiler optimization, here splitting into two lines:
          Scalar w = (input_data_val > 0) ? Scalar(1) : weight_val;
          input_grad_data[i] = w * grad_out_data_val;
          // to allow for compiler optimization, here splitting into two lines:
          Scalar mask = (input_data_val > 0) ? Scalar(0) : Scalar(1);
          partial_sum += mask * input_data_val * grad_out_data_val;
        }
        return partial_sum;
      }, std::plus<Scalar>());
      weight_grad_data[0] = sum;
        */
}


#[inline] pub fn prelu_cpu_backward_kernel_multi_weights<Scalar>(
    input:                 &Tensor,
    weight:                &Tensor,
    grad_out:              &Tensor,
    input_grad:            &mut Tensor,
    weight_grad_collector: &mut Tensor,
    input_dim0_size:       i64,
    channel_size:          i64,
    input_stride0:         i64,
    input_stride1:         i64)  {

    todo!();
    /*
       auto input_data = input.data_ptr<Scalar>();
       auto weight_data = weight.data_ptr<Scalar>();
      auto grad_out_data = grad_out.data_ptr<Scalar>();
      auto input_grad_data = input_grad.data_ptr<Scalar>();
      auto weight_grad_collector_data = weight_grad_collector.data_ptr<Scalar>();

      auto loop = [&](i64 start, i64 end) {
        for (const auto i : c10::irange(start, end)) {
          for (const auto j : c10::irange(channel_size)) {
            for (const auto k : c10::irange(input_stride1)) {
              i64 pos = i * input_stride0 + j * input_stride1 + k;
              Scalar weight_data_val = weight_data[j];
              Scalar input_data_val = input_data[pos];
              Scalar grad_out_data_val = grad_out_data[pos];
              // to allow for compiler optimization, here splitting into two lines:
              Scalar w = (input_data_val > 0) ? Scalar(1) : weight_data_val;
              input_grad_data[pos] = w * grad_out_data_val;
              // to allow for compiler optimization, here splitting into two lines:
              Scalar mask = (input_data_val > 0) ? Scalar(0) : Scalar(1);
              weight_grad_collector_data[pos] = mask * input_data_val * grad_out_data_val;
            }
          }
        }
      };
      if (input.numel() > 1000) {
        at::parallel_for(0, input_dim0_size, 0, loop);
      } else {
        loop(0, input_dim0_size);
      }
        */
}

pub fn prelu_backward_cpu(
    grad_out: &Tensor,
    self_:    &Tensor,
    weight:   &Tensor) -> (Tensor,Tensor) {
    
    todo!();
        /*
            auto input = self.contiguous();
      auto grad_out = grad_out_.contiguous();
      auto weight = weight_.contiguous();

      TORCH_CHECK(input.is_contiguous());
      TORCH_CHECK(grad_out.is_contiguous());
      TORCH_CHECK(weight.is_contiguous());

      i64 weight_num = weight.numel();
      auto strides = input.strides();
      auto dims = input.dim();

      Tensor input_grad = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      Tensor weight_grad = at::empty_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      Tensor weight_grad_collector = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      // case1: shared parameter for all channels
      if (weight_num == 1) {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "prelu_backward_cpu", [&] {
          prelu_cpu_backward_kernel_share_weights<Scalar>(input, weight, grad_out, input_grad, weight_grad);
        });
      }
      else { // case2: multiple parameters, one for each channel
        i64 input_ndim = input.dim();
        TORCH_CHECK(input_ndim > 0, "Not allow zero-dim input tensor.");

        i64 channel_size = 1; // channel_size default to 1
        i64 input_dim0_size = 1, input_stride0 = 1, input_stride1 = 1;

        if (input_ndim > 1) {
          channel_size = input.size(1); // channel is the 2nd dim of input
          input_dim0_size = input.size(0);
          input_stride0 = strides[0];
          input_stride1 = strides[1];
        }
        TORCH_CHECK(channel_size == weight_num,
          "Mismatch of parameter numbers and input channel size. Found parameter numbers = ", weight_num,
          " and channel size = ", channel_size, ".");

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "prelu_backward_cpu", [&] {
          prelu_cpu_backward_kernel_multi_weights<Scalar>(
            input,
            weight,
            grad_out,
            input_grad,
            weight_grad_collector,
            input_dim0_size,
            channel_size,
            input_stride0,
            input_stride1);
        });
        // update weight_grad
        std::vector<i64> reduce_dims;
        reduce_dims.push_back(0);
        if (dims > 2) {
          for(i64 i = 2; i < dims; i++) reduce_dims.push_back(i);
        }
        weight_grad = weight_grad_collector.sum(reduce_dims);
      }
      return std::tuple<Tensor, Tensor>{input_grad, weight_grad};
        */
}

pub fn infinitely_differentiable_gelu_backward(
    grad:  &Tensor,
    self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            constexpr double kAlpha = M_2_SQRTPI * M_SQRT1_2 * 0.5;
      Tensor cdf = (1.0 + (self * M_SQRT1_2).erf_()).mul_(0.5);
      Tensor pdf = (-0.5 * self * self).exp_();
      return cdf.addcmul_(self, pdf, kAlpha).mul_(grad);
        */
}

pub fn log_sigmoid_forward_cpu(input: &Tensor) -> (Tensor,Tensor) {
    
    todo!();
        /*
            // FIXME: do these actually need to be zeros_like or can they be empty_like?
      auto result = at::zeros_like(input, at::MemoryFormat::Contiguous);
      auto buffer = at::zeros_like(input, at::MemoryFormat::Contiguous);
      log_sigmoid_cpu_stub(kCPU, result, buffer, input.contiguous());
      return std::make_tuple(result, buffer);
        */
}

pub fn log_sigmoid_forward_out_cpu<'a>(
        input:  &Tensor,
        result: &mut Tensor,
        buffer: &mut Tensor) -> (&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
            result.resize_as_(input);
      buffer.resize_as_(input, at::MemoryFormat::Contiguous);
      TORCH_CHECK(buffer.is_contiguous(), "Contiguous buffer required for log_sigmoid with out parameter");
      Tensor result_tmp = result.is_contiguous() ? result : at::empty_like(result, at::MemoryFormat::Contiguous);
      log_sigmoid_cpu_stub(kCPU, result_tmp, buffer, input.contiguous());
      if (!result.is_contiguous()) {
        result.copy_(result_tmp);
      }
      return std::forward_as_tuple(result, buffer);
        */
}


pub fn log_sigmoid_out<'a>(
    self_:  &Tensor,
    output: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            Tensor buffer = at::empty({0}, self.options());
      return std::get<0>(at::log_sigmoid_forward_out(output, buffer, self));
        */
}

pub fn log_sigmoid(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return std::get<0>(at::log_sigmoid_forward(self));
        */
}

pub fn log_sigmoid_backward_cpu(
        grad_output: &Tensor,
        input:       &Tensor,
        buffer:      &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor grad_input;
      auto iter = at::TensorIteratorConfig()
        .add_output(grad_input)
        .add_input(input)
        .add_input(buffer)
        .add_input(grad_output)
        .build();
      log_sigmoid_backward_cpu_stub(kCPU, iter);
      return iter.output();
        */
}

pub fn log_sigmoid_backward_out_cpu<'a>(
    grad_output: &Tensor,
    input:       &Tensor,
    buffer:      &Tensor,
    grad_input:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            auto iter = TensorIteratorConfig()
        .add_output(grad_input)
        .add_input(input)
        .add_input(buffer)
        .add_input(grad_output)
        .build();
      log_sigmoid_backward_cpu_stub(kCPU, iter);
      return grad_input;
        */
}

define_dispatch!(GeluKernel);
define_dispatch!(GeluBackwardKernel);
