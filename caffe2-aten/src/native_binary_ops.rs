crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/BinaryOps.h]

#[inline] pub fn alpha_check(
        dtype: ScalarType,
        alpha: &Scalar)  {
    
    todo!();
        /*
            TORCH_CHECK(! alpha.isBoolean() || dtype == ScalarType::Bool,
                  "Boolean alpha only supported for Boolean results.");
      TORCH_CHECK(isFloatingType(dtype) || isComplexType(dtype)
                  || alpha.isIntegral(true),
                  "For integral input tensors, argument alpha must not be a floating point number.");
      TORCH_CHECK(isComplexType(dtype) || !alpha.isComplex(),
                  "For non-complex input tensors, argument alpha must not be a complex number.")
        */
}

/**
  | Basic checking for all sub functions.
  |
  */
#[inline] pub fn sub_check(
        self_: &Tensor,
        other: &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(self.scalar_type() != kBool || other.scalar_type() != kBool,
                  "Subtraction, the `-` operator, with two bool tensors is not supported. "
                  "Use the `^` or `logical_xor()` operator instead.")
      TORCH_CHECK(self.scalar_type() != kBool && other.scalar_type() != kBool,
                  "Subtraction, the `-` operator, with a bool tensor is not supported. "
                  "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
        */
}

#[inline] pub fn sub_check_with_scalar(
        self_:  &Tensor,
        scalar: &Scalar)  {
    
    todo!();
        /*
            TORCH_CHECK(self.scalar_type() != kBool || !scalar.isBoolean(),
                  "Subtraction, the `-` operator, with two bool tensors is not supported."
                  "Use the `^` or `logical_xor()` operator instead.")
      TORCH_CHECK(self.scalar_type() != kBool && !scalar.isBoolean(),
                  "Subtraction, the `-` operator, with a bool tensor is not supported. "
                  "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
        */
}

pub type StructuredBinaryFnAlpha = fn(_0: &mut TensorIteratorBase, alpha: &Scalar) -> ();
pub type StructuredBinaryFn      = fn(_0: &mut TensorIteratorBase) -> ();
pub type BinaryFnAlpha           = fn(_0: &mut TensorIterator, alpha: &Scalar) -> ();
pub type BinaryFnDouble          = fn(_0: &mut TensorIterator, _1: f64) -> ();
pub type BinaryFn                = fn(_0: &mut TensorIterator) -> ();

pub type BinaryClampFnAlpha = fn(
        _0:      &mut TensorIterator,
        alpha:   &Scalar,
        min_val: &Scalar,
        max_val: &Scalar
) -> ();

declare_dispatch!{structured_binary_fn_alpha, add_stub}
declare_dispatch!{binary_clamp_fn_alpha, add_clamp_stub}
declare_dispatch!{structured_binary_fn_alpha, sub_stub}
declare_dispatch!{structured_binary_fn, mul_stub}
declare_dispatch!{structured_binary_fn, div_true_stub}
declare_dispatch!{structured_binary_fn, div_floor_stub}
declare_dispatch!{structured_binary_fn, div_trunc_stub}
declare_dispatch!{structured_binary_fn, atan2_stub}
declare_dispatch!{structured_binary_fn, remainder_stub}
declare_dispatch!{binary_fn, bitwise_and_stub}
declare_dispatch!{binary_fn, bitwise_or_stub}
declare_dispatch!{binary_fn, bitwise_xor_stub}
declare_dispatch!{binary_fn, lshift_stub}
declare_dispatch!{binary_fn, rshift_stub}
declare_dispatch!{binary_fn, logical_xor_stub}
declare_dispatch!{binary_fn, logical_and_stub}
declare_dispatch!{binary_fn, logical_or_stub}
declare_dispatch!{binary_fn, lt_stub}
declare_dispatch!{binary_fn, le_stub}
declare_dispatch!{binary_fn, gt_stub}
declare_dispatch!{binary_fn, ge_stub}
declare_dispatch!{binary_fn, eq_stub}
declare_dispatch!{binary_fn, ne_stub}
declare_dispatch!{binary_fn, max_elementwise_stub}
declare_dispatch!{binary_fn, min_elementwise_stub}
declare_dispatch!{structured_binary_fn, maximum_stub}
declare_dispatch!{structured_binary_fn, minimum_stub}
declare_dispatch!{structured_binary_fn, fmax_stub}
declare_dispatch!{structured_binary_fn, fmin_stub}
declare_dispatch!{binary_fn_double, smooth_l1_stub}
declare_dispatch!{binary_fn_double, huber_stub}
declare_dispatch!{binary_fn, sigmoid_backward_stub}
declare_dispatch!{binary_fn_alpha, logit_backward_stub}
declare_dispatch!{binary_fn, tanh_backward_stub}
declare_dispatch!{binary_fn, mse_stub}
declare_dispatch!{binary_fn, fmod_stub}
declare_dispatch!{structured_binary_fn, logaddexp_stub}
declare_dispatch!{structured_binary_fn, logaddexp2_stub}
declare_dispatch!{structured_binary_fn, gcd_stub}
declare_dispatch!{structured_binary_fn, lcm_stub}
declare_dispatch!{structured_binary_fn, hypot_stub}
declare_dispatch!{structured_binary_fn, igamma_stub}
declare_dispatch!{structured_binary_fn, igammac_stub}
declare_dispatch!{structured_binary_fn, nextafter_stub}
declare_dispatch!{structured_binary_fn, heaviside_stub}
declare_dispatch!{structured_binary_fn, copysign_stub}
declare_dispatch!{binary_fn, xlogy_stub}
declare_dispatch!{structured_binary_fn, xlog1py_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/BinaryOps.cpp]

lazy_static!{
    /*
    TORCH_META_FUNC2(add, Tensor) (
      const Tensor& self, const Tensor& other, const Scalar& alpha
    ) {
      build_borrowing_binary_op(maybe_get_output(), self, other);
      native::alpha_check(dtype(), alpha);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC2(sub, Tensor) (
      const Tensor& self, const Tensor& other, const Scalar& alpha
    ) {
      native::sub_check(self, other);
      build_borrowing_binary_op(maybe_get_output(), self, other);
      native::alpha_check(dtype(), alpha);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC2(mul, Tensor) (
      const Tensor& self, const Tensor& other
    ) {
      build_borrowing_binary_op(maybe_get_output(), self, other);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC2(div, Tensor) (const Tensor& self, const Tensor& other) {
      build_borrowing_binary_float_op(maybe_get_output(), self, other);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC2(div, Tensor_mode) (const Tensor& self, const Tensor& other, optional<string_view> rounding_mode) {
      if (!rounding_mode.has_value()) {
        build_borrowing_binary_float_op(maybe_get_output(), self, other);
      } else if (*rounding_mode == "trunc") {
        build_borrowing_binary_op(maybe_get_output(), self, other);
      } else if (*rounding_mode == "floor") {
        build_borrowing_binary_op(maybe_get_output(), self, other);
      } else {
        TORCH_CHECK(false,
            "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
            "but found '", *rounding_mode, "'");
      }
    }
    */
}


lazy_static!{
    /*
    TORCH_META_FUNC(special_xlog1py) (const Tensor& self, const Tensor& other) {
      build_borrowing_binary_float_op(maybe_get_output(), self, other);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC2(copysign, Tensor) (
      const Tensor& self, const Tensor& other
    ) {
      build_borrowing_binary_float_op(maybe_get_output(), self, other);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(heaviside) (
      const Tensor& self, const Tensor& other
    ) {
      TORCH_CHECK(!self.is_complex() && !other.is_complex() &&
                  (maybe_get_output().defined() ? !maybe_get_output().is_complex() : true),
                  "heaviside is not yet implemented for complex tensors.");
      TORCH_CHECK(self.dtype() == other.dtype() &&
                  (maybe_get_output().defined() ? maybe_get_output().dtype() == self.dtype() : true),
                  "heaviside is not yet implemented for tensors with different dtypes.");

      build_binary_op(maybe_get_output(), self, other);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(atan2) (const Tensor& self, const Tensor& other) {
      build_borrowing_binary_float_op(maybe_get_output(), self, other);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC2(remainder, Tensor)(const Tensor& self, const Tensor& other) {
      build_borrowing_binary_op(maybe_get_output(), self, other);
    }
    */
}

/// These are normal binary ops that preserve
/// dtype
///
#[macro_export] macro_rules! create_binary_meta_func {
    ($func:ident) => {
        /*
        
          TORCH_META_FUNC(func) (const Tensor& self, const Tensor& other) {   
            build_borrowing_binary_op(maybe_get_output(), self, other);                 
          }
        */
    }
}

create_binary_meta_func!(logaddexp);
create_binary_meta_func!(logaddexp2);
create_binary_meta_func!(gcd);
create_binary_meta_func!(lcm);
create_binary_meta_func!(hypot);
create_binary_meta_func!(igamma);
create_binary_meta_func!(igammac);
create_binary_meta_func!(nextafter);

lazy_static!{
    /*
    TORCH_META_FUNC(maximum) (const Tensor& self, const Tensor& other) {
      TORCH_CHECK(!self.is_complex() && !other.is_complex(), "maximum not implemented for complex tensors.");
      build_borrowing_binary_op(maybe_get_output(), self, other);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(minimum) (const Tensor& self, const Tensor& other) {
      TORCH_CHECK(!self.is_complex() && !other.is_complex(), "minimum not implemented for complex tensors.");
      build_borrowing_binary_op(maybe_get_output(), self, other);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(fmax) (const Tensor& self, const Tensor& other) {
        TORCH_CHECK(!self.is_complex() && !other.is_complex(), "fmax not implemented for complex tensors.");
        build_binary_op(maybe_get_output(), self, other);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(fmin) (const Tensor& self, const Tensor& other) {
        TORCH_CHECK(!self.is_complex() && !other.is_complex(), "fmin not implemented for complex tensors.");
        build_binary_op(maybe_get_output(), self, other);
    }
    */
}

define_dispatch!{add_stub}
define_dispatch!{add_clamp_stub}
define_dispatch!{sub_stub}
define_dispatch!{mul_stub}
define_dispatch!{div_true_stub}
define_dispatch!{div_floor_stub}
define_dispatch!{div_trunc_stub}
define_dispatch!{remainder_stub}
define_dispatch!{atan2_stub}
define_dispatch!{bitwise_and_stub}
define_dispatch!{bitwise_or_stub}
define_dispatch!{bitwise_xor_stub}
define_dispatch!{lshift_stub}
define_dispatch!{rshift_stub}
define_dispatch!{logical_and_stub}
define_dispatch!{logical_or_stub}
define_dispatch!{logical_xor_stub}
define_dispatch!{lt_stub}
define_dispatch!{le_stub}
define_dispatch!{gt_stub}
define_dispatch!{ge_stub}
define_dispatch!{eq_stub}
define_dispatch!{ne_stub}
define_dispatch!{sigmoid_backward_stub}
define_dispatch!{logit_backward_stub}
define_dispatch!{tanh_backward_stub}
define_dispatch!{maximum_stub}
define_dispatch!{minimum_stub}
define_dispatch!{fmax_stub}
define_dispatch!{fmin_stub}
define_dispatch!{fmod_stub}
define_dispatch!{logaddexp_stub}
define_dispatch!{logaddexp2_stub}
define_dispatch!{gcd_stub}
define_dispatch!{lcm_stub}
define_dispatch!{hypot_stub}
define_dispatch!{igamma_stub}
define_dispatch!{igammac_stub}
define_dispatch!{nextafter_stub}
define_dispatch!{heaviside_stub}
define_dispatch!{copysign_stub}
define_dispatch!{xlogy_stub}
define_dispatch!{xlog1py_stub}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(add_out) (
      const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& result
    ) {
      add_stub(device_type(), *this, alpha);
      TORCH_INTERNAL_ASSERT(result.scalar_type() == output().dtype());
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(sub_out) (
      const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& result
    ) {
      sub_stub(device_type(), *this, alpha);
      TORCH_INTERNAL_ASSERT(result.scalar_type() == output().dtype());
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(mul_out) (
      const Tensor& self, const Tensor& other, const Tensor& result
    ) {
      mul_stub(device_type(), *this);
    }
    */
}


lazy_static!{
    /*
    TORCH_IMPL_FUNC(div_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
      div_true_stub(device_type(), *this);
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(div_out_mode) (
      const Tensor& self, const Tensor& other, optional<string_view> rounding_mode, const Tensor& result
    ) {
      if (!rounding_mode.has_value()) {
        div_true_stub(device_type(), *this);
      } else if (*rounding_mode == "trunc") {
        div_trunc_stub(device_type(), *this);
      } else if (*rounding_mode == "floor") {
        div_floor_stub(device_type(), *this);
      }
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(special_xlog1py_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
      xlog1py_stub(device_type(), *this);
    }
    */
}

#[macro_export] macro_rules! create_binary_torch_impl_func {
    ($func_out:ident, $func_stub:ident) => {
        /*
        
        TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& other, const Tensor& result) {  
          func_stub(device_type(), *this);                                                           
        }
        */
    }
}

create_binary_torch_impl_func!(maximum_out, maximum_stub);
create_binary_torch_impl_func!(minimum_out, minimum_stub);
create_binary_torch_impl_func!(fmax_out, fmax_stub);
create_binary_torch_impl_func!(fmin_out, fmin_stub);
create_binary_torch_impl_func!(logaddexp_out, logaddexp_stub);
create_binary_torch_impl_func!(logaddexp2_out, logaddexp2_stub);
create_binary_torch_impl_func!(gcd_out, gcd_stub);
create_binary_torch_impl_func!(lcm_out, lcm_stub);
create_binary_torch_impl_func!(hypot_out, hypot_stub);
create_binary_torch_impl_func!(igamma_out, igamma_stub);
create_binary_torch_impl_func!(igammac_out, igammac_stub);
create_binary_torch_impl_func!(nextafter_out, nextafter_stub);
create_binary_torch_impl_func!(remainder_out, remainder_stub);

pub fn special_xlog1py_scalar_tensor(
        x: &Scalar,
        y: &Tensor) -> Tensor {
    
    todo!();
        /*
            return special_xlog1py(wrapped_scalar_tensor(x), y);
        */
}

pub fn special_xlog1py_tensor_scalar(
        x: &Tensor,
        y: &Scalar) -> Tensor {
    
    todo!();
        /*
            return special_xlog1py(x, wrapped_scalar_tensor(y));
        */
}

pub fn special_xlog1py_out_scalar_tensor(
        self_:  &Scalar,
        other:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return special_xlog1py_out(result, wrapped_scalar_tensor(self), other);
        */
}

pub fn special_xlog1py_out_tensor_scalar(
        self_:  &Tensor,
        other:  &Scalar,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return special_xlog1py_out(result, self, wrapped_scalar_tensor(other));
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(atan2_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
      atan2_stub(device_type(), *this);
    }
    */
}

pub fn add_relu_impl(
        result: &mut Tensor,
        self_:  &Tensor,
        other:  &Tensor,
        alpha:  &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            auto iter = TensorIterator::binary_op(result, self, other);
      Scalar min_val;
      Scalar max_val;
      if (self.dtype() == kInt) {
        min_val = 0;
        max_val = i32::max;
      } else if (self.dtype() == kLong) {
        min_val = 0;
        max_val = i64::max;
      } else if (self.dtype() == kShort) {
        min_val = 0;
        max_val = i16::max;
      } else if (self.dtype() == kChar) {
        min_val = 0;
        max_val = i8::max;
      } else if (self.dtype() == kFloat) {
        min_val = 0.0;
        max_val = float::max;
      } else if (self.dtype() == kDouble) {
        min_val = 0.0;
        max_val = double::max;
      } else {
        TORCH_INTERNAL_ASSERT(
            "Unsupported datatype for add_relu:", self.dtype().name());
      }

      result = iter.output();
      add_clamp_stub(iter.device_type(), iter, alpha, min_val, max_val);
      return result;
        */
}

pub fn add_relu_out(
        self_:  &Tensor,
        other:  &Tensor,
        alpha:  &Scalar,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return add_relu_impl(result, self, other, alpha);
        */
}

pub fn add_relu(
        self_: &Tensor,
        other: &Tensor,
        alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor result;
      return add_relu_impl(result, self, other, alpha);
        */
}

pub fn add_relu_mut_tensor_tensor_with_alpha(
        self_: &mut Tensor,
        other: &Tensor,
        alpha: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return add_relu_impl(self, self, other, alpha);
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(copysign_out) (
      const Tensor& self, const Tensor& other, const Tensor& result
    ) {
      copysign_stub(device_type(), *this);
    }
    */
}

pub fn copysign(
        self_: &Tensor,
        other: &Scalar) -> Tensor {
    
    todo!();
        /*
            // redispatch!
      return copysign(self, wrapped_scalar_tensor(other));
        */
}

pub fn copysign_mut_tensor_scalar(
        self_: &mut Tensor,
        other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            // redispatch!
      return self.copysign_(wrapped_scalar_tensor(other));
        */
}

pub fn copysign_out(
        self_:  &Tensor,
        other:  &Scalar,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // redispatch!
      return copysign_out(result, self, wrapped_scalar_tensor(other));
        */
}

/**
  | WARNING: There doesn't appear to be
  | any testing for this function with sparse
  | self input.
  |
  */
pub fn div(
        self_: &Tensor,
        other: &Scalar) -> Tensor {
    
    todo!();
        /*
            return self.div(wrapped_scalar_tensor(other)); // redispatch!
        */
}

/**
  | WARNING: This function, with a sparse self, is
  | currently only exercised by
  | DistributedDataParallelTest.test_sparse_gradients
  | (you need to exercise it from C++, because this
  | overload is never used for Python)
  |
  */
pub fn div_mut_tensor_scalar(
        self_: &mut Tensor,
        other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return self.div_(wrapped_scalar_tensor(other)); // redispatch!
        */
}

pub fn div_tensor_scalar_with_rounding_mode(
        self_:         &Tensor,
        other:         &Scalar,
        rounding_mode: Option<StringView>) -> Tensor {
    
    todo!();
        /*
            return self.div(wrapped_scalar_tensor(other), move(rounding_mode)); // redispatch!
        */
}

pub fn div_mut_tensor_scalar_with_rounding_mode(
        self_:         &mut Tensor,
        other:         &Scalar,
        rounding_mode: Option<StringView>) -> &mut Tensor {
    
    todo!();
        /*
            return self.div_(wrapped_scalar_tensor(other), move(rounding_mode)); // redispatch!
        */
}

/**
  | divide, alias for div
  |
  */
pub fn divide_out(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return div_out(result, self, other);
        */
}


pub fn divide(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.div(other);
        */
}


pub fn divide_mut_tensor_tensor(
        self_: &mut Tensor,
        other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return self.div_(other);
        */
}


pub fn divide_tensor_scalar(
        self_: &Tensor,
        other: &Scalar) -> Tensor {
    
    todo!();
        /*
            return self.div(other);
        */
}


pub fn divide_mut_tensor_scalar(
        self_: &mut Tensor,
        other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return self.div_(other);
        */
}

pub fn divide_out_with_rounding_mode(
        self_:         &Tensor,
        other:         &Tensor,
        rounding_mode: Option<StringView>,
        result:        &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return div_out(result, self, other, move(rounding_mode));
        */
}

pub fn divide_tensor_tensor_with_rounding_mode(
    self_:         &Tensor,
    other:         &Tensor,
    rounding_mode: Option<StringView>) -> Tensor {

    todo!();
        /*
            return self.div(other, move(rounding_mode));
        */
}

pub fn divide_mut_tensor_tensor_with_rounding_mode(
    self_:         &mut Tensor,
    other:         &Tensor,
    rounding_mode: Option<StringView>) -> &mut Tensor {
    
    todo!();
        /*
            return self.div_(other, move(rounding_mode));
        */
}

pub fn divide_tensor_scalar_with_rounding_mode(
    self_:         &Tensor,
    other:         &Scalar,
    rounding_mode: Option<StringView>) -> Tensor {
    
    todo!();
        /*
            return self.div(other, move(rounding_mode));
        */
}

pub fn divide_mut_tensor_scalar_with_rounding_mode(
    self_:         &mut Tensor,
    other:         &Scalar,
    rounding_mode: Option<StringView>) -> &mut Tensor {
    
    todo!();
        /*
            return self.div_(other, move(rounding_mode));
        */
}

/**
  | true_divide, an alias for div
  |
  */
pub fn true_divide_out(
    self_:   &Tensor,
    divisor: &Tensor,
    result:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return div_out(result, self, divisor);
        */
}

pub fn true_divide(
    self_:   &Tensor,
    divisor: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.div(divisor);
        */
}

pub fn true_divide_mut(
    self_:   &mut Tensor,
    divisor: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return self.div_(divisor);
        */
}

pub fn true_divide_tensor_scalar(
    self_:   &Tensor,
    divisor: &Scalar) -> Tensor {

    todo!();
        /*
            return self.div(divisor);
        */
}

pub fn true_divide_mut_tensor_scalar(
    self_:   &mut Tensor,
    divisor: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return self.div_(divisor);
        */
}

pub fn floor_divide_out(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_WARN_ONCE(
        "floor_divide is deprecated, and will be removed in a future version of pytorch. "
        "It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). "
        "This results in incorrect rounding for negative values.\n"
        "To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), "
        "or for actual floor division, use torch.div(a, b, rounding_mode='floor')."
      );
      // FIXME: Not actually doing floor division (#43874)
      auto iter = TensorIterator::binary_op(result, self, other);
      div_trunc_stub(iter.device_type(), iter);
      if (!result.defined()) {
        result = iter.output();
      }
      return result;
        */
}

pub fn floor_divide(
    self_: &Tensor,
    other: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_WARN_ONCE(
        "floor_divide is deprecated, and will be removed in a future version of pytorch. "
        "It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). "
        "This results in incorrect rounding for negative values.\n"
        "To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), "
        "or for actual floor division, use torch.div(a, b, rounding_mode='floor')."
      );
      // FIXME: Not actually doing floor division (#43874)
      Tensor result;
      auto iter = TensorIterator::binary_op(result, self, other);
      div_trunc_stub(iter.device_type(), iter);
      return iter.output();
        */
}

pub fn floor_divide_mut(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {

    todo!();
        /*
            return native::floor_divide_out(self, other, self);
        */
}

/**
  | TODO: Make this structured to undo the
  | perf regression from native:: removal
  | in call here
  |
  */
pub fn mul(
        self_: &Tensor,
        other: &Scalar) -> Tensor {
    
    todo!();
        /*
            return mul(self, wrapped_scalar_tensor(other)); // redispatch!
        */
}

pub fn mul_mut(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return mul_out(self, wrapped_scalar_tensor(other), self); // redispatch!
        */
}

/**
  | multiply, alias for mul
  |
  */
pub fn multiply_out(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return mul_out(result, self, other);
        */
}


pub fn multiply(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.mul(other);
        */
}


pub fn multiply_mut(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return self.mul_(other);
        */
}

pub fn multiply_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {
    
    todo!();
        /*
            return self.mul(other);
        */
}

pub fn multiply_mut_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return self.mul_(other);
        */
}

pub fn sub(
    self_: &Tensor,
    other: &Scalar,
    alpha: &Scalar) -> Tensor {

    todo!();
        /*
            return sub(self, wrapped_scalar_tensor(other), alpha); // redispatch!
        */
}

pub fn sub_mut(
    self_: &mut Tensor,
    other: &Scalar,
    alpha: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return self.sub_(wrapped_scalar_tensor(other), alpha); // redispatch!
        */
}

/**
  | subtract, alias for sub
  |
  */
pub fn subtract_out(
    self_:  &Tensor,
    other:  &Tensor,
    alpha:  &Scalar,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            return sub_out(result, self, other, alpha);
        */
}

pub fn subtract(
    self_: &Tensor,
    other: &Tensor,
    alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            return self.sub(other, alpha);
        */
}

pub fn subtract_mut(
    self_: &mut Tensor,
    other: &Tensor,
    alpha: &Scalar) -> &mut Tensor {

    todo!();
        /*
            return self.sub_(other, alpha);
        */
}


pub fn subtract_tensor_scalar_scalar(
        self_: &Tensor,
        other: &Scalar,
        alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            return self.sub(other, alpha);
        */
}

pub fn subtract_mut_tensor_scalar_scalar(
    self_: &mut Tensor,
    other: &Scalar,
    alpha: &Scalar) -> &mut Tensor {

    todo!();
        /*
            return self.sub_(other, alpha);
        */
}


pub fn sigmoid_backward_out(
        grad_output: &Tensor,
        output:      &Tensor,
        result:      &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto iter = TensorIterator::binary_op(result, grad_output, output);
      sigmoid_backward_stub(iter.device_type(), iter);
      return result;
        */
}


pub fn sigmoid_backward(
        grad_output: &Tensor,
        output:      &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result;
      auto iter = TensorIterator::binary_op(result, grad_output, output);
      sigmoid_backward_stub(iter.device_type(), iter);
      return iter.output();
        */
}


pub fn logit_backward_out(
        grad_output: &Tensor,
        input:       &Tensor,
        eps:         Option<f64>,
        result:      &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto iter = TensorIterator::binary_op(result, grad_output, input);
      logit_backward_stub(
          iter.device_type(), iter, Scalar(eps ? eps.value() : -1.0));
      return result;
        */
}


pub fn logit_backward(
        grad_output: &Tensor,
        input:       &Tensor,
        eps:         Option<f64>) -> Tensor {
    
    todo!();
        /*
            Tensor result;
      auto iter = TensorIterator::binary_op(result, grad_output, input);
      logit_backward_stub(
          iter.device_type(), iter, Scalar(eps ? eps.value() : -1.0));
      return iter.output();
        */
}


pub fn tanh_backward_out(
        grad_output: &Tensor,
        output:      &Tensor,
        result:      &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto iter = TensorIterator::binary_op(result, grad_output, output);
      tanh_backward_stub(iter.device_type(), iter);
      return result;
        */
}


pub fn tanh_backward(
        grad_output: &Tensor,
        output:      &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result;
      auto iter = TensorIterator::binary_op(result, grad_output, output);
      tanh_backward_stub(iter.device_type(), iter);
      return iter.output();
        */
}


pub fn rsub_tensor_tensor_scalar(
        self_: &Tensor,
        other: &Tensor,
        alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            return sub(other, self, alpha); // redispatch!
        */
}

/**
  | These are still needed because we don't
  | have C++ conversions from number types
  | (int, float, etc.) to Tensor (only to
  | Scalar). They're not exposed to Python.
  |
  */
pub fn check_convert(
        scalar:      &Scalar,
        scalar_type: ScalarType)  {
    
    todo!();
        /*
            // Validate that is possible to convert scalar to tensor dtype without overflow
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half, scalarType, "check_convert", [&]{
        scalar.to<Scalar>();
      });
        */
}


pub fn wrapped_scalar_tensor_and_check_convert(
        scalar: &Scalar,
        tensor: Tensor) -> Tensor {
    
    todo!();
        /*
            check_convert(scalar, tensor.scalar_type());
      return wrapped_scalar_tensor(scalar);
        */
}

/**
  | TODO: Make this structured to undo the
  | perf regression from native:: removal
  | in call here
  |
  */
pub fn add(
        self_: &Tensor,
        other: &Scalar,
        alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            return add(self, wrapped_scalar_tensor(other), alpha);
        */
}

pub fn add_mut_tensor_scalar_scalar(
    self_: &mut Tensor,
    other: &Scalar,
    alpha: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return self.add_(wrapped_scalar_tensor(other), alpha);
        */
}

pub fn remainder_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {

    todo!();
        /*
            // redispatch
      return remainder(self, wrapped_scalar_tensor(other));
        */
}

pub fn remainder_mut(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {

    todo!();
        /*
            // redispatch
      return self.remainder_(wrapped_scalar_tensor(other));
        */
}

pub fn remainder_out(
    self_:  &Tensor,
    other:  &Scalar,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            // redispatch
      return remainder_out(result, self, wrapped_scalar_tensor(other));
        */
}

pub fn remainder_scalar_tensor(
        self_: &Scalar,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return remainder(wrapped_scalar_tensor(self), other);
        */
}

pub fn rsub_tensor_scalar_scalar(
        self_: &Tensor,
        other: &Scalar,
        alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            return native::rsub(self, wrapped_scalar_tensor(other), alpha);
        */
}

pub fn bitwise_and_out(
    self_:  &Tensor,
    other:  &Tensor,
    result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto iter = TensorIterator::binary_op(result, self, other);
      bitwise_and_stub(iter.device_type(), iter);
      return result;
        */
}

pub fn bitwise_and(
    self_: &Tensor,
    other: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      bitwise_and_out(result, self, other);
      return result;
        */
}

pub fn bitwise_and_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return bitwise_and_out(self, self, other);
        */
}

pub fn bitwise_and_out_tensor_scalar(
    self_:  &Tensor,
    other:  &Scalar,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            return bitwise_and_out(result, self, wrapped_scalar_tensor(other));
        */
}

pub fn bitwise_and_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {

    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return bitwise_and_out(result, self, other);
        */
}

pub fn bitwise_and_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {

    todo!();
        /*
            return bitwise_and_out(self, self, other);
        */
}

/**
  | Legacy and interfaces. They are aliased
  | to bitwise_and* functions
  |
  */
pub fn and_tensor_tensor(
    self_: &Tensor,
    other: &Tensor) -> Tensor {

    todo!();
        /*
            return bitwise_and(self, other);
        */
}

pub fn and_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {

    todo!();
        /*
            return bitwise_and(self, other);
        */
}

pub fn iand_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return self.bitwise_and_(other);
        */
}

pub fn iand_mut_tensor_scalar(
        self_: &mut Tensor,
        other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return self.bitwise_and_(other);
        */
}

pub fn bitwise_or_out(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto iter = TensorIterator::binary_op(result, self, other);
      bitwise_or_stub(iter.device_type(), iter);
      return result;
        */
}

pub fn bitwise_or(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      bitwise_or_out(result, self, other);
      return result;
        */
}

pub fn bitwise_or_tensor_mut(
        self_: &mut Tensor,
        other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return bitwise_or_out(self, self, other);
        */
}

pub fn bitwise_or_out_tensor_scalar(
        self_:  &Tensor,
        other:  &Scalar,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return bitwise_or_out(result, self, wrapped_scalar_tensor(other));
        */
}


pub fn bitwise_or_tensor_scalar(
        self_: &Tensor,
        other: &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return bitwise_or_out(result, self, other);
        */
}

pub fn bitwise_or_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return bitwise_or_out(self, self, other);
        */
}

/**
  | Legacy or interfaces. They are aliased
  | to bitwise_or* functions
  |
  */
pub fn or_tensor_tensor(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return bitwise_or(self, other);
        */
}

pub fn or_tensor_scalar(
        self_: &Tensor,
        other: &Scalar) -> Tensor {
    
    todo!();
        /*
            return bitwise_or(self, other);
        */
}


pub fn ior_mut_tensor_tensor(
        self_: &mut Tensor,
        other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return self.bitwise_or_(other);
        */
}

pub fn ior_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return self.bitwise_or_(other);
        */
}


pub fn bitwise_xor_out_tensor_tensor(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto iter = TensorIterator::binary_op(result, self, other);
      bitwise_xor_stub(iter.device_type(), iter);
      return result;
        */
}


pub fn bitwise_xor(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      bitwise_xor_out(result, self, other);
      return result;
        */
}

pub fn bitwise_xor_mut(
        self_: &mut Tensor,
        other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return bitwise_xor_out(self, self, other);
        */
}


pub fn bitwise_xor_out_tensor_scalar(
        self_:  &Tensor,
        other:  &Scalar,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return bitwise_xor_out(result, self, wrapped_scalar_tensor(other));
        */
}

pub fn bitwise_xor_tensor_scalar(
        self_: &Tensor,
        other: &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return bitwise_xor_out(result, self, other);
        */
}


pub fn bitwise_xor_mut_tensor_scalar(
        self_: &mut Tensor,
        other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return bitwise_xor_out(self, self, other);
        */
}

/**
  | Legacy xor interfaces. They are aliased
  | to bitwise_xor* functions
  |
  */
pub fn xor(
    self_: &Tensor,
    other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return bitwise_xor(self, other);
        */
}

pub fn xor_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {

    todo!();
        /*
            return bitwise_xor(self, other);
        */
}

pub fn ixor_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return self.bitwise_xor_(other);
        */
}

pub fn ixor_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return self.bitwise_xor_(other);
        */
}

pub fn lshift(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result;
      auto iter = TensorIterator::binary_op(result, self, other);
      lshift_stub(iter.device_type(), iter);
      return iter.output();
        */
}

pub fn lshift_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor result;
      auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
      auto iter = TensorIterator::binary_op(result, self, wrapper);
      lshift_stub(iter.device_type(), iter);
      return iter.output();
        */
}

pub fn ilshift_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto iter = TensorIterator::binary_op(self, self, other);
      lshift_stub(iter.device_type(), iter);
      return self;
        */
}

pub fn ilshift_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
      auto iter = TensorIterator::binary_op(self, self, wrapper);
      lshift_stub(iter.device_type(), iter);
      return self;
        */
}

pub fn rshift_tensor_tensor(
    self_: &Tensor,
    other: &Tensor) -> Tensor {

    todo!();
        /*
            Tensor result;
      auto iter = TensorIterator::binary_op(result, self, other);
      rshift_stub(iter.device_type(), iter);
      return iter.output();
        */
}

pub fn rshift_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor result;
      auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
      auto iter = TensorIterator::binary_op(result, self, wrapper);
      rshift_stub(iter.device_type(), iter);
      return iter.output();
        */
}

pub fn irshift_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto iter = TensorIterator::binary_op(self, self, other);
      rshift_stub(iter.device_type(), iter);
      return self;
        */
}

pub fn irshift_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {

    todo!();
        /*
            auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
      auto iter = TensorIterator::binary_op(self, self, wrapper);
      rshift_stub(iter.device_type(), iter);
      return self;
        */
}

pub fn comparison_op_out_mut_with_stub<Stub>(
    result: &mut Tensor,
    self_:  &Tensor,
    other:  &Tensor,
    stub:   &mut Stub) -> &mut Tensor {

    todo!();
    /*
            // Validate that is possible to convert zero-dim tensor's dtype to other dtype without overflow
      if (self.scalar_type() != other.scalar_type()) {
        if (self.dim() != 0 && other.dim() == 0) {
          check_convert(other.item(), self.scalar_type());
        } else if (self.dim() == 0 && other.dim() != 0) {
          check_convert(self.item(), other.scalar_type());
        }
      }
      auto iter = TensorIterator::comparison_op(result, self, other);
      stub(iter.device_type(), iter);
      return result;
        */
}

pub fn comparison_op_tensor_tensor<OutImpl>(
    self_:    &Tensor,
    other:    &Tensor,
    out_impl: &mut OutImpl) -> Tensor {

    todo!();
        /*
            Tensor result = empty({0}, self.options().dtype(kBool));
      return out_impl(result, self, other);
        */
}

/**
  | To avoid overflow during type promotion
  | we will check that both dtypes of self
  | and other are same
  |
  */
pub fn comparison_op_mut<OutImpl>(
    self_:    &mut Tensor,
    other:    &Tensor,
    out_impl: &mut OutImpl) -> &mut Tensor {

    todo!();
        /*
            TORCH_CHECK(self.dtype() == other.dtype(),
                  "Expected object of scalar type ", self.dtype(), " but got scalar type ",
                  other.dtype(), " for argument 'other'");
      return out_impl(self, self, other);
        */
}


/**
  | validates that is possible to convert
  | Scalar other to self's dtype without
  | overflow.
  | 
  | This behavior is unique to comparison
  | ops; arithmetic operations don't do
  | this.
  | 
  | In the future, we should reconsider
  | this inconsistency and decide if we
  | want to add the same check to arithmetic
  | ops.
  |
  */
pub fn comparison_op_out<OutImpl>(
    result:   &mut Tensor,
    self_:    &Tensor,
    other:    &Scalar,
    out_impl: &mut OutImpl) -> &mut Tensor {

    todo!();
        /*
            return out_impl(result, self, wrapped_scalar_tensor_and_check_convert(other, self));
        */
}

pub fn comparison_op_tensor_scalar<OutImpl>(
    self_:    &Tensor,
    other:    &Scalar,
    out_impl: &mut OutImpl) -> Tensor {

    todo!();
        /*
            return comparison_op(self, wrapped_scalar_tensor_and_check_convert(other, self), out_impl);
        */
}

pub fn comparison_op_mut_tensor_scalar<OutImpl>(
    self_:    &mut Tensor,
    other:    &Scalar,
    out_impl: &mut OutImpl) -> &mut Tensor {

    todo!();
        /*
            return out_impl(self, self, wrapped_scalar_tensor_and_check_convert(other, self));
        */
}

/**
  | We need explicit cast to OutFunc because each
  | *_out func is overloaded twice. Without An
  | explicit cast, merely referring to *_out
  | function is ambiguious.
  |
  */
lazy_static!{
    /*
    using OutFunc = add_const<Tensor&(&)(Tensor&, const Tensor&, const Tensor&)>::type;
    */
}

pub fn lt_out(
    self_:  &Tensor,
    other:  &Tensor,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            return comparison_op_out(result, self, other, lt_stub);
        */
}

pub fn lt(
    self_: &Tensor,
    other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(lt_out));
        */
}

pub fn lt_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {

    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(lt_out));
        */
}

pub fn lt_out_tensor_scalar(
    self_:  &Tensor,
    other:  &Scalar,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            return comparison_op_out(result, self, other, static_cast<OutFunc>(lt_out));
        */
}

pub fn lt_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {
    
    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(lt_out));
        */
}

pub fn lt_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(lt_out));
        */
}

/**
  | less, alias for torch.lt
  |
  */
pub fn less_out(
    self_:  &Tensor,
    other:  &Tensor,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            return lt_out(result, self, other);
        */
}

pub fn less_tensor_tensor(
    self_: &Tensor,
    other: &Tensor) -> Tensor {

    todo!();
        /*
            return self.lt(other);
        */
}

pub fn less_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {

    todo!();
        /*
            return self.lt_(other);
        */
}

pub fn less_out_tensor_scalar(
    self_:  &Tensor,
    other:  &Scalar,
    result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return lt_out(result, self, other);
        */
}

pub fn less_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {
    
    todo!();
        /*
            return self.lt(other);
        */
}

pub fn less_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return self.lt_(other);
        */
}

pub fn le_out_tensor_tensor(
    self_:  &Tensor,
    other:  &Tensor,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            return comparison_op_out(result, self, other, le_stub);
        */
}

pub fn le_tensor_tensor(
    self_: &Tensor,
    other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(le_out));
        */
}

pub fn le_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(le_out));
        */
}

pub fn le_out_tensor_scalar(
        self_:  &Tensor,
        other:  &Scalar,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_out(result, self, other, static_cast<OutFunc>(le_out));
        */
}

pub fn le_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {
    
    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(le_out));
        */
}

pub fn le_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(le_out));
        */
}

/**
  | less_equal, alias for torch.le
  |
  */
pub fn less_equal_out_tensor_tensor(
    self_:  &Tensor,
    other:  &Tensor,
    result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return le_out(result, self, other);
        */
}

pub fn less_equal_tensor_tensor(
    self_: &Tensor,
    other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.le(other);
        */
}

pub fn less_equal_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return self.le_(other);
        */
}

pub fn less_equal_out_tensor_scalar(
    self_:  &Tensor,
    other:  &Scalar,
    result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return le_out(result, self, other);
        */
}

pub fn less_equal_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {
    
    todo!();
        /*
            return self.le(other);
        */
}

pub fn less_equal_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return self.le_(other);
        */
}

pub fn gt_out_tensor_tensor(
    self_:  &Tensor,
    other:  &Tensor,
    result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_out(result, self, other, gt_stub);
        */
}

pub fn gt_tensor_tensor(
    self_: &Tensor,
    other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(gt_out));
        */
}

pub fn gt_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {

    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(gt_out));
        */
}

pub fn gt_out_tensor_scalar(
    self_:  &Tensor,
    other:  &Scalar,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            return comparison_op_out(result, self, other, static_cast<OutFunc>(gt_out));
        */
}

pub fn gt_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {
    
    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(gt_out));
        */
}

pub fn gt_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(gt_out));
        */
}

/**
  | greater, alias for torch.gt
  |
  */
pub fn greater_out_tensor_tensor(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return gt_out(result, self, other);
        */
}

pub fn greater_tensor_tensor(
    self_: &Tensor,
    other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.gt(other);
        */
}

pub fn greater_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return self.gt_(other);
        */
}

pub fn greater_out_tensor_scalar(
    self_:  &Tensor,
    other:  &Scalar,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            return gt_out(result, self, other);
        */
}

pub fn greater_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {

    todo!();
        /*
            return self.gt(other);
        */
}

pub fn greater_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return self.gt_(other);
        */
}

pub fn ge_out_tensor_tensor(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_out(result, self, other, ge_stub);
        */
}

pub fn ge_tensor_tensor(
    self_: &Tensor,
    other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(ge_out));
        */
}

pub fn ge_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(ge_out));
        */
}

pub fn ge_out_tensor_scalar(
        self_:  &Tensor,
        other:  &Scalar,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_out(result, self, other, static_cast<OutFunc>(ge_out));
        */
}

pub fn ge_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {
    
    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(ge_out));
        */
}

pub fn ge_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(ge_out));
        */
}

/**
  | greater_equal, alias for torch.ge
  |
  */
pub fn greater_equal_out_tensor_tensor(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return ge_out(result, self, other);
        */
}

pub fn greater_equal_tensor_tensor(
    self_: &Tensor,
    other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.ge(other);
        */
}

pub fn greater_equal_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return self.ge_(other);
        */
}

pub fn greater_equal_out_tensor_scalar(
    self_:  &Tensor,
    other:  &Scalar,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            return ge_out(result, self, other);
        */
}

pub fn greater_equal_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {

    todo!();
        /*
            return self.ge(other);
        */
}

pub fn greater_equal_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return self.ge_(other);
        */
}

pub fn eq_out(
    self_:  &Tensor,
    other:  &Tensor,
    result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_out(result, self, other, eq_stub);
        */
}

pub fn eq_tensor_tensor(
    self_: &Tensor,
    other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(eq_out));
        */
}

pub fn eq_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(eq_out));
        */
}

pub fn eq_out_tensor_scalar(
    self_:  &Tensor,
    other:  &Scalar,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            return comparison_op_out(result, self, other, static_cast<OutFunc>(eq_out));
        */
}

pub fn eq_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {
    
    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(eq_out));
        */
}

pub fn eq_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(eq_out));
        */
}

pub fn ne_out(
    self_:  &Tensor,
    other:  &Tensor,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            return comparison_op_out(result, self, other, ne_stub);
        */
}

pub fn ne_tensor_tensor(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(ne_out));
        */
}

pub fn ne_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(ne_out));
        */
}

pub fn ne_out_tensor_scalar(
    self_:  &Tensor,
    other:  &Scalar,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            return comparison_op_out(result, self, other, static_cast<OutFunc>(ne_out));
        */
}

pub fn ne_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {

    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(ne_out));
        */
}

pub fn ne_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(ne_out));
        */
}

/**
  | not_equal, alias for torch.ne
  |
  */
pub fn not_equal_out_tensor_tensor(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return ne_out(result, self, other);
        */
}

pub fn not_equal_tensor_tensor(
    self_: &Tensor,
    other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.ne(other);
        */
}

pub fn not_equal_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return self.ne_(other);
        */
}

pub fn not_equal_out_tensor_scalar(
    self_:  &Tensor,
    other:  &Scalar,
    result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return ne_out(result, self, other);
        */
}

pub fn not_equal_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {

    todo!();
        /*
            return self.ne(other);
        */
}

pub fn not_equal_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return self.ne_(other);
        */
}

pub fn logical_and_out_tensor_tensor(
    self_:  &Tensor,
    other:  &Tensor,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            return comparison_op_out(result, self, other, logical_and_stub);
        */
}

pub fn logical_and_tensor_tensor(
    self_: &Tensor,
    other: &Tensor) -> Tensor {

    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(logical_and_out));
        */
}

pub fn logical_and_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(logical_and_out));
        */
}

pub fn logical_and_out(
    result: &mut Tensor,
    self_:  &Tensor,
    other:  &Scalar) -> &mut Tensor {

    todo!();
        /*
            return comparison_op_out(result, self, other, static_cast<OutFunc>(logical_and_out));
        */
}

pub fn logical_and_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {
    
    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(logical_and_out));
        */
}

pub fn logical_and_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(logical_and_out));
        */
}

pub fn logical_or_out(
    self_:  &Tensor,
    other:  &Tensor,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            return comparison_op_out(result, self, other, logical_or_stub);
        */
}

pub fn logical_or_tensor_tensor(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(logical_or_out));
        */
}

pub fn logical_or_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {

    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(logical_or_out));
        */
}

pub fn logical_or_out_mut_tensor_tensor(
    result: &mut Tensor,
    self_:  &Tensor,
    other:  &Scalar) -> &mut Tensor {

    todo!();
        /*
            return comparison_op_out(result, self, other, static_cast<OutFunc>(logical_or_out));
        */
}

pub fn logical_or_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {

    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(logical_or_out));
        */
}

pub fn logical_or_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(logical_or_out));
        */
}

pub fn logical_xor_out_tensor_tensor(
    self_:  &Tensor,
    other:  &Tensor,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            return comparison_op_out(result, self, other, logical_xor_stub);
        */
}

pub fn logical_xor_tensor_tensor(
    self_: &Tensor,
    other: &Tensor) -> Tensor {

    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(logical_xor_out));
        */
}

pub fn logical_xor_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {

    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(logical_xor_out));
        */
}

pub fn logical_xor_out_mut_tensor_tensor(
    result: &mut Tensor,
    self_:  &Tensor,
    other:  &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_out(result, self, other, static_cast<OutFunc>(logical_xor_out));
        */
}

pub fn logical_xor_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {
    
    todo!();
        /*
            return comparison_op(self, other, static_cast<OutFunc>(logical_xor_out));
        */
}

pub fn logical_xor_mut_tensor_scalar(
        self_: &mut Tensor,
        other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return comparison_op_(self, other, static_cast<OutFunc>(logical_xor_out));
        */
}

/**
  | binary max, alias for maximum
  |
  */
pub fn max_out(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return maximum_out(result, self, other);
        */
}

pub fn max_tensor_tensor(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return maximum(self, other);
        */
}

/**
  | binary min, alias for minimum
  |
  */
pub fn min_out(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return minimum_out(result, self, other);
        */
}


pub fn min_tensor_tensor(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return minimum(self, other);
        */
}

pub fn fmin_out(
    self_:  &Tensor,
    other:  &Tensor,
    result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!self.is_complex() && !other.is_complex(), "fmin not implemented for complex tensors.");

      auto iter = TensorIterator::binary_op(result, self, other);
      fmin_stub(iter.device_type(), iter);
      return result;
        */
}

pub fn fmin_tensor_tensor(
    self_: &Tensor,
    other: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!self.is_complex() && !other.is_complex(), "fmin not implemented for complex tensors.");

      Tensor result;
      auto iter = TensorIterator::binary_op(result, self, other);
      fmin_stub(iter.device_type(), iter);
      return iter.output();
        */
}

pub trait FloorDivide<X> {
    type Output;
    fn floor_divide(&self, x: X) -> Self::Output;
}

pub trait FloorDivideInplace<X> {
    fn floor_divide_inplace(&mut self, x: X);
}

impl FloorDivide<&Scalar> for Tensor {

    type Output = Tensor;

    fn floor_divide(&self, x: &Scalar) -> Tensor {

        todo!();
            /*
                return floor_divide(self, wrapped_scalar_tensor(other));
            */
    }
}

impl FloorDivideInplace<&Scalar> for Tensor {

    fn floor_divide_inplace(&mut self, x: &Scalar) {

        todo!();
            /*
                return floor_divide_out(self, self, wrapped_scalar_tensor(other));
            */
    }
}

pub fn fmod_out(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto iter = TensorIterator::binary_op(result, self, other);
      fmod_stub(iter.device_type(), iter);
      return result;
        */
}

pub fn fmod_out_tensor_scalar(
    self_:  &Tensor,
    other:  &Scalar,
    result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return native::fmod_out(self, wrapped_scalar_tensor(other), result);
        */
}

pub fn fmod_tensor_tensor(
    self_: &Tensor,
    other: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result;
      auto iter = TensorIterator::binary_op(result, self, other);
      fmod_stub(iter.device_type(), iter);
      return iter.output();
        */
}

pub fn fmod_tensor_scalar(
    self_: &Tensor,
    other: &Scalar) -> Tensor {
    
    todo!();
        /*
            return native::fmod(self, wrapped_scalar_tensor(other));
        */
}

pub fn fmod_mut_tensor_tensor(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return native::fmod_out(self, other, self);
        */
}

pub fn fmod_mut_tensor_scalar(
    self_: &mut Tensor,
    other: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return native::fmod_(self, wrapped_scalar_tensor(other));
        */
}

/**
  | -----------
  | @note
  | 
  | this function is only for testing.
  | 
  | It is undocumented and should not be
  | used outside of tests.
  |
  */
pub fn test_serialization_subcmul(
        self_: &Tensor,
        other: &Tensor,
        alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            return self - (other * alpha);
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(heaviside_out) (
      const Tensor& self, const Tensor& other, const Tensor& result
    ) {
      heaviside_stub(device_type(), *this);
    }
    */
}

pub fn ldexp_out_tensor_tensor(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return mul_out(result, self, pow(2.0, other));
        */
}

pub fn ldexp_tensor_tensor(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return mul(self, pow(2.0, other));
        */
}

pub fn ldexp_mut_tensor_tensor(
        self_: &mut Tensor,
        other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return ldexp_out(self, self, other);
        */
}

pub fn xlogy_out(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto iter = TensorIterator::binary_float_op(result, self, other);
      xlogy_stub(iter.device_type(), iter);
      return result;
        */
}

pub fn xlogy_out_scalar_tensor(
        self_:  &Scalar,
        other:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return xlogy_out(result, wrapped_scalar_tensor(self), other);
        */
}


pub fn xlogy_out_tensor_scalar(
        self_:  &Tensor,
        other:  &Scalar,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return xlogy_out(result, self, wrapped_scalar_tensor(other));
        */
}


pub fn xlogy_tensor_tensor(
        x: &Tensor,
        y: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result;
      auto iter = TensorIterator::binary_float_op(result, x, y);
      xlogy_stub(iter.device_type(), iter);
      return iter.output();
        */
}

pub fn xlogy_scalar_tensor(
        x: &Scalar,
        y: &Tensor) -> Tensor {
    
    todo!();
        /*
            return xlogy(wrapped_scalar_tensor(x), y);
        */
}


pub fn xlogy_tensor_scalar(
        x: &Tensor,
        y: &Scalar) -> Tensor {
    
    todo!();
        /*
            return xlogy(x, wrapped_scalar_tensor(y));
        */
}


pub fn xlogy_mut_tensor_tensor(
        x: &mut Tensor,
        y: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return xlogy_out(x, x, y);
        */
}


pub fn xlogy_mut_tensor_scalar(
        x: &mut Tensor,
        y: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return xlogy_out(x, x, wrapped_scalar_tensor(y));
        */
}
