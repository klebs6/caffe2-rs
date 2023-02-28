crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Pow.h]

/**
  | integral power in pytorch allows for negative
  | exponents, giving truncated integral results.
  |
  | e.g. since 2**-1==0.5, the truncated integral
  | result is zero. 1**negative_exponent is the
  | only non-zero result.
  |
  */
#[inline] pub fn powi_impl(a: T, b: T) -> T where T: Integer {
//#[__ubsan_ignore_signed_int_overflow__] 
    
    todo!();
        /*
            T result = 1;
      while (b) {
        if (b & 1) {
           result *= a;
        }
        b /= 2;
        a *= a;
      }
      return result;
        */
}

#[inline] pub fn powi_unsigned(a: T, b: T) -> T where T: Integer + Unsigned {
    
    todo!();
        /*
            return powi_impl(a, b);
        */
}

#[inline] pub fn powi_signed(a: T, b: T) -> T where T: Integer + Signed {
    
    todo!();
        /*
            if ( b < 0 ) {
          if ( a == 1 ) {
              return 1;
          } else if ( a == -1 ) {
              auto negative = (-b) % static_cast<T>(2);
              return negative ? -1 : 1;
          } else {
              return 0;
          }
      }
      return powi_impl(a, b);
        */
}

lazy_static!{
    /*
    using pow_tensor_tensor_fn = void (*)(TensorIteratorBase&);
    using pow_tensor_scalar_fn = void (*)(TensorIteratorBase&, const Scalar&);
    */
}

declare_dispatch!{pow_tensor_tensor_fn, pow_tensor_tensor_stub}
declare_dispatch!{pow_tensor_scalar_fn, pow_tensor_scalar_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Pow.cpp]

lazy_static!{
    /*
    TORCH_META_FUNC2(pow, Tensor_Tensor) (const Tensor& base, const Tensor& exp) {
      build_borrowing_binary_op(maybe_get_output(), base, exp);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC2(pow, Tensor_Scalar) (const Tensor& base, const Scalar& exp) {
      // Numpy compatibility check:
      TORCH_CHECK(!(isIntegralType(base.scalar_type(), true) &&
                  exp.isIntegral(true) && exp.toLong() < 0),
                  "Integers to negative integer powers are not allowed.");

      auto common_dtype = result_type(base, exp);
      build_unary_op(maybe_get_output(), base.to(common_dtype));
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC2(pow, Scalar) (const Scalar& base, const Tensor& exp) {
        // This overload doesn't directly use TensorIterator. It attempts to short-circuit,
        // but otherwise redispatches to the Tensor_Tensor overload.
        auto dtype = maybe_get_output().defined() ? maybe_get_output().scalar_type() : result_type(base, exp);
        set_output(0, exp.sizes(), {}, exp.options().dtype(dtype), exp.has_names() ? exp.names() : ArrayRef<Dimname>());
    }
    */
}

define_dispatch!{pow_tensor_tensor_stub}
define_dispatch!{pow_tensor_scalar_stub}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(pow_Tensor_Tensor_out) (const Tensor& base, const Tensor& exp, const Tensor& out) {
      if (exp.dim() == 0 && exp.device().is_cpu() && base.is_cuda()) {
        pow_out(const_cast<Tensor&>(out), base, exp.item()); // redispatch!
      } else {
        pow_tensor_tensor_stub(device_type(), *this);
      }
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(pow_Tensor_Scalar_out) (const Tensor& base, const Scalar& exp, const Tensor& out) {
      if (exp.equal(0.0)) {
        out.fill_(1);
      } else if (exp.equal(1.0)) {
        out.copy_(base);
      } else {
        pow_tensor_scalar_stub(device_type(), *this, exp);
      }
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(pow_Scalar_out) (const Scalar& base, const Tensor& exp, const Tensor& out) {
      if (base.isComplex() && base.toComplexDouble() == 1.0) {
        out.fill_(1);
      } else if (!base.isComplex() && base.toDouble() == 1.0) {
        out.fill_(1);
      } else {
        pow_out(const_cast<Tensor&>(out), wrapped_scalar_tensor(base, exp.device()), exp); // redispatch!
      }
    }
    */
}

#[deprecated = "figure out a better way to handle the overloads"]
pub fn float_power_out_a(
    base:   &Tensor,
    exp:    &Tensor,
    result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto dtype = (isComplexType(base.scalar_type()) || isComplexType(exp.scalar_type())) ?
                    kComplexDouble : kDouble;
      TORCH_CHECK(result.scalar_type() == dtype,
                  "the output given to float_power has dtype ", result.scalar_type(),
                  " but the operation's result requires dtype ", dtype);

      return pow_out(result, base.to(dtype), exp.to(dtype));
        */
}

#[deprecated = "figure out a better way to handle the overloads"]
pub fn float_power_out_b(
    base:   &Tensor,
    exp:    &Scalar,
    result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto dtype = (isComplexType(base.scalar_type()) || exp.isComplex()) ? kComplexDouble : kDouble;
      TORCH_CHECK(result.scalar_type() == dtype,
                  "the output given to float_power has dtype ", result.scalar_type(),
                  " but the operation's result requires dtype ", dtype);

      // Note: need the casts inside the ternary because conversion functions return e.g. complex,
      // which causes a complex scalar to always be returned.
      auto casted_exp = (dtype == kComplexDouble) ? Scalar(exp.toComplexDouble()) : Scalar(exp.toDouble());
      return pow_out(result, base.to(dtype), casted_exp);
        */
}

#[deprecated = "figure out a better way to handle the overloads"]
pub fn float_power_out_c(
    base:   &Scalar,
    exp:    &Tensor,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            auto dtype = (isComplexType(exp.scalar_type()) || base.isComplex()) ? kComplexDouble : kDouble;
      TORCH_CHECK(result.scalar_type() == dtype,
                  "the output given to float_power has dtype ", result.scalar_type(),
                  " but the operation's result requires dtype ", dtype);

      auto casted_base = (dtype == kComplexDouble) ? Scalar(base.toComplexDouble()) : Scalar(base.toDouble());
      return pow_out(result, casted_base, exp.to(dtype));
        */
}

#[deprecated = "figure out a better way to handle the overloads"]
pub fn float_power_a(
        base: &Tensor,
        exp:  &Scalar) -> Tensor {
    
    todo!();
        /*
            auto dtype = (isComplexType(base.scalar_type()) || exp.isComplex()) ? kComplexDouble : kDouble;
      auto casted_exp = (dtype == kComplexDouble) ? Scalar(exp.toComplexDouble()) : Scalar(exp.toDouble());
      return pow(base.to(dtype), casted_exp);
        */
}


#[deprecated = "figure out a better way to handle the overloads"]
pub fn float_power_b(
        base: &Scalar,
        exp:  &Tensor) -> Tensor {
    
    todo!();
        /*
            auto dtype = (isComplexType(exp.scalar_type()) || base.isComplex()) ? kComplexDouble : kDouble;
      auto casted_base = (dtype == kComplexDouble) ? Scalar(base.toComplexDouble()) : Scalar(base.toDouble());
      return pow(casted_base, exp.to(dtype));
        */
}


#[deprecated = "figure out a better way to handle the overloads"]
pub fn float_power_c(
        base: &Tensor,
        exp:  &Tensor) -> Tensor {
    
    todo!();
        /*
            auto dtype = (isComplexType(base.scalar_type()) || isComplexType(exp.scalar_type())) ? kComplexDouble : kDouble;
      return pow(base.to(dtype), exp.to(dtype));
        */
}


#[deprecated = "figure out a better way to handle the overloads"]
pub fn float_power_d(
        base: &mut Tensor,
        exp:  &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto dtype = (isComplexType(base.scalar_type()) || isComplexType(exp.scalar_type())) ? kComplexDouble : kDouble;
      TORCH_CHECK(base.scalar_type() == dtype,
                  "the base given to float_power_ has dtype ", base.scalar_type(),
                  " but the operation's result requires dtype ", dtype);

      return base.pow_(exp.to(dtype));
        */
}

#[deprecated = "figure out a better way to handle the overloads"]
pub fn float_power_e(
        base: &mut Tensor,
        exp:  &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            auto dtype = (isComplexType(base.scalar_type()) || exp.isComplex()) ? kComplexDouble : kDouble;
      TORCH_CHECK(base.scalar_type() == dtype,
                  "the base given to float_power_ has dtype ", base.scalar_type(),
                  " but the operation's result requires dtype ", dtype);

      auto casted_exp = (dtype == kComplexDouble) ? Scalar(exp.toComplexDouble()) : Scalar(exp.toDouble());
      return base.pow_(casted_exp);
        */
}
