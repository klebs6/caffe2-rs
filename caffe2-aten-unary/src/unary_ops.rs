crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/UnaryOps.h]

pub type UnaryFn           = fn(_0: &mut TensorIteratorBase) -> ();
pub type UnaryFnWithScalar = fn(_0: &mut TensorIteratorBase, a: &Scalar) -> ();

declare_dispatch!{unary_fn, abs_stub}
declare_dispatch!{unary_fn, angle_stub}
declare_dispatch!{unary_fn, real_stub}
declare_dispatch!{unary_fn, imag_stub}
declare_dispatch!{unary_fn, conj_physical_stub}
declare_dispatch!{unary_fn, acos_stub}
declare_dispatch!{unary_fn, acosh_stub}
declare_dispatch!{unary_fn, asinh_stub}
declare_dispatch!{unary_fn, atanh_stub}
declare_dispatch!{unary_fn, asin_stub}
declare_dispatch!{unary_fn, atan_stub}
declare_dispatch!{unary_fn, bitwise_not_stub}
declare_dispatch!{unary_fn, logical_not_stub}
declare_dispatch!{unary_fn, ceil_stub}
declare_dispatch!{unary_fn, cos_stub}
declare_dispatch!{unary_fn, cosh_stub}
declare_dispatch!{unary_fn, digamma_stub}
declare_dispatch!{unary_fn, special_entr_stub}
declare_dispatch!{unary_fn, erf_stub}
declare_dispatch!{unary_fn, erfc_stub}
declare_dispatch!{unary_fn, erfinv_stub}
declare_dispatch!{unary_fn, exp_stub}
declare_dispatch!{unary_fn, exp2_stub}
declare_dispatch!{unary_fn, expm1_stub}
declare_dispatch!{unary_fn, floor_stub}
declare_dispatch!{unary_fn, frac_stub}
declare_dispatch!{unary_fn, frexp_stub}
declare_dispatch!{unary_fn, i0_stub}
declare_dispatch!{unary_fn, special_i0e_stub}
declare_dispatch!{unary_fn, special_i1_stub}
declare_dispatch!{unary_fn, special_i1e_stub}
declare_dispatch!{unary_fn, log_stub}
declare_dispatch!{unary_fn, log10_stub}
declare_dispatch!{unary_fn, log1p_stub}
declare_dispatch!{unary_fn, log2_stub}
declare_dispatch!{unary_fn, neg_stub}

declare_dispatch!{unary_fn, reciprocal_stub}
declare_dispatch!{unary_fn, round_stub}
declare_dispatch!{unary_fn, rsqrt_stub}
declare_dispatch!{unary_fn, sigmoid_stub}
declare_dispatch!{unary_fn_with_scalar, logit_stub}
declare_dispatch!{unary_fn, sign_stub}
declare_dispatch!{unary_fn, signbit_stub}
declare_dispatch!{unary_fn, sgn_stub}
declare_dispatch!{unary_fn, sin_stub}
declare_dispatch!{unary_fn, sinc_stub}
declare_dispatch!{unary_fn, sinh_stub}
declare_dispatch!{unary_fn, sqrt_stub}
declare_dispatch!{unary_fn, tan_stub}
declare_dispatch!{unary_fn, tanh_stub}
declare_dispatch!{unary_fn, trigamma_stub}
declare_dispatch!{unary_fn, trunc_stub}
declare_dispatch!{unary_fn, lgamma_stub}

// NB: these are actually defined in Distribution
declare_dispatch!{
    fn(
        _0: &mut Tensor,
        _1: &Tensor,
        _2: Option<Generator>
    ) -> (),
    bernoulli_tensor_stub
}

declare_dispatch!{
    fn(
        _0: &mut Tensor,
        _1: f64,
        _2: Option<Generator>
    ) -> (), 
    bernoulli_scalar_stub
}

declare_dispatch!{
    fn(
        _0: &mut TensorIteratorBase,
        _1: f64,
        _2: f64,
        _3: Option<Generator>
    ) -> (),
    cauchy_stub
}

declare_dispatch!{
    fn(
        _0: &mut TensorIteratorBase,
        _1: f64,
        _2: Option<Generator>
    ) -> (),
    exponential_stub
}

declare_dispatch!{
    fn(
        _0: &mut TensorIteratorBase,
        _1: f64,
        _2: Option<Generator>
    ) -> (),
    geometric_stub
}

declare_dispatch!{
    fn(
        _0: &mut TensorIteratorBase,
        _1: f64,
        _2: f64,
        _3: Option<Generator>
    ) -> (),
    log_normal_stub
}

declare_dispatch!{
    fn(
        _0: &mut TensorIteratorBase,
        _1: f64,
        _2: f64,
        _3: Option<Generator>
    ) -> (),
    uniform_stub
}

declare_dispatch!{
    fn(
        _0: &mut Tensor,
        _1: f64,
        _2: f64,
        _3: Option<Generator>
    ) -> (),
    normal_stub
}

declare_dispatch!{
    fn(
        _0: &mut TensorIteratorBase,
        _1: u64,
        _2: i64,
        _3: Option<Generator>
    ) -> (),
    random_from_to_stub
}

declare_dispatch!{
    fn(
        _0: &mut TensorIteratorBase, 
        _1: Option<Generator>
    ) -> (),
    random_full_64_bits_range_stub
}

declare_dispatch!{
    fn(
        _0: &mut TensorIteratorBase, 
        _1: Option<Generator>
    ) -> (),
    random_stub
}

declare_dispatch!{
    fn(
        _0: &mut TensorIteratorBase,
        _1: i64,
        _2: f64
    ) -> (),
    kaiser_window_stub
}

declare_dispatch!{
    fn(
        _0: &mut TensorIteratorBase, 
        _1: i64
    ) -> (),
    polygamma_stub
}

declare_dispatch!{
    fn(
        _0: &mut TensorIteratorBase,
        a:  &Scalar,
        b:  &Scalar
    ) -> (),
    clamp_stub
}

declare_dispatch!{
    fn(
        _0: &mut Tensor,
        _1: &Tensor,
        _2: i64,
        _3: Option<Generator>
    ) -> c_void,
    multinomial_with_replacement_stub
}

declare_dispatch!{
    fn(
        _0: &mut TensorIteratorBase,
        _1: Option<f64>,
        _2: Option<f64>,
        _3: Option<f64>
    ) -> c_void,
    nan_to_num_stub
}

// Missing unary functions
// digamma
// lgamma
// erfinv
// clone
// contiguous
// zero
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/UnaryOps.cpp]

/**
  | Unary float operations always produce floating
  | point outputs for floating point and integral
  | types
  |
  | For complex inputs, the output type should be
  | the same as input type.
  |
  */
#[macro_export] macro_rules! create_unary_float_meta_func {
    ($func:ident) => {
        /*
        
          TORCH_META_FUNC(func) (const Tensor& self) {        
            build_unary_float_op(maybe_get_output(), self);   
          }
        */
    }
}

create_unary_float_meta_func!{acos}
create_unary_float_meta_func!{acosh}
create_unary_float_meta_func!{asin}
create_unary_float_meta_func!{asinh}
create_unary_float_meta_func!{atan}
create_unary_float_meta_func!{atanh}
create_unary_float_meta_func!{cos}
create_unary_float_meta_func!{cosh}
create_unary_float_meta_func!{digamma}
create_unary_float_meta_func!{erf}
create_unary_float_meta_func!{erfc}
create_unary_float_meta_func!{erfinv}
create_unary_float_meta_func!{exp}
create_unary_float_meta_func!{exp2}
create_unary_float_meta_func!{expm1}
create_unary_float_meta_func!{i0}
create_unary_float_meta_func!{lgamma}
create_unary_float_meta_func!{log}
create_unary_float_meta_func!{log10}
create_unary_float_meta_func!{log1p}
create_unary_float_meta_func!{log2}
create_unary_float_meta_func!{reciprocal}
create_unary_float_meta_func!{rsqrt}
create_unary_float_meta_func!{sigmoid}
create_unary_float_meta_func!{sin}
create_unary_float_meta_func!{sinc}
create_unary_float_meta_func!{sinh}
create_unary_float_meta_func!{special_entr}
create_unary_float_meta_func!{special_i0e}
create_unary_float_meta_func!{special_i1}
create_unary_float_meta_func!{special_i1e}
create_unary_float_meta_func!{sqrt}
create_unary_float_meta_func!{tan}
create_unary_float_meta_func!{tanh}

lazy_static!{
    /*
    TORCH_META_FUNC(polygamma)(i64 n, const Tensor& self) {
      TORCH_CHECK(n >= 0, "polygamma(n, x) does not support negative n.");
      build_unary_float_op(maybe_get_output(), self);
    }
    */
}

/**
  | These are normal unary ops that preserve
  | dtype
  |
  */
#[macro_export] macro_rules! create_unary_meta_func {
    ($func:ident) => {
        /*
        
          TORCH_META_FUNC(func) (const Tensor& self) {        
            build_unary_op(maybe_get_output(), self);   
          }
        */
    }
}

create_unary_meta_func!{bitwise_not}
create_unary_meta_func!{frac}
create_unary_meta_func!{round}
create_unary_meta_func!{sgn}

lazy_static!{
    /*
    TORCH_META_FUNC(neg)(const Tensor& self) {
      TORCH_CHECK(self.scalar_type() != kBool,
                  "Negation, the `-` operator, on a bool tensor is not supported. "
                  "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
      build_unary_op(maybe_get_output(), self);
    }

    TORCH_META_FUNC(trunc) (const Tensor& self) {
      // Note: this is consistent with NumPy
      TORCH_CHECK(!self.is_complex(),
        "trunc is not supported for complex inputs");
      build_unary_op(maybe_get_output(), self);
    }

    TORCH_META_FUNC(floor) (const Tensor& self) {
      // Note: this is consistent with NumPy
      TORCH_CHECK(!self.is_complex(),
        "floor is not supported for complex inputs");
      build_unary_op(maybe_get_output(), self);
    }

    TORCH_META_FUNC(sign) (const Tensor& self) {
      TORCH_CHECK(!self.is_complex(),
                  "Unlike NumPy, torch.sign is not intended to support complex numbers. Please use torch.sgn instead.");
      build_unary_op(maybe_get_output(), self);
    }

    TORCH_META_FUNC(ceil) (const Tensor& self) {
      // Note: this is consistent with NumPy
      TORCH_CHECK(!self.is_complex(),
        "ceil is not supported for complex inputs");
      build_unary_op(maybe_get_output(), self);
    }
    */
}

/**
  | NOTE: These are helper functions that reduce
  | redundant code in implementing the most typical
  | kind of unary operators.
  |
  | YOU ARE NOT OBLIGED TO USE THESE HELPERS---if
  | you're writing something more specialized,
  | please don't try to make them work for your
  | case, but just write something new
  | instead. Here we use helper functions instead
  | of a flat fat macro that implements everything,
  | because the former allows some simple
  | preprocessing that are unique to some operators
  | (more is foreseeable) and is more flexible and
  | elegant than the latter.
  |
  */
#[macro_export] macro_rules! create_unary_torch_impl_func {
    ($func_out:ident, $func_stub:ident) => {
        /*
        
        TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& result) {  
          func_stub(device_type(), *this);                                      
        }
        */
    }
}

create_unary_torch_impl_func!{acos_out, acos_stub}
create_unary_torch_impl_func!{acosh_out, acosh_stub}
create_unary_torch_impl_func!{asin_out, asin_stub}
create_unary_torch_impl_func!{asinh_out, asinh_stub}
create_unary_torch_impl_func!{atan_out, atan_stub}
create_unary_torch_impl_func!{atanh_out, atanh_stub}
create_unary_torch_impl_func!{bitwise_not_out, bitwise_not_stub}
create_unary_torch_impl_func!{ceil_out, ceil_stub}
create_unary_torch_impl_func!{cos_out, cos_stub}
create_unary_torch_impl_func!{cosh_out, cosh_stub}
create_unary_torch_impl_func!{digamma_out, digamma_stub}
create_unary_torch_impl_func!{erf_out, erf_stub}
create_unary_torch_impl_func!{erfc_out, erfc_stub}
create_unary_torch_impl_func!{erfinv_out, erfinv_stub}
create_unary_torch_impl_func!{exp_out, exp_stub}
create_unary_torch_impl_func!{exp2_out, exp2_stub}
create_unary_torch_impl_func!{expm1_out, expm1_stub}
create_unary_torch_impl_func!{floor_out, floor_stub}
create_unary_torch_impl_func!{frac_out, frac_stub}
create_unary_torch_impl_func!{i0_out, i0_stub}
create_unary_torch_impl_func!{lgamma_out, lgamma_stub}
create_unary_torch_impl_func!{log_out, log_stub}
create_unary_torch_impl_func!{log10_out, log10_stub}
create_unary_torch_impl_func!{log1p_out, log1p_stub}
create_unary_torch_impl_func!{log2_out, log2_stub}
create_unary_torch_impl_func!{neg_out, neg_stub}
create_unary_torch_impl_func!{reciprocal_out, reciprocal_stub}
create_unary_torch_impl_func!{round_out, round_stub}
create_unary_torch_impl_func!{rsqrt_out, rsqrt_stub}
create_unary_torch_impl_func!{sigmoid_out, sigmoid_stub}
create_unary_torch_impl_func!{sign_out, sign_stub}
create_unary_torch_impl_func!{sin_out, sin_stub}
create_unary_torch_impl_func!{sinc_out, sinc_stub}
create_unary_torch_impl_func!{sinh_out, sinh_stub}
create_unary_torch_impl_func!{special_entr_out, special_entr_stub}
create_unary_torch_impl_func!{special_i0e_out, special_i0e_stub}
create_unary_torch_impl_func!{special_i1e_out, special_i1e_stub}
create_unary_torch_impl_func!{special_i1_out, special_i1_stub}
create_unary_torch_impl_func!{sqrt_out, sqrt_stub}
create_unary_torch_impl_func!{tan_out, tan_stub}
create_unary_torch_impl_func!{tanh_out, tanh_stub}
create_unary_torch_impl_func!{trunc_out, trunc_stub}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(polygamma_out)
    (i64 n, const Tensor& self, const Tensor& result) {
      polygamma_stub(device_type(), *this, n);
    }
    */
}

/**
  | since polygamma_ has different signature
  | from its out and functional variant,
  | we explicitly define it (instead of
  | using structured kernel).
  |
  */
pub fn polygamma<'a>(
        self_: &mut Tensor,
        n:     i64) -> &'a mut Tensor {
    
    todo!();
        /*
            return polygamma_out(self, n, self);
        */
}

#[inline] pub fn unary_op_impl_out<'a,Stub>(
        result: &mut Tensor,
        self_:  &Tensor,
        stub:   &mut Stub) -> &'a mut Tensor {

    todo!();
        /*
            auto iter = TensorIterator::unary_op(result, self);
      stub(iter.device_type(), iter);
      return result;
        */
}

#[inline] pub fn unary_op_impl_float_out<'a,Stub, Args>(
        result: &mut Tensor,
        self_:  &Tensor,
        stub:   &mut Stub,
        args:   Args) -> &'a mut Tensor {

    todo!();
        /*
            auto iter = TensorIterator::unary_float_op(result, self);
      stub(iter.device_type(), iter, args...);
      iter.cast_outputs();
      return result;
        */
}

#[inline] pub fn unary_op_impl_float<Stub, Args>(
        self_: &Tensor,
        stub:  &mut Stub,
        args:  Args) -> Tensor {

    todo!();
        /*
            Tensor result;
      auto iter = TensorIterator::unary_float_op(result, self);
      stub(iter.device_type(), iter, args...);
      return iter.output();
        */
}

/**
  | An alternate version of unary_op_impl_out that
  | follows the same pattern for non-complex
  | inputs, but returns a floating point tensor for
  | complex inputs by default.
  |
  | Note: This is done by running the operation as
  | usual and then copying the operation's result
  | to the expected result type.
  |
  */
#[inline] pub fn unary_op_impl_with_complex_to_float_out<'a,Stub>(
    result:                    &mut Tensor,
    self_:                     &Tensor,
    stub:                      &mut Stub,
    promotes_integer_to_float: bool) -> &'a mut Tensor {

    todo!();
        /*
            if (self.is_complex() && !result.is_complex()) {
          // Checks if the corresponding float type can be cast to the desired dtype
          const auto float_type = toValueType(self.scalar_type());
          TORCH_CHECK(canCast(float_type, result.scalar_type()),
                "result type ", float_type, " can't be cast to the desired output type ",
                result.scalar_type());

          // Runs the function complex->complex, as TensorIterator expects
          Tensor complex_result = empty({0}, self.options());
          auto iter = TensorIterator::unary_op(complex_result, self);
          stub(iter.device_type(), iter);

          // Copies the complex result to the actual result and returns it
          native::resize_output(result, complex_result.sizes());
          result.copy_(real(complex_result));
          return result;
        }

        if (promotes_integer_to_float) {
          return unary_op_impl_float_out(result, self, stub);
        }

        return unary_op_impl_out(result, self, stub);
        */
}

/**
  | out_impl passed into unary_op_impl and
  | unary_op_impl_  must go through  device
  | dispatch otherwise it won't dispatch to
  | out-of-source devices like XLA.
  |
  | For example it must be bitwise_not_out instead
  | of bitwise_not_out(which is native!).
  |
  */
#[inline] pub fn unary_op_impl<OutImpl>(
    self_:    &Tensor,
    out_impl: &mut OutImpl) -> Tensor {

    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return out_impl(result, self);
        */
}

/**
  | An alternate version of unary_op_impl that
  | follows the same pattern for non-complex
  | inputs, but returns a floating point tensor for
  | complex inputs by default.
  |
  */
#[inline] pub fn unary_op_impl_with_complex_to_float<OutImpl>(
        self_:    &Tensor,
        out_impl: &mut OutImpl) -> Tensor {

    todo!();
        /*
            if (self.is_complex()) {
        const auto float_type = toValueType(self.scalar_type());
        Tensor result = empty({0}, self.options().dtype(float_type));
        return out_impl(result, self);
      }

      Tensor result = empty({0}, self.options());
      return out_impl(result, self);
        */
}

#[inline] pub fn unary_op_impl_mut<'a,OutImpl>(
    self_:    &mut Tensor,
    out_impl: &mut OutImpl) -> &'a mut Tensor {

    todo!();
        /*
            return out_impl(self, self);
        */
}

/// arccos, alias for acos
pub fn arccos_out<'a>(
    self_:  &Tensor,
    result: &mut Tensor) -> &'a mut Tensor {

    todo!();
        /*
            return acos_out(result, self);
        */
}

pub fn arccos_a<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.acos();
        */
}

pub fn arccos_b<'a>(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return self.acos_();
        */
}

pub fn rad2deg_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!self.is_complex(), "rad2deg is not supported for complex tensors.");
      constexpr double M_180_PI = 57.295779513082320876798154814105170332405472466564;
      return mul_out(result, self, wrapped_scalar_tensor(Scalar(M_180_PI)));
        */
}

pub fn rad2deg_a<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            // Note: int-> float promotion handled differently from other Unary ops,
      // as it does not use the usual TensorIterator + Kernel Dispatch pattern.
      auto options = self.options();
      if (isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
        options = options.dtype(get_default_dtype());
      }
      auto result = empty_like(self, options);
      rad2deg_out(result, self);
      return result;
        */
}

pub fn rad2deg_b<'a>(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return unary_op_impl_(self, rad2deg_out);
        */
}

pub fn deg2rad_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!self.is_complex(), "deg2rad is not supported for complex tensors.");
      constexpr double M_PI_180 = 0.017453292519943295769236907684886127134428718885417;
      return mul_out(result, self, wrapped_scalar_tensor(Scalar(M_PI_180)));
        */
}

pub fn deg2rad_a<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            // Note: int-> float promotion handled differently from other Unary ops,
      // as it does not use the usual TensorIterator + Kernel Dispatch pattern.
      auto options = self.options();
      if (isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
        options = options.dtype(get_default_dtype());
      }
      auto result = empty_like(self, options);
      deg2rad_out(result, self);
      return result;
        */
}

pub fn deg2rad_b<'a>(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return unary_op_impl_(self, deg2rad_out);
        */
}

// arcsin, alias of asin
pub fn arcsin_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return asin_out(result, self);
        */
}

pub fn arcsin<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.asin();
        */
}

pub fn arcsin_mut<'a>(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return self.asin_();
        */
}

/**
  | arctan, alias of atan
  |
  */
pub fn arctan_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return atan_out(result, self);
        */
}

pub fn arctan<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.atan();
        */
}

pub fn arctan_mut<'a>(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return self.atan_();
        */
}

/**
  | Note [Complex abs and angle]
  |
  | Complex inputs to abs and angle return float
  | results by default.
  |
  | abs and angle, in both NumPy and C++, returns
  | a float result when given a complex input. This
  | makes sense mathematically since the absolute
  | value and angle of a complex number has no
  | imaginary part.
  |
  */
pub fn abs_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return unary_op_impl_with_complex_to_float_out(result, self, abs_stub, /*promotes_integer_to_float=*/false);
        */
}

pub fn abs<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return unary_op_impl_with_complex_to_float(self, abs_out);
        */
}


pub fn abs_mut<'a>(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!self.is_complex(), "In-place abs is not supported for complex tensors.");
      return unary_op_impl_(self, abs_out);
        */
}

/**
  | Absolute, alias for abs
  |
  */
pub fn absolute_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return abs_out(result, self);
        */
}

pub fn absolute<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.abs();
        */
}

pub fn absolute_mut<'a>(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return self.abs_();
        */
}


pub fn angle_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return unary_op_impl_with_complex_to_float_out(result, self, angle_stub, /*promotes_integer_to_float=*/true);
        */
}


pub fn angle(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            if (self.is_complex()) {
        const auto float_type = toValueType(self.scalar_type());
        Tensor result = empty({0}, self.options().dtype(float_type));
        return angle_out(result, self);
      }

      return unary_op_impl_float(self, angle_stub);
        */
}


pub fn real(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            if (self.is_complex()) {
        // real is never affected by conjugate bit, safe to use physical version
        auto real_tensor = _view_as_real_physical(self);
        return select(real_tensor, real_tensor.dim() - 1, 0);
      } else {
        TORCH_CHECK(false, "real is not implemented for tensors with non-complex dtypes.");
      }
        */
}


pub fn imag(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            if (self.is_complex()) {
        auto real_tensor = view_as_real(self);
        return select(real_tensor, real_tensor.dim() - 1, 1);
      } else {
        TORCH_CHECK(false, "imag is not implemented for tensors with non-complex dtypes.");
      }
        */
}


pub fn conj_physical_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return unary_op_impl_out(result, self, conj_physical_stub);
        */
}


pub fn conj_physical_a(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            if (self.is_conj()) {
        return self.conj().clone();
      }
      return unary_op_impl(self, conj_physical_out);
        */
}


pub fn conj_physical_b<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            if (!self.is_complex()) return self;
      return _conj_physical(self);
        */
}

pub fn conj_physical_c<'a>(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            if (!self.is_complex()) return self;
      return unary_op_impl_out(self, self, conj_physical_stub);
        */
}


pub fn resolve_conj(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            if (!self.is_conj()) { return self; }
      // conjugation is handled in `copy_()` that clone ultimately calls into
      return self.clone(self.suggest_memory_format());
        */
}


pub fn conj(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor self_ = self.alias();
      self_._set_conj(!self.is_conj());
      namedinference::propagate_names(self_, self);
      return self_;
        */

    todo!();
        /*
            // This might look like an infinite recursion but it's not.
      // This actually calls into `conj()` defined in the Tensor class.
      return self.conj();
        */
}

/**
  | special_exp2, alias for exp2
  |
  */
pub fn special_exp2_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return exp2_out(result, self);
        */
}

pub fn special_exp2(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.exp2();
        */
}

/**
  | special_expm1, alias for expm1
  |
  */
pub fn special_expm1_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return expm1_out(result, self);
        */
}


pub fn special_expm1(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.expm1();
        */
}

/**
  | special_erf, alias for erf
  |
  */
pub fn special_erf_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return erf_out(result, self);
        */
}


pub fn special_erf(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.erf();
        */
}

/**
  | special_erfc, alias for erfc
  |
  */
pub fn special_erfc_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return erfc_out(result, self);
        */
}


pub fn special_erfc(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.erfc();
        */
}

/**
  | special_erfinv, alias for erfinv
  |
  */
pub fn special_erfinv_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return erfinv_out(result, self);
        */
}


pub fn special_erfinv(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.erfinv();
        */
}

/**
  | special_psi, alias for digamma
  |
  */
pub fn special_psi_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return digamma_out(result, self);
        */
}


pub fn special_psi(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.digamma();
        */
}

/**
  | special_digamma, alias for digamma
  |
  */
pub fn special_digamma_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return digamma_out(result, self);
        */
}

pub fn special_digamma(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.digamma();
        */
}

/**
  | special_i0, alias for i0
  |
  */
pub fn special_i0_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return i0_out(result, self);
        */
}

pub fn special_i0(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.i0();
        */
}


#[inline] pub fn calc_ndtr(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            auto x_sqrt_2 = self / sqrt(2.);
      return (1 + erf(x_sqrt_2)) * 0.5;
        */
}

/**
  | special_ndtr
  |
  */
pub fn special_ndtr_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          self.device() == result.device(),
          "Expected all tensors to be on the same device, but found at least two devices, ",
          self.device(),
          " and ",
          result.device(),
          "!");

      auto ndtr = calc_ndtr(self);
      TORCH_CHECK(
          can_cast(ndtr.scalar_type(), result.scalar_type()),
          "result type ",
          ndtr.scalar_type(),
          " can't be cast to the desired output type ",
          result.scalar_type());

      native::resize_output(result, ndtr.sizes());
      return result.copy_(ndtr);
        */
}


pub fn special_ndtr(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return calc_ndtr(self);
        */
}

// FIXME: remove const_cast once unary_op_impl_out is updated
lazy_static!{
    /*
    TORCH_IMPL_FUNC(sgn_out) (const Tensor& self, const Tensor& result) {
      if (self.is_complex()) {
        sgn_stub(device_type(), *this);
      } else {
        sign_stub(device_type(), *this);
      }
    }
    */
}

/**
  | arccosh, alias for acosh
  |
  */

pub fn arccosh_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return acosh_out(result, self);
        */
}


pub fn arccosh<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return acosh(self);
        */
}

pub fn arccosh_mut<'a>(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return acosh_(self);
        */
}

/**
  | arcsinh, alias for asinh
  |
  */
pub fn arcsinh_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return asinh_out(result, self);
        */
}

pub fn arcsinh<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.asinh();
        */
}

pub fn arcsinh_mut<'a>(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return self.asinh_();
        */
}

/**
  | arctanh, alias for atanh
  |
  */
pub fn arctanh_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return atanh_out(result, self);
        */
}

pub fn arctanh<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.atanh();
        */
}

pub fn arctanh_mut<'a>(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return self.atanh_();
        */
}


pub fn square_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return pow_out(result, self, 2);
        */
}


pub fn square<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return pow(self, 2);
        */
}


pub fn square_mut<'a>(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return self.pow_(2);
        */
}


pub fn logit_out<'a>(
        self_:  &Tensor,
        eps:    Option<f64>,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return unary_op_impl_float_out(
          result, self, logit_stub, Scalar(eps ? eps.value() : -1.0));
        */
}


pub fn logit(
        self_: &Tensor,
        eps:   Option<f64>) -> Tensor {
    
    todo!();
        /*
            return unary_op_impl_float(
          self, logit_stub, Scalar(eps ? eps.value() : -1.0));
        */
}


pub fn logit_mut<'a>(
        self_: &mut Tensor,
        eps:   Option<f64>) -> &'a mut Tensor {
    
    todo!();
        /*
            return logit_out(self, self, eps);
        */
}


pub fn special_logit_out<'a>(
        self_:  &Tensor,
        eps:    Option<f64>,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return logit_out(result, self, eps);
        */
}


pub fn special_logit(
        self_: &Tensor,
        eps:   Option<f64>) -> Tensor {
    
    todo!();
        /*
            return self.logit(eps);
        */
}

/**
  | special_expit, alias for sigmoid
  |
  */
pub fn special_expit_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return sigmoid_out(result, self);
        */
}

pub fn special_expit(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.sigmoid();
        */
}

pub fn nan_to_num_out<'a>(
        self_:   &Tensor,
        nan:     Option<f64>,
        pos_inf: Option<f64>,
        neg_inf: Option<f64>,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          self.scalar_type() == result.scalar_type(),
          "nan_to_num: dtype of out: ",
          result.scalar_type(),
          " should be same as input: ",
          self.scalar_type());

      if (isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
        native::resize_output(result, self.sizes());
        result.copy_(self);
        return result;
      }

      auto iter = TensorIterator::unary_op(result, self);
      nan_to_num_stub(iter.device_type(), iter, nan, pos_inf, neg_inf);
      return result;
        */
}

pub fn nan_to_num(
        self_:   &Tensor,
        nan:     Option<f64>,
        pos_inf: Option<f64>,
        neg_inf: Option<f64>) -> Tensor {
    
    todo!();
        /*
            auto result = empty_like(self);
      return nan_to_num_out(result, self, nan, pos_inf, neg_inf);
        */
}

pub fn nan_to_num_mut<'a>(
    self_:   &mut Tensor,
    nan:     Option<f64>,
    pos_inf: Option<f64>,
    neg_inf: Option<f64>) -> &'a mut Tensor {
    
    todo!();
        /*
            return nan_to_num_out(self, self, nan, pos_inf, neg_inf);
        */
}

/**
  | Alias for trunc
  |
  */
pub fn fix_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return trunc_out(result, self);
        */
}

pub fn fix<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.trunc();
        */
}

pub fn fix_mut<'a>(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return self.trunc_();
        */
}

pub fn positive(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.scalar_type() != kBool, "The `+` operator, on a bool tensor is not supported.");
      return self;
        */
}

pub fn negative_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return neg_out(result, self);
        */
}

pub fn negative<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.neg();
        */
}

pub fn negative_mut<'a>(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return self.neg_();
        */
}

pub fn logical_not<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options().dtype(kBool));
      return logical_not_out(result, self);
        */
}

pub fn logical_not_mut<'a>(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return logical_not_out(self, self);
        */
}

pub fn logical_not_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(result)
        .add_input(self)
        .build();
      logical_not_stub(iter.device_type(), iter);
      return result;
        */
}


pub fn signbit_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!self.is_complex(), "signbit is not implemented for complex tensors.");
      TORCH_CHECK(result.scalar_type() == kBool, "signbit does not support non-boolean outputs.");
      native::resize_output(result, self.sizes());

      if (self.dtype() == kBool) {
        return result.fill_(false);
      } else {
        TensorIterator iter = TensorIteratorConfig()
          .check_all_same_dtype(false)
          .add_output(result)
          .add_input(self)
          .build();
        signbit_stub(iter.device_type(), iter);
      }
      return result;
        */
}


pub fn signbit(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options().dtype(kBool));
      return signbit_out(result, self);
        */
}

pub const HALF: f64 = 0.5;
pub const QUARTER: f64 = 0.25;

#[inline] pub fn mvlgamma_check(
        self_: &Tensor,
        p:     i64)  {
    
    todo!();
        /*
            TORCH_CHECK((self > HALF * (p - 1)).all().item<bool>(),
                  "All elements must be greater than (p-1)/2");
      TORCH_CHECK(p >= 1, "p has to be greater than or equal to 1");
        */
}



pub fn mvlgamma_a(
        self_: &Tensor,
        p:     i64) -> Tensor {
    
    todo!();
        /*
            mvlgamma_check(self, p);
      auto dtype = scalarTypeToTypeMeta(self.scalar_type());
      if (isIntegralType(self.scalar_type(), /*include_bool=*/true)) {
        // int -> float promotion
        dtype = get_default_dtype();
      }
      Tensor args = native::arange(
          -p * HALF + HALF,
          HALF,
          HALF,
          optTypeMetaToScalarType(dtype),
          self.options().layout_opt(),
          self.options().device_opt(),
          self.options().pinned_memory_opt());
      args = args.add(self.unsqueeze(-1));
      const auto p2_sub_p = static_cast<double>(p * (p - 1));
      return args.lgamma_().sum(-1).add_(p2_sub_p * log(pi<double>) * QUARTER);
        */
}

pub fn mvlgamma_b<'a>(
    self_: &mut Tensor,
    p:     i64) -> &'a mut Tensor {
    
    todo!();
        /*
            mvlgamma_check(self, p);
      Tensor args = native::arange(
          -p *HALF  + HALF,
          HALF,
          HALF,
          optTypeMetaToScalarType(self.options().dtype_opt()),
          self.options().layout_opt(),
          self.options().device_opt(),
          self.options().pinned_memory_opt());
      args = args.add(self.unsqueeze(-1));
      const auto p2_sub_p = static_cast<double>(p * (p - 1));
      return self.copy_(args.lgamma_().sum(-1).add_(p2_sub_p * log(pi<double>) * QUARTER));
        */
}


pub fn frexp(self_: &Tensor) -> (Tensor,Tensor) {
    
    todo!();
        /*
            Tensor mantissa = empty_like(self);
      Tensor exponent = empty_like(self, self.options().dtype(kInt));

      frexp_out(mantissa, exponent, self);
      return tuple<Tensor, Tensor>(mantissa, exponent);
        */
}


pub fn frexp_out<'a>(
        self_:    &Tensor,
        mantissa: &mut Tensor,
        exponent: &mut Tensor) -> (&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
            // torch.frexp is implemented for floating-point dtypes for now,
      // should add support for integral dtypes in the future.
      TORCH_CHECK(isFloatingType(self.scalar_type()),
                  "torch.frexp() only supports floating-point dtypes");

      TORCH_CHECK(mantissa.dtype() == self.dtype(),
                  "torch.frexp() expects mantissa to have dtype ", self.dtype(),
                  " but got ", mantissa.dtype());
      TORCH_CHECK(exponent.dtype() == kInt,
                  "torch.frexp() expects exponent to have int dtype "
                  "but got ", exponent.dtype());

      auto iter = TensorIteratorConfig()
        .add_output(mantissa)
        .add_output(exponent)
        .add_input(self)
        .check_all_same_dtype(false)
        .set_check_mem_overlap(true)
        .build();
      frexp_stub(iter.device_type(), iter);

      return tuple<Tensor&, Tensor&>(mantissa, exponent);
        */
}

/**
  | alias for lgamma, implements special.gammanln
  | equivalent to scipy.special.gammaln
  |
  */
pub fn special_gammaln(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.lgamma();
        */
}

pub fn special_gammaln_out<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return lgamma_out(result, self);
        */
}

define_dispatch!{abs_stub}
define_dispatch!{angle_stub}
define_dispatch!{real_stub}
define_dispatch!{imag_stub}
define_dispatch!{conj_physical_stub}
define_dispatch!{acos_stub}
define_dispatch!{acosh_stub}
define_dispatch!{asinh_stub}
define_dispatch!{atanh_stub}
define_dispatch!{asin_stub}
define_dispatch!{atan_stub}
define_dispatch!{bitwise_not_stub}
define_dispatch!{ceil_stub}
define_dispatch!{cos_stub}
define_dispatch!{cosh_stub}
define_dispatch!{digamma_stub}
define_dispatch!{special_entr_stub}
define_dispatch!{erf_stub}
define_dispatch!{erfc_stub}
define_dispatch!{erfinv_stub}
define_dispatch!{exp_stub}
define_dispatch!{exp2_stub}
define_dispatch!{expm1_stub}
define_dispatch!{floor_stub}
define_dispatch!{frac_stub}
define_dispatch!{frexp_stub}
define_dispatch!{i0_stub}
define_dispatch!{special_i0e_stub}
define_dispatch!{special_i1_stub}
define_dispatch!{special_i1e_stub}
define_dispatch!{log_stub}
define_dispatch!{log10_stub}
define_dispatch!{log1p_stub}
define_dispatch!{log2_stub}
define_dispatch!{logical_not_stub}
define_dispatch!{neg_stub}
define_dispatch!{nan_to_num_stub}
define_dispatch!{polygamma_stub}
define_dispatch!{reciprocal_stub}
define_dispatch!{round_stub}
define_dispatch!{rsqrt_stub}
define_dispatch!{sigmoid_stub}
define_dispatch!{logit_stub}
define_dispatch!{sign_stub}
define_dispatch!{signbit_stub}
define_dispatch!{sgn_stub}
define_dispatch!{sin_stub}
define_dispatch!{sinc_stub}
define_dispatch!{sinh_stub}
define_dispatch!{sqrt_stub}
define_dispatch!{tan_stub}
define_dispatch!{tanh_stub}
define_dispatch!{trigamma_stub}
define_dispatch!{trunc_stub}
define_dispatch!{lgamma_stub}
