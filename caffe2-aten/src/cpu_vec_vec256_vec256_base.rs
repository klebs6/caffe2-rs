/*!
  | DO NOT DEFINE STATIC DATA IN THIS HEADER!
  | See Note [Do not compile initializers with AVX]
  |
  | Note [Do not compile initializers with AVX]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ If
  | you define a static initializer in this file,
  | the initialization will use AVX instructions
  | because these object files are compiled with
  | AVX enabled.
  |
  | We need to avoid non-trivial global data in
  | these architecture specific files because
  | there's no way to guard the global initializers
  | with CPU capability detection.
  |
  | See
  | https://github.com/pytorch/pytorch/issues/37577
  | for an instance of this bug in the past.
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/vec256_base.h]

// NOTE: If you specialize on a type, you must define all operations!

// emulates vectorized types
pub struct Vectorized<T> {

    values: Align32<[T; 32 / sizeof(T)]>,

    //using value_type = T;
    //using size_type = int;
}

impl Default for Vectorized {
    
    fn default() -> Self {
        todo!();
        /*


            : values{0}
        */
    }
}

impl Vectorized<T> {

  // Note [constexpr static function to avoid
  // odr-usage compiler bug]
  //
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //
  // Why, you might ask, is size defined to be
  // a static constexpr function, rather than
  // a more ordinary 'static constexpr int size;'
  // variable? The problem lies within ODR rules
  // for static constexpr members versus static
  // constexpr functions.  First, recall that this
  // class (along with all of its derivations)
  // live in an anonymous namespace: they are
  // intended to be *completely* inlined at their
  // use-sites, because we need to compile it
  // multiple times for different instruction
  // sets.
  //
  // Because of this constraint, we CANNOT provide
  // a single definition for any static members in
  // this class; since we want to compile the
  // class multiple times, there wouldn't actually
  // be any good place to put the definition.  Now
  // here is the problem: if we ODR-use a static
  // constexpr member, we are *obligated* to
  // provide a definition.  Without the
  // definition, you get a compile error like:
  //
  //    relocation R_X86_64_PC32 against undefined symbol
  //    `_ZN2at6vec25612_GLOBAL__N_16VectorizedIdE4sizeE' can not be used when making
  //    a shared object; recompile with -fPIC
  //
  // If this were C++17, we could replace a static
  // constexpr variable with an inline variable
  // which doesn't require one definition. But we
  // are not C++17.  So the next best thing is to
  // replace the member with a static constexpr
  // (and therefore inline) function, which does
  // not require ODR either.
  //
  // Also, technically according to the C++
  // standard, we don't have to define a constexpr
  // variable if we never odr-use it.  But it
  // seems that some versions GCC/Clang have buggy
  // determinations on whether or not an
  // identifier is odr-used or not, and in any
  // case it's hard to tell if a variable is
  // odr-used or not.  So best to just cut the
  // problem at the root.
    pub fn size() -> SizeType {
        
        todo!();
        /*
            return 32 / sizeof(T);
        */
    }
    
    pub fn new(val: T) -> Self {
    
        todo!();
        /*


            for (int i = 0; i != size(); i++) {
          values[i] = val;
        }
        */
    }
    
    pub fn new(vals: Args) -> Self {
    
        todo!();
        /*


            : values{vals...}
        */
    }

    // This also implies const T& operator[](int
    // idx) const
    //
    #[inline] pub fn operator_const_tptr(&mut self) -> TPtr {
        
        todo!();
        /*
            return values;
        */
    }

    // This also implies T& operator[](int idx)
    #[inline] pub fn operator_tptr(&mut self) -> TPtr {
        
        todo!();
        /*
            return values;
        */
    }
    
    pub fn blend<const mask_: i64>(
        a: &Vectorized<T>,
        b: &Vectorized<T>) -> Vectorized<T> {
    
        todo!();
        /*
            i64 mask = mask_;
        Vectorized vec;
        for (i64 i = 0; i < size(); i++) {
          if (mask & 0x01) {
            vec[i] = b[i];
          } else {
            vec[i] = a[i];
          }
          mask = mask >> 1;
        }
        return vec;
        */
    }
    
    pub fn blendv(
        a:    &Vectorized<T>,
        b:    &Vectorized<T>,
        mask: &Vectorized<T>) -> Vectorized<T> {
        
        todo!();
        /*
            Vectorized vec;
        int_same_size_t<T> buffer[size()];
        mask.store(buffer);
        for (i64 i = 0; i < size(); i++) {
          if (buffer[i] & 0x01)
           {
            vec[i] = b[i];
          } else {
            vec[i] = a[i];
          }
        }
        return vec;
        */
    }

    /**
      | step sometimes requires a higher precision
      | type (e.g., T=int, step_t=double)
      |
      */
    pub fn arange<step_t>(
        base: T,
        step: Step) -> Vectorized<T> {
        let base: T = base.unwrap_or(0);
        let step: Step = step.unwrap_or(1);
        todo!();
        /*
            Vectorized vec;
        for (i64 i = 0; i < size(); i++) {
          vec.values[i] = base + i * step;
        }
        return vec;
        */
    }
    
    pub fn set(
        a:     &Vectorized<T>,
        b:     &Vectorized<T>,
        count: i64) -> Vectorized<T> {
        
        todo!();
        /*
            Vectorized vec;
        for (i64 i = 0; i < size(); i++) {
          if (i < count) {
            vec[i] = b[i];
          } else {
            vec[i] = a[i];
          }
        }
        return vec;
        */
    }
    
    pub fn loadu(ptr: *const void) -> Vectorized<T> {
        
        todo!();
        /*
            Vectorized vec;
        memcpy(vec.values, ptr, 32);
        return vec;
        */
    }
    
    pub fn loadu(
        ptr:   *const void,
        count: i64) -> Vectorized<T> {
        
        todo!();
        /*
            Vectorized vec;
        memcpy(vec.values, ptr, count * sizeof(T));
        return vec;
        */
    }
    
    pub fn store(&self, 
        ptr:   *mut void,
        count: i32)  {
        
        todo!();
        /*
            memcpy(ptr, values, count * sizeof(T));
        */
    }
    
    pub fn zero_mask(&self) -> i32 {
        
        todo!();
        /*
            // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
        int mask = 0;
        for (int i = 0; i < size(); ++ i) {
          if (values[i] == static_cast<T>(0)) {
            mask |= (1 << i);
          }
        }
        return mask;
        */
    }
    
    pub fn isnan(&self) -> Vectorized<T> {
        
        todo!();
        /*
            Vectorized<T> vec;
        for (i64 i = 0; i != size(); i++) {
          if (_isnan(values[i])) {
            memset(static_cast<void*>(vec.values + i), 0xFF, sizeof(T));
          } else {
            memset(static_cast<void*>(vec.values + i), 0, sizeof(T));
          }
        }
        return vec;
        */
    }
    
    pub fn map(&self, f: fn(_0: T) -> T) -> Vectorized<T> {
        
        todo!();
        /*
            Vectorized<T> ret;
        for (i64 i = 0; i != size(); i++) {
          ret[i] = f(values[i]);
        }
        return ret;
        */
    }
    
    pub fn map(&self, f: fn(_0: &T) -> T) -> Vectorized<T> {
        
        todo!();
        /*
            Vectorized<T> ret;
        for (i64 i = 0; i != size(); i++) {
          ret[i] = f(values[i]);
        }
        return ret;
        */
    }

    //template <typename other_t_abs = T, typename enable_if<!is_floating_point<other_t_abs>::value && !is_complex<other_t_abs>::value, int>::type = 0>
    pub fn abs(&self) -> Vectorized<T> {
        
        todo!();
        /*
            // other_t_abs is for SFINAE and clarity. Make sure it is not changed.
        static_assert(is_same<other_t_abs, T>::value, "other_t_abs must be T");
        return map([](T x) -> T { return x < static_cast<T>(0) ? -x : x; });
        */
    }

    //template <typename float_t_abs = T, typename enable_if<is_floating_point<float_t_abs>::value, int>::type = 0>
    pub fn abs(&self) -> Vectorized<T> {
        
        todo!();
        /*
            // float_t_abs is for SFINAE and clarity. Make sure it is not changed.
        static_assert(is_same<float_t_abs, T>::value, "float_t_abs must be T");
        // Specifically deal with floating-point because the generic code above won't handle -0.0 (which should result in
        // 0.0) properly.
        return map([](T x) -> T { return abs(x); });
        */
    }

    //template <typename complex_t_abs = T, typename enable_if<is_complex<complex_t_abs>::value, int>::type = 0>
    pub fn abs(&self) -> Vectorized<T> {
        
        todo!();
        /*
            // complex_t_abs is for SFINAE and clarity. Make sure it is not changed.
        static_assert(is_same<complex_t_abs, T>::value, "complex_t_abs must be T");
        // Specifically map() does not perform the type conversion needed by abs.
        return map([](T x) { return static_cast<T>(abs(x)); });
        */
    }

    //template <typename other_t_sgn = T, typename enable_if<is_complex<other_t_sgn>::value, int>::type = 0>
    pub fn sgn(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(native::sgn_impl);
        */
    }

    //template <typename other_t_angle = T, typename enable_if<!is_complex<other_t_angle>::value, int>::type = 0>
    pub fn angle(&self) -> Vectorized<T> {
        
        todo!();
        /*
            // other_t_angle is for SFINAE and clarity. Make sure it is not changed.
        static_assert(is_same<other_t_angle, T>::value, "other_t_angle must be T");
        return map(native::angle_impl<T>);  // compiler is unable to resolve the overload without <T>
        */
    }

    //template <typename complex_t_angle = T, typename enable_if<is_complex<complex_t_angle>::value, int>::type = 0>
    pub fn angle(&self) -> Vectorized<T> {
        
        todo!();
        /*
            // complex_t_angle is for SFINAE and clarity. Make sure it is not changed.
        static_assert(is_same<complex_t_angle, T>::value, "complex_t_angle must be T");
        return map([](T x) { return static_cast<T>(arg(x)); });
        */
    }

    //template <typename other_t_real = T, typename enable_if<!is_complex<other_t_real>::value, int>::type = 0>
    pub fn real(&self) -> Vectorized<T> {
        
        todo!();
        /*
            // other_t_real is for SFINAE and clarity. Make sure it is not changed.
        static_assert(is_same<other_t_real, T>::value, "other_t_real must be T");
        return *this;
        */
    }

    //template <typename complex_t_real = T, typename enable_if<is_complex<complex_t_real>::value, int>::type = 0>
    pub fn real(&self) -> Vectorized<T> {
        
        todo!();
        /*
            // complex_t_real is for SFINAE and clarity. Make sure it is not changed.
        static_assert(is_same<complex_t_real, T>::value, "complex_t_real must be T");
        return map([](T x) { return static_cast<T>(x.real()); });
        */
    }

    //template <typename other_t_imag = T, typename enable_if<!is_complex<other_t_imag>::value, int>::type = 0>
    pub fn imag(&self) -> Vectorized<T> {
        
        todo!();
        /*
            // other_t_imag is for SFINAE and clarity. Make sure it is not changed.
        static_assert(is_same<other_t_imag, T>::value, "other_t_imag must be T");
        return Vectorized(0);
        */
    }

    //template <typename complex_t_imag = T, typename enable_if<is_complex<complex_t_imag>::value, int>::type = 0>
    pub fn imag(&self) -> Vectorized<T> {
        
        todo!();
        /*
            // complex_t_imag is for SFINAE and clarity. Make sure it is not changed.
        static_assert(is_same<complex_t_imag, T>::value, "complex_t_imag must be T");
        return map([](T x) { return static_cast<T>(x.imag()); });
        */
    }

    //template <typename other_t_conj = T, typename enable_if<!is_complex<other_t_conj>::value, int>::type = 0>
    pub fn conj(&self) -> Vectorized<T> {
        
        todo!();
        /*
            // other_t_conj is for SFINAE and clarity. Make sure it is not changed.
        static_assert(is_same<other_t_conj, T>::value, "other_t_conj must be T");
        return *this;
        */
    }

    //template <typename complex_t_conj = T, typename enable_if<is_complex<complex_t_conj>::value, int>::type = 0>
    pub fn conj(&self) -> Vectorized<T> {
        
        todo!();
        /*
            // complex_t_conj is for SFINAE and clarity. Make sure it is not changed.
        static_assert(is_same<complex_t_conj, T>::value, "complex_t_conj must be T");
        return map([](T x) { return static_cast<T>(conj(x)); });
        */
    }
    
    pub fn acos(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(acos);
        */
    }
    
    pub fn asin(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(asin);
        */
    }
    
    pub fn atan(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(atan);
        */
    }
    
    pub fn atan2(&self, exp: &Vectorized<T>) -> Vectorized<T> {
        
        todo!();
        /*
            Vectorized<T> ret;
        for (i64 i = 0; i < size(); i++) {
          ret[i] = atan2(values[i], exp[i]);
        }
        return ret;
        */
    }

  //template < typename U = T, typename enable_if_t<is_floating_point<U>::value, int> = 0>
    pub fn copysign(&self, sign: &Vectorized<T>) -> Vectorized<T> {
        
        todo!();
        /*
            Vectorized<T> ret;
        for (size_type i = 0; i < size(); i++) {
          ret[i] = copysign(values[i], sign[i]);
        }
        return ret;
        */
    }
    
    pub fn erf(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(erf);
        */
    }
    
    pub fn erfc(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(erfc);
        */
    }
    
    pub fn erfinv(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(calc_erfinv);
        */
    }
    
    pub fn exp(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(exp);
        */
    }
    
    pub fn expm1(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(expm1);
        */
    }
    
    pub fn frac(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return *this - this->trunc();
        */
    }

    //template < typename U = T, typename enable_if_t<is_floating_point<U>::value, int> = 0>
    pub fn fmod(&self, q: &Vectorized<T>) -> Vectorized<T> {
        
        todo!();
        /*
            // U is for SFINAE purposes only. Make sure it is not changed.
        static_assert(is_same<U, T>::value, "U must be T");
        Vectorized<T> ret;
        for (i64 i = 0; i < size(); i++) {
          ret[i] = fmod(values[i], q[i]);
        }
        return ret;
        */
    }
    
    pub fn log(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(log);
        */
    }
    
    pub fn log10(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(log10);
        */
    }
    
    pub fn log1p(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(log1p);
        */
    }

    //template <typename other_t_log2 = T, typename enable_if<!is_complex<other_t_log2>::value, int>::type = 0>
    pub fn log2(&self) -> Vectorized<T> {
        
        todo!();
        /*
            // other_t_log2 is for SFINAE and clarity. Make sure it is not changed.
        static_assert(is_same<other_t_log2, T>::value, "other_t_log2 must be T");
        return map(log2);
        */
    }

    //template <typename complex_t_log2 = T, typename enable_if<is_complex<complex_t_log2>::value, int>::type = 0>
    pub fn log2(&self) -> Vectorized<T> {
        
        todo!();
        /*
            // complex_t_log2 is for SFINAE and clarity. Make sure it is not changed.
        static_assert(is_same<complex_t_log2, T>::value, "complex_t_log2 must be T");
        const T log_2 = T(log(2.0));
        return Vectorized(map(log))/Vectorized(log_2);
        */
    }
    
    pub fn ceil(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(native::ceil_impl);
        */
    }
    
    pub fn cos(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(cos);
        */
    }
    
    pub fn cosh(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(cosh);
        */
    }
    
    pub fn floor(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(native::floor_impl);
        */
    }
    
    pub fn hypot(&self, b: &Vectorized<T>) -> Vectorized<T> {
        
        todo!();
        /*
            Vectorized<T> ret;
        for (i64 i = 0; i < size(); i++) {
          ret[i] = hypot(values[i], b[i]);
        }
        return ret;
        */
    }
    
    pub fn i0(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(calc_i0);
        */
    }
    
    pub fn i0e(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(calc_i0e);
        */
    }
    
    pub fn igamma(&self, x: &Vectorized<T>) -> Vectorized<T> {
        
        todo!();
        /*
            Vectorized<T> ret;
        for (i64 i = 0; i < size(); i++) {
          ret[i] = calc_igamma(values[i], x[i]);
        }
        return ret;
        */
    }
    
    pub fn igammac(&self, x: &Vectorized<T>) -> Vectorized<T> {
        
        todo!();
        /*
            Vectorized<T> ret;
        for (i64 i = 0; i < size(); i++) {
          ret[i] = calc_igammac(values[i], x[i]);
        }
        return ret;
        */
    }
    
    pub fn neg(&self) -> Vectorized<T> {
        
        todo!();
        /*
            // NB: the trailing return type is needed because we need to coerce the
        // return value back to T in the case of unary operator- incuring a
        // promotion
        return map([](T x) -> T { return -x; });
        */
    }
    
    pub fn nextafter(&self, b: &Vectorized<T>) -> Vectorized<T> {
        
        todo!();
        /*
            Vectorized<T> ret;
        for (i64 i = 0; i < size(); i++) {
          ret[i] = nextafter(values[i], b[i]);
        }
        return ret;
        */
    }
    
    pub fn round(&self) -> Vectorized<T> {
        
        todo!();
        /*
            // We do not use round because we would like to round midway numbers to the nearest even integer.
        return map(native::round_impl);
        */
    }
    
    pub fn sin(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(sin);
        */
    }
    
    pub fn sinh(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(sinh);
        */
    }
    
    pub fn tan(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(tan);
        */
    }
    
    pub fn tanh(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(tanh);
        */
    }
    
    pub fn trunc(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(native::trunc_impl);
        */
    }
    
    pub fn lgamma(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(lgamma);
        */
    }
    
    pub fn sqrt(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map(sqrt);
        */
    }
    
    pub fn reciprocal(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map([](T x) { return (T)(1) / x; });
        */
    }
    
    pub fn rsqrt(&self) -> Vectorized<T> {
        
        todo!();
        /*
            return map([](T x) { return (T)1 / sqrt(x); });
        */
    }
    
    pub fn pow(&self, exp: &Vectorized<T>) -> Vectorized<T> {
        
        todo!();
        /*
            Vectorized<T> ret;
        for (i64 i = 0; i < size(); i++) {
          ret[i] = pow(values[i], exp[i]);
        }
        return ret;
        */
    }
    
    #[inline] pub fn binary_pred<Op>(&self, 
        other: &Vectorized<T>,
        op:    Op) -> Vectorized<T> {
    
        todo!();
        /*
            // All bits are set to 1 if the pred is true, otherwise 0.
        Vectorized<T> vec;
        for (i64 i = 0; i != size(); i++) {
          if (op(values[i], other.values[i])) {
            memset(static_cast<void*>(vec.values + i), 0xFF, sizeof(T));
          } else {
            memset(static_cast<void*>(vec.values + i), 0, sizeof(T));
          }
        }
        return vec;
        */
    }
    
    
    #[inline] pub fn binary_pred_bool<Op>(&self, 
        other: &Vectorized<T>,
        op:    Op) -> Vectorized<T> {
    
        todo!();
        /*
            // 1 if the pred is true, otherwise 0.
        Vectorized<T> vec;
        for (int i = 0; i != size(); ++ i) {
          vec[i] = bool(op(values[i], other.values[i]));
        }
        return vec;
        */
    }

    pub fn eq(&self, other: &Vectorized<T>) -> Vectorized<T> {
        
        todo!();
        /*
            return binary_pred_bool(other, equal_to<T>());
        */
    }
    
    pub fn ne(&self, other: &Vectorized<T>) -> Vectorized<T> {
        
        todo!();
        /*
            return binary_pred_bool(other, not_equal_to<T>());
        */
    }
    
    pub fn gt(&self, other: &Vectorized<T>) -> Vectorized<T> {
        
        todo!();
        /*
            return binary_pred_bool(other, greater<T>());
        */
    }
    
    pub fn ge(&self, other: &Vectorized<T>) -> Vectorized<T> {
        
        todo!();
        /*
            return binary_pred_bool(other, greater_equal<T>());
        */
    }
    
    pub fn lt(&self, other: &Vectorized<T>) -> Vectorized<T> {
        
        todo!();
        /*
            return binary_pred_bool(other, less<T>());
        */
    }
    
    pub fn le(&self, other: &Vectorized<T>) -> Vectorized<T> {
        
        todo!();
        /*
            return binary_pred_bool(other, less_equal<T>());
        */
    }
}

lazy_static!{
    /*
    Vectorized<T> operator==(const Vectorized<T>& other) const { return binary_pred(other, equal_to<T>()); }
      Vectorized<T> operator!=(const Vectorized<T>& other) const { return binary_pred(other, not_equal_to<T>()); }
      Vectorized<T> operator>=(const Vectorized<T>& other) const { return binary_pred(other, greater_equal<T>()); }
      Vectorized<T> operator<=(const Vectorized<T>& other) const { return binary_pred(other, less_equal<T>()); }
      Vectorized<T> operator>(const Vectorized<T>& other) const { return binary_pred(other, greater<T>()); }
      Vectorized<T> operator<(const Vectorized<T>& other) const { return binary_pred(other, less<T>()); }
    */
}

lazy_static!{
    /*
    template <class T> Vectorized<T> inline operator+(const Vectorized<T> &a, const Vectorized<T> &b) {
      Vectorized<T> c;
      for (int i = 0; i != Vectorized<T>::size(); i++) {
        c[i] = a[i] + b[i];
      }
      return c;
    }

    template <class T> Vectorized<T> inline operator-(const Vectorized<T> &a, const Vectorized<T> &b) {
      Vectorized<T> c;
      for (int i = 0; i != Vectorized<T>::size(); i++) {
        c[i] = a[i] - b[i];
      }
      return c;
    }

    template <class T> Vectorized<T> inline operator*(const Vectorized<T> &a, const Vectorized<T> &b) {
      Vectorized<T> c;
      for (int i = 0; i != Vectorized<T>::size(); i++) {
        c[i] = a[i] * b[i];
      }
      return c;
    }

    template <class T> Vectorized<T> inline operator/(const Vectorized<T> &a, const Vectorized<T> &b) __ubsan_ignore_float_divide_by_zero__ {
      Vectorized<T> c;
      for (int i = 0; i != Vectorized<T>::size(); i++) {
        c[i] = a[i] / b[i];
      }
      return c;
    }

    template <class T> Vectorized<T> inline operator||(
        const Vectorized<T> &a, const Vectorized<T> &b) {
      Vectorized<T> c;
      for (int i = 0; i != Vectorized<T>::size(); i++) {
        c[i] = a[i] || b[i];
      }
      return c;
    }
    */
}

// Implements the IEEE 754 201X `maximum`
// operation, which propagates NaN if either input
// is a NaN.
//
// template <class T, typename enable_if<!is_complex<T>::value, int>::type = 0>
//
#[inline] pub fn maximum<T>(
        a: &Vectorized<T>,
        b: &Vectorized<T>) -> Vectorized<T> {
    
    todo!();
        /*
            Vectorized<T> c;
      for (int i = 0; i != Vectorized<T>::size(); i++) {
        c[i] = (a[i] > b[i]) ? a[i] : b[i];
        if (_isnan(a[i])) {
          // If either input is NaN, propagate a NaN.
          // NOTE: The case where b[i] was NaN is handled correctly by the naive
          // ternary operator above.
          c[i] = a[i];
        }
      }
      return c;
        */
}

// template <class T, typename enable_if<is_complex<T>::value, int>::type = 0>
#[inline] pub fn maximum_complex<T: Complex>(
        a: &Vectorized<T>,
        b: &Vectorized<T>) -> Vectorized<T> {
    
    todo!();
        /*
            Vectorized<T> c;
      for (int i = 0; i != Vectorized<T>::size(); i++) {
        c[i] = (abs(a[i]) > abs(b[i])) ? a[i] : b[i];
        if (_isnan(a[i])) {
          // If either input is NaN, propagate a NaN.
          // NOTE: The case where b[i] was NaN is handled correctly by the naive
          // ternary operator above.
          c[i] = a[i];
        }
      }
      return c;
        */
}

// Implements the IEEE 754 201X `minimum`
// operation, which propagates NaN if either input
// is a NaN.
//
// template <class T, typename enable_if<!is_complex<T>::value, int>::type = 0>
//
#[inline] pub fn minimum<T>(
        a: &Vectorized<T>,
        b: &Vectorized<T>) -> Vectorized<T> {
    
    todo!();
        /*
            Vectorized<T> c;
      for (int i = 0; i != Vectorized<T>::size(); i++) {
        c[i] = (a[i] < b[i]) ? a[i] : b[i];
        if (_isnan(a[i])) {
          // If either input is NaN, propagate a NaN.
          // NOTE: The case where b[i] was NaN is handled correctly by the naive
          // ternary operator above.
          c[i] = a[i];
        }
      }
      return c;
        */
}

// template <class T, typename enable_if<is_complex<T>::value, int>::type = 0>
#[inline] pub fn minimum_complex<T: Complex>(
        a: &Vectorized<T>,
        b: &Vectorized<T>) -> Vectorized<T> {
    
    todo!();
        /*
            Vectorized<T> c;
      for (int i = 0; i != Vectorized<T>::size(); i++) {
        c[i] = (abs(a[i]) < abs(b[i])) ? a[i] : b[i];
        if (_isnan(a[i])) {
          // If either input is NaN, propagate a NaN.
          // NOTE: The case where b[i] was NaN is handled correctly by the naive
          // ternary operator above.
          c[i] = a[i];
        }
      }
      return c;
        */
}

// template <class T, typename enable_if<!is_complex<T>::value, int>::type = 0>
#[inline] pub fn clamp(
        a:       &Vectorized<T>,
        min_vec: &Vectorized<T>,
        max_vec: &Vectorized<T>) -> Vectorized<T> {
    
    todo!();
        /*
            Vectorized<T> c;
      for (int i = 0; i != Vectorized<T>::size(); i++) {
        c[i] = min(max(a[i], min_vec[i]), max_vec[i]);
      }
      return c;
        */
}

//template <class T, typename enable_if<!is_complex<T>::value, int>::type = 0>
#[inline] pub fn clamp_max(
        a:       &Vectorized<T>,
        max_vec: &Vectorized<T>) -> Vectorized<T> {
    
    todo!();
        /*
            Vectorized<T> c;
      for (int i = 0; i != Vectorized<T>::size(); i++) {
        c[i] = a[i] > max_vec[i] ? max_vec[i] : a[i];
      }
      return c;
        */
}

//template <class T, typename enable_if<!is_complex<T>::value, int>::type = 0>
#[inline] pub fn clamp_min(
        a:       &Vectorized<T>,
        min_vec: &Vectorized<T>) -> Vectorized<T> {
    
    todo!();
        /*
            Vectorized<T> c;
      for (int i = 0; i != Vectorized<T>::size(); i++) {
        c[i] = a[i] < min_vec[i] ? min_vec[i] : a[i];
      }
      return c;
        */
}

lazy_static!{
    /*
    #ifdef target_feature = "avx2"

    #[inline] pub fn bitwise_binary_op<T, Op>(
            a:  &Vectorized<T>,
            b:  &Vectorized<T>,
            op: Op) -> Vectorized<T> {

        todo!();
            /*
                __m256i buffer;
          __m256i a_buffer = _mm256_loadu_si256(reinterpret_cast<const __m256i*>((const T*)a));
          __m256i b_buffer = _mm256_loadu_si256(reinterpret_cast<const __m256i*>((const T*)b));
          buffer = op(a_buffer, b_buffer);
          __at_align32__ T results[Vectorized<T>::size()];
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(results), buffer);
          return Vectorized<T>::loadu(results);
            */
    }



    template<class T, typename enable_if_t<!is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
    inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
      // We enclose _mm256_and_si256 with lambda because it is always_inline
      return bitwise_binary_op(a, b, [](__m256i a, __m256i b) { return _mm256_and_si256(a, b); });
    }
    template<class T, typename enable_if_t<!is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
    inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
      // We enclose _mm256_or_si256 with lambda because it is always_inline
      return bitwise_binary_op(a, b, [](__m256i a, __m256i b) { return _mm256_or_si256(a, b); });
    }
    template<class T, typename enable_if_t<!is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
    inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
      // We enclose _mm256_xor_si256 with lambda because it is always_inline
      return bitwise_binary_op(a, b, [](__m256i a, __m256i b) { return _mm256_xor_si256(a, b); });
    }

    #else

    #[inline] pub fn bitwise_binary_op<T, Op>(
            a:  &Vectorized<T>,
            b:  &Vectorized<T>,
            op: Op) -> Vectorized<T> {

        todo!();
            /*
                static constexpr u32 element_no = 32 / sizeof(intmax_t);
          __at_align32__ intmax_t buffer[element_no];
          const intmax_t *a_ptr = reinterpret_cast<const intmax_t*>((const T*) a);
          const intmax_t *b_ptr = reinterpret_cast<const intmax_t*>((const T*) b);
          for (u32 i = 0U; i < element_no; ++ i) {
            buffer[i] = op(a_ptr[i], b_ptr[i]);
          }
          return Vectorized<T>::loadu(buffer);
            */
    }



    template<class T, typename enable_if_t<!is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
    inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
      return bitwise_binary_op(a, b, bit_and<intmax_t>());
    }
    template<class T, typename enable_if_t<!is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
    inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
      return bitwise_binary_op(a, b, bit_or<intmax_t>());
    }
    template<class T, typename enable_if_t<!is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
    inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
      return bitwise_binary_op(a, b, bit_xor<intmax_t>());
    }

    #endif
    */
}

lazy_static!{
    /*
    template<class T, typename enable_if_t<!is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
    inline Vectorized<T> operator~(const Vectorized<T>& a) {
      Vectorized<T> ones;  // All bits are 1
      memset((T*) ones, 0xFF, 32);
      return a ^ ones;
    }

    template <typename T>
    inline Vectorized<T>& operator += (Vectorized<T>& a, const Vectorized<T>& b) {
      a = a + b;
      return a;
    }
    template <typename T>
    inline Vectorized<T>& operator -= (Vectorized<T>& a, const Vectorized<T>& b) {
      a = a - b;
      return a;
    }
    template <typename T>
    inline Vectorized<T>& operator /= (Vectorized<T>& a, const Vectorized<T>& b) {
      a = a / b;
      return a;
    }
    template <typename T>
    inline Vectorized<T>& operator %= (Vectorized<T>& a, const Vectorized<T>& b) {
      a = a % b;
      return a;
    }
    template <typename T>
    inline Vectorized<T>& operator *= (Vectorized<T>& a, const Vectorized<T>& b) {
      a = a * b;
      return a;
    }
    */
}

#[inline] pub fn fmadd<T>(
        a: &Vectorized<T>,
        b: &Vectorized<T>,
        c: &Vectorized<T>) -> Vectorized<T> {

    todo!();
        /*
            return a * b + c;
        */
}

lazy_static!{
    /*
    template <i64 scale = 1, typename T = void>
    enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<T>>
    inline gather(T const* base_addr, const Vectorized<int_same_size_t<T>>& vindex) {
      static constexpr int size = Vectorized<T>::size();
      int_same_size_t<T> index_arr[size];
      vindex.store(static_cast<void*>(index_arr));
      T buffer[size];
      for (i64 i = 0; i < size; i++) {
        buffer[i] = base_addr[index_arr[i] * scale / sizeof(T)];
      }
      return Vectorized<T>::loadu(static_cast<void*>(buffer));
    }

    template <i64 scale = 1, typename T = void>
    enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<T>>
    inline mask_gather(const Vectorized<T>& src, T const* base_addr,
                       const Vectorized<int_same_size_t<T>>& vindex, Vectorized<T>& mask) {
      static constexpr int size = Vectorized<T>::size();
      T src_arr[size];
      int_same_size_t<T> mask_arr[size];  // use int type so we can logical and
      int_same_size_t<T> index_arr[size];
      src.store(static_cast<void*>(src_arr));
      mask.store(static_cast<void*>(mask_arr));
      vindex.store(static_cast<void*>(index_arr));
      T buffer[size];
      for (i64 i = 0; i < size; i++) {
        if (mask_arr[i] & 0x01) {  // check highest bit
          buffer[i] = base_addr[index_arr[i] * scale / sizeof(T)];
        } else {
          buffer[i] = src_arr[i];
        }
      }
      mask = Vectorized<T>();  // "zero out" mask
      return Vectorized<T>::loadu(static_cast<void*>(buffer));
    }

    // Cast a given vector to another type without changing the bits representation.
    // So a Vec<double> of 256 bits containing all ones can be cast to a
    // Vec<i64> of 256 bits containing all ones (i.e., four negative 1s).
    namespace {
      // There is a struct here because we don't have static_if and I can't
      // partially specialize a templated function.
      template<typename dst_t, typename src_t>
      struct CastImpl {
        static inline Vectorized<dst_t> apply(const Vectorized<src_t>& src) {
          src_t src_arr[Vectorized<src_t>::size()];
          src.store(static_cast<void*>(src_arr));
          return Vectorized<dst_t>::loadu(static_cast<const void*>(src_arr));
        }
      };

      template<typename Scalar>
      struct CastImpl<Scalar, Scalar> {
        static inline Vectorized<Scalar> apply(const Vectorized<Scalar>& src) {
          return src;
        }
      };
    }
    template<typename dst_t, typename src_t>
    inline Vectorized<dst_t> cast(const Vectorized<src_t>& src) {
      return CastImpl<dst_t, src_t>::apply(src);
    }

    template <typename T>
    inline Vectorized<int_same_size_t<T>> convert_to_int_of_same_size(const Vectorized<T>& src) {
      static constexpr int size = Vectorized<T>::size();
      T src_arr[size];
      src.store(static_cast<void*>(src_arr));
      int_same_size_t<T> buffer[size];
      for (i64 i = 0; i < size; i++) {
        buffer[i] = static_cast<int_same_size_t<T>>(src_arr[i]);
      }
      return Vectorized<int_same_size_t<T>>::loadu(static_cast<void*>(buffer));
    }

    // E.g., inputs: a           Vectorized<float>   = {a0, b0, a1, b1, a2, b2, a3, b3}
    //               b           Vectorized<float>   = {a4, b4, a5, b5, a6, b6, a7, b7}
    //       returns:            Vectorized<float>   = {a0, a1, a2, a3, a4, a5, a6, a7}
    //                           Vectorized<float>   = {b0, b1, b2, b3, b4, b5, b6, b7}
    template <typename T>
    inline enable_if_t<Vectorized<T>::size() % 2 == 0, pair<Vectorized<T>, Vectorized<T>>>
    deinterleave2(const Vectorized<T>& a, const Vectorized<T>& b) {
      static constexpr int size = Vectorized<T>::size();
      static constexpr int half_size = size / 2;
      T a_arr[size];
      T b_arr[size];
      T buffer1[size];
      T buffer2[size];
      a.store(static_cast<void*>(a_arr));
      b.store(static_cast<void*>(b_arr));
      for (i64 i = 0; i < half_size; i++) {
        buffer1[i] = a_arr[i * 2];
        buffer1[half_size + i] = b_arr[i * 2];
        buffer2[i] = a_arr[i * 2 + 1];
        buffer2[half_size + i] = b_arr[i * 2 + 1];
      }
      return make_pair(Vectorized<T>::loadu(static_cast<void*>(buffer1)),
                            Vectorized<T>::loadu(static_cast<void*>(buffer2)));
    }

    // inverse operation of deinterleave2
    // E.g., inputs: a           Vectorized<float>   = {a0, a1, a2, a3, a4, a5, a6, a7}
    //               b           Vectorized<float>   = {b0, b1, b2, b3, b4, b5, b6, b7}
    //       returns:            Vectorized<float>   = {a0, b0, a1, b1, a2, b2, a3, b3}
    //                           Vectorized<float>   = {a4, b4, a5, b5, a6, b6, a7, b7}
    template <typename T>
    inline enable_if_t<Vectorized<T>::size() % 2 == 0, pair<Vectorized<T>, Vectorized<T>>>
    interleave2(const Vectorized<T>& a, const Vectorized<T>& b) {
      static constexpr int size = Vectorized<T>::size();
      static constexpr int half_size = size / 2;
      T a_arr[size];
      T b_arr[size];
      T buffer1[size];
      T buffer2[size];
      a.store(static_cast<void*>(a_arr));
      b.store(static_cast<void*>(b_arr));
      for (i64 i = 0; i < half_size; i++) {
        buffer1[i * 2] = a_arr[i];
        buffer1[i * 2 + 1] = b_arr[i];
        buffer2[i * 2] = a_arr[half_size + i];
        buffer2[i * 2 + 1] = b_arr[half_size + i];
      }
      return make_pair(Vectorized<T>::loadu(static_cast<void*>(buffer1)),
                            Vectorized<T>::loadu(static_cast<void*>(buffer2)));
    }

    template <typename src_T, typename dst_T>
    inline void convert(const src_T *src, dst_T *dst, i64 n) {
    #ifndef _MSC_VER
    # pragma unroll
    #endif
      for (i64 i = 0; i < n; i++) {
        *dst = static_cast_with_inter_type<dst_T, src_T>::apply(*src);
        src++;
        dst++;
      }
    }
    */
}

