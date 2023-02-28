crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp]

/**
  | -----------
  | @note
  | 
  | Undefined behavior when performing
  | addition is intentionally ignored.
  |
  */
pub fn add_kernel(
        iter:         &mut TensorIteratorBase,
        alpha_scalar: &Scalar)  {
    
    todo!();
        /*
            if (iter.dtype() == ScalarType::Bool) {
          using Scalar = bool;
          auto alpha = alpha_scalar.to<Scalar>();
          cpu_kernel(iter,
            [=](Scalar a, Scalar b) __ubsan_ignore_undefined__ -> Scalar { return a + alpha * b; });
      } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "add_cpu/sub_cpu", [&]() {
          auto alpha = alpha_scalar.to<Scalar>();
          auto alpha_vec = Vectorized<Scalar>(alpha);
          cpu_kernel_vec(iter,
            [=](Scalar a, Scalar b) __ubsan_ignore_undefined__ -> Scalar { return a + alpha * b; },
            [=](Vectorized<Scalar> a, Vectorized<Scalar> b) __ubsan_ignore_undefined__ {
              return vec::fmadd(b, alpha_vec, a);
            });
          });
      }
        */
}

pub fn add_clamp_kernel(
        iter:         &mut TensorIterator,
        alpha_scalar: &Scalar,
        min_val:      &Scalar,
        max_val:      &Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES(iter.dtype(), "add_clamp_cpu", [&]() {
        auto alpha = alpha_scalar.to<Scalar>();
        auto alpha_vec = Vectorized<Scalar>(alpha);
        auto min_scalar = min_val.to<Scalar>();
        auto min_vec = Vectorized<Scalar>(min_scalar);
        auto max_scalar = max_val.to<Scalar>();
        auto max_vec = Vectorized<Scalar>(max_scalar);
        cpu_kernel_vec(iter,
          [=](Scalar a, Scalar b) __ubsan_ignore_undefined__ -> Scalar {
            return min(max_scalar, max(min_scalar, static_cast<Scalar>(a + alpha * b)));
          },
          [=](Vectorized<Scalar> a, Vectorized<Scalar> b) __ubsan_ignore_undefined__ {
            auto add_clamp_res = vec::fmadd(b, alpha_vec, a);
            add_clamp_res = vec::clamp_min(add_clamp_res, min_vec);
            add_clamp_res = vec::clamp_max(add_clamp_res, max_vec);
            return add_clamp_res;
          });
        });
        */
}

pub fn atan2_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "atan2_cpu", [&]() {
        cpu_kernel_vec(iter, [=](Scalar a, Scalar b) -> Scalar {
        return atan2(a, b);
      },
        [=](Vectorized<Scalar> a, Vectorized<Scalar> b) {
          return a.atan2(b);
        });
      });
        */
}

/**
  | -----------
  | @note
  | 
  | Undefined behavior when performing
  | subtraction is intentionally ignored.
  |
  */
#[__ubsan_ignore_undefined__]
pub fn sub_kernel(
        iter:         &mut TensorIteratorBase,
        alpha_scalar: &Scalar)  {
    
    todo!();
        /*
            add_kernel(iter, -alpha_scalar);
        */
}

pub fn mul_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            if (iter.dtype() == ScalarType::Bool) {
        cpu_kernel(iter, [=](bool a, bool b) -> bool { return a && b; });
      } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "mul_cpu", [&]() {
          cpu_kernel_vec(iter,
            [=](Scalar a, Scalar b) -> Scalar { return a * b; },
            [=](Vectorized<Scalar> a, Vectorized<Scalar> b) {
              return a * b;
            });
        });
      }
        */
}

pub fn div_true_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "div_cpu", [&]() {
        cpu_kernel_vec(iter,
          [](Scalar a, Scalar b) __ubsan_ignore_float_divide_by_zero__ -> Scalar {
            return a / b;
          },
          [](Vectorized<Scalar> a, Vectorized<Scalar> b) {
            return a / b;
          });
      });
        */
}

pub fn div_trunc_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            const auto dtype = iter.common_dtype();
      if (isIntegralType(dtype, /*includeBool*/ false)) {
        // There's no SIMD integer division, so don't try to vectorize it.
        // TODO: if the divisor is a scalar, rewrite as multiplication by a constant.
        AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_trunc_cpu", [&]() {
          cpu_kernel(iter, [](Scalar a, Scalar b) -> Scalar {
            TORCH_CHECK(b != 0, "ZeroDivisionError");
            return a / b;
          });
        });
      } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, dtype, "div_trunc_cpu", [&]() {
          cpu_kernel_vec(iter,
            [](Scalar a, Scalar b) __ubsan_ignore_float_divide_by_zero__ -> Scalar {
              return trunc(a / b);
            },
            [](Vectorized<Scalar> a, Vectorized<Scalar> b) {
              return (a / b).trunc();
            });
        });
      }
        */
}

/**
  | NOTE: [Floor Division in Python]
  |
  | Python's __floordiv__ operator is more
  | complicated than just floor(a / b).
  |
  | It aims to maintain the property: a == (a // b)
  | * b + remainder(a, b) which can otherwise fail
  | due to rounding errors in the remainder.
  |
  | So, instead it is calculated as: a // b = (a
  | - remainder(a, b)) / b With some additional
  | fix-ups added to the result.
  |
  | For reference, see CPython's implementation:
  | https://github.com/python/cpython/blob/ace008c531dd685a30c1dd68f9b5ba35f20171cf/Objects/floatobject.c#L636
  */
pub fn div_floor_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            const auto dtype = iter.common_dtype();
      if (dtype == kByte) {
        // In the special case of unsigned integer division, floor division is
        // equivalent to truncation division (since the signs of the divisor and
        // dividend are always the same)
        return div_trunc_kernel(iter);
      } else if (isIntegralType(dtype, /*includeBool*/ false)) {
        // There's no SIMD integer division, so don't try to vectorize it.
        AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_floor_cpu", [&]() {
          cpu_kernel(iter, [](Scalar a, Scalar b) -> Scalar {
            TORCH_CHECK(b != 0, "ZeroDivisionError");
            if ((a < 0) != (b < 0)) {
              // Subtracts one from the results of truncation division if the
              // divisor and dividend have different sign(bit)s and the remainder of
              // the division is nonzero
              const auto quot = a / b;
              const auto rem = a % b;
              return rem ? quot - 1 : quot;
            }

            return a / b;
          });
        });
      } else {
        // See NOTE: [Floor Division in Python]
        AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, dtype, "div_floor_cpu", [&]() {
          using vec_t = Vectorized<Scalar>;
          cpu_kernel_vec(iter,
              [](Scalar a, Scalar b) __ubsan_ignore_float_divide_by_zero__ -> Scalar {
                if (C10_UNLIKELY(b == 0)) {
                  // Divide by zero: return standard IEEE result
                  return a / b;
                }

                auto mod = fmod(a, b);
                auto div = (a - mod) / b;
                if ((mod != 0) && (b < 0) != (mod < 0)) {
                  div -= Scalar(1);
                }

                Scalar floordiv;
                if (div != 0) {
                  floordiv = floor(div);
                  if (div - floordiv > Scalar(0.5)) {
                    floordiv += Scalar(1.0);
                  }
                } else {
                  floordiv = copysign(Scalar(0), a / b);
                }
                return floordiv;
              },
              [](vec_t a, vec_t b) -> vec_t {
                auto mod = a.fmod(b);
                auto div = (a - mod) / b;
                const auto zero = vec_t(0);
                auto mask = (mod != zero) & ((b < zero) ^ (mod < zero));
                const auto one = vec_t(1);
                div = vec_t::blendv(div, div - one, mask);
                auto floordiv = div.floor();
                mask = (div - floordiv) > vec_t(0.5);
                floordiv = vec_t::blendv(floordiv, floordiv + one, mask);
                const auto basic_div = a / b;
                floordiv = vec_t::blendv(floordiv, zero.copysign(basic_div), div == zero);
                floordiv = vec_t::blendv(floordiv, basic_div, b == zero);
                return floordiv;
              });
        });
      }
        */
}

pub fn remainder_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            if (isIntegralType(iter.common_dtype(), /*includeBool*/ false)) {
        AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "remainder_cpu", [&]() {
          cpu_kernel(iter, [](Scalar a, Scalar b) -> Scalar {
            TORCH_CHECK(b != 0, "ZeroDivisionError");
            Scalar r = a % b;
            if ((r != 0) && ((r < 0) != (b < 0))) {
              r += b;
            }
            return r;
          });
        });
      } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.common_dtype(), "remainder_cpu", [&]() {
          cpu_kernel_vec(iter,
            [=](Scalar a, Scalar b) __ubsan_ignore_float_divide_by_zero__ -> Scalar {
              Scalar mod = fmod(a, b);
              if ((mod != 0) && ((b < 0) != (mod < 0))) mod += b;
              return mod;
            },
            [=](Vectorized<Scalar> a, Vectorized<Scalar> b) {
              auto mod = a.fmod(b);
              const auto zero = Vectorized<Scalar>(0);
              auto mask = (mod != zero) & ((b < zero) ^ (mod < zero));
              return Vectorized<Scalar>::blendv(mod, mod + b, mask);
            });
        });
      }
        */
}

pub fn bitwise_and_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            if (iter.dtype() == ScalarType::Bool) {
        cpu_kernel(
            iter,
            [](bool a, bool b) {
              return a && b;
            });
      } else {
        AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_and_cpu", [&]() {
          cpu_kernel_vec(
              iter,
              [](Scalar a, Scalar b) -> Scalar {
                return a & b;
              },
              [](Vectorized<Scalar> a, Vectorized<Scalar> b) {
                return a & b;
              });
        });
      }
        */
}

pub fn bitwise_or_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            if (iter.dtype() == ScalarType::Bool) {
        cpu_kernel(
            iter,
            [](bool a, bool b) {
              return a || b;
            });
      } else {
        AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_or_cpu", [&]() {
          cpu_kernel_vec(
              iter,
              [](Scalar a, Scalar b) -> Scalar {
                return a | b;
              },
              [](Vectorized<Scalar> a, Vectorized<Scalar> b) {
                return a | b;
              });
        });
      }
        */
}

pub fn bitwise_xor_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            if (iter.dtype() == ScalarType::Bool) {
        // Boolean type does not work with ^ (bitwise XOR) in C++. bitwise_xor wraps this operation for both Boolean and
        // integral types.
        cpu_kernel(
              iter,
              [](bool a, bool b) {
                return a != b;
              });
      } else {
        AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_xor_cpu", [&]() {
          cpu_kernel_vec(
              iter,
              [](Scalar a, Scalar b) -> Scalar {
                return a ^ b;
              },
              [](Vectorized<Scalar> a, Vectorized<Scalar> b) {
                return a ^ b;
              });
        });
      }
        */
}

pub fn lshift_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            if (iter.dtype() == ScalarType::Float || iter.dtype() == ScalarType::Double) {
        AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "lshift_cpu", [&]() {
          auto base_vec = Vectorized<Scalar>((Scalar)(2));
          cpu_kernel_vec(
            iter,
            [=](Scalar a, Scalar b) -> Scalar {
              return a * pow((Scalar)(2), b);
            },
            [=](Vectorized<Scalar> a, Vectorized<Scalar> b) {
              return a * base_vec.pow(b);
          });
        });
      } else {
        AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "lshift_cpu", [&]() {
          cpu_kernel(iter,
            [](Scalar a, Scalar b) -> Scalar {
              return static_cast<make_unsigned_t<Scalar>>(a) << b;
          });
        });
      }
        */
}


pub fn logical_and_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            // See Note [special-case bool outputs]
      if (iter.dtype() == ScalarType::Bool) {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "logical_and_cpu", [&]() {
          cpu_kernel(iter,
            [](Scalar a, Scalar b) -> bool {
              return a && b;
            });
        });
      } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.common_dtype(), "logical_and_cpu", [&]() {
          cpu_kernel(iter,
            [](Scalar a, Scalar b) -> Scalar {
              return static_cast<Scalar>(a && b);
            });
        });
      }
        */
}

pub fn logical_or_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            // See Note [special-case bool outputs]
      if (iter.dtype() == ScalarType::Bool) {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "logical_or_cpu", [&]() {
          cpu_kernel(iter,
            [](Scalar a, Scalar b) -> bool {
              return a || b;
            });
        });
      } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "logical_or_cpu", [&]() {
          cpu_kernel(iter,
            [](Scalar a, Scalar b) -> Scalar {
              return static_cast<Scalar>(a || b);
            });
        });
      }
        */
}

pub fn logical_xor_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            // See Note [special-case bool outputs]
      if (iter.dtype() == ScalarType::Bool) {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "logical_xor_cpu", [&]() {
          cpu_kernel(iter,
            [](Scalar a, Scalar b) -> bool {
              return bool(a) != bool(b);
            });
        });
      } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.common_dtype(), "logical_xor_cpu", [&]() {
          cpu_kernel(iter,
            [](Scalar a, Scalar b) -> Scalar {
              return static_cast<Scalar>(bool(a) != bool(b));
            });
        });
      }
        */
}


pub fn rshift_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            if (iter.dtype() == ScalarType::Float || iter.dtype() == ScalarType::Double) {
        AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "rshift_cpu", [&]() {
          auto base_vec = Vectorized<Scalar>((Scalar)(2));
          cpu_kernel_vec(
            iter,
            [=](Scalar a, Scalar b) -> Scalar {
              return a / pow((Scalar)(2), b);
            },
            [=](Vectorized<Scalar> a, Vectorized<Scalar> b) {
              return a / base_vec.pow(b);
          });
        });
      } else {
        AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "rshift_cpu", [&]() {
          cpu_kernel(iter,
            [](Scalar a, Scalar b) -> Scalar {
              return a >> b;
            });
        });
      }
        */
}


pub fn lt_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            // See Note [special-case bool outputs]
      if (iter.dtype() == ScalarType::Bool) {
        AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "lt_cpu", [&]() {
          cpu_kernel(iter,
            [](Scalar a, Scalar b) -> bool {
              return a < b;
            });
        });
      } else {
        AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "lt_cpu", [&]() {
          cpu_kernel_vec(
            iter,
            [](Scalar a, Scalar b) -> Scalar {
              return a < b;
            },
            [](Vectorized<Scalar> a, Vectorized<Scalar> b) -> Vectorized<Scalar> {
              return a.lt(b);
            });
        });
      }
        */
}


pub fn le_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            // See Note [special-case bool outputs]
      if (iter.dtype() == ScalarType::Bool) {
        AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "le_cpu", [&]() {
          cpu_kernel(iter,
            [](Scalar a, Scalar b) -> bool {
              return a <= b;
            });
        });
      } else {
        AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "le_cpu", [&]() {
          cpu_kernel_vec(
            iter,
            [](Scalar a, Scalar b) -> Scalar {
              return a <= b;
            },
            [](Vectorized<Scalar> a, Vectorized<Scalar> b) -> Vectorized<Scalar> {
              return a.le(b);
            });
        });
      }
        */
}


pub fn gt_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            // See Note [special-case bool outputs]
      if (iter.dtype() == ScalarType::Bool) {
        AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "gt_cpu", [&]() {
          cpu_kernel(iter,
            [](Scalar a, Scalar b) -> bool {
              return a > b;
            });
        });
      } else {
        AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "gt_cpu", [&]() {
          cpu_kernel_vec(
            iter,
            [](Scalar a, Scalar b) -> Scalar {
              return a > b;
            },
            [](Vectorized<Scalar> a, Vectorized<Scalar> b) -> Vectorized<Scalar> {
              return a.gt(b);
            });
        });
      }
        */
}


pub fn ge_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            // See Note [special-case bool outputs]
      if (iter.dtype() == ScalarType::Bool) {
        AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "ge_cpu", [&]() {
          cpu_kernel(iter,
            [](Scalar a, Scalar b) -> bool {
              return a >= b;
            });
        });
      } else {
        AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "ge_cpu", [&]() {
          cpu_kernel_vec(
            iter,
            [](Scalar a, Scalar b) -> Scalar {
              return a >= b;
            },
            [](Vectorized<Scalar> a, Vectorized<Scalar> b) -> Vectorized<Scalar> {
              return a.ge(b);
            });
        });
      }
        */
}

pub fn eq_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            // See Note [special-case bool outputs]
      if (iter.dtype() == ScalarType::Bool) {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "eq_cpu", [&]() {
          cpu_kernel(iter,
            [](Scalar a, Scalar b) -> bool {
              return a == b;
            });
        });
      } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.common_dtype(), "eq_cpu", [&]() {
          cpu_kernel_vec(
            iter,
            [](Scalar a, Scalar b) -> Scalar {
              return a == b;
            },
            [](Vectorized<Scalar> a, Vectorized<Scalar> b) -> Vectorized<Scalar> {
              return a.eq(b);
            });
        });
      }
        */
}

pub fn ne_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            // See Note [special-case bool outputs]
      if (iter.dtype() == ScalarType::Bool) {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "ne_cpu", [&]() {
          cpu_kernel(iter,
            [](Scalar a, Scalar b) -> bool {
              return a != b;
            });
        });
      } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.common_dtype(), "ne_cpu", [&]() {
          cpu_kernel_vec(
            iter,
            [](Scalar a, Scalar b) -> Scalar {
              return a != b;
            },
            [](Vectorized<Scalar> a, Vectorized<Scalar> b) -> Vectorized<Scalar> {
              return a.ne(b);
            });
        });
      }
        */
}


pub fn maximum_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            if (iter.dtype() == ScalarType::Bool) {
        cpu_kernel(iter,
          [](bool a, bool b) -> bool {
            return a || b;
          });
      } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
        AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "maximum_cpu", [&]() {
          cpu_kernel_vec(iter,
            [](Scalar a, Scalar b) -> Scalar { return max(a, b); },
            [](Vectorized<Scalar> a, Vectorized<Scalar> b) { return vec::maximum(a, b); });
        });
      } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "maximum_cpu", [&]() {
          cpu_kernel_vec(iter,
            [](Scalar a, Scalar b) -> Scalar {
              if (a != a || b != b) {
                return numeric_limits<Scalar>::quiet_NaN();
              } else {
                return max(a, b);
              }
            },
            [](Vectorized<Scalar> a, Vectorized<Scalar> b) { return vec::maximum(a, b); });
        });
      }
        */
}

pub fn minimum_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            if (iter.dtype() == ScalarType::Bool) {
        cpu_kernel(iter,
          [](bool a, bool b) -> bool {
            return a && b;
          });
      } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
        AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "minimum_cpu", [&]() {
          cpu_kernel_vec(iter,
            [](Scalar a, Scalar b) -> Scalar { return min(a, b); },
            [](Vectorized<Scalar> a, Vectorized<Scalar> b) { return vec::minimum(a, b); });
        });
      } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "minimum_cpu", [&]() {
          cpu_kernel_vec(iter,
            [](Scalar a, Scalar b) -> Scalar {
              if (a != a || b != b) {
                return numeric_limits<Scalar>::quiet_NaN();
              } else {
                return min(a, b);
              }
            },
            [](Vectorized<Scalar> a, Vectorized<Scalar> b) { return vec::minimum(a, b); });
        });
      }
        */
}


pub fn fmax_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            if (isFloatingType(iter.common_dtype())) {
        AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "fmax_cpu", [&]() {
          cpu_kernel(iter,
            [](Scalar a, Scalar b) -> Scalar {
              return fmax(a, b);
            });
        });
      } else {
        maximum_kernel(iter);
      }
        */
}


pub fn fmin_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            if (isFloatingType(iter.common_dtype())) {
        AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "fmin_cpu", [&]() {
          cpu_kernel(iter,
            [](Scalar a, Scalar b) -> Scalar {
              return fmin(a, b);
            });
        });
      } else {
        minimum_kernel(iter);
      }
        */
}


pub fn smooth_l1_kernel(
        iter: &mut TensorIterator,
        beta: f64)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(
            kBFloat16, kHalf, iter.dtype(), "smooth_l1_cpu", [&]() {
            using Vec = Vectorized<Scalar>;
            const Scalar beta_val(beta);
            const Vec beta_val_vec(beta_val);
            const Vec point_five_vec(static_cast<Scalar>(0.5));
            cpu_kernel_vec(
                iter,
                [&beta_val](Scalar a, Scalar b) -> Scalar {
                  auto z = abs(a - b);
                  return z < beta_val
                      ? static_cast<Scalar>(0.5) * z * z / beta_val
                      : z - static_cast<Scalar>(0.5) * beta_val;
                },
                [&beta_val_vec, &point_five_vec](Vec a, Vec b) {
                  auto z = (a - b).abs();
                  return Vec::blendv(
                      point_five_vec * z * z / beta_val_vec, z - point_five_vec * beta_val_vec, z >= beta_val_vec);
                });
          });
        */
}


pub fn huber_kernel(
        iter:  &mut TensorIterator,
        delta: f64)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "huber_cpu", [&]() {
        using Vec = Vectorized<Scalar>;
        const Scalar delta_val(delta);
        const Vec delta_val_vec(delta_val);
        const Vec point_five_vec(static_cast<Scalar>(0.5));
        cpu_kernel_vec(
          iter,
          [&delta_val](Scalar a, Scalar b) -> Scalar {
            auto z = abs(a - b);
            return z < delta_val ? static_cast<Scalar>(0.5) * z * z :
            delta_val * (z - static_cast<Scalar>(0.5) * delta_val);
          },
          [&delta_val_vec, &point_five_vec](Vec a, Vec b) {
            auto z = (a - b).abs();
            return Vec::blendv(point_five_vec * z * z,
              delta_val_vec * (z - point_five_vec * delta_val_vec),
              z >= delta_val_vec);
        });
      });
        */
}


pub fn sigmoid_backward_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "sigmoid_backward_cpu", [&]() {
        auto one_vec = Vectorized<Scalar>((Scalar)(1));
        cpu_kernel_vec(iter,
          [=](Scalar a, Scalar b) -> Scalar {
            return a * (Scalar(1) - b) * b;
          },
          [=](Vectorized<Scalar> a, Vectorized<Scalar> b) {
            return a * (one_vec - b) * b;
          });
      });
        */
}


pub fn logit_backward_kernel(
        iter:       &mut TensorIterator,
        eps_scalar: &Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND(
          kBFloat16, iter.dtype(), "logit_backward_cpu", [&]() {
            const Scalar eps = eps_scalar.to<Scalar>();
            const Vectorized<Scalar> kZeroVec(Scalar(0));
            const Vectorized<Scalar> kOneVec(Scalar(1));
            if (eps < Scalar(0)) {
              const Vectorized<Scalar> kNanVec(
                  numeric_limits<Scalar>::quiet_NaN());
              cpu_kernel_vec(
                  iter,
                  [](Scalar dy, Scalar x) {
                    return (x < Scalar(0) || x > Scalar(1))
                        ? numeric_limits<Scalar>::quiet_NaN()
                        : ((x == Scalar(0) || x == Scalar(1))
                               ? (dy * numeric_limits<Scalar>::infinity())
                               : (dy / (x * (Scalar(1) - x))));
                  },
                  [kZeroVec, kOneVec, kNanVec](
                      Vectorized<Scalar> dy_vec, Vectorized<Scalar> x_vec) {
                    return Vectorized<Scalar>::blendv(
                        kNanVec,
                        dy_vec / (x_vec * (kOneVec - x_vec)),
                        (x_vec >= kZeroVec) & (x_vec <= kOneVec));
                  });
            } else {
              const Scalar lo = eps;
              const Scalar hi = Scalar(1) - eps;
              const Vectorized<Scalar> lo_vec(lo);
              const Vectorized<Scalar> hi_vec(hi);
              cpu_kernel_vec(
                  iter,
                  [lo, hi](Scalar dy, Scalar x) {
                    return (x < lo || x > hi)
                        ? Scalar(0)
                        : ((x == Scalar(0) || x == Scalar(1))
                               ? dy * numeric_limits<Scalar>::infinity()
                               : dy / (x * (Scalar(1) - x)));
                  },
                  [kZeroVec, kOneVec, lo_vec, hi_vec](
                      Vectorized<Scalar> dy_vec, Vectorized<Scalar> x_vec) {
                    return Vectorized<Scalar>::blendv(
                        kZeroVec,
                        dy_vec / (x_vec * (kOneVec - x_vec)),
                        (x_vec >= lo_vec) & (x_vec <= hi_vec));
                  });
            }
          });
        */
}


pub fn tanh_backward_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            if (isComplexType(iter.dtype())) {
        AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "tanh_backward_cpu", [&]() {
          auto one_vec = Vectorized<Scalar>(Scalar{1});
        cpu_kernel_vec(
          iter,
          [=](Scalar a, Scalar b) -> Scalar {
            return a * conj(Scalar{1} - b * b);
          },
          [=](Vectorized<Scalar> a, Vectorized<Scalar> b) {
            return a * (one_vec - b * b).conj();
          });
      });
      } else {
        AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "tanh_backward_cpu", [&]() {
          auto one_vec = Vectorized<Scalar>(Scalar{1});
          cpu_kernel_vec(
            iter,
            [=](Scalar a, Scalar b) -> Scalar {
              return a * (Scalar{1} - b * b);
            },
            [=](Vectorized<Scalar> a, Vectorized<Scalar> b) {
              return a * (one_vec - b * b);
            });
        });
      }
        */
}


pub fn mse_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            if (iter.dtype() == ScalarType::Half) {
        TORCH_WARN_ONCE("Applying the CPU mse kernel on half-type tensors. "
                        "This may be slower than using float or double-type tensors.");
      }

      AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "mse_cpu", [&]() {
        cpu_kernel_vec(iter,
          [=](Scalar a, Scalar b) -> Scalar {
            auto diff = a - b;
            return diff * diff;
          },
          [=](Vectorized<Scalar> a, Vectorized<Scalar> b) {
          auto diff =  a - b;
          return diff * diff;
          });
      });
        */
}


pub fn fmod_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            if (isIntegralType(iter.common_dtype(), /*includeBool=*/ false)) {
        AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "fmod_cpu", [&]() {
          cpu_kernel(iter, [=](Scalar x, Scalar d) -> Scalar {
            TORCH_CHECK(d != 0, "ZeroDivisionError");
            return x % d;
          });
        });
      } else {
        AT_DISPATCH_FLOATING_TYPES_AND(kHalf, iter.common_dtype(), "fmod_cpu", [&]() {
          cpu_kernel_vec(
            iter,
            [](Scalar x, Scalar d) -> Scalar {
              return fmod(x, d);
            },
            [](Vectorized<Scalar> x, Vectorized<Scalar> d) {
              return x.fmod(d);
            });
        });
      }
        */
}

pub fn logaddexp_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logaddexp_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](Scalar a, Scalar b) -> Scalar {
              if (isinf(a) && a == b) {
                return a;
              } else {
                Scalar m = max(a, b);
                return m + log((Scalar)(1.0) + exp(-abs(a - b)));
              }
            },
            [=](Vectorized<Scalar> a, Vectorized<Scalar> b) {
              Vectorized<Scalar> inf(numeric_limits<Scalar>::infinity());
              Vectorized<Scalar> one(1.0);
              Vectorized<Scalar> m = maximum(a, b);
              return Vectorized<Scalar>::blendv(
                  m + (one + (a - b).abs().neg().exp()).log(),
                  a,
                  (a == b) & (a.abs() == inf));
            });
      });
        */
}


pub fn logaddexp2_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logaddexp2_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](Scalar a, Scalar b) -> Scalar {
              if (isinf(a) && a == b) {
                return a;
              } else {
                Scalar m = max(a, b);
                return m + log2((Scalar)(1.0) + pow((Scalar)(2), -abs(a - b)));
              }
            },
            [=](Vectorized<Scalar> a, Vectorized<Scalar> b) {
              Vectorized<Scalar> inf(numeric_limits<Scalar>::infinity());
              Vectorized<Scalar> one(1.0);
              Vectorized<Scalar> two(2.0);
              Vectorized<Scalar> m = maximum(a, b);
              return Vectorized<Scalar>::blendv(
                  m + (one + two.pow((a - b).abs().neg())).log2(),
                  a,
                  (a == b) & (a.abs() == inf));
            });
      });
        */
}


pub fn gcd_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "gcd_cpu", [&]() {
          cpu_kernel(
              iter,
              [](Scalar a, Scalar b) -> Scalar {
                return calc_gcd(a, b);
              });
        });
        */
}


pub fn lcm_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "lcm_cpu", [&]() {
          cpu_kernel(
              iter,
              [](Scalar a, Scalar b) -> Scalar {
                Scalar g = calc_gcd(a, b);
                return (g == 0) ? 0 : abs(a / g * b);
              });
        });
        */
}


pub fn hypot_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.dtype(), "hypot_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](Scalar a, Scalar b) -> Scalar {
                return hypot(a, b);
            },
            [=](Vectorized<Scalar> a, Vectorized<Scalar> b) {
                return a.hypot(b);
            });
      });
        */
}


pub fn igamma_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "igamma_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](Scalar a, Scalar b) -> Scalar {
                return calc_igamma(a, b);
            },
            [=](Vectorized<Scalar> a, Vectorized<Scalar> b) {
                return a.igamma(b);
            });
      });
        */
}


pub fn igammac_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "igammac_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](Scalar a, Scalar b) -> Scalar {
                return calc_igammac(a, b);
            },
            [=](Vectorized<Scalar> a, Vectorized<Scalar> b) {
                return a.igammac(b);
            });
      });
        */
}


pub fn nextafter_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "nextafter_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](Scalar a, Scalar b) -> Scalar {
                return nextafter(a, b);
            },
            [=](Vectorized<Scalar> a, Vectorized<Scalar> b) {
                return a.nextafter(b);
            });
      });
        */
}


pub fn heaviside_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, iter.dtype(), "heaviside_cpu", [&]() {
        cpu_kernel(iter, [](Scalar a, Scalar b) -> Scalar {
            return a == 0 ? b : static_cast<Scalar>(a > 0);
        });
      });
        */
}


pub fn copysign_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "copysign_cpu", [&]() {
        cpu_kernel_vec(iter,
          [](Scalar a, Scalar b) -> Scalar {
            return copysign(a, b);
          },
          [](Vectorized<Scalar> a, Vectorized<Scalar> b) -> Vectorized<Scalar> {
            return a.copysign(b);
          });
      });
        */
}


pub fn xlogy_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "xlogy_cpu", [&]() {
        cpu_kernel(iter, [](Scalar x, Scalar y) -> Scalar {
          if (_isnan(y)){
            return NAN;
          }
          if (x == 0){
            return 0;
          }
          return x * log(y);
        });
      });
        */
}


pub fn xlog1py_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "xlog1py_cpu", [&]() {
        cpu_kernel(iter, [](Scalar x, Scalar y) -> Scalar {
          if (_isnan(y)){
            return NAN;
          }
          if (x == 0){
            return 0;
          }
          return x * log1p(y);
        });
      });
        */
}

register_dispatch!{add_stub              , &add_kernel}
register_dispatch!{add_clamp_stub        , &add_clamp_kernel}
register_dispatch!{sub_stub              , &sub_kernel}
register_dispatch!{mul_stub              , &mul_kernel}
register_dispatch!{div_true_stub         , &div_true_kernel}
register_dispatch!{div_trunc_stub        , &div_trunc_kernel}
register_dispatch!{div_floor_stub        , &div_floor_kernel}
register_dispatch!{remainder_stub        , &remainder_kernel}
register_dispatch!{atan2_stub            , &atan2_kernel}
register_dispatch!{bitwise_and_stub      , &bitwise_and_kernel}
register_dispatch!{bitwise_or_stub       , &bitwise_or_kernel}
register_dispatch!{bitwise_xor_stub      , &bitwise_xor_kernel}
register_dispatch!{lshift_stub           , &lshift_kernel}
register_dispatch!{rshift_stub           , &rshift_kernel}
register_dispatch!{logical_xor_stub      , &logical_xor_kernel}
register_dispatch!{logical_and_stub      , &logical_and_kernel}
register_dispatch!{logical_or_stub       , &logical_or_kernel}
register_dispatch!{lt_stub               , &lt_kernel}
register_dispatch!{le_stub               , &le_kernel}
register_dispatch!{gt_stub               , &gt_kernel}
register_dispatch!{ge_stub               , &ge_kernel}
register_dispatch!{eq_stub               , &eq_kernel}
register_dispatch!{ne_stub               , &ne_kernel}
register_dispatch!{maximum_stub          , &maximum_kernel}
register_dispatch!{minimum_stub          , &minimum_kernel}
register_dispatch!{fmax_stub             , &fmax_kernel}
register_dispatch!{fmin_stub             , &fmin_kernel}
register_dispatch!{smooth_l1_stub        , &smooth_l1_kernel}
register_dispatch!{huber_stub            , &huber_kernel}
register_dispatch!{sigmoid_backward_stub , &sigmoid_backward_kernel}
register_dispatch!{logit_backward_stub   , &logit_backward_kernel}
register_dispatch!{tanh_backward_stub    , &tanh_backward_kernel}
register_dispatch!{mse_stub              , &mse_kernel}
register_dispatch!{fmod_stub             , &fmod_kernel}
register_dispatch!{logaddexp_stub        , &logaddexp_kernel}
register_dispatch!{logaddexp2_stub       , &logaddexp2_kernel}
register_dispatch!{gcd_stub              , &gcd_kernel}
register_dispatch!{lcm_stub              , &lcm_kernel}
register_dispatch!{hypot_stub            , &hypot_kernel}
register_dispatch!{igamma_stub           , &igamma_kernel}
register_dispatch!{igammac_stub          , &igammac_kernel}
register_dispatch!{nextafter_stub        , &nextafter_kernel}
register_dispatch!{heaviside_stub        , &heaviside_kernel}
register_dispatch!{copysign_stub         , &copysign_kernel}
register_dispatch!{xlogy_stub            , &xlogy_kernel}
register_dispatch!{xlog1py_stub          , &xlog1py_kernel}
