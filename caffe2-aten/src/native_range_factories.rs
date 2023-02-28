crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/RangeFactories.cpp]

declare_dispatch!{
    fn(
        _0: &mut TensorIterator,
        _1: &Scalar,
        _2: &Scalar,
        _3: &Scalar
    ) -> (),
    arange_stub
}

declare_dispatch!{
    fn(
        _0: &mut TensorIterator,
        _1: &Scalar,
        _2: &Scalar,
        _3: i64
    ) -> (),
    linspace_stub
}

pub fn linspace_cpu_out(
    start:          &Scalar,
    end:            &Scalar,
    optional_steps: Option<i64>,
    result:         &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            const auto steps = optional_steps.value_or(100);
      TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

      if (!optional_steps.has_value()) {
        TORCH_WARN_ONCE(
          "Not providing a value for linspace's steps is deprecated and will "
          "throw a runtime error in a future release. This warning will appear "
          "only once per process.");
      }

      if (result.numel() != steps) {
        result.resize_({steps});
      }

      if (steps == 0) {
        // skip
      } else if (steps == 1) {
        result.fill_(start);
      } else {
        Tensor r = result.is_contiguous() ? result : result.contiguous();
        auto iter = TensorIterator::borrowing_nullary_op(r);
        linspace_stub(iter.device_type(), iter, start, end, steps);
        if (!result.is_contiguous()) {
          result.copy_(r);
        }
      }

      return result;
        */
}


pub fn logspace_cpu_out(
        start:          &Scalar,
        end:            &Scalar,
        optional_steps: Option<i64>,
        base:           f64,
        result:         &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            const auto steps = optional_steps.value_or(100);
      TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

      if (!optional_steps.has_value()) {
        TORCH_WARN_ONCE(
          "Not providing a value for logspace's steps is deprecated and will "
          "throw a runtime error in a future release. This warning will appear "
          "only once per process.");
      }

      if (result.numel() != steps) {
        result.resize_({steps});
      }
      Tensor r = result.is_contiguous() ? result : result.contiguous();

      if (steps == 0) {
        // skip
      } else if (steps == 1) {
        if (isComplexType(r.scalar_type())){
          r.fill_(pow(base, start.to<complex<double>>()));
        } else {
          r.fill_(pow(base, start.to<double>()));
        }
      } else if (isComplexType(r.scalar_type())) {
        AT_DISPATCH_COMPLEX_TYPES(r.scalar_type(), "logspace_cpu", [&]() {
          Scalar scalar_base = static_cast<Scalar>(base);
          Scalar scalar_start = start.to<Scalar>();
          Scalar scalar_end = end.to<Scalar>();
          Scalar *data_ptr = r.data_ptr<Scalar>();
          Scalar step = (scalar_end - scalar_start) / static_cast<Scalar>(steps - 1);
          const i64 halfway = steps / 2;
          parallel_for(0, steps, internal::GRAIN_SIZE, [&](i64 p_begin, i64 p_end) {
            Scalar is = static_cast<Scalar>(p_begin);
            for (i64 i = p_begin; i < p_end; ++i, is+=1) { //complex does not support ++operator
              if (i < halfway) {
                data_ptr[i] = pow(scalar_base, scalar_start + step*is);
              } else {
                data_ptr[i] = pow(scalar_base, scalar_end - (step * static_cast<Scalar>(steps - i - 1)));
              }
            }
          });
        });
      } else {
        AT_DISPATCH_ALL_TYPES_AND(kBFloat16, r.scalar_type(), "logspace_cpu", [&]() {
          double scalar_base = static_cast<double>(base); // will be autopromoted anyway
          Scalar scalar_start = start.to<Scalar>();
          Scalar scalar_end = end.to<Scalar>();
          Scalar *data_ptr = r.data_ptr<Scalar>();
          double step = static_cast<double>(scalar_end - scalar_start) / (steps - 1);
          const i64 halfway = steps / 2;
          parallel_for(0, steps, internal::GRAIN_SIZE, [&](i64 p_begin, i64 p_end) {
            for (i64 i=p_begin; i < p_end; i++) {
              if (i < halfway) {
                data_ptr[i] = pow(scalar_base, scalar_start + step*i);
              } else {
                data_ptr[i] = pow(scalar_base, scalar_end - step * (steps - i - 1));
              }
            }
          });
        });
      }

      if (!result.is_contiguous()) {
        result.copy_(r);
      }
      return result;
        */
}


pub fn range_cpu_out(
        start:  &Scalar,
        end:    &Scalar,
        step:   &Scalar,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES(result.scalar_type(), "range_cpu", [&]() {
        using accscalar_t = acc_type<Scalar, false>;
        auto xstart = start.to<accscalar_t>();
        auto xend = end.to<accscalar_t>();
        auto xstep = step.to<accscalar_t>();

        TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
        TORCH_CHECK(isfinite(static_cast<double>(xstart)) &&
                 isfinite(static_cast<double>(xend)),
                 "unsupported range: ", xstart, " -> ", xend);
        TORCH_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
                 "upper bound and larger bound inconsistent with step sign");
        i64 size = static_cast<i64>(((xend - xstart) / xstep) + 1);
        if (result.numel() != size) {
          result.resize_({size});
        }
        Tensor r = result.is_contiguous() ? result : result.contiguous();
        Scalar *data_ptr = r.data_ptr<Scalar>();

        parallel_for(0, size, internal::GRAIN_SIZE, [&](i64 p_begin, i64 p_end) {
          Scalar is = p_begin;
          for (i64 i = p_begin; i < p_end; ++i, ++is) {
            data_ptr[i] = xstart + is * xstep;
          }
        });
        if (!result.is_contiguous()) {
          result.copy_(r);
        }
      });

      return result;
        */
}


pub fn arange_cpu_out(
        start:  &Scalar,
        end:    &Scalar,
        step:   &Scalar,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES(result.scalar_type(), "arange_cpu", [&]() {
        using accscalar_t = acc_type<Scalar, false>;
        auto xstart = start.to<accscalar_t>();
        auto xend = end.to<accscalar_t>();
        auto xstep = step.to<accscalar_t>();

        // we use double precision for (start - end) / step
        // to compute size_d for consistency across devices.
        // The problem with using accscalar_t is that accscalar_t might be float32 on gpu for a float32 Scalar,
        // but double on cpu for the same,
        // and the effective output size starts differing on CPU vs GPU because of precision issues, which
        // we dont want.
        // the corner-case we do want to take into account is i64, which has higher precision than double
        double size_d;
        if (is_same<Scalar, i64>::value) {
          size_d = ceil(static_cast<double>(end.to<accscalar_t>() - start.to<accscalar_t>())
                             / step.to<accscalar_t>());
        } else {
          size_d = ceil(static_cast<double>(end.to<double>() - start.to<double>())
                             / step.to<double>());
        }

        TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
        TORCH_CHECK(isfinite(static_cast<double>(xstart)) &&
                 isfinite(static_cast<double>(xend)),
                 "unsupported range: ", xstart, " -> ", xend);
        TORCH_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
                 "upper bound and larger bound inconsistent with step sign");

        TORCH_CHECK(size_d >= 0 && size_d <= static_cast<double>(i64::max),
                 "invalid size, possible overflow?");

        i64 size = static_cast<i64>(size_d);
        i64 numel = result.numel();

        if (numel != size) {
          if(numel > 0){
            TORCH_WARN("The number of elements in the out tensor of shape ", result.sizes(),
                        " is ", numel, " which does not match the computed number of elements ", size,
                        ". Note that this may occur as a result of rounding error. "
                        "The out tensor will be resized to a tensor of shape (", size, ",).");
          }
          result.resize_({size});
        }

        Tensor r = result.is_contiguous() ? result : result.contiguous();
        auto iter = TensorIterator::borrowing_nullary_op(r);
        arange_stub(iter.device_type(), iter, start, size, step);
        if (!result.is_contiguous()) {
          result.copy_(r);
        }
      });

      return result;
        */
}

define_dispatch!{arange_stub}
define_dispatch!{linspace_stub}
