crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/RangeFactoriesKernel.cpp]

pub fn arange_kernel(
        iter:         &mut TensorIterator,
        scalar_start: &Scalar,
        scalar_steps: &Scalar,
        scalar_step:  &Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES(iter.dtype(), "arange_cpu", [&]() {
        using accscalar_t = acc_type<Scalar, false>;
        auto start = scalar_start.to<accscalar_t>();
        auto steps = scalar_steps.to<accscalar_t>();
        auto step = scalar_step.to<accscalar_t>();
        parallel_for(0, steps, internal::GRAIN_SIZE, [&](i64 p_begin, i64 p_end) {
          i64 idx(p_begin);
          TensorIterator it(iter);
          cpu_serial_kernel_vec(
              it,
              [start, step, &idx]() -> Scalar {
                return start + step * (idx++);
              },
              [start, step, &idx]() -> Vectorized<Scalar> {
                Vectorized<Scalar> res;
                res = Vectorized<Scalar>::arange(start + step * idx, step);
                idx += Vectorized<Scalar>::size();
                return res;
              }, {p_begin, p_end});
        });
      });
        */
}

pub fn linspace_kernel(
        iter:         &mut TensorIterator,
        scalar_start: &Scalar,
        scalar_end:   &Scalar,
        steps:        i64)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, iter.dtype(), "linspace_cpu", [&]() {
        // step should be of double type for all integral types
        using step_t = conditional_t<is_integral<Scalar>::value, double, Scalar>;
        const Scalar start = scalar_start.to<Scalar>();
        const Scalar end = scalar_end.to<Scalar>();
        // Cast `end` and `start` to `step_t`, since range can be larger than Scalar for integral types
        const step_t step = (static_cast<step_t>(end) - static_cast<step_t>(start)) / (steps - 1);
        i64 halfway = steps / 2;
        parallel_for(0, steps, internal::GRAIN_SIZE, [&](i64 p_begin, i64 p_end) {
          i64 idx(p_begin);
          TensorIterator it(iter);
          cpu_serial_kernel_vec(
              it,
              [start, end, step, halfway, steps, &idx]() -> Scalar {
                if (idx < halfway) {
                  return start + step * (idx++);
                } else {
                  return end - step * (steps - (idx++) - 1);
                }
              },
              [start, end, step, halfway, steps, &idx]() -> Vectorized<Scalar> {
                Vectorized<Scalar> res;
                if (idx < halfway) {
                  res = Vectorized<Scalar>::arange(start + step * idx, step);
                } else {
                  res = Vectorized<Scalar>::arange(
                      end - step * (steps - idx - 1), step);
                }
                idx += Vectorized<Scalar>::size();
                return res;
              }, {p_begin, p_end});
        });
      });
        */
}

register_dispatch!{arange_stub   , &arange_kernel}
register_dispatch!{linspace_stub , &linspace_kernel}
