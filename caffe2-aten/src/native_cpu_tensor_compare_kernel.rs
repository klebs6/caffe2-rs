crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/TensorCompareKernel.cpp]

#[inline] pub fn compare_base_kernel_core<Scalar, Scalar2, Loop1d>(
    result1: &mut Tensor,
    result2: &mut Tensor,
    self_:   &Tensor,
    dim:     i64,
    keepdim: bool,
    loop_:   &Loop1d)  {

    todo!();
        /*
            auto self_sizes = ensure_nonempty_vec(self.sizes().vec());
      self_sizes[dim] = 1;

      // result1 and result2 may be a empty tensor, if not,
      // reshape them as self dims
      if (!keepdim) {
        if (result1.ndimension() >= dim) {
          result1.unsqueeze_(dim);
        }
        if (result2.ndimension() >= dim) {
          result2.unsqueeze_(dim);
        }
      }

      native::resize_output(result1, self_sizes);
      native::resize_output(result2, self_sizes);

      auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .declare_static_shape(self.sizes(), /*squash_dims=*/dim)
        .add_output(result1)
        .add_output(result2)
        .add_input(self)
        .build();

      iter.for_each(loop, /* grain_size */ 1);

      if (!keepdim) {
        result1.squeeze_(dim);
        result2.squeeze_(dim);
      }
        */
}

//Scalar2 = i64,
#[inline] pub fn compare_base_kernel<Scalar, Scalar2, func_t>(
    result1: &mut Tensor,
    result2: &mut Tensor,
    self_:   &Tensor,
    dim:     i64,
    keepdim: bool,
    f:       &Func)  {

    todo!();
    /*
      auto self_dim_stride = ensure_nonempty_stride(self, dim);

      auto loop = [&](char** data, const i64* strides, i64 n) {
        auto* result1_data_bytes = data[0];
        auto* result2_data_bytes = data[1];
        const auto* self_data_bytes = data[2];
        for (i64 i = 0; i < n; ++i) {
          f((Scalar*)result1_data_bytes,
            (scalar_t_2*)result2_data_bytes,
            (Scalar*)self_data_bytes,
            self_dim_stride);
          result1_data_bytes += strides[0];
          result2_data_bytes += strides[1];
          self_data_bytes += strides[2];
        }
      };

      compare_base_kernel_core<Scalar, scalar_t_2>(
          result1, result2, self, dim, keepdim, loop);
        */
}

pub fn min_kernel_impl(
    result:  &mut Tensor,
    indice:  &mut Tensor,
    self_:   &Tensor,
    dim:     i64,
    keepdim: bool)  {

    todo!();
        /*
            auto wrap_dim = maybe_wrap_dim(dim, self.dim());
      i64 self_dim_size = ensure_nonempty_size(self, wrap_dim);

      TORCH_CHECK(result.scalar_type() == self.scalar_type() && indice.scalar_type() == kLong,
        "Expect dtype ", self.scalar_type(), "and torch.long, but got ", result.scalar_type(), "and", indice.scalar_type());

      AT_DISPATCH_ALL_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, self.scalar_type(), "min_cpu", [&] {
        compare_base_kernel<Scalar>(result, indice, self, wrap_dim, keepdim, [&] (
          Scalar* result_data, i64* indice_data,
          const Scalar* self_data, auto self_dim_stride) {
            using Value = typename scalar_Valueype<Scalar>::type;
            Value (*zabs_)(Scalar) = zabs<Scalar, Value>;
            Scalar min_number = self_data[0];
            i64 index = 0;
            for (i64 i = 0; i < self_dim_size; ++i) {
              Scalar value = self_data[i * self_dim_stride];
              if (!(zabs_(value) >= zabs_(min_number))) {
                min_number = value;
                index = i;
                if (_isnan<Scalar>(value)) {
                  break;
                }
              }
            }
            *result_data = min_number;
            *indice_data = index;
          }
        );
      });
        */
}

pub fn max_kernel_impl(
    result:  &mut Tensor,
    indice:  &mut Tensor,
    self_:   &Tensor,
    dim:     i64,
    keepdim: bool)  {

    todo!();
        /*
            auto wrap_dim = maybe_wrap_dim(dim, self.dim());
      i64 self_dim_size = ensure_nonempty_size(self, wrap_dim);

      TORCH_CHECK(result.scalar_type() == self.scalar_type() && indice.scalar_type() == kLong,
        "Expect dtype ", self.scalar_type(), "and torch.long, but got ", result.scalar_type(), "and", indice.scalar_type());

      AT_DISPATCH_ALL_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, self.scalar_type(), "max_cpu", [&] {
        compare_base_kernel<Scalar>(result, indice, self, wrap_dim, keepdim, [&] (
          Scalar* result_data, i64* indice_data,
          const Scalar* self_data, auto self_dim_stride) {
            using Value = typename scalar_Valueype<Scalar>::type;
            Value (*zabs_)(Scalar) = zabs<Scalar, Value>;
            Scalar max_number = self_data[0];
            i64 index = 0;
            for (i64 i = 0; i < self_dim_size; ++i) {
              Scalar value = self_data[i * self_dim_stride];
              if (!(zabs_(value) <= zabs_(max_number))) {
                max_number = value;
                index = i;
                if (_isnan<Scalar>(value)) {
                  break;
                }
              }
            }
            *result_data = max_number;
            *indice_data = index;
          }
        );
      });
        */
}

pub fn aminmax_kernel_impl(
    min_result: &mut Tensor,
    max_result: &mut Tensor,
    self_:      &Tensor,
    dim:        i64,
    keepdim:    bool)  {
    
    todo!();
        /*
            auto wrap_dim = maybe_wrap_dim(dim, self.dim());
      i64 self_dim_size = ensure_nonempty_size(self, wrap_dim);

      TORCH_CHECK(min_result.scalar_type() == self.scalar_type() && max_result.scalar_type() == self.scalar_type(),
        "Expect min and max dtype ", self.scalar_type(),
        " but got ", min_result.scalar_type(), " and ", max_result.scalar_type());

      AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, self.scalar_type(), "_aminmax_cpu", [&] {
        compare_base_kernel<Scalar, Scalar>(min_result, max_result, self, wrap_dim, keepdim, [&] (
          Scalar* min_result_data, Scalar* max_result_data,
          const Scalar* self_data, auto self_dim_stride) {
            Scalar min_number = self_data[0];
            Scalar max_number = self_data[0];
            for (i64 i = 0; i < self_dim_size; ++i) {
              Scalar value = self_data[i * self_dim_stride];
              // note: comparison is written this way to handle NaN correctly
              if (!(value >= min_number)) {
                min_number = value;
                if (_isnan<Scalar>(value)) {
                  max_number = value;
                  break;
                }
              } else if (!(value <= max_number)) {
                max_number = value;
              }
            }
            *min_result_data = min_number;
            *max_result_data = max_number;
          }
        );
      });
        */
}

pub fn where_kernel_impl(
    iter:           &mut TensorIterator,
    condition_type: ScalarType)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool,
        iter.dtype(), "where_cpu", [&] {
        if (condition_type == ScalarType::Byte) {
          cpu_kernel(
            iter,
            [=](u8 cond_val, Scalar self_val, Scalar other_val) -> Scalar {
              return cond_val ? self_val : other_val;
            });
        } else {
          cpu_kernel(
            iter,
            [=](bool cond_val, Scalar self_val, Scalar other_val) -> Scalar {
              return cond_val ? self_val : other_val;
            });
        }
      });
        */
}

pub fn isposinf_kernel_impl(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.input_dtype(), "isposinf_cpu", [&]() {
        cpu_kernel(iter, [](Scalar a) -> bool { return a == numeric_limits<Scalar>::infinity(); });
      });
        */
}

pub fn isneginf_kernel_impl(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.input_dtype(), "isneginf_cpu", [&]() {
        cpu_kernel(iter, [](Scalar a) -> bool { return a == -numeric_limits<Scalar>::infinity(); });
      });
        */
}

pub fn mode_kernel_impl(
        values:  &mut Tensor,
        indices: &mut Tensor,
        self_:   &Tensor,
        dim:     i64,
        keepdim: bool)  {
    
    todo!();
        /*
            auto self_dim_size = ensure_nonempty_size(self, dim);
      auto self_dim_stride = ensure_nonempty_stride(self, dim);

      AT_DISPATCH_ALL_TYPES_AND3(
          kHalf, kBFloat16, kBool, self.scalar_type(), "mode_cpu", [&] {
            auto loop = [&](char** data, const i64* strides, i64 n) {
              auto* values_data_bytes = data[0];
              auto* indices_data_bytes = data[1];
              const auto* self_data_bytes = data[2];

              vector<pair<Scalar, i64>> elements(self_dim_size);

              for (i64 k = 0; k < n; ++k) {
                Scalar* values_data = (Scalar*)values_data_bytes;
                i64* indices_data = (i64*)indices_data_bytes;
                const Scalar* self_data = (Scalar*)self_data_bytes;

                Scalar mode = 0;
                i64 modei = 0;
                i64 temp_freq = 0;
                i64 max_freq = 0;

                for (i64 i = 0; i < self_dim_size; i++) {
                  elements[i] = make_pair(self_data[i * self_dim_stride], i);
                }

                // Even though, theoretically, we don't need to specify this lambda
                // (it's basically the same as less), doing so degrades
                // performance. That is because its implementation for pair
                // uses 3 comparisons.
                sort(
                    elements.begin(),
                    elements.end(),
                    [=](const auto& i, const auto& j) {
                      return i.first < j.first;
                    });

                for (i64 i = 0; i < self_dim_size; i++) {
                  temp_freq++;
                  if ((i == self_dim_size - 1) ||
                      (elements[i].first != elements[i + 1].first)) {
                    if (temp_freq > max_freq) {
                      mode = elements[i].first;
                      modei = elements[i].second;
                      max_freq = temp_freq;
                    }
                    temp_freq = 0;
                  }
                }

                *values_data = mode;
                *indices_data = modei;

                values_data_bytes += strides[0];
                indices_data_bytes += strides[1];
                self_data_bytes += strides[2];
              }
            };

            compare_base_kernel_core<Scalar>(
                values, indices, self, dim, keepdim, loop);
          });
        */
}

/**
  | Default brute force implementation of
  | isin(). Used when the number of test elements
  | is small.
  |
  | Iterates through each element and checks it
  | against each test element.
  |
  */
pub fn isin_default_kernel_cpu(
        elements:      &Tensor,
        test_elements: &Tensor,
        invert:        bool,
        out:           &Tensor)  {
    
    todo!();
        /*
            // Since test elements is not an input of the TensorIterator, type promotion
      // must be done manually.
      ScalarType common_type = result_type(elements, test_elements);
      Tensor test_elements_flat = test_elements.to(common_type).ravel();
      Tensor promoted_elements = elements.to(common_type);
      auto iter = TensorIteratorConfig()
        .add_output(out)
        .add_input(promoted_elements)
        .check_all_same_dtype(false)
        .build();
      // Dispatch based on promoted type.
      AT_DISPATCH_ALL_TYPES(iter.dtype(1), "isin_default_cpu", [&]() {
        cpu_kernel(iter, [&](Scalar element_val) -> bool {
          const auto* test_element_data = reinterpret_cast<Scalar*>(test_elements_flat.data_ptr());
          for (auto j = 0; j < test_elements_flat.numel(); ++j) {
            if (element_val == test_element_data[j]) {
              return !invert;
            }
          }
          return invert;
        });
      });
        */
}

pub fn clamp_kernel_impl(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.common_dtype(), "clamp_cpu", [&]() {
        cpu_kernel_vec(iter,
          [](Scalar a, Scalar min, Scalar max) -> Scalar {
            return min(max(a, min), max);
          },
          [](Vectorized<Scalar> a, Vectorized<Scalar> min, Vectorized<Scalar> max) {
            return vec::clamp(a, min, max);
          });
      });
        */
}

pub fn clamp_scalar_kernel_impl(
        iter: &mut TensorIterator,
        min:  Scalar,
        max:  Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.common_dtype(), "clamp_scalar_cpu", [&]() {
        const auto min = min_.to<Scalar>();
        const auto max = max_.to<Scalar>();
        const Vectorized<Scalar> min_vec(min);
        const Vectorized<Scalar> max_vec(max);
        cpu_kernel_vec(iter,
          [=](Scalar a) -> Scalar {
            return min(max(a, min), max);
          },
          [=](Vectorized<Scalar> a) {
            return vec::clamp(a, min_vec, max_vec);
          });
      });
        */
}

pub fn clamp_max_kernel_impl(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.common_dtype(), "clamp_max_cpu", [&]() {
        cpu_kernel_vec(iter,
          [](Scalar a, Scalar max) -> Scalar {
            return min(a, max);
          },
          [](Vectorized<Scalar> a, Vectorized<Scalar> max) {
            return vec::clamp_max(a, max);
          });
      });
        */
}

pub fn clamp_max_scalar_kernel_impl(
        iter: &mut TensorIterator,
        max:  Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.common_dtype(), "clamp_max_scalar_cpu", [&]() {
        const auto max = max_.to<Scalar>();
        const Vectorized<Scalar> max_vec(max);
        cpu_kernel_vec(iter,
          [=](Scalar a) -> Scalar {
            return min(a, max);
          },
          [=](Vectorized<Scalar> a) {
            return vec::clamp_max(a, max_vec);
          });
      });
        */
}

pub fn clamp_min_kernel_impl(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.common_dtype(), "clamp_min_cpu", [&]() {
        cpu_kernel_vec(iter,
            [](Scalar a, Scalar min) -> Scalar {
              return max(a, min);
            },
            [](Vectorized<Scalar> a, Vectorized<Scalar> min) {
              return vec::clamp_min(a, min);
            });
      });
        */
}

pub fn clamp_min_scalar_kernel_impl(
        iter: &mut TensorIterator,
        min:  Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.common_dtype(), "clamp_min_cpu", [&]() {
        const auto min = min_.to<Scalar>();
        const Vectorized<Scalar> min_vec(min);
        cpu_kernel_vec(iter,
            [=](Scalar a) -> Scalar {
              return max(a, min);
            },
            [=](Vectorized<Scalar> a) {
              return vec::clamp_min(a, min_vec);
            });
      });
        */
}

register_dispatch!{max_stub              , &max_kernel_impl}
register_dispatch!{min_stub              , &min_kernel_impl}
register_dispatch!{_aminmax_stub         , &_aminmax_kernel_impl}
register_dispatch!{where_kernel          , &where_kernel_impl}
register_dispatch!{isposinf_stub         , &isposinf_kernel_impl}
register_dispatch!{isneginf_stub         , &isneginf_kernel_impl}
register_dispatch!{mode_stub             , &mode_kernel_impl}
register_dispatch!{clamp_stub            , &clamp_kernel_impl}
register_dispatch!{clamp_min_stub        , &clamp_min_kernel_impl}
register_dispatch!{clamp_max_stub        , &clamp_max_kernel_impl}
register_dispatch!{clamp_scalar_stub     , &clamp_scalar_kernel_impl}
register_dispatch!{clamp_min_scalar_stub , &clamp_min_scalar_kernel_impl}
register_dispatch!{clamp_max_scalar_stub , &clamp_max_scalar_kernel_impl}
register_dispatch!{isin_default_stub     , &isin_default_kernel_cpu}
