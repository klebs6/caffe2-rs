crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/IndexKernel.cpp]

pub struct Indexer {
    num_indexers:     i64,
    indexers:         *mut *mut u8,
    indexer_strides:  *const i64,
    original_strides: *const i64,
    original_sizes:   *const i64,
}

impl Indexer {
    
    pub fn new(
        num_indexers:     i64,
        indexers:         *mut *mut u8,
        indexer_strides:  *const i64,
        original_sizes:   &[i32],
        original_strides: &[i32]) -> Self {
    
        todo!();
        /*


            : num_indexers(num_indexers)
        , indexers(indexers)
        , indexer_strides(indexer_strides)
        , original_strides(original_strides.data())
        , original_sizes(original_sizes.data()) 

        AT_ASSERT(original_strides.size() == num_indexers);
        AT_ASSERT(original_sizes.size() == num_indexers);
        */
    }
    
    pub fn get(&mut self, idx: i64) -> i64 {
        
        todo!();
        /*
            i64 offset = 0;
        for (int j = 0; j < num_indexers; j++) {
          i64 value = *(i64*)&indexers[j][idx * indexer_strides[j]];
          i64 size = original_sizes[j];
          TORCH_CHECK_INDEX(value >= -size && value < size,
                            "index ", value, " is out of bounds for dimension ", j, " with size ", size);
          if (value < 0) {
            value += size;
          }
          offset += value * original_strides[j];
        }
        return offset;
        */
    }
}

pub fn is_constant_index(
        ntensor: i32,
        strides: *const i64) -> bool {
    
    todo!();
        /*
            AT_ASSERT(ntensor >= 3);
      for (int arg = 2; arg < ntensor; arg++) {
        if (strides[arg] != 0) {
          return false;
        }
      }
      return true;
        */
}

pub fn cpu_index_kernel<Scalar, func_t>(
        iter:             &mut TensorIterator,
        index_size:       &[i32],
        index_stride:     &[i32],
        f:                &Func,
        serial_execution: bool)  {
    let serial_execution: bool =
                 serial_execution.unwrap_or(false);
    todo!();
        /*
            int ntensor = iter.ntensors();
      // When launch the index parallel version, set a relative samll grain size less than the INTERNAL::GRAIN_SIZE
      // to make the whole available thread numbers get more balanced work load and a better cache location.
      // The grain size here is chosen by the op benchmark to overcome the thread launch overhead
      const int index_parallel_grain_size = 3000;
      auto loop = [&](char** data, const i64* strides, i64 n) {
        auto indexer = Indexer(ntensor - 2, &data[2], &strides[2], index_size, index_stride);
        char* dst = data[0];
        char* src = data[1];
        if (is_constant_index(ntensor, strides)) {
          // specialization for when every element uses the same index
          i64 offset = indexer.get(0);
          if (strides[0] == sizeof(Scalar) && strides[1] == sizeof(Scalar)) {
            for (i64 i = 0; i < n; i++) {
              f(dst + strides[0] * i, src + strides[1] * i, offset);
            }
          } else {
            for (i64 i = 0; i < n; i++) {
              f(dst + strides[0] * i, src + strides[1] * i, offset);
            }
          }
        } else {
          for (i64 i = 0; i < n; i++) {
            i64 offset = indexer.get(i);
            f(dst + strides[0] * i, src + strides[1] * i, offset);
          }
        }
      };
      if (serial_execution) {
        iter.serial_for_each(loop, {0, iter.numel()});
      } else {
        iter.for_each(loop, index_parallel_grain_size);
      }
        */
}

pub fn index_kernel(
        iter:         &mut TensorIterator,
        index_size:   &[i32],
        index_stride: &[i32])  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
        iter.dtype(), "index_cpu", [&] {
        cpu_index_kernel<Scalar>(iter, index_size, index_stride, [](char* dst, char* src, i64 offset) {
          *(Scalar*)dst = *(Scalar*)(src + offset);
        });
      });
        */
}

/**
  | Given a linear index, returns the offset of the
  | tensor.
  |
  | Implements the same algorithm as its (legacy)
  | GPU version IndexToOffset OffsetCalculator
  | implements yet again the same algorithm but in
  | a column-major order
  |
  */
pub struct IndexToOffset {
    sizes:   &[i32],
    strides: &[i32],
    ndim:    i32,
}

impl IndexToOffset {
    
    pub fn new(tensor: &Tensor) -> Self {
    
        todo!();
        /*
        : sizes(tensor.sizes()),
        : strides(tensor.strides()),
        : ndim(tensor.dim()),

        
        */
    }
    
    pub fn get(&self, linear_index: i64) -> i64 {
        
        todo!();
        /*
            i64 offset = 0;
        for (int i = ndim - 1; i > 0; i--) {
          offset += (linear_index % sizes[i]) * strides[i];
          linear_index /= sizes[i];
        }
        return offset + linear_index * strides[0];
        */
    }
}

pub fn cpu_take_put_kernel<Scalar, func_t>(
        iter:             &mut TensorIterator,
        indexed:          &Tensor,
        f:                &Func,
        serial_execution: bool)  {

    let serial_execution: bool = serial_execution.unwrap_or(false);

    todo!();
        /*
            // This kernel follows the same strategy as `cpu_index_kernel`
      // Even though the indexed_tensor is const, we modify it through the data_ptr
      // This is a bit dirty, but otherwise it would be necessary to innecessarily add tensor
      // with zero strides to `iter` which would not be much better

      // When launch the parallel version, set a relative small grain size less than the INTERNAL::GRAIN_SIZE
      // to make the whole available thread numbers get more balanced work load and a better cache location.
      // The grain size here is chosen by the op benchmark to overcome the thread launch overhead
      // Perhaps tweak this number for `put_`? This number was tweaked for `index_put`
      constexpr int parallel_grain_size = 3000;
      const bool is_contiguous = indexed.is_contiguous();
      const auto numel = indexed.numel();
      const auto offset_indexed = IndexToOffset(indexed);

      auto* indexed_data = indexed.data_ptr<Scalar>();
      auto loop = [&](char** data, const i64* strides, i64 n) {
        auto* iterated_data_bytes = data[0];
        auto* index_data_bytes = data[1];
        for (i64 elem = 0; elem < n; ++elem) {
          auto idx = *reinterpret_cast<i64*>(index_data_bytes);
          auto& iterated = *reinterpret_cast<Scalar*>(iterated_data_bytes);

          TORCH_CHECK_INDEX(idx >= -numel && idx < numel,
                            "out of range: tried to access index ",
                            idx, " on a tensor of ", numel, " elements.");
          if (idx < 0) {
            idx += numel;
          }
          if (!is_contiguous) {
            idx = offset_indexed.get(idx);
          }
          f(iterated, indexed_data, idx);
          iterated_data_bytes += strides[0];
          index_data_bytes += strides[1];
        }
      };
      if (serial_execution) {
        iter.serial_for_each(loop, {0, iter.numel()});
      } else {
        iter.for_each(loop, parallel_grain_size);
      }
        */
}

pub fn put_kernel(
        iter:       &mut TensorIterator,
        self_:      &Tensor,
        accumulate: bool)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
        iter.dtype(), "take_put_cpu", [&] {
      // iter could be const, but for_each does not have a const version
        if (accumulate) {
          // nb. This deterministic issue the same as that of `index_put_kernel`
          // See Note [Enabling Deterministic Operations]
          // Parallel cpu_put_kernel with accumulation is nondeterministic, so we
          // must enable serial execution if deterministic algorithms are enabled.
          bool is_deterministic = globalContext().deterministicAlgorithms();
          bool use_parallel_for = (!is_deterministic) && (
            (iter.numel() >= internal::GRAIN_SIZE) && (get_num_threads() > 1));
          if (use_parallel_for && iter.dtype() == ScalarType::Float) {
            cpu_take_put_kernel<float>(iter, self,
                [](float& iterated, float* indexed, const i64 idx) {
                    cpu_atomic_add_float(indexed+idx, iterated);
                  });
          } else {
            // TODO: investigate parallelization of the accumulate kernel.
            // Unlike the non-accumulate case, this needs to be thread-safe.
            cpu_take_put_kernel<Scalar>(iter, self,
                [](Scalar& iterated, Scalar* indexed, const i64 idx) {
                    indexed[idx] += iterated;
                  },
                /*serial_execution=*/true);
          }
        } else {
          cpu_take_put_kernel<Scalar>(iter, self,
              [](Scalar& iterated, Scalar* indexed, const i64 idx) {
                  indexed[idx] = iterated;
                });
        }
      });
        */
}

pub fn take_kernel(
        iter:  &mut TensorIterator,
        input: &Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
        iter.dtype(), "take_cpu", [&] {
          cpu_take_put_kernel<Scalar>(iter, input,
              [](Scalar& iterated, Scalar* indexed, const i64 idx) {
                  iterated = indexed[idx];
                });
        });
        */
}

pub fn index_put_kernel(
        iter:         &mut TensorIterator,
        index_size:   &[i32],
        index_stride: &[i32],
        accumulate:   bool)  {
    
    todo!();
        /*
            // NOTE: duplicate indices are only supported if accumulate is true.
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
        iter.dtype(), "index_put", [&] {
        // See Note [Enabling Deterministic Operations]
        // Parallel cpu_index_kernel with accumulation is nondeterministic, so we
        // must enable serial execution if deterministic algorithms are enabled.
        const bool is_deterministic = globalContext().deterministicAlgorithms();
        if (accumulate) {
          bool use_parallel_for = (!is_deterministic) && (
            (iter.numel() >= internal::GRAIN_SIZE) && (get_num_threads() > 1));
          if (use_parallel_for && iter.dtype() == ScalarType::Float) {
            cpu_index_kernel<float>(iter, index_size, index_stride, [](char* dst, char* src, i64 offset) {
              cpu_atomic_add_float((float*)(dst + offset), *(float*)src);
            });
          } else {
            // TODO: investigate parallelization of the accumulate kernel. Unlike the non-accumulate case,
            // this needs to be thread-safe.
            cpu_index_kernel<Scalar>(iter, index_size, index_stride, [](char* dst, char* src, i64 offset) {
              *(Scalar*)(dst + offset) += *(Scalar*)src;
            }, /*serial_execution=*/true);
          }
        } else {
          cpu_index_kernel<Scalar>(iter, index_size, index_stride, [](char* dst, char* src, i64 offset) {
            *(Scalar*)(dst + offset) = *(Scalar*)src;
          }, /*serial_execution=*/is_deterministic);
        }
      });
        */
}

pub fn index_fill_kernel(
        iter:            &mut TensorIterator,
        dim:             i64,
        self_dim_size:   i64,
        self_dim_stride: i64,
        source:          &Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
        iter.dtype(), "index_fill_cpu", [&] {
        auto fill_val = source.to<Scalar>();
        auto handle_nonzero_idx_stride = [&](char** data, const i64* strides, i64 n) {
          auto* self_data_bytes = data[0];
          auto* index_data_bytes = data[1];
          for (i64 elem = 0; elem < n; ++elem) {
            auto* self_data = reinterpret_cast<Scalar*>(self_data_bytes);
            auto idx = *reinterpret_cast<i64*>(index_data_bytes);
            TORCH_CHECK_INDEX(idx >= -self_dim_size && idx < self_dim_size,
                              "index ", idx, " is out of bounds for dimension ",
                              dim, " with size ", self_dim_size);
            if (idx < 0) {
              idx += self_dim_size;
            }

            self_data[idx * self_dim_stride] = fill_val;

            self_data_bytes += strides[0];
            index_data_bytes += strides[1];
          }
        };
        auto handle_zero_idx_stride = [&](char** data, const i64* strides, i64 n) {
          auto* self_data_bytes = data[0];
          auto* index_data_bytes = data[1];
          auto idx = *reinterpret_cast<i64*>(index_data_bytes);
          TORCH_CHECK_INDEX(idx >= -self_dim_size && idx < self_dim_size,
                            "index ", idx, " is out of bounds for dimension ",
                            dim, " with size ", self_dim_size);
          if (idx < 0) {
            idx += self_dim_size;
          }
          for (i64 elem = 0; elem < n; ++elem) {
            auto* self_data = reinterpret_cast<Scalar*>(self_data_bytes);

            self_data[idx * self_dim_stride] = fill_val;

            self_data_bytes += strides[0];
          }
        };

        auto loop = [&](char** data, const i64* strides, i64 n) {
          auto idx_stride = strides[1];
          if (idx_stride) {
            handle_nonzero_idx_stride(data, strides, n);
          }
          else {
            handle_zero_idx_stride(data, strides, n);
          }
        };
        iter.for_each(loop);
      });
        */
}

pub fn index_copy_kernel(
        iter:            &mut TensorIterator,
        dim:             i64,
        self_dim_size:   i64,
        self_dim_stride: i64)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
        iter.dtype(), "index_copy_cpu", [&] {
        auto handle_nonzero_idx_stride = [&](char** data, const i64* strides, i64 n) {
          auto* self_data_bytes = data[0];
          auto* index_data_bytes = data[1];
          auto* source_data_bytes = data[2];
          for (i64 elem = 0; elem < n; ++elem) {
            auto* self_data = reinterpret_cast<Scalar*>(self_data_bytes);
            auto idx = *reinterpret_cast<i64*>(index_data_bytes);
            auto* source_data = reinterpret_cast<Scalar*>(source_data_bytes);
            TORCH_CHECK_INDEX(idx >= 0 && idx < self_dim_size,
                  "index_copy_(): index ", idx, " is out of bounds for dimension ",
                  dim, " with size ", self_dim_size);

            self_data[idx * self_dim_stride] = *source_data;

            self_data_bytes += strides[0];
            index_data_bytes += strides[1];
            source_data_bytes += strides[2];
          }
        };
        auto handle_zero_idx_stride = [&](char** data, const i64* strides, i64 n) {
          auto* self_data_bytes = data[0];
          auto* index_data_bytes = data[1];
          auto* source_data_bytes = data[2];
          auto idx = *reinterpret_cast<i64*>(index_data_bytes);
          TORCH_CHECK_INDEX(idx >= 0 && idx < self_dim_size,
                "index_copy_(): index ", idx, " is out of bounds for dimension ",
                dim, " with size ", self_dim_size);
          for (i64 elem = 0; elem < n; ++elem) {
            auto* self_data = reinterpret_cast<Scalar*>(self_data_bytes);
            auto* source_data = reinterpret_cast<Scalar*>(source_data_bytes);

            self_data[idx * self_dim_stride] = *source_data;

            self_data_bytes += strides[0];
            source_data_bytes += strides[2];
          }
        };

        auto loop = [&](char** data, const i64* strides, i64 n) {
          auto idx_stride = strides[1];
          if (idx_stride) {
            handle_nonzero_idx_stride(data, strides, n);
          }
          else {
            handle_zero_idx_stride(data, strides, n);
          }
        };
        bool is_deterministic = globalContext().deterministicAlgorithms();
        if (is_deterministic) {
          iter.serial_for_each(loop, {0, iter.numel()});
        } else {
          iter.for_each(loop);
        }
      });
        */
}

pub fn cpu_masked_fill_kernel<Scalar, mask_t>(
        iter:  &mut TensorIterator,
        value: Scalar)  {

    todo!();
        /*
            auto is_mask_bool = is_same<mask_t, bool>::value;
      auto loop = [&](char** data, const i64* strides, i64 n) {
        char* dst = data[0];
        char* mask = data[1];
        for (i64 i = 0; i < n; i++) {
          mask_t mask_value = *(mask_t*)(mask + strides[1] * i);
          if (!is_mask_bool) {
            TORCH_CHECK(mask_value == 0 || mask_value == 1, "Mask tensor can take 0 and 1 values only");
          }
          if (mask_value) {
            *(Scalar*)(dst + strides[0] * i) = value;
          }
        }
      };
      iter.for_each(loop);
        */
}

pub fn masked_fill_kernel(
        iter:  &mut TensorIterator,
        value: &Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
        iter.dtype(), "masked_fill", [&] {
          Scalar scalar_val = value.to<Scalar>();
          auto mask_dtype = iter.input_dtype(0);
          if (mask_dtype == ScalarType::Bool) {
            cpu_masked_fill_kernel<Scalar, bool>(iter, scalar_val);
          } else {
            cpu_masked_fill_kernel<Scalar, unsigned char>(iter, scalar_val);
          }
        });
        */
}

pub fn cpu_masked_scatter_kernel<Scalar, mask_t>(
        iter:   &mut TensorIterator,
        source: &Tensor)  {

    todo!();
        /*
            auto is_mask_bool = is_same<mask_t, bool>::value;
      ptrdiff_t source_cntr = 0;
      Scalar* source_ptr = source.data_ptr<Scalar>();
      auto numel = source.numel();

      auto loop = [&](char** data, const i64* strides, i64 n) {
        char* dst = data[0];
        const i64 dst_stride = strides[0];
        char* mask = data[1];
        const i64 mask_stride = strides[1];
        for (i64 i = 0; i < n; i++) {
          mask_t mask_value = *(mask_t*)(mask + mask_stride * i);
          if (!is_mask_bool) {
            TORCH_CHECK(mask_value <= static_cast<mask_t>(1), "Mask tensor can take 0 and 1 values only");
          }
          if (mask_value) {
            TORCH_CHECK(source_cntr < numel, "Number of elements of source < number of ones in mask");
            *(Scalar*)(dst + dst_stride * i) = *(source_ptr);
            source_ptr++;
            source_cntr++;
          }
        }
      };
      iter.serial_for_each(loop, {0, iter.numel()});
        */
}

pub fn masked_scatter_kernel(
        iter:   &mut TensorIterator,
        source: &Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
          ScalarType::Bool,
          ScalarType::BFloat16,
          ScalarType::Half,
          iter.dtype(),
          "masked_scatter",
          [&] {
            auto mask_dtype = iter.input_dtype(0);
            if (mask_dtype == ScalarType::Bool) {
              cpu_masked_scatter_kernel<Scalar, bool>(iter, source);
            } else {
              cpu_masked_scatter_kernel<Scalar, unsigned char>(iter, source);
            }
          });
        */
}

pub fn cpu_masked_select_serial_kernel<Scalar, mask_t, func_t>(
        iter: &mut TensorIterator,
        f:    &Func)  {

    todo!();
        /*
            auto is_mask_bool = is_same<mask_t, bool>::value;
      i64 offset = 0;
      auto loop = [&](char** data, const i64* strides, i64 n) {
        char* dst = data[0];
        char* src = data[1];
        char* mask = data[2];
        for (i64 i = 0; i < n; i++) {
          mask_t mask_value = *(mask_t*)(mask + strides[2] * i);
          if (!is_mask_bool) {
            TORCH_CHECK(mask_value == 0 || mask_value == 1, "Mask tensor can take 0 and 1 values only");
          }
          if (mask_value) {
            i64 offset_bytes = offset * sizeof(Scalar);
            f(dst, src + strides[1] * i, offset_bytes);
            offset++;
          }
        }
      };
      iter.serial_for_each(loop, {0, iter.numel()});
        */
}

pub fn masked_select_serial_kernel(
        iter:          &mut TensorIterator,
        result_stride: i64)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
        iter.dtype(), "masked_select", [&] {
          auto mask_dtype = iter.input_dtype(1);
          if (mask_dtype == ScalarType::Bool) {
            cpu_masked_select_serial_kernel<Scalar, bool>(iter, [result_stride](char* dst, char* src, i64 offset) {
              *(Scalar*)(dst + offset*result_stride) = *(Scalar*)src;
            });
          } else {
            cpu_masked_select_serial_kernel<Scalar, unsigned char>(iter, [result_stride](char* dst, char* src, i64 offset) {
              *(Scalar*)(dst + offset*result_stride) = *(Scalar*)src;
            });
          }
        });
        */
}

pub fn cpu_masked_select_kernel<Scalar, mask_t, func_t>(
        iter: &mut TensorIterator,
        f:    &Func)  {

    todo!();
        /*
            auto is_mask_bool = is_same<mask_t, bool>::value;
      auto loop = [&](char** data, const i64* strides, i64 n) {
        char* dst = data[0];
        char* src = data[1];
        char* mask = data[2];
        char* mask_prefix_sum = data[3];
        for (i64 i = 0; i < n; i++) {
          mask_t mask_value = *(mask_t*)(mask + strides[2] * i);
          if (!is_mask_bool) {
            TORCH_CHECK(mask_value == 0 || mask_value == 1, "Mask tensor can take 0 and 1 values only");
          }
          if (mask_value) {
            i64 offset = *(i64*)(mask_prefix_sum + strides[3] * i);
            i64 offset_bytes = (offset - 1) * sizeof(Scalar);
            f(dst, src + strides[1] * i, offset_bytes);
          }
        }
      };
      iter.for_each(loop);
        */
}

pub fn masked_select_kernel(
        iter:          &mut TensorIterator,
        result_stride: i64)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
        iter.dtype(), "masked_select", [&] {
          auto mask_dtype = iter.input_dtype(1);
          if (mask_dtype == ScalarType::Bool) {
            cpu_masked_select_kernel<Scalar, bool>(iter, [result_stride](char* dst, char* src, i64 offset) {
              *(Scalar*)(dst + offset*result_stride) = *(Scalar*)src;
            });
          } else {
            cpu_masked_select_kernel<Scalar, unsigned char>(iter, [result_stride](char* dst, char* src, i64 offset) {
              *(Scalar*)(dst + offset*result_stride) = *(Scalar*)src;
            });
          }
        });
        */
}

pub fn flip_kernel(
        iter:      &mut TensorIterator,
        quantized: bool)  {
    
    todo!();
        /*
            if (quantized) {
        AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(iter.dtype(), "flip_quantized_cpu",
            [&iter] { cpu_kernel(iter,
              [](Scalar a, Scalar /*dummy input*/) -> Scalar {
                return a;
            });
        });
      } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kHalf, kBFloat16, iter.dtype(), "flip_cpu",
            [&iter] { cpu_kernel_vec(iter,
              [](Scalar a, Scalar /*dummy input*/) -> Scalar {
                return a;
            },
              [](Vectorized<Scalar> a, Vectorized<Scalar> /*dummy input*/) -> Vectorized<Scalar> {
                return a;
            });
        });
      }
        */
}

register_dispatch!{index_stub                , &index_kernel}
register_dispatch!{index_fill_stub           , &index_fill_kernel}
register_dispatch!{index_copy_stub           , &index_copy_kernel}
register_dispatch!{index_put_stub            , &index_put_kernel}
register_dispatch!{put_stub                  , &put_kernel}
register_dispatch!{take_stub                 , &take_kernel}
register_dispatch!{masked_fill_stub          , &masked_fill_kernel}
register_dispatch!{masked_select_serial_stub , &masked_select_serial_kernel}
register_dispatch!{masked_select_stub        , &masked_select_kernel}
register_dispatch!{masked_scatter_stub       , &masked_scatter_kernel}
register_dispatch!{flip_stub                 , &flip_kernel}
