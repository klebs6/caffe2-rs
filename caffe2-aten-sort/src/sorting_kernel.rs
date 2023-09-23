crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/SortingKernel.cpp]

pub fn fill_indices(
    indices: &mut Tensor,
    dim:     i64)  {
    
    todo!();
        /*
            auto dim_size = indices.size(dim);
      auto idx_dim = at::arange(0, dim_size, indices.options().dtype(at::kLong));
      auto idx_dim_sizes = std::vector<i64>(indices.dim(), 1);
      auto idx_dim_strides = std::vector<i64>(indices.dim(), 0);
      idx_dim_sizes[dim] = dim_size;
      idx_dim_strides[dim] = 1;
      auto idx_dim_restrided = idx_dim.as_strided(idx_dim_sizes, idx_dim_strides);
      indices.copy_(idx_dim_restrided);
        */
}

pub fn dim_apply<func_t>(
    values:      &mut Tensor,
    indices:     &mut Tensor,
    dim:         i64,
    method_name: &String,
    f:           &Func)  {

    todo!();
        /*
            dim = maybe_wrap_dim(dim, values.dim());
      TORCH_CHECK(
        dim >= 0 && dim < values.dim(),
        method_name, "(): invalid dimension parameter ", dim
      );

      auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .declare_static_shape(values.sizes(), /*squash_dims=*/dim)
        .add_output(values)
        .add_output(indices)
        .build();

      auto values_dim_stride = values.stride(dim);
      auto indices_dim_stride = indices.stride(dim);
      auto dim_size = values.size(dim);

      AT_DISPATCH_ALL_TYPES_AND3(
        ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, iter.dtype(),
        "sorting_kernel_method_name", [&] {
          auto loop = [&](char** data, const i64* strides, i64 n) {
            auto* values_data_bytes = data[0];
            auto* indices_data_bytes = data[1];

            for (i64 i = 0; i < n; ++i) {
              f(
                reinterpret_cast<Scalar*>(values_data_bytes),
                values_dim_stride,
                reinterpret_cast<i64*>(indices_data_bytes),
                indices_dim_stride,
                dim_size
              );

              values_data_bytes += strides[0];
              indices_data_bytes += strides[1];
            }
          };

          iter.for_each(loop);
        }
      );
        */
}

pub struct KeyValueCompAsc<Scalar> {

}

impl KeyValueCompAsc<Scalar> {
    
    pub fn invoke<LHS, RHS>(&self, 
        lhs: LHS,
        rhs: RHS) -> bool {
    
        todo!();
        /*
            return (!_isnan<Scalar>(get<0>(lhs)) && _isnan<Scalar>(get<0>(rhs)))
          || (get<0>(lhs) < get<0>(rhs));
        */
    }
}

pub struct KeyValueCompDesc<Scalar> {

}

impl KeyValueCompDesc<Scalar> {
    
    pub fn invoke<LHS, RHS>(&self, 
        lhs: LHS,
        rhs: RHS) -> bool {
    
        todo!();
        /*
            return (_isnan<Scalar>(get<0>(lhs)) && !_isnan<Scalar>(get<0>(rhs)))
          || (get<0>(lhs) > get<0>(rhs));
        */
    }
}

pub fn sort_kernel(
        values:     &mut Tensor,
        indices:    &mut Tensor,
        dim:        i64,
        descending: bool,
        stable:     bool)  {
    
    todo!();
        /*
            dim = maybe_wrap_dim(dim, values.dim());
      _fill_indices(indices, dim);
      _dim_apply(
        values, indices, dim,
        "sort_cpu", [&](
          auto* values, i64 values_dim_stride,
          auto* indices, i64 indices_dim_stride,
          i64 dim_size
        ) {
          using Scalar = typename std::remove_pointer<decltype(values)>::type;
          auto values_accessor = StridedRandomAccessor<Scalar>(
            values, values_dim_stride);
          auto indices_accessor = StridedRandomAccessor<i64>(
            indices, indices_dim_stride);
          auto composite_accessor = CompositeRandomAccessorCPU<
            decltype(values_accessor), decltype(indices_accessor)
          >(values_accessor, indices_accessor);

          if (descending) {
            if (stable) {
              std::stable_sort(composite_accessor, composite_accessor + dim_size,
                KeyValueCompDesc<Scalar>());
            }
            else {
              std::sort(composite_accessor, composite_accessor + dim_size,
                KeyValueCompDesc<Scalar>());
            }
          }
          else {
            if (stable) {
              std::stable_sort(composite_accessor, composite_accessor + dim_size,
                KeyValueCompAsc<Scalar>());
            }
            else {
              std::sort(composite_accessor, composite_accessor + dim_size,
                KeyValueCompAsc<Scalar>());
            }
          }
        }
      );
        */
}

pub fn topk_kernel(
        values:  &Tensor,
        indices: &Tensor,
        self_:   &Tensor,
        k:       i64,
        dim:     i64,
        largest: bool,
        sorted:  bool)  {
    
    todo!();
        /*
            auto sizes = self.sizes();
      auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .declare_static_shape(sizes, /*squash_dims=*/dim)
        .add_output(values)
        .add_output(indices)
        .add_input(self)
        .build();

      auto mode_values_stride = values.strides()[dim];
      auto mode_indices_stride = indices.strides()[dim];
      auto tmp_values_stride = self.strides()[dim];

      AT_DISPATCH_ALL_TYPES(self.scalar_type(), "topk_cpu", [&] {
        auto loop = [&](char** data, const i64* strides, i64 n) {
          return topk_impl_loop<Scalar>(
              mode_values_stride, mode_indices_stride, tmp_values_stride,
              k, sizes[dim], largest, sorted, data, strides, n);
        };

        i64 grain_size = internal::GRAIN_SIZE / std::max(i64{1}, sizes[dim]);
        iter.for_each(loop, /*grain_size=*/grain_size);
      });
        */
}

register_dispatch!{sort_stub, &sort_kernel}
register_dispatch!{topk_stub, &topk_kernel}
