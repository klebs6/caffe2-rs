crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/SortingUtils.h]

/**
  | ensure we get good values and indices
  | for kthvalue, mode this will always
  | be with the reducing dim as 1-d
  |
  */
#[inline] pub fn reduction_with_indices_allocate_or_resize_output(
        values:  &mut Tensor,
        indices: &mut Tensor,
        self_:   &Tensor,
        dim:     i64,
        keepdim: bool)  {
    
    todo!();
        /*
            i64 dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
      auto result_sizes = self.sizes().vec();
      if (result_sizes.size() > 0) {
        result_sizes[dim] = 1;
      }
      if (values.defined()) {
        TORCH_CHECK(
            self.options().type_equal(values.options()),
            "output values must be of same type as input");
        if (!keepdim && values.dim() == self.dim() - 1) {
          // unsqueeze to preserve passed in noncontiguous tensor in resize
          values.unsqueeze_(dim);
        }
        resize_output(values, result_sizes);
      } else {
        values = empty(result_sizes, self.options());
      }
      if (indices.defined()) {
        TORCH_CHECK(
            indices.dtype() == kLong, "output indices must be of scalar type Long");
        TORCH_CHECK(
            indices.device() == self.device(),
            "output indices must be on same device as input");
        if (!keepdim && indices.dim() == self.dim() - 1) {
          // unsqueeze to preserve passed in noncontiguous tensor in resize
          indices.unsqueeze_(dim);
        }
        resize_output(indices, result_sizes);
      } else {
        indices = empty(result_sizes, self.options().dtype(kLong));
      }
        */
}

/**
  | ensure we get good values and indices
  | for topk
  |
  */
#[inline] pub fn allocate_or_resize_output_with_indices(
        values:  &mut Tensor,
        indices: &mut Tensor,
        self_:   &Tensor,
        dim:     i64,
        k:       i64)  {
    
    todo!();
        /*
            i64 dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
      auto result_sizes = self.sizes().vec();
      if (result_sizes.size() > 0) {
        result_sizes[dim] = k;
      }
      if (values.defined()) {
        TORCH_CHECK(
            self.options().type_equal(values.options()),
            "output values must be of same type as input");
        values.resize_(result_sizes);
      } else {
        values = empty(result_sizes, self.options());
      }
      if (indices.defined()) {
        TORCH_CHECK(
            indices.dtype() == kLong, "output indices must be of scalar type Long");
        TORCH_CHECK(
            indices.device() == self.device(),
            "output indices must be on same device as input");
        indices.resize_(result_sizes);
      } else {
        indices = empty(result_sizes, self.options().dtype(kLong));
      }
        */
}

/**
  | Core topk loop, shared between CPU and
  | QuantizedCPU
  |
  */
pub fn topk_impl_loop<Scalar>(
    mode_values_stride:  i64,
    mode_indices_stride: i64,
    tmp_values_stride:   i64,
    k:                   i64,
    dim_size:            i64,
    largest:             bool,
    sorted:              bool,
    data:                *mut *mut u8,
    strides:             *const i64,
    n:                   i64)  {

    todo!();
    /*
            for (i64 i = 0; i < n; ++i) {
        TensorAccessor<Scalar, 1> mode_values(
            reinterpret_cast<Scalar*>(data[0] + i * strides[0]),
            &k, &mode_values_stride);
        TensorAccessor<i64, 1> mode_indices(
            reinterpret_cast<i64*>(data[1] + i * strides[1]),
            &k, &mode_indices_stride);
        TensorAccessor<Scalar, 1> tmp_values(
            reinterpret_cast<Scalar*>(data[2] + i * strides[2]),
            &dim_size, &tmp_values_stride);

        auto n = dim_size;
        auto use_partial_sort = k * 64 <= n;

        using elem_t = pair<Scalar, i64>;
        vector<elem_t> queue(n);
        for (i64 j = 0; j < n; j++) {
          queue[j].first = tmp_values[j];
          queue[j].second = j;
        }

        // we want nan to be sorted as top for numpy compatibility
        if (use_partial_sort) {
          if (largest) {
            partial_sort(queue.begin(), queue.begin() + k, queue.end(),
              [](const elem_t& x, const elem_t& y) -> bool {
                return ((_isnan<Scalar>(x.first) && !_isnan<Scalar>(y.first)) || (x.first > y.first));
              });
          } else {
            partial_sort(queue.begin(), queue.begin() + k, queue.end(),
              [](const elem_t& x, const elem_t& y) -> bool {
                return ((!_isnan<Scalar>(x.first) && _isnan<Scalar>(y.first)) || (x.first < y.first));
              });
          }
        } else {
          if (largest) {
            nth_element(queue.begin(), queue.begin() + k - 1, queue.end(),
              [](const elem_t& x, const elem_t& y) -> bool {
                return ((_isnan<Scalar>(x.first) && !_isnan<Scalar>(y.first)) || (x.first > y.first));
              });
            if (sorted) {
              sort(queue.begin(), queue.begin() + k - 1,
                [](const elem_t& x, const elem_t& y) -> bool {
                  return ((_isnan<Scalar>(x.first) && !_isnan<Scalar>(y.first)) || (x.first > y.first));
                });
            }
          } else {
            nth_element(queue.begin(), queue.begin() + k -1, queue.end(),
              [](const elem_t& x, const elem_t& y) -> bool {
                return ((!_isnan<Scalar>(x.first) && _isnan<Scalar>(y.first)) || (x.first < y.first));
              });
            if (sorted) {
              sort(queue.begin(), queue.begin() + k -1,
                [](const elem_t& x, const elem_t& y) -> bool {
                  return ((!_isnan<Scalar>(x.first) && _isnan<Scalar>(y.first)) || (x.first < y.first));
                });
            }
          }
        }

        for (i64 j = 0; j < k; j++) {
          mode_values[j] = queue[j].first;
          mode_indices[j] = queue[j].second;
        }
      }
        */
}
