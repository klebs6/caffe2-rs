crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/ScatterGatherKernel.cpp]

/**
  | Implement as functors since lambdas
  | don't get optimized.
  |
  */
pub struct ReduceMultiply {

}

impl ReduceMultiply {
    
    pub fn invoke<Scalar>(&self, 
        self_data: *mut Scalar,
        src_data:  *mut Scalar)  {
    
        todo!();
        /*
            *self_data *= *src_data;
        */
    }
    
    pub fn invoke(&self, 
        self_data: *mut bool,
        src_data:  *mut bool)  {
        
        todo!();
        /*
            *self_data = *self_data && *src_data;
        */
    }
}

lazy_static!{
    /*
    static ReduceMultiply reduce_multiply;
    */
}

pub struct ReduceAdd {

}

impl ReduceAdd {
    
    pub fn invoke<Scalar>(&self, 
        self_data: *mut Scalar,
        src_data:  *mut Scalar)  {
    
        todo!();
        /*
            *self_data += *src_data;
        */
    }
}

lazy_static!{
    /*
    static ReduceAdd reduce_add;
    */
}

pub struct TensorAssign {

}

impl TensorAssign {
    
    pub fn invoke<Scalar>(&self, 
        self_data: *mut Scalar,
        src_data:  *mut Scalar)  {
    
        todo!();
        /*
            *self_data = *src_data;
        */
    }
}

lazy_static!{
    /*
    static TensorAssign tensor_assign;
    */
}

pub struct CpuScatterGatherDimLoop<const is_scatter_like: bool = true> {

}

impl<const is_scatter_like: bool> CpuScatterGatherDimLoop<is_scatter_like> {
    
    pub fn invoke<Scalar, func_t>(&mut self, 
        self_data:         *mut Scalar,
        self_dim_stride:   i64,
        index_data:        *mut i64,
        index_dim_stride:  i64,
        src_data:          *mut Scalar,
        src_dim_stride:    i64,
        dim:               i64,
        index_dim_size:    i64,
        index_upper_bound: i64,
        f:                 &mut Func)  {
    
        todo!();
        /*
            for (i64 i = 0; i < index_dim_size; ++i) {
          i64 idx_dim = index_data[i * index_dim_stride];
          // we are not putting idx_dim in the error message because it disables
          // loop optimization in clang-7
          TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
            "index ", index_data[i * index_dim_stride],
            " is out of bounds for dimension ", dim,
            " with size ", index_upper_bound
          );

          f(
            self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
            src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride
          );
        }
        */
    }
    
    pub fn invoke<Scalar, func_t>(&mut self, 
        self_data:         *mut Scalar,
        self_dim_stride:   i64,
        index_data:        *mut i64,
        index_dim_stride:  i64,
        value:             Scalar,
        dim:               i64,
        index_dim_size:    i64,
        index_upper_bound: i64,
        f:                 &mut Func)  {
    
        todo!();
        /*
            for (i64 i = 0; i < index_dim_size; ++i) {
          i64 idx_dim = index_data[i * index_dim_stride];
          // we are not putting idx_dim in the error message because it disables
          // loop optimization in clang-7
          TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
            "index ", index_data[i * index_dim_stride],
            " is out of bounds for dimension ", dim,
            " with size ", index_upper_bound
          );
          auto temp = value.to<Scalar>();
          f(
            self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride, &temp
          );
        }
        */
    }
}

pub struct CpuScatterGatherBaseKernel<const is_scatter_like: bool = true> {

}

impl<const is_scatter_like: bool> CpuScatterGatherBaseKernel<is_scatter_like> {
    
    pub fn invoke<func_t>(&mut self, 
        self_:       &mut Tensor,
        dim:         i64,
        index:       &Tensor,
        value:       &Scalar,
        method_name: &String,
        kernel_func: &mut Func)  {
    
        todo!();
        /*
            // no-op if index is empty
        if (index.numel() == 0) {
          return;
        }

        dim = maybe_wrap_dim(dim, self.dim());

        auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
        auto index_strides = ensure_nonempty_vec(index.strides().vec());

        // `dim` is traversed in the kernel,
        // that is why index.stride(dim) = 0 and index.size(dim) = 1.
        // Also, index.size(dim) = 1 makes sure that TensorIterator.DimCounter
        // has the following form : (i_1,..., i_{dim-1}, 0, i_{dim+1},...,i_n).
        index_sizes[dim] = 1;
        index_strides[dim] = 0;

        auto iter = TensorIteratorConfig()
          .check_all_same_dtype(false)
          .resize_outputs(false)
          .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
          .add_output(self)
          .add_input(index)
          .build();

        auto self_dim_stride = ensure_nonempty_stride(self, dim);
        auto self_dim_size = ensure_nonempty_size(self, dim);

        auto index_dim_stride = ensure_nonempty_stride(index, dim);
        auto index_dim_size = ensure_nonempty_size(index, dim);

        auto index_upper_bound = self_dim_size;

        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
          ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, iter.dtype(),
          "scatter_gather_scalar_cpu", [&] {
            constexpr auto SELF_ITER_STRIDE_IDX = 0;
            constexpr auto INDEX_ITER_STRIDE_IDX = 1;

            auto loop = [&](char** data, const i64* strides, i64 n) {
              auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
              auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
              // we change the order of TensorIterator-dim loop
              // vs dim-TensorIterator loop order depending on
              // whether dim is the last dimension and/or
              // whether `n` is smaller than `index_dim_size`

              if ((dim== self.dim() - 1) || (n < index_dim_size)) {
                for (i64 nelem = 0; nelem < n; ++nelem) {
                  // dim loop is a separate code block
                  // for better performance
                  _cpu_scatter_gather_dim_loop<is_scatter_like>()(
                    (Scalar*)self_data_bytes, self_dim_stride,
                    (i64*)index_data_bytes, index_dim_stride,
                    value, dim, index_dim_size, index_upper_bound,
                    kernel_func);

                  self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
                  index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
                }
              }
              else {
                for (i64 i = 0; i < index_dim_size; ++i) {
                  auto* self_data = self_data_bytes;
                  auto* index_data = (char*)((i64*)index_data_bytes + i * index_dim_stride);
                  for (i64 nelem = 0; nelem < n; ++nelem) {
                    i64 idx_dim = *(i64*)index_data;
                    // we are not putting idx_dim in the error message because it disables
                    // loop optimization in clang-7
                    TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                                "index ", *(i64*)index_data,
                                " is out of bounds for dimension ", dim,
                                " with size ", index_upper_bound);

                    auto temp = value.to<Scalar>();
                    kernel_func((Scalar*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride, &temp);

                    self_data += strides[SELF_ITER_STRIDE_IDX];
                    index_data += strides[INDEX_ITER_STRIDE_IDX];
                  }
                }
              }
            };
            iter.for_each(loop);
          }
        );
        */
    }
    
    pub fn invoke<func_t>(&mut self, 
        self_:       &mut Tensor,
        dim:         i64,
        index:       &Tensor,
        src:         &Tensor,
        method_name: &String,
        kernel_func: &mut Func)  {
    
        todo!();
        /*
            // no-op if index is empty
        if (index.numel() == 0) {
          return;
        }

        dim = maybe_wrap_dim(dim, self.dim());

        scatter_gather_dtype_check(method_name, self, index, src);
        if (!is_scatter_like) {
          gather_shape_check(self, dim, index, src);
        }

        auto iter = TensorIteratorConfig()
          .check_all_same_dtype(false)
          .resize_outputs(false)
          .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
          .add_output(self)
          .add_input(src)
          .add_input(index)
          .build();

        auto self_dim_stride = ensure_nonempty_stride(self, dim);
        auto self_dim_size = ensure_nonempty_size(self, dim);

        auto index_dim_stride = ensure_nonempty_stride(index, dim);
        auto index_dim_size = ensure_nonempty_size(index, dim);

        auto src_dim_stride = ensure_nonempty_stride(src, dim);
        auto src_dim_size = ensure_nonempty_size(src, dim);

        auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;

        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
          ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, iter.dtype(),
          "scatter_gather_tensor_cpu", [&] {
            constexpr auto SELF_ITER_STRIDE_IDX = 0;
            constexpr auto INDEX_ITER_STRIDE_IDX = 2;
            constexpr auto SRC_ITER_STRIDE_IDX = 1;
            auto loop = [&](char** data, const i64* strides, i64 n) {
              auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
              auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
              auto* src_data_bytes = data[SRC_ITER_STRIDE_IDX];
              // we change the order of TensorIterator-dim loop
              // vs dim-TensorIterator loop order depending on
              // whether dim is the last dimension and/or
              // whether `n` is smaller than `index_dim_size`
              if ((dim== self.dim() - 1) || (n < index_dim_size)) {
                for (i64 nelem = 0; nelem < n; ++nelem) {
                  // dim loop is a separate code block
                  // for better performance
                  _cpu_scatter_gather_dim_loop<is_scatter_like>()(
                     (Scalar*)self_data_bytes, self_dim_stride,
                     (i64*)index_data_bytes, index_dim_stride,
                     (Scalar*)src_data_bytes, src_dim_stride,
                     dim, index_dim_size, index_upper_bound,
                     kernel_func
                   );

                  self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
                  index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
                  src_data_bytes += strides[SRC_ITER_STRIDE_IDX];
                }
              }
              else {
                for (i64 i = 0; i < index_dim_size; ++i) {
                  auto* self_data = self_data_bytes;
                  auto* index_data = (char*)((i64*)index_data_bytes + i * index_dim_stride);
                  auto* src_data = src_data_bytes;
                  for (i64 nelem = 0; nelem < n; ++nelem) {
                    i64 idx_dim = *(i64*)index_data;
                    // we are not putting idx_dim in the error message because it disables
                    // loop optimization in clang-7
                    TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                                "index ", *(i64*)index_data,
                                " is out of bounds for dimension ", dim,
                                " with size ", index_upper_bound);

                    kernel_func(
                      (Scalar*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
                      (Scalar*)src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride);

                    self_data += strides[SELF_ITER_STRIDE_IDX];
                    index_data += strides[INDEX_ITER_STRIDE_IDX];
                    src_data += strides[SRC_ITER_STRIDE_IDX];
                  }
                }
              }
            };
            iter.for_each(loop);
          }
        );
        */
    }
}

pub fn gather_cpu_kernel(
        result: &mut Tensor,
        self_:  &Tensor,
        dim:    i64,
        index:  &Tensor)  {
    
    todo!();
        /*
            cpu_scatter_gather_base_kernel</*is_scatter_like=*/false>()(
        result, dim, index, self,
        "gather_out_cpu", tensor_assign);
        */
}

pub fn scatter_cpu_kernel(
        self_: &mut Tensor,
        dim:   i64,
        index: &Tensor,
        src:   &Tensor)  {
    
    todo!();
        /*
            cpu_scatter_gather_base_kernel<>()(
        self, dim, index, src, "scatter_cpu_", tensor_assign);
        */
}

pub fn scatter_fill_cpu_kernel(
        self_: &mut Tensor,
        dim:   i64,
        index: &Tensor,
        value: &Scalar)  {
    
    todo!();
        /*
            cpu_scatter_gather_base_kernel<>()(
        self, dim, index, value, "scatter_fill_cpu_", tensor_assign);
        */
}

pub fn scatter_add_cpu_kernel(
        self_: &mut Tensor,
        dim:   i64,
        index: &Tensor,
        src:   &Tensor)  {
    
    todo!();
        /*
            cpu_scatter_gather_base_kernel()(
        self, dim, index, src,
        "scatter_add_", reduce_add);
        */
}

pub fn scatter_reduce_cpu_kernel(
        self_:  &mut Tensor,
        dim:    i64,
        index:  &Tensor,
        src:    &Tensor,
        reduce: &ScatterGatherOp)  {
    
    todo!();
        /*
            switch (reduce) {
      case SCATTER_GATHER_OP::REDUCE_ADD :
        cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                           "scatter_reduce_add_", reduce_add);
        break;
      case SCATTER_GATHER_OP::REDUCE_MULTIPLY :
        cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                           "scatter_reduce_multiply_", reduce_multiply);
        break;
      }
        */
}

pub fn scatter_scalar_reduce_cpu_kernel(
        self_:  &mut Tensor,
        dim:    i64,
        index:  &Tensor,
        value:  &Scalar,
        reduce: &ScatterGatherOp)  {
    
    todo!();
        /*
            switch (reduce) {
      case SCATTER_GATHER_OP::REDUCE_ADD :
        cpu_scatter_gather_base_kernel<>()(self, dim, index, value,
                                           "scatter_scalar_reduce_add_", reduce_add);
        break;
      case SCATTER_GATHER_OP::REDUCE_MULTIPLY :
        cpu_scatter_gather_base_kernel<>()(self, dim, index, value,
                                           "scatter_scalar_reduce_multiply_", reduce_multiply);
        break;
      }
        */
}

register_dispatch!{gather_stub                , &gather_cpu_kernel}
register_dispatch!{scatter_stub               , &scatter_cpu_kernel}
register_dispatch!{scatter_fill_stub          , &scatter_fill_cpu_kernel}
register_dispatch!{scatter_add_stub           , &scatter_add_cpu_kernel}
register_dispatch!{scatter_reduce_stub        , &scatter_reduce_cpu_kernel}
register_dispatch!{scatter_scalar_reduce_stub , &scatter_scalar_reduce_cpu_kernel}
