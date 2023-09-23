crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TriangularOps.cpp]

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triu/tril ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn apply_triu_tril_single<Scalar, const upper: bool>(
        result:          *mut Scalar,
        self_:           *mut Scalar,
        inplace:         bool,
        k:               i64,
        n:               i64,
        m:               i64,
        res_row_stride:  i64,
        res_col_stride:  i64,
        self_row_stride: i64,
        self_col_stride: i64)  {

    todo!();
        /*
            constexpr i64 zero = 0;

      if (upper) {
        parallel_for(0, n, 0, [&](i64 start, i64 end) {
          for (auto i = start; i < end; i++) {
            for (i64 j = 0; j < min(m, i + k); j++) {
              result[i * res_row_stride + j * res_col_stride] = 0;
            }
            if (!inplace) {  // copy the rest of the self if not inplace
              for (i64 j = max(zero, i + k); j < m; j++) {
                result[i * res_row_stride + j * res_col_stride] = self[i * self_row_stride + j * self_col_stride];
              }
            }
          }
        });
      } else {
        parallel_for(0, n, 0, [&](i64 start, i64 end) {
          for (auto i = start; i < end; i++) {
            for (i64 j = max(zero, i + k + 1); j < m; j++) {
              result[i * res_row_stride + j * res_col_stride] = 0;
            }
            if (!inplace) {  // copy the rest of the self if not inplace
              for (i64 j = zero; j < min(m, i + k + 1); j++) {
                result[i * res_row_stride + j * res_col_stride] = self[i * self_row_stride + j * self_col_stride];
              }
            }
          }
        });
      }
        */
}


pub fn apply_triu_tril<Scalar, const upper: bool>(
        result:  &mut Tensor,
        self_:   &Tensor,
        inplace: bool,
        k:       i64)  {

    todo!();
        /*
            auto n = self.size(-2);
      auto m = self.size(-1);
      auto self_data = self.data_ptr<Scalar>();
      auto self_stride = (self.dim() > 2 && self.stride(-3) > 0) ? self.stride(-3) : 1;
      auto batchsize = batchCountTrilTriu(result);
      auto self_row_stride = self.stride(-2);
      auto self_column_stride = self.stride(-1);

      auto result_data = result.data_ptr<Scalar>();
      i64 result_stride, result_row_stride, result_column_stride;
      if (result_data != self_data) {
        result_stride = (result.dim() > 2 && result.stride(-3) > 0) ? result.stride(-3) : 1;
        result_row_stride = result.stride(-2);
        result_column_stride = result.stride(-1);
      } else {
        result_stride = self_stride;
        result_row_stride = self_row_stride;
        result_column_stride = self_column_stride;
      }

      parallel_for(0, batchsize, 0, [&](i64 start, i64 end) {
        for (auto b = start; b < end; b++) {
          Scalar* self_batch = &self_data[b * self_stride];
          Scalar* result_batch = &result_data[b * result_stride];
          apply_triu_tril_single<Scalar, upper>(
              result_batch, self_batch, inplace, k, n, m,
              result_row_stride, result_column_stride, self_row_stride, self_column_stride);
        }
      });
        */
}


pub fn tril(
        self_: &Tensor,
        k:     i64) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      tril_out(result, self, k);
      return result;
        */
}


pub fn tril_cpu<'a>(
        self_: &mut Tensor,
        k:     i64) -> &'a mut Tensor {
    
    todo!();
        /*
            if (self.numel() == 0) {
        return self;
      }
      bool inplace;
      Tensor self_c;
      tie(inplace, self_c) = checkTrilTriuBatchContiguous(self, true);
      Tensor result = inplace ? self : empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(ScalarType::Half, ScalarType::Bool, self.scalar_type(), "tril", [&]{
        apply_triu_tril<Scalar, false>(result, self_c, inplace, k);
      });
      if (!inplace) self.copy_(result);
      return self;
        */
}


pub fn tril_cpu_out<'a>(
        self_:  &Tensor,
        k:      i64,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            native::resize_output(result, self.sizes());
      if (self.numel() == 0) {
        return result;
      }
      Tensor self_c;
      tie(ignore, self_c) = checkTrilTriuBatchContiguous(self, false);
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(ScalarType::Half, ScalarType::Bool, self.scalar_type(), "tril", [&]{
        apply_triu_tril<Scalar, false>(result, self_c, false, k);
      });
      return result;
        */
}


pub fn triu(
        self_: &Tensor,
        k:     i64) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      triu_out(result, self, k);
      return result;
        */
}


pub fn triu_cpu<'a>(
        self_: &mut Tensor,
        k:     i64) -> &'a mut Tensor {
    
    todo!();
        /*
            if (self.numel() == 0) {
        return self;
      }
      bool inplace;
      Tensor self_c;
      tie(inplace, self_c) = checkTrilTriuBatchContiguous(self, true);
      Tensor result = inplace ? self : empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(ScalarType::Half, ScalarType::Bool, self.scalar_type(), "triu", [&]{
        apply_triu_tril<Scalar, true>(result, self_c, inplace, k);
      });
      if (!inplace) self.copy_(result);
      return self;
        */
}


pub fn triu_cpu_out<'a>(
        self_:  &Tensor,
        k:      i64,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            native::resize_output(result, self.sizes());
      if (self.numel() == 0) {
        return result;
      }
      Tensor self_c;
      tie(ignore, self_c) = checkTrilTriuBatchContiguous(self, false);
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(ScalarType::Half, ScalarType::Bool, self.scalar_type(), "triu", [&]{
        apply_triu_tril<Scalar, true>(result, self_c, false, k);
      });
      return result;
        */
}


pub fn trace_backward(
        grad:  &Tensor,
        sizes: &[i32]) -> Tensor {
    
    todo!();
        /*
            if (sizes.size() != 2) {
        throw runtime_error("expected matrix input");
      }

      auto grad_input = zeros(sizes[0] * sizes[1], grad.options());
      auto indices = arange(0, grad_input.numel(), sizes[1] + 1, grad.options().dtype(kLong));
      grad_input.index_fill_(0, indices, grad);
      return grad_input.view(sizes);
        */
}
