crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cuda/Blas.cpp]

#[inline] pub fn prepare_matrix_for_cublas(
        tensor:           &Tensor,
        transpose_tensor: &mut bool) -> MaybeOwned<Tensor> {
    
    todo!();
        /*
            if (tensor.is_non_overlapping_and_dense()) { // common case
          transpose_tensor = tensor.is_contiguous();
          return MaybeOwned<Tensor>::borrowed(tensor);
      }
      IntArrayRef tensor_strides = tensor.strides();
      IntArrayRef tensor_sizes = tensor.sizes();
      if ((tensor_strides[0] == 1) && (tensor_strides[1] >= max<i64>(1, tensor_sizes[0]))) {
        transpose_tensor = false;
        return MaybeOwned<Tensor>::borrowed(tensor);
      } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= max<i64>(1, tensor_sizes[1]))) {
        transpose_tensor = true;
        return MaybeOwned<Tensor>::borrowed(tensor);
      } else {
        transpose_tensor = true;
        return MaybeOwned<Tensor>::owned(tensor.clone(MemoryFormat::Contiguous));
      }
        */
}

pub fn prepare_batch_matrix_for_cublas(
        tensor:           &Tensor,
        transpose_tensor: &mut bool,
        ld_tensor:        &mut i64,
        transpose_result: bool,
        m:                i64,
        n:                i64) -> MaybeOwned<Tensor> {
    
    todo!();
        /*
            IntArrayRef tensor_strides = tensor.strides();
      MaybeOwned<Tensor> tensor_;
      int fast_dim = transpose_result ? 2 : 1;
      int leading_dim = transpose_result ? 1 : 2;

      if (tensor_strides[fast_dim] == 1 &&
        (tensor_strides[leading_dim] >= max<i64>(1, m))) {
        transpose_tensor = false;
        tensor_ = MaybeOwned<Tensor>::borrowed(tensor);
        ld_tensor = tensor_strides[leading_dim];
      } else if ((tensor_strides[leading_dim] == 1) &&
        (tensor_strides[fast_dim] >= max<i64>(1, n))) {
        transpose_tensor = true;
        tensor_ = MaybeOwned<Tensor>::borrowed(tensor);
        ld_tensor = tensor_strides[fast_dim];
      } else {
        transpose_tensor = !transpose_result;
        // gemm call requires leading dimension and stride parameters to be non-zero
        bool is_stride_non_zero = tensor.strides()[1] != 0 && tensor.strides()[2] != 0;
        if (tensor.is_contiguous() && is_stride_non_zero) {
          tensor_ = MaybeOwned<Tensor>::borrowed(tensor);
        } else {
          tensor_ = MaybeOwned<Tensor>::owned(tensor.clone(MemoryFormat::Contiguous));
        }
        ld_tensor = tensor_->strides()[1];
      }

      return tensor_;
        */
}

pub fn addmm_out_cuda_impl(
        result: &mut Tensor,
        self_:  &Tensor,
        mat1:   &Tensor,
        mat2:   &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            // Make sure to keep addmm_cuda below in sync with this code; it
      // preflights a check to try to avoid actually needing to call
      // expand().
      TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");

      TensorArg args[]{{result, "out", 0}, {self, "self", 1}, {mat1, "mat1", 2}, {mat2, "mat2", 3}};
      checkAllSameGPU(__func__, args);

      IntArrayRef mat1_sizes = mat1.sizes();
      IntArrayRef mat2_sizes = mat2.sizes();
      IntArrayRef self__sizes;
      MaybeOwned<Tensor> self_;
      if (&result != &self) {
        self_ = expand_size(self, {mat1_sizes[0], mat2_sizes[1]}, "addmm");
        self__sizes = self_->sizes();
      } else {
        self_ = MaybeOwned<Tensor>::borrowed(self);
        self__sizes = self_->sizes();
        TORCH_CHECK(result.dim() == 2, "tensors must be 2-D");
        TORCH_CHECK(self__sizes[0] == mat1_sizes[0], "self_ dim 0 must match mat1 dim 0");
        TORCH_CHECK(self__sizes[1] == mat2_sizes[1], "self_ dim 1 must match mat2 dim 1");
      }

      if (&result != &self) {
        native::resize_output(result, self__sizes);
        if (beta.toComplexDouble() != 0.0) {
          native::copy_(result, *self_);
        }
      }

      IntArrayRef result_sizes = result.sizes();
      if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
        return result;
      }

      bool transpose_result;
      MaybeOwned<Tensor> result_ = prepare_matrix_for_cublas(result, transpose_result);
      bool transpose_mat1;
      bool transpose_mat2;
      MaybeOwned<Tensor> mat1_ = prepare_matrix_for_cublas(transpose_result ? mat2 : mat1, transpose_mat1);
      MaybeOwned<Tensor> mat2_ = prepare_matrix_for_cublas(transpose_result ? mat1 : mat2, transpose_mat2);

      if (transpose_result) {
        transpose_mat1 = !transpose_mat1;
        transpose_mat2 = !transpose_mat2;
        mat1_sizes = mat1_->sizes();
        mat2_sizes = mat2_->sizes();
      }

      i64 m = mat1_sizes[transpose_result ? 1 : 0];
      i64 k = mat1_sizes[transpose_result ? 0 : 1];
      i64 n = mat2_sizes[transpose_result ? 0 : 1];
      i64 mat1_ld = mat1_->stride((transpose_mat1 == transpose_result) ? 1 : 0);
      i64 mat2_ld = mat2_->stride((transpose_mat2 == transpose_result) ? 1 : 0);
      i64 result_ld = result_->stride(transpose_result ? 0 : 1);
      ScalarType scalar_type = self_->scalar_type();

      if (mat1.numel() == 0) {
        // By definition, when beta==0, values in self should be ignored. nans and infs
        // should not propagate
        if (beta.toComplexDouble() == 0.) {
          return result.zero_();
        }
        // TODO: We could squeeze some perf by calling mul_out here instead, to bypass the dispatcher.
        // That requires some fixing some internal build dependencies though.
        return mul_out(
            result,
            self,
            native::scalar_tensor(
                beta,
                self.scalar_type(),
                nullopt /* layout */,
                kCPU,
                nullopt /* pin_memory */));
      }

      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, scalar_type, "addmm_cuda", [&] {
        Scalar alpha_val = alpha.to<Scalar>();
        Scalar beta_val = beta.to<Scalar>();
        Scalar* mat1_ptr = mat1_->data_ptr<Scalar>();
        Scalar* mat2_ptr = mat2_->data_ptr<Scalar>();
        Scalar* result_ptr = result_->data_ptr<Scalar>();
        blas::gemm<Scalar>(
          transpose_mat1 ? 't' : 'n',
          transpose_mat2 ? 't' : 'n',
          m, n, k,
          alpha_val,
          mat1_ptr, mat1_ld,
          mat2_ptr, mat2_ld,
          beta_val,
          result_ptr, result_ld
        );
      });
      if (!result.is_same(*result_)) {
        result.copy_(*result_);
      }
      return result;
        */
}

pub fn baddbmm_out_cuda_impl(
        result: &mut Tensor,
        self_:  &Tensor,
        batch1: &Tensor,
        batch2: &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.dim() == 3, "self must be a 3D tensor");
      TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
      TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");

      TensorArg args[]{{result, "out", 0}, {self, "self", 1}, {batch1, "batch1", 2}, {batch2, "batch2", 3}};
      checkAllSameGPU(__func__, args);

      IntArrayRef batch1_sizes = batch1.sizes();
      IntArrayRef batch2_sizes = batch2.sizes();
      IntArrayRef self_sizes = self.sizes();

      TORCH_CHECK(self_sizes[0] == batch1_sizes[0], "self dim 0 must match batch1 dim 0");
      TORCH_CHECK(self_sizes[0] == batch2_sizes[0], "self dim 0 must match batch2 dim 0");
      TORCH_CHECK(self_sizes[1] == batch1_sizes[1], "self dim 1 must match batch1 dim 1");
      TORCH_CHECK(self_sizes[2] == batch2_sizes[2], "self dim 2 must match batch2 dim 2");
      TORCH_CHECK(batch1_sizes[2] == batch2_sizes[1], "batch1 dim 2 must match batch2 dim 1");

      if (!result.is_same(self)) {
        result.resize_as_(self);
        if (beta.to<complex<double>>() != 0.0) {
          result.copy_(self);
        }
      }

      // handle pathological cases that blas may not like
      if (result.numel() == 0) {
        return result;
      } else if (batch1_sizes[2] == 0) {
        if (beta.to<complex<double>>() == 0.0) {
          return result.zero_();
        } else {
          return result.mul_(beta);
        }
      }

      bool transpose_result = false;
      MaybeOwned<Tensor> result_;
      IntArrayRef result_strides = result.strides();
      IntArrayRef result_sizes = result.sizes();

      if ((result_strides[1] == 1) &&
          ((result_sizes[2] == 1) || (result_strides[2] >= max<i64>(1, result_sizes[1])))) {
        result_ = MaybeOwned<Tensor>::borrowed(result);
      } else if ((result_strides[2] == 1) &&
        (result_sizes[1] == 1 || (result_strides[1] >= max<i64>(1, result_sizes[2])))) {
        transpose_result = true;
        result_ = MaybeOwned<Tensor>::borrowed(result);
      } else {
        result_ = MaybeOwned<Tensor>::owned(result.transpose(1, 2).clone(MemoryFormat::Contiguous).transpose(1, 2));
      }

      int leading_dim = transpose_result ? 1 : 2;

      i64 m = result_sizes[transpose_result ? 2 : 1];
      i64 n = result_sizes[leading_dim];
      i64 k = (transpose_result ? batch2 : batch1).sizes()[leading_dim];

      i64 lda, ldb, ldc;
      bool transpose_batch1, transpose_batch2;
      auto batch1_ = prepare_batch_matrix_for_cublas(transpose_result ? batch2 : batch1, transpose_batch1, lda, transpose_result, m, k);
      auto batch2_ = prepare_batch_matrix_for_cublas(transpose_result ? batch1 : batch2, transpose_batch2, ldb, transpose_result, k, n);

      ldc = result_->strides()[leading_dim];
      i64 num_batches = result_->sizes()[0];

      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, self.scalar_type(), "baddbmm_cuda", [&] {
        Scalar alpha_val = alpha.to<Scalar>();
        Scalar beta_val = beta.to<Scalar>();
        Scalar* batch1_ptr = batch1_->data_ptr<Scalar>();
        Scalar* batch2_ptr = batch2_->data_ptr<Scalar>();
        Scalar* result_ptr = result_->data_ptr<Scalar>();
        blas::bgemm<Scalar>(
          transpose_batch1 ? 't' : 'n',
          transpose_batch2 ? 't' : 'n',
          m, n, k,
          alpha_val,
          batch1_ptr, lda, batch1_->strides()[0],
          batch2_ptr, ldb, batch2_->strides()[0],
          beta_val,
          result_ptr, ldc, result_->strides()[0],
          num_batches
        );
      });
      if (!result.is_same(*result_)) {
        result.copy_(*result_);
      }
      return result;
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(addmm_out_cuda)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, const Tensor& result) {
      addmm_out_cuda_impl(const_cast<Tensor&>(result), self, mat1, mat2, beta, alpha);
    }

    TORCH_IMPL_FUNC(mm_out_cuda)(const Tensor& self, const Tensor& mat2, const Tensor& result) {
      addmm_out_cuda_impl(const_cast<Tensor&>(result), result, self, mat2, 0, 1);
    }
    */
}

pub fn baddbmm_out_cuda(
        self_:  &Tensor,
        batch1: &Tensor,
        batch2: &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto self_ = &result == &self
        ? MaybeOwned<Tensor>::borrowed(self)
        : expand_size(self, {batch1.size(0), batch1.size(1), batch2.size(2)}, "baddbmm");
      {
        NoNamesGuard guard;
        baddbmm_out_cuda_impl(result, *self_, batch1, batch2, beta, alpha);
      }
      namedinference::propagate_names_if_nonempty(
           result,
           namedinference::compute_baddbmm_outnames(result, batch1, batch2, self));
      return result;
        */
}

pub fn baddbmm_cuda(
        self_:  &Tensor,
        batch1: &Tensor,
        batch2: &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor out = empty({0}, self.options());
      return baddbmm_out_cuda(self, batch1, batch2, beta, alpha, out);
        */
}

pub fn baddbmm_cuda_mut(
        self_:  &mut Tensor,
        batch1: &Tensor,
        batch2: &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return baddbmm_out_cuda(self, batch1, batch2, beta, alpha, self);
        */
}

pub fn bmm_out_cuda(
        batch1: &Tensor,
        batch2: &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
      TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");
      native::resize_output(result, {batch1.sizes()[0], batch1.sizes()[1], batch2.sizes()[2]});
      Scalar beta(0.0);
      Scalar alpha(1.0);
      {
        NoNamesGuard guard;
        baddbmm_out_cuda_impl(result, result, batch1, batch2, beta, alpha);
      }
      namedinference::propagate_names_if_nonempty(
          result,
          namedinference::compute_bmm_outnames(result, batch1, batch2));
      return result;
        */
}

pub fn bmm_cuda(
        self_: &Tensor,
        mat2:  &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.dim() == 3, "self must be a 3D tensor");
      TORCH_CHECK(mat2.dim() == 3, "batch2 must be a 3D tensor");
      Tensor result = empty({self.sizes()[0], self.sizes()[1], mat2.sizes()[2]}, self.options());
      return native::bmm_out_cuda(self, mat2, result);
        */
}

#[inline] pub fn dot_check(
        self_: &Tensor,
        other: &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(
          self.dim() == 1 && other.dim() == 1,
          "1D tensors expected, but got ",
          self.dim(),
          "D and ",
          other.dim(),
          "D tensors");
      TORCH_CHECK(
          self.scalar_type() == other.scalar_type(),
          "dot : expected both vectors to have same dtype, but found ",
          self.scalar_type(),
          " and ",
          other.scalar_type());
      TORCH_CHECK(
          self.numel() == other.numel(),
          "inconsistent tensor size, expected tensor [",
          self.numel(),
          "] and src [",
          other.numel(),
          "] to have the same number of elements, but got ",
          self.numel(),
          " and ",
          other.numel(),
          " elements respectively");
      TORCH_CHECK(
          self.device() == other.device(),
          "expected all tensors to be on the same device. Found: ",
          self.device(),
          ", ",
          other.device());
      TORCH_CHECK(
          (self.numel() <= INT_MAX) && (self.stride(0) <= INT_MAX) &&
              (other.stride(0) <= INT_MAX),
          "dot only supports n, incx, incy with the bound [val] <= %d",
          INT_MAX);
        */
}

pub fn dot_cuda(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            NoNamesGuard guard;

      dot_check(self, other);

      const int n = static_cast<int>(self.numel());
      int incx = static_cast<int>(self.stride(0));
      int incy = static_cast<int>(other.stride(0));
      if (n == 1) {
        incx = 1;
        incy = 1;
      }

    return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
          ScalarType::Half, ScalarType::BFloat16,
          self.scalar_type(), "dot",
          [&] {
            Tensor result = empty({}, self.options());

            auto handle = getCurrentCUDABlasHandle();
            blas::PointerModeGuard pointerModeGuard(handle, CUBLAS_POINTER_MODE_DEVICE);
            blas::dot<Scalar>(
                handle,
                n,
                self.data_ptr<Scalar>(),
                incx,
                other.data_ptr<Scalar>(),
                incy,
                result.data_ptr<Scalar>());

            return result;
          });
        */
}

pub fn vdot_cuda(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            if (!self.is_complex()) {
        return dot_cuda(self, other);
      }

      NoNamesGuard guard;
      dot_check(self, other);

      const int n = static_cast<int>(self.numel());
      int incx = static_cast<int>(self.stride(0));
      int incy = static_cast<int>(other.stride(0));
      if (n == 1) {
        incx = 1;
        incy = 1;
      }

      return AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "vdot", [&] {
        Tensor result = empty({}, self.options());

        auto handle = getCurrentCUDABlasHandle();
        blas::PointerModeGuard pointerModeGuard(
            handle, CUBLAS_POINTER_MODE_DEVICE);
        blas::vdot<Scalar>(
            handle,
            n,
            self.data_ptr<Scalar>(),
            incx,
            other.data_ptr<Scalar>(),
            incy,
            result.data_ptr<Scalar>());

        return result;
      });
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(addmv_out_cuda)(const Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar& beta_, const Scalar& alpha_, const Tensor& result) {
      MaybeOwned<Tensor> self_ = expand_size(self, {mat.size(0)});
      auto betaval = beta_.toComplexDouble();
      if (mat.numel() == 0) {
        // shortcut for an empty matrix
        // By definition, when beta==0, values in self should be ignored. nans and infs
        // should not propagate
        if (betaval == 0.0) {
          result.zero_();
        } else {
          mul_out(
              const_cast<Tensor&>(result),
              self,
              native::scalar_tensor(
                  beta_, self.scalar_type(), nullopt /* layout */, kCPU, nullopt /* pin_memory */));
        }
      } else {
        if (!result.is_same(*self_) && betaval != 0.0) { //if beta is 0, result contents will be zeroed later
          native::copy_(const_cast<Tensor&>(result), *self_);
        }
        if (result.numel() != 0) {
          auto r_stride = result.stride(0);
          auto vec_stride = vec.stride(0);

          // Check for contiguity of `vec` and update `vec_stride` accordingly
          const auto vec_contiguous = vec_stride == 0 ? vec.contiguous() : vec;
          vec_stride = vec_contiguous.stride(0);

          AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, mat.scalar_type(), "addmv_impl_cuda", [&] {
            auto beta = beta_.to<Scalar>();
            auto alpha = alpha_.to<Scalar>();
            if (mat.stride(0) == 1 && mat.stride(1) >= max<i64>(1, mat.size(0))) {
              blas::gemv<Scalar>('n',
                mat.size(0), mat.size(1), alpha, mat.data_ptr<Scalar>(), mat.stride(1), vec_contiguous.data_ptr<Scalar>(),
                vec_stride, beta, result.data_ptr<Scalar>(), r_stride);
            }
            else if (mat.stride(1) == 1 && mat.stride(0) >= max<i64>(1, mat.size(1))) {
              blas::gemv<Scalar>('t',
                mat.size(1), mat.size(0), alpha, mat.data_ptr<Scalar>(), mat.stride(0),
                vec_contiguous.data_ptr<Scalar>(), vec_stride, beta, result.data_ptr<Scalar>(), r_stride);
            }
            else {
              Tensor cmat = mat.contiguous();
              blas::gemv<Scalar>('t',
                  mat.size(1), mat.size(0), alpha, cmat.data_ptr<Scalar>(), cmat.stride(0),
                  vec_contiguous.data_ptr<Scalar>(), vec_stride, beta, result.data_ptr<Scalar>(), r_stride);
            }
          });
        }
      }
    }
    */
}
