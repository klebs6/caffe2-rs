crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Blas.cpp]

lazy_static!{
    /*
    TORCH_META_FUNC(addmv)(const Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar& beta, const Scalar& alpha) {
      TORCH_CHECK((mat.dim() == 2 && vec.dim() == 1 && self.dim() <= 1),
        "vector + matrix @ vector expected, got ", self.dim(), ", ", mat.dim(), ", ", vec.dim());

      TORCH_CHECK(mat.size(1) == vec.size(0) && (mat.size(0) == self.numel() || self.numel() == 1),
         "size mismatch, got ", self.size(0), ", ", mat.size(0), "x", mat.size(1), ",", vec.size(0));
      auto names = namedinference::propagate_names_for_addmv(mat, vec, self);
      set_output(0, IntArrayRef(mat.sizes().data(), 1), {}, mat.options(), names);
      auto result = maybe_get_output(0);
      //this check can fire for inplace op only, for all other versions result is guaranteed to be correct size
      TORCH_CHECK(result.dim() == 1 && result.sizes()[0] == mat.sizes()[0], "output of addmv operation should be 1D with ",
      "size equal to mat.size(0), yet got output size ", result.sizes(), " and mat.size(0) ", mat.size(0));
    }
    */
}

pub fn gemv(
        trans: u8,
        m:     i64,
        n:     i64,
        alpha: Scalar,
        a:     *mut Scalar,
        lda:   i64,
        x:     *mut Scalar,
        incx:  i64,
        beta:  Scalar,
        y:     *mut Scalar,
        incy:  i64)  {
    
    todo!();
        /*
        
        */
}

pub fn dot_impl(
        n:    i64,
        x:    *mut Scalar,
        incx: i64,
        y:    *mut Scalar,
        incy: i64) -> Scalar {
    
    todo!();
        /*
        
        */
}

pub fn vdot_impl(
        n:    i64,
        x:    *mut Scalar,
        incx: i64,
        y:    *mut Scalar,
        incy: i64) -> Scalar {
    
    todo!();
        /*
        
        */
}

#[inline] pub const fn lda_cond(
        m:   i64,
        n:   i64,
        lda: i64) -> bool {
    
    todo!();
        /*
            return n == 1 || lda >= max<i64>(1L, m);
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(addmv_out_cpu)(const Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar& beta_, const Scalar& alpha_, const Tensor& result) {
      MaybeOwned<Tensor> self_ = expand_size(self, {mat.size(0)});
      auto betaval = beta_.toComplexDouble();
      if (mat.numel() == 0) {
        // shortcut for an empty matrix
        // By definition, when beta==0, values in self should be ignored. nans and infs
        // should not propagate
        if (betaval == 0.0) {
          result.zero_();
        } else {
          cpu::mul_out(
              const_cast<Tensor&>(result),
              self,
              native::scalar_tensor(
                  beta_, self.scalar_type(), nullopt /* layout */, kCPU, nullopt /* pin_memory */));
        }
      } else {
        if (!result.is_same(*self_) && betaval != 0.0) { //if beta is 0, result contents is ignored
          native::copy_(const_cast<Tensor&>(result), *self_);
        }
        if (result.numel() != 0) {
          auto r_stride = result.stride(0);
          AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, mat.scalar_type(), "addmv_impl_cpu", [&] {
            auto beta = beta_.to<Scalar>();
            auto alpha = alpha_.to<Scalar>();
            if (mat.stride(0) == 1 && lda_cond(mat.size(0), mat.size(1), mat.stride(1))) {
              gemv<Scalar>('n', mat.size(0), mat.size(1), alpha, mat.data_ptr<Scalar>(), mat.stride(1),
                  vec.data_ptr<Scalar>(), vec.stride(0), beta, result.data_ptr<Scalar>(), r_stride);
            }
            else if (mat.stride(1) == 1 && lda_cond(mat.size(1), mat.size(0), mat.stride(0))) {
              gemv<Scalar>('t', mat.size(1), mat.size(0), alpha, mat.data_ptr<Scalar>(), mat.stride(0),
                  vec.data_ptr<Scalar>(), vec.stride(0), beta, result.data_ptr<Scalar>(), r_stride);
            }
            else {
              Tensor cmat = mat.contiguous();
              gemv<Scalar>('t', mat.size(1), mat.size(0), alpha, cmat.data_ptr<Scalar>(), cmat.stride(0),
                  vec.data_ptr<Scalar>(), vec.stride(0), beta, result.data_ptr<Scalar>(), r_stride);
            }
          });
        }
      }
    }
    */
}

pub fn mv_out<'a>(
        self_:  &Tensor,
        vec:    &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            //self arg sent to addmv_out cannot be resized
      //here we use result as self argument for addmv, and result is user supplied and can be wrong size
      //it's not a hard error, because we allow resizing result, but it becomes a hard error
      //in addmv, because addmv expects self to satisfy proper conditions
      //to avoid this, supply correctly sized self, its contents doesn't matter because beta is 0
      if (result.dim() > 1 || (result.numel() != self.size(0) || result.numel() !=1)) {
        Tensor self_addmv = empty({self.size(0)}, self.options());
        return addmv_out(result, self_addmv, self, vec, 0, 1);
      }
      return addmv_out(result, result, self, vec, 0, 1);
        */
}

pub fn mv(
        self_: &Tensor,
        vec:   &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({self.size(0)}, self.options());
      //inplace version is more efficient if we can use it
      return addmv_(result, self, vec, 0, 1);
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
        */
}


pub fn dot(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            NoNamesGuard guard;

      dot_check(self, other);

      return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(ScalarType::Half, self.scalar_type(), "dot", [&] {
        Tensor result = empty({}, self.options());
        result.fill_(dot_impl<Scalar>(self.numel(), self.data_ptr<Scalar>(), self.stride(0), other.data_ptr<Scalar>(), other.stride(0)));
        return result;
      });
        */
}

pub fn vdot(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            NoNamesGuard guard;

      // Dispatch to `dot` for real dtypes.
      if (!self.is_complex()){
        return dot(self, other);
      }

      // For complex dtypes.
      dot_check(self, other);
      return AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "vdot", [&] {
        Tensor result = empty({}, self.options());
        result.fill_(vdot_impl<Scalar>(self.numel(), self.data_ptr<Scalar>(), self.stride(0), other.data_ptr<Scalar>(), other.stride(0)));
        return result;
      });
        */
}
