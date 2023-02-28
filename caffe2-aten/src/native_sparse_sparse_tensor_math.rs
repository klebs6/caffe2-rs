crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/sparse/SparseTensorMath.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/sparse/SparseTensorMath.cpp]

#[inline] pub fn get_result_tensor_for_unary_op(input: &SparseTensor) -> SparseTensor {
    
    todo!();
        /*
        if (isIntegralType(input.scalar_type(), /*includeBool=*/true)) {
          return empty_like(input, input.options().dtype(get_default_dtype()));
        }
        return empty_like(input);
        */
}

// --------------------------------------------------------------------
// zero_(SparseTensor)
// --------------------------------------------------------------------

// hummu hummu
pub fn zero_sparse(self_: &mut SparseTensor) -> &mut SparseTensor {
    
    todo!();
        /*
            AT_ASSERT(self.is_sparse());
      zeros_out(self, get_sparse_impl(self)->sizes());
      return self._coalesced_(true);
        */
}

// NB: Don't need zeros, zeros_like, already implemented in TensorFactories

// --------------------------------------------------------------------
// mul(SparseTensor, Scalar)
// --------------------------------------------------------------------

pub fn mul_out_sparse_zerodim(
    r:     &mut SparseTensor,
    t:     &SparseTensor,
    value: &Tensor) -> &mut SparseTensor {
    
    todo!();
        /*
            AT_ASSERT(r.is_sparse());
      AT_ASSERT(t.is_sparse());
      AT_ASSERT(value.dim() == 0);

      if (is_same_tensor(r, t)) {
        r._values().mul_(value);
      } else {
        r.resize_as_(t);
        auto indices = r._indices();
        indices.resize_as_(t._indices());
        indices.copy_(t._indices());
        Tensor r_values = r._values(); // Sigh... needed because mul_out takes Tensor&
        mul_out(r_values, t._values(), value);
        get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
        r._coalesced_(t.is_coalesced());
      }
      return r;
        */
}

pub fn mul_out_sparse_scalar(
    r:     &mut SparseTensor,
    t:     &SparseTensor,
    value: &Scalar) -> &mut SparseTensor {
    
    todo!();
        /*
            return mul_out_sparse_zerodim(r, t, wrapped_scalar_tensor(value));
        */
}

// --------------------------------------------------------------------
// log1p(SparseTensor)
// --------------------------------------------------------------------

/**
  | In-place log1p on uncoalesced tensors is not
  | supported since the operation is not a linear
  | map.
  |
  | Values of uncoalesced tensor corresponding to
  | the same indices are summed and
  | log1p(summed_value) != log1p(v1) + log1p(v2)
  */
pub fn log1p_out_sparse(
    t: &SparseTensor,
    r: &mut SparseTensor) -> &mut SparseTensor {

    todo!();
        /*
            TORCH_CHECK(r.is_sparse(), "Tensor should be sparse");
      TORCH_CHECK(t.is_sparse(), "Tensor should be sparse");
      TORCH_CHECK(
          !isIntegralType(r.scalar_type(), /*includeBool=*/true),
          "log1p: result type cannot be Integral, got:",
          r.scalar_type());

      if (is_same_tensor(r, t)) {
        // don't have in-place log1p for uncoalesced input because coalesce() is not in-place
        TORCH_CHECK(r.is_coalesced(), "log1p: in-place on uncoalesced tensors is not supported");
      }
      else {
        copy_sparse_to_sparse_(r, t.coalesce());
      }
      r._values().log1p_();
      return r;
        */
}

pub fn log1p_sparse(t: &SparseTensor) -> SparseTensor {
    
    todo!();
        /*
            auto result = get_result_tensor_for_unary_op(t);
      return log1p_out_sparse(t, result);
        */
}

pub fn log1p_sparse_mut(t: &mut SparseTensor) -> &mut SparseTensor {
    
    todo!();
        /*
            return log1p_out_sparse(t, t);
        */
}

// --------------------------------------------------------------------
// neg(SparseTensor)
// --------------------------------------------------------------------

pub fn neg_out_sparse(
    t: &SparseTensor,
    r: &mut SparseTensor) -> &mut SparseTensor {
    
    todo!();
        /*
            TORCH_CHECK(r.is_sparse(), "Tensor should be sparse");
      TORCH_CHECK(t.is_sparse(), "Tensor should be sparse");

      // copy_sparse_ does not perform the copy if it is the same tensor
      copy_sparse_to_sparse_(r, t);
      r._values().neg_();
      return r;
        */
}

pub fn neg_sparse_a(t: &SparseTensor) -> SparseTensor {
    
    todo!();
        /*
            SparseTensor r = get_result_tensor_for_unary_op(t);
      neg_out_sparse(t, r);
      return r;
        */
}

pub fn neg_sparse_b(t: &mut SparseTensor) -> &mut SparseTensor {
    
    todo!();
        /*
            return neg_out_sparse(t, t);
        */
}

// --------------------------------------------------------------------
// asin(SparseTensor)
// --------------------------------------------------------------------

/**
  | In-place asin on uncoalesced tensors is not
  | supported since the operation is not a linear
  | map.
  |
  | Values of uncoalesced tensor corresponding to
  | the same indices are summed and
  | asin(summed_value) != asin(v1) + asin(v2)
  */
pub fn asin_out_sparse(
    t: &SparseTensor,
    r: &mut SparseTensor) -> &mut SparseTensor {

    todo!();
        /*
            TORCH_CHECK(r.is_sparse(), "Tensor should be sparse");
      TORCH_CHECK(t.is_sparse(), "Tensor should be sparse");
      TORCH_CHECK(
          !isIntegralType(r.scalar_type(), /*includeBool=*/true),
          "asin: result type cannot be Integral, got:",
          r.scalar_type());

      if (is_same_tensor(r, t)) {
        // don't have in-place asin for uncoalesced input because coalesce() is not in-place, see above comment
        TORCH_CHECK(r.is_coalesced(), "asin: in-place on uncoalesced tensors is not supported");
      } else {
        copy_sparse_to_sparse_(r, t.coalesce());
      }
      r._values().asin_();
      return r;
        */
}

pub trait AsinSparse {

    fn asin_sparse(&self) -> SparseTensor;
}

impl AsinSparse for SparseTensor {

    fn asin_sparse(&self) -> SparseTensor {
        
        todo!();
            /*
                auto result = get_result_tensor_for_unary_op(t);
          return asin_out_sparse(t, result);
            */
    }
}

pub trait AsinSparseInplace {

    fn asin_sparse(&mut self) -> &mut Self;
}

impl AsinSparseInplace for SparseTensor {

    fn asin_sparse(&mut self) -> &mut SparseTensor {
        
        todo!();
            /*
                return asin_out_sparse(t, t);
            */
    }
}

// --------------------------------------------------------------------
// sqrt(SparseTensor)
// --------------------------------------------------------------------

// TODO: add in-place variant

pub fn sqrt_out_sparse(
    t: &SparseTensor,
    r: &mut SparseTensor) -> &mut SparseTensor {
    
    todo!();
        /*
            TORCH_CHECK(r.is_sparse(), "Tensor should be sparse");
      TORCH_CHECK(t_.is_sparse(), "Tensor should be sparse");

      // This coalesce is why we can't easily provide an inplace variant
      SparseTensor t = t_.coalesce();

      r.resize_as_(t);
      auto indices = r._indices();
      indices.resize_as_(t._indices());
      indices.copy_(t._indices());
      Tensor r_values = r._values();
      sqrt_out(r_values, t._values());
      get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
      return r._coalesced_(t.is_coalesced());
        */
}

pub fn sqrt_sparse(t: &SparseTensor) -> SparseTensor {
    
    todo!();
        /*
            SparseTensor r = get_result_tensor_for_unary_op(t);
      sqrt_out_sparse(t, r);
      return r;
        */
}


// --------------------------------------------------------------------
// pow(SparseTensor, Scalar)
// --------------------------------------------------------------------

// TODO: add in-place variant

pub fn pow_out_sparse_scalar(
        t:     &SparseTensor,
        value: &Scalar,
        r:     &mut SparseTensor) -> &mut SparseTensor {
    
    todo!();
        /*
            AT_ASSERT(r.is_sparse());
      AT_ASSERT(t_.is_sparse());
      TORCH_CHECK(value.toDouble() != 0, "pow: cannot raise to zeroth power on sparse tensor; it would make the result tensor dense");

      // This coalesce is why we can't easily provide an inplace variant
      SparseTensor t = t_.coalesce();

      r.resize_as_(t);
      auto indices = r._indices();
      indices.resize_as_(t._indices());
      indices.copy_(t._indices());
      Tensor r_values = r._values(); // Sigh... needed because pow_out takes Tensor&
      pow_out(r_values, t._values(), value);
      get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
      return r._coalesced_(t.is_coalesced());
        */
}


pub fn pow_sparse_scalar(
        t:     &SparseTensor,
        value: &Scalar) -> SparseTensor {
    
    todo!();
        /*
            SparseTensor r = empty({0}, t.options());
      pow_out_sparse_scalar(t, value, r);
      return r;
        */
}

// --------------------------------------------------------------------
// div(SparseTensor, Scalar)
// --------------------------------------------------------------------

pub fn coalesce(tensor: &mut SparseTensor) -> &mut SparseTensor {
    
    todo!();
        /*
            if (tensor.is_coalesced()) {
        return tensor;
      }

      SparseTensor coalesced = tensor.coalesce();
      tensor._values().resize_as_(coalesced._values());
      tensor._indices().resize_as_(coalesced._indices());
      tensor._values().copy_(coalesced._values());
      tensor._indices().copy_(coalesced._indices());
      tensor._coalesced_(true);
      return tensor;
        */
}

/**
  | Note [Sparse Floor Division]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  |
  | Uncoalesced sparse tensors cannot be floor
  | divided correctly. Integer division is
  | considered a special-case of floor division for
  | purposes of this note.
  |
  | For example, an integer tensor with values=[3,
  | 3] divided by 2 would produce values=[1, 1],
  | which sum to 2 instead of 3 (=6/2).
  |
  | A float tensor with values=[3., 3.] floor
  | divided by 2 would also produce values=[1., 1.]
  | (after truncation), which sum to 2.f instead of
  | 3.f.
  |
  | To perform floor division the sparse tensor
  | must be coalesced first.
  */
pub fn div_out_sparse_zerodim_a(
        t:             &SparseTensor,
        value:         &Tensor,
        rounding_mode: Option<StringView>,
        r:             &mut SparseTensor) -> &mut SparseTensor {
    
    todo!();
        /*
            TORCH_CHECK(value.dim() == 0, "Sparse division requires a scalar or ",
        "zero-dim dense tensor divisor (got shape ", value.sizes(), " for divisor)");
      TORCH_CHECK(!value.is_sparse(), "Sparse division requires a scalar or ",
        "zero-dim dense tensor divisor (got a sparse divisor)");

      AT_ASSERT(r.is_sparse());
      AT_ASSERT(t.is_sparse());

      // See note "Sparse Floor Division"
      const bool should_coalesce = rounding_mode.has_value() && !t.is_coalesced();
      if (is_same_tensor(r, t)) {
        if (should_coalesce) {
          coalesce_(r);
        }
        r._values().div_(value, rounding_mode);
      } else {
        Tensor t_tmp = t;
        if (should_coalesce) {
          t_tmp = t.coalesce();
        }
        r.resize_as_(t_tmp);
        auto indices = r._indices();
        indices.resize_as_(t_tmp._indices());
        indices.copy_(t_tmp._indices());
        Tensor r_values = r._values(); // Sigh... needed because div_out takes Tensor&
        div_out(r_values, t_tmp._values(), value, rounding_mode);
        get_sparse_impl(r)->set_nnz_and_narrow(t_tmp._nnz());
        r._coalesced_(t_tmp.is_coalesced());
      }
      return r;
        */
}

pub fn div_out_sparse_zerodim_b(
        t:     &SparseTensor,
        value: &Tensor,
        r:     &mut SparseTensor) -> &mut SparseTensor {
    
    todo!();
        /*
            return div_out_sparse_zerodim(t, value, /*rounding_mode=*/nullopt, r);
        */
}

pub fn div_sparse_a(
        self_: &Tensor,
        value: &Tensor) -> Tensor {
    
    todo!();
        /*
            auto commonDtype = result_type(self, value);
      if (isIntegralType(commonDtype, /*includeBool=*/true)) {
        commonDtype = typeMetaToScalarType(get_default_dtype());
      }
      Tensor result = empty({0}, self.options().dtype(commonDtype));
      return div_out_sparse_zerodim(self, value, result);
        */
}

pub fn div_sparse_b(
        self_: &mut Tensor,
        value: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return div_out_sparse_zerodim(self, value, self);
        */
}

pub fn div_out_sparse_scalar_a(
        t:     &SparseTensor,
        value: Scalar,
        r:     &mut SparseTensor) -> &mut SparseTensor {
    
    todo!();
        /*
            return div_out_sparse_zerodim(t, wrapped_scalar_tensor(value), r);
        */
}

pub fn div_sparse_c(
        self_:         &Tensor,
        value:         &Tensor,
        rounding_mode: Option<StringView>) -> Tensor {
    
    todo!();
        /*
            auto commonDtype = result_type(self, value);
      if (isIntegralType(commonDtype, /*include_bool=*/true) && !rounding_mode.has_value()) {
        commonDtype = typeMetaToScalarType(get_default_dtype());
      }
      Tensor result = empty({0}, self.options().dtype(commonDtype));
      return div_out_sparse_zerodim(self, value, move(rounding_mode), result);
        */
}

pub fn div_sparse_d(
        self_:         &mut Tensor,
        value:         &Tensor,
        rounding_mode: Option<StringView>) -> &mut Tensor {
    
    todo!();
        /*
            return div_out_sparse_zerodim(self, value, move(rounding_mode), self);
        */
}

pub fn div_out_sparse_scalar_b(
    t:             &SparseTensor,
    value:         Scalar,
    rounding_mode: Option<StringView>,
    r:             &mut SparseTensor) -> &mut SparseTensor {

    todo!();
        /*
            return div_out_sparse_zerodim(t, wrapped_scalar_tensor(value), move(rounding_mode), r);
        */
}

// --------------------------------------------------------------------
// floor_divide(SparseTensor, Scalar)
// --------------------------------------------------------------------

pub fn floor_divide_out_sparse_zerodim(
    dividend: &SparseTensor,
    divisor:  &Tensor,
    result:   &mut SparseTensor) -> &mut SparseTensor {
    
    todo!();
        /*
            TORCH_CHECK(divisor.dim() == 0, "Sparse floor division requires a scalar or ",
        "zero-dim dense tensor divisor (got shape ", divisor.sizes(), " for divisor)");
      TORCH_CHECK(!divisor.is_sparse(), "Sparse floor division requires a scalar or ",
        "zero-dim dense tensor divisor (got a sparse divisor)");

      AT_ASSERT(result.is_sparse());
      AT_ASSERT(dividend.is_sparse());

      // Case 1: result and dividend are the same tensor
      // Performs floor division in-place
      if (is_same_tensor(result, dividend)) {

        // See note "Sparse Floor Division"
        if (!result.is_coalesced()) {
          coalesce_(result);
        }

        result._values().floor_divide_(divisor);
        return result;
      }

      // Case 2: result and dividend are different tensors
      Tensor dividend_tmp = dividend;

      // Ensures dividend_tmp is coalesced (see note above)
      if (!dividend.is_coalesced()) {
        dividend_tmp = dividend.coalesce();
      }

      // Resizes and indexes result like dividend_tmp
      result.resize_as_(dividend_tmp);
      result._indices().resize_as_(dividend_tmp._indices());
      result._indices().copy_(dividend_tmp._indices());

      // Computes result
      Tensor result_values = result._values();
      floor_divide_out(result_values, dividend_tmp._values(), divisor);
      get_sparse_impl(result)->set_nnz_and_narrow(dividend_tmp._nnz());
      result._coalesced_(dividend_tmp.is_coalesced());
      return result;
        */
}


pub fn floor_divide_sparse_a(
        self_: &Tensor,
        value: &Tensor) -> Tensor {
    
    todo!();
        /*
            auto commonDtype = result_type(self, value);
      Tensor result = empty({0}, self.options().dtype(commonDtype));
      return floor_divide_out_sparse_zerodim(self, value, result);
        */
}

pub fn floor_divide_sparse_b(
        self_: &mut Tensor,
        value: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return floor_divide_out_sparse_zerodim(self, value, self);
        */
}


pub fn floor_divide_out_sparse_scalar(
        r:     &mut SparseTensor,
        t:     &SparseTensor,
        value: &Scalar) -> &mut SparseTensor {
    
    todo!();
        /*
            return floor_divide_out_sparse_zerodim(t, wrapped_scalar_tensor(value), r);
        */
}

// --------------------------------------------------------------------
// norm(SparseTensor, Scalar)
// --------------------------------------------------------------------

/// Only supports floating point, FYI
///
pub fn norm_sparse_a(
        self_: &SparseTensor,
        p:     &Scalar) -> Tensor {
    
    todo!();
        /*
            AT_ASSERT(self.is_sparse());
      return norm_sparse(self, p, IntArrayRef{}, false, nullopt);
        */
}

pub fn norm_sparse_b(
        self_:   &SparseTensor,
        p:       &Option<Scalar>,
        dim:     &[i32],
        keepdim: bool,
        dtype:   Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            AT_ASSERT(self.is_sparse());
      if (dim.size() > 0) {
        // Only full reductions are supported, so check if that is the case
        i64 ndim = self.dim();
        bool passed_full_reduction_check = static_cast<usize>(ndim) == dim.size();
        if (passed_full_reduction_check) {
          auto dim_ = dim.vec();
          maybe_wrap_dims(dim_, ndim);
          vector<bool> dims_check(ndim, false);
          // Need to check for duplicates, and fail if any are found
          for (auto dim_ind : dim_) {
            if (dims_check[dim_ind]) {
              passed_full_reduction_check = false;
              break;
            }
            dims_check[dim_ind] = true;
          }
        }
        TORCH_CHECK(passed_full_reduction_check,
          "norm_sparse currently only supports full reductions, so 'dim' must either be empty or contain all dimensions of the input");
      }
      TORCH_CHECK(keepdim == false, "norm_sparse currently does not support keepdim=True");
      TORCH_CHECK(!dtype.has_value(), "norm_sparse currently does not support 'dtype' argument");
      constexpr auto TWO = 2.0;
      auto p_ = p.value_or(TWO);
      return self.coalesce()._values().norm(p_);
        */
}

// --------------------------------------------------------------------
// mv(SparseTensor, Tensor)
// --------------------------------------------------------------------

pub fn mv_sparse(
        self_: &SparseTensor,
        vec:   &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.ndimension() == 2 &&
                  vec.ndimension() == 1,
                  "mv: two tensor dim should be 2 and 1, but got ",
                  "SparseTensor Dim: ", self.ndimension(), "Tensor Dim: ", vec.ndimension());

      TORCH_CHECK(vec.size(-1) == self.size(-1),
                  "mv: expected self.size(-1) == vec.size(-1)");

      auto result = self.matmul(vec.unsqueeze(-1));

      return result.squeeze(-1);
        */
}

// --------------------------------------------------------------------
// add(SparseTensor, SparseTensor, Scalar)  [broadcasts]
// --------------------------------------------------------------------

pub fn add_sparse_a(
    self_: &Tensor,
    other: &Tensor,
    alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            // TODO: Why?! Can't we just flip the order here...
      TORCH_CHECK(!(self.is_sparse() && !other.is_sparse()),
                  "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
      auto commonDtype = result_type(self, other);
      alpha_check(commonDtype, alpha);
      Tensor result = empty({0}, self.options().dtype(commonDtype));
      return add_out(result, self, other, alpha);  // redispatch!
        */
}

pub fn add_sparse_b(
    self_: &mut Tensor,
    other: &Tensor,
    alpha: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return add_out(self, self, other, alpha);  // redispatch!
        */
}

// There's actually nothing sparse specific about these implementations

pub fn sub_sparse_a(
        self_: &Tensor,
        other: &Tensor,
        alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            sub_check(self, other);
      return native::add_sparse(self, other, -alpha);
        */
}

pub fn sub_sparse_b(
        self_: &mut Tensor,
        other: &Tensor,
        alpha: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            sub_check(self, other);
      return native::add_sparse_(self, other, -alpha);
        */
}


pub fn sub_out_sparse(
        self_: &Tensor,
        other: &Tensor,
        alpha: &Scalar,
        r:     &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            sub_check(self, other);
      return add_out(r, self, other, -alpha);  // redispatch!
        */
}


pub fn add_out_sparse_contiguous(
        r:            &mut SparseTensor,
        t:            &SparseTensor,
        src:          &SparseTensor,
        value:        &Scalar,
        common_dtype: ScalarType) -> &mut SparseTensor {
    
    todo!();
        /*
            // saving those because they can be overwritten when doing in-place operations
        i64 t_nnz = t._nnz(), s_nnz = src._nnz(), max_nnz = t_nnz + s_nnz;
        bool coalesced = t.is_coalesced() && src.is_coalesced();
        i64 sparse_dim = src.sparse_dim();

        Tensor r_indices = empty({src.sparse_dim(), max_nnz}, t._indices().options());

        Tensor t_values = t._values().to(commonDtype);
        Tensor s_values = src._values().to(commonDtype);

        Tensor r_values = new_values_with_size_of(s_values, max_nnz).zero_();

        i64 blockSize = r_values.stride(0);
        i64 r_i = 0, t_i = 0, s_i = 0;
        auto t_indices = t._indices();
        auto src_indices = src._indices();

        // NB: relies on nnz tests above
        auto t_indices_accessor = t_indices.accessor<i64, 2>();
        auto r_indices_accessor = r_indices.accessor<i64, 2>();
        auto src_indices_accessor = src_indices.accessor<i64, 2>();

        AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
            commonDtype, "cadd_sparse", [&] {
              Scalar* t_values_ptr = t_values.data_ptr<Scalar>();
              Scalar* s_values_ptr = s_values.data_ptr<Scalar>();
              Scalar* r_values_ptr = r_values.data_ptr<Scalar>();
              Scalar cast_value = value.to<Scalar>();
              while (t_i < t_nnz || s_i < s_nnz) {
                i64 cmp;
                if (t_i >= t_nnz) {
                  cmp = -1;
                } else if (s_i >= s_nnz) {
                  cmp = 1;
                } else {
                  cmp = 0;
                  for (auto d: irange(sparse_dim)) {
                    if (t_indices_accessor[d][t_i] < src_indices_accessor[d][s_i]) {
                      cmp = 1;
                      break;
                    }
                    if (t_indices_accessor[d][t_i] > src_indices_accessor[d][s_i]) {
                      cmp = -1;
                      break;
                    }
                  }
                }
                if (cmp >= 0) {
                  for (auto d: irange(sparse_dim)) {
                    r_indices_accessor[d][r_i] = t_indices_accessor[d][t_i];
                  }
                  if (t_values.numel() > 0) {  // We add all elements from t_values to r_values only if t_values is not an empty tensor
                    native::cpublas::axpy<Scalar>(blockSize, 1,
                      t_values_ptr + t_i * blockSize, 1,
                      r_values_ptr + r_i * blockSize, 1);
                  }
                  t_i++;
                }
                if (cmp <= 0) {
                  for (auto d: irange(sparse_dim)) {
                    r_indices_accessor[d][r_i] = src_indices_accessor[d][s_i];
                  }
                  if (s_values.numel() > 0) {  // We add all elements from s_values to r_values only if s_values is not an empty tensor
                    native::cpublas::axpy<Scalar>(blockSize, cast_value,
                      s_values_ptr + s_i * blockSize, 1,
                      r_values_ptr + r_i * blockSize, 1);
                  }
                  s_i++;
                }
                r_i++;
              }
            }
        );

        if (r.scalar_type() != commonDtype) {
          r_values = r_values.to(r.scalar_type());
        }
        get_sparse_impl(r)->set_indices_and_values_unsafe(r_indices, r_values);
        get_sparse_impl(r)->set_nnz_and_narrow(r_i);

        // TODO: I think it may be possible to track inside the loop and
        // detect when we are uncoalesced (e.g., by observing that an
        // index goes backwards) which may be more precise than using the
        // coalesced flag here.  But this is easy.
        return r._coalesced_(coalesced);
        */
}


pub fn add_out_sparse_non_contiguous(
        r:            &mut SparseTensor,
        t:            &SparseTensor,
        src:          &SparseTensor,
        value:        &Scalar,
        common_dtype: ScalarType) -> &mut SparseTensor {
    
    todo!();
        /*
            Tensor t_values = t._values().to(commonDtype);
        Tensor s_values = src._values().to(commonDtype);

        // If `t` or `src` contains non-contiguous `values`, `native::cpublas::axpy` doesn't work
        // and we concat the indices and values tensors instead.
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
          commonDtype, "add_out_sparse_cpu", [&] {
              if (value.to<Scalar>() != static_cast<Scalar>(1)) {
                s_values = s_values.mul(value);
              }
            });

        Tensor r_indices = cat({t._indices(), src._indices()}, 1);
        Tensor r_values = cat({t_values, s_values}, 0).to(r.scalar_type());
        alias_into_sparse(r, r_indices, r_values);

        // Prevent unbounded growth of nnz
        // TODO: Improved heuristic on when to coalesce or remove need to coalesce
        if (r._nnz() > r.numel()) {
          auto c = r.coalesce();
          alias_into_sparse(r, c._indices(), c._values());
        }

        return r;
        */
}

pub fn add_out_sparse_cpu(
        t:     &SparseTensor,
        src:   &SparseTensor,
        value: &Scalar,
        r:     &mut SparseTensor) -> &mut SparseTensor {
    
    todo!();
        /*
            if (!t.is_sparse()) {
        return add_out_dense_sparse_cpu(r, t, src, value);
      }
      // TODO: This test seems a bit goofy
      TORCH_CHECK(src.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
      AT_ASSERT(!t.is_cuda());  // the dispatch argument
      TORCH_CHECK(!r.is_cuda(), "add: expected 'out' to be CPU tensor, but got CUDA tensor");
      TORCH_CHECK(!src.is_cuda(), "add: expected 'other' to be a CPU tensor, but got a CUDA tensor");

      TORCH_CHECK(t.sizes().equals(src.sizes()), "add: expected sizes of 'self' and 'other' to match, but ", t.sizes(), " != ", src.sizes());

      auto commonDtype = promoteTypes(t.scalar_type(), src.scalar_type());

      TORCH_CHECK(canCast(commonDtype, r.scalar_type()), "Can't convert result type ", commonDtype, " to output ", r.scalar_type(), " in add operation");

      if (src._nnz() == 0) {
        return copy_sparse_to_sparse_(r, t);
      }
      if (t._nnz() == 0) {
        return mul_out_sparse_scalar(r, src, value);
      }

      TORCH_CHECK(is_same_density(t, src), "add: expected 'self' and 'other' to have same density, but 'self' has ", t.sparse_dim(), " sparse dimensions while 'other' has ", src.sparse_dim(), " sparse dimensions");

      r.resize_as_(src);

      if (src._values().is_contiguous() && t._values().is_contiguous()) {
        return add_out_sparse_contiguous(r, t, src, value, commonDtype);
      } else {
        return add_out_sparse_non_contiguous(r, t, src, value, commonDtype);
      }
        */
}

// --------------------------------------------------------------------
// add(Tensor, SparseTensor, Scalar)
//    formerly known as spcadd
// --------------------------------------------------------------------

pub fn add_dense_sparse_worker_cpu<Scalar>(
        r:       &mut Tensor,
        value:   &Scalar,
        sparse:  &SparseTensor,
        indices: &Tensor,
        values:  &Tensor)  {

    todo!();
        /*
            auto indices_accessor = indices.accessor<i64, 2>();
      auto values_accessor = values.accessor<Scalar, 1>();

      Scalar* r_ptr = r.data_ptr<Scalar>();
      Scalar cast_value = value.to<Scalar>();

      parallel_for(0, sparse._nnz(), 0, [&](i64 start, i64 end) {
        for (auto k: irange(start, end)) {
          i64 index = r.storage_offset();
          for (auto d: irange(sparse.sparse_dim())) {
            index += r.stride(d) * indices_accessor[d][k];
          }
          r_ptr[index] += cast_value * values_accessor[k];
        }
      });
        */
}


pub fn add_out_dense_sparse_cpu(
        r:      &mut Tensor,
        dense:  &Tensor,
        sparse: &SparseTensor,
        value:  &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            AT_ASSERT(!r.is_sparse());
      AT_ASSERT(!dense.is_sparse());
      AT_ASSERT(sparse_.is_sparse());

      AT_ASSERT(!dense.is_cuda()); // dispatch argument
      TORCH_CHECK(!r.is_cuda(), "add: expected 'out' to be CPU tensor, but got CUDA tensor");
      TORCH_CHECK(!sparse_.is_cuda(), "add: expected 'other' to be a CPU tensor, but got a CUDA tensor");

      TORCH_CHECK(dense.sizes().equals(sparse_.sizes()), "add: expected 'self' and 'other' to have same size, but self has size ",
        dense.sizes(), " while other has size ", sparse_.sizes(), " (FYI: dense-sparse addition does not currently support broadcasting)");

      auto commonDtype = promoteTypes(dense.scalar_type(), sparse_.scalar_type());
      TORCH_CHECK(canCast(commonDtype, r.scalar_type()), "Can't convert result type ", commonDtype, " to output ", r.scalar_type(), " in add operation");

      r.resize_as_(dense);
      SparseTensor sparse = sparse_.coalesce();

      Tensor indices = sparse._indices();
      Tensor values = sparse._values();
      i64 nDim = dense.dim();
      i64 nDimI = sparse.sparse_dim();

      if (sparse._nnz() == 0) {
        if (!is_same_tensor(r, dense)) r.copy_(dense);
        return r;
      }

      Tensor valuesBuffer = values.to(commonDtype);
      Tensor resultBuffer = r;
      if (r.scalar_type() != commonDtype) {
        resultBuffer = dense.to(commonDtype);
      } else if (!is_same_tensor(r, dense)) {
        resultBuffer.copy_(dense);
      }

      // accessors rely on nnz test
      if (nDim > nDimI) {
        auto indices_accessor = indices.accessor<i64, 2>();
        for (i64 k = 0; k < sparse._nnz(); k++) {
          Tensor dstBuffer = resultBuffer;
          for (i64 d = 0; d < sparse.sparse_dim(); d++) {
            dstBuffer = dstBuffer.select(0, indices_accessor[d][k]);
          }
          Tensor srcBuffer = valuesBuffer.select(0, k);
          dstBuffer.add_(srcBuffer, value);
        }
      } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(ScalarType::Bool,
            commonDtype, "add_dense_sparse", [&] {
              add_dense_sparse_worker_cpu<Scalar>(resultBuffer, value, sparse, indices, valuesBuffer);
            });
      }
      if (r.scalar_type() != commonDtype) {
        r.copy_(resultBuffer);
      }
      return r;
        */
}

// --------------------------------------------------------------------
// mul(SparseTensor, SparseTensor)  [broadcasts]
// --------------------------------------------------------------------

pub fn mul_sparse_a(
    self_: &Tensor,
    other: &Tensor) -> Tensor {
    
    todo!();
        /*
            auto commonDtype = result_type(self, other);
      Tensor result = empty({0}, self.options().dtype(commonDtype));
      return mul_out(result, self, other);  // redispatch!
        */
}

pub fn mul_sparse_b(
    self_: &mut Tensor,
    other: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return mul_out(self, self, other);  // redispatch!
        */
}


pub fn mul_out_sparse_cpu(
        t:   &Tensor,
        src: &Tensor,
        r:   &mut SparseTensor) -> &mut SparseTensor {
    
    todo!();
        /*
            if (src_.dim() == 0) {
        return mul_out_sparse_zerodim(r, t_, src_);
      } else if (t_.dim() == 0) {
        return mul_out_sparse_zerodim(r, src_, t_);
      }

      TORCH_CHECK(t_.sizes().equals(src_.sizes()), "mul operands have incompatible sizes");
      AT_ASSERT(!t_.is_cuda()); // dispatch argument
      TORCH_CHECK(!r.is_cuda(), "mul: expected 'out' to be CPU tensor, but got CUDA tensor");
      TORCH_CHECK(!src_.is_cuda(), "mul: expected 'other' to be a CPU tensor, but got a CUDA tensor");

      TORCH_CHECK(t_.sizes().equals(src_.sizes()), "mul: expected 'self' and 'other' to have same sizes, but ", t_.sizes(), " != ", src_.sizes());

      if (src_._nnz() == 0 || t_._nnz() == 0) {
        r.resize_as_(src_);
        return r.zero_();
      }

      SparseTensor t = t_.coalesce();
      SparseTensor src = src_.coalesce();

      // saving those because they can be overwritten when doing in-place operations
      i64 t_nnz = t._nnz(), s_nnz = src._nnz();
      i64 max_nnz = min(t_nnz, s_nnz);  // multiply by zero is zero, and can be dropped
      i64 sparse_dim = src.sparse_dim();
      Tensor t_indices = t._indices();
      Tensor src_indices = src._indices();
      Tensor r_indices = empty({sparse_dim, max_nnz}, t_indices.options());

      i64 r_i = 0, t_i = 0, s_i = 0;

      auto commonDtype = promoteTypes(t_.scalar_type(), src_.scalar_type());
      TORCH_CHECK(canCast(commonDtype, r.scalar_type()), "Can't convert result type ", commonDtype, " to output ", r.scalar_type(), " in mul operation");

      Tensor t_values = t._values().to(commonDtype);
      Tensor s_values = src._values().to(commonDtype);

      Tensor r_buffer = new_values_with_size_of(t_values, max_nnz).zero_();

      // NB: relies on nnz test above
      auto t_indices_accessor = t_indices.accessor<i64, 2>();
      auto r_indices_accessor = r_indices.accessor<i64, 2>();
      auto src_indices_accessor = src_indices.accessor<i64, 2>();

      // Check if we can find matching indices, and if so, write an
      // entry to the result indices vector.  Returns true if matching
      // indices were found.
      auto index_preamble = [&]() {
        for (auto d: irange(sparse_dim)) {
          if (t_indices_accessor[d][t_i] < src_indices_accessor[d][s_i]) {
            t_i++;
            return false;
          }
          if (t_indices_accessor[d][t_i] > src_indices_accessor[d][s_i]) {
            s_i++;
            return false;
          }
        }
        for (auto d: irange(sparse_dim)) {
          r_indices_accessor[d][r_i] = t_indices_accessor[d][t_i];
        }
        return true;
      };

      if (t_values.dim() > 1) {
        while (t_i < t_nnz && s_i < s_nnz) {
          if (!index_preamble()) continue;
          r_buffer.select(0, r_i).addcmul_(t_values.select(0, t_i), s_values.select(0, s_i));
          r_i++;
          t_i++;
          s_i++;
        }
      } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
            commonDtype, "mul_out_sparse", [&] {
              auto r_accessor = r_buffer.accessor<Scalar, 1>();
              auto t_accessor = t_values.accessor<Scalar, 1>();
              auto s_accessor = s_values.accessor<Scalar, 1>();

              while (t_i < t_nnz && s_i < s_nnz) {
                if (!index_preamble()) continue;
                r_accessor[r_i] = t_accessor[t_i] * s_accessor[s_i];
                r_i++;
                t_i++;
                s_i++;
              }
            }
        );
      }

      r.resize_as_(src);
      Tensor r_values = r_buffer.to(r.scalar_type());
      get_sparse_impl(r)->set_indices_and_values_unsafe(r_indices, r_values);
      get_sparse_impl(r)->set_nnz_and_narrow(r_i);
      return r._coalesced_(true);
        */
}

/**
  | --------------------------------------------------------------------
  | addmm(D1, S, D2, beta, alpha) -> D  [broadcasts]
  |
  | D = beta * D1 + alpha * mm(S, D2)
  | --------------------------------------------------------------------
  */
pub fn s_addmm_out_sparse_dense_worker<Scalar>(
        nnz:     i64,
        dim_i:   i64,
        dim_j:   i64,
        dim_k:   i64,
        r:       &mut Tensor,
        beta:    &Scalar,
        t:       &Tensor,
        alpha:   &Scalar,
        indices: &Tensor,
        values:  &Tensor,
        dense:   &Tensor)  {

    todo!();
        /*
            // r_ = alpha * sparse * dense
      Scalar cast_alpha = alpha.to<Scalar>();
      Scalar cast_beta = beta.to<Scalar>();

      if (cast_beta == static_cast<Scalar>(0)) {
        r.zero_();
      } else if (cast_beta == static_cast<Scalar>(1)) {
        if (!is_same_tensor(r, t)) {
          r.copy_(t);
        }
      } else {
        mul_out(r, t, scalar_to_tensor(beta));
      }

      auto indices_accessor = indices.accessor<i64, 2>();

      auto values_accessor = values.accessor<Scalar, 1>();
      Scalar* dense_ptr = dense.data_ptr<Scalar>();
      Scalar* r_ptr = r.data_ptr<Scalar>();

      i64 dense_stride0 = dense.stride(0);
      i64 dense_stride1 = dense.stride(1);
      i64 r_stride0 = r.stride(0);
      i64 r_stride1 = r.stride(1);
      for (auto i: irange(nnz)) {
        Scalar val = values_accessor[i];
        i64 row = indices_accessor[0][i];
        i64 col = indices_accessor[1][i];
        if (col >= 0 && col < dim_j && row >= 0 && row < dim_i) {
          native::cpublas::axpy<Scalar>(dim_k,
                cast_alpha * val,
                dense_ptr + col * dense_stride0, dense_stride1,
                r_ptr + row * r_stride0, r_stride1);
        } else {
          if (col < 0 || col >= dim_j) {
            AT_ERROR("addmm: index out of column bound: ", col, " not between 1 and ", dim_j);
          } else {
            AT_ERROR("addmm: index out of row bound: ", row, " not between 1 and ", dim_i);
          }
        }
      }
        */
}


pub fn s_addmm_out_sparse_dense_cpu(
        r:      &mut Tensor,
        t:      &Tensor,
        sparse: &SparseTensor,
        dense:  &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            // TODO: This error message seems awfully opaque
      TORCH_CHECK(!t.is_cuda(),  "Expected all tensors to be on the same device. addmm expected 't' to be CPU tensor, but got CUDA tensor");
      TORCH_CHECK(!r.is_cuda(), "Expected all tensors to be on the same device. addmm: expected 'out' to be CPU tensor, but got CUDA tensor");
      TORCH_CHECK(!sparse_.is_cuda(), "Expected all tensors to be on the same device. addmm: expected 'mat1' to be a CPU tensor, but got a CUDA tensor");
      TORCH_CHECK(!dense.is_cuda(), "Expected all tensors to be on the same device. addmm: expected 'mat2' to be a CPU tensor, but got a CUDA tensor");

      TORCH_CHECK(sparse_.sparse_dim() == 2, "addmm: matrices expected, got ", sparse_.sparse_dim(), "D tensor");
      TORCH_CHECK(sparse_.dense_dim() == 0, "addmm: scalar values expected, got ", sparse_.dense_dim(), "D values");
      TORCH_CHECK(dense.dim() == 2, "addmm: matrices expected, got ", dense.dim(), "D tensor");

      // ixj * jxk = ixk
      i64 dim_i = sparse_.size(0);
      i64 dim_j = sparse_.size(1);
      i64 dim_k = dense.size(1);

      TORCH_CHECK(dense.size(0) == dim_j,
          "addmm: Argument #3 (dense): Expected dim 0 size ", dim_j, ", got ", dense.size(0));
      TORCH_CHECK(t.size(0) == dim_i,
          "addmm: Argument #1 (t): Expected dim 0 size ", dim_i, ", got ", t.size(0));
      TORCH_CHECK(t.size(1) == dim_k,
          "addmm: Argument #1 (t): Expected dim 1 size ", dim_k, ", got ", t.size(1));

      r.resize_({dim_i, dim_k});

      i64 nnz        = sparse_._nnz();

      if (nnz == 0) {
        mul_out(r, t, scalar_tensor(beta, r.options()));
        return r;
      }

      Tensor indices = sparse_._indices();
      Tensor values      = sparse_._values();

      AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
          values.scalar_type(), "addmm_sparse_dense", [&] {
            s_addmm_out_sparse_dense_worker<Scalar>(nnz, dim_i, dim_j, dim_k, r, beta, t, alpha, indices, values, dense);
          }
      );

      return r;
        */
}


pub fn addmm_out_sparse_dense_cpu(
        self_:  &Tensor,
        mat1:   &SparseTensor,
        mat2:   &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            MaybeOwned<Tensor> b_self = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
      return s_addmm_out_sparse_dense_cpu(result, *b_self, mat1, mat2, beta, alpha);
        */
}


pub fn s_addmm_sparse_dense_cpu_a(
        t:      &Tensor,
        sparse: &SparseTensor,
        dense:  &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor r = empty({0}, t.options());
      s_addmm_out_sparse_dense_cpu(r, t, sparse, dense, beta, alpha);
      return r;
        */
}


pub fn addmm_sparse_dense_cpu(
        self_: &Tensor,
        mat1:  &SparseTensor,
        mat2:  &Tensor,
        beta:  &Scalar,
        alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            MaybeOwned<Tensor> b_self = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
      return s_addmm_sparse_dense_cpu(*b_self, mat1, mat2, beta, alpha);
        */
}


pub fn s_addmm_sparse_dense_cpu_b(
        t:      &mut Tensor,
        sparse: &SparseTensor,
        dense:  &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return s_addmm_out_sparse_dense_cpu(t, t, sparse, dense, beta, alpha);
        */
}

// NB: Purposely no broadcasting version of addmm inplace

pub fn sparse_addmm(
        t:      &Tensor,
        sparse: &SparseTensor,
        dense:  &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar) -> Tensor {
    
    todo!();
        /*
            // _sparse_addmm forward is functionally equivalent to addmm; it's
      // just the backward that is different.  This technically does an
      // unnecessary redispatch, I was too lazy to make it not do that
      return addmm(t, sparse, dense, beta, alpha);
        */
}


pub fn sparse_mm(
        sparse: &SparseTensor,
        dense:  &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor t = zeros({}, dense.options());
      return _sparse_addmm(t, sparse, dense, 0, 1);  // redispatch!
        */
}

/**
  | NB: Despite its suggestive name, this actually
  | only exists so that we can redispatch to
  | addmm_out;
  |
  | this is NOT an implementation of the sparse
  | masking version of mm
  |
  */
pub fn sparse_mm_out(
        sparse: &SparseTensor,
        dense:  &Tensor,
        result: &mut SparseTensor) -> &mut SparseTensor {
    
    todo!();
        /*
            Tensor t = zeros({}, dense.options());
      return addmm_out(result, t, sparse, dense, 0, 1);  // redispatch!
        */
}

// --------------------------------------------------------------------
// hspmm(SparseTensor mat1, Tensor mat2)
// --------------------------------------------------------------------

pub fn hspmm_out_sparse_cpu(
        sparse: &SparseTensor,
        dense:  &Tensor,
        r:      &mut SparseTensor) -> &mut SparseTensor {
    
    todo!();
        /*
            // TODO: Make this a real argument
      Scalar alpha = 1;

      AT_ASSERT(!sparse_.is_cuda()); // dispatch argument
      TORCH_CHECK(!r.is_cuda(), "hspmm: expected 'out' to be CPU tensor, but got CUDA tensor");
      TORCH_CHECK(!dense.is_cuda(), "hspmm: expected 'other' to be a CPU tensor, but got a CUDA tensor");

      TORCH_CHECK(sparse_.sparse_dim() == 2,
          "hspmm: Argument #2: matrices expected, got ", sparse_.sparse_dim(), "D tensor");
      TORCH_CHECK(sparse_.dense_dim() == 0,
          "hspmm: Argument #2: scalar values expected, got ", sparse_.dense_dim(), "D values");
      TORCH_CHECK(dense.dim() == 2,
          "hspmm: Argument #3: matrices expected, got ", dense.dim(), "D tensor");

      i64 m = sparse_.size(0);
      i64 k = sparse_.size(1);
      i64 n = dense.size(1);

      TORCH_CHECK(dense.size(0) == k,
          "hspmm: Argument #3: Expected dim 0 size ", k, ", got ", dense.size(0));

      get_sparse_impl(r)->raw_resize_(1, 1, {m, n});

      SparseTensor sparse = sparse_.coalesce();

      i64 nnz = sparse._nnz();

      if (nnz == 0) {
        r.zero_();
        return r;
      }

      Tensor indices = empty({1, nnz}, initialTensorOptions().dtype(kLong));

      // Initialize the sparse matrix that will be used with spaddmm to send rows
      // from the dense matrix to rows of the output's value tensor
      SparseTensor newSparse = sparse.clone();
      Tensor spIndices = newSparse._indices();
      Tensor valueIndices = spIndices.select(0, 0);

      // Compute output indices
      auto valueIndices_accessor = valueIndices.accessor<i64, 1>();
      auto indices_accessor = indices.accessor<i64, 2>();

      i64 i = -1, prevIdx = -1;
      for (i64 j = 0; j < nnz; j++) {
        i64 currIdx = valueIndices_accessor[j];
        if (currIdx != prevIdx) {
          indices_accessor[0][++i] = currIdx;
          prevIdx = currIdx;
        }
        valueIndices_accessor[j] = i;
      }
      i64 outNnz = i + 1;
      indices.resize_({1, outNnz});
      Tensor values = empty({outNnz, n}, dense.options());

      vector<i64> new_size = get_sparse_impl(newSparse)->sizes().vec();
      new_size[0] = outNnz;
      get_sparse_impl(newSparse)->raw_resize_(get_sparse_impl(newSparse)->sparse_dim(), get_sparse_impl(newSparse)->dense_dim(), new_size);

      // Compute output values tensor with sparse * dense multiplication
      s_addmm_out_sparse_dense_cpu(values, values, newSparse, dense, 0, alpha);
      get_sparse_impl(r)->set_indices_and_values_unsafe(indices, values);

      return r;
        */
}

pub fn hspmm_sparse_cpu(
    sparse: &SparseTensor,
    dense:  &Tensor) -> SparseTensor {
    
    todo!();
        /*
            SparseTensor r = empty({0}, sparse.options());
      hspmm_out_sparse_cpu(sparse, dense, r);
      return r;
        */
}

// --------------------------------------------------------------------
// sspaddmm(S1, S2, D, beta, alpha) -> S
//
// S = beta * S1 + alpha * mm(S2, D)
// --------------------------------------------------------------------

pub fn sspaddmm_out_cpu(
        t:      &SparseTensor,
        sparse: &SparseTensor,
        dense:  &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar,
        r:      &mut SparseTensor) -> &mut SparseTensor {
    
    todo!();
        /*
            AT_ASSERT(!t.is_cuda()); // dispatch argument
      TORCH_CHECK(!r.is_cuda(), "sspaddmm: expected 'out' to be CPU tensor, but got CUDA tensor");
      TORCH_CHECK(!sparse_.is_cuda(), "sspaddmm: expected 'mat1' to be a CPU tensor, but got a CUDA tensor");
      TORCH_CHECK(!dense.is_cuda(), "sspaddmm: expected 'mat2' to be a CPU tensor, but got a CUDA tensor");

      TORCH_CHECK(sparse_.sparse_dim() == 2,
          "sspaddmm: Argument #2: matrices expected, got ", sparse_.sparse_dim(), "D tensor");
      TORCH_CHECK(sparse_.dense_dim() == 0,
          "sspaddmm: Argument #2: scalar values expected, got ", sparse_.dense_dim(), "D values");
      TORCH_CHECK(dense.dim() == 2,
          "sspaddmm: Argument #2: matrices expected, got ", dense.dim(), "D tensor");

      SparseTensor sparse = sparse_.coalesce();

      // ixj * jxk = ixk
      i64 dim_i = sparse.size(0);
      i64 dim_j = sparse.size(1);
      i64 dim_k = dense.size(1);

      // NB: This has to occur before the checks, because r may alias t.
      // See test_saddmm
      get_sparse_impl(r)->raw_resize_(2, 0, {dim_i, dim_k});

      TORCH_CHECK(dense.size(0) == dim_j,
          "sspaddmm: Argument #3: Expected dim 0 size ", dim_j, ", got ", dense.size(0));
      TORCH_CHECK(t.size(0) == dim_i,
          "sspaddmm: Argument #1: Expected dim 0 size ", dim_i, ", got ", t.size(0));
      TORCH_CHECK(t.size(1) == dim_k,
          "sspaddmm: Argument #1: Expected dim 1 size ", dim_k, ", got ", t.size(1));

      i64 nnz        = sparse._nnz();
      // We have to make indices contiguous as we use indices.data_ptr in _to_csr which assumes row-contiguous storage
      Tensor indices = sparse._indices().contiguous();
      Tensor values      = sparse._values();

      Tensor csr = coo_to_csr(indices.data_ptr<i64>(), dim_i, nnz);

      i64 t_nnz = t._nnz();
      i64 r_nnz = nnz * dim_k + t_nnz;
      Tensor newi = empty({2, r_nnz}, kLong);
      Tensor newv = native::zeros(
          {r_nnz},
          optTypeMetaToScalarType(values.options().dtype_opt()),
          values.options().layout_opt(),
          values.options().device_opt(),
          values.options().pinned_memory_opt());

      if (t_nnz != 0) {
        Tensor narrowi = newi.narrow(1, 0, t_nnz);
        Tensor narrowv = newv.narrow(0, 0, t_nnz);

        narrowi.copy_(t._indices());
        narrowv.copy_(t._values());
        newv.mul_(beta);
      }

      // sparse = sparse * dense
      i64 p = t_nnz;

      auto csr_accessor = csr.accessor<i64, 1>();
      auto indices_accessor = indices.accessor<i64, 2>();
      auto newi_accessor = newi.accessor<i64, 2>();

      i64 dense_stride0 = dense.stride(0);
      i64 dense_stride1 = dense.stride(1);
      i64 newv_stride0 = newv.stride(0);

      AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
          values.scalar_type(), "sspmm", [&] {
            auto values_accessor = values.accessor<Scalar, 1>();
            Scalar* dense_ptr = dense.data_ptr<Scalar>();
            Scalar* newv_ptr = newv.data_ptr<Scalar>();
            Scalar cast_alpha = alpha.to<Scalar>();

            for (i64 h = 0; h < dim_i; h++) {
              i64 i_start = csr_accessor[h];
              i64 i_end = csr_accessor[h+1];
              for (i64 i = i_start; i < i_end; i++) {
                Scalar val = values_accessor[i];
                i64 col = indices_accessor[1][i];
                if (col >= 0 && col < dim_j) {
                  native::cpublas::axpy<Scalar>(dim_k,
                      cast_alpha * val,
                      dense_ptr + col * dense_stride0, dense_stride1,
                      newv_ptr + p * newv_stride0, 1);
                } else {
                  AT_ERROR("index out of bound. sspmm: ", col, " not between 1 and ", dim_j);
                }
              }
              // Fill up the indices with the right values
              if (i_start != i_end) {
                for (i64 i = 0; i < dim_k; i++) {
                  newi_accessor[0][p+i] = h;
                  newi_accessor[1][p+i] = i;
                }
                p += dim_k;
              }
            }
          }
      );

      // to avoid a clone
      get_sparse_impl(r)->set_indices_and_values_unsafe(newi, newv);
      get_sparse_impl(r)->set_nnz_and_narrow(p);

      return r;
        */
}

/// sparse, sparse, sparse, dense, real, real -> sparse
///
pub fn sspaddmm_out_only_sparse(
        self_:  &Tensor,
        mat1:   &Tensor,
        mat2:   &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            AT_ERROR("tensor.sspaddmm(...) can only be called on sparse tensors");
        */
}

/// sparse, dense -> sparse
pub fn smm(
        self_: &Tensor,
        mat2:  &Tensor) -> Tensor {
    
    todo!();
        /*
            auto result = empty({0}, self.options());
      sspaddmm_out(result, result, self, mat2, 0.0, 1.0);
      return result;
        */
}

/// sparse, sparse, dense, real, real -> sparse
///
pub fn sspaddmm(
        self_: &Tensor,
        mat1:  &Tensor,
        mat2:  &Tensor,
        beta:  &Scalar,
        alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            auto result = empty({0}, self.options());
      sspaddmm_out(result, self, mat1, mat2, beta, alpha);
      return result;
        */
}

/**
  | sparse.sum()
  |
  | This implementation calls coalesce() to do the
  | sum reduction on sparse dims. Ideally in the
  | future there should be unified reduction
  | function for ops like sum, max, and min.
  */
pub fn sparse_sum_a(input: &SparseTensor) -> Tensor {
    
    todo!();
        /*
            return input.coalesce().values().sum();
        */
}

pub fn sparse_sum_b(
    input: &SparseTensor,
    dtype: ScalarType) -> Tensor {

    todo!();
        /*
            // don't have to do a conversion to the correct dtype first
      // just need to setup the accumulator correctly
      return input.coalesce().values().sum(dtype);
        */
}

pub fn sparse_sum_c(
    input:       &SparseTensor,
    dims_to_sum: &[i32],
    dtype:       ScalarType) -> Tensor {
    
    todo!();
        /*
            return _sparse_sum(input.to(dtype), dims_to_sum);
        */
}

pub fn sparse_sum_d(
    input:       &SparseTensor,
    dims_to_sum: &[i32]) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(input._nnz() > 0, "_sparse_sum: sparse tensor input._nnz() == 0, please call torch.sparse.sum(input) instead.")

      const i64 input_dim = input.dim();
      auto dims_to_sum_b = dim_list_to_bitset(dims_to_sum, input_dim);
      auto dims_to_sum_v = dims_to_sum.vec();
      maybe_wrap_dims(dims_to_sum_v, input_dim);

      Tensor indices = input._indices();
      Tensor values = input._values();
      IntArrayRef sizes = input.sizes();
      const i64 sparse_dim = input.sparse_dim();
      // const i64 dense_dim = input.dense_dim();

      auto dims_to_keep_v = vector<i64>();
      auto dense_dims_to_sum_v = vector<i64>();
      for (i64 d = 0; d < input_dim; d++) {
        if (dims_to_sum_b[d]) {
          if (d >= sparse_dim) dense_dims_to_sum_v.emplace_back(d + 1 - sparse_dim);
        }
        else {
          dims_to_keep_v.emplace_back(d);
        }
      }
      const i64 sparse_dims_to_sum_size = dims_to_sum_v.size() - dense_dims_to_sum_v.size();
      const bool sum_all_sparse_dim = (sparse_dim == sparse_dims_to_sum_size);
      const bool sum_dense_dim = (dense_dims_to_sum_v.size() > 0);

      // new values
      Tensor new_values;
      if (sum_dense_dim) {
        new_values = values.sum(dense_dims_to_sum_v);
      }
      else {
        new_values = values.clone(MemoryFormat::Contiguous);
      }

      if (sum_all_sparse_dim) {
        // return a dense tensor if sum over all sparse dims
        new_values = new_values.sum(0);
        return new_values;
      }
      else { // !sum_all_sparse_dim
        // new indices
        Tensor new_indices;
        if (sparse_dims_to_sum_size == 0) {
          new_indices = indices.clone(MemoryFormat::Contiguous);
        }
        else {
          new_indices = empty({sparse_dim - sparse_dims_to_sum_size, input._nnz()}, indices.options());
          for (auto i: irange(dims_to_keep_v.size())) {
            i64 d = dims_to_keep_v[i];
            if (d < sparse_dim) new_indices[i].copy_(indices[d]);
            else break;
          }
        }

        // new size
        i64 new_sparse_dim = new_indices.size(0);
        i64 new_dense_dim = new_values.dim() - 1; // exclude nnz dim
        vector<i64> new_sizes;
        new_sizes.reserve(dims_to_keep_v.size());
        for (auto d : dims_to_keep_v) new_sizes.emplace_back(sizes[d]);
        if (sum_all_sparse_dim) new_sizes.emplace(new_sizes.begin(), 1);

        // use coalesce() to do sum reduction
        SparseTensor new_sparse = _sparse_coo_tensor_with_dims_and_tensors(new_sparse_dim, new_dense_dim, new_sizes, new_indices, new_values, input.options());
        new_sparse = new_sparse.coalesce();
        return new_sparse;
      }
        */
}

/**
  | NOTE [ sparse.sum() backward ]
  |
  | When sum over sparse_dim, backward scatters
  | gradients from grad tensor to input tensor.
  |
  | Grad and input need to align indices over
  | sparse_dim that are not summed (given
  | input.spares_dim >=
  | grad.sparse_dim). Implementation here compares
  | each pair of indices between grad and
  | input. When a matching indices pair (input_i,
  | grad_i) is found, copy grad.values[grad_i] ->
  | input_grad.values[input_i]. E.g.,
  |
  |  input.sparse_dim = [5, 5]
  |  input.indices = [[0, 0, 1, 2, 2, 3, 4, 4],
  |                   [1, 4, 4, 0, 1, 3, 2, 4]]
  |  input.values =   [0, 1, 2, 3, 4, 5, 6, 7]
  |  ...
  |  sparse.sum(input, [0])
  |  backward(...)
  |  ...
  |  grad.indices = [[0, 1, 2, 3]]
  |  grad.values =   [1, 2, 0, 4]
  |
  | # after indices matching
  |         input         grad
  |        [[0, 1],   ->  [1]
  |         [0, 4],   ->  [ ]
  |         [1, 4],   ->  [ ]
  |         [2, 0],   ->  [0]
  |         [2, 1],   ->  [1]
  |         [3, 3],   ->  [3]
  |         [4, 2],   ->  [2]
  |         [4, 4]])  ->  [ ]
  |
  | input_grad.indices = [[0, 0, 1, 2, 2, 3, 4, 4],
  |                       [1, 4, 4, 0, 1, 3, 2, 4]]
  | input_grad.values =   [2, 0, 0, 1, 2, 4, 0, 0]
  |
  | Note that we allow input to be uncoalesced in
  | the forward, we have to coalesce input at the
  | backward, because grad-of-input take the same
  | indices as input, if input is not coalesced,
  | then coalescing grad-of-input may add up grad
  | values for a duplicate indices, and hence
  | generates a wrong grad-of-input.
  |
  | Other edge cases:
  |
  | - assign zero values to input gradients if
  | cannot find matched indices at grad
  |
  | - grad.values might have zeros
  |
  */
pub fn sparse_sum_backward_cpu(
    grad:        &Tensor,
    input:       &SparseTensor,
    dims_to_sum: &[i32]) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!grad_.is_cuda(), "_sparse_sum_backward_cpu: expected 'grad_' to be CPU tensor, but got CUDA tensor");
      TORCH_CHECK(!input_.is_cuda(), "_sparse_sum_backward_cpu: expected 'input_' to be CPU tensor, but got CUDA tensor");

      auto input = input_.coalesce();
      const i64 input_dim = input.dim();
      auto dims_to_sum_b = dim_list_to_bitset(dims_to_sum, input_dim);
      auto dims_to_sum_v = dims_to_sum.vec();
      maybe_wrap_dims(dims_to_sum_v, input_dim);

      Tensor input_indices = input._indices();
      Tensor input_values = input._values();
      IntArrayRef input_sizes = input.sizes();
      const i64 input_sparse_dim = input.sparse_dim();
      const i64 input_dense_dim = input.dense_dim();
      const i64 input_nnz = input._nnz();

      i64 sparse_dims_to_sum_size = 0;
      auto sparse_dims_to_keep_v = vector<i64>();
      auto dense_dims_to_sum_v = vector<i64>();
      for (auto d: irange(input_dim)) {
        if (dims_to_sum_b[d]) {
          if (d < input_sparse_dim) sparse_dims_to_sum_size ++;
          else dense_dims_to_sum_v.emplace_back(d + 1 - input_sparse_dim);
        }
        else {
          if (d < input_sparse_dim) sparse_dims_to_keep_v.emplace_back(d);
        }
      }

      const bool sum_all_sparse_dim = (input_sparse_dim == sparse_dims_to_sum_size);
      const bool sum_dense_dim = (dense_dims_to_sum_v.size() > 0);
      const bool sum_sparse_dim = (sparse_dims_to_sum_size > 0);

      if (sum_all_sparse_dim) {
        TORCH_CHECK(!grad_.is_sparse(), "_sparse_sum_backward_cpu: expected grad_ Tensor to be dense since all sparse dims are summed");
        auto grad_input_values = grad_;
        auto expand_size = input_values.sizes().vec();
        if (sum_dense_dim) {
          auto dense_expand_size = vector<i64>(expand_size);
          dense_expand_size.erase(dense_expand_size.begin());
          AT_ASSERT(dense_expand_size.size() == static_cast<usize>(input_values.dim() - 1));
          for (auto d : dense_dims_to_sum_v) grad_input_values = grad_input_values.unsqueeze(d - 1);  // -1 since grad has no nnz dim
          grad_input_values = grad_input_values.expand(dense_expand_size);
        }
        grad_input_values = grad_input_values.expand(expand_size).clone(MemoryFormat::Contiguous);
        return _sparse_coo_tensor_with_dims_and_tensors(input_sparse_dim, input_dense_dim, input_sizes, input_indices.clone(MemoryFormat::Contiguous), grad_input_values, input.options().dtype(grad_.dtype())); // convert to grad dtype
      }
      else {
        TORCH_CHECK(grad_.is_sparse(), "_sparse_sum_backward_cpu: expected grad_ Tensor to be sparse, but got dense");
        auto grad = grad_.coalesce();
        Tensor grad_indices = grad._indices();
        Tensor grad_values = grad._values();
        const i64 grad_sparse_dim = grad.sparse_dim();
        const i64 grad_nnz = grad._nnz();

        Tensor grad_values_expand = grad_values;
        if (sum_dense_dim) {
          auto expand_size = input_values.sizes().vec();
          if (sum_sparse_dim) expand_size[0] = grad_values.size(0);
          for (auto d : dense_dims_to_sum_v) grad_values_expand = grad_values_expand.unsqueeze(d);
          grad_values_expand = grad_values_expand.expand(expand_size).clone(MemoryFormat::Contiguous);
        }

        Tensor grad_input_values;
        if (sum_sparse_dim) {
          // see NOTE [ sparse.sum() backward ]
          grad_input_values = zeros_like(input_values, grad_values.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);

          // get flatten indices for grad and input
          auto grad_sparse_dim_to_keep_v = vector<i64>(grad_sparse_dim);
          iota(grad_sparse_dim_to_keep_v.begin(), grad_sparse_dim_to_keep_v.end(), 0);

          auto grad_indices_1D = flatten_indices_by_dims(grad_indices, grad.sizes(), grad_sparse_dim_to_keep_v); // flatten indices on all sparse_dim of grad, output indices is coalesced and sorted
          auto grad_indices_1D_accessor = grad_indices_1D.accessor<i64, 1>();
          auto input_indices_1D = flatten_indices_by_dims(input_indices, input_sizes, sparse_dims_to_keep_v);
          auto input_indices_1D_accessor = input_indices_1D.accessor<i64, 1>();

          // binary search to find matching indices

          parallel_for(0, input_nnz, 0, [&](i64 start, i64 end) {
            for (auto i: irange(start, end)) {
              i64 input_idx = input_indices_1D_accessor[i];
              i64 l = 0, r = grad_nnz - 1;
              while (l <= r) {
                i64 m = l + (r - l) / 2;
                if (grad_indices_1D_accessor[m] == input_idx) {
                  grad_input_values[i].copy_(grad_values_expand[m]);
                  break;
                }
                if (grad_indices_1D_accessor[m] < input_idx) {
                  l = m + 1;
                }
                else {
                  r = m - 1;
                }
              }
            }
          });
        }
        else {
          grad_input_values = grad_values_expand;
        }
        return _sparse_coo_tensor_with_dims_and_tensors(input_sparse_dim, input_dense_dim, input_sizes, input_indices.clone(MemoryFormat::Contiguous), grad_input_values, grad.options());
      }
        */
}

pub fn isnan_sparse(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(self.is_sparse());
      SparseTensor out =  sparse_coo_tensor({0}, self.options().dtype(kBool));
      out.resize_as_(self);
      auto indices = out._indices();
      indices.resize_as_(self._indices());
      indices.copy_(self._indices());
      Tensor out_values = out._values();
      out_values.resize_as_(self._values());
      Tensor nan_values = isnan(self._values());
      out_values.copy_(nan_values);
      return out;
        */
}

pub fn any_sparse(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(self.is_sparse());

      return any(self._values());
        */
}

pub fn bmm_sparse_cpu(
        self_: &SparseTensor,
        mat2:  &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({}, mat2.options());
      return bmm_out_sparse_cpu(self, mat2, result);
        */
}

/**
  | Search a sorted strided array for the rightmost
  | instance of a value.
  |
  | Array must be sorted from lowest to highest.
  |
  | Returns the index of the found element.
  |
  | Returns by reference `found`, true if search
  | value was found, false otherwise
  |
  */
pub fn binary_search_strided_rightmost<Scalar>(
        search_val:           Scalar,
        sorted_arr_accessor:  &mut TensorAccessor<Scalar,1>,
        sorted_arr_begin_idx: i64,
        length:               i64,
        found:                *mut bool) -> Scalar {

    todo!();
        /*
            if (length == 0) {
        *found = false;
        return -1;
      }

      i64 left_ind = 0;
      i64 right_ind = length - 1;
      i64 mid_ind; // NOLINT(cppcoreguidelines-init-variables)
      bool done_searching = false;

      while (!done_searching) {
        mid_ind = (left_ind+right_ind) >> 1;
        Scalar mid_val = sorted_arr_accessor[sorted_arr_begin_idx + mid_ind];

        if (mid_val > search_val) {
          right_ind = mid_ind-1;
        } else if((mid_val == search_val) && (
          (mid_ind == length - 1) || (sorted_arr_accessor[sorted_arr_begin_idx + mid_ind + 1] != search_val)
        )) {
          done_searching = true;
          *found = true;
        } else {
          left_ind = mid_ind+1;
        }

        if (left_ind > right_ind) {
          done_searching = true;
          *found = false;
          mid_ind = -1;
        }
      }

      return mid_ind;
        */
}

pub fn bmm_out_sparse_cpu(
    self_:  &SparseTensor,
    mat2:   &Tensor,
    result: &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            TORCH_CHECK(!mat2.is_sparse(), "bmm_sparse: Tensor 'mat2' must be dense");

      TORCH_CHECK(self.dense_dim() == 0, "bmm_sparse: Tensor 'self' must have 0 dense dims, but has ", self.dense_dim());
      TORCH_CHECK(self.sparse_dim() == 3, "bmm_sparse: Tensor 'self' must have 3 sparse dims, but has ", self.sparse_dim());
      TORCH_CHECK(mat2.dim() == 3, "bmm_sparse: Tensor 'mat2' must have 3 dims, but has ", mat2.dim());

      TORCH_CHECK(self.size(0) == mat2.size(0), "bmm_sparse: 'self.size(0)' and 'mat2.size(0)' must match");
      TORCH_CHECK(self.size(2) == mat2.size(1), "bmm_sparse: 'self.size(2)' and 'mat2.size(1)' must match");

      result.resize_({self.size(0), self.size(1), mat2.size(2)});

      if (self._nnz() == 0) {
        result.zero_();
        return result;
      }

      // First need to coalesce to get all of the first dimension indices
      // in order since we'll be sending each matrix into the MM operation
      SparseTensor self_coalesced = self.coalesce();

      i64 nnz =        self_coalesced._nnz();
      Tensor indices = self_coalesced._indices();
      Tensor values =      self_coalesced._values();

      Tensor indices_dim0 = indices[0];
      auto indices_dim0_accessor = indices_dim0.accessor<i64, 1>();
      Tensor indices_dim1_dim2 = indices.slice(0, 1, 3);

      i64 dim_i = self_coalesced.size(1);
      i64 dim_j = self_coalesced.size(2);
      i64 dim_k = mat2.size(2);

      Scalar beta = 0;
      Tensor t_dummy;
      Scalar alpha = 1;

      i64 mat_el_begin_idx = 0;

      i64 num_matrices = self_coalesced.size(0);

      // Iterate through each set of 2D matrices within the 3D
      // tensor inputs, performing a matrix multiply with each one.
      i64 start_mat_num = indices_dim0_accessor[0];
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
        values.scalar_type(), "bmm_sparse_dense", [&] {
          for (i64 cur_mat_num = 0;
            (cur_mat_num < num_matrices);
            cur_mat_num++
          ) {
            // If there are sparse matrices at the beginning or end that
            // have all zero elements, we need to zero out the result matrix.
            if ((cur_mat_num < start_mat_num) || (mat_el_begin_idx >= nnz)) {
              result[cur_mat_num].zero_();
              continue;
            }

            // Search for the range of sparse tensor elements that
            // correspond to the current matrix number. We already know
            // where the current matrix begins, so we just need to find
            // the end. The search excludes everything to the left of
            // the starting point, for best performance
            bool mat_end_found;
            i64 mat_el_end_idx = binary_search_strided_rightmost(
              cur_mat_num,
              indices_dim0_accessor,
              mat_el_begin_idx,
              nnz-mat_el_begin_idx,
              &mat_end_found
            ) + mat_el_begin_idx;

            if (mat_end_found) {
              mat_el_end_idx++;

              // Create tensors to view just the current set of matrices
              const Tensor dense_matrix = mat2[cur_mat_num];
              Tensor result_matrix = result[cur_mat_num];
              Tensor sparse_indices = indices_dim1_dim2.slice(1, mat_el_begin_idx, mat_el_end_idx);
              Tensor sparse_values = values.slice(0, mat_el_begin_idx, mat_el_end_idx);
              i64 sparse_nnz = mat_el_end_idx - mat_el_begin_idx;

              s_addmm_out_sparse_dense_worker<Scalar>(
                sparse_nnz,
                dim_i, dim_j, dim_k,
                result_matrix,
                beta, t_dummy, alpha,
                sparse_indices, sparse_values,
                dense_matrix
              );
              mat_el_begin_idx = mat_el_end_idx;

            // If no elements for this sparse matrix are found, then
            // it's a zero matrix and we need to zero out the result
            } else {
              result[cur_mat_num].zero_();
            }
          }
        }
      );
      return result;
        */
}

// Tensor conj_physical_sparse(const Tensor& input) {
//   if (!input.is_complex()) {
//     return input;
//   }
//   Tensor result = native::empty_like(input);
//   return conj_physical_out_sparse(input, result);
// }
pub fn conj_physical_out_sparse(
        input:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(input.is_sparse());
      if (!is_same_tensor(result, input)) {
        copy_sparse_to_sparse_(result, input);
      }
      if (!input.is_complex()) {
        return result;
      }
      Tensor result_values = result._values();
      conj_physical_out(result_values, input._values());
      return result;
        */
}
