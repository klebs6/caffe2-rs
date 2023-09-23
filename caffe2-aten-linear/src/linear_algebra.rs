crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/LinearAlgebra.h]

pub type AddrFn = fn(
        _0:    &mut TensorIterator,
        beta:  &Scalar,
        alpha: &Scalar
) -> c_void;


declare_dispatch!{addr_fn, addr_stub}

pub type LinalgVectorNormFn = fn(_0: &mut TensorIterator, _1: Scalar) -> ();

declare_dispatch!{linalg_vector_norm_fn, linalg_vector_norm_stub}

pub type UnpackPivotsFn = fn(iter: &mut TensorIterator, dim_size: i64) -> ();

declare_dispatch!{unpack_pivots_fn, unpack_pivots_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/LinearAlgebra.cpp]

lazy_static!{
    /*
    TORCH_META_FUNC(addmm)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
      TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");
      TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");

      auto names = namedinference::propagate_names_for_addmm(mat1, mat2, self);
      set_output(0, {mat1.sizes()[0], mat2.sizes()[1]}, {}, self.options(), names);
      const auto& result = maybe_get_output(0);
      //this check can fire for inplace op only, for all other versions result is guaranteed to be correct size
      TORCH_CHECK(((result.dim() == 2) && (result.sizes()[0] == mat1.sizes()[0]) && (result.sizes()[1] == mat2.sizes()[1])),
      "The input tensor must be a matrix with size ", mat1.sizes()[0], "x", mat2.sizes()[1], ", but got a ", result.dim(),
      "-D tensor with size ", result.sizes()[0], "x", result.sizes()[1]);
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(mm)(const Tensor & self, const Tensor & mat2) {
      TORCH_CHECK(self.dim() == 2, "self must be a matrix");
      TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");

      auto names = namedinference::compute_matmul_outnames(self, mat2);
      set_output(0, {self.sizes()[0], mat2.sizes()[1]}, {}, self.options(), names);
      const auto& result = maybe_get_output(0);
      //this check can fire for inplace op only, for all other versions result is guaranteed to be correct size
      TORCH_CHECK(((result.dim() == 2) && (result.sizes()[0] == self.sizes()[0]) && (result.sizes()[1] == mat2.sizes()[1])),
      "The input tensor must be a matrix with size ", self.sizes()[0], "x", mat2.sizes()[1], ", but got a ", result.dim(),
      "-D tensor with size ", result.sizes()[0], "x", result.sizes()[1]);
    }
    */
}

define_dispatch!{addr_stub}

define_dispatch!{linalg_vector_norm_stub}

/**
  | Helper function for det methods.
  | For pivoted LU factorization A = P * L * U. Since we always have det(L) = 1,
  | det(P) = \pm 1, this method returns a 3-tuple:
  |   (det(P), diag(U), info),
  | where info helps us identify singular matrices.
  */
#[inline] pub fn lu_det_p_diag_u(self_: &Tensor) -> (Tensor,Tensor) {
    
    todo!();
        /*
            Tensor pivs, lu, infos;
      tie(lu, pivs, infos) = _lu_with_info(self, /*pivot=*/true, /*check_errors=*/false);
      TORCH_CHECK(infos.ge(0).all().item<u8>(), "Invalid argument passed to lu");
      auto n = self.size(-1);
      auto num_exchanges = (arange(1, n + 1, pivs.options()) != pivs)
        .sum(-1, /*keepdim=*/false, /*dtype=*/kLong).fmod_(2);
      auto u_diagonal = lu.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1);
      return tuple<Tensor, Tensor>(num_exchanges.mul_(-2).add_(1), u_diagonal);
        */
}

/// torch.det, alias for torch.linalg.det
pub fn det(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return linalg_det(self);
        */
}

pub fn linalg_det_out<'a>(
        self_: &Tensor,
        out:   &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            checkSameDevice("torch.linalg.det", out, self, "out");
      checkLinalgCompatibleDtype("torch.linalg.det", out, self, "out");
      squareCheckInputs(self);
      TORCH_CHECK((isFloatingType(self.scalar_type()) || isComplexType(self.scalar_type())),
                  "Expected a floating point or complex tensor as input");

      IntArrayRef out_sizes(self.sizes().data(), self.dim() - 2);
      native::resize_output(out, out_sizes);

      Tensor det_P, diag_U;
      tie(det_P, diag_U) = _lu_det_P_diag_U(self);
      // complete_det is 0 when U is singular (U(i, i) = 0 for some i in [1, self.size(-1)]).
      // The product accumulation takes care of this case, and hence no special case handling is required.
      prod_out(out, diag_U, -1);
      out.mul_(det_P);
      return out;
        */
}

pub fn linalg_det(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            auto out = empty({0}, self.options());
      native::linalg_det_out(self, out);
      return out;
        */
}

pub fn logdet(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            squareCheckInputs(self);
      TORCH_CHECK((isFloatingType(self.scalar_type()) || isComplexType(self.scalar_type())),
                  "Expected a floating point tensor as input");

      Tensor det_P, diag_U;
      tie(det_P, diag_U) = _lu_det_P_diag_U(self);
      Tensor det_sign = diag_U.sign().prod(-1).mul_(det_P);

      // If det_sign > 0, diag_U.abs_().log_().sum(-1) gives logdet (this means U is not singular).
      // If det_sign <= 0, then we get proper nan (when det < 0, i.e., det_sign) or -inf (when det = 0, i.e., U is singular).
      // U is singular when U(i, i) = 0 for some i in [1, self.size(-1)].
      Tensor logdet_vals = diag_U.abs_().log_().sum(-1);
      if (self.dim() > 2) {
        auto indices = toListOfOptionalTensors((det_sign < 0).nonzero_numpy());
        logdet_vals.index_put_(move(indices), full({}, NAN, self.options()));
      } else if (det_sign.item<double>() < 0) {
        logdet_vals.fill_(NAN);
      }
      return logdet_vals;
        */
}

pub fn linalg_slogdet(self_: &Tensor) -> (Tensor,Tensor) {
    
    todo!();
        /*
            squareCheckInputs(self);
      ScalarType t = self.scalar_type();
      TORCH_CHECK(t == ScalarType::Double || t == ScalarType::Float || t == ScalarType::ComplexFloat || t == ScalarType::ComplexDouble,
                  "linalg_slogdet: expected a tensor of float, double, cfloat or cdouble types but got ", t);

      Tensor det_P, diag_U;
      tie(det_P, diag_U) = _lu_det_P_diag_U(self);
      auto det_sign = diag_U.sgn().prod(-1).mul_(det_P);
      // abslogdet_val is -inf if U is singular, in which case diag_U.abs_().log_().sum(-1) will return -inf.
      // U is singular when U(i, i) = 0 for some i in [1, self.size(-1)].
      // Since abslogdet_val cannot take nan, no special case handling is required.
      // in-place abs is not supported for complex tensors
      auto abslogdet_val = isComplexType(t) ? diag_U.abs().log_().sum(-1) : diag_U.abs_().log_().sum(-1);
      return make_tuple(det_sign, abslogdet_val);
        */
}

/**
  | TODO: implement _out variant avoiding
  | copy and using already allocated storage
  | directly
  |
  */
pub fn linalg_slogdet_out<'a>(
        input:     &Tensor,
        sign:      &mut Tensor,
        logabsdet: &mut Tensor) -> (&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
            checkSameDevice("linalg_slogdet", sign, input, "sign");
      checkSameDevice("linalg_slogdet", logabsdet, input, "logabsdet");
      checkLinalgCompatibleDtype("linalg_slogdet", sign, input, "sign");
      ScalarType real_dtype = toValueType(input.scalar_type());
      // logabsdet is always real-valued here
      checkLinalgCompatibleDtype("linalg_slogdet", logabsdet.scalar_type(), real_dtype, "logabsdet");

      Tensor sign_tmp, logabsdet_tmp;
      tie(sign_tmp, logabsdet_tmp) = linalg_slogdet(input);

      native::resize_output(sign, sign_tmp.sizes());
      sign.copy_(sign_tmp);
      native::resize_output(logabsdet, logabsdet_tmp.sizes());
      logabsdet.copy_(logabsdet_tmp);

      return tuple<Tensor&, Tensor&>(sign, logabsdet);
        */
}

pub fn slogdet(self_: &Tensor) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return linalg_slogdet(self);
        */
}

pub fn linalg_pinv_a(
        input:     &Tensor,
        rcond:     &Tensor,
        hermitian: bool) -> Tensor {
    
    todo!();
        /*
            NoTF32Guard disable_tf32;
      ScalarType t = input.scalar_type();
      TORCH_CHECK((t == ScalarType::Double || t == ScalarType::Float || t == ScalarType::ComplexFloat || t == ScalarType::ComplexDouble)
                  && input.dim() >= 2,
                  "linalg_pinv(", t, "{", input.sizes(), "}): expected a tensor with 2 or more dimensions "
                  "of float, double, cfloat or cdouble types");
      TORCH_CHECK(rcond.device() == input.device(),
                  "Expected rcond and input to be on the same device, but found rcond on ",
                  rcond.device(), " and input on ", input.device(), " instead.");
      TORCH_CHECK(!isComplexType(rcond.scalar_type()),
                  "linalg_pinv: rcond tensor of complex type is not supported.");

      if (input.numel() == 0) {
        // The implementation below uses operations that do not work for zero numel tensors
        // therefore we need this early return for 'input.numel() == 0' case
        Tensor U, S, V;
        // TODO: replace input.svd with linalg_svd when torch/xla can work with linalg_svd
        tie(U, S, V) = input.svd();
        return matmul(V * S.reciprocal().unsqueeze(-2), U.conj().transpose(-2, -1));
      }

      // If not Hermitian use singular value decomposition, else use eigenvalue decomposition
      if (!hermitian) {
        Tensor U, S, V;
        // TODO: replace input.svd with linalg_svd
        // using linalg_svd breaks pytorch/xla, see https://github.com/pytorch/xla/issues/2755
        tie(U, S, V) = input.svd();
        Tensor max_val = narrow(S, /*dim=*/-1, /*start=*/0, /*length=*/1);  // singular values are sorted in descending order
        Tensor S_pseudoinv = where(S > (rcond.unsqueeze(-1) * max_val), S.reciprocal(), zeros({}, S.options())).to(input.dtype());
        // computes V @ diag(S_pseudoinv) @ U.conj().T
        return matmul(V * S_pseudoinv.unsqueeze(-2), U.conj().transpose(-2, -1));
      } else {
        Tensor S, U;
        tie(S, U) = linalg_eigh(input);
        // For Hermitian matrices, singular values equal to abs(eigenvalues)
        Tensor S_abs = S.abs();
        // eigenvalues are sorted in ascending order starting with negative values, we need a maximum value of abs(eigenvalues)
        Tensor max_val = S_abs.amax(/*dim=*/-1, /*keepdim=*/true);
        Tensor S_pseudoinv = where(S_abs > (rcond.unsqueeze(-1) * max_val), S.reciprocal(), zeros({}, S.options())).to(input.dtype());
        // computes U @ diag(S_pseudoinv) @ U.conj().T
        return matmul(U * S_pseudoinv.unsqueeze(-2), U.conj().transpose(-2, -1));
      }
        */
}

pub fn linalg_pinv_b(
        input:     &Tensor,
        rcond:     f64,
        hermitian: bool) -> Tensor {
    
    todo!();
        /*
            Tensor rcond_tensor = full({}, rcond, input.options().dtype(ScalarType::Double));
      return linalg_pinv(input, rcond_tensor, hermitian);
        */
}

/**
  | TODO: implement _out variant avoiding
  | copy and using already allocated storage
  | directly
  |
  */
pub fn linalg_pinv_out_a<'a>(
        input:     &Tensor,
        rcond:     &Tensor,
        hermitian: bool,
        result:    &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            checkSameDevice("linalg_pinv", result, input);
      checkLinalgCompatibleDtype("linalg_pinv", result, input);

      Tensor result_tmp = linalg_pinv(input, rcond, hermitian);
      native::resize_output(result, result_tmp.sizes());
      result.copy_(result_tmp);
      return result;
        */
}

pub fn linalg_pinv_out_b<'a>(
        input:     &Tensor,
        rcond:     f64,
        hermitian: bool,
        result:    &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            Tensor rcond_tensor = full({}, rcond, input.options().dtype(ScalarType::Double));
      return linalg_pinv_out(result, input, rcond_tensor, hermitian);
        */
}

pub fn pinverse(
        self_: &Tensor,
        rcond: f64) -> Tensor {
    
    todo!();
        /*
            return linalg_pinv(self, rcond, /*hermitian=*/false);
        */
}

/**
  | matrix_power implementation
  | 
  | -----------
  | @brief
  | 
  | Raises the input matrix to the given
  | power n
  | 
  | If the exponent n is negative, the inverse
  | of the input matrix will be raised to
  | power abs(n).
  | 
  | -----------
  | @param self
  | 
  | (batched) square matrix to raise to
  | power n
  | ----------
  | @param n
  | 
  | exponent to raise matrix (or matrices
  | in batch) to
  | ----------
  | @param _out
  | 
  | optional tensor to write the output
  | to
  | 
  | 
  | -----------
  | @return
  | 
  | Tensor input matrix raised to power
  | n
  |
  */
pub fn linalg_matrix_power_impl(
    self_: &Tensor,
    n:     i64,
    out:   Option<Tensor>) -> Tensor {
    
    todo!();
        /*
            auto out = _out.value_or(Tensor());

      squareCheckInputs(self);
      if (_out.has_value()) {
        checkSameDevice("matrix_power", out, self);
        checkLinalgCompatibleDtype("matrix_power", out, self);
        native::resize_output(out, self.sizes());
      }

      // For n=0 we return the identity matrix of the same shape as input.
      if (n == 0) {
        if (!_out.has_value()) {
          // Clone input to include result in the autograd graph
          out = self.clone(MemoryFormat::Contiguous);
        }
        return out.copy_(eye(self.size(-2), self.options()));
      }
      if (n == 1) {
        return _out.has_value() ? out.copy_(self)
                                : self.clone(MemoryFormat::Contiguous);
      }
      if (n == -1) {
        return _out.has_value() ? linalg_inv_out(out, self)
                                : linalg_inv(self);
      }

      // For negative n we inverte the input matrix before raising to power abs(n)
      auto a = n < 0 ? linalg_inv(self) : self;
      n = abs(n);

      // Fast paths for small powers
      if (n == 2) {
        return _out.has_value() ? matmul_out(out, a, a) : matmul(a, a);
      }
      if (n == 3) {
        return _out.has_value() ? matmul_out(out, matmul(a, a), a)
                                : matmul(matmul(a, a), a);
      }

      // This is a binary decomposition of n.
      // Moving from the least significant bit to the most significant bit
      // This is done to reduce the number of matrix multiplications
      // by raising the input matrix in powers of 2
      // The total number of matrix multiplications are
      // number of bits + number of bits that equal 1 ~ O(log n)
      // instead of O(n)
      Tensor z, result;
      while (n > 0) {
        const auto bit = n % 2;
        n = n / 2;
        z = z.defined() ? matmul(z, z) : a;
        if (bit == 1) {
          if (_out.has_value() && n <= 0) {
            // Last multiplication can use the out version
            return result.defined() ? matmul_out(out, result, z) : out.copy_(z);
          }
          result = result.defined() ? matmul(result, z) : z;
        }
      }

      return result;
        */
}

pub fn linalg_matrix_power_out<'a>(
        self_:  &Tensor,
        n:      i64,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            linalg_matrix_power_impl(self, n, result);
      return result;
        */
}


pub fn linalg_matrix_power(
        self_: &Tensor,
        n:     i64) -> Tensor {
    
    todo!();
        /*
            return linalg_matrix_power_impl(self, n, nullopt);
        */
}

pub fn matrix_power_out<'a>(
        self_:  &Tensor,
        n:      i64,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::linalg_matrix_power_out(self, n, result);
        */
}

pub fn matrix_power(
        self_: &Tensor,
        n:     i64) -> Tensor {
    
    todo!();
        /*
            return native::linalg_matrix_power(self, n);
        */
}

/**
  | Computes the rank of 'input' and saves the
  | result in-place in 'result' 'hermitian'
  | controls whether SVD or eigendecomposition is
  | used for computing the singular values 'atol'
  | and 'rtol' are the absolute and relative
  | tolerances, respectively.
  |
  | TODO: this function can be made public, see:
  | https://github.com/pytorch/pytorch/issues/54151
  */
pub fn linalg_matrix_rank_out_helper<'a>(
        input:     &Tensor,
        atol:      &Tensor,
        rtol:      &Tensor,
        hermitian: bool,
        result:    &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            checkSameDevice("torch.linalg.matrix_rank", result, input);
      checkSameDevice("torch.linalg.matrix_rank", atol, input, "atol");
      checkSameDevice("torch.linalg.matrix_rank", rtol, input, "rtol");
      ScalarType output_type = ScalarType::Long;
      checkLinalgCompatibleDtype("torch.linalg.matrix_rank", result.scalar_type(), output_type);

      // Matrices or batch of matrices are allowed
      TORCH_CHECK(input.dim() >= 2, "torch.linalg.matrix_rank: Expected as input a matrix or a batch of matrices, but got a tensor of size: ", input.sizes());

      TORCH_CHECK(!isComplexType(atol.scalar_type()),
                  "torch.linalg.matrix_rank: atol tensor of complex type is not supported.");
      TORCH_CHECK(!isComplexType(rtol.scalar_type()),
                  "torch.linalg.matrix_rank: rtol tensor of complex type is not supported.");

      // matrix_rank assigns a scalar value for each matrix in the batch so
      // result's shape is equal to input.shape[0:input.ndim-2]
      // for single matrix result_shape = {}
      auto result_shape = IntArrayRef(input.sizes().cbegin(), input.sizes().cend() - 2);
      native::resize_output(result, result_shape);

      // NumPy doesn't take into account possible input with no elements and it errors on max not defined for this case
      // Let's output 0 for this case, since that kind of matrices have zero number of non-zero rows, hence rank is 0.
      if (input.numel() == 0) {
        result.fill_(0);
        return result;
      }

      // We compute matrix rank as the number of singular or absolute eigen values
      // that are above max(atol, rtol * max(S)) threshold
      Tensor S, max_S;
      if (!hermitian) {
        S = linalg_svdvals(input);
        // singular values are sorted in descending order
        max_S = narrow(S, /*dim=*/-1, /*start=*/0, /*length=*/1);
      } else {
        S = linalg_eigvalsh(input);
        S = S.abs();
        // eigenvalues are sorted in ascending order starting with negative values, we need a maximum value of abs(eigenvalues)
        max_S = S.amax(/*dim=*/-1, /*keepdim=*/true);
      }

      Tensor tol = max(atol.unsqueeze(-1), rtol * max_S);

      result = sum_out(result, S > tol, /*dim=*/-1);
      return result;
        */
}

pub fn linalg_matrix_rank_out_a<'a>(
        input:     &Tensor,
        tol:       &Tensor,
        hermitian: bool,
        result:    &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            // For NumPy compatibility tol is not scaled with max(singular_value) if the value for tol is provided
      // It is assumed that the provided value is the absolute tolerance
      Tensor rtol = zeros({}, tol.options());
      result = linalg_matrix_rank_out_helper(input, tol, rtol, hermitian, result);
      return result;
        */
}

pub fn linalg_matrix_rank_out_b<'a>(
        input:     &Tensor,
        tol:       Option<f64>,
        hermitian: bool,
        result:    &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            double tol_value;
      Tensor atol, rtol;
      if (tol.has_value()) {
        tol_value = tol.value();
        // For NumPy compatibility tol is not scaled with max(singular_value) if the value for tol is provided
        // It is assumed that the provided value is the absolute tolerance
        atol = full({}, tol_value, input.options().dtype(ScalarType::Double));
        rtol = zeros({}, input.options().dtype(ScalarType::Double));
      } else {
        ScalarType real_dtype = toValueType(input.scalar_type());
        // This is NumPy compatible default value
        tol_value = _get_epsilon(real_dtype) * max(input.size(-1), input.size(-2));
        // It is assumed that the default tolerance is the relative tolerance
        atol = zeros({}, input.options().dtype(ScalarType::Double));
        rtol = full({}, tol_value, input.options().dtype(ScalarType::Double));
      }

      result = linalg_matrix_rank_out_helper(input, atol, rtol, hermitian, result);
      return result;
        */
}

pub fn linalg_matrix_rank_a(
        input:     &Tensor,
        tol:       &Tensor,
        hermitian: bool) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, input.options().dtype(ScalarType::Long));
      result = linalg_matrix_rank_outf(input, tol, hermitian, result);
      return result;
        */
}

pub fn linalg_matrix_rank_b(
        input:     &Tensor,
        tol:       Option<f64>,
        hermitian: bool) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, input.options().dtype(ScalarType::Long));
      result = linalg_matrix_rank_outf(input, tol, hermitian, result);
      return result;
        */
}

pub fn matrix_rank_a(
    self_:     &Tensor,
    tol:       f64,
    symmetric: bool) -> Tensor {

    todo!();
        /*
            TORCH_WARN_ONCE(
        "torch.matrix_rank is deprecated in favor of torch.linalg.matrix_rank",
        "and will be removed in a future PyTorch release. The parameter 'symmetric' was ",
        "renamed in torch.linalg.matrix_rank to 'hermitian'."
      );
      return linalg_matrix_rank(self, optional<double>(tol), symmetric);
        */
}

pub fn matrix_rank_b(
    self_:     &Tensor,
    symmetric: bool) -> Tensor {

    todo!();
        /*
            TORCH_WARN_ONCE(
        "torch.matrix_rank is deprecated in favor of torch.linalg.matrix_rank",
        "and will be removed in a future PyTorch release. The parameter 'symmetric' was ",
        "renamed in torch.linalg.matrix_rank to 'hermitian'."
      );
      return linalg_matrix_rank(self, nullopt, symmetric);
        */
}

/**
  | multi_dot helper functions
  | 
  | -----------
  | @brief
  | 
  | Computes the optimal matrix chain multiplication
  | order
  | 
  | Follows the dynamic programming algorithm
  | from Cormen et al, "Introduction to
  | Algorithms, Third Edition", Chapter
  | 15.2, p. 370-378. Note that the book
  | uses 1-based indexing.
  | 
  | The cost of multiplying two matrices
  | with sizes p x q and q x r is defined here
  | as p * q * r. The optimal multiplication
  | order is the one that minimizes the total
  | cost.
  | 
  | -----------
  | @param tensors
  | 
  | list of 2D tensors
  | 
  | 
  | -----------
  | @return
  | 
  | a 2D vector s used by #matrix_chain_multiplication
  | to construct the optimal matrix multiplication
  | order. The optimal multiplication
  | order for multiplying tensors i...j
  | is to multiply tensors i...s[i, j] and
  | tensors (s[i, j] + 1)...j first and then
  | the result of that.
  |
  */
pub fn matrix_chain_order(tensors: &[Tensor]) -> Vec<Vec<i64>> {
    
    todo!();
        /*
            const usize n = tensors.size();

      // Tensor i has dimensions p[i] x p[i + 1]
      vector<i64> p(n + 1);
      for (const auto i : irange(n)) {
        p[i] = tensors[i].size(0);
      }
      p[n] = tensors[n - 1].size(1);

      // m[i, j] = k where k is the minimum cost for multiplying tensors i...j
      vector<vector<i64>> m(n, vector<i64>(n, 0));

      // s[i, j] = k where k is the index at which to split the list such that
      // optimally multiplying matrices i...k and k...j first and then the resulting
      // matrices is the optimal order for multiplying matrices i...j.
      vector<vector<i64>> s(n, vector<i64>(n));

      // Compute the optimal multiplication order
      for (const auto l : irange(1, n)) {
        for (const auto i : irange(n - l)) {
          const auto j = i + l;
          m[i][j] = i64::max;
          for (const auto k : irange(i, j)) {
            const auto q = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1];
            if (q < m[i][j]) {
              m[i][j] = q;
              s[i][j] = k;
            }
          }
        }
      }

      return s;
        */
}

/**
  | -----------
  | @brief
  | 
  | Recursively multiplies the tensors
  | i...j using the given order
  | 
  | -----------
  | @param tensors
  | 
  | matrices to multiply togther
  | ----------
  | @param order
  | 
  | optimal chain multiplication order
  | from #matrix_chain_order
  | ----------
  | @param i
  | 
  | index of first tensor to be multiplied
  | ----------
  | @param j
  | 
  | index of last tensor to be multiplied
  | 
  | 
  | -----------
  | @return
  | 
  | Tensor result of multiplying tensors[i...j]
  | together.
  |
  */
pub fn matrix_chain_multiplication(
        tensors: &[Tensor],
        order:   &Vec<Vec<i64>>,
        i:       i64,
        j:       i64) -> Tensor {
    
    todo!();
        /*
            if (i == j) {
        return tensors[i];
      }
      return mm(
          matrix_chain_multiplication(tensors, order, i, order[i][j]),
          matrix_chain_multiplication(tensors, order, order[i][j] + 1, j));
        */
}

/// Implements torch.linalg.multi_dot
pub fn multi_dot_impl(
        tensors: &[Tensor],
        out:     Option<Tensor>) -> Tensor {
    
    todo!();
        /*
            const usize n = _tensors.size();
      TORCH_CHECK(n >= 2, "multi_dot(): expected at least 2 tensors but got ", n);

      vector<i64> out_shape;
      vector<Tensor> tensors(n);

      // If the first tensor is 1D of size n view it as a row vector (1, n)
      if (_tensors[0].dim() == 1) {
        tensors[0] = _tensors[0].unsqueeze(0);
      } else if (_tensors[0].dim() == 2) {
        tensors[0] = _tensors[0];
        out_shape.emplace_back(tensors[0].size(0));
      } else {
        TORCH_CHECK(
            false,
            "multi_dot(): the first tensor must be 1D or 2D but got ",
            _tensors[0].dim(),
            "D");
      }

      // If the last tensor is 1D of size n view it as a column vector (n, 1)
      if (_tensors[n - 1].dim() == 1) {
        tensors[n - 1] = _tensors[n - 1].unsqueeze(-1);
      } else if (_tensors[n - 1].dim() == 2) {
        tensors[n - 1] = _tensors[n - 1];
        out_shape.emplace_back(tensors[n - 1].size(1));
      } else {
        TORCH_CHECK(
            false,
            "multi_dot(): the last tensor must be 1D or 2D but got ",
            _tensors[n - 1].dim(),
            "D");
      }

      // Ensure middle tensors are 2D
      for (const auto i : irange(1, n - 1)) {
        TORCH_CHECK(
            _tensors[i].dim() == 2,
            "multi_dot(): tensor ",
            i,
            " must be 2D but got ",
            _tensors[i].dim(),
            "D");
        tensors[i] = _tensors[i];
      }

      // Ensure all tensors have the same device and dtype and check
      // that the shapes can be multiplied
      const auto dtype = tensors[0].dtype();
      const auto device = tensors[0].device();
      for (const auto i : irange(1, n)) {
        TORCH_CHECK(
            tensors[i].dtype() == dtype,
            "multi_dot(): all tensors must have be the same dtype but tensor 0 is ",
            dtype,
            " and tensor ",
            i,
            " ",
            tensors[i].dtype());
        TORCH_CHECK(
            tensors[i].device() == device,
            "multi_dot(): all tensors must be on the same device but tensor 0 is on ",
            device,
            " and tensor ",
            i,
            " on ",
            tensors[i].device());
        TORCH_CHECK(
            tensors[i - 1].size(-1) == tensors[i].size(0),
            "multi_dot(): tensors ",
            i - 1,
            " and ",
            i,
            " with shapes ",
            _tensors[i - 1].sizes(),
            " and ",
            _tensors[i].sizes(),
            " cannot be multiplied")
      }

      Tensor result;

      if (_out.has_value()) {
        auto out = *_out;
        TORCH_CHECK(
            dtype == out.dtype(),
            "multi_dot(): expected out tensor to have dtype ",
            dtype,
            " but got ",
            out.dtype());
        TORCH_CHECK(
            device == out.device(),
            "multi_dot(): expected out tensor to be on device ",
            device,
            " but got ",
            out.device());

        // If the last and last tensors have shapes (a, b) and (b, c) the
        // output has shape (a, c). If either the first or last tensor is 1D
        // a and/or c dimensions will be implicitely size 1 and will be ommited
        // from the output. e.g. for inputs (a, b) x (b) the output has shape (a,).
        native::resize_output(out, out_shape);

        // View output as 2D for simplicity of computation.
        result = out.view({tensors[0].size(0), tensors.back().size(-1)});
      }

      // The resize_ and view calls below are to ensure the
      // output shape respects the original dimensionality of
      // the first and last tensors which we are now viewed as 2D

      if (tensors.size() == 2) {
        return _out.has_value() ? mm_out(result, tensors[0], tensors[1])
                             : mm(tensors[0], tensors[1]).view(out_shape);
      }

      // Why the separate implementation for 3 matrices?
      // The logic for three matrices is much faster when done directly
      // Requires 1 comparison to 4 comparisons and fewer arithmetic operations
      if (tensors.size() == 3) {
        const auto a = tensors[0].size(0);
        const auto b = tensors[1].size(0);
        const auto c = tensors[2].size(0);
        const auto d = tensors[2].size(1);

        // The matrices are of size (a x b), (b x c), (c x d)
        // cost_1 is the cost of parenthesizing (a x b) and (b x c) and then
        // combining (c x d) cost_2 is the cost of parenthesizing (b x c) and (c x
        // d) and then combining (a x b)
        const auto cost_1 = (a * c) * (b + d);
        const auto cost_2 = (b * d) * (a + c);

        if (cost_1 > cost_2) {
          return _out.has_value()
              ? mm_out(result, tensors[0], mm(tensors[1], tensors[2]))
              : mm(tensors[0], mm(tensors[1], tensors[2])).view(out_shape);
        } else {
          return _out.has_value()
              ? mm_out(result, mm(tensors[0], tensors[1]), tensors[2])
              : mm(mm(tensors[0], tensors[1]), tensors[2]).view(out_shape);
        }
      }

      // Algorithm for multiplying 4 or more matrices
      const auto order = matrix_chain_order(tensors);
      const i64 i = 0;
      const i64 j = n - 1;

      if (_out.has_value()) {
        // We manually implement the first recursive layer here so we can use mm_out
        // for the final multiplication
        return mm_out(
            result,
            matrix_chain_multiplication(tensors, order, i, order[i][j]),
            matrix_chain_multiplication(tensors, order, order[i][j] + 1, j));
      }
      return matrix_chain_multiplication(tensors, order, i, j).view(out_shape);
        */
}

pub fn linalg_multi_dot(tensors: &[Tensor]) -> Tensor {
    
    todo!();
        /*
            return multi_dot_impl(tensors, nullopt);
        */
}

pub fn linalg_multi_dot_out<'a>(
        tensors: &[Tensor],
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            multi_dot_impl(tensors, result);
      return result;
        */
}

pub fn chain_matmul(matrices: &[Tensor]) -> Tensor {
    
    todo!();
        /*
            TORCH_WARN_ONCE(
          "torch.chain_matmul is deprecated and will be removed in a future PyTorch release. ",
          "Use torch.linalg.multi_dot instead, which accepts a list of two or more tensors rather than ",
          "multiple parameters."
      );
      checkAllSameDim(matrices, 2);

      TORCH_CHECK(
          matrices.size() > 0, "chain_matmul(): Expected one or more matrices");

      if (matrices.size() == 1) {
        return matrices[0].clone();
      }

      return native::linalg_multi_dot(matrices);
        */
}

pub fn chain_matmul_out<'a>(
        matrices: &[Tensor],
        result:   &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_WARN_ONCE(
          "torch.chain_matmul is deprecated and will be removed in a future PyTorch release. ",
          "Use torch.linalg.multi_dot instead, which accepts a list of two or more tensors rather than ",
          "multiple parameters."
      );
      checkAllSameDim(matrices, 2);

      TORCH_CHECK(
          matrices.size() > 0, "chain_matmul(): Expected one or more matrices");

      if (matrices.size() == 1) {
        native::resize_output(result, matrices[0].sizes());
        return result.copy_(matrices[0]);
      }

      return native::linalg_multi_dot_out(matrices, result);
        */
}

pub fn check_1d(
        t:   &Tensor,
        arg: *const u8,
        fn_: *const u8)  {
    
    todo!();
        /*
            TORCH_CHECK(t.dim() == 1, fn, ": Expected 1-D argument ", arg, ", but got ", t.dim(), "-D");
        */
}

pub fn check_addr_scalar(
        dtype:       ScalarType,
        scalar:      &Scalar,
        scalar_name: &String)  {
    
    todo!();
        /*
            TORCH_CHECK(
        !scalar.isBoolean() || dtype == ScalarType::Bool,
        "Boolean ", scalar_name, " only supported for Boolean results.");
      TORCH_CHECK(
        isFloatingType(dtype) || isComplexType(dtype) || scalar.isIntegral(true),
        "For integral input tensors, "
        "argument ", scalar_name ," must not be a floating point number.");
        */
}

pub fn build_addr_iter(
        result: &mut Tensor,
        self_:  &Tensor,
        vec1:   &Tensor,
        vec2:   &Tensor) -> TensorIterator {
    
    todo!();
        /*
            check_1d(vec1, "vec1", "addr");
      check_1d(vec2, "vec2", "addr");

      const auto vec1_size0 = vec1.sizes()[0];
      const auto vec2_size0 = vec2.sizes()[0];
      auto self_ = &result == &self
        ? MaybeOwned<Tensor>::borrowed(self)
        : expand_size(self, {vec1_size0, vec2_size0}, "addr");
      TORCH_CHECK(
        self_->dim() == 2,
        "2D tensor expected, got ", self_->dim(), "D tensor for input"
      );
      TORCH_CHECK(
        self_->sizes()[0] == vec1_size0 && self_->sizes()[1] == vec2_size0,
        "size mismatch, input: ", self_->sizes(),
        ", v1: ", vec1.sizes(),
        ", v2: ", vec2.sizes()
      );

      auto iter = TensorIteratorConfig()
        .set_check_mem_overlap(true)
        .add_output(result)
        .add_owned_input(*self_)
        .add_owned_input(vec1.reshape({vec1_size0, 1}))
        .add_input(vec2)
        .allow_cpu_scalars(true)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .enforce_safe_casting_to_output(true)
        .build();
      return iter;
        */
}

pub fn addr_a(
    self_: &Tensor,
    vec1:  &Tensor,
    vec2:  &Tensor,
    beta:  &Scalar,
    alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor result;
      auto iter = build_addr_iter(result, self, vec1, vec2);

      check_addr_scalar(iter.dtype(), beta, "beta");
      check_addr_scalar(iter.dtype(), alpha, "alpha");

      addr_stub(iter.device_type(), iter, beta, alpha);
      return iter.output();
        */
}

pub fn addr_b<'a>(
    self_: &mut Tensor,
    vec1:  &Tensor,
    vec2:  &Tensor,
    beta:  &Scalar,
    alpha: &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            return addr_out(self, self, vec1, vec2, beta, alpha);
        */
}


pub fn addr_out<'a>(
        self_:  &Tensor,
        vec1:   &Tensor,
        vec2:   &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            auto iter = build_addr_iter(result, self, vec1, vec2);

      check_addr_scalar(iter.dtype(), beta, "beta");
      check_addr_scalar(iter.dtype(), alpha, "alpha");

      addr_stub(iter.device_type(), iter, beta, alpha);
      return result;
        */
}

/**
  | The math_addr and math_addr_out functions
  | support backends other than CPU and CUDA, such
  | as XLA.
  |
  | They are implemented using the composition of
  | existing ops
  */
pub fn math_addr(
        self_: &Tensor,
        vec1:  &Tensor,
        vec2:  &Tensor,
        beta:  &Scalar,
        alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            // when beta==0, values in self should be ignored,
      // nans and infs in self should not propagate.
      if (beta.toComplexDouble() == 0.0) {
        if (alpha.toComplexDouble() == 1.0) {
          return outer(vec1, vec2);
        }
        return alpha * outer(vec1, vec2);
      }

      if (beta.toComplexDouble() == 1.0) {
        if (alpha.toComplexDouble() == 1.0) {
          return self + outer(vec1, vec2);
        }
        return self + alpha * outer(vec1, vec2);
      }

      if (alpha.toComplexDouble() == 1.0) {
        return beta * self + outer(vec1, vec2);
      }
      return beta * self + alpha * outer(vec1, vec2);
        */
}

pub fn math_addr_out<'a>(
        self_:  &Tensor,
        vec1:   &Tensor,
        vec2:   &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            auto addr_result = addr(self, vec1, vec2, beta, alpha);

      // Validates safe casting
      const auto result_dtype = addr_result.scalar_type();
      TORCH_CHECK(canCast(result_dtype, result.scalar_type()),
                  "result type ", result_dtype,
                  " can't be cast to the desired output type ", result.scalar_type());

      native::resize_output(result, addr_result.sizes().vec());
      result.copy_(addr_result);
      return result;
        */
}

/**
  | torch.ger, alias for torch.outer
  |
  */
pub fn ger_out<'a>(
        self_:  &Tensor,
        vec2:   &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_WARN("torch.ger is deprecated and will be removed in a future PyTorch release. "
                 "Use torch.outer instead.");
      return outer_out(result, self, vec2);
        */
}

pub fn ger(
        self_: &Tensor,
        vec2:  &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.outer(vec2);
        */
}

pub fn inner_out<'a>(
        self_: &Tensor,
        other: &Tensor,
        out:   &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            checkDeviceType("inner()", {out, self, other}, self.device().type());

      // If either self or other is a scalar just multiply them
      if (self.dim() == 0 || other.dim() == 0) {
        mul_out(out, self, other);
        return out;
      }

      // Last dimension should match (tensordot does not enforce this)
      TORCH_CHECK(
          self.size(-1) == other.size(-1),
          "inner() the last dimension must match on both input tensors but got shapes ",
          self.sizes(),
          " and ",
          other.sizes());

      tensordot_out(out, self, other, -1, -1);
      return out;
        */
}

pub fn inner(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            checkDeviceType("inner()", {self, other}, self.device().type());

      // If either self or other is a scalar just multiply them
      if (self.dim() == 0 || other.dim() == 0) {
        return self * other;
      }

      // Last dimension should match (tensordot does not enforce this)
      TORCH_CHECK(
          self.size(-1) == other.size(-1),
          "inner() the last dimension must match on both input tensors but got shapes ",
          self.sizes(),
          " and ",
          other.sizes());

      return tensordot(self, other, -1, -1);
        */
}

pub fn outer_out<'a>(
        self_:  &Tensor,
        vec2:   &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            check_1d(self, "self", "outer");
      check_1d(vec2, "vec2", "outer");

      // torch.outer is implemented as a composite op using reshape and mul
      mul_out(result, self.reshape({self.size(0), 1}), vec2);
      return result;
        */
}

pub fn outer(
        self_: &Tensor,
        vec2:  &Tensor) -> Tensor {
    
    todo!();
        /*
            check_1d(self, "self", "outer");
      check_1d(vec2, "vec2", "outer");

      return self.reshape({self.size(0), 1}) * vec2;
        */
}

pub fn addmm_impl_cpu(
        result: &mut Tensor,
        self_:  &Tensor,
        m1:     Tensor,
        m2:     Tensor,
        beta:   &Scalar,
        alpha:  &Scalar)  {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(self.dim() == 2 && m1.dim() == 2 && m2.dim() == 2);

      // Array access is faster than .size(n) and .stride(n)
      const auto self_sizes = self.sizes();
      auto m1_strides = m1.strides();
      auto m1_sizes = m1.sizes();
      auto m2_strides = m2.strides();
      auto m2_sizes = m2.sizes();

      // keeping TORCH_CHECKs here because othe mm methods also utilize this impl.
      // TODO move this to meta once all methods have migrated to structured kernel.
      TORCH_CHECK(
          m1_sizes[1] == m2_sizes[0], "mat1 and mat2 shapes cannot be multiplied (",
          m1_sizes[0], "x", m1_sizes[1], " and ", m2_sizes[0], "x", m2_sizes[1], ")");

      TORCH_CHECK(
          self_sizes[0] == m1_sizes[0] && self_sizes[1] == m2_sizes[1],
          "input shape is incompatible with matrix multiplication (",
          m1_sizes[0], "x", m1_sizes[1], " @ ", m2_sizes[0], "x", m2_sizes[1], " != ",
          self_sizes[0], "x", self_sizes[1], ")");

      native::resize_output(result, self_sizes);
      const auto result_strides = result.strides();
      const auto result_sizes = result.sizes();

      if (result.numel() == 0) {
        return;
      }

      if (beta.toComplexDouble() != 0.0 && !self.is_same(result)) {
        result.copy_(self);
      }

      bool transpose_c = false;
      Tensor c;

      // Cast result as matrix a
      if (result_strides[0] == 1 &&
          (result_sizes[1] == 1 || result_strides[1] >= max(i64{1}, result_sizes[0]))) {
        transpose_c = false;
        c = result;
      } else if (result_strides[1] == 1 &&
                 (result_sizes[0] == 1 || result_strides[0] >= max(i64{1}, result_sizes[1]))) {
        swap(m1, m2);
        swap(m1_sizes, m2_sizes);
        swap(m1_strides, m2_strides);
        transpose_c = true;
        c = result;
      } else {
        transpose_c = false;
        // make c FORTRAN contiguous
        c = result.transpose(0, 1).contiguous().transpose_(0, 1);
      }

      const i64 m = result_sizes[transpose_c ? 1 : 0];
      const i64 n = result_sizes[transpose_c ? 0 : 1];
      const i64 k = m1_sizes[transpose_c ? 0 : 1];

      // Cast m1 as matrix a
      bool transpose_a = false;
      Tensor a;
      /* Need lda >= max(1, (transpose_a ? k : m)) */
      if (m1_strides[transpose_c ? 1 : 0] == 1 &&
          m1_strides[transpose_c ? 0 : 1] >= max(i64{1}, m)) {
        transpose_a = false;
        a = m1;
      } else if (m1_strides[transpose_c ? 0 : 1] == 1 &&
                 m1_strides[transpose_c ? 1 : 0] >= max(i64{1}, k)) {
        transpose_a = true;
        a = m1;
      } else {
        transpose_a = !transpose_c;
        a = m1.clone(MemoryFormat::Contiguous);
      }

      // Cast m2 as matrix b
      bool transpose_b = false;
      Tensor b;
      /* Need ldm2_ >= max(1, (transpose_m2 == 'n' ? k : n)) */
      if (m2_strides[transpose_c ? 1 : 0] == 1 &&
          m2_strides[transpose_c ? 0 : 1] >= max(i64{1}, k)) {
        transpose_b = false;
        b = m2;
      } else if (m2_strides[transpose_c ? 0 : 1] == 1 &&
                 m2_strides[transpose_c ? 1 : 0] >= max(i64{1}, n)) {
        transpose_b = true;
        b = m2;
      } else {
        transpose_b = !transpose_c;
        b = m2.clone(MemoryFormat::Contiguous);
      }

      const i64 lda = a.strides()[(transpose_a == transpose_c) ? 1 : 0];
      const i64 ldb = b.strides()[(transpose_b == transpose_c) ? 1 : 0];
      const i64 ldc = c.strides()[transpose_c ? 0 : 1];

      // Apply BLAS routine
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16,
          result.scalar_type(), "addmm_impl_cpu_",
          [&]{
            native::cpublas::gemm(
                transpose_a ? cpublas::Transpose : cpublas::NoTranspose,
                transpose_b ? cpublas::Transpose : cpublas::NoTranspose,
                m, n, k,
                alpha.to<Scalar>(),
                a.data_ptr<Scalar>(), lda,
                b.data_ptr<Scalar>(), ldb,
                beta.to<Scalar>(),
                c.data_ptr<Scalar>(), ldc);
          });

      if (!c.is_same(result)) {
        result.copy_(c);
      }
        */
}

pub fn addbmm_impl(
        result: &mut Tensor,
        self_:  &Tensor,
        batch1: &Tensor,
        batch2: &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar)  {
    
    todo!();
        /*
            TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
      TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");
      TORCH_CHECK(batch1.size(0) == batch2.size(0),
          "batch1 and batch2 must have same number of batches, got ",
          batch1.size(0), " and ", batch2.size(0));
      TORCH_CHECK(batch1.size(2) == batch2.size(1),
          "Incompatible matrix sizes for bmm (",
          batch1.size(1), "x", batch1.size(2), " and ",
          batch2.size(1), "x", batch2.size(2), ")");

      const i64 dim1 = batch1.size(1);
      const i64 dim2 = batch2.size(2);
      TORCH_CHECK(self.size(0) == dim1 && self.size(1) == dim2,
          "self tensor does not match matmul output shape");

      result.resize_as_(self);

      if (beta.to<complex<double>>() != 0.0 && !self.is_same(result)) {
        result.copy_(self);
      }

      const i64 num_batches = batch1.size(0);

      if (num_batches == 0) {
        if (beta.to<complex<double>>() != 0.0) {
          result.mul_(beta);
        } else {
          result.zero_();
        }
        return;
      }

      auto adjusted_beta(beta);
      for (i64 batch = 0; batch < num_batches; ++batch) {
        result.addmm_(batch1[batch], batch2[batch], adjusted_beta, alpha);
        adjusted_beta = 1; // accumulate output once
      }
        */
}

pub fn addbmm_out<'a>(
        self_:  &Tensor,
        batch1: &Tensor,
        batch2: &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            auto b_self = expand_size(self, {batch1.size(1), batch2.size(2)}, "addbmm_out");
      {
        NoNamesGuard guard;
        addbmm_impl_(result, *b_self, batch1, batch2, beta, alpha);
      }
      auto names = namedinference::propagate_names_for_addmm(batch1, batch2, self);
      namedinference::propagate_names_if_nonempty(result, names);
      return result;
        */
}


pub fn addbmm_a<'a>(
        self_:  &mut Tensor,
        batch1: &Tensor,
        batch2: &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::addbmm_out(self, batch1, batch2, beta, alpha, self);
        */
}

pub fn addbmm_b(
        self_:  &Tensor,
        batch1: &Tensor,
        batch2: &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return native::addbmm_out(self, batch1, batch2, beta, alpha, result);
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(addmm_out_cpu)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, const Tensor &result) {
      auto b_self = expand_size(self, {mat1.sizes()[0], mat2.sizes()[1]}, "addmm_out");
      {
        NoNamesGuard guard;
        addmm_impl_cpu_(const_cast<Tensor&>(result), *b_self, mat1, mat2, beta, alpha);
      }
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(mm_out_cpu)(const Tensor & self, const Tensor & mat2, const Tensor & result) {
      {
        NoNamesGuard guard;
        addmm_impl_cpu_(const_cast<Tensor&>(result), result, self, mat2, 0, 1);
      }
    }
    */
}

#[inline] pub fn baddbmm_cpu_kernel<Scalar, const is_bmm: bool>(
        result: &Tensor,
        self_:  &Tensor,
        mat2:   &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar)  {

    todo!();
        /*
            i64 bs = result.size(0);
      i64 is = result.size(1);
      i64 js = result.size(2);
      i64 ks = self.size(2);

      Scalar alpha = alpha_.to<Scalar>();
      Scalar beta = beta_.to<Scalar>();

      auto r0 = result.accessor<Scalar, 3>();
      auto s0 = self.accessor<Scalar, 3>();
      auto m0 = mat2.accessor<Scalar, 3>();

      i64 grain_size = min(internal::GRAIN_SIZE / (is * js * ks), (i64)1);
      parallel_for(0, bs, grain_size, [&](i64 b_begin, i64 b_end) {
          for (i64 b = b_begin; b < b_end; b++) {
            auto r1 = r0[b];
            auto s1 = s0[b];
            auto m1 = m0[b];
            for (i64 i = 0; i < is; i++) {
              auto r2 = r1[i];
              auto s2 = s1[i];
              for (i64 j = 0; j < js; j++) {
                Scalar &r = r2[j];
                if (is_bmm) {
                  r = 0;
                  for (i64 k = 0; k < ks; k++) {
                    r += s2[k] * m1[k][j];
                  }
                } else {
                  r *= beta;
                  for (i64 k = 0; k < ks; k++) {
                    r += alpha * s2[k] * m1[k][j];
                  }
                }
              }
            }
          }
        });
        */
}

/**
  | This tries to apply some optimizations to
  | bmm/baddbmm:
  |
  | - When the operand size is small, computation
  |   are parallelized over the batch dimension
  |   using OMP and naive matrix multiplication is
  |   applied.
  |
  | - When the operand size is larger than the
  | threshold, if compiled with MKL, MKL's batch
  | gemm is used.
  |
  | - Otherwise, we use a series of matrix
  | multiplications.
  |
  | The threshold of 400 for the first has not been
  | thoroughly benchmarked yet and may have room
  | for further optimization, it likely depends on
  | the characteristics of the CPU, MKL will be
  | different from non-MKL etc., but this seems to
  | be a first starting point.
  */
#[inline] pub fn bmm_out_or_baddbmm<'a>(
        self_or_result: &mut Tensor,
        batch1:         &Tensor,
        batch2:         &Tensor,
        beta:           &Scalar,
        alpha:          &Scalar,
        is_bmm_out:     bool) -> &'a mut Tensor {
    
    todo!();
        /*
            // is_bmm_out: true for bmm_out, false for baddbmm_
      // self_or_result is "self" for baddbmm_ and "result" for bmm_out
      CheckedFrom c = (is_bmm_out ? "bmm" : "baddbmm");

      auto checkOnCPU = [](const Tensor& t, CheckedFrom c) {
        TORCH_CHECK(
            !t.is_cuda(),
            "Expect tensor to have CPU backend, but got tensor with ",
            toString(t.options().backend()),
            " Backend (while checking arguments for ",
            c);
      };

      checkOnCPU(self_or_result, c);
      checkOnCPU(batch1, c);
      checkOnCPU(batch2, c);

      checkDim(c, batch1, "batch1", /* pos */ 1, /* dim */ 3);
      checkDim(c, batch2, "batch2", /* pos */ 2, /* dim */ 3);

      const auto batch1_sizes = batch1.sizes();
      const auto batch2_sizes = batch2.sizes();

      i64 bs = batch1_sizes[0];
      i64 contraction_size = batch1_sizes[2];
      i64 res_rows = batch1_sizes[1];
      i64 res_cols = batch2_sizes[2];

      TORCH_CHECK(batch2_sizes[0] == bs && batch2_sizes[1] == contraction_size);

      if (is_bmm_out) {
        // Here it is result
        self_or_result.resize_({bs, res_rows, res_cols});
      } else {
        const auto self_sizes = self_or_result.sizes();
        TORCH_CHECK(self_sizes[0] == bs && self_sizes[1] == res_rows && self_sizes[2] == res_cols);
      }

      // handle pathological cases that blas may not like
      if (self_or_result.numel() == 0) {
        return self_or_result;
      } else if (contraction_size == 0) {
        if (is_bmm_out || (beta.to<complex<double>>() == 0.0)) {
          return self_or_result.zero_();
        } else {
          return self_or_result.mul_(beta);
        }
      }

      auto batch_items_contiguous_or_transposed = [&](const Tensor& t) {
        const auto sizes = t.sizes();
        const auto strides = t.strides();
        return (strides[2] == 1 && strides[1] >= sizes[2])
                || (strides[1] == 1 && strides[2] >= sizes[1]);
      };

      if (contraction_size * res_rows * res_cols < 400) {
        if (is_bmm_out) {
          AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, batch1.scalar_type(), "bmm", [&] {
              baddbmm_cpu_kernel<Scalar, true>(self_or_result, batch1, batch2, beta, alpha);
            });
        } else {
          AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, batch1.scalar_type(), "baddbmm", [&] {
              baddbmm_cpu_kernel<Scalar, false>(self_or_result, batch1, batch2, beta, alpha);
            });
        }
      } else if (hasMKL() && ((
                self_or_result.scalar_type() != kHalf &&
                self_or_result.scalar_type() != kBFloat16 &&
                native::is_floating_point(self_or_result)) ||
                native::is_complex(self_or_result))
                && batch_items_contiguous_or_transposed(batch1)
                && batch_items_contiguous_or_transposed(batch2)
                && self_or_result.is_contiguous()) {
        native::_baddbmm_mkl_(self_or_result, batch1, batch2, beta, alpha);
      } else { // split along batch dimension
    #ifdef C10_MOBILE
        /*
         * We only do multithreading when Inference mode is enabled because various
         * thread local state is not appropriately propagated through
         * parallel_for. e.g. RecordFunction related state, dispatchKeySet Big
         * concern with this is that if we use parallel_for where state is not
         * propagated then dispatch machinery may work differently on main thread
         * vs. other threads, leading to undefined behavior.
         * Thus it is recommended to not use parallel_for where lambdas do
         * ops that go through dispatcher.
         * For now we circument this by InferenceMode guard in order to unlock
         * performance.
         * Longer term we probably want a separate API that explicitly calls out
         * the TLS that it propagates.
         * Also note that this is enabled for mobile only because blas
         * implementation for non-mobile build is already multithreaded.
         */
        // Benchmarking was done as follows:
        // bmm_test: operator benchmark under
        // benchmarks/operator_benchmarks/pt/bmm_test.py Ran this benchmark for
        // various matrix sizes on Samsung S8U
        const bool enable_multithreaded_bmm = InferenceMode::is_enabled() &&
            bs >= 4 && res_rows >= 4 && res_cols >= 16 && contraction_size >= 16;
    #else
        const bool enable_multithreaded_bmm{false};
    #endif
        if (is_bmm_out) {
          if (enable_multithreaded_bmm) {
            auto bmm_out_fn = [&](u64 start, u64 end) {
              InferenceMode guard;
              for (i64 b = start; b < end; b++) {
                auto r = self_or_result.select(0, b);
                addmm_impl_cpu_(
                    r, r, batch1.select(0, b), batch2.select(0, b), 0, 1);
              }
            };
            parallel_for(0, bs, 1, bmm_out_fn);
          } else {
            for (i64 b = 0; b < bs; b++) {
              auto r = self_or_result.select(0, b);
              addmm_impl_cpu_(r, r, batch1.select(0, b), batch2.select(0, b), 0, 1);
            }
          }
        } else {
          if (enable_multithreaded_bmm) {
            auto bmm_fn = [&](u64 start, u64 end) {
              InferenceMode guard;
              for (i64 b = start; b < end; b++) {
                self_or_result.select(0, b).addmm_(
                    batch1.select(0, b), batch2.select(0, b), beta, alpha);
              }
            };
            parallel_for(0, bs, 1, bmm_fn);
          } else {
            for (i64 b = 0; b < bs; b++) {
              self_or_result.select(0, b).addmm_(
                  batch1.select(0, b), batch2.select(0, b), beta, alpha);
            }
          }
        }
      }
      return self_or_result;
        */
}

pub fn baddbmm_cpu_a(
        self_:  &Tensor,
        batch1: &Tensor,
        batch2: &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return native::baddbmm_out_cpu(self, batch1, batch2, beta, alpha, result);
        */
}

pub fn baddbmm_out_cpu<'a>(
        self_:  &Tensor,
        batch1: &Tensor,
        batch2: &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            auto self = expand_size(self_, {batch1.size(0), batch1.size(1), batch2.size(2)}, "baddbmm");
      result.resize_(self->sizes());
      result.copy_(*self);
      return native::baddbmm__cpu(result, batch1, batch2, beta, alpha);
        */
}

pub fn baddbmm_cpu_b<'a>(
        self_:  &mut Tensor,
        batch1: &Tensor,
        batch2: &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            return bmm_out_or_baddbmm_(self, batch1, batch2, beta, alpha, false);
        */
}

pub fn bmm_cpu(
        self_: &Tensor,
        mat2:  &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return native::bmm_out_cpu(self, mat2, result);
        */
}

pub fn bmm_out_cpu<'a>(
        batch1: &Tensor,
        batch2: &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            Scalar beta(0.0);
      Scalar alpha(1.0);
      {
      NoNamesGuard guard;
      bmm_out_or_baddbmm_(result, batch1, batch2, beta, alpha, true);
      }
      namedinference::propagate_names_if_nonempty(
          result,
          namedinference::compute_bmm_outnames(result, batch1, batch2));
      return result;
        */
}

pub fn dot_out<'a>(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            auto output_device = result.device();
      auto input1_device = self.device();
      auto input2_device = other.device();
      // check if the input & output tensors are on the same device.
      TORCH_CHECK(
        (output_device == input1_device) && (input1_device == input2_device),
        "dot: Expected the output and input tensors to be on the "
        "same device, but got the output tensor on ", output_device,
        ", the 'input' tensor on ", input1_device, ", and the 'other' tensor on ", input2_device);
      native::resize_output(result, {});
      TORCH_CHECK(result.scalar_type() == self.scalar_type(),
               "result dtype ", result.scalar_type(), " does not match input dtype ", self.scalar_type());
      return result.fill_(self.dot(other));
        */
}

pub fn vdot_out<'a>(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            auto output_device = result.device();
      auto input1_device = self.device();
      auto input2_device = other.device();
      // check if the input & output tensors are on the same device.
      TORCH_CHECK(
        (output_device == input1_device) && (input1_device == input2_device),
        "vdot: Expected the output and input tensors to be on the "
        "same device, but got the output tensor on ", output_device,
        ", the 'input' tensor on ", input1_device, ", and the 'other' tensor on ", input2_device);
      native::resize_output(result, {});
      TORCH_CHECK(result.scalar_type() == self.scalar_type(),
               "result dtype ", result.scalar_type(), " does not match input dtype ", self.scalar_type());
      return result.fill_(self.vdot(other));
        */
}

/**
  | Matrix product of two Tensors.
  | 
  | The behavior depends on the dimensionality
  | of the Tensors as follows:
  | 
  | - If both Tensors are 1-dimensional,
  | the dot product (scalar) is returned.
  | 
  | - If both arguments are 2-dimensional,
  | the matrix-matrix product is returned.
  | 
  | - If the first argument is 1-dimensional
  | and the second argument is 2-dimensional,
  | a 1 is prepended to its dimension for
  | the purpose of the matrix multiply.
  | 
  | After the matrix multiply, the prepended
  | dimension is removed.
  | 
  | - If the first argument is 2-dimensional
  | and the second argument is 1-dimensional,
  | the matrix-vector product is returned.
  | 
  | - If both arguments are at least 1-dimensional
  | and at least one argument is
  | 
  | N-dimensional (where N > 2), then a batched
  | matrix multiply is returned. If the
  | first argument is 1-dimensional, a
  | 1 is prepended to its dimension for the
  | purpose of the batched matrix multiply
  | and removed after. If the second argument
  | is 1-dimensional, a 1 is appended to
  | its dimension for the purpose of the
  | batched matrix multiple and removed
  | after.
  | 
  | The non-matrix (i.e. batch) dimensions
  | are broadcasted (and thus must be broadcastable).
  | For example, if tensor1 is a (j x 1 x n x
  | m) Tensor and tensor2 is a (k x m x p) Tensor,
  | the returned tensor will be an (j x k x
  | n x p) Tensor.
  |
  */
pub fn matmul_a(
        out_opt: Option<Tensor>,
        tensor1: &Tensor,
        tensor2: &Tensor) -> Tensor {
    
    todo!();
        /*
            NoNamesGuard guard;
      auto dim_tensor1 = tensor1.dim();
      auto dim_tensor2 = tensor2.dim();
      auto has_out = out_opt.has_value();
      Tensor out = out_opt.value_or(Tensor());

      if (dim_tensor1 == 1 && dim_tensor2 == 1) {
        return has_out ? native::dot_out(tensor1, tensor2, out) : tensor1.dot(tensor2);
      } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
        return has_out ? mv_out(out, tensor1, tensor2) : tensor1.mv(tensor2);
      } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
        return has_out ? mm_out(out, tensor1.unsqueeze(0), tensor2).squeeze_(0)
                       : tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
      } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
        return has_out ? mm_out(out, tensor1, tensor2) : tensor1.mm(tensor2);
      } else if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
        // optimization: use mm instead of bmm by folding tensor1's batch into
        // its leading matrix dimension.

        Tensor t2 = dim_tensor2 == 1 ? tensor2.unsqueeze(-1) : tensor2;
        auto size1 = tensor1.sizes();
        auto size2 = t2.sizes();
        vector<i64> output_size;
        output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
        if (dim_tensor2 > 1) {
          output_size.push_back(size2[dim_tensor2 - 1]);
        }

        // fold the batch into the first dimension
        Tensor t1 = tensor1.expect_contiguous()->view({-1, size1[size1.size() - 1]});
        Tensor output = has_out ? _unsafe_view(mm_out(out, t1, t2), output_size)
                                : _unsafe_view(t1.mm(t2), output_size);
        return has_out ? out.set_(output) : output;
      } else if ((dim_tensor1 == 1 || dim_tensor1 == 2) && dim_tensor2 >= 3) {
        // optimization: transpose the inner dimensions of the arguments, call
        // matmul on the swapped arguments, then transpose the inner dimensions
        // of the result.
        const i64 n = dim_tensor1 == 2 ? tensor1.size(-2) : 1;
        const i64 m = tensor1.size(-1);
        const i64 p = tensor2.size(-1);

        const Tensor t2_T = tensor2.transpose(-1, -2);
        const Tensor t1_T = dim_tensor1 == 2 ? tensor1.t() : tensor1.reshape({n, m}).t();
        const Tensor res_T = matmul(out_opt, t2_T, t1_T);

        if (dim_tensor1 == 2) {
          Tensor res = res_T.transpose(-1, -2).contiguous();
          return has_out ? out.set_(res) : res;
        }
        else {
          vector<i64> shape = tensor2.sizes().slice(0, dim_tensor2 - 2).vec();
          shape.push_back(p);

          Tensor res = res_T.reshape(shape).contiguous();
          return has_out ? out.set_(res) : res;
        }
      } else if ((dim_tensor1 >= 1 && dim_tensor2 >= 1) && (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
        // We are multiplying b1 x n x m1 by x2 x m2 x p (where b1 can be a list);
        // we track m1 vs m2 separately even though they must match for nicer error messages
        i64 n = dim_tensor1 > 1 ? tensor1.size(-2) : 1;
        i64 m1 = tensor1.size(-1);
        IntArrayRef batch_tensor1(tensor1.sizes().data(), max<i64>(dim_tensor1 - 2, 0));
        i64 m2 = dim_tensor2 > 1 ? tensor2.size(-2) : 1;
        i64 p = tensor2.size(-1);
        IntArrayRef batch_tensor2(tensor2.sizes().data(), max<i64>(dim_tensor2 - 2, 0));

        // expand the batch portion (i.e. cut off matrix dimensions and expand rest)
        vector<i64> expand_batch_portion = infer_size(batch_tensor1, batch_tensor2);

        vector<i64> tensor1_expand_size(expand_batch_portion);
        tensor1_expand_size.insert(tensor1_expand_size.end(), {n, m1});

        vector<i64> tensor2_expand_size(expand_batch_portion);
        tensor2_expand_size.insert(tensor2_expand_size.end(), {m2, p});

        const i64 expand_batch_product =
            multiply_integers(expand_batch_portion);

        vector<i64> tensor1_bmm_view({expand_batch_product});
        tensor1_bmm_view.insert(tensor1_bmm_view.end(), {n, m1});

        vector<i64> tensor2_bmm_view({expand_batch_product});
        tensor2_bmm_view.insert(tensor2_bmm_view.end(), {m2, p});

        // flatten expanded batches
        Tensor tensor1_expanded = tensor1.expand(tensor1_expand_size).reshape(tensor1_bmm_view);
        Tensor tensor2_expanded = tensor2.expand(tensor2_expand_size).reshape(tensor2_bmm_view);

        // reshape batches back into result
        vector<i64> output_shape(expand_batch_portion);
        if (dim_tensor1 > 1) {
          output_shape.push_back(n);
        }
        if (dim_tensor2 > 1) {
          output_shape.push_back(p);
        }

        Tensor output = has_out ? _unsafe_view(bmm_out(out, tensor1_expanded, tensor2_expanded), output_shape)
                                : _unsafe_view(tensor1_expanded.bmm(tensor2_expanded), output_shape);

        return has_out ? out.set_(output) : output;
      }

     AT_ERROR("both arguments to matmul need to be at least 1D, but they are ",
              dim_tensor1, "D and ", dim_tensor2, "D");
        */
}

pub fn matmul_b(
        tensor1: &Tensor,
        tensor2: &Tensor) -> Tensor {
    
    todo!();
        /*
            auto maybe_outnames = namedinference::compute_matmul_outnames(tensor1, tensor2);
      auto result = native::matmul(nullopt, tensor1, tensor2);
      namedinference::propagate_names_if_nonempty(result, maybe_outnames);
      return result;
        */
}

pub fn matmul_out<'a>(
        tensor1: &Tensor,
        tensor2: &Tensor,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            auto maybe_outnames = namedinference::compute_matmul_outnames(tensor1, tensor2);
      native::matmul(optional<Tensor>(result), tensor1, tensor2);
      namedinference::propagate_names_if_nonempty(result, maybe_outnames);
      return result;
        */
}

/**
  | we consider 6 Taylor expansions of degree
  | 1, 2, 4, 8, 12, 18
  |
  */
pub const TOTAL_N_DEGS: i32 = 6;

pub fn operator_1_norm(tensor: &Tensor) -> Tensor {
    
    todo!();
        /*
            return get<0>(tensor.abs().sum(-2).max(-1));
        */
}

/**
  | Allocates a buffers of uninitialized
  | or zero values of shape [n_copies, a.size()]
  |
  */
pub fn allocate_buffer(
        a:        &Tensor,
        n_copies: i32,
        is_zero:  bool) -> Tensor {
    let is_zero: bool = is_zero.unwrap_or(false);

    todo!();
        /*
            auto res = empty(
        {n_copies, a.size(0), a.size(1), a.size(2)},
        a.options().memory_format(MemoryFormat::Contiguous)
      );

      if (is_zero) {
        res.zero_();
      }

      return res;
        */
}

/**
  | Makes `buffer` to store `num_matrices` number of matrices needed for
  | compute the matrix exponentials of different orders, i.e.
  | first `num_matrices` matrices from the list l := {I, A, A^2, A^3, A^6}
  | in a contiguous block of memory such that
  | buffer[0, ...] = l[0], // I
  | buffer[1, ...] = l[1], // A
  | ...
  | buffer[num_matrices - 1, ...] = l[num_matries - 1]
  */
pub fn fill_matrix_powers(
        buffer:       &mut Tensor,
        a:            &Tensor,
        num_matrices: i32)  {
    
    todo!();
        /*
            auto a_sizes_minus_last = a.sizes().vec();
      a_sizes_minus_last.pop_back();
      // fill I
      buffer.select(0, 0).copy_(
        diag_embed(
          ones({1}, buffer.options())
            .expand(a_sizes_minus_last)
        )
      );

      // fill a
      buffer.select(0, 1).copy_(a);

      // fill a^2
      if (2 <= num_matrices - 1) {
        native::matmul(
          buffer.select(0, 2), // out for a^2
          buffer.select(0, 1),
          buffer.select(0, 1)
        );
      }

      // fill a^3
      if (3 <= num_matrices - 1) {
        native::matmul(
          buffer.select(0, 3), // out for a^3
          buffer.select(0, 1),
          buffer.select(0, 2)
        );
      }

      // fill a^6
      if (4 <= num_matrices - 1) {
        native::matmul(
          buffer.select(0, 4),
          buffer.select(0, 3),
          buffer.select(0, 3)
        );
      }
        */
}

#[inline] pub fn move_memory_if_cuda_input(
        mem: &Tensor,
        in_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return (in.device().type() == kCUDA)
        ? mem.to(device_of(in).value())
        : mem;
        */
}

/**
  | convert a 1D blob to a 2D Tensor of size [1,
  | blob.size()] such that blob.device() ==
  | in.device()) designed to be used with
  | _compute_linear_combination
  */
#[inline] pub fn blob_to_tensor<Scalar>(
        blob: InitializerList<Scalar>,
        in_:  &Tensor) -> Tensor {

    todo!();
        /*
            // we convert to void* expecitly because begin() returns
      // a pointer to a constant.
      // Blob is assumed to be a 1D array, that is why
      // we also insert a fake dimension so that the result could directly
      // be used in _compute_linear_combination
      auto tensor = from_blob((void*)blob.begin(), blob.size(),
        toValueType(in.scalar_type())).unsqueeze(0);
      return _move_memory_if_cuda_input(tensor, in);
        */
}

// I + A
pub fn compute_t1(A: &Tensor) -> Tensor {
    
    todo!();
        /*
            // 2 for {I, A}
      auto As = _allocate_buffer(A, 2);
      _fill_matrix_powers(As, A, 2);
      return As.sum(0);
        */
}

// I + A + A^2 / 2
pub fn compute_t2(A: &Tensor) -> Tensor {
    
    todo!();
        /*
            auto As = _allocate_buffer(A, 3);
      // 3 for {I, A, A^2}
      _fill_matrix_powers(As, A, 3);
      As.select(0, 2).div_(2.0);
      return As.sum(0);
        */
}

/// I + A + A^2 * (I / 2 + A / 6 + A^2 / 24)
pub fn compute_t4<Scalar>(A: &Tensor) -> Tensor {

    todo!();
        /*
            auto As = _allocate_buffer(A, 4);
      // 3 for {I, A, A^2}
      _fill_matrix_powers(As, A, 3);

      native::matmul(
        // output for A^2 * (I / 2 + A / 6 + A^2 / 24)
        As.select(0, 3),
        // contains A^2
        As.select(0, 2),
        // computes (I / 2 + A / 6 + A^2 / 24)
        native::_compute_linear_combination(
          As.narrow(0, 0, 3),
          _blob_to_Tensor<Scalar>({1 / 2.0, 1 / 6.0, 1 / 24.0}, A)
        )
      );

      // I + A + A^2 * (I / 2 + A / 6 + A^2 / 24)
      return native::_compute_linear_combination(
        As, _blob_to_Tensor<Scalar>({1.0, 1.0, 0.0, 1.0}, A)
      );
        */
}

pub fn compute_t8<Scalar>(A: &Tensor) -> Tensor {

    todo!();
        /*
            constexpr Scalar sqrt_177 = 0.1330413469565007072504e+2;
      constexpr Scalar x3 = 2. / 3.;
      constexpr Scalar x1 = x3 * ((1. + sqrt_177) / 88.);
      constexpr Scalar x2 = x3 * ((1. + sqrt_177) / 352.);
      constexpr Scalar x4 = (-271. + 29. * sqrt_177) / (315. * x3);
      constexpr Scalar x5 = (-11. + 11. * sqrt_177) / (1260. * x3);
      constexpr Scalar x6 = (-99. + 11. * sqrt_177) / (5040. * x3);
      constexpr Scalar x7 = (89. - sqrt_177) / (5040. * x3);
      constexpr Scalar y2 = (857. - 58. * sqrt_177) / 630.;

      auto As = _allocate_buffer(A, 5);
      // 3 for {I, A, A^2}
      _fill_matrix_powers(As, A, 3);

      // A4 =  A2 * (x1 * A + x2 * A2)
      native::matmul(
        // output for A4
        As.select(0, 3),
        // As.select(0, 2) = A^2
        As.select(0, 2),
        native::_compute_linear_combination(
          // extract {A, A^2} from As
          As.narrow(0, 1, 2),
          _blob_to_Tensor<Scalar>({x1, x2}, A)
        )
      );

      // A8 = (x3 * A2 + A4) * (x4 * I + x5 * A + x6 * A2 + x7 * A4)
      native::matmul(
        // output for A8
        As.select(0, 4),
        // x3 * A2 + A4
        native::_compute_linear_combination(
          As.narrow(0, 2, 2),
          _blob_to_Tensor<Scalar>({x3, 1.0}, A)
        ),
        native::_compute_linear_combination(
          As.narrow(0, 0, 4),
          _blob_to_Tensor<Scalar>({x4, x5, x6, x7}, A)
        )
      );

      // return I + A + y2 * A2 + A8;
      return native::_compute_linear_combination(
        As,
        _blob_to_Tensor<Scalar>({1.0, 1.0, y2, 0.0, 1.0}, A)
      );
        */
}

pub fn compute_t12<Scalar>(A: &Tensor) -> Tensor {

    todo!();
        /*
            constexpr int num_prods = 4;
      array2d<Scalar, num_prods, num_prods> b = {{
        {
          9.0198e-16,
          0.46932117595418237389,
          -0.20099424927047284052,
          -0.04623946134063071740
        },
        {
          5.31597895759871264183,
          1.19926790417132231573,
          0.01179296240992997031,
          0.01108844528519167989
        },
        {
          0.18188869982170434744,
          0.05502798439925399070,
          0.09351590770535414968,
          0.00610700528898058230
        },
        {
          -2.0861320e-13,
          -0.13181061013830184015,
          -0.02027855540589259079,
          -0.00675951846863086359
        }
      }};

      // gather coefficients `b` from above into a tensor,
      // and move them to device `device_of(A)`
      auto bs = from_blob(
        reinterpret_cast<void*>(&b),
        {num_prods, num_prods},
        {num_prods, 1},
        toValueType(A.scalar_type())
      );
      bs = _move_memory_if_cuda_input(bs, A);

      auto As = _allocate_buffer(A, num_prods);
      _fill_matrix_powers(As, A, num_prods);

      auto Bs = native::_compute_linear_combination(As, bs);

      // compute A6
      Bs.select(0, 2).add_(native::matmul(
        // tmp buffer for this matrix product
        As.select(0, 0),
        Bs.select(0, 3),
        Bs.select(0, 3)
      ));

      return Bs.select(0,0).add_(native::matmul(
        // tmp buffer for this matrix product
        As.select(0, 0),
        Bs.select(0, 1).add_(Bs.select(0, 2)),
        Bs.select(0, 2)
      ));
        */
}

pub fn compute_t18<Scalar>(A: &Tensor) -> Tensor {

    todo!();
        /*
            constexpr int num_prods = 5;
      array2d<Scalar, num_prods, num_prods> b = {{
        {
          0.,
          -1.00365581030144618291e-01,
          -8.02924648241156932449e-03,
          -8.92138498045729985177e-04,
          0.
        },
        {
          0.,
          3.97849749499645077844e-01,
          1.36783778460411720168e+00,
          4.98289622525382669416e-01,
          -6.37898194594723280150e-04
        },
        {
          -1.09676396052962061844e+01,
          1.68015813878906206114e+00,
          5.71779846478865511061e-02,
          -6.98210122488052056106e-03,
          3.34975017086070470649e-05
        },
        {
          -9.04316832390810593223e-02,
          -6.76404519071381882256e-02,
          6.75961301770459654925e-02,
          2.95552570429315521194e-02,
          -1.39180257516060693404e-05
        },
        {
          0.,
          0.,
          -9.23364619367118555360e-02,
          -1.69364939002081722752e-02,
          -1.40086798182036094347e-05
        }
      }};

      // gather coefficients `b` from above into a tensor,
      // and move them to device `device_of(A)`
      auto bs = from_blob(
        reinterpret_cast<void*>(&b),
        {num_prods, num_prods},
        {num_prods, 1},
        toValueType(A.scalar_type())
      );
      bs = _move_memory_if_cuda_input(bs, A);

      auto As = _allocate_buffer(A, num_prods);
      _fill_matrix_powers(As, A, num_prods);

      auto Bs = native::_compute_linear_combination(As, bs);

      // compute A9
      Bs.select(0, 3).add_(native::matmul(
        // tmp buffer for this matrix product
        As.select(0, 0),
        Bs.select(0, 0),
        Bs.select(0, 4))
      );

      return Bs.select(0, 1).add_(native::matmul(
        // tmp buffer for this matrix product
        As.select(0, 0),
        Bs.select(0, 2).add_(Bs.select(0, 3)),
        Bs.select(0, 3)
      ));
        */
}

pub fn compute_t18_scale_square<Scalar>(
        mexp_out: &mut Tensor,
        a:        &Tensor,
        norm:     &Tensor,
        theta:    Scalar)  {

    todo!();
        /*
            // Scale
      const auto s = max(
        zeros_like(norm),
        ceil(log2(norm / theta))
      ).unsqueeze(-1).unsqueeze(-1).to(kLong);
      const auto pow2s = pow(2, s);
      const auto a_scaled = a / pow2s;

      // Square
      auto mexp_scaled = native::compute_T18<Scalar>(a_scaled);
      auto s_cpu = (s.device().type() == kCPU)
        ? s : s.to(kCPU);
      for (i64 i = 0; i < mexp_scaled.size(0); ++i) {
        auto s_val = s_cpu.select(0, i).template item<i64>();
        auto mexp = mexp_scaled.select(0, i);
        for (i64 p = 0; p < s_val; ++p) {
          mexp = matmul(mexp, mexp);
        }
        mexp_out.select(0, i).copy_(mexp);
      }
        */
}

pub fn mexp_impl<Scalar>(
    a:                             &Tensor,
    thetas:                        Array<Scalar,TotalNDegs>,
    compute_highest_degree_approx: bool) -> Tensor {

    let compute_highest_degree_approx: bool = compute_highest_degree_approx.unwrap_or(false);

    todo!();
        /*
            auto res = empty_like(a);
      const auto norm = operator_1_norm(a);
      // `norm_cpu` is used to decide which Tensors require which approximation
      // based on their norm. This decision takes place on CPU.
      // It requires moving data back and forth between devices when `a` is on CUDA,
      // but at the cost of only one sigle CPU-CUDA synchronization (instead of 6),
      // and better performance overall (benchmarked).
      const auto norm_cpu = (a.device().type() == kCUDA)
        ? norm.to(kCPU) : norm;

      if (!compute_highest_degree_approx) {
        constexpr array<
          Tensor(*)(const Tensor&),
          total_n_degs - 1>
        compute_Ts = {
          compute_T1, compute_T2, compute_T4<Scalar>,
          compute_T8<Scalar>, compute_T12<Scalar>
        };

        for (int i = 0; i < total_n_degs - 1; ++i) {
          auto norm_lower_bound = (i == 0) ? static_cast<Scalar>(-1) : thetas[i - 1];
          auto norm_upper_bound = thetas[i];
          // nonzero returns a 2D tensor, hence squeeze(-1) to make it 1D
          auto idx_curr_norm_interval = (
            (norm_lower_bound < norm_cpu) * (norm_cpu <= norm_upper_bound)
          ).nonzero().squeeze(-1);

          if (idx_curr_norm_interval.numel()) {
            auto idx_to_device = _move_memory_if_cuda_input(
              idx_curr_norm_interval, a
            );
            auto sub_a = index_select(a, 0, idx_to_device);
            res.index_put_({idx_to_device}, compute_Ts[i](sub_a));
          }
        }

        // nonzero returns a 2D tensor, hence squeeze(-1) to make it 1D
        auto idx_large_norm = (norm_cpu >= thetas[total_n_degs - 2])
          .nonzero().squeeze(-1);

        if (idx_large_norm.numel()) {
          auto idx_to_device = _move_memory_if_cuda_input(
            idx_large_norm, a
          );
          auto a_large_norm = index_select(a, 0, idx_to_device);
          auto large_norm_subset = index_select(norm, 0, idx_to_device);
          auto mexp_out = empty_like(a_large_norm);

          compute_T18_scale_square(
            mexp_out,
            a_large_norm,
            large_norm_subset,
            thetas[total_n_degs - 1]
          );
          res.index_put_({idx_large_norm}, mexp_out);
        }

        return res;
      }

      compute_T18_scale_square(
        res, a, norm,
        thetas[total_n_degs - 1]
      );

      return res;
        */
}

/// matrix exponential
pub fn mexp(
    a:                             &Tensor,
    compute_highest_degree_approx: bool) -> Tensor {

    let compute_highest_degree_approx: bool = compute_highest_degree_approx.unwrap_or(false);

    todo!();
        /*
            // squash batch dimensions to one dimension for simplicity
      const auto a_3d = a.view({-1, a.size(-2), a.size(-1)});

      if (a.scalar_type() == ScalarType::Float
          || a.scalar_type() == ScalarType::ComplexFloat) {
        constexpr array<float, total_n_degs> thetas_float = {
          1.192092800768788e-07, // deg 1
          5.978858893805233e-04, // deg 2
          5.116619363445086e-02, // deg 4
          5.800524627688768e-01, // deg 8
          1.461661507209034e+00, // deg 12
          3.010066362817634e+00  // deg 18
        };

        return mexp_impl<float>(a_3d, thetas_float, compute_highest_degree_approx)
          .view(a.sizes());
      }
      else { // if Double or ComplexDouble
        constexpr array<double, total_n_degs> thetas_double = {
          2.220446049250313e-16, // deg 1
          2.580956802971767e-08, // deg 2
          3.397168839976962e-04, // deg 4
          4.991228871115323e-02, // deg 8
          2.996158913811580e-01, // deg 12
          1.090863719290036e+00  // deg 18
        };

        return mexp_impl<double>(a_3d, thetas_double, compute_highest_degree_approx)
          .view(a.sizes());
      }
        */
}

/**
  | Based on:
  |
  | Mathias, Roy.
  | A Chain Rule for Matrix Functions and Applications.
  | SIAM J. Matrix Anal. Appl. 17 (1996): 610-620.
  |
  */
pub fn backward_analytic_function_of_a_matrix<func_t>(
        self_:                &Tensor,
        grad:                 &Tensor,
        function_of_a_matrix: &Func) -> Tensor {

    todo!();
        /*
            auto self_transposed = self.transpose(-2, -1).conj();
      auto self_transposed_sizes = self_transposed.sizes().vec();
      self_transposed_sizes[self.dim() - 2] <<= 1;
      self_transposed_sizes[self.dim() - 1] <<= 1;

      auto n = self_transposed.size(-1);
      auto meta_grad = zeros(self_transposed_sizes, grad.options());
      meta_grad.narrow(-2, 0, n).narrow(-1, 0, n).copy_(self_transposed);
      meta_grad.narrow(-2, n, n).narrow(-1, n, n).copy_(self_transposed);
      meta_grad.narrow(-2, 0, n).narrow(-1, n, n).copy_(grad);

      auto grad_input = function_of_a_matrix(meta_grad)
        .narrow(-2, 0, n).narrow(-1, n, n);
      return grad_input;
        */
}

/**
  | Computes the matrix exponential for
  | a given batch of squared matrices.
  | 
  | The implementaion is based on:
  | 
  | Bader, P.; Blanes, S.; Casas, F.
  | 
  | Computing the Matrix Exponential with
  | an Optimized Taylor Polynomial Approximation.
  | 
  | Mathematics 2019, 7, 1174.
  |
  */
pub fn matrix_exp(a: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(a.dim() >= 2
              && (isFloatingType(a.scalar_type())
               || isComplexType(a.scalar_type())),
                  "matrix_exp(", a.scalar_type(), "{", a.sizes(), "}): expected a tensor "
                  "of floating or complex types with dim at least 2");
      TORCH_CHECK(a.size(-1) == a.size(-2),
                  "matrix_exp(", a.scalar_type(), "{", a.sizes(), "}): expected a tensor "
                  "of squared matrices");

      NoTF32Guard disable_tf32;

      if (a.size(-1) == 1) {
        return a.exp();
      }

      return mexp(a);
        */
}

pub fn matrix_exp_backward(
        self_: &Tensor,
        grad:  &Tensor) -> Tensor {
    
    todo!();
        /*
            NoTF32Guard disable_tf32;
      return backward_analytic_function_of_a_matrix(
        self, grad,
        [](const Tensor& a) {
          return a.matrix_exp();
        }
      );
        */
}

pub fn frobenius_norm_a(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return norm(self);
        */
}

pub fn frobenius_norm_b(
        self_:   &Tensor,
        dim:     &[i32],
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            // NOTE: As frobenius_norm_out is currently implemented, it will always produce a
      //    strided tensor result, even if the input is sparse.
      auto options = self.options().layout(Layout::Strided).dtype(toValueType(self.scalar_type()));
      Tensor result = empty({0}, options);
      return native::frobenius_norm_out(self, dim, keepdim, result);
        */
}

pub fn frobenius_norm_out<'a>(
        self_:   &Tensor,
        dim:     &[i32],
        keepdim: bool,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          dim.size() <= 2,
          "Expected at most 2 dimensions, but got ",
          dim.size(),
          " dimensions instead.");
      Tensor result_;
      if (dim.size() == 1 || dim.size() == 0) {
        result_ = norm(self, 2, dim, keepdim);
      } else {
        auto dim_ = dim.vec();
        maybe_wrap_dims(dim_, self.dim());
        TORCH_CHECK(dim_[0] != dim_[1], "Expected dims to be different, got ", dim, " instead");
        if (self.is_complex()){
          result_ = sqrt(sum(real(self.conj() * self), dim_, keepdim));
        } else {
          result_ = sqrt(sum((self * self), dim_, keepdim));
        }
      }
      // NOTE: It would be better to avoid resize and copy by using norm_out and sqrt_out above.
      //    However, norm_out and sqrt_out do not support automatic differentiation.
      //    More details here: https://github.com/pytorch/pytorch/pull/44095#discussion_r486673947
      native::resize_output(result, result_.sizes());
      result.copy_(result_);
      return result;
        */
}


pub fn nuclear_norm_a(
        self_:   &Tensor,
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          self.dim() == 2,
          "Expected a tensor with 2 dimensions, but got a tensor with ",
          self.dim(), " dimension", self.dim()==1 ? "" : "s", " instead.");
      return native::nuclear_norm(self, IntArrayRef({0, 1}), keepdim);
        */
}

pub fn nuclear_norm_out_a<'a>(
        self_:   &Tensor,
        keepdim: bool,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          self.dim() == 2,
          "Expected a tensor with 2 dimensions, but got a tensor with ",
          self.dim(), " dimension", self.dim()==1 ? "" : "s", " instead.");
      return native::nuclear_norm_out(self, IntArrayRef({0, 1}), keepdim, result);
        */
}

pub fn nuclear_norm_b(
        self_:   &Tensor,
        dim:     &[i32],
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options().dtype(toValueType(self.scalar_type())));
      return native::nuclear_norm_out(self, dim, keepdim, result);
        */
}

pub fn nuclear_norm_out_b<'a>(
        self_:   &Tensor,
        dim:     &[i32],
        keepdim: bool,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(dim.size() == 2, "nuclear norm requires a 'dim' argument of size 2");
      auto dim_ = dim.vec();
      maybe_wrap_dims(dim_, self.dim());

      auto permutation = create_dim_backshift_permutation(dim_[0], dim_[1], self.dim());
      Tensor p = self.permute(permutation);
      // NOTE: U and V are computed only if gradmode is enabled, since the backward for nuclear
      //       norm uses svd_backward, which requires them.
      Tensor result_ = sum(get<1>(svd(p, /*some=*/true,
                      /*compute_uv=*/GradMode::is_enabled() && self.requires_grad())), -1, keepdim);
      if (keepdim) {
        result_.unsqueeze_(-1);
        auto permutation_reverse = create_reverse_permutation(permutation);
        result_ = result_.permute(permutation_reverse);
      }
      native::resize_output(result, result_.sizes());
      result.copy_(result_);
      return result;
        */
}

/**
  | Creates a vector of length ndim with
  | values equal to its indices (e.g. [0,
  | 1, 2, ..., ndim-1])
  |
  */
pub fn make_dim_list(ndim: i64) -> Vec<i64> {
    
    todo!();
        /*
            vector<i64> dim_list(ndim);
      for (i64 ind = 0; ind < ndim; ind++) {
        dim_list[ind] = ind;
      }
      return dim_list;
        */
}

/**
  | Checks for valid arguments to linalg_norm
  | when type(ord) == str
  |
  */
pub fn check_str_ord_valid(
        str_ord: StringView,
        opt_dim: Option<&[i32]>,
        ndim:    i64)  {
    
    todo!();
        /*
            TORCH_CHECK((str_ord == "nuc") || (str_ord == "fro"), "Invalid norm order: ", str_ord);
      bool dims_valid = (ndim == 2 && !opt_dim.has_value()) || (opt_dim.has_value() && opt_dim.value().size() == 2);
      TORCH_CHECK(dims_valid, "order \"", str_ord,
        "\" can only be used if either len(dim) == 2 or (self.dim() == 2 and dim is None)");
        */
}

/**
  | Performs vector norm for ord = +/-infinity, and
  | the second dimension reduction for matrix
  | norms.
  |
  */
pub fn norm_min_max(
        self_:   &mut Tensor,
        ord:     f64,
        dim:     i64,
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            Tensor result;
      if (self.numel() == 0 && self.sizes()[dim] > 0) {
        // This special case is needed in matrix norm for tensors with 3 or more dims,
        // or in vector norm for order inf and -inf for tesnsors with 2 or more dims.
        // When the sizes of the dims to be reduced are greater than 0 but another dim
        // in the tensor is size 0 (thus numel == 0), we must either flatten or resize
        // the second reduction dim to 1, to avoid calling min/max, which would throw
        // an error.
        if (self.sizes()[dim] != 1) {
          auto new_sizes = self.sizes().vec();
          new_sizes[dim] = 1;
          self.resize_(new_sizes);
        }
        result = keepdim ? self : self.flatten(dim);
      } else {
        if (ord > 0) {
          result = get<0>(self.max(dim, keepdim));
        } else {
          result = get<0>(self.min(dim, keepdim));
        }
      }
      return result;
        */
}

/// Performs matrix norm
pub fn linalg_norm_matrix_out<'a>(
        result:    &mut Tensor,
        self_:     &Tensor,
        opt_ord:   &Option<Scalar>,
        dim:       &[i32],
        keepdim:   bool,
        opt_dtype: Option<ScalarType>) -> &'a mut Tensor {
    
    todo!();
        /*
            Tensor result_;
      auto ord = opt_ord.value_or(2.0).toDouble();
      TORCH_CHECK(self.layout() == Layout::Strided,
                  "matrix norm only supports strided layout, got: ", self.layout());

      TORCH_CHECK(dim.size() == 2, "_linalg_norm_matrix: 'dim' must either specify 2 dimensions. ",
        "Got 'dim' specifying ", dim.size(), " dims");
      auto dim_ = dim.vec();
      maybe_wrap_dims(dim_, self.dim());
      TORCH_CHECK(dim_[0] != dim_[1],
        "Expected dims to be different, got (", dim[0], ", ", dim[1], ") instead");

      ScalarType scalarType = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
      TORCH_CHECK(
          isFloatingType(scalarType) || isComplexType(scalarType),
          "Can only calculate the mean of floating and complex types. Got ",
          toString(scalarType), " instead.");

      Tensor self_;
      if (opt_dtype.has_value()) {
        self_ = self.to(scalarType);
      } else {
        self_ = self;
      }

      if (abs(ord) == 2) {
        // Need to shift the reduction dims to the back, because svd will only operate on
        // the last 2 dimensions
        auto permutation = create_dim_backshift_permutation(dim_[0], dim_[1], self.dim());
        auto permutation_reverse = create_reverse_permutation(permutation);

        result_ = linalg_svdvals(self_.permute(permutation));
        result_ = _norm_min_max(result_, ord, result_.dim() - 1, keepdim);

        if (keepdim) {
          result_.unsqueeze_(-1);
          result_ = result_.permute(permutation_reverse);
        }
      } else {
        // abs(p) == infinity and abs(p) == 1 will perform identical reductions, except
        // that the order of the two dims is swapped. So we can swap the dims if
        // abs(p) == infinity to simplify the rest of the operation's logic.
        if (abs(ord) == INFINITY) {
          swap(dim_[0], dim_[1]);
        }
        // If the dim of the second reduction is greater than that of the first reduction
        // and we are not keeping the dims, then the fact that the output of the first
        // reduction will have one fewer dimension means that the second reduction dim
        // will be off by one, so we need to correct that.
        if ((dim_[1] > dim_[0]) && !keepdim) {
          dim_[1]--;
        }
        if (abs(ord) == 1 || abs(ord) == INFINITY) {
          result_ = self_.abs().sum(dim_[0], keepdim);
          result_ = _norm_min_max(result_, ord, dim_[1], keepdim);
        } else {
          TORCH_CHECK(false, "Order ", ord, " not supported for matrix norm");
        }
      }
      native::resize_output(result, result_.sizes());
      result.copy_(result_);
      return result;
        */
}

pub fn linalg_norm_out_impl<'a>(
    result:      &mut Tensor,
    self_:       &Tensor,
    opt_num_ord: &Option<Scalar>,
    opt_str_ord: Option<StringView>,
    opt_dim:     Option<&[i32]>,
    keepdim:     bool,
    opt_dtype:   Option<ScalarType>) -> &'a mut Tensor {

    todo!();
        /*
            // Callers must give the ord argument as either a number, a string, or neither.
      // Since the user-facing API has no direct control over how this function is called, this is an internal assert.
      TORCH_INTERNAL_ASSERT(!(opt_num_ord.has_value() && opt_str_ord.has_value()));
      if (opt_dtype.has_value()) {
        auto dtype = opt_dtype.value();
        TORCH_CHECK(dtype == result.scalar_type(), "provided dtype must match dtype of result, but got",
          "dtype = ", dtype, ", out.dtype = ", result.scalar_type());
      }
      i64 ndim = self.dim();
      if (opt_str_ord.has_value()) {
        // 'ord' is string
        auto str_ord = opt_str_ord.value();
        check_str_ord_valid(str_ord, opt_dim, ndim);
        Tensor self_ = opt_dtype.has_value() ? self.to(opt_dtype.value()) : self;
        if (str_ord == "fro") {
          frobenius_norm_out(result, self_, opt_dim.value_or(IntArrayRef({0, 1})), keepdim);
        } else if (str_ord == "nuc") {
          if (opt_dim.has_value()) {
            nuclear_norm_out(result, self_, opt_dim.value(), keepdim);
          } else {
            nuclear_norm_out(result, self_, keepdim);
          }
        }
      } else {
        // 'ord' is int or None
        vector<i64> dim_ = opt_dim.has_value() ? opt_dim.value().vec() : make_dim_list(ndim);
        if (!opt_num_ord.has_value() || dim_.size() == 1) {
          Tensor result_ = linalg_vector_norm(
              self, opt_num_ord.value_or(2), opt_dim, keepdim, opt_dtype);
          // TODO: Resize and copy should be avoided with
          //       https://github.com/pytorch/pytorch/issues/52712
          native::resize_output(result, result_.sizes());
          result.copy_(result_);
        } else if (dim_.size() == 2) {
          _linalg_norm_matrix_out(result, self, opt_num_ord.value(), dim_, keepdim, opt_dtype);
        } else {
          TORCH_CHECK(false, "'dim' must specify 1 or 2 dimensions when order is numerical and input is "
            "not 1-D or 2-D");
        }
      }
      return result;
        */
}

pub fn linalg_vector_norm_impl<'a>(
        self_:      &Tensor,
        scalar_ord: &Scalar,
        opt_dim:    Option<&[i32]>,
        keepdim:    bool,
        opt_dtype:  Option<ScalarType>,
        result:     &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            // Casting a large integer to a double will introduce some error, but for
      // practical purposes, it won't matter since a large order will usually
      // give an infinite result
      auto ord = scalar_ord.toDouble();

      TORCH_CHECK(self.device().type() == DeviceType_CPU || self.device().type() == DeviceType::Cuda,
                  "linalg.vector_norm only supports CPU and CUDA device types, but got: ",
                  self.device().type());
      TORCH_CHECK(self.layout() == Layout::Strided,
                  "linalg.vector_norm only supports strided layout, but got: ", self.layout());

      if (opt_dtype.has_value() && isComplexType(self.scalar_type())) {
        TORCH_CHECK(isComplexType(opt_dtype.value()),
          "linalg.vector_norm expected complex 'dtype', since input is complex, ",
          "but got ", opt_dtype.value());
      }

      ScalarType in_dtype = opt_dtype.value_or(self.scalar_type());
      TORCH_CHECK(
          isFloatingType(in_dtype) || isComplexType(in_dtype),
          "linalg.vector_norm only supports floating point and complex dtypes, but got: ",
          toString(in_dtype));

      IntArrayRef dim = opt_dim.value_or(IntArrayRef{});

      if (self.numel() == 0) {
        // TODO: The question about how to handle negative orders when the input
        // is empty has not been settled yet. For now, we raise an error. Issue:
        // https://github.com/pytorch/pytorch/issues/52783
        TORCH_CHECK(ord >= 0,
          "linalg.vector_norm of negative order cannot be performed on an empty tensor");

        // For NumPy compatibility, we can only perform order infinity reduction
        // (max/min) on a tensor with zero elements if the dimensions to reduce are
        // nonzero. Otherwise, throw an error.
        if (ord == INFINITY) {
          bool has_identity = true;

          if (dim.size() == 0) {
            has_identity = false;
          } else {
            for (i64 dim_num : dim) {
              if (self.size(dim_num) == 0) {
                has_identity = false;
                break;
              }
            }
          }
          TORCH_CHECK(has_identity,
            "linalg.vector_norm cannot compute the infinity norm on an empty ",
            "dimension because the operation does not have an identity");
        }
      }
      Tensor self_;
      if (self.device().type() == kCPU && isComplexType(self.scalar_type()) && abs(ord) == INFINITY) {
        // TODO: This abs() call is used so that the abs() call in the
        // backward function produces an identical result for complex inputs.
        // However, it would be ideal if we could incorporate this into
        // linalg_vector_norm_stub. See issue:
        // https://github.com/pytorch/pytorch/issues/52648
        self_ = self.to(in_dtype).abs();
        in_dtype = toValueType(in_dtype);
      } else {
        self_ = self;
      }
      ScalarType out_dtype = opt_dtype.value_or(toValueType(self.scalar_type()));
      TORCH_CHECK(!result.defined() || out_dtype == result.scalar_type(),
        "linalg.vector_norm expected out tensor dtype ", out_dtype,
        " but got: ", result.scalar_type());
      // omit in_dtype in the following call, to avoid make_reduction explicitly casting input to out_dtype
      auto iter = isComplexType(self.scalar_type()) ?
          make_reduction("vector_norm", result, self_, dim, keepdim, in_dtype, out_dtype) :
          make_reduction("vector_norm", result, self_, dim, keepdim, out_dtype);

      linalg_vector_norm_stub(iter.device_type(), iter, ord);
      return result;
        */
}

pub fn linalg_vector_norm(
        self_:     &Tensor,
        ord:       &Scalar,
        opt_dim:   Option<&[i32]>,
        keepdim:   bool,
        opt_dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            ScalarType out_dtype = opt_dtype.value_or(toValueType(self.scalar_type()));
      Tensor result = create_reduction_result(self, opt_dim.value_or(IntArrayRef{}), keepdim, out_dtype);
      return native::linalg_vector_norm_impl(self, ord, opt_dim, keepdim, opt_dtype, result);
        */
}

pub fn linalg_vector_norm_out<'a>(
        self_:     &Tensor,
        ord:       &Scalar,
        opt_dim:   Option<&[i32]>,
        keepdim:   bool,
        opt_dtype: Option<ScalarType>,
        result:    &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::linalg_vector_norm_impl(self, ord, opt_dim, keepdim, opt_dtype, result);
        */
}

/**
  | Only performs checks not performed
  | by linalg.norm
  |
  */
pub fn check_linalg_matrix_norm_args(
    self_: &Tensor,
    dim:   &[i32],
    dtype: Option<ScalarType>)  {
    
    todo!();
        /*
            TORCH_CHECK(
          self.ndimension() >= 2,
          "linalg.matrix_norm(): input tensor must be a matrix or batch of matrices");
      ScalarType in_dtype = dtype.value_or(self.scalar_type());
      TORCH_CHECK(
          in_dtype == kFloat || in_dtype == kDouble || in_dtype == kComplexFloat ||
              in_dtype == kComplexDouble,
          "linalg.matrix_norm(): only supports the float, double, cfloat and cdouble dtypes, but got: ",
          toString(in_dtype));
      TORCH_CHECK(
          dim.size() == 2, "linalg.matrix_norm(): dim must be a 2-tuple of ints");
        */
}

pub fn linalg_matrix_norm_a(
        self_:   &Tensor,
        ord:     &Scalar,
        dim:     &[i32],
        keepdim: bool,
        dtype:   Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            check_linalg_matrix_norm_args(self, dim, dtype);
      return native::linalg_norm(self, ord, dim, keepdim, dtype);
        */
}

pub fn linalg_matrix_norm_out_a<'a>(
        self_:   &Tensor,
        ord:     &Scalar,
        dim:     &[i32],
        keepdim: bool,
        dtype:   Option<ScalarType>,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            check_linalg_matrix_norm_args(self, dim, dtype);
      return native::linalg_norm_out(self, ord, dim, keepdim, dtype, result);
        */
}

pub fn linalg_matrix_norm_b(
    self_:   &Tensor,
    ord:     StringView,
    dim:     &[i32],
    keepdim: bool,
    dtype:   Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            check_linalg_matrix_norm_args(self, dim, dtype);
      return native::linalg_norm(self, ord, dim, keepdim, dtype);
        */
}

pub fn linalg_matrix_norm_out_b<'a>(
        self_:   &Tensor,
        ord:     StringView,
        dim:     &[i32],
        keepdim: bool,
        dtype:   Option<ScalarType>,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            check_linalg_matrix_norm_args(self, dim, dtype);
      return native::linalg_norm_out(self, ord, dim, keepdim, dtype, result);
        */
}

/// Numerical or None norms
///
pub fn linalg_norm_with_numerical_or_none_norms(
        self_:     &Tensor,
        opt_ord:   &Option<Scalar>,
        opt_dim:   Option<&[i32]>,
        keepdim:   bool,
        opt_dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            auto options = TensorOptions().dtype(opt_dtype.has_value() ? opt_dtype.value() : toValueType(self.scalar_type())).device(self.device());
      Tensor result = empty({0}, options);
      return native::linalg_norm_out(
          self, opt_ord, opt_dim, keepdim, opt_dtype, result);
        */
}

/// Frobenius and nuclear norms
///
pub fn linalg_norm_with_frobenius_and_nuclear_norms(
    self_:     &Tensor,
    ord:       StringView,
    opt_dim:   Option<&[i32]>,
    keepdim:   bool,
    opt_dtype: Option<ScalarType>) -> Tensor {

    todo!();
        /*
            auto options = TensorOptions().dtype(opt_dtype.has_value() ? opt_dtype.value() : toValueType(self.scalar_type())).device(self.device());
      Tensor result = empty({0}, options);
      return native::linalg_norm_out(
          self, ord, opt_dim, keepdim, opt_dtype, result);
        */
}

/// Numerical or None norms
///
pub fn linalg_norm_out_with_numerical_or_none_norms<'a>(
        self_:     &Tensor,
        opt_ord:   &Option<Scalar>,
        opt_dim:   Option<&[i32]>,
        keepdim:   bool,
        opt_dtype: Option<ScalarType>,
        result:    &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return linalg_norm_out_impl(result, self, opt_ord, nullopt, opt_dim, keepdim, opt_dtype);
        */
}

/// Frobenius and nuclear norms
///
pub fn linalg_norm_out_with_frobenius_and_nuclear_norms<'a>(
    self_:     &Tensor,
    ord:       StringView,
    opt_dim:   Option<&[i32]>,
    keepdim:   bool,
    opt_dtype: Option<ScalarType>,
    result:    &mut Tensor) -> &'a mut Tensor {

    todo!();
        /*
            return linalg_norm_out_impl(result, self, nullopt, ord, opt_dim, keepdim, opt_dtype);
        */
}

/**
  | This function helps to dispatch norm
  | computations depending on 'ord' of variant
  | type
  |
  */
pub fn linalg_cond_helper(
        self_:       &Tensor,
        ord_variant: Variant<Scalar,StringView>) -> Tensor {
    
    todo!();
        /*
            Tensor inverse, info;
      tie(inverse, info) = linalg_inv_ex(self);
      info.unsqueeze_(-1).unsqueeze_(-1);
      inverse.masked_fill_(info > 0, INFINITY);

      return visit([&](auto&& ord) {
        Tensor norm_self = linalg_matrix_norm(self, ord);
        Tensor norm_inverse = linalg_matrix_norm(inverse, ord);
        Tensor result = norm_self * norm_inverse;
        // fix multiplication of zero and infinity for NumPy compatibility
        result.nan_to_num_(INFINITY, INFINITY, -INFINITY);
        return result;
      }, ord_variant);
        */
}

/**
  | Return zero for each matrix in the batch
  |
  */
pub fn linalg_cond_empty_matrix(
        self_: &Tensor,
        dtype: ScalarType) -> Tensor {
    
    todo!();
        /*
            auto result_shape = IntArrayRef(self.sizes().cbegin(), self.sizes().cend()-2);
      TensorOptions options = self.options().dtype(toValueType(self.scalar_type()));
      return zeros(result_shape, options);
        */
}

pub fn linalg_cond_check_ord(ord_variant: Variant<Scalar,StringView>)  {
    
    todo!();
        /*
            if (ord_variant.index() == 0) {
        Scalar* ord = get_if<Scalar>(&ord_variant);
        double abs_ord = abs(ord->toDouble());
        TORCH_CHECK(abs_ord == 2.0 || abs_ord == 1.0 || abs_ord == INFINITY,
          "linalg_cond got an invalid norm type: ", ord->toDouble());
      } else if (ord_variant.index() == 1) {
        string_view* ord = get_if<string_view>(&ord_variant);
        TORCH_CHECK(*ord == "fro" || *ord == "nuc",
          "linalg_cond got an invalid norm type: ", *ord);
      } else {
        TORCH_CHECK(false,
          "linalg_cond: something went wrong while checking the norm type");
      }
        */
}

/// Numerical or None norms
pub fn linalg_cond_with_numerical_or_none_norms(
        self_:   &Tensor,
        opt_ord: &Option<Scalar>) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.dim() >= 2, "linalg_cond only supports matrices or batches of matrices, but got a tensor with ",
        self.dim(), " dimensions.");

      // The default case is using 2-norm
      Scalar ord = opt_ord.has_value() ? opt_ord.value() : 2;

      variant<Scalar, string_view> ord_variant = ord;
      _linalg_cond_check_ord(ord_variant);

      // NumPy doesn't define the condition number for 0x0 matrices, we return 0.0 for such input
      if (self.numel() == 0) {
        auto real_dtype = toValueType(typeMetaToScalarType(self.dtype()));
        return _linalg_cond_empty_matrix(self, real_dtype);
      }

      // If ord == None or ord == ±2
      if (abs(ord.toDouble()) == 2.0) {
        auto singular_values = get<1>(svd(self));
        // singular values are sorted in descending order
        auto s_max = narrow(singular_values, /*dim=*/-1, /*start=*/0, /*length=*/1);
        auto s_min = narrow(singular_values, /*dim=*/-1, /*start=*/-1, /*length=*/1);
        Tensor result;
        if (ord.toDouble() == -2.0) {
          result = s_min / s_max;
        } else {
          result = s_max / s_min;
        }
        // squeeze the result for NumPy compatibility
        return result.squeeze(-1);
      }

      // ord == ±1 ord == ±inf
      // since inverse is used in the implementation, self has to be a tensor consisting of square matrices
      // the same check as squareCheckInputs(self) but with a slightly more informative error message
      TORCH_CHECK(self.size(-1) == self.size(-2),
                  "linalg_cond with ±1 or ±inf norm types only supports square matrices or batches of square matrices "
                  "but got ", self.size(-1), " by ", self.size(-2), " matrices");

      return _linalg_cond_helper(self, ord_variant);
        */
}

pub fn linalg_cond_out_a<'a>(
    self_:   &Tensor,
    opt_ord: &Option<Scalar>,
    result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            checkSameDevice("linalg_cond", result, self);
      ScalarType real_dtype = toValueType(self.scalar_type());
      checkLinalgCompatibleDtype("linalg_cond", result.scalar_type(), real_dtype);

      Tensor result_tmp = linalg_cond(self, opt_ord);
      native::resize_output(result, result_tmp.sizes());
      result.copy_(result_tmp);
      return result;
        */
}

/// Frobenius or nuclear norms
///
pub fn linalg_cond_with_frobenius_or_nuclear_norms(
        self_: &Tensor,
        ord:   StringView) -> Tensor {
    
    todo!();
        /*
            // the same checks as squareCheckInputs(self) but with a slightly more informative error message
      TORCH_CHECK(self.dim() >= 2, "linalg_cond only supports matrices or batches of matrices, but got a tensor with ",
        self.dim(), " dimensions.");
      TORCH_CHECK(self.size(-1) == self.size(-2),
                  "linalg_cond with frobenius or nuclear norm types only supports square matrices or batches of square matrices "
                  "but got ", self.size(-1), " by ", self.size(-2), " matrices");

      variant<Scalar, string_view> ord_variant = ord;
      _linalg_cond_check_ord(ord_variant);

      // NumPy doesn't define the condition number for 0x0 matrices, we return 0.0 for such input
      if (self.numel() == 0) {
        return _linalg_cond_empty_matrix(self, self.scalar_type());
      }

      if (ord == "nuc") {
        // calling matrix_norm with "nuc" on inputs with infinities raises an error
        // therefore we use the mathematical definition of nuclear norm directly
        // instead of going through the matrix_norm
        auto singular_values = linalg_svdvals(self);
        return singular_values.sum(-1) * (singular_values.reciprocal().sum(-1));
      }

      return _linalg_cond_helper(self, ord_variant);
        */
}

/**
  | TODO: implement _out variant avoiding
  | copy and using already allocated storage
  | directly
  |
  */
pub fn linalg_cond_out_b<'a>(
    self_:  &Tensor,
    ord:    StringView,
    result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            checkSameDevice("linalg_cond", result, self);
      ScalarType real_dtype = toValueType(self.scalar_type());
      checkLinalgCompatibleDtype("linalg_cond", result.scalar_type(), real_dtype);

      Tensor result_tmp = linalg_cond(self, ord);
      native::resize_output(result, result_tmp.sizes());
      result.copy_(result_tmp);
      return result;
        */
}

pub fn linalg_tensorinv(
        self_: &Tensor,
        ind:   i64) -> Tensor {
    
    todo!();
        /*
            /*
      The idea is to reduce the problem to 2D square matrix inversion.
      Step 1. Calculate the shape of the result and the shape of the intermediate 2D matrix.
      Step 2. Reshape `self` to 2D matrix.
      Step 3. Invert the 2D matrix self.to_2D()
              There is no quick way to find out whether the matrix is invertible,
              so at this stage an error from inverse can be thrown.
              Note that for CUDA this causes cross-device memory synchronization that can be slow.
      Step 4. reshape the result.
      */
      TORCH_CHECK(ind > 0, "Expected a strictly positive integer for 'ind', but got ", ind);

      // self[ind:]
      vector<i64> shape_ind_end = self.sizes().slice(ind).vec();
      // self[:ind]
      vector<i64> shape_start_ind = self.sizes().slice(0, ind).vec();

      i64 prod_ind_end = multiply_integers(shape_ind_end.cbegin(), shape_ind_end.cend());
      i64 prod_start_ind = multiply_integers(shape_start_ind.cbegin(), shape_start_ind.cend());

      // Check whether the self tensor can be reshaped to the 2D square matrix
      TORCH_CHECK(prod_ind_end == prod_start_ind,
        "Expected self to satisfy the requirement prod(self.shape[ind:]) == prod(self.shape[:ind]), but got ",
        prod_ind_end, " != ", prod_start_ind);

      // Concatenate shape_ind_end and shape_start_ind to form the shape of the result
      // self[ind:] + self[:ind]
      shape_ind_end.insert(shape_ind_end.cend(), shape_start_ind.cbegin(), shape_start_ind.cend());

      // If the reshaped self is not invertible catch this error
      Tensor result;
      try {
        result = inverse(self.reshape({prod_ind_end, prod_ind_end}));
      } catch (...) {
        TORCH_CHECK(false, "Failed to invert the input tensor, because it is singular.");
      }

      return result.reshape(shape_ind_end);
        */
}

/**
  | TODO: implement _out variant avoiding
  | copy and using already allocated storage
  | directly
  |
  */
pub fn linalg_tensorinv_out<'a>(
        self_:  &Tensor,
        ind:    i64,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            checkSameDevice("tensorinv", result, self);
      checkLinalgCompatibleDtype("tensorinv", result, self);

      Tensor result_tmp = linalg_tensorinv(self, ind);
      native::resize_output(result, result_tmp.sizes());
      result.copy_(result_tmp);
      return result;
        */
}

pub fn linalg_tensorsolve(
        self_: &Tensor,
        other: &Tensor,
        dims:  Option<&[i32]>) -> Tensor {
    
    todo!();
        /*
            /*
      The idea is to reduce the problem to 2D matrix solve.
      Step 1. (optional) `self` is permuted with `dims` such that dimensions from `dims` are moved to the right.
      For example, if we have 4D input with the shape (1, 2, 3, 4) and dims=(0, 2),
      then the result of permutation would have the shape (2, 4, 1, 3).
      Step 2. reshape `self` to 2D matrix.
      Step 3. solve the matrix equation self.to_2D() @ result = other.to_1D()
      Step 4. reshape the result.
      */
      i64 ndim = self.dim();
      Tensor self_ = self;

      // move dimensions of `self_` from `dims` to the end
      if (dims.has_value()) {
        DimVector dest_axes(dims.value().size());
        iota(dest_axes.begin(), dest_axes.end(), ndim - dest_axes.size());
        self_ = movedim(self_, dims.value(), dest_axes);
      }

      // result_shape is self_.sizes[-(an-other.dim):]
      vector<i64> result_shape = self_.sizes().slice(other.dim(), ndim - other.dim()).vec();

      i64 result_product = multiply_integers(result_shape.begin(), result_shape.end());
      i64 other_product = multiply_integers(other.sizes().begin(), other.sizes().end());

      // Check whether the self tensor can be reshaped to the 2D square matrix
      TORCH_CHECK(result_product == other_product,
        "Expected self to satisfy the requirement prod(self.shape[other.ndim:]) == prod(self.shape[:other.ndim]), but got ",
        result_product, " != ", other_product);

      self_ = self_.reshape({result_product, result_product});

      // normally `other` would be flattened by linalg_solve expects 2D input
      Tensor result = linalg_solve(self_, other.flatten());
      return result.reshape(result_shape);
        */
}

pub fn linalg_tensorsolve_out<'a>(
        self_:  &Tensor,
        other:  &Tensor,
        dims:   Option<&[i32]>,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            checkSameDevice("tensorsolve", result, self);
      checkLinalgCompatibleDtype("tensorsolve", result, self);

      Tensor result_tmp = linalg_tensorsolve(self, other, dims);
      native::resize_output(result, result_tmp.sizes());
      result.copy_(result_tmp);
      return result;
        */
}

pub struct KronImpl {
    maxdim:         i64,
    self_view:      Tensor,
    other_view:     Tensor,
    result_reshape: SmallVector<i64,10>,
    a_reshape:      SmallVector<i64,10>,
    b_reshape:      SmallVector<i64,10>,
}

impl KronImpl {
    
    pub fn new<'a>(
        self_: &Tensor,
        other: &Tensor) -> Self {
    
        todo!();
        /*


            maxdim = max(self.dim(), other.dim());
          i64 pad_self = maxdim - self.dim();
          i64 pad_other = maxdim - other.dim();
          a_reshape = SmallVector<i64, 10>(2 * maxdim);
          b_reshape = SmallVector<i64, 10>(2 * maxdim);
          result_reshape = SmallVector<i64, 10>(maxdim);
          for (i64 i = 0; i < maxdim; i++) {
            a_reshape[2 * i] = (i >= pad_self ? self.sizes()[i - pad_self] : 1);
            a_reshape[2 * i + 1] = 1;
            b_reshape[2 * i] = 1;
            b_reshape[2 * i + 1] = (i >= pad_other ? other.sizes()[i - pad_other] : 1);
            result_reshape[i] = a_reshape[2 * i] * b_reshape[2 * i + 1];
          }
          self_view = _unsafe_view(self, a_reshape);
          other_view = _unsafe_view(other, b_reshape);
        */
    }
    
    pub fn kron_out<'a>(&self, result: &mut Tensor) -> &'a mut Tensor {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(result.defined(), "Cannot call kron_out with an undefined result tensor as the out argument. Please allocate a Tensor before calling kron_out with it.");

          SmallVector<i64, 10> mul_shape(2 * maxdim);
          for (i64 i = 0; i < maxdim; i++) {
            mul_shape[2 * i] = a_reshape[2 * i];
            mul_shape[2 * i + 1] = b_reshape[2 * i + 1];
          }
          native::resize_output(result, result_reshape);
          auto result_mul = _unsafe_view(result, mul_shape);
          mul_out(result_mul, self_view, other_view);

          return result;
        */
    }
    
    pub fn kron(&self) -> Tensor {
        
        todo!();
        /*
            return _unsafe_view(mul(self_view, other_view), result_reshape);
        */
    }
}

define_dispatch!{unpack_pivots_stub}

pub fn lu_unpack(
    lu_data:       &Tensor,
    lu_pivots:     &Tensor,
    unpack_data:   bool,
    unpack_pivots: bool) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            TORCH_CHECK(LU_pivots.is_contiguous() && (LU_pivots.scalar_type() == kInt),
          "lu_unpack: LU_pivots is expected to be a contiguous tensor of torch.int32 dtype."
          "Note: this function is intended to be used with the output produced by torch{.linalg}.lu");

      // trivial case
      if (!unpack_data && !unpack_pivots) {
        return make_tuple(Tensor(), Tensor(), Tensor());
      }

      Tensor L, U;
      // In the generalized LU factorization, the following shape relations hold:
      // A.shape[-2:] == (m, n),
      // P.shape[-2:] == (m, m),
      // U.shape[-2:] == (m, k),
      // L.shape[-2:] == (k, n),
      // where k = min(m, n)
      i64 m = LU_data.size(-2);
      i64 n = LU_data.size(-1);
      i64 k = min(m, n);

      if (unpack_data) {
        U = LU_data.triu();
        if (m != k) {
          U = U.narrow(-2, 0, k);
        }

        L = LU_data.tril();
        if (k != n) {
          L = L.narrow(-1, 0, k);
        }
        L.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(1);
      }

      if (!unpack_pivots) {
        return make_tuple(Tensor(), L, U);
      }

      auto unpacked_pivots_sizes = LU_pivots.sizes().vec();
      unpacked_pivots_sizes[LU_pivots.dim() - 1] = m;
      auto unpacked_pivots = empty(
        unpacked_pivots_sizes,
        LU_pivots.options().memory_format(MemoryFormat::Contiguous)
      );

      // Fill `unpacked_pivots` with identity permutation
      auto id_perm = arange(m, LU_pivots.options());
      unpacked_pivots.copy_(id_perm);

      // WARNING: we assume that unchanged LAPACK pivots are provided.
      // Since LAPACK relies on the FORTRAN's 1-based indexing,
      // we subtract 1 to convert the pivots to the C-style 0-based indexing.
      // This behaviour could change in the future.
      auto LU_pivots_zero_idx = LU_pivots - 1;

      auto iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .declare_static_shape(LU_pivots.sizes(), /*squash_dim=*/LU_pivots.dim() - 1)
        .add_output(unpacked_pivots)
        .add_input(LU_pivots_zero_idx)
        .build();
      // }

      unpack_pivots_stub(
        LU_pivots.device().type(),
        iter,
        LU_pivots.size(-1)
      );

      // The permutation matrix is converted to LU_data.dtype
      // because `matmul` does not work with integer matrices.
      unpacked_pivots_sizes.push_back(m);
      auto permutation_matrix = zeros(
        unpacked_pivots_sizes,
        LU_data.options().memory_format(MemoryFormat::Contiguous)
      );

      // now that we know the final permutation,
      // scatter 1s at proper locations.
      permutation_matrix.scatter_(
        -2,
        unpacked_pivots.unsqueeze(-2).to(kLong),
        ones({1}, permutation_matrix.options()).expand(permutation_matrix.sizes())
      );

      return make_tuple(permutation_matrix, L, U);
        */
}

pub type TupleTensorRefs3 = (&mut Tensor,&mut Tensor,&mut Tensor);

pub fn lu_unpack_out(
        lu_data:       &Tensor,
        lu_pivots:     &Tensor,
        unpack_data:   bool,
        unpack_pivots: bool,
        P:             &mut Tensor,
        L:             &mut Tensor,
        U:             &mut Tensor) -> TupleTensorRefs3 {
    
    todo!();
        /*
            Tensor P_tmp, L_tmp, U_tmp;
      tie(P_tmp, L_tmp, U_tmp) = lu_unpack(LU_data, LU_pivots, unpack_data, unpack_pivots);

      if (unpack_pivots) {
        checkSameDevice("lu_unpack", P, LU_data, "P");
        // Note that lu_unpack returns P such that P.dtype == LU_data.dtype,
        // because otherwise we cannot use P in matric products (no int -> float promotion)
        checkLinalgCompatibleDtype("lu_unpack", P, LU_data, "L");

        native::resize_output(P, P_tmp.sizes());
        P.copy_(P_tmp);
      }

      if (unpack_data) {
        checkSameDevice("lu_unpack", L, LU_data, "L");
        checkSameDevice("lu_unpack", U, LU_data, "U");
        checkLinalgCompatibleDtype("lu_unpack", L, LU_data, "L");
        checkLinalgCompatibleDtype("lu_unpack", U, LU_data, "U");

        native::resize_output(L, L_tmp.sizes());
        native::resize_output(U, U_tmp.sizes());
        L.copy_(L_tmp);
        U.copy_(U_tmp);
      }

      return TupleTensorRefs3(P, L, U);
        */
}

/**
  | Calculates the Kronecker product between
  | two Tensors.
  |
  */
pub fn kron_out<'a>(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return KronImpl(self, other).kron_out(result);
        */
}

pub fn kron(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return KronImpl(self, other).kron();
        */
}
