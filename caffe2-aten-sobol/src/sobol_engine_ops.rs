crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/SobolEngineOps.cpp]

/**
  | This is the core function to draw samples from
  | a `SobolEngine` given its state variables
  | (`sobolstate` and `quasi`).
  |
  | `dimension` can be inferred from `sobolstate`,
  | but choosing to pass it explicitly to avoid an
  | extra operation to obtain the size of the
  | first dimension of `sobolstate`.
  */
pub fn sobol_engine_draw(
        quasi:         &Tensor,
        n:             i64,
        sobolstate:    &Tensor,
        dimension:     i64,
        num_generated: i64,
        dtype:         Option<ScalarType>) -> (Tensor,Tensor) {
    
    todo!();
        /*
            TORCH_CHECK(sobolstate.dtype() == kLong,
               "sobolstate needs to be of type ", kLong);
      TORCH_CHECK(quasi.dtype() == kLong,
               "quasi needs to be of type ", kLong);

      Tensor wquasi = quasi.clone(MemoryFormat::Contiguous);
      auto result_dtype = dtype.has_value() ? dtype.value() : kFloat;
      Tensor result = empty({n, dimension}, sobolstate.options().dtype(result_dtype));

      AT_DISPATCH_FLOATING_TYPES(result_dtype, "_sobol_engine_draw", [&]() -> void {
        // We deal with `data` and `strides` due to performance issues.
        i64 l;
        i64* wquasi_data = wquasi.data_ptr<i64>();
        i64* sobolstate_data = sobolstate.data_ptr<i64>();
        Scalar* result_data = result.data_ptr<Scalar>();

        i64 wquasi_stride = wquasi.stride(0);
        i64 sobolstate_row_stride = sobolstate.stride(0), sobolstate_col_stride = sobolstate.stride(1);
        i64 result_row_stride = result.stride(0), result_col_stride = result.stride(1);

        for (i64 i = 0; i < n; i++, num_generated++) {
          l = rightmost_zero(num_generated);
          for (i64 j = 0; j < dimension; j++) {
            wquasi_data[j * wquasi_stride] ^= sobolstate_data[j * sobolstate_row_stride + l * sobolstate_col_stride];
            result_data[i * result_row_stride + j * result_col_stride] = wquasi_data[j * wquasi_stride];
          }
        }
      });

      result.mul_(RECIPD);
      return tuple<Tensor, Tensor>(result, wquasi);
        */
}

/**
  | This is the core function to fast-forward
  | a `SobolEngine` given its state variables
  | (`sobolstate` and `quasi`).
  |
  | `dimension` can be inferred from `sobolstate`,
  | but is passed as an argument for the same
  | reasons specified above.
  |
  */
pub fn sobol_engine_ff<'a>(
    quasi:         &mut Tensor,
    n:             i64,
    sobolstate:    &Tensor,
    dimension:     i64,
    num_generated: i64) -> &'a mut Tensor {

    todo!();
        /*
            TORCH_CHECK(sobolstate.dtype() == kLong,
               "sobolstate needs to be of type ", kLong);
      TORCH_CHECK(quasi.dtype() == kLong,
               "quasi needs to be of type ", kLong);

      // We deal with `data` and `strides` due to performance issues.
      i64 l;
      i64* quasi_data = quasi.data_ptr<i64>();
      i64* sobolstate_data = sobolstate.data_ptr<i64>();

      i64 quasi_stride = quasi.stride(0);
      i64 sobolstate_row_stride = sobolstate.stride(0), sobolstate_col_stride = sobolstate.stride(1);

      for (i64 i = 0; i < n; i++, num_generated++) {
        l = rightmost_zero(num_generated);
        for (i64 j = 0; j < dimension; j++) {
          quasi_data[j * quasi_stride] ^= sobolstate_data[j * sobolstate_row_stride + l * sobolstate_col_stride];
        }
      }
      return quasi;
        */
}

/**
  | This is an implicit function used for
  | randomizing the state variables of
  | the. `SobolEngine`. Arguments are a randomized
  | `sobolstate` state variables and a list of
  | random lower triangular matrices consisting of
  | 0s and 1s.
  |
  | `dimension` is passed explicitly again.
  |
  */
pub fn sobol_engine_scramble<'a>(
    sobolstate: &mut Tensor,
    ltm:        &Tensor,
    dimension:  i64) -> &'a mut Tensor {

    todo!();
        /*
            TORCH_CHECK(sobolstate.dtype() == kLong,
               "sobolstate needs to be of type ", kLong);

      /// Require a tensor accessor for `sobolstate`
      auto ss_a = sobolstate.accessor<i64, 2>();

      /// For every tensor in the list of tensors, the diagonals are made 1
      /// Require a dot product of every row with a specific vector of each of the matrices in `ltm`.
      /// Instead, we perform an element-wise product of all the matrices and sum over the last dimension.
      /// The required product of the m^{th} row in the d^{th} square matrix in `ltm` can be accessed
      /// using ltm_d_a[d][m] m and d are zero-indexed
      Tensor diag_true = ltm.clone(MemoryFormat::Contiguous);
      diag_true.diagonal(0, -2, -1).fill_(1);
      Tensor ltm_dots = cdot_pow2(diag_true);
      auto ltm_d_a = ltm_dots.accessor<i64, 2>();

      /// Main scrambling loop
      for (i64 d = 0; d < dimension; ++d) {
        for (i64 j = 0; j < MAXBIT; ++j) {
          i64 vdj = ss_a[d][j], l = 1, t2 = 0;
          for (i64 p = MAXBIT - 1; p >= 0; --p) {
            i64 lsmdp = ltm_d_a[d][p];
            i64 t1 = 0;
            for (i64 k = 0; k < MAXBIT; ++k) {
              t1 += (bitsubseq(lsmdp, k, 1) * bitsubseq(vdj, k, 1));
            }
            t1 = t1 % 2;
            t2 = t2 + t1 * l;
            l = l << 1;
          }
          ss_a[d][j] = t2;
        }
      }
      return sobolstate;
        */
}

/**
  | This is a core function to initialize
  | the main state variable of a `SobolEngine`.
  | `dimension` is passed explicitly as
  | well (see why above)
  |
  */
pub fn sobol_engine_initialize_state<'a>(
        sobolstate: &mut Tensor,
        dimension:  i64) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(sobolstate.dtype() == kLong,
               "sobolstate needs to be of type ", kLong);

      /// Use a tensor accessor for `sobolstate`
      auto ss_a = sobolstate.accessor<i64, 2>();

      /// First row of `sobolstate` is all 1s
      for (i64 m = 0; m < MAXBIT; ++m) {
        ss_a[0][m] = 1;
      }

      /// Remaining rows of sobolstate (row 2 through dim, indexed by [1:dim])
      for (i64 d = 1; d < dimension; ++d) {
        i64 p = poly[d];
        i64 m = bit_length(p) - 1;

        // First m elements of row d comes from initsobolstate
        for (i64 i = 0; i < m; ++i) {
          ss_a[d][i] = initsobolstate[d][i];
        }

        // Fill in remaining elements of v as in Section 2 (top of pg. 90) of:
        // P. Bratley and B. L. Fox. Algorithm 659: Implementing sobol's
        // quasirandom sequence generator. ACM Trans.
        // Math. Softw., 14(1):88-100, Mar. 1988.
        for (i64 j = m; j < MAXBIT; ++j) {
          i64 newv = ss_a[d][j - m];
          i64 pow2 = 1;
          for (i64 k = 0; k < m; ++k) {
            pow2 <<= 1;
            if ((p >> (m - 1 - k)) & 1) {
              newv = newv ^ (pow2 * ss_a[d][j - k - 1]);
            }
          }
          ss_a[d][j] = newv;
        }
      }

      /// Multiply each column of sobolstate by power of 2:
      /// sobolstate * [2^(maxbit-1), 2^(maxbit-2),..., 2, 1]
      Tensor pow2s = pow(
          2,
          native::arange(
              (MAXBIT - 1),
              -1,
              -1,
              optTypeMetaToScalarType(sobolstate.options().dtype_opt()),
              sobolstate.options().layout_opt(),
              sobolstate.options().device_opt(),
              sobolstate.options().pinned_memory_opt()));
      sobolstate.mul_(pow2s);
      return sobolstate;
        */
}
