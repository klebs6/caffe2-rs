// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/TH/generic/THTensorMoreMath.cpp]
lazy_static!{
    /*
    #ifndef TH_GENERIC_FILE
    #define TH_GENERIC_FILE "TH/generic/THTensorMoreMath.cpp"
    #else




    ptrdiff_t THTensor_(numel)(THTensor *t)
    {
      return THTensor_(nElement)(t);
    }

    #if !defined(TH_REAL_IS_BFLOAT16) && !defined(TH_REAL_IS_HALF)

    // Helper function to be used in a reduction operation.
    // Due to resize semantics of outputs, if the specified output tensor r_ has
    // same size as the output of the reduction operation, then any noncontiguities
    // in r_ should be preserved.
    // The reduction operation, however, needs to act on r_ with an extra dimension
    // (the reduced dimension), so this function "resizes" r_ and preserves its
    // noncontiguities if necessary.
    void THTensor_(preserveReduceDimSemantics)(
        THTensor *r_, int in_dims, int reduce_dimension, int keepdim) {
      if (r_ && !keepdim &&
          THTensor_(nDimensionLegacyAll)(r_) == in_dims - 1 &&
          THTensor_(nDimensionLegacyAll)(r_) != 0) {
        THTensor_(unsqueeze1d)(r_, r_, reduce_dimension);
      }
    }

    #if !defined(TH_REAL_IS_BOOL) /* non bool only part */

    #define ARR(III) arr[(III)*stride]
    #define IDX(III) idx[(III)*stride]

    #define LONG_SWAP(AAA, BBB) swap = AAA; AAA = BBB; BBB = swap
    #define REAL_SWAP(AAA, BBB) rswap = AAA; AAA = BBB; BBB = rswap

    #define ARR_SWAP(III, JJJ) \
      REAL_SWAP(ARR(III), ARR(JJJ));

    #define BOTH_SWAP(III, JJJ) \
      REAL_SWAP(ARR(III), ARR(JJJ)); \
      LONG_SWAP(IDX(III), IDX(JJJ))

    /* Emulate NumPy behavior of putting NaNs
     * at the end of an ascending list. */
    #define GT_OR_NAN(x, y) \
      ((th_isnan(x) && !(th_isnan(y))) || (x > y))

    /* Implementation of the Quickselect algorithm, based on Nicolas Devillard's
    public domain implementation at http://ndevilla.free.fr/median/median/
    Adapted similarly to the above Quicksort algorithm. */
    static void THTensor_(quickselect)(Scalar *arr, i64 *idx, i64 k, i64 elements, i64 stride)
    {
      i64 P, L, R, i, j, swap;
      Scalar rswap, piv;
      L = 0;
      R = elements-1;

      do {
        if (R <= L) /* One element only */
          return;

        if (R == L+1) {  /* Two elements only */
          if (ARR(L) > ARR(R)) {
            BOTH_SWAP(L, R);
          }
          return;
        }

        /* Use median of three for pivot choice */
        P=(L+R)>>1;
        BOTH_SWAP(P, L+1);
        if (ARR(L+1) > ARR(R)) { BOTH_SWAP(L+1, R); }
        if (ARR(L) > ARR(R)) { BOTH_SWAP(L, R); }
        if (ARR(L+1) > ARR(L)) { BOTH_SWAP(L+1, L); }

        i = L+1;
        j = R;
        piv = ARR(L);
        do {
          do i++; while(ARR(i) < piv);
          do j--; while(ARR(j) > piv);
          if (j < i)
            break;
          BOTH_SWAP(i, j);
        } while(1);
        BOTH_SWAP(L, j);

        /* Re-set active partition */
        if (j <= k) L=i;
        if (j >= k) R=j-1;
      } while(1);
    }

    #undef ARR
    #undef IDX
    #undef LONG_SWAP
    #undef REAL_SWAP
    #undef BOTH_SWAP

    void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, i64 k, int dimension, int keepdim)
    {
      THTensor *temp_;
      THLongTensor *tempi_;
      Scalar *temp__data;
      i64 *tempi__data;
      i64 t_size_dim;

      THArgCheck(dimension >= 0 && dimension < THTensor_(nDimensionLegacyAll)(t), 3, "dimension out of range");
      THArgCheck(k > 0 && k <= THTensor_sizeLegacyNoScalars(t, dimension), 2, "selected index out of range");

      int in_dims = THTensor_(nDimensionLegacyAll)(t);
      THTensor_(preserveReduceDimSemantics)(values_, in_dims, dimension, keepdim);
      THLongTensor_preserveReduceDimSemantics(indices_, in_dims, dimension, keepdim);
      vector<i64> dim = THTensor_sizesLegacyNoScalars(t);
      dim[dimension] = 1;
      THTensor_(resize)(values_, dim, {});
      THLongTensor_resize(indices_, dim, {});

      t_size_dim = THTensor_sizeLegacyNoScalars(t, dimension);

      temp_ = THTensor_(new)();
      THTensor_(resize1d)(temp_, t_size_dim);
      temp__data = temp_->data<Scalar>();

      tempi_ = THLongTensor_new();
      THLongTensor_resize1d(tempi_, t_size_dim);
      tempi__data = THLongTensor_data(tempi_);

      TH_TENSOR_DIM_APPLY3(Scalar, t, Scalar, values_, i64, indices_, dimension,
                           TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
                           i64 i;
                           for(i = 0; i < t_size_dim; i++)
                              temp__data[i] = t_data[i*t_stride];
                           for(i = 0; i < t_size_dim; i++)
                              tempi__data[i] = i;
                           THTensor_(quickselect)(temp__data, tempi__data, k - 1, t_size_dim, 1);
                           *values__data = temp__data[k-1];
                           *indices__data = tempi__data[k-1];);

      raw::intrusive_ptr::decref(temp_);
      THLongTensor_free(tempi_);
      if (!keepdim) {
        THTensor_(squeeze1d)(values_, values_, dimension);
        THLongTensor_squeeze1d(indices_, indices_, dimension);
      }
    }

    static void THTensor_(propagate_names_if_named_tensor_enabled)(THTensor* result, THTensor* src) {
      namedinference::propagate_names(result, src);
    }

    #define LAB_IMPLEMENT_BASIC_FUNCTION_3_ARGS(NAME, CFUNC, THRESHOLD) \
      void THTensor_(NAME)(THTensor *r_, THTensor *t) \
      { \
        THTensor_(resizeAs)(r_, t); \
        ptrdiff_t r_Size = THTensor_(nElement)(r_); \
        int r_Contig = THTensor_(isContiguous)(r_); \
        int tContig = THTensor_(isContiguous)(t); \
        TH_TENSOR_APPLY2_PARALLEL(r_Size, r_Contig, tContig, Scalar, r_, Scalar, t, *r__data = CFUNC(*t_data);, THRESHOLD); \
        THTensor_(propagate_names_if_named_tensor_enabled)(r_, t); \
      }

    #define LAB_IMPLEMENT_BASIC_FUNCTION_2_ARGS(NAME, CFUNC) \
      LAB_IMPLEMENT_BASIC_FUNCTION_3_ARGS(NAME, CFUNC, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD)

    #define LAB_IMPLEMENT_VECTORIZED_FUNCTION_3_ARGS(NAME, CFUNC, THRESHOLD) \
      void THTensor_(NAME)(THTensor *r_, THTensor *t) \
      { \
        THTensor_(resizeAs)(r_, t); \
        ptrdiff_t r_Size = THTensor_(nElement)(r_); \
        int r_Contig = THTensor_(isContiguous)(r_); \
        int tContig = THTensor_(isContiguous)(t); \
        if (r_Contig && tContig) { \
          TH_TENSOR_APPLY2_CONTIG(Scalar, r_, Scalar, t, THVector_(NAME)(r__data, t_data, r__len);); \
        } else { \
          TH_TENSOR_APPLY2_PARALLEL(r_Size, r_Contig, tContig, Scalar, r_, Scalar, t, *r__data = CFUNC(*t_data);, THRESHOLD); \
        } \
        THTensor_(propagate_names_if_named_tensor_enabled)(r_, t); \
      }

    #define LAB_IMPLEMENT_VECTORIZED_FUNCTION_2_ARGS(NAME, CFUNC) \
      LAB_IMPLEMENT_VECTORIZED_FUNCTION_3_ARGS(NAME, CFUNC, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD)

    #define EXPAND(...) __VA_ARGS__

    #define GET_4TH_ARG(ARG0, ARG1, ARG2, ARG3, ...) ARG3

    #define LAB_IMPLEMENT_BASIC_FUNCTION_CHOOSE(...) \
      EXPAND(GET_4TH_ARG(__VA_ARGS__, LAB_IMPLEMENT_BASIC_FUNCTION_3_ARGS, LAB_IMPLEMENT_BASIC_FUNCTION_2_ARGS, ))

    #define LAB_IMPLEMENT_VECTORIZED_FUNCTION_CHOOSE(...) \
      EXPAND(GET_4TH_ARG(__VA_ARGS__, LAB_IMPLEMENT_VECTORIZED_FUNCTION_3_ARGS, LAB_IMPLEMENT_VECTORIZED_FUNCTION_2_ARGS, ))

    #define LAB_IMPLEMENT_BASIC_FUNCTION(...) EXPAND(LAB_IMPLEMENT_BASIC_FUNCTION_CHOOSE(__VA_ARGS__)(__VA_ARGS__))

    #define LAB_IMPLEMENT_VECTORIZED_FUNCTION(...) EXPAND(LAB_IMPLEMENT_VECTORIZED_FUNCTION_CHOOSE(__VA_ARGS__)(__VA_ARGS__))

    /*
     * LAB_IMPLEMENT_BASIC_FUNCTION is a macro with optional parameters, you can use it flexibly.
     * The macro will discard the invalid threshold if parallelization is unavailable.
     * The macro will give a default threshold even if you forget to pass one.
     * In other word,
     * (A), If parallelization is UNavailable, the two usage below is both right.
     *      (1) LAB_IMPLEMENT_BASIC_FUNCTION(type_func, func_entity, OMP_OVERHEAD_THRESHOLD) // discard the invalid threshold
     *      (2) LAB_IMPLEMENT_BASIC_FUNCTION(type_func, func_entity)
     * (B), If parallelization is available, the two usage below is also both right.
     *      (1) LAB_IMPLEMENT_BASIC_FUNCTION(type_func, func_entity, OMP_OVERHEAD_THRESHOLD)
     *      (2) LAB_IMPLEMENT_BASIC_FUNCTION(type_func, func_entity) // pass the default threshold
     * So do LAB_IMPLEMENT_VECTORIZED_FUNCTION.
    */

    LAB_IMPLEMENT_BASIC_FUNCTION(neg,-)

    #if defined(TH_REAL_IS_LONG)
    LAB_IMPLEMENT_BASIC_FUNCTION(abs,abs)
    #endif /* i64 only part */

    #if defined(TH_REAL_IS_SHORT) || defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_CHAR)
    LAB_IMPLEMENT_BASIC_FUNCTION(abs,abs)
    #endif /* int only part */

    #if defined(TH_REAL_IS_BYTE)
    LAB_IMPLEMENT_BASIC_FUNCTION(abs,)
    #endif /* for byte, identity due to it being unsigned */

    /* floating point only now */
    #if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

    #if defined (TH_REAL_IS_FLOAT)
    #define TH_MATH_NAME(fn) fn##f
    #else
    #define TH_MATH_NAME(fn) fn
    #endif

    LAB_IMPLEMENT_BASIC_FUNCTION(abs,TH_MATH_NAME(fabs))

    LAB_IMPLEMENT_BASIC_FUNCTION(cosh,TH_MATH_NAME(cosh),HYPER_TH_OMP_OVERHEAD_THRESHOLD)
    LAB_IMPLEMENT_BASIC_FUNCTION(tanh,TH_MATH_NAME(tanh),HYPER_TH_OMP_OVERHEAD_THRESHOLD)

    void THTensor_(histc)(THTensor *hist, THTensor *tensor, i64 nbins, Scalar minvalue, Scalar maxvalue)
    {
      if (nbins <= 0) {
          THError("bins must be > 0");
      }
      Scalar minval;
      Scalar maxval;
      Scalar *h_data;

      THTensor_(resize1d)(hist, nbins);
      THTensor_wrap(hist).zero_();
      minval = minvalue;
      maxval = maxvalue;
      if (minval == maxval)
      {
        minval = THTensor_wrap(tensor).min().item<Scalar>();
        maxval = THTensor_wrap(tensor).max().item<Scalar>();
      }
      if (minval == maxval)
      {
        minval = minval - 1;
        maxval = maxval + 1;
      }

      TORCH_CHECK(!(isinf(minval) || isinf(maxval) || isnan(minval) || isnan(maxval)), "range of [", minval, ", ", maxval, "] is not finite");
      TORCH_CHECK(minval < maxval, "max must be larger than min");

      h_data = hist->data<Scalar>();

      TH_TENSOR_APPLY(Scalar, tensor,
        if (*tensor_data >= minval && *tensor_data <= maxval) {
          const int bin = (int)((*tensor_data-minval) / (maxval-minval) * nbins);
          h_data[THMin(bin, nbins-1)] += 1;
        }
      );
    }

    #endif

    #undef TH_MATH_NAME
    #endif /* floating point only part */
    #undef IS_NONZERO

    #endif /* !defined(TH_REAL_IS_BOOL) */

    #endif /* TH_GENERIC_FILE */
    */
}

