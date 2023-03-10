// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/THC/generic/THCTensorMathMagma.h]
lazy_static!{
    /*
    #ifndef THC_GENERIC_FILE
    #define THC_GENERIC_FILE "THC/generic/THCTensorMathMagma.h"
    #else

    #if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)

    // MAGMA (i.e. CUDA implementation of LAPACK functions)
    TORCH_CUDA_CU_API void THCTensor_(gels)(
        THCState* state,
        THCTensor* rb_,
        THCTensor* ra_,
        THCTensor* b_,
        THCTensor* a_);

    #endif // defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)

    #endif
    //-------------------------------------------[.cpp/pytorch/aten/src/THC/generic/THCTensorMathMagma.cpp]
    #ifndef THC_GENERIC_FILE
    #define THC_GENERIC_FILE "THC/generic/THCTensorMathMagma.cpp"
    #else

    #if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)

    static THCTensor* THCTensor_(newColumnMajor)(THCState *state, THCTensor *self, THCTensor *src)
    {
      THAssert(src->dim() == 2);
      if (self == src && self->stride(0) == 1 && self->stride(1) == self->size(0))
      {
        THCTensor_(retain)(state, self);
        return self;
      }

      if (self == src)
        self = THCTensor_(new)(state);
      else
        THCTensor_(retain)(state, self);

      i64 size[2] = { src->size(0), src->size(1) };
      i64 stride[2] = { 1, src->size(0) };

      THCTensor_(resizeNd)(state, self, 2, size, stride);
      THCTensor_(copy)(state, self, src);
      return self;
    }

    void THCTensor_(gels)(THCState *state, THCTensor *rb_, THCTensor *ra_, THCTensor *b_, THCTensor *a_)
    {
    #ifdef USE_MAGMA
      THArgCheck(!a_->is_empty() && a_->dim() == 2, 1, "A should be (non-empty) 2 dimensional");
      THArgCheck(!b_->is_empty() && b_->dim() == 2, 1, "b should be (non-empty) 2 dimensional");
      TORCH_CHECK(a_->size(0) == b_->size(0), "Expected A and b to have same size "
          "at dim 0, but A has ", a_->size(0), " rows and B has ", b_->size(0), " rows");
      THArgCheck(a_->size(0) >= a_->size(1), 2, "Expected A with shape (m x n) to have "
          "m >= n. The case for m < n is not implemented yet.");

      THCTensor *a = THCTensor_(newColumnMajor)(state, ra_, a_);
      THCTensor *b = THCTensor_(newColumnMajor)(state, rb_, b_);
      Scalar *a_data = THCTensor_(data)(state, a);
      Scalar *b_data = THCTensor_(data)(state, b);

      i64 m = a->size(0);
      i64 n = a->size(1);
      i64 nrhs = b->size(1);
      Scalar wkopt;

      int info;
      {
        native::MagmaStreamSyncGuard guard;
    #if defined(THC_REAL_IS_FLOAT)
        magma_sgels_gpu(MagmaNoTrans, m, n, nrhs, a_data, m, b_data, m, &wkopt, -1, &info);
    #else
        magma_dgels_gpu(MagmaNoTrans, m, n, nrhs, a_data, m, b_data, m, &wkopt, -1, &info);
    #endif

        Scalar *hwork = th_magma_malloc_pinned<Scalar>((usize)wkopt);

    #if defined(THC_REAL_IS_FLOAT)
        magma_sgels_gpu(MagmaNoTrans, m, n, nrhs, a_data, m, b_data, m, hwork, (int)wkopt, &info);
    #else
        magma_dgels_gpu(MagmaNoTrans, m, n, nrhs, a_data, m, b_data, m, hwork, (int)wkopt, &info);
    #endif

        magma_free_pinned(hwork);
      }

      if (info != 0)
        THError("MAGMA gels : Argument %d : illegal value", -info);

      THCTensor_(freeCopyTo)(state, a, ra_);
      THCTensor_(freeCopyTo)(state, b, rb_);
    #else
      THError(NoMagma(gels));
    #endif
    }

    #endif

    #endif
    */
}

