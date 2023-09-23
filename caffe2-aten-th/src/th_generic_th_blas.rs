// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/TH/generic/THBlas.h]
//-------------------------------------------[.cpp/pytorch/aten/src/TH/generic/THBlas.cpp]

lazy_static!{
    /*
    #ifndef TH_GENERIC_FILE
    #define TH_GENERIC_FILE "TH/generic/THBlas.h"
    #else

    /* Level 1 */
    TH_API void THBlas_(swap)(i64 n, Scalar *x, i64 incx, Scalar *y, i64 incy);

    #endif
    #ifndef TH_GENERIC_FILE
    #define TH_GENERIC_FILE "TH/generic/THBlas.cpp"
    #else

    #ifdef BLAS_F2C
    # define ffloat double
    #else
    # define ffloat float
    #endif

    TH_EXTERNC void dswap_(int *n, double *x, int *incx, double *y, int *incy);
    TH_EXTERNC void sswap_(int *n, float *x, int *incx, float *y, int *incy);

    void THBlas_(swap)(i64 n, Scalar *x, i64 incx, Scalar *y, i64 incy)
    {
      if(n == 1)
      {
        incx = 1;
        incy = 1;
      }

    #if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
      if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
      {
        int i_n = (int)n;
        int i_incx = (int)incx;
        int i_incy = (int)incy;

    #if defined(TH_REAL_IS_DOUBLE)
        dswap_(&i_n, x, &i_incx, y, &i_incy);
    #else
        sswap_(&i_n, x, &i_incx, y, &i_incy);
    #endif
        return;
      }
    #endif
      {
        i64 i;
        for(i = 0; i < n; i++)
        {
          Scalar z = x[i*incx];
          x[i*incx] = y[i*incy];
          y[i*incy] = z;
        }
      }
    }

    #endif
    */
}

