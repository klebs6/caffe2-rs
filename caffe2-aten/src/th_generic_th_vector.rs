crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/TH/generic/THVector.h]
lazy_static!{
    /*
    #ifndef TH_GENERIC_FILE
    #define TH_GENERIC_FILE "TH/generic/THVector.h"
    #else
    #if !defined(TH_REAL_IS_BOOL) /* non bool only part */

    TH_API void THVector_(neg)(Scalar *y, const Scalar *x, const ptrdiff_t n);

    #endif /* non bool only part */

    /* floating point only now */
    #if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

    TH_API void THVector_(erfc)(Scalar *y, const Scalar *x, const ptrdiff_t n);
    TH_API void THVector_(pow)(Scalar *y, const Scalar *x, const Scalar c, const ptrdiff_t n);

    #endif /* floating point only part */

    #endif
    */
}

