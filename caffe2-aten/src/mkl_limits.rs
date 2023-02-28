crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/mkl/Limits.h]

/**
  | Since size of MKL_LONG varies on different
  | platforms (linux 64 bit, windows
  | 32 bit), we need to programmatically
  | calculate the max.
  */
lazy_static!{
    /*
    static i64 MKL_LONG_MAX = ((1LL << (sizeof(MKL_LONG) * 8 - 2)) - 1) * 2 + 1;
    */
}
