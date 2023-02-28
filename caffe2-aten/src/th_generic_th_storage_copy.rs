// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/TH/generic/THStorageCopy.h]
lazy_static!{
    /*
    #ifndef TH_GENERIC_FILE
    #define TH_GENERIC_FILE "TH/generic/THStorageCopy.h"
    #else

    /* Support for copy between different Storage types */
    TH_API void THStorage_(copy)(THStorage *storage, THStorage *src);

    // TODO: Add cross-dtype storage copy for complex storage
    #if !defined(TH_REAL_IS_COMPLEXFLOAT) && !defined(TH_REAL_IS_COMPLEXDOUBLE)
      TH_API void THStorage_(copyByte)(THStorage *storage, struct THByteStorage *src);
      TH_API void THStorage_(copyChar)(THStorage *storage, struct THCharStorage *src);
      TH_API void THStorage_(copyShort)(THStorage *storage, struct THShortStorage *src);
      TH_API void THStorage_(copyInt)(THStorage *storage, struct THIntStorage *src);
      TH_API void THStorage_(copyLong)(THStorage *storage, struct THLongStorage *src);
      TH_API void THStorage_(copyFloat)(THStorage *storage, struct THFloatStorage *src);
      TH_API void THStorage_(copyDouble)(THStorage *storage, struct THDoubleStorage *src);
      TH_API void THStorage_(copyHalf)(THStorage *storage, struct THHalfStorage *src);
      TH_API void THStorage_(copyBool)(THStorage *storage, struct THBoolStorage *src);
      TH_API void THStorage_(copyBFloat16)(THStorage *storage, struct THBFloat16Storage *src);
      #ifdef THQUINT8
        TH_API void THStorage_(copyQUInt8)(THStorage *storage, struct THQUInt8Storage *src);
      #endif
      #ifdef THQINT8
        TH_API void THStorage_(copyQInt8)(THStorage *storage, struct THQInt8Storage *src);
      #endif
      #ifdef THQINT32
        TH_API void THStorage_(copyQInt32)(THStorage *storage, struct THQInt32Storage *src);
      #endif
    #else
      TH_API void THStorage_(copyComplexFloat)(THStorage *storage, struct THComplexFloatStorage *src);
      TH_API void THStorage_(copyComplexDouble)(THStorage *storage, struct THComplexDoubleStorage *src);
    #endif

    #endif
    //-------------------------------------------[.cpp/pytorch/aten/src/TH/generic/THStorageCopy.cpp]
    #ifndef TH_GENERIC_FILE
    #define TH_GENERIC_FILE "TH/generic/THStorageCopy.cpp"
    #else

    void THStorage_(copy)(THStorage *storage, THStorage *src)
    {
      THArgCheck(storage->nbytes() == src->nbytes(), 2, "size mismatch");
      Scalar *scalar_src = THStorage_(data)(src);
      Scalar *data = THStorage_(data)(storage);
      u64 numel = storage->nbytes() / sizeof(Scalar);
      for (u64 i = 0; i < numel; ++i) {
        data[i] = scalar_src[i];
      }
    }

    // NOTE: for performance, these macros generally use the raw data pointer in the inner loops,
    // rather than repeated THStorage_(data) calls.

    #define IMPLEMENT_THStorage_COPY(TYPENAMESRC)                \
      void THStorage_(copy##TYPENAMESRC)(                        \
          THStorage * storage, TH##TYPENAMESRC##Storage * src) { \
        auto data = THStorage_(data)(storage);                   \
        auto src_data = TH##TYPENAMESRC##Storage_data(src);      \
        u64 numel = storage->nbytes() / sizeof(Scalar);   \
        for (u64 i = 0; i < numel; i++)                     \
          data[i] = static_cast<Scalar>(src_data[i]);          \
      }

    // TODO: Add cross-dtype storage copy for complex storage
    #if !defined(TH_REAL_IS_COMPLEXFLOAT) && !defined(TH_REAL_IS_COMPLEXDOUBLE)
      IMPLEMENT_THStorage_COPY(Byte)
      IMPLEMENT_THStorage_COPY(Char)
      IMPLEMENT_THStorage_COPY(Short)
      IMPLEMENT_THStorage_COPY(Int)
      IMPLEMENT_THStorage_COPY(Long)
      IMPLEMENT_THStorage_COPY(Float)
      IMPLEMENT_THStorage_COPY(Double)
      IMPLEMENT_THStorage_COPY(Half)
      IMPLEMENT_THStorage_COPY(Bool)
      IMPLEMENT_THStorage_COPY(BFloat16)
      #ifdef THQUINT8
        IMPLEMENT_THStorage_COPY(QUInt8)
      #endif
      #ifdef THQINT8
        IMPLEMENT_THStorage_COPY(QInt8)
      #endif
      #ifdef THQINT32
        IMPLEMENT_THStorage_COPY(QInt32)
      #endif
    #else
      IMPLEMENT_THStorage_COPY(ComplexFloat)
      IMPLEMENT_THStorage_COPY(ComplexDouble)
    #endif

    #endif
    */
}

