crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/TH/generic/THTensorFastGetSet.hpp]
lazy_static!{
    /*
    #ifndef TH_GENERIC_FILE
    #define TH_GENERIC_FILE "TH/generic/THTensorFastGetSet.hpp"
    #else

    static inline Scalar THTensor_(fastGetLegacy1dNoScalars)(THTensor *self, i64 x0) {
      return self->unsafe_data<Scalar>()[x0*THTensor_strideLegacyNoScalars(self, 0)];
    }

    static inline Scalar THTensor_(fastGet1d)(THTensor *self, i64 x0) {
      return self->unsafe_data<Scalar>()[x0*self->stride(0)];
    }

    static inline Scalar THTensor_(fastGet2d)(THTensor *self, i64 x0, i64 x1) {
      return self->unsafe_data<Scalar>()[x0*self->stride(0)+x1*self->stride(1)];
    }

    static inline Scalar THTensor_(fastGet3d)(THTensor *self, i64 x0, i64 x1, i64 x2) {
      return self->unsafe_data<Scalar>()[x0*self->stride(0)+x1*self->stride(1)+x2*self->stride(2)];
    }

    static inline Scalar THTensor_(fastGet4d)(THTensor *self, i64 x0, i64 x1, i64 x2, i64 x3) {
      return self->unsafe_data<Scalar>()[x0*self->stride(0)+x1*self->stride(1)+x2*self->stride(2)+x3*self->stride(3)];
    }

    static inline Scalar THTensor_(fastGet5d)(THTensor *self, i64 x0, i64 x1, i64 x2, i64 x3, i64 x4) {
      return self->unsafe_data<Scalar>()[x0*self->stride(0)+x1*self->stride(1)+x2*self->stride(2)+x3*self->stride(3)+(x4)*self->stride(4)];
    }

    static inline void THTensor_(fastSet1d)(THTensor *self, i64 x0, Scalar value) {
      self->unsafe_data<Scalar>()[x0*self->stride(0)] = value;
    }

    static inline void THTensor_(fastSet2d)(THTensor *self, i64 x0, i64 x1, Scalar value) {
      self->unsafe_data<Scalar>()[x0*self->stride(0)+x1*self->stride(1)] = value;
    }

    static inline void THTensor_(fastSet3d)(THTensor *self, i64 x0, i64 x1, i64 x2, Scalar value) {
      self->unsafe_data<Scalar>()[x0*self->stride(0)+x1*self->stride(1)+x2*self->stride(2)] = value;
    }

    static inline void THTensor_(fastSet4d)(THTensor *self, i64 x0, i64 x1, i64 x2, i64 x3, Scalar value) {
      self->unsafe_data<Scalar>()[x0*self->stride(0)+x1*self->stride(1)+x2*self->stride(2)+x3*self->stride(3)] = value;
    }

    static inline void THTensor_(fastSet5d)(THTensor *self, i64 x0, i64 x1, i64 x2, i64 x3, i64 x4, Scalar value) {
      self->unsafe_data<Scalar>()[x0*self->stride(0)+x1*self->stride(1)+x2*self->stride(2)+x3*self->stride(3)+(x4)*self->stride(4)] = value;
    }

    #endif
    */
}

