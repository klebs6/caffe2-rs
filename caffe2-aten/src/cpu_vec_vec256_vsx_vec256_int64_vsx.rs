// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/vsx/vec256_int64_vsx.h]
lazy_static!{
    /*
    #pragma once

    #include <ATen/cpu/vec/vec256/intrinsics.h>
    #include <ATen/cpu/vec/vec256/vec256_base.h>
    #include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
    namespace at {
    namespace vec {
    // See Note [Acceptable use of anonymous namespace in header]
    namespace {

    template <>
    class Vectorized<i64> {
     private:
      union {
        struct {
          vint64 _vec0;
          vint64 _vec1;
        };
        struct {
          vbool64 _vecb0;
          vbool64 _vecb1;
        };

      } __attribute__((__may_alias__));

     public:
      using value_type = i64;
      using vec_internal_type = vint64;
      using vec_internal_mask_type = vbool64;
      using Sizeype = int;
      static constexpr Sizeype size() {
        return 4;
      }
      Vectorized() {}
      C10_ALWAYS_INLINE Vectorized(vint64 v) : _vec0{v}, _vec1{v} {}
      C10_ALWAYS_INLINE Vectorized(vbool64 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
      C10_ALWAYS_INLINE Vectorized(vint64 v1, vint64 v2) : _vec0{v1}, _vec1{v2} {}
      C10_ALWAYS_INLINE Vectorized(vbool64 v1, vbool64 v2) : _vecb0{v1}, _vecb1{v2} {}
      C10_ALWAYS_INLINE Vectorized(i64 scalar)
          : _vec0{vec_splats(scalar)}, _vec1{vec_splats(scalar)} {}
      C10_ALWAYS_INLINE Vectorized(
          i64 scalar1,
          i64 scalar2,
          i64 scalar3,
          i64 scalar4)
          : _vec0{vint64{scalar1, scalar2}}, _vec1{vint64{scalar3, scalar4}} {}

      C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
        return _vec0;
      }
      C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
        return _vec1;
      }

      template <u64 mask>
      static std::enable_if_t<mask == 0, Vectorized<i64>> C10_ALWAYS_INLINE
      blend(const Vectorized<i64>& a, const Vectorized<i64>& b) {
        return a;
      }

      template <u64 mask>
      static std::enable_if_t<mask == 3, Vectorized<i64>> C10_ALWAYS_INLINE
      blend(const Vectorized<i64>& a, const Vectorized<i64>& b) {
        return {b._vec0, a._vec1};
      }

      template <u64 mask>
      static std::enable_if_t<(mask & 15) == 15, Vectorized<i64>> C10_ALWAYS_INLINE
      blend(const Vectorized<i64>& a, const Vectorized<i64>& b) {
        return b;
      }

      template <u64 mask>
      static std::enable_if_t<(mask > 0 && mask < 3), Vectorized<i64>> C10_ALWAYS_INLINE
      blend(const Vectorized<i64>& a, const Vectorized<i64>& b) {
        constexpr u64 g0 = (mask & 1) * 0xffffffffffffffff;
        constexpr u64 g1 = ((mask & 2) >> 1) * 0xffffffffffffffff;
        const vbool64 mask_1st = (vbool64){g0, g1};
        return {(vint64)vec_sel(a._vec0, b._vec0, (vbool64)mask_1st), a._vec1};
      }

      template <u64 mask>
      static std::enable_if_t<(mask > 3) && (mask & 3) == 0, Vectorized<i64>>
          C10_ALWAYS_INLINE blend(const Vectorized<i64>& a, const Vectorized<i64>& b) {
        constexpr u64 g0_2 = ((mask & 4) >> 2) * 0xffffffffffffffff;
        constexpr u64 g1_2 = ((mask & 8) >> 3) * 0xffffffffffffffff;

        const vbool64 mask_2nd = (vbool64){g0_2, g1_2};
        return {a._vec0, (vint64)vec_sel(a._vec1, b._vec1, (vbool64)mask_2nd)};
      }

      template <u64 mask>
      static std::enable_if_t<
          (mask > 3) && (mask & 3) != 0 && (mask & 15) != 15,
          Vectorized<i64>>
          C10_ALWAYS_INLINE blend(const Vectorized<i64>& a, const Vectorized<i64>& b) {
        constexpr u64 g0 = (mask & 1) * 0xffffffffffffffff;
        constexpr u64 g1 = ((mask & 2) >> 1) * 0xffffffffffffffff;
        constexpr u64 g0_2 = ((mask & 4) >> 2) * 0xffffffffffffffff;
        constexpr u64 g1_2 = ((mask & 8) >> 3) * 0xffffffffffffffff;

        const vbool64 mask_1st = (vbool64){g0, g1};
        const vbool64 mask_2nd = (vbool64){g0_2, g1_2};
        return {
            (vint64)vec_sel(a._vec0, b._vec0, (vbool64)mask_1st),
            (vint64)vec_sel(a._vec1, b._vec1, (vbool64)mask_2nd)};
      }

      static Vectorized<i64> C10_ALWAYS_INLINE blendv(
          const Vectorized<i64>& a,
          const Vectorized<i64>& b,
          const Vectorized<i64>& mask) {
        // the mask used here returned by comparision of vec256

        return {
            vec_sel(a._vec0, b._vec0, mask._vecb0),
            vec_sel(a._vec1, b._vec1, mask._vecb1)};
      }
      template <typename step_t>
      static Vectorized<i64> arange(i64 base = 0., step_t step = static_cast<step_t>(1)) {
        return Vectorized<i64>(base, base + step, base + 2 * step, base + 3 * step);
      }

      static Vectorized<i64> C10_ALWAYS_INLINE
      set(const Vectorized<i64>& a,
          const Vectorized<i64>& b,
          usize count = size()) {
        switch (count) {
          case 0:
            return a;
          case 1:
            return blend<1>(a, b);
          case 2:
            return blend<3>(a, b);
          case 3:
            return blend<7>(a, b);
        }

        return b;
      }
      static Vectorized<value_type> C10_ALWAYS_INLINE
      loadu(const void* ptr, int count = size()) {
        if (count == size()) {
          static_assert(sizeof(double) == sizeof(value_type));
          const double* dptr = reinterpret_cast<const double*>(ptr);
          return {// treat it as double load
                  (vint64)vec_vsx_ld(offset0, dptr),
                  (vint64)vec_vsx_ld(offset16, dptr)};
        }

        __at_align32__ double tmp_values[size()];
        std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(value_type));

        return {
            (vint64)vec_vsx_ld(offset0, tmp_values),
            (vint64)vec_vsx_ld(offset16, tmp_values)};
      }
      void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
        if (count == size()) {
          double* dptr = reinterpret_cast<double*>(ptr);
          vec_vsx_st((vfloat64)_vec0, offset0, dptr);
          vec_vsx_st((vfloat64)_vec1, offset16, dptr);
        } else if (count > 0) {
          __at_align32__ double tmp_values[size()];
          vec_vsx_st((vfloat64)_vec0, offset0, tmp_values);
          vec_vsx_st((vfloat64)_vec1, offset16, tmp_values);
          std::memcpy(
              ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
        }
      }
      const i64& operator[](int idx) const = delete;
      i64& operator[](int idx) = delete;

      Vectorized<i64> angle() const {
        return blendv(
          Vectorized<i64>(0), Vectorized<i64>(c10::pi<i64>), *this < Vectorized<i64>(0));
      }
      Vectorized<i64> real() const {
        return *this;
      }
      Vectorized<i64> imag() const {
        return Vectorized<i64>{0};
      }
      Vectorized<i64> conj() const {
        return *this;
      }

      Vectorized<i64> C10_ALWAYS_INLINE abs() const {
        return {vec_abs(_vec0), vec_abs(_vec1)};
      }

      Vectorized<i64> C10_ALWAYS_INLINE neg() const {
        return {vec_neg(_vec0), vec_neg(_vec1)};
      }

      DEFINE_MEMBER_UNARY_OP(operator~, i64, vec_not)
      DEFINE_MEMBER_OP(operator==, i64, vec_cmpeq)
      DEFINE_MEMBER_OP(operator!=, i64, vec_cmpne)
      DEFINE_MEMBER_OP(operator<, i64, vec_cmplt)
      DEFINE_MEMBER_OP(operator<=, i64, vec_cmple)
      DEFINE_MEMBER_OP(operator>, i64, vec_cmpgt)
      DEFINE_MEMBER_OP(operator>=, i64, vec_cmpge)
      DEFINE_MEMBER_OP_AND_ONE(eq, i64, vec_cmpeq)
      DEFINE_MEMBER_OP_AND_ONE(ne, i64, vec_cmpne)
      DEFINE_MEMBER_OP_AND_ONE(lt, i64, vec_cmplt)
      DEFINE_MEMBER_OP_AND_ONE(le, i64, vec_cmple)
      DEFINE_MEMBER_OP_AND_ONE(gt, i64, vec_cmpgt)
      DEFINE_MEMBER_OP_AND_ONE(ge, i64, vec_cmpge)
      DEFINE_MEMBER_OP(operator+, i64, vec_add)
      DEFINE_MEMBER_OP(operator-, i64, vec_sub)
      DEFINE_MEMBER_OP(operator*, i64, vec_mul)
      DEFINE_MEMBER_OP(operator/, i64, vec_div)
      DEFINE_MEMBER_OP(maximum, i64, vec_max)
      DEFINE_MEMBER_OP(minimum, i64, vec_min)
      DEFINE_MEMBER_OP(operator&, i64, vec_and)
      DEFINE_MEMBER_OP(operator|, i64, vec_or)
      DEFINE_MEMBER_OP(operator^, i64, vec_xor)
    };

    template <>
    Vectorized<i64> inline maximum(
        const Vectorized<i64>& a,
        const Vectorized<i64>& b) {
      return a.maximum(b);
    }

    template <>
    Vectorized<i64> inline minimum(
        const Vectorized<i64>& a,
        const Vectorized<i64>& b) {
      return a.minimum(b);
    }

    } // namespace
    } // namespace vec
    } // namespace at
    */
}

