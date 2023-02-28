// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/vsx/vec256_int32_vsx.h]

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
    class Vectorized<i32> {
     private:
      union {
        struct {
          vint32 _vec0;
          vint32 _vec1;
        };
        struct {
          vbool32 _vecb0;
          vbool32 _vecb1;
        };

      } __attribute__((__may_alias__));

     public:
      using value_type = i32;
      using vec_internal_type = vint32;
      using vec_internal_mask_type = vbool32;
      using Sizeype = int;
      static constexpr Sizeype size() {
        return 8;
      }
      Vectorized() {}
      C10_ALWAYS_INLINE Vectorized(vint32 v) : _vec0{v}, _vec1{v} {}
      C10_ALWAYS_INLINE Vectorized(vbool32 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
      C10_ALWAYS_INLINE Vectorized(vint32 v1, vint32 v2) : _vec0{v1}, _vec1{v2} {}
      C10_ALWAYS_INLINE Vectorized(vbool32 v1, vbool32 v2) : _vecb0{v1}, _vecb1{v2} {}
      C10_ALWAYS_INLINE Vectorized(i32 scalar)
          : _vec0{vec_splats(scalar)}, _vec1{vec_splats(scalar)} {}
      C10_ALWAYS_INLINE Vectorized(
          i32 scalar1,
          i32 scalar2,
          i32 scalar3,
          i32 scalar4,
          i32 scalar5,
          i32 scalar6,
          i32 scalar7,
          i32 scalar8)
          : _vec0{vint32{scalar1, scalar2, scalar3, scalar4}},
            _vec1{vint32{scalar5, scalar6, scalar7, scalar8}} {}
      C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
        return _vec0;
      }
      C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
        return _vec1;
      }

      template <u64 mask>
      static std::enable_if_t<mask == 0, Vectorized<i32>> C10_ALWAYS_INLINE
      blend(const Vectorized<i32>& a, const Vectorized<i32>& b) {
        return a;
      }

      template <u64 mask>
      static std::enable_if_t<(mask & 255) == 255, Vectorized<i32>> C10_ALWAYS_INLINE
      blend(const Vectorized<i32>& a, const Vectorized<i32>& b) {
        return b;
      }

      template <u64 mask>
      static std::enable_if_t<mask == 15, Vectorized<i32>> C10_ALWAYS_INLINE
      blend(const Vectorized<i32>& a, const Vectorized<i32>& b) {
        return {b._vec0, a._vec1};
      }

      template <u64 mask>
      static std::enable_if_t<(mask > 0 && mask < 15), Vectorized<i32>>
          C10_ALWAYS_INLINE blend(const Vectorized<i32>& a, const Vectorized<i32>& b) {
        constexpr u32 g0 = (mask & 1) * 0xffffffff;
        constexpr u32 g1 = ((mask & 2) >> 1) * 0xffffffff;
        constexpr u32 g2 = ((mask & 4) >> 2) * 0xffffffff;
        constexpr u32 g3 = ((mask & 8) >> 3) * 0xffffffff;
        const vbool32 mask_1st = (vbool32){g0, g1, g2, g3};

        return {(vint32)vec_sel(a._vec0, b._vec0, (vbool32)mask_1st), a._vec1};
      }

      template <u64 mask>
      static std::enable_if_t<
          (mask > 15 && (mask & 255) != 255 && ((mask & 15) == 15)),
          Vectorized<i32>>
          C10_ALWAYS_INLINE blend(const Vectorized<i32>& a, const Vectorized<i32>& b) {
        constexpr u32 mask2 = (mask & 255) >> 4;
        constexpr u32 g0_2 = (mask2 & 1) * 0xffffffff;
        constexpr u32 g1_2 = ((mask2 & 2) >> 1) * 0xffffffff;
        constexpr u32 g2_2 = ((mask2 & 4) >> 2) * 0xffffffff;
        constexpr u32 g3_2 = ((mask2 & 8) >> 3) * 0xffffffff;

        const vbool32 mask_2nd = (vbool32){g0_2, g1_2, g2_2, g3_2};
        // generated masks
        return {b._vec0, (vint32)vec_sel(a._vec1, b._vec1, (vbool32)mask_2nd)};
      }

      template <u64 mask>
      static std::enable_if_t<
          (mask > 15 && ((mask & 255) != 255) && ((mask & 15) == 0)),
          Vectorized<i32>>
          C10_ALWAYS_INLINE blend(const Vectorized<i32>& a, const Vectorized<i32>& b) {
        constexpr u32 mask2 = (mask & 255) >> 4;
        constexpr u32 g0_2 = (mask2 & 1) * 0xffffffff;
        constexpr u32 g1_2 = ((mask2 & 2) >> 1) * 0xffffffff;
        constexpr u32 g2_2 = ((mask2 & 4) >> 2) * 0xffffffff;
        constexpr u32 g3_2 = ((mask2 & 8) >> 3) * 0xffffffff;

        const vbool32 mask_2nd = (vbool32){g0_2, g1_2, g2_2, g3_2};
        // generated masks
        return {a, (vint32)vec_sel(a._vec1, b._vec1, (vbool32)mask_2nd)};
      }

      template <u64 mask>
      static std::enable_if_t<
          (mask > 15 && ((mask & 255) != 255) && ((mask & 15) != 0) &&
           ((mask & 15) != 15)),
          Vectorized<i32>>
          C10_ALWAYS_INLINE blend(const Vectorized<i32>& a, const Vectorized<i32>& b) {
        constexpr u32 g0 = (mask & 1) * 0xffffffff;
        constexpr u32 g1 = ((mask & 2) >> 1) * 0xffffffff;
        constexpr u32 g2 = ((mask & 4) >> 2) * 0xffffffff;
        constexpr u32 g3 = ((mask & 8) >> 3) * 0xffffffff;
        constexpr u32 mask2 = (mask & 255) >> 4;
        constexpr u32 g0_2 = (mask2 & 1) * 0xffffffff;
        constexpr u32 g1_2 = ((mask2 & 2) >> 1) * 0xffffffff;
        constexpr u32 g2_2 = ((mask2 & 4) >> 2) * 0xffffffff;
        constexpr u32 g3_2 = ((mask2 & 8) >> 3) * 0xffffffff;

        const vbool32 mask_1st = (vbool32){g0, g1, g2, g3};
        const vbool32 mask_2nd = (vbool32){g0_2, g1_2, g2_2, g3_2};
        // generated masks
        return {
            (vint32)vec_sel(a._vec0, b._vec0, (vbool32)mask_1st),
            (vint32)vec_sel(a._vec1, b._vec1, (vbool32)mask_2nd)};
      }

      static Vectorized<i32> C10_ALWAYS_INLINE blendv(
          const Vectorized<i32>& a,
          const Vectorized<i32>& b,
          const Vectorized<i32>& mask) {
        // the mask used here returned by comparision of vec256
        // assuming this we can use the same mask directly with vec_sel
        // warning intel style mask will not work properly
        return {
            vec_sel(a._vec0, b._vec0, mask._vecb0),
            vec_sel(a._vec1, b._vec1, mask._vecb1)};
      }

      template <typename step_t>
      static Vectorized<i32> arange(i32 base = 0.f, step_t step = static_cast<step_t>(1)) {
        return Vectorized<i32>(
            base,
            base + step,
            base + 2 * step,
            base + 3 * step,
            base + 4 * step,
            base + 5 * step,
            base + 6 * step,
            base + 7 * step);
      }
      static Vectorized<i32> set(
          const Vectorized<i32>& a,
          const Vectorized<i32>& b,
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
          case 4:
            return blend<15>(a, b);
          case 5:
            return blend<31>(a, b);
          case 6:
            return blend<63>(a, b);
          case 7:
            return blend<127>(a, b);
        }

        return b;
      }
      static Vectorized<value_type> C10_ALWAYS_INLINE
      loadu(const void* ptr, int count = size()) {
        if (count == size()) {
          return {
              vec_vsx_ld(offset0, reinterpret_cast<const value_type*>(ptr)),
              vec_vsx_ld(offset16, reinterpret_cast<const value_type*>(ptr))};
        }

        __at_align32__ value_type tmp_values[size()];
        std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(value_type));

        return {vec_vsx_ld(offset0, tmp_values), vec_vsx_ld(offset16, tmp_values)};
      }
      void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
        if (count == size()) {
          vec_vsx_st(_vec0, offset0, reinterpret_cast<value_type*>(ptr));
          vec_vsx_st(_vec1, offset16, reinterpret_cast<value_type*>(ptr));
        } else if (count > 0) {
          __at_align32__ value_type tmp_values[size()];
          vec_vsx_st(_vec0, offset0, tmp_values);
          vec_vsx_st(_vec1, offset16, tmp_values);
          std::memcpy(
              ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
        }
      }
      const i32& operator[](int idx) const = delete;
      i32& operator[](int idx) = delete;

      Vectorized<i32> angle() const {
        return blendv(
          Vectorized<i32>(0), Vectorized<i32>(c10::pi<i32>), *this < Vectorized<i32>(0));
      }
      Vectorized<i32> real() const {
        return *this;
      }
      Vectorized<i32> imag() const {
        return Vectorized<i32>{0};
      }
      Vectorized<i32> conj() const {
        return *this;
      }

      Vectorized<i32> C10_ALWAYS_INLINE abs() const {
        return {vec_abs(_vec0), vec_abs(_vec1)};
      }

      Vectorized<i32> C10_ALWAYS_INLINE neg() const {
        return {vec_neg(_vec0), vec_neg(_vec1)};
      }

      DEFINE_MEMBER_UNARY_OP(operator~, i32, vec_not)
      DEFINE_MEMBER_OP(operator==, i32, vec_cmpeq)
      DEFINE_MEMBER_OP(operator!=, i32, vec_cmpne)
      DEFINE_MEMBER_OP(operator<, i32, vec_cmplt)
      DEFINE_MEMBER_OP(operator<=, i32, vec_cmple)
      DEFINE_MEMBER_OP(operator>, i32, vec_cmpgt)
      DEFINE_MEMBER_OP(operator>=, i32, vec_cmpge)
      DEFINE_MEMBER_OP_AND_ONE(eq, i32, vec_cmpeq)
      DEFINE_MEMBER_OP_AND_ONE(ne, i32, vec_cmpne)
      DEFINE_MEMBER_OP_AND_ONE(lt, i32, vec_cmplt)
      DEFINE_MEMBER_OP_AND_ONE(le, i32, vec_cmple)
      DEFINE_MEMBER_OP_AND_ONE(gt, i32, vec_cmpgt)
      DEFINE_MEMBER_OP_AND_ONE(ge, i32, vec_cmpge)
      DEFINE_MEMBER_OP(operator+, i32, vec_add)
      DEFINE_MEMBER_OP(operator-, i32, vec_sub)
      DEFINE_MEMBER_OP(operator*, i32, vec_mul)
      DEFINE_MEMBER_EMULATE_BINARY_OP(operator/, i32, /)
      DEFINE_MEMBER_OP(maximum, i32, vec_max)
      DEFINE_MEMBER_OP(minimum, i32, vec_min)
      DEFINE_MEMBER_OP(operator&, i32, vec_and)
      DEFINE_MEMBER_OP(operator|, i32, vec_or)
      DEFINE_MEMBER_OP(operator^, i32, vec_xor)
    };

    template <>
    Vectorized<i32> inline maximum(
        const Vectorized<i32>& a,
        const Vectorized<i32>& b) {
      return a.maximum(b);
    }

    template <>
    Vectorized<i32> inline minimum(
        const Vectorized<i32>& a,
        const Vectorized<i32>& b) {
      return a.minimum(b);
    }

    } // namespace
    } // namespace vec
    } // namespace at
    */
}

