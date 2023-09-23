// # vim: ft=none
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/vsx/vec256_int16_vsx.h]
crate::ix!();

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
    class Vectorized<i16> {
     private:
      union {
        struct {
          vint16 _vec0;
          vint16 _vec1;
        };
        struct {
          vbool16 _vecb0;
          vbool16 _vecb1;
        };

      } __attribute__((__may_alias__));

     public:
      using value_type = i16;
      using vec_internal_type = vint16;
      using vec_internal_mask_type = vbool16;
      using Sizeype = int;
      static constexpr Sizeype size() {
        return 16;
      }
      Vectorized() {}
      C10_ALWAYS_INLINE Vectorized(vint16 v) : _vec0{v}, _vec1{v} {}
      C10_ALWAYS_INLINE Vectorized(vbool16 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
      C10_ALWAYS_INLINE Vectorized(vint16 v1, vint16 v2) : _vec0{v1}, _vec1{v2} {}
      C10_ALWAYS_INLINE Vectorized(vbool16 v1, vbool16 v2) : _vecb0{v1}, _vecb1{v2} {}
      C10_ALWAYS_INLINE Vectorized(i16 scalar)
          : _vec0{vec_splats(scalar)}, _vec1{vec_splats(scalar)} {}

      C10_ALWAYS_INLINE Vectorized(
          i16 scalar1,
          i16 scalar2,
          i16 scalar3,
          i16 scalar4,
          i16 scalar5,
          i16 scalar6,
          i16 scalar7,
          i16 scalar8,
          i16 scalar9,
          i16 scalar10,
          i16 scalar11,
          i16 scalar12,
          i16 scalar13,
          i16 scalar14,
          i16 scalar15,
          i16 scalar16)
          : _vec0{vint16{
                scalar1,
                scalar2,
                scalar3,
                scalar4,
                scalar5,
                scalar6,
                scalar7,
                scalar8}},
            _vec1{vint16{
                scalar9,
                scalar10,
                scalar11,
                scalar12,
                scalar13,
                scalar14,
                scalar15,
                scalar16}} {}
      C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
        return _vec0;
      }
      C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
        return _vec1;
      }

      template <u64 mask>
      static std::enable_if_t<mask == 0, Vectorized<i16>> C10_ALWAYS_INLINE
      blend(const Vectorized<i16>& a, const Vectorized<i16>& b) {
        return a;
      }

      template <u64 mask>
      static std::enable_if_t<(mask & 65535) == 65535, Vectorized<i16>>
          C10_ALWAYS_INLINE blend(const Vectorized<i16>& a, const Vectorized<i16>& b) {
        return b;
      }

      template <u64 mask>
      static std::enable_if_t<mask == 255, Vectorized<i16>> C10_ALWAYS_INLINE
      blend(const Vectorized<i16>& a, const Vectorized<i16>& b) {
        return {b._vec0, a._vec1};
      }

      template <u64 mask>
      static std::enable_if_t<(mask > 0 && mask < 255), Vectorized<i16>>
          C10_ALWAYS_INLINE blend(const Vectorized<i16>& a, const Vectorized<i16>& b) {
        constexpr i16 g0 = (mask & 1) * 0xffff;
        constexpr i16 g1 = ((mask & 2) >> 1) * 0xffff;
        constexpr i16 g2 = ((mask & 4) >> 2) * 0xffff;
        constexpr i16 g3 = ((mask & 8) >> 3) * 0xffff;
        constexpr i16 g4 = ((mask & 16) >> 4) * 0xffff;
        constexpr i16 g5 = ((mask & 32) >> 5) * 0xffff;
        constexpr i16 g6 = ((mask & 64) >> 6) * 0xffff;
        constexpr i16 g7 = ((mask & 128) >> 7) * 0xffff;
        const vint16 mask_1st = vint16{g0, g1, g2, g3, g4, g5, g6, g7};

        return {(vint16)vec_sel(a._vec0, b._vec0, (vbool16)mask_1st), a._vec1};
      }

      template <u64 mask>
      static std::enable_if_t<
          (mask > 255 && (mask & 65535) != 65535 && ((mask & 255) == 255)),
          Vectorized<i16>>
          C10_ALWAYS_INLINE blend(const Vectorized<i16>& a, const Vectorized<i16>& b) {
        constexpr i16 g0_2 = (mask & 1) * 0xffff;
        constexpr i16 g1_2 = ((mask & 2) >> 1) * 0xffff;
        constexpr i16 g2_2 = ((mask & 4) >> 2) * 0xffff;
        constexpr i16 g3_2 = ((mask & 8) >> 3) * 0xffff;
        constexpr i16 g4_2 = ((mask & 16) >> 4) * 0xffff;
        constexpr i16 g5_2 = ((mask & 32) >> 5) * 0xffff;
        constexpr i16 g6_2 = ((mask & 64) >> 6) * 0xffff;
        constexpr i16 g7_2 = ((mask & 128) >> 7) * 0xffff;

        const vint16 mask_2nd =
            vint16{g0_2, g1_2, g2_2, g3_2, g4_2, g5_2, g6_2, g7_2};
        // generated masks
        return {b._vec0, (vint16)vec_sel(a._vec1, b._vec1, (vbool16)mask_2nd)};
      }

      template <u64 mask>
      static std::enable_if_t<
          (mask > 255 && ((mask & 65535) != 65535) && ((mask & 255) == 0)),
          Vectorized<i16>>
          C10_ALWAYS_INLINE blend(const Vectorized<i16>& a, const Vectorized<i16>& b) {
        constexpr i16 mask2 = (mask & 65535) >> 16;
        constexpr i16 g0_2 = (mask & 1) * 0xffff;
        constexpr i16 g1_2 = ((mask & 2) >> 1) * 0xffff;
        constexpr i16 g2_2 = ((mask & 4) >> 2) * 0xffff;
        constexpr i16 g3_2 = ((mask & 8) >> 3) * 0xffff;
        constexpr i16 g4_2 = ((mask & 16) >> 4) * 0xffff;
        constexpr i16 g5_2 = ((mask & 32) >> 5) * 0xffff;
        constexpr i16 g6_2 = ((mask & 64) >> 6) * 0xffff;
        constexpr i16 g7_2 = ((mask & 128) >> 7) * 0xffff;

        const vint16 mask_2nd =
            vint16{g0_2, g1_2, g2_2, g3_2, g4_2, g5_2, g6_2, g7_2};
        // generated masks
        return {a, (vint16)vec_sel(a._vec1, b._vec1, (vbool16)mask_2nd)};
      }

      template <u64 mask>
      static std::enable_if_t<
          (mask > 255 && ((mask & 65535) != 65535) && ((mask & 255) != 0) &&
           ((mask & 255) != 255)),
          Vectorized<i16>>
          C10_ALWAYS_INLINE blend(const Vectorized<i16>& a, const Vectorized<i16>& b) {
        constexpr i16 g0 = (mask & 1) * 0xffff;
        constexpr i16 g1 = ((mask & 2) >> 1) * 0xffff;
        constexpr i16 g2 = ((mask & 4) >> 2) * 0xffff;
        constexpr i16 g3 = ((mask & 8) >> 3) * 0xffff;
        constexpr i16 g4 = ((mask & 16) >> 4) * 0xffff;
        constexpr i16 g5 = ((mask & 32) >> 5) * 0xffff;
        constexpr i16 g6 = ((mask & 64) >> 6) * 0xffff;
        constexpr i16 g7 = ((mask & 128) >> 7) * 0xffff;
        constexpr i16 mask2 = (mask & 65535) >> 16;
        constexpr i16 g0_2 = (mask & 1) * 0xffff;
        constexpr i16 g1_2 = ((mask & 2) >> 1) * 0xffff;
        constexpr i16 g2_2 = ((mask & 4) >> 2) * 0xffff;
        constexpr i16 g3_2 = ((mask & 8) >> 3) * 0xffff;
        constexpr i16 g4_2 = ((mask & 16) >> 4) * 0xffff;
        constexpr i16 g5_2 = ((mask & 32) >> 5) * 0xffff;
        constexpr i16 g6_2 = ((mask & 64) >> 6) * 0xffff;
        constexpr i16 g7_2 = ((mask & 128) >> 7) * 0xffff;

        const vint16 mask_1st = vint16{g0, g1, g2, g3, g4, g5, g6, g7};
        const vint16 mask_2nd =
            vint16{g0_2, g1_2, g2_2, g3_2, g4_2, g5_2, g6_2, g7_2};
        // generated masks
        return {
            (vint16)vec_sel(a._vec0, b._vec0, (vbool16)mask_1st),
            (vint16)vec_sel(a._vec1, b._vec1, (vbool16)mask_2nd)};
      }

      static Vectorized<i16> C10_ALWAYS_INLINE blendv(
          const Vectorized<i16>& a,
          const Vectorized<i16>& b,
          const Vectorized<i16>& mask) {
        // the mask used here returned by comparision of vec256
        // assuming this we can use the same mask directly with vec_sel
        // warning intel style mask will not work properly
        return {
            vec_sel(a._vec0, b._vec0, mask._vecb0),
            vec_sel(a._vec1, b._vec1, mask._vecb1)};
      }

      template <typename step_t>
      static Vectorized<i16> arange(i16 base = 0, step_t step = static_cast<step_t>(1)) {
        return Vectorized<i16>(
            base,
            base + step,
            base + 2 * step,
            base + 3 * step,
            base + 4 * step,
            base + 5 * step,
            base + 6 * step,
            base + 7 * step,
            base + 8 * step,
            base + 9 * step,
            base + 10 * step,
            base + 11 * step,
            base + 12 * step,
            base + 13 * step,
            base + 14 * step,
            base + 15 * step);
      }
      static Vectorized<i16> set(
          const Vectorized<i16>& a,
          const Vectorized<i16>& b,
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
          case 8:
            return blend<255>(a, b);
          case 9:
            return blend<511>(a, b);
          case 10:
            return blend<1023>(a, b);
          case 11:
            return blend<2047>(a, b);
          case 12:
            return blend<4095>(a, b);
          case 13:
            return blend<8191>(a, b);
          case 14:
            return blend<16383>(a, b);
          case 15:
            return blend<32767>(a, b);
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
          std::memcpy(ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
        }
      }
      const i16& operator[](int idx) const = delete;
      i16& operator[](int idx) = delete;

      Vectorized<i16> angle() const {
        return blendv(
          Vectorized<i16>(0), Vectorized<i16>(c10::pi<i16>), *this < Vectorized<i16>(0));
      }
      Vectorized<i16> real() const {
        return *this;
      }
      Vectorized<i16> imag() const {
        return Vectorized<i16>{0};
      }
      Vectorized<i16> conj() const {
        return *this;
      }

      Vectorized<i16> C10_ALWAYS_INLINE abs() const {
        return {vec_abs(_vec0), vec_abs(_vec1)};
      }

      Vectorized<i16> C10_ALWAYS_INLINE neg() const {
        return {vec_neg(_vec0), vec_neg(_vec1)};
      }

      DEFINE_MEMBER_UNARY_OP(operator~, i16, vec_not)
      DEFINE_MEMBER_OP(operator==, i16, vec_cmpeq)
      DEFINE_MEMBER_OP(operator!=, i16, vec_cmpne)
      DEFINE_MEMBER_OP(operator<, i16, vec_cmplt)
      DEFINE_MEMBER_OP(operator<=, i16, vec_cmple)
      DEFINE_MEMBER_OP(operator>, i16, vec_cmpgt)
      DEFINE_MEMBER_OP(operator>=, i16, vec_cmpge)
      DEFINE_MEMBER_OP_AND_ONE(eq, i16, vec_cmpeq)
      DEFINE_MEMBER_OP_AND_ONE(ne, i16, vec_cmpne)
      DEFINE_MEMBER_OP_AND_ONE(lt, i16, vec_cmplt)
      DEFINE_MEMBER_OP_AND_ONE(le, i16, vec_cmple)
      DEFINE_MEMBER_OP_AND_ONE(gt, i16, vec_cmpgt)
      DEFINE_MEMBER_OP_AND_ONE(ge, i16, vec_cmpge)
      DEFINE_MEMBER_OP(operator+, i16, vec_add)
      DEFINE_MEMBER_OP(operator-, i16, vec_sub)
      DEFINE_MEMBER_OP(operator*, i16, vec_mul)
      DEFINE_MEMBER_EMULATE_BINARY_OP(operator/, i16, /)
      DEFINE_MEMBER_OP(maximum, i16, vec_max)
      DEFINE_MEMBER_OP(minimum, i16, vec_min)
      DEFINE_MEMBER_OP(operator&, i16, vec_and)
      DEFINE_MEMBER_OP(operator|, i16, vec_or)
      DEFINE_MEMBER_OP(operator^, i16, vec_xor)
    };

    template <>
    Vectorized<i16> inline maximum(
        const Vectorized<i16>& a,
        const Vectorized<i16>& b) {
      return a.maximum(b);
    }

    template <>
    Vectorized<i16> inline minimum(
        const Vectorized<i16>& a,
        const Vectorized<i16>& b) {
      return a.minimum(b);
    }


    } // namespace
    } // namespace vec
    } // namespace at
    */
}

