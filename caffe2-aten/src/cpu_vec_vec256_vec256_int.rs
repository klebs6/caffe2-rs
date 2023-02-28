crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/vec256_int.h]

#[cfg(target_feature = "avx2")]
#[derive(Default)]
pub struct Vectorizedi {
    values: __m256i,
}

#[cfg(target_feature = "avx2")]
impl Vectorizedi {
    
    #[inline] pub fn invert(v: &__m256i) -> __m256i {
        
        todo!();
        /*
            const auto ones = _mm256_set1_epi64x(-1);
        return _mm256_xor_si256(ones, v);
        */
    }
    
    pub fn new(v: __m256i) -> Self {
    
        todo!();
        /*
        : values(v),

        
        */
    }
    
    pub fn operator_m_256i(&self) -> __m256i {
        
        todo!();
        /*
            return values;
        */
    }
}

/**
  | dummy definition to make Vectorizedi
  | always defined
  |
  */
#[cfg(not(target_feature = "avx2"))]
pub struct Vectorizedi {}  

lazy_static!{
    /*
    #ifdef target_feature = "avx2"

    class VectorizedInt64_t : public Vectorizedi {

      static const Vectorized<i64> ones;

      using value_type = i64;
      using Sizeype = int;
      static constexpr Sizeype size() {
        return 4;
      }
      using Vectorizedi::Vectorizedi;
      Vectorized() {}
      Vectorized(i64 v) { values = _mm256_set1_epi64x(v); }
      Vectorized(i64 val1, i64 val2, i64 val3, i64 val4) {
        values = _mm256_setr_epi64x(val1, val2, val3, val4);
      }
      template <i64 mask>
      static Vectorized<i64> blend(Vectorized<i64> a, Vectorized<i64> b) {
        __at_align32__ i64 tmp_values[size()];
        a.store(tmp_values);
        if (mask & 0x01)
          tmp_values[0] = _mm256_extract_epi64(b.values, 0);
        if (mask & 0x02)
          tmp_values[1] = _mm256_extract_epi64(b.values, 1);
        if (mask & 0x04)
          tmp_values[2] = _mm256_extract_epi64(b.values, 2);
        if (mask & 0x08)
          tmp_values[3] = _mm256_extract_epi64(b.values, 3);
        return loadu(tmp_values);
      }
      static Vectorized<i64> blendv(const Vectorized<i64>& a, const Vectorized<i64>& b,
                                    const Vectorized<i64>& mask) {
        return _mm256_blendv_epi8(a.values, b.values, mask.values);
      }
      template <typename step_t>
      static Vectorized<i64> arange(i64 base = 0, step_t step = static_cast<step_t>(1)) {
        return Vectorized<i64>(base, base + step, base + 2 * step, base + 3 * step);
      }
      static Vectorized<i64>
      set(Vectorized<i64> a, Vectorized<i64> b, i64 count = size()) {
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
      static Vectorized<i64> loadu(const void* ptr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
      }
      static Vectorized<i64> loadu(const void* ptr, i64 count) {
        __at_align32__ i64 tmp_values[size()];
        // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
        // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
        // instructions while a loop would be compiled to one instruction.
        for (auto i = 0; i < size(); ++i) {
          tmp_values[i] = 0;
        }
        memcpy(tmp_values, ptr, count * sizeof(i64));
        return loadu(tmp_values);
      }
      void store(void* ptr, int count = size()) const {
        if (count == size()) {
          // ptr need not to be aligned here. See
          // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm256-storeu-si256.html
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
        } else if (count > 0) {
          __at_align32__ i64 tmp_values[size()];
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
          memcpy(ptr, tmp_values, count * sizeof(i64));
        }
      }
      const i64& operator[](int idx) const  = delete;
      i64& operator[](int idx)  = delete;
      Vectorized<i64> abs() const {
        auto zero = _mm256_set1_epi64x(0);
        auto is_larger = _mm256_cmpgt_epi64(zero, values);
        auto inverse = _mm256_xor_si256(values, is_larger);
        return _mm256_sub_epi64(inverse, is_larger);
      }
      Vectorized<i64> real() const {
        return *this;
      }
      Vectorized<i64> imag() const {
        return _mm256_set1_epi64x(0);
      }
      Vectorized<i64> conj() const {
        return *this;
      }
      Vectorized<i64> frac() const;
      Vectorized<i64> neg() const;
      Vectorized<i64> operator==(const Vectorized<i64>& other) const {
        return _mm256_cmpeq_epi64(values, other.values);
      }
      Vectorized<i64> operator!=(const Vectorized<i64>& other) const {
        return invert(_mm256_cmpeq_epi64(values, other.values));
      }
      Vectorized<i64> operator<(const Vectorized<i64>& other) const {
        return _mm256_cmpgt_epi64(other.values, values);
      }
      Vectorized<i64> operator<=(const Vectorized<i64>& other) const {
        return invert(_mm256_cmpgt_epi64(values, other.values));
      }
      Vectorized<i64> operator>(const Vectorized<i64>& other) const {
        return _mm256_cmpgt_epi64(values, other.values);
      }
      Vectorized<i64> operator>=(const Vectorized<i64>& other) const {
        return invert(_mm256_cmpgt_epi64(other.values, values));
      }

      Vectorized<i64> eq(const Vectorized<i64>& other) const;
      Vectorized<i64> ne(const Vectorized<i64>& other) const;
      Vectorized<i64> gt(const Vectorized<i64>& other) const;
      Vectorized<i64> ge(const Vectorized<i64>& other) const;
      Vectorized<i64> lt(const Vectorized<i64>& other) const;
      Vectorized<i64> le(const Vectorized<i64>& other) const;
    };

    class VectorizedInt32_t : public Vectorizedi {

      static const Vectorized<i32> ones;

      using value_type = i32;
      static constexpr int size() {
        return 8;
      }
      using Vectorizedi::Vectorizedi;
      Vectorized() {}
      Vectorized(i32 v) { values = _mm256_set1_epi32(v); }
      Vectorized(i32 val1, i32 val2, i32 val3, i32 val4,
             i32 val5, i32 val6, i32 val7, i32 val8) {
        values = _mm256_setr_epi32(val1, val2, val3, val4, val5, val6, val7, val8);
      }
      template <i64 mask>
      static Vectorized<i32> blend(Vectorized<i32> a, Vectorized<i32> b) {
        return _mm256_blend_epi32(a, b, mask);
      }
      static Vectorized<i32> blendv(const Vectorized<i32>& a, const Vectorized<i32>& b,
                                    const Vectorized<i32>& mask) {
        return _mm256_blendv_epi8(a.values, b.values, mask.values);
      }
      template <typename step_t>
      static Vectorized<i32> arange(i32 base = 0, step_t step = static_cast<step_t>(1)) {
        return Vectorized<i32>(
          base,            base +     step, base + 2 * step, base + 3 * step,
          base + 4 * step, base + 5 * step, base + 6 * step, base + 7 * step);
      }
      static Vectorized<i32>
      set(Vectorized<i32> a, Vectorized<i32> b, i32 count = size()) {
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
      static Vectorized<i32> loadu(const void* ptr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
      }
      static Vectorized<i32> loadu(const void* ptr, i32 count) {
        __at_align32__ i32 tmp_values[size()];
        // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
        // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
        // instructions while a loop would be compiled to one instruction.
        for (auto i = 0; i < size(); ++i) {
          tmp_values[i] = 0;
        }
        memcpy(tmp_values, ptr, count * sizeof(i32));
        return loadu(tmp_values);
      }
      void store(void* ptr, int count = size()) const {
        if (count == size()) {
          // ptr need not to be aligned here. See
          // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm256-storeu-si256.html
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
        } else if (count > 0) {
          __at_align32__ i32 tmp_values[size()];
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
          memcpy(ptr, tmp_values, count * sizeof(i32));
        }
      }
      void dump() const {
          for (usize i = 0; i < size(); ++i) {
              cout << (int)((value_type*)&values)[i] << " ";
          }
          cout << endl;
      }
      const i32& operator[](int idx) const  = delete;
      i32& operator[](int idx)  = delete;
      Vectorized<i32> abs() const {
        return _mm256_abs_epi32(values);
      }
      Vectorized<i32> real() const {
        return *this;
      }
      Vectorized<i32> imag() const {
        return _mm256_set1_epi32(0);
      }
      Vectorized<i32> conj() const {
        return *this;
      }
      Vectorized<i32> frac() const;
      Vectorized<i32> neg() const;
      Vectorized<i32> operator==(const Vectorized<i32>& other) const {
        return _mm256_cmpeq_epi32(values, other.values);
      }
      Vectorized<i32> operator!=(const Vectorized<i32>& other) const {
        return invert(_mm256_cmpeq_epi32(values, other.values));
      }
      Vectorized<i32> operator<(const Vectorized<i32>& other) const {
        return _mm256_cmpgt_epi32(other.values, values);
      }
      Vectorized<i32> operator<=(const Vectorized<i32>& other) const {
        return invert(_mm256_cmpgt_epi32(values, other.values));
      }
      Vectorized<i32> operator>(const Vectorized<i32>& other) const {
        return _mm256_cmpgt_epi32(values, other.values);
      }
      Vectorized<i32> operator>=(const Vectorized<i32>& other) const {
        return invert(_mm256_cmpgt_epi32(other.values, values));
      }
      Vectorized<i32> eq(const Vectorized<i32>& other) const;
      Vectorized<i32> ne(const Vectorized<i32>& other) const;
      Vectorized<i32> gt(const Vectorized<i32>& other) const;
      Vectorized<i32> ge(const Vectorized<i32>& other) const;
      Vectorized<i32> lt(const Vectorized<i32>& other) const;
      Vectorized<i32> le(const Vectorized<i32>& other) const;
    };

    template <>
    inline void convert(const i32 *src, float *dst, i64 n) {
      i64 i;
      // i32 and float have same size
    #ifndef _MSC_VER
    # pragma unroll
    #endif
      for (i = 0; i <= (n - Vectorized<i32>::size()); i += Vectorized<i32>::size()) {
        auto input_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        auto output_vec = _mm256_cvtepi32_ps(input_vec);
        _mm256_storeu_ps(reinterpret_cast<float*>(dst + i), output_vec);
      }
    #ifndef _MSC_VER
    # pragma unroll
    #endif
      for (; i < n; i++) {
        dst[i] = static_cast<float>(src[i]);
      }
    }

    template <>
    inline void convert(const i32 *src, double *dst, i64 n) {
      i64 i;
      // i32 has half the size of double
    #ifndef _MSC_VER
    # pragma unroll
    #endif
      for (i = 0; i <= (n - Vectorized<double>::size()); i += Vectorized<double>::size()) {
        auto input_128_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        auto output_vec = _mm256_cvtepi32_pd(input_128_vec);
        _mm256_storeu_pd(reinterpret_cast<double*>(dst + i), output_vec);
      }
    #ifndef _MSC_VER
    # pragma unroll
    #endif
      for (; i < n; i++) {
        dst[i] = static_cast<double>(src[i]);
      }
    }

    template <>
    class Vectorized<i16> : public Vectorizedi {

      static const Vectorized<i16> ones;

      using value_type = i16;
      static constexpr int size() {
        return 16;
      }
      using Vectorizedi::Vectorizedi;
      Vectorized() {}
      Vectorized(i16 v) { values = _mm256_set1_epi16(v); }
      Vectorized(i16 val1, i16 val2, i16 val3, i16 val4,
             i16 val5, i16 val6, i16 val7, i16 val8,
             i16 val9, i16 val10, i16 val11, i16 val12,
             i16 val13, i16 val14, i16 val15, i16 val16) {
        values = _mm256_setr_epi16(val1, val2, val3, val4, val5, val6, val7, val8,
                                   val9, val10, val11, val12, val13, val14, val15, val16);
      }
      template <i64 mask>
      static Vectorized<i16> blend(Vectorized<i16> a, Vectorized<i16> b) {
        __at_align32__ i16 tmp_values[size()];
        a.store(tmp_values);
        if (mask & 0x01)
          tmp_values[0] = _mm256_extract_epi16(b.values, 0);
        if (mask & 0x02)
          tmp_values[1] = _mm256_extract_epi16(b.values, 1);
        if (mask & 0x04)
          tmp_values[2] = _mm256_extract_epi16(b.values, 2);
        if (mask & 0x08)
          tmp_values[3] = _mm256_extract_epi16(b.values, 3);
        if (mask & 0x10)
          tmp_values[4] = _mm256_extract_epi16(b.values, 4);
        if (mask & 0x20)
          tmp_values[5] = _mm256_extract_epi16(b.values, 5);
        if (mask & 0x40)
          tmp_values[6] = _mm256_extract_epi16(b.values, 6);
        if (mask & 0x80)
          tmp_values[7] = _mm256_extract_epi16(b.values, 7);
        if (mask & 0x100)
          tmp_values[8] = _mm256_extract_epi16(b.values, 8);
        if (mask & 0x200)
          tmp_values[9] = _mm256_extract_epi16(b.values, 9);
        if (mask & 0x400)
          tmp_values[10] = _mm256_extract_epi16(b.values, 10);
        if (mask & 0x800)
          tmp_values[11] = _mm256_extract_epi16(b.values, 11);
        if (mask & 0x1000)
          tmp_values[12] = _mm256_extract_epi16(b.values, 12);
        if (mask & 0x2000)
          tmp_values[13] = _mm256_extract_epi16(b.values, 13);
        if (mask & 0x4000)
          tmp_values[14] = _mm256_extract_epi16(b.values, 14);
        if (mask & 0x8000)
          tmp_values[15] = _mm256_extract_epi16(b.values, 15);
        return loadu(tmp_values);
      }
      static Vectorized<i16> blendv(const Vectorized<i16>& a, const Vectorized<i16>& b,
                                    const Vectorized<i16>& mask) {
        return _mm256_blendv_epi8(a.values, b.values, mask.values);
      }
      template <typename step_t>
      static Vectorized<i16> arange(i16 base = 0, step_t step = static_cast<step_t>(1)) {
        return Vectorized<i16>(
          base,             base +      step, base +  2 * step, base +  3 * step,
          base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
          base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
          base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step);
      }
      static Vectorized<i16>
      set(Vectorized<i16> a, Vectorized<i16> b, i16 count = size()) {
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
      static Vectorized<i16> loadu(const void* ptr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
      }
      static Vectorized<i16> loadu(const void* ptr, i16 count) {
        __at_align32__ i16 tmp_values[size()];
        // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
        // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
        // instructions while a loop would be compiled to one instruction.
        for (auto i = 0; i < size(); ++i) {
          tmp_values[i] = 0;
        }
        memcpy(tmp_values, ptr, count * sizeof(i16));
        return loadu(tmp_values);
      }
      void store(void* ptr, int count = size()) const {
        if (count == size()) {
          // ptr need not to be aligned here. See
          // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm256-storeu-si256.html
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
        } else if (count > 0) {
          __at_align32__ i16 tmp_values[size()];
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
          memcpy(ptr, tmp_values, count * sizeof(i16));
        }
      }
      const i16& operator[](int idx) const  = delete;
      i16& operator[](int idx)  = delete;
      Vectorized<i16> abs() const {
        return _mm256_abs_epi16(values);
      }
      Vectorized<i16> real() const {
        return *this;
      }
      Vectorized<i16> imag() const {
        return _mm256_set1_epi16(0);
      }
      Vectorized<i16> conj() const {
        return *this;
      }
      Vectorized<i16> frac() const;
      Vectorized<i16> neg() const;
      Vectorized<i16> operator==(const Vectorized<i16>& other) const {
        return _mm256_cmpeq_epi16(values, other.values);
      }
      Vectorized<i16> operator!=(const Vectorized<i16>& other) const {
        return invert(_mm256_cmpeq_epi16(values, other.values));
      }
      Vectorized<i16> operator<(const Vectorized<i16>& other) const {
        return _mm256_cmpgt_epi16(other.values, values);
      }
      Vectorized<i16> operator<=(const Vectorized<i16>& other) const {
        return invert(_mm256_cmpgt_epi16(values, other.values));
      }
      Vectorized<i16> operator>(const Vectorized<i16>& other) const {
        return _mm256_cmpgt_epi16(values, other.values);
      }
      Vectorized<i16> operator>=(const Vectorized<i16>& other) const {
        return invert(_mm256_cmpgt_epi16(other.values, values));
      }

      Vectorized<i16> eq(const Vectorized<i16>& other) const;
      Vectorized<i16> ne(const Vectorized<i16>& other) const;
      Vectorized<i16> gt(const Vectorized<i16>& other) const;
      Vectorized<i16> ge(const Vectorized<i16>& other) const;
      Vectorized<i16> lt(const Vectorized<i16>& other) const;
      Vectorized<i16> le(const Vectorized<i16>& other) const;
    };

    template <>
    class Vectorized<i8> : public Vectorizedi {

      static const Vectorized<i8> ones;

      using value_type = i8;
      static constexpr int size() {
        return 32;
      }
      using Vectorizedi::Vectorizedi;
      Vectorized() {}
      Vectorized(i8 v) { values = _mm256_set1_epi8(v); }
      Vectorized(i8 val1, i8 val2, i8 val3, i8 val4,
             i8 val5, i8 val6, i8 val7, i8 val8,
             i8 val9, i8 val10, i8 val11, i8 val12,
             i8 val13, i8 val14, i8 val15, i8 val16,
             i8 val17, i8 val18, i8 val19, i8 val20,
             i8 val21, i8 val22, i8 val23, i8 val24,
             i8 val25, i8 val26, i8 val27, i8 val28,
             i8 val29, i8 val30, i8 val31, i8 val32) {
        values = _mm256_setr_epi8(val1, val2, val3, val4, val5, val6, val7, val8,
                                  val9, val10, val11, val12, val13, val14, val15, val16,
                                  val17, val18, val19, val20, val21, val22, val23, val24,
                                  val25, val26, val27, val28, val29, val30, val31, val32);
      }
      template <i64 mask>
      static Vectorized<i8> blend(Vectorized<i8> a, Vectorized<i8> b) {
        __at_align32__ i8 tmp_values[size()];
        a.store(tmp_values);
        if (mask & 0x01)
          tmp_values[0] = _mm256_extract_epi8(b.values, 0);
        if (mask & 0x02)
          tmp_values[1] = _mm256_extract_epi8(b.values, 1);
        if (mask & 0x04)
          tmp_values[2] = _mm256_extract_epi8(b.values, 2);
        if (mask & 0x08)
          tmp_values[3] = _mm256_extract_epi8(b.values, 3);
        if (mask & 0x10)
          tmp_values[4] = _mm256_extract_epi8(b.values, 4);
        if (mask & 0x20)
          tmp_values[5] = _mm256_extract_epi8(b.values, 5);
        if (mask & 0x40)
          tmp_values[6] = _mm256_extract_epi8(b.values, 6);
        if (mask & 0x80)
          tmp_values[7] = _mm256_extract_epi8(b.values, 7);
        if (mask & 0x100)
          tmp_values[8] = _mm256_extract_epi8(b.values, 8);
        if (mask & 0x200)
          tmp_values[9] = _mm256_extract_epi8(b.values, 9);
        if (mask & 0x400)
          tmp_values[10] = _mm256_extract_epi8(b.values, 10);
        if (mask & 0x800)
          tmp_values[11] = _mm256_extract_epi8(b.values, 11);
        if (mask & 0x1000)
          tmp_values[12] = _mm256_extract_epi8(b.values, 12);
        if (mask & 0x2000)
          tmp_values[13] = _mm256_extract_epi8(b.values, 13);
        if (mask & 0x4000)
          tmp_values[14] = _mm256_extract_epi8(b.values, 14);
        if (mask & 0x8000)
          tmp_values[15] = _mm256_extract_epi8(b.values, 15);
        if (mask & 0x010000)
          tmp_values[16] = _mm256_extract_epi8(b.values, 16);
        if (mask & 0x020000)
          tmp_values[17] = _mm256_extract_epi8(b.values, 17);
        if (mask & 0x040000)
          tmp_values[18] = _mm256_extract_epi8(b.values, 18);
        if (mask & 0x080000)
          tmp_values[19] = _mm256_extract_epi8(b.values, 19);
        if (mask & 0x100000)
          tmp_values[20] = _mm256_extract_epi8(b.values, 20);
        if (mask & 0x200000)
          tmp_values[21] = _mm256_extract_epi8(b.values, 21);
        if (mask & 0x400000)
          tmp_values[22] = _mm256_extract_epi8(b.values, 22);
        if (mask & 0x800000)
          tmp_values[23] = _mm256_extract_epi8(b.values, 23);
        if (mask & 0x1000000)
          tmp_values[24] = _mm256_extract_epi8(b.values, 24);
        if (mask & 0x2000000)
          tmp_values[25] = _mm256_extract_epi8(b.values, 25);
        if (mask & 0x4000000)
          tmp_values[26] = _mm256_extract_epi8(b.values, 26);
        if (mask & 0x8000000)
          tmp_values[27] = _mm256_extract_epi8(b.values, 27);
        if (mask & 0x10000000)
          tmp_values[28] = _mm256_extract_epi8(b.values, 28);
        if (mask & 0x20000000)
          tmp_values[29] = _mm256_extract_epi8(b.values, 29);
        if (mask & 0x40000000)
          tmp_values[30] = _mm256_extract_epi8(b.values, 30);
        if (mask & 0x80000000)
          tmp_values[31] = _mm256_extract_epi8(b.values, 31);
        return loadu(tmp_values);
      }
      static Vectorized<i8> blendv(const Vectorized<i8>& a, const Vectorized<i8>& b,
                                   const Vectorized<i8>& mask) {
        return _mm256_blendv_epi8(a.values, b.values, mask.values);
      }
      template <typename step_t>
      static Vectorized<i8> arange(i8 base = 0, step_t step = static_cast<step_t>(1)) {
        return Vectorized<i8>(
          base,             base +      step, base +  2 * step, base +  3 * step,
          base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
          base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
          base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step,
          base + 16 * step, base + 17 * step, base + 18 * step, base + 19 * step,
          base + 20 * step, base + 21 * step, base + 22 * step, base + 23 * step,
          base + 24 * step, base + 25 * step, base + 26 * step, base + 27 * step,
          base + 28 * step, base + 29 * step, base + 30 * step, base + 31 * step);
      }
      static Vectorized<i8>
      set(Vectorized<i8> a, Vectorized<i8> b, i8 count = size()) {
        switch (count) {
          case 0:
            return a;
          case 1:
            return blend<0x1>(a, b);
          case 2:
            return blend<0x3>(a, b);
          case 3:
            return blend<0x7>(a, b);
          case 4:
            return blend<0xF>(a, b);
          case 5:
            return blend<0x1F>(a, b);
          case 6:
            return blend<0x3F>(a, b);
          case 7:
            return blend<0x7F>(a, b);
          case 8:
            return blend<0xFF>(a, b);
          case 9:
            return blend<0x1FF>(a, b);
          case 10:
            return blend<0x3FF>(a, b);
          case 11:
            return blend<0x7FF>(a, b);
          case 12:
            return blend<0xFFF>(a, b);
          case 13:
            return blend<0x1FFF>(a, b);
          case 14:
            return blend<0x3FFF>(a, b);
          case 15:
            return blend<0x7FFF>(a, b);
          case 16:
            return blend<0xFFFF>(a, b);
          case 17:
            return blend<0x1FFFF>(a, b);
          case 18:
            return blend<0x3FFFF>(a, b);
          case 19:
            return blend<0x7FFFF>(a, b);
          case 20:
            return blend<0xFFFFF>(a, b);
          case 21:
            return blend<0x1FFFFF>(a, b);
          case 22:
            return blend<0x3FFFFF>(a, b);
          case 23:
            return blend<0x7FFFFF>(a, b);
          case 24:
            return blend<0xFFFFFF>(a, b);
          case 25:
            return blend<0x1FFFFFF>(a, b);
          case 26:
            return blend<0x3FFFFFF>(a, b);
          case 27:
            return blend<0x7FFFFFF>(a, b);
          case 28:
            return blend<0xFFFFFFF>(a, b);
          case 29:
            return blend<0x1FFFFFFF>(a, b);
          case 30:
            return blend<0x3FFFFFFF>(a, b);
          case 31:
            return blend<0x7FFFFFFF>(a, b);
        }
        return b;
      }
      static Vectorized<i8> loadu(const void* ptr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
      }
      static Vectorized<i8> loadu(const void* ptr, i8 count) {
        __at_align32__ i8 tmp_values[size()];
        // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
        // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
        // instructions while a loop would be compiled to one instruction.
        for (usize i = 0; i < size(); ++i) {
          tmp_values[i] = 0;
        }
        memcpy(tmp_values, ptr, count * sizeof(i8));
        return loadu(tmp_values);
      }
      void store(void* ptr, int count = size()) const {
        if (count == size()) {
          // ptr need not to be aligned here. See
          // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm256-storeu-si256.html
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
        } else if (count > 0) {
          __at_align32__ i8 tmp_values[size()];
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
          memcpy(ptr, tmp_values, count * sizeof(i8));
        }
      }
      const i8& operator[](int idx) const  = delete;
      i8& operator[](int idx)  = delete;
      Vectorized<i8> abs() const {
        return _mm256_abs_epi8(values);
      }
      Vectorized<i8> real() const {
        return *this;
      }
      Vectorized<i8> imag() const {
        return _mm256_set1_epi8(0);
      }
      Vectorized<i8> conj() const {
        return *this;
      }
      Vectorized<i8> frac() const;
      Vectorized<i8> neg() const;
      Vectorized<i8> operator==(const Vectorized<i8>& other) const {
        return _mm256_cmpeq_epi8(values, other.values);
      }
      Vectorized<i8> operator!=(const Vectorized<i8>& other) const {
        return invert(_mm256_cmpeq_epi8(values, other.values));
      }
      Vectorized<i8> operator<(const Vectorized<i8>& other) const {
        return _mm256_cmpgt_epi8(other.values, values);
      }
      Vectorized<i8> operator<=(const Vectorized<i8>& other) const {
        return invert(_mm256_cmpgt_epi8(values, other.values));
      }
      Vectorized<i8> operator>(const Vectorized<i8>& other) const {
        return _mm256_cmpgt_epi8(values, other.values);
      }
      Vectorized<i8> operator>=(const Vectorized<i8>& other) const {
        return invert(_mm256_cmpgt_epi8(other.values, values));
      }

      Vectorized<i8> eq(const Vectorized<i8>& other) const;
      Vectorized<i8> ne(const Vectorized<i8>& other) const;
      Vectorized<i8> gt(const Vectorized<i8>& other) const;
      Vectorized<i8> ge(const Vectorized<i8>& other) const;
      Vectorized<i8> lt(const Vectorized<i8>& other) const;
      Vectorized<i8> le(const Vectorized<i8>& other) const;
    };

    template <>
    Vectorized<i64> inline operator+(const Vectorized<i64>& a, const Vectorized<i64>& b) {
      return _mm256_add_epi64(a, b);
    }

    template <>
    Vectorized<i32> inline operator+(const Vectorized<i32>& a, const Vectorized<i32>& b) {
      return _mm256_add_epi32(a, b);
    }

    template <>
    Vectorized<i16> inline operator+(const Vectorized<i16>& a, const Vectorized<i16>& b) {
      return _mm256_add_epi16(a, b);
    }

    template <>
    Vectorized<i8> inline operator+(const Vectorized<i8>& a, const Vectorized<i8>& b) {
      return _mm256_add_epi8(a, b);
    }

    template <>
    Vectorized<i64> inline operator-(const Vectorized<i64>& a, const Vectorized<i64>& b) {
      return _mm256_sub_epi64(a, b);
    }

    template <>
    Vectorized<i32> inline operator-(const Vectorized<i32>& a, const Vectorized<i32>& b) {
      return _mm256_sub_epi32(a, b);
    }

    template <>
    Vectorized<i16> inline operator-(const Vectorized<i16>& a, const Vectorized<i16>& b) {
      return _mm256_sub_epi16(a, b);
    }

    template <>
    Vectorized<i8> inline operator-(const Vectorized<i8>& a, const Vectorized<i8>& b) {
      return _mm256_sub_epi8(a, b);
    }

    // Negation. Defined here so we can utilize operator-
    Vectorized<i64> Vectorized<i64>::neg() const {
      return Vectorized<i64>(0) - *this;
    }

    Vectorized<i32> Vectorized<i32>::neg() const {
      return Vectorized<i32>(0) - *this;
    }

    Vectorized<i16> Vectorized<i16>::neg() const {
      return Vectorized<i16>(0) - *this;
    }

    Vectorized<i8> Vectorized<i8>::neg() const {
      return Vectorized<i8>(0) - *this;
    }

    // Emulate operations with no native 64-bit support in avx,
    // by extracting each element, performing the operation pointwise,
    // then combining the results into a vector.
    template <typename op_t>
    Vectorized<i64> inline emulate(const Vectorized<i64>& a, const Vectorized<i64>& b, const op_t& op) {
      i64 a0 = _mm256_extract_epi64(a, 0);
      i64 a1 = _mm256_extract_epi64(a, 1);
      i64 a2 = _mm256_extract_epi64(a, 2);
      i64 a3 = _mm256_extract_epi64(a, 3);

      i64 b0 = _mm256_extract_epi64(b, 0);
      i64 b1 = _mm256_extract_epi64(b, 1);
      i64 b2 = _mm256_extract_epi64(b, 2);
      i64 b3 = _mm256_extract_epi64(b, 3);

      i64 c0 = op(a0, b0);
      i64 c1 = op(a1, b1);
      i64 c2 = op(a2, b2);
      i64 c3 = op(a3, b3);

      return _mm256_set_epi64x(c3, c2, c1, c0);
    }

    template <typename op_t>
    Vectorized<i64> inline emulate(const Vectorized<i64>& a, const Vectorized<i64>& b, const Vectorized<i64>& c, const op_t& op) {
      i64 a0 = _mm256_extract_epi64(a, 0);
      i64 a1 = _mm256_extract_epi64(a, 1);
      i64 a2 = _mm256_extract_epi64(a, 2);
      i64 a3 = _mm256_extract_epi64(a, 3);

      i64 b0 = _mm256_extract_epi64(b, 0);
      i64 b1 = _mm256_extract_epi64(b, 1);
      i64 b2 = _mm256_extract_epi64(b, 2);
      i64 b3 = _mm256_extract_epi64(b, 3);

      i64 c0 = _mm256_extract_epi64(c, 0);
      i64 c1 = _mm256_extract_epi64(c, 1);
      i64 c2 = _mm256_extract_epi64(c, 2);
      i64 c3 = _mm256_extract_epi64(c, 3);

      i64 d0 = op(a0, b0, c0);
      i64 d1 = op(a1, b1, c1);
      i64 d2 = op(a2, b2, c2);
      i64 d3 = op(a3, b3, c3);

      return _mm256_set_epi64x(d3, d2, d1, d0);
    }

    // AVX2 has no intrinsic for i64 multiply so it needs to be emulated
    // This could be implemented more efficiently using epi32 instructions
    // This is also technically avx compatible, but then we'll need AVX
    // code for add as well.
    // Note: intentionally ignores undefined behavior like (-lowest * -1).
    template <>
    Vectorized<i64> inline operator*(const Vectorized<i64>& a, const Vectorized<i64>& b) {
      return emulate(a, b, [](i64 a_point, i64 b_point) __ubsan_ignore_undefined__ {return a_point * b_point;});
    }

    template <>
    Vectorized<i32> inline operator*(const Vectorized<i32>& a, const Vectorized<i32>& b) {
      return _mm256_mullo_epi32(a, b);
    }

    template <>
    Vectorized<i16> inline operator*(const Vectorized<i16>& a, const Vectorized<i16>& b) {
      return _mm256_mullo_epi16(a, b);
    }

    template <typename T, typename Op>
    Vectorized<T> inline int_elementwise_binary_256(const Vectorized<T>& a, const Vectorized<T>& b, Op op) {
      T values_a[Vectorized<T>::size()];
      T values_b[Vectorized<T>::size()];
      a.store(values_a);
      b.store(values_b);
      for (int i = 0; i != Vectorized<T>::size(); i++) {
        values_a[i] = op(values_a[i], values_b[i]);
      }
      return Vectorized<T>::loadu(values_a);
    }

    template <>
    Vectorized<i8> inline operator*(const Vectorized<i8>& a, const Vectorized<i8>& b) {
      // We don't have an instruction for multiplying i8
      return int_elementwise_binary_256(a, b, multiplies<i8>());
    }

    template <>
    Vectorized<i64> inline minimum(const Vectorized<i64>& a, const Vectorized<i64>& b) {
      return emulate(a, b, [](i64 a_point, i64 b_point) {return min(a_point, b_point);});
    }

    template <>
    Vectorized<i32> inline minimum(const Vectorized<i32>& a, const Vectorized<i32>& b) {
      return _mm256_min_epi32(a, b);
    }

    template <>
    Vectorized<i16> inline minimum(const Vectorized<i16>& a, const Vectorized<i16>& b) {
      return _mm256_min_epi16(a, b);
    }

    template <>
    Vectorized<i8> inline minimum(const Vectorized<i8>& a, const Vectorized<i8>& b) {
      return _mm256_min_epi8(a, b);
    }

    template <>
    Vectorized<i64> inline maximum(const Vectorized<i64>& a, const Vectorized<i64>& b) {
      return emulate(a, b, [](i64 a_point, i64 b_point) {return max(a_point, b_point);});
    }

    template <>
    Vectorized<i32> inline maximum(const Vectorized<i32>& a, const Vectorized<i32>& b) {
      return _mm256_max_epi32(a, b);
    }

    template <>
    Vectorized<i16> inline maximum(const Vectorized<i16>& a, const Vectorized<i16>& b) {
      return _mm256_max_epi16(a, b);
    }

    template <>
    Vectorized<i8> inline maximum(const Vectorized<i8>& a, const Vectorized<i8>& b) {
      return _mm256_max_epi8(a, b);
    }

    template <>
    Vectorized<i64> inline clamp(const Vectorized<i64>& a, const Vectorized<i64>& min_val, const Vectorized<i64>& max_val) {
      return emulate(a, min_val, max_val, [](i64 a_point, i64 min_point, i64 max_point) {return min(max_point, max(a_point, min_point));});
    }

    template <>
    Vectorized<i32> inline clamp(const Vectorized<i32>& a, const Vectorized<i32>& min_val, const Vectorized<i32>& max_val) {
      return _mm256_min_epi32(max_val, _mm256_max_epi32(a, min_val));
    }

    template <>
    Vectorized<i16> inline clamp(const Vectorized<i16>& a, const Vectorized<i16>& min_val, const Vectorized<i16>& max_val) {
      return _mm256_min_epi16(max_val, _mm256_max_epi16(a, min_val));
    }

    template <>
    Vectorized<i8> inline clamp(const Vectorized<i8>& a, const Vectorized<i8>& min_val, const Vectorized<i8>& max_val) {
      return _mm256_min_epi8(max_val, _mm256_max_epi8(a, min_val));
    }

    template <>
    Vectorized<i64> inline clamp_max(const Vectorized<i64>& a, const Vectorized<i64>& max_val) {
      return emulate(a, max_val, [](i64 a_point, i64 max_point) {return min(max_point, a_point);});
    }

    template <>
    Vectorized<i32> inline clamp_max(const Vectorized<i32>& a, const Vectorized<i32>& max_val) {
      return _mm256_min_epi32(max_val, a);
    }

    template <>
    Vectorized<i16> inline clamp_max(const Vectorized<i16>& a, const Vectorized<i16>& max_val) {
      return _mm256_min_epi16(max_val, a);
    }

    template <>
    Vectorized<i8> inline clamp_max(const Vectorized<i8>& a, const Vectorized<i8>& max_val) {
      return _mm256_min_epi8(max_val, a);
    }

    template <>
    Vectorized<i64> inline clamp_min(const Vectorized<i64>& a, const Vectorized<i64>& min_val) {
      return emulate(a, min_val, [](i64 a_point, i64 min_point) {return max(min_point, a_point);});
    }

    template <>
    Vectorized<i32> inline clamp_min(const Vectorized<i32>& a, const Vectorized<i32>& min_val) {
      return _mm256_max_epi32(min_val, a);
    }

    template <>
    Vectorized<i16> inline clamp_min(const Vectorized<i16>& a, const Vectorized<i16>& min_val) {
      return _mm256_max_epi16(min_val, a);
    }

    template <>
    Vectorized<i8> inline clamp_min(const Vectorized<i8>& a, const Vectorized<i8>& min_val) {
      return _mm256_max_epi8(min_val, a);
    }

    template<typename T>
    Vectorized<i32> inline convert_to_int32(const T* ptr) {
      return Vectorized<i32>::loadu(ptr);
    }

    template<>
    Vectorized<i32> inline convert_to_int32<i8>(const i8* ptr) {
      return _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr)));
    }

    template<>
    Vectorized<i32> inline convert_to_int32<u8>(const u8* ptr) {
      return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr)));
    }

    template <>
    Vectorized<i64> inline operator/(const Vectorized<i64>& a, const Vectorized<i64>& b) {
      return int_elementwise_binary_256(a, b, divides<i64>());
    }
    template <>
    Vectorized<i32> inline operator/(const Vectorized<i32>& a, const Vectorized<i32>& b) {
      return int_elementwise_binary_256(a, b, divides<i32>());
    }
    template <>
    Vectorized<i16> inline operator/(const Vectorized<i16>& a, const Vectorized<i16>& b) {
      return int_elementwise_binary_256(a, b, divides<i16>());
    }
    template <>
    Vectorized<i8> inline operator/(const Vectorized<i8>& a, const Vectorized<i8>& b) {
      return int_elementwise_binary_256(a, b, divides<i8>());
    }

    template<class T, typename enable_if_t<is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
    inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
      return _mm256_and_si256(a, b);
    }
    template<class T, typename enable_if_t<is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
    inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
      return _mm256_or_si256(a, b);
    }
    template<class T, typename enable_if_t<is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
    inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
      return _mm256_xor_si256(a, b);
    }
    template<class T, typename enable_if_t<is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
    inline Vectorized<T> operator~(const Vectorized<T>& a) {
      return _mm256_xor_si256(a, _mm256_set1_epi32(-1));
    }

    Vectorized<i64> Vectorized<i64>::eq(const Vectorized<i64>& other) const {
      return (*this == other) & Vectorized<i64>(1);
    }

    Vectorized<i64> Vectorized<i64>::ne(const Vectorized<i64>& other) const {
      return (*this != other) & Vectorized<i64>(1);
    }

    Vectorized<i64> Vectorized<i64>::gt(const Vectorized<i64>& other) const {
      return (*this > other) & Vectorized<i64>(1);
    }

    Vectorized<i64> Vectorized<i64>::ge(const Vectorized<i64>& other) const {
      return (*this >= other) & Vectorized<i64>(1);
    }

    Vectorized<i64> Vectorized<i64>::lt(const Vectorized<i64>& other) const {
      return (*this < other) & Vectorized<i64>(1);
    }

    Vectorized<i64> Vectorized<i64>::le(const Vectorized<i64>& other) const {
      return (*this <= other) & Vectorized<i64>(1);
    }

    Vectorized<i32> Vectorized<i32>::eq(const Vectorized<i32>& other) const {
      return (*this == other) & Vectorized<i32>(1);
    }

    Vectorized<i32> Vectorized<i32>::ne(const Vectorized<i32>& other) const {
      return (*this != other) & Vectorized<i32>(1);
    }

    Vectorized<i32> Vectorized<i32>::gt(const Vectorized<i32>& other) const {
      return (*this > other) & Vectorized<i32>(1);
    }

    Vectorized<i32> Vectorized<i32>::ge(const Vectorized<i32>& other) const {
      return (*this >= other) & Vectorized<i32>(1);
    }

    Vectorized<i32> Vectorized<i32>::lt(const Vectorized<i32>& other) const {
      return (*this < other) & Vectorized<i32>(1);
    }

    Vectorized<i32> Vectorized<i32>::le(const Vectorized<i32>& other) const {
      return (*this <= other) & Vectorized<i32>(1);
    }

    Vectorized<i16> Vectorized<i16>::eq(const Vectorized<i16>& other) const {
      return (*this == other) & Vectorized<i16>(1);
    }

    Vectorized<i16> Vectorized<i16>::ne(const Vectorized<i16>& other) const {
      return (*this != other) & Vectorized<i16>(1);
    }

    Vectorized<i16> Vectorized<i16>::gt(const Vectorized<i16>& other) const {
      return (*this > other) & Vectorized<i16>(1);
    }

    Vectorized<i16> Vectorized<i16>::ge(const Vectorized<i16>& other) const {
      return (*this >= other) & Vectorized<i16>(1);
    }

    Vectorized<i16> Vectorized<i16>::lt(const Vectorized<i16>& other) const {
      return (*this < other) & Vectorized<i16>(1);
    }

    Vectorized<i16> Vectorized<i16>::le(const Vectorized<i16>& other) const {
      return (*this <= other) & Vectorized<i16>(1);
    }

    Vectorized<i8> Vectorized<i8>::eq(const Vectorized<i8>& other) const {
      return (*this == other) & Vectorized<i8>(1);
    }

    Vectorized<i8> Vectorized<i8>::ne(const Vectorized<i8>& other) const {
      return (*this != other) & Vectorized<i8>(1);
    }

    Vectorized<i8> Vectorized<i8>::gt(const Vectorized<i8>& other) const {
      return (*this > other) & Vectorized<i8>(1);
    }

    Vectorized<i8> Vectorized<i8>::ge(const Vectorized<i8>& other) const {
      return (*this >= other) & Vectorized<i8>(1);
    }

    Vectorized<i8> Vectorized<i8>::lt(const Vectorized<i8>& other) const {
      return (*this < other) & Vectorized<i8>(1);
    }

    Vectorized<i8> Vectorized<i8>::le(const Vectorized<i8>& other) const {
      return (*this <= other) & Vectorized<i8>(1);
    }

    #endif
    */
}

