crate::ix!();


/**
  | NOTE: clang-format wants to use a different
  | formatting but the current formatting should
  | be easier to read.
  |
  | clang-format off
  */
pub const ld_st_masks: Align64::<[[i32; 8]; 8]> = Align64([
    [  0,  0,  0,  0,  0,  0,  0,  0, ],
    [ -1,  0,  0,  0,  0,  0,  0,  0, ],
    [ -1, -1,  0,  0,  0,  0,  0,  0, ],
    [ -1, -1, -1,  0,  0,  0,  0,  0, ],
    [ -1, -1, -1, -1,  0,  0,  0,  0, ],
    [ -1, -1, -1, -1, -1,  0,  0,  0, ],
    [ -1, -1, -1, -1, -1, -1,  0,  0, ],
    [ -1, -1, -1, -1, -1, -1, -1,  0, ],
]);

/**
  | convert to float16 reducing mantissa,
  | preserving exponent
  |
  */
#[inline] pub fn fp32_to_bfp16(
    source: *const f32,
    size:   usize,
    dest:   *mut f32)  {
    
    todo!();
    /*
        // Results on a 1 sign, 8 exponent, 7 mantissa
      constexpr int mask = 0xFFFF0000;
      __m256 wmask = _mm256_broadcast_ss(reinterpret_cast<const float*>(&mask));

      size_t i = 0;
      for (; i < (size / 8) * 8; i += 8) {
        __m256 data = _mm256_loadu_ps(&source[i]);
        _mm256_storeu_ps(&dest[i], _mm256_and_ps(wmask, data));
      }
      if (i < size) {
        __m256i ld_st_mask = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(ld_st_masks[size - i]));
        __m256 data = _mm256_maskload_ps(&source[i], ld_st_mask);
        _mm256_maskstore_ps(&dest[i], ld_st_mask, _mm256_and_ps(wmask, data));
      }
    */
}

/// convert to float24 reducing mantissa, preserving exponent
#[inline] pub fn fp32_to_bfp24(
    source: *const f32,
    size:   usize,
    dest:   *mut f32)  {
    
    todo!();
    /*
        // Results on a 1 sign, 8 exponent, 7 mantissa
      constexpr int mask = 0xFFFFFF00;
      __m256 wmask = _mm256_broadcast_ss(reinterpret_cast<const float*>(&mask));

      size_t i = 0;
      for (; i < (size / 8) * 8; i += 8) {
        __m256 data = _mm256_loadu_ps(&source[i]);
        _mm256_storeu_ps(&dest[i], _mm256_and_ps(wmask, data));
      }
      if (i < size) {
        __m256i ld_st_mask = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(ld_st_masks[size - i]));
        __m256 data = _mm256_maskload_ps(&source[i], ld_st_mask);
        _mm256_maskstore_ps(&dest[i], ld_st_mask, _mm256_and_ps(wmask, data));
      }
    */
}

/// convert to float14 reducing mantissa, preserving exponent
#[inline] pub fn fp32_to_bfp14(
    source: *const f32,
    size:   usize,
    dest:   *mut f32)  {
    
    todo!();
    /*
        // Results on a 1 sign, 8 exponent, 7 mantissa
      constexpr int mask = 0xFFFC0000;
      __m256 wmask = _mm256_broadcast_ss((float*)(&mask));

      size_t i = 0;
      for (; i < (size / 8) * 8; i += 8) {
        __m256 data = _mm256_loadu_ps(&source[i]);
        _mm256_storeu_ps(&dest[i], _mm256_and_ps(wmask, data));
      }
      if (i < size) {
        __m256i ld_st_mask = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(ld_st_masks[size - i]));
        __m256 data = _mm256_maskload_ps(&source[i], ld_st_mask);
        _mm256_maskstore_ps(&dest[i], ld_st_mask, _mm256_and_ps(wmask, data));
      }
    */
}

#[inline] pub fn fp32_to_bfp16_scalar(
    source: *const f32,
    size:   usize,
    dest:   *mut f32)  {
    
    todo!();
    /*
        constexpr int mask = 0xFFFF0000;
      for (const auto i : c10::irange(size)) {
        *(int*)(dest + i) = *(int*)(source + i) & mask;
      }
    */
}

/// convert to IEEE float16
#[inline] pub fn fp32_to_fp16(
    source: *const f32,
    size:   usize,
    dest:   *mut f32)  {
    
    todo!();
    /*
        size_t i = 0;
      for (; i < (size / 8) * 8; i += 8) {
        __m128i vin_fp16 = _mm256_cvtps_ph(_mm256_loadu_ps(&source[i]), 0);
        _mm256_storeu_ps(&dest[i], _mm256_cvtph_ps(vin_fp16));
      }
      if (i < size) {
        __m256i ld_st_mask = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(ld_st_masks[size - i]));
        __m128i vin_fp16 =
            _mm256_cvtps_ph(_mm256_maskload_ps(&source[i], ld_st_mask), 0);
        _mm256_maskstore_ps(&dest[i], ld_st_mask, _mm256_cvtph_ps(vin_fp16));
      }
    */
}

/// fp32 -> int32 -> += 1<< 15 -> fp32 -> truncation
#[inline] pub fn fp32_to_bfp16_round(
    source: *const f32,
    size:   usize,
    dest:   *mut f32)  {
    
    todo!();
    /*
        constexpr int offset = 0x00008000; // 1 << 15
      constexpr int mask = 0xFFFF0000;

      __m256i woffset = _mm256_set1_epi32(offset);
      __m256i wmask = _mm256_set1_epi32(mask);

      size_t i = 0;
      for (; i < (size / 8) * 8; i += 8) {
        __m256i v32int = _mm256_add_epi32(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&source[i])),
            woffset);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(&dest[i]), _mm256_and_si256(wmask, v32int));
      }
      if (i < size) {
        __m256i ld_st_mask = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(ld_st_masks[size - i]));
        __m256i v32int = _mm256_add_epi32(
            _mm256_maskload_epi32(
                reinterpret_cast<const int*>(&source[i]), ld_st_mask),
            woffset);
        _mm256_maskstore_epi32(
            reinterpret_cast<int*>(&dest[i]),
            ld_st_mask,
            _mm256_and_si256(wmask, v32int));
      }
    */
}
