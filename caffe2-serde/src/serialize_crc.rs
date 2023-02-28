#[cfg(use_external_mzcrc)]
#[inline] pub fn mz_crc32(
    crc:     u32,
    ptr:     *const u8,
    buf_len: usize) -> u32 {
    
    todo!();
    /*
        auto z = crc32_fast(ptr, buf_len, crc);
      return z;
    */
}
