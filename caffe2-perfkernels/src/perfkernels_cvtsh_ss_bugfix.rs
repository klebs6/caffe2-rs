crate::ix!();

/**
  | It seems that microsoft msvc does not
  | have a _cvtsh_ss implementation so
  | we will add a dummy version to it.
  |
  */
#[inline] pub fn cvtsh_ss(x: u16) -> f32 {
    
    todo!();
    /*
        union {
        std::uint32_t intval;
        float floatval;
      } t1;
      std::uint32_t t2, t3;
      t1.intval = x & 0x7fff; // Non-sign bits
      t2 = x & 0x8000; // Sign bit
      t3 = x & 0x7c00; // Exponent
      t1.intval <<= 13; // Align mantissa on MSB
      t2 <<= 16; // Shift sign bit into position
      t1.intval += 0x38000000; // Adjust bias
      t1.intval = (t3 == 0 ? 0 : t1.intval); // Denormals-as-zero
      t1.intval |= t2; // Re-insert sign bit
      return t1.floatval;
    */
}

#[inline] pub fn cvtss_sh(x: f32, imm8: i32) -> u16 {
    
    todo!();
    /*
        unsigned short ret;
      *reinterpret_cast<at::Half*>(&ret) = x;
      return ret;
    */
}
