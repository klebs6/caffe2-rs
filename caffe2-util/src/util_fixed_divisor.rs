crate::ix!();

/**
  | Utility class for quickly calculating
  | quotients and remainders for a known integer
  | divisor
  |
  | Works for any positive divisor, 1 to
  | INT_MAX. One 64-bit multiplication and one
  | 64-bit shift is used to calculate the result.
  */
pub struct FixedDivisorI32 {
    d:  i32, // default = 1

    #[cfg(not(__hip_platform_hcc__))]
    magic:  u64,

    #[cfg(not(__hip_platform_hcc__))]
    shift:  i32,
}

impl FixedDivisorI32 {
    
    pub fn new(d: i32) -> Self {
    
        todo!();
        /*
            : d_(d) 

    #ifndef __HIP_PLATFORM_HCC__
        CalcSignedMagic();
    #endif // __HIP_PLATFORM_HCC__
        */
    }
    
    #[inline] pub fn d(&self) -> i32 {
        
        todo!();
        /*
            return d_;
        */
    }
    
    #[cfg_attr(any(cuda, hip), host, device)]
    #[inline] pub fn magic(&self) -> u64 {
        
        todo!();
        /*
            return magic_;
        */
    }
    
    #[cfg_attr(any(cuda, hip), host, device)]
    #[inline] pub fn shift(&self) -> i32 {
        
        todo!();
        /*
            return shift_;
        */
    }
    
    /// Calculates `q = n / d`.
    #[cfg_attr(any(cuda, hip), host, device)]
    #[inline] pub fn div(&self, n: i32) -> i32 {
        
        todo!();
        /*
            #ifdef __HIP_PLATFORM_HCC__
        return n / d_;
    #else // __HIP_PLATFORM_HCC__
        // In lieu of a mulhi instruction being available, perform the
        // work in uint64
        return (int32_t)((magic_ * (uint64_t)n) >> shift_);
    #endif // __HIP_PLATFORM_HCC__
        */
    }
    
    /// Calculates `r = n % d`.
    #[cfg_attr(any(cuda, hip), host, device)]
    #[inline] pub fn modulo(&self, n: i32) -> i32 {
        
        todo!();
        /*
            return n - d_ * Div(n);
        */
    }
    
    /// Calculates `q = n / d` and `r = n % d` together.
    #[cfg_attr(any(cuda, hip), host, device)]
    #[inline] pub fn div_mod(&self, 
        n: i32,
        q: *mut i32,
        r: *mut i32)  {

        todo!();
        /*
            *q = Div(n);
        *r = n - d_ * *q;
        */
    }
    
    /**
      | Calculates magic multiplicative value and
      | shift amount for calculating `q = n / d`
      | for signed 32-bit integers.
      |
      | Implementation taken from Hacker's Delight
      | section 10.
      */
    #[cfg(not(__hip_platform_hcc__))]
    #[inline] pub fn calc_signed_magic(&mut self)  {
        
        todo!();
        /*
            if (d_ == 1) {
          magic_ = UINT64_C(0x1) << 32;
          shift_ = 32;
          return;
        }

        const std::uint32_t two31 = UINT32_C(0x80000000);
        const std::uint32_t ad = std::abs(d_);
        const std::uint32_t t = two31 + ((uint32_t)d_ >> 31);
        const std::uint32_t anc = t - 1 - t % ad; // Absolute value of nc.
        std::uint32_t p = 31; // Init. p.
        std::uint32_t q1 = two31 / anc; // Init. q1 = 2**p/|nc|.
        std::uint32_t r1 = two31 - q1 * anc; // Init. r1 = rem(2**p, |nc|).
        std::uint32_t q2 = two31 / ad; // Init. q2 = 2**p/|d|.
        std::uint32_t r2 = two31 - q2 * ad; // Init. r2 = rem(2**p, |d|).
        std::uint32_t delta = 0;
        do {
          ++p;
          q1 <<= 1; // Update q1 = 2**p/|nc|.
          r1 <<= 1; // Update r1 = rem(2**p, |nc|).
          if (r1 >= anc) { // (Must be an unsigned
            ++q1; // comparison here).
            r1 -= anc;
          }
          q2 <<= 1; // Update q2 = 2**p/|d|.
          r2 <<= 1; // Update r2 = rem(2**p, |d|).
          if (r2 >= ad) { // (Must be an unsigned
            ++q2; // comparison here).
            r2 -= ad;
          }
          delta = ad - r2;
        } while (q1 < delta || (q1 == delta && r1 == 0));
        std::int32_t magic = q2 + 1;
        if (d_ < 0) {
          magic = -magic;
        }
        shift_ = p;
        magic_ = (std::uint64_t)(std::uint32_t)magic;
        */
    }
}
