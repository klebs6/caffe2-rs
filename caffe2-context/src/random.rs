crate::ix!();

#[cfg(all(not(caffe2_is_xplat_build), not(c10_mobile)))]
pub type RandGenType = CPUGeneratorImpl;

#[cfg(any(caffe2_is_xplat_build, c10_mobile))]
pub type RandGenType = mersenne_twister::MT19937;

/**
  | A function to generate a random number
  | seed that is unique in a best-effort
  | basis, using an ever-incrementing
  | seed and the current time.
  |
  */
#[inline] pub fn random_number_seed() -> u32 {
    
    todo!();
    /*
        // Originally copied from folly::randomNumberSeed (at 418ad4)
      // modified to use chrono instead of sys/time.h
      static std::atomic<uint32_t> seedInput(0);
      auto tv = std::chrono::system_clock::now().time_since_epoch();
      uint64_t usec = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(tv).count());
      uint32_t tv_sec = usec / 1000000;
      uint32_t tv_usec = usec % 1000000;
      const uint32_t kPrime0 = 51551;
      const uint32_t kPrime1 = 61631;
      const uint32_t kPrime2 = 64997;
      const uint32_t kPrime3 = 111857;
      return kPrime0 * (seedInput++) + kPrime1 * static_cast<uint32_t>(getpid()) +
          kPrime2 * tv_sec + kPrime3 * tv_usec;
    */
}
