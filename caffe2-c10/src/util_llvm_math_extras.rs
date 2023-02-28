/*!
 | -------------------------------------------[.cpp/pytorch/c10/util/llvmMathExtras.h]
 | ===-- llvm/Support/MathExtras.h - Useful math functions -------*- C++ -*-===//
 |
 |  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 |  See https://llvm.org/LICENSE.txt for license information.
 |  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 |
 | ===----------------------------------------------------------------------===//
 |
 |  This file contains some functions that are useful for math stuff.
 |
 | ===----------------------------------------------------------------------===//
 */

crate::ix!();

macro_rules! has_builtin {
    ($x:ident) => {
        /*
                0
        */
    }
}

/**
  | The behavior an operation has on an input
  | of 0.
  |
  */
pub enum ZeroBehavior {

    /// The returned value is undefined.
    Undefined,

    /// The returned value is T::max
    Max,

    /// The returned value is numeric_limits<T>::digits
    Width
}

/**
  | Count number of 0's from the least significant
  |   bit to the most stopping at the first 1.
  |
  | Only unsigned integral types are allowed.
  |
  | \param ZB the behavior on an input of 0. Only
  |   ZeroBehavior::Width and ZeroBehavior::Undefined are valid
  |   arguments.
  |
  */
pub fn count_trailing_zeros<T>(
        val: T,
        ZB:  Option<ZeroBehavior>) -> usize {

    let ZB: ZeroBehavior = ZB.unwrap_or(ZeroBehavior::Width);

    todo!();
        /*
            static_assert(
          numeric_limits<T>::is_integer && !numeric_limits<T>::is_signed,
          "Only unsigned integral types are allowed.");
      return llvm::TrailingZerosCounter<T, sizeof(T)>::count(Val, ZB);
        */
}

/**
  | Count number of 0's from the most significant
  | bit to the least stopping at the first
  | 1.
  | 
  | Only unsigned integral types are allowed.
  | 
  | -----------
  | @param ZB
  | 
  | the behavior on an input of 0. Only ZeroBehavior::Width
  | and ZeroBehavior::Undefined are valid arguments.
  |
  */
pub fn count_leading_zeros<T>(
        val: T,
        ZB:  Option<ZeroBehavior>) -> usize {

    let ZB: ZeroBehavior = ZB.unwrap_or(ZeroBehavior::Width);

    todo!();
        /*
            static_assert(
          numeric_limits<T>::is_integer && !numeric_limits<T>::is_signed,
          "Only unsigned integral types are allowed.");
      return llvm::LeadingZerosCounter<T, sizeof(T)>::count(Val, ZB);
        */
}

/**
  | Get the index of the first set bit starting
  | from the least significant bit.
  | 
  | Only unsigned integral types are allowed.
  | 
  | -----------
  | @param ZB
  | 
  | the behavior on an input of 0. Only ZeroBehavior::Max
  | and ZeroBehavior::Undefined are valid arguments.
  |
  */
pub fn find_first_set<T>(
        val: T,
        ZB:  Option<ZeroBehavior>) -> T {

    let ZB: ZeroBehavior = ZB.unwrap_or(ZeroBehavior::Max);

    todo!();
        /*
            if (ZB == ZeroBehavior::Max && Val == 0)
        return T::max;

      return countTrailingZeros(Val, ZeroBehavior::Undefined);
        */
}

/**
  | Create a bitmask with the N right-most bits
  | set to 1, and all other bits set to 0.
  | Only unsigned types are allowed.
  |
  */
pub fn mask_trailing_ones<T>(N: u32) -> T {

    todo!();
        /*
            static_assert(is_unsigned<T>::value, "Invalid type!");
      const unsigned Bits = CHAR_BIT * sizeof(T);
      assert(N <= Bits && "Invalid bit index");
      return N == 0 ? 0 : (T(-1) >> (Bits - N));
        */
}

/**
  | Create a bitmask with the N left-most bits set
  | to 1, and all other bits set to 0.  Only
  | unsigned types are allowed.
  |
  */
pub fn mask_leading_ones<T>(N: u32) -> T {

    todo!();
        /*
            return ~maskTrailingOnes<T>(CHAR_BIT * sizeof(T) - N);
        */
}

/**
  | Create a bitmask with the N right-most bits
  | set to 0, and all other bits set to 1.  Only
  | unsigned types are allowed.
  |
  */
pub fn mask_trailing_zeros<T>(N: u32) -> T {

    todo!();
        /*
            return maskLeadingOnes<T>(CHAR_BIT * sizeof(T) - N);
        */
}

/**
  | Create a bitmask with the N left-most bits set
  | to 0, and all other bits set to 1.  Only
  | unsigned types are allowed.
  |
  */
pub fn mask_leading_zeros<T>(N: u32) -> T {

    todo!();
        /*
            return maskTrailingOnes<T>(CHAR_BIT * sizeof(T) - N);
        */
}

/**
  | Get the index of the last set bit starting
  | from the least significant bit.
  | 
  | Only unsigned integral types are allowed.
  | 
  | -----------
  | @param ZB
  | 
  | the behavior on an input of 0. Only
  | 
  | ZeroBehavior::Max and ZeroBehavior::Undefined are valid arguments.
  |
  */
pub fn find_last_set<T>(
        val: T,
        ZB:  Option<ZeroBehavior>) -> T {

    let ZB: ZeroBehavior = ZB.unwrap_or(ZeroBehavior::Max);

    todo!();
        /*
            if (ZB == ZeroBehavior::Max && Val == 0)
        return T::max;

      // Use ^ instead of - because both gcc and llvm can remove the associated ^
      // in the __builtin_clz intrinsic on x86.
      return countLeadingZeros(Val, ZeroBehavior::Undefined) ^
          (numeric_limits<T>::digits - 1);
        */
}

/**
  | Macro compressed bit reversal table for 256
  | bits.
  |
  | http://graphics.stanford.edu/~seander/bithacks.html#BitReverseTable
  |
  */
lazy_static!{
    /*
    static const unsigned char BitReverseTable256[256] = {
    #define R2(n) n, n + 2 * 64, n + 1 * 64, n + 3 * 64
    #define R4(n) R2(n), R2(n + 2 * 16), R2(n + 1 * 16), R2(n + 3 * 16)
    #define R6(n) R4(n), R4(n + 2 * 4), R4(n + 1 * 4), R4(n + 3 * 4)
        R6(0),
        R6(2),
        R6(1),
        R6(3)
    #undef R2
    #undef R4
    #undef R6
    };
    */
}

/// Reverse the bits in \p Val.
pub fn reverse_bits<T>(val: T) -> T {

    todo!();
        /*
            unsigned char in[sizeof(Val)];
      unsigned char out[sizeof(Val)];
      memcpy(in, &Val, sizeof(Val));
      for (unsigned i = 0; i < sizeof(Val); ++i)
        out[(sizeof(Val) - i) - 1] = BitReverseTable256[in[i]];
      memcpy(&Val, out, sizeof(Val));
      return Val;
        */
}



// NOTE: The following support functions use the
// _32/_64 extensions instead of type overloading
// so that signed and unsigned integers can be
// used without ambiguity.

/// Return the high 32 bits of a 64 bit value.
#[inline] pub const fn hi_32(value: u64) -> u32 {
    
    todo!();
        /*
            return static_cast<uint32_t>(Value >> 32);
        */
}

/// Return the low 32 bits of a 64 bit value.
#[inline] pub const fn lo_32(value: u64) -> u32 {
    
    todo!();
        /*
            return static_cast<uint32_t>(Value);
        */
}

/// Make a 64-bit integer from a high / low pair
/// of 32-bit integers.
///
#[inline] pub const fn make_64(
        high: u32,
        low:  u32) -> u64 {
    
    todo!();
        /*
            return ((uint64_t)High << 32) | (uint64_t)Low;
        */
}

/**
  | Checks if an integer fits into the given
  | bit width.
  |
  */
#[inline] pub const fn is_int<const N: u32>(x: i64) -> bool {

    todo!();
        /*
            return N >= 64 ||
          (-(INT64_C(1) << (N - 1)) <= x && x < (INT64_C(1) << (N - 1)));
        */
}

lazy_static!{
    /*
    // Template specializations to get better code for
    // common cases.
    //
    template <>
    constexpr inline bool isInt<8>(int64_t x) {
      return static_cast<int8_t>(x) == x;
    }
    template <>
    constexpr inline bool isInt<16>(int64_t x) {
      return static_cast<int16_t>(x) == x;
    }
    template <>
    constexpr inline bool isInt<32>(int64_t x) {
      return static_cast<int32_t>(x) == x;
    }
    */
}

/// Checks if a signed integer is an N bit number
/// shifted left by S.
///
#[inline] pub const fn is_shifted_int<const N: u32, const S: u32>(x: i64) -> bool {

    todo!();
        /*
            static_assert(
          N > 0, "isShiftedInt<0> doesn't make sense (refers to a 0-bit number.");
      static_assert(N + S <= 64, "isShiftedInt<N, S> with N + S > 64 is too wide.");
      return isInt<N + S>(x) && (x % (UINT64_C(1) << S) == 0);
        */
}

/**
  | Checks if an unsigned integer fits into the
  | given bit width.
  |
  | This is written as two functions rather than
  | as simply
  |
  |   return N >= 64 || X < (UINT64_C(1) << N);
  |
  | to keep MSVC from (incorrectly) warning on
  | isUInt<64> that we're shifting left too many
  | places.
  |
  */
lazy_static!{
    /*
    template <unsigned N>
    constexpr inline typename enable_if<(N < 64), bool>::type isUInt(
        uint64_t X) {
      static_assert(N > 0, "isUInt<0> doesn't make sense");
      return X < (UINT64_C(1) << (N));
    }
    template <unsigned N>
    constexpr inline typename enable_if<N >= 64, bool>::type isUInt(
        uint64_t X) {
      return true;
    }

    // Template specializations to get better code for common cases.
    template <>
    constexpr inline bool isUInt<8>(uint64_t x) {
      return static_cast<uint8_t>(x) == x;
    }
    template <>
    constexpr inline bool isUInt<16>(uint64_t x) {
      return static_cast<uint16_t>(x) == x;
    }
    template <>
    constexpr inline bool isUInt<32>(uint64_t x) {
      return static_cast<uint32_t>(x) == x;
    }
    */
}

/// Checks if a unsigned integer is an N bit
/// number shifted left by S.
///
#[inline] pub const fn is_shifted_uint<const N: u32, const S: u32>(x: u64) -> bool {

    todo!();
        /*
            static_assert(
          N > 0, "isShiftedUInt<0> doesn't make sense (refers to a 0-bit number)");
      static_assert(
          N + S <= 64, "isShiftedUInt<N, S> with N + S > 64 is too wide.");
      // Per the two static_asserts above, S must be strictly less than 64.  So
      // 1 << S is not undefined behavior.
      return isUInt<N + S>(x) && (x % (UINT64_C(1) << S) == 0);
        */
}

/// Gets the maximum value for a N-bit unsigned
/// integer.
///
#[inline] pub fn max_uintn(N: u64) -> u64 {
    
    todo!();
        /*
            assert(N > 0 && N <= 64 && "integer width out of range");

      // uint64_t(1) << 64 is undefined behavior, so we can't do
      //   (uint64_t(1) << N) - 1
      // without checking first that N != 64.  But this works and doesn't have a
      // branch.
      return UINT64_MAX >> (64 - N);
        */
}

/// Gets the minimum value for a N-bit signed
/// integer.
///
#[inline] pub fn min_intn(N: i64) -> i64 {
    
    todo!();
        /*
            assert(N > 0 && N <= 64 && "integer width out of range");

      return -(UINT64_C(1) << (N - 1));
        */
}

/// Gets the maximum value for a N-bit signed
/// integer.
///
#[inline] pub fn max_intn(N: i64) -> i64 {
    
    todo!();
        /*
            assert(N > 0 && N <= 64 && "integer width out of range");

      // This relies on two's complement wraparound when N == 64, so we convert to
      // int64_t only at the very end to avoid UB.
      return (UINT64_C(1) << (N - 1)) - 1;
        */
}

/// Checks if an unsigned integer fits into the
/// given (dynamic) bit width.
///
#[inline] pub fn is_uintn(N: u32, x: u64) -> bool {
    
    todo!();
        /*
            return N >= 64 || x <= maxUIntN(N);
        */
}

/// Checks if an signed integer fits into the
/// given (dynamic) bit width.
///
#[inline] pub fn is_intn(N: u32, x: i64) -> bool {
    
    todo!();
        /*
            return N >= 64 || (minIntN(N) <= x && x <= maxIntN(N));
        */
}

/// Return true if the argument is a non-empty
/// sequence of ones starting at the least
/// significant bit with the remainder zero (32
/// bit version).
///
/// Ex. isMask_32(0x0000FFFFU) == true.
///
#[inline] pub const fn is_mask_32(value: u32) -> bool {
    
    todo!();
        /*
            return Value && ((Value + 1) & Value) == 0;
        */
}

/**
  | Return true if the argument is a non-empty
  | sequence of ones starting at the least
  | significant bit with the remainder zero (64
  | bit version).
  |
  */
#[inline] pub const fn is_mask_64(value: u64) -> bool {
    
    todo!();
        /*
            return Value && ((Value + 1) & Value) == 0;
        */
}

/**
  | Return true if the argument contains
  | a non-empty sequence of ones with the
  | remainder zero (32 bit version.)
  | Ex. isShiftedMask_32(0x0000FF00U) == true.
  |
  */
#[inline] pub const fn is_shifted_mask_32(value: u32) -> bool {
    
    todo!();
        /*
            return Value && isMask_32((Value - 1) | Value);
        */
}

/**
  | Return true if the argument contains
  | a non-empty sequence of ones with the
  | remainder zero (64 bit version.)
  |
  */
#[inline] pub const fn is_shifted_mask_64(value: u64) -> bool {
    
    todo!();
        /*
            return Value && isMask_64((Value - 1) | Value);
        */
}

/// Return true if the argument is a power of two > 0.
///
/// Ex. isPowerOf2_32(0x00100000U) == true (32 bit edition.)
///
#[inline] pub const fn is_power_of2_32(value: u32) -> bool {
    
    todo!();
        /*
            return Value && !(Value & (Value - 1));
        */
}

/// Return true if the argument is a power of two > 0 (64 bit edition.)
///
#[inline] pub const fn is_power_of2_64(value: u64) -> bool {
    
    todo!();
        /*
            return Value && !(Value & (Value - 1));
        */
}

/**
  | Count the number of ones from the most
  | significant bit to the first zero bit.
  | 
  | Ex. countLeadingOnes(0xFF0FFF00)
  | == 8. Only unsigned integral types are
  | allowed.
  | 
  | -----------
  | @param ZB
  | 
  | the behavior on an input of all ones.
  | Only ZeroBehavior::Width and ZeroBehavior::Undefined are
  | valid arguments.
  |
  */
pub fn count_leading_ones<T>(
        value: T,
        ZB:    Option<ZeroBehavior>) -> usize {

    let ZB: ZeroBehavior = ZB.unwrap_or(ZeroBehavior::Width);

    todo!();
        /*
            static_assert(
          numeric_limits<T>::is_integer && !numeric_limits<T>::is_signed,
          "Only unsigned integral types are allowed.");
      return countLeadingZeros<T>(~Value, ZB);
        */
}

/**
  | Count the number of ones from the least
  | significant bit to the first zero bit.
  |
  | Ex. countTrailingOnes(0x00FF00FF) == 8. Only
  | unsigned integral types are allowed.
  |
  | \param ZB the behavior on an input of all
  | ones. Only ZeroBehavior::Width and ZeroBehavior::Undefined are valid
  | arguments.
  */
pub fn count_trailing_ones<T>(
        value: T,
        ZB:    Option<ZeroBehavior>) -> usize {

    let ZB: ZeroBehavior = ZB.unwrap_or(ZeroBehavior::Width);

    todo!();
        /*
            static_assert(
          numeric_limits<T>::is_integer && !numeric_limits<T>::is_signed,
          "Only unsigned integral types are allowed.");
      return countTrailingZeros<T>(~Value, ZB);
        */
}

/// Count the number of set bits in a value.
/// Ex. countPopulation(0xF000F000) = 8
/// Returns 0 if the word is zero.
#[inline] pub fn count_population<T>(value: T) -> u32 {

    todo!();
        /*
            static_assert(
          numeric_limits<T>::is_integer && !numeric_limits<T>::is_signed,
          "Only unsigned integral types are allowed.");
      return PopulationCounter<T, sizeof(T)>::count(Value);
        */
}

/// Return the log base 2 of the specified value.
#[inline] pub fn log2(value: f64) -> f64 {
    
    todo!();
        /*
            #if defined(__ANDROID_API__) && __ANDROID_API__ < 18
      return __builtin_log(Value) / __builtin_log(2.0);
    #else
      return log2(Value);
    #endif
        */
}

/// Return the floor log base 2 of the specified
/// value, -1 if the value is zero.
///
/// (32 bit edition.)
///
/// Ex. Log2_32(32) == 5, Log2_32(1) == 0, Log2_32(0) == -1, Log2_32(6) == 2
///
#[inline] pub fn log2_32(value: u32) -> u32 {
    
    todo!();
        /*
            return static_cast<unsigned>(31 - countLeadingZeros(Value));
        */
}

/// Return the floor log base 2 of the specified
/// value, -1 if the value is zero.
///
/// (64 bit edition.)
#[inline] pub fn log2_64(value: u64) -> u32 {
    
    todo!();
        /*
            return static_cast<unsigned>(63 - countLeadingZeros(Value));
        */
}

/**
  | Return the ceil log base 2 of the specified
  | value, 32 if the value is zero.
  |
  | (32 bit edition).
  |
  | Ex. Log2_32_Ceil(32) == 5, Log2_32_Ceil(1) == 0, Log2_32_Ceil(6) == 3
  */
#[inline] pub fn log2_32_ceil(value: u32) -> u32 {
    
    todo!();
        /*
            return static_cast<unsigned>(32 - countLeadingZeros(Value - 1));
        */
}

/**
  | Return the ceil log base 2 of the specified
  | value, 64 if the value is zero.
  |
  | (64 bit edition.)
  */
#[inline] pub fn log2_64_ceil(value: u64) -> u32 {
    
    todo!();
        /*
            return static_cast<unsigned>(64 - countLeadingZeros(Value - 1));
        */
}

/**
  | Return the greatest common divisor
  | of the values using Euclid's algorithm.
  |
  */
#[inline] pub fn greatest_common_divisor64(A: u64, B: u64) -> u64 {
    
    todo!();
        /*
            while (B) {
        uint64_t T = B;
        B = A % B;
        A = T;
      }
      return A;
        */
}

/**
  | This function takes a 64-bit integer
  | and returns the bit equivalent double.
  |
  */
#[inline] pub fn bits_to_double(bits: u64) -> f64 {
    
    todo!();
        /*
            double D;
      static_assert(sizeof(uint64_t) == sizeof(double), "Unexpected type sizes");
      memcpy(&D, &Bits, sizeof(Bits));
      return D;
        */
}

/**
  | This function takes a 32-bit integer
  | and returns the bit equivalent float.
  |
  */
#[inline] pub fn bits_to_float(bits: u32) -> f32 {
    
    todo!();
        /*
            // TODO: Use bit_cast once C++20 becomes available.
      float F;
      static_assert(sizeof(uint32_t) == sizeof(float), "Unexpected type sizes");
      memcpy(&F, &Bits, sizeof(Bits));
      return F;
        */
}

/**
  | This function takes a double and returns the
  | bit equivalent 64-bit integer.
  |
  | Note that copying doubles around changes the
  | bits of NaNs on some hosts, notably x86, so
  | this routine cannot be used if these bits are
  | needed.
  */
#[inline] pub fn double_to_bits(double: f64) -> u64 {
    
    todo!();
        /*
            uint64_t Bits;
      static_assert(sizeof(uint64_t) == sizeof(double), "Unexpected type sizes");
      memcpy(&Bits, &Double, sizeof(Double));
      return Bits;
        */
}

/**
  | This function takes a float and returns the
  | bit equivalent 32-bit integer.
  |
  | Note that copying floats around changes the
  | bits of NaNs on some hosts, notably x86, so
  | this routine cannot be used if these bits are
  | needed.
  |
  */
#[inline] pub fn float_to_bits(float: f32) -> u32 {
    
    todo!();
        /*
            uint32_t Bits;
      static_assert(sizeof(uint32_t) == sizeof(float), "Unexpected type sizes");
      memcpy(&Bits, &Float, sizeof(Float));
      return Bits;
        */
}

/**
  | A and B are either alignments or
  | offsets. Return the minimum alignment that may
  | be assumed after adding the two together.
  |
  */
#[inline] pub const fn min_align(A: u64, B: u64) -> u64 {
    
    todo!();
        /*
            // The largest power of 2 that divides both A and B.
      //
      // Replace "-Value" by "1+~Value" in the following commented code to avoid
      // MSVC warning C4146
      //    return (A | B) & -(A | B);
      return (A | B) & (1 + ~(A | B));
        */
}

/**
  | Aligns \c Addr to \c Alignment bytes, rounding
  | up.
  |
  | Alignment should be a power of two.  This
  | method rounds up, so alignAddr(7, 4) == 8 and
  | alignAddr(8, 4) == 8.
  |
  */
#[inline] pub fn align_addr(
    addr:      *const c_void,
    alignment: usize) -> libc::uintptr_t {

    todo!();
        /*
            assert(
          Alignment && isPowerOf2_64((uint64_t)Alignment) &&
          "Alignment is not a power of two!");

      assert((uintptr_t)Addr + Alignment - 1 >= (uintptr_t)Addr);

      return (((uintptr_t)Addr + Alignment - 1) & ~(uintptr_t)(Alignment - 1));
        */
}

/**
  | Returns the necessary adjustment for
  | aligning \c Ptr to \c Alignment bytes,
  | rounding up.
  |
  */
#[inline] pub fn alignment_adjustment(
    ptr:       *const c_void,
    alignment: usize) -> usize {
    
    todo!();
        /*
            return alignAddr(Ptr, Alignment) - (uintptr_t)Ptr;
        */
}

/**
  | Returns the next power of two (in 64-bits)
  | that is strictly greater than A.
  |
  | Returns zero on overflow.
  |
  */
#[inline] pub fn next_power_of2(A: u64) -> u64 {
    
    todo!();
        /*
            A |= (A >> 1);
      A |= (A >> 2);
      A |= (A >> 4);
      A |= (A >> 8);
      A |= (A >> 16);
      A |= (A >> 32);
      return A + 1;
        */
}

/**
  | Returns the power of two which is less than or
  | equal to the given value.
  |
  | Essentially, it is a floor operation across
  | the domain of powers of two.
  |
  */
#[inline] pub fn power_of_2floor(A: u64) -> u64 {
    
    todo!();
        /*
            if (!A)
        return 0;
      return 1ull << (63 - countLeadingZeros(A, ZeroBehavior::Undefined));
        */
}

/**
  | Returns the power of two which is greater than
  | or equal to the given value.
  |
  | Essentially, it is a ceil operation across the
  | domain of powers of two.
  |
  */
#[inline] pub fn power_of_2ceil(A: u64) -> u64 {
    
    todo!();
        /*
            if (!A)
        return 0;
      return NextPowerOf2(A - 1);
        */
}

/**
  | Returns the next integer (mod 2**64) that is greater than or equal to
  | \p Value and is a multiple of \p Align. \p Align must be non-zero.
  |
  | If non-zero \p Skew is specified, the return value will be a minimal
  | integer that is greater than or equal to \p Value and equal to
  | \p Align * N + \p Skew for some integer N. If \p Skew is larger than
  | \p Align, its value is adjusted to '\p Skew mod \p Align'.
  |
  | Examples:
  | \code
  |   alignTo(5, 8) = 8
  |   alignTo(17, 8) = 24
  |   alignTo(~0LL, 8) = 0
  |   alignTo(321, 255) = 510
  |
  |   alignTo(5, 8, 7) = 7
  |   alignTo(17, 8, 1) = 17
  |   alignTo(~0LL, 8, 3) = 3
  |   alignTo(321, 255, 42) = 552
  | \endcode
  */
#[inline] pub fn align_to_a(
        value: u64,
        align: u64,
        skew:  Option<u64>) -> u64 {

    let skew: u64 = skew.unwrap_or(0);

    todo!();
        /*
            assert(Align != 0u && "Align can't be 0.");
      Skew %= Align;
      return (Value + Align - 1 - Skew) / Align * Align + Skew;
        */
}

/**
  | Returns the next integer (mod 2**64) that is
  | greater than or equal to
  |
  | \p Value and is a multiple of \c Align. \c
  | Align must be non-zero.
  |
  */
#[inline] pub const fn align_to_b<const Align: u64>(value: u64) -> u64 {

    todo!();
        /*
            static_assert(Align != 0u, "Align must be non-zero");
      return (Value + Align - 1) / Align * Align;
        */
}

/// Returns the integer ceil(Numerator / Denominator).
///
#[inline] pub fn divide_ceil(
        numerator:   u64,
        denominator: u64) -> u64 {
    
    todo!();
        /*
            return alignTo(Numerator, Denominator) / Denominator;
        */
}

/**
  | \c alignTo for contexts where a constant
  | expression is required.
  |
  | \sa alignTo
  |
  | \todo FIXME: remove when \c constexpr becomes
  | really \c constexpr
  |
  */
lazy_static!{
    /*
    template <uint64_t Align>
    struct AlignTo {

      static_assert(Align != 0u, "Align must be non-zero");
      template <uint64_t Value>
      struct from_value {
        static const uint64_t value = (Value + Align - 1) / Align * Align;
      };
    };
    */
}


/**
  | Returns the largest uint64_t less than or
  | equal to \p Value and is
  |
  | \p Skew mod \p Align. \p Align must be
  | non-zero
  |
  */
#[inline] pub fn align_down(
        value: u64,
        align: u64,
        skew:  Option<u64>) -> u64 {

    let skew: u64 = skew.unwrap_or(0);

    todo!();
        /*
            assert(Align != 0u && "Align can't be 0.");
      Skew %= Align;
      return (Value - Skew) / Align * Align + Skew;
        */
}


/**
  | Returns the offset to the next integer (mod
  | 2**64) that is greater than or equal to \p
  | Value and is a multiple of \p Align. \p Align
  | must be non-zero.
  |
  */
#[inline] pub fn offset_to_alignment(
        value: u64,
        align: u64) -> u64 {
    
    todo!();
        /*
            return alignTo(Value, Align) - Value;
        */
}


/**
  | Sign-extend the number in the bottom B bits of
  | X to a 32-bit integer.
  |
  | Requires 0 < B <= 32.
  |
  */
#[inline] pub const fn sign_extend32_a<const B: u32>(X: u32) -> i32 {

    todo!();
        /*
            static_assert(B > 0, "Bit width can't be 0.");
      static_assert(B <= 32, "Bit width out of range.");
      return int32_t(X << (32 - B)) >> (32 - B);
        */
}

/**
  | Sign-extend the number in the bottom B bits of
  | X to a 32-bit integer.
  |
  | Requires 0 < B < 32.
  */
#[inline] pub fn sign_extend32_b(X: u32, B: u32) -> i32 {
    
    todo!();
        /*
            assert(B > 0 && "Bit width can't be 0.");
      assert(B <= 32 && "Bit width out of range.");
      return int32_t(X << (32 - B)) >> (32 - B);
        */
}

/**
  | Sign-extend the number in the bottom B bits of
  | X to a 64-bit integer.
  |
  | Requires 0 < B < 64.
  |
  */
#[inline] pub const fn sign_extend64_a<const B: u32>(x: u64) -> i64 {

    todo!();
        /*
            static_assert(B > 0, "Bit width can't be 0.");
      static_assert(B <= 64, "Bit width out of range.");
      return int64_t(x << (64 - B)) >> (64 - B);
        */
}

/**
  | Sign-extend the number in the bottom B bits of
  | X to a 64-bit integer.
  |
  | Requires 0 < B < 64.
  |
  */
#[inline] pub fn sign_extend64_b(X: u64, B: u32) -> i64 {
    
    todo!();
        /*
            assert(B > 0 && "Bit width can't be 0.");
      assert(B <= 64 && "Bit width out of range.");
      return int64_t(X << (64 - B)) >> (64 - B);
        */
}

/**
  | Subtract two unsigned integers, X and Y, of
  | type T and return the absolute value of the
  | result.
  |
  */
//typename enable_if<is_unsigned<T>::value, T>::type 
pub fn absolute_difference<T>(X: T, Y: T) -> T {

    todo!();
        /*
            return max(X, Y) - min(X, Y);
        */
}

/**
  | Add two unsigned integers, X and Y, of type T.
  | Clamp the result to the maximum representable
  | value of T on overflow.  ResultOverflowed
  | indicates if the result is larger than the
  | maximum representable value of type T.
  |
  */
// typename enable_if<is_unsigned<T>::value, T>::type 
pub fn saturating_add<T>(
        X:                 T,
        Y:                 T,
        result_overflowed: Option<*mut bool>) -> T {

    todo!();
        /*
            bool Dummy;
      bool& Overflowed = ResultOverflowed ? *ResultOverflowed : Dummy;
      // Hacker's Delight, p. 29
      T Z = X + Y;
      Overflowed = (Z < X || Z < Y);
      if (Overflowed)
        return T::max;
      else
        return Z;
        */
}

/**
  | Multiply two unsigned integers, X and Y, of
  | type T.  Clamp the result to the maximum
  | representable value of T on overflow.
  | ResultOverflowed indicates if the result is
  | larger than the maximum representable value of
  | type T.
  |
  */
// typename enable_if<is_unsigned<T>::value, T>::type
pub fn saturating_multiply<T>(
    X:                 T,
    Y:                 T,
    result_overflowed: Option<*mut bool>) -> T {

    todo!();
        /*
            bool Dummy;
      bool& Overflowed = ResultOverflowed ? *ResultOverflowed : Dummy;

      // Hacker's Delight, p. 30 has a different algorithm, but we don't use that
      // because it fails for uint16_t (where multiplication can have undefined
      // behavior due to promotion to int), and requires a division in addition
      // to the multiplication.

      Overflowed = false;

      // Log2(Z) would be either Log2Z or Log2Z + 1.
      // Special case: if X or Y is 0, Log2_64 gives -1, and Log2Z
      // will necessarily be less than Log2Max as desired.
      int Log2Z = Log2_64(X) + Log2_64(Y);
      const T Max = T::max;
      int Log2Max = Log2_64(Max);
      if (Log2Z < Log2Max) {
        return X * Y;
      }
      if (Log2Z > Log2Max) {
        Overflowed = true;
        return Max;
      }

      // We're going to use the top bit, and maybe overflow one
      // bit past it. Multiply all but the bottom bit then add
      // that on at the end.
      T Z = (X >> 1) * Y;
      if (Z & ~(Max >> 1)) {
        Overflowed = true;
        return Max;
      }
      Z <<= 1;
      if (X & 1)
        return SaturatingAdd(Z, Y, ResultOverflowed);

      return Z;
        */
}

/**
  | Multiply two unsigned integers, X and Y, and
  | add the unsigned integer, A to the product.
  |
  | Clamp the result to the maximum representable
  | value of T on overflow.
  |
  | ResultOverflowed indicates if the result is
  | larger than the maximum representable value of
  | type T.
  |
  */
// typename enable_if<is_unsigned<T>::value, T>::type
pub fn saturating_multiply_add<T>(
        X:                 T,
        Y:                 T,
        A:                 T,
        result_overflowed: Option<*mut bool>) -> T {

    todo!();
        /*
            bool Dummy;
      bool& Overflowed = ResultOverflowed ? *ResultOverflowed : Dummy;

      T Product = SaturatingMultiply(X, Y, &Overflowed);
      if (Overflowed)
        return Product;

      return SaturatingAdd(A, Product, &Overflowed);
        */
}

/// Use this rather than HUGE_VALF; the latter
/// causes warnings on MSVC.
///
lazy_static!{
    /*
    extern const float huge_valf;
    */
}
