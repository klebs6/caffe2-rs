crate::ix!();

use std::process::Output;


/**
  Similar to Axpy that calculate y = a * x + y, but allowing x and y to be
  of different data types.
  It also provides a performance optimization hint (use_a) to see if a is going
  to be 1 or not.
  */
#[inline] pub fn typed_axpy<IN, OUT>(
    n: i32,
    a: OUT,
    x: *const IN,
    y: *mut OUT)  {

    todo!();
    /*
    
    */
}

#[inline] pub fn typed_axpy__base(
    n: i32,
    a: f32,
    x: *const f32,
    y: *mut f32)  
{
    todo!();
    /*
        for (int i = 0; i < N; ++i) {
        y[i] += a * x[i];
      }
    */
}

#[inline] pub fn typed_axpyf32f32(
    n: i32,
    a: f32,
    x: *const f32,
    y: *mut f32)  {
    
    todo!();
    /*
        AVX2_FMA_DO(TypedAxpy, N, a, x, y);
      AVX_F16C_DO(TypedAxpy, N, a, x, y);
      BASE_DO(TypedAxpy, N, a, x, y);
    */
}

#[inline] pub fn typed_axpy_halffloat__base(
    n: i32,
    a: f32,
    x: *const f16,
    y: *mut f32)
{
    todo!();
    /*
        for (int i = 0; i < N; ++i) {
        union {
          uint32_t intval;
          float floatval;
        } t1;
        uint32_t t2, t3;
        t1.intval = x[i].x & 0x7fff; // Non-sign bits
        t2 = x[i].x & 0x8000; // Sign bit
        t3 = x[i].x & 0x7c00; // Exponent
        t1.intval <<= 13; // Align mantissa on MSB
        t2 <<= 16; // Shift sign bit into position
        t1.intval += 0x38000000; // Adjust bias
        t1.intval = (t3 == 0 ? 0 : t1.intval); // Denormals-as-zero
        t1.intval |= t2; // Re-insert sign bit
        y[i] += t1.floatval * a;
      }
    */
}

#[inline] pub fn typed_axpyf16f32(
    n: i32,
    a: f32,
    x: *const f16,
    y: *mut f32)  {
    
    todo!();
    /*
        AVX2_FMA_DO(TypedAxpyHalffloat, N, a, x, y);
      AVX_F16C_DO(TypedAxpyHalffloat, N, a, x, y);
      BASE_DO(TypedAxpyHalffloat, N, a, x, y);
    */
}


#[inline] pub fn typed_axpy_uint8_float__base(
    n: i32,
    a: f32,
    x: *const u8,
    y: *mut f32)  
{
    todo!();
    /*
        for (int i = 0; i < N; ++i) {
        y[i] += (float)(x[i]) * a;
      }
    */
}

#[inline] pub fn typed_axpyu8f32(
    n: i32,
    a: f32,
    x: *const u8,
    y: *mut f32)  {
    
    todo!();
    /*
        AVX2_FMA_DO(TypedAxpy_uint8_float, N, a, x, y);
      BASE_DO(TypedAxpy_uint8_float, N, a, x, y);
    */
}
