crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/vsx/vec256_common_vsx.h]

define_clamp_funcs!{qui8}
define_clamp_funcs!{qi8}
define_clamp_funcs!{qi32}
define_clamp_funcs!{i16}
define_clamp_funcs!{i32}
define_clamp_funcs!{i64}
define_clamp_funcs!{f32}
define_clamp_funcs!{f64}

#[inline(always)]
pub fn fmadd_f64(
    a: &Vectorized<f64>,
    b: &Vectorized<f64>,
    c: &Vectorized<f64>) -> Vectorized<f64> {

    todo!();
        /*
            return Vectorized<double>{
          vec_madd(a.vec0(), b.vec0(), c.vec0()),
          vec_madd(a.vec1(), b.vec1(), c.vec1())};
        */
}

#[inline(always)]
pub fn fmadd_i64(
    a: &Vectorized<i64>,
    b: &Vectorized<i64>,
    c: &Vectorized<i64>) -> Vectorized<i64> {
    
    todo!();
        /*
            return Vectorized<i64>{
          a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
        */
}

#[inline(always)]
pub fn fmadd_i32(
    a: &Vectorized<i32>,
    b: &Vectorized<i32>,
    c: &Vectorized<i32>) -> Vectorized<i32> {
    
    todo!();
        /*
            return Vectorized<i32>{
          a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
        */
}

#[inline(always)]
pub fn fmadd_i16(
    a: &Vectorized<i16>,
    b: &Vectorized<i16>,
    c: &Vectorized<i16>) -> Vectorized<i16> {
    
    todo!();
        /*
            return Vectorized<i16>{
          a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
        */
}

define_reinterpret_cast_to_all_funcs!{f32}
define_reinterpret_cast_to_all_funcs!{f64}
define_reinterpret_cast_to_all_funcs!{i64}
define_reinterpret_cast_to_all_funcs!{i32}
define_reinterpret_cast_to_all_funcs!{i16}

#[inline(always)]
pub fn convert_double_to_int_of_same_size(src: &Vectorized<f64>) -> Vectorized<i64> {
    
    todo!();
        /*
            return Vectorized<i64>{vec_signed(src.vec0()), vec_signed(src.vec1())};
        */
}

#[inline(always)]
pub fn convert_float_to_int_of_same_size(src: &Vectorized<f32>) -> Vectorized<i32> {
    
    todo!();
        /*
            return Vectorized<i32>{vec_signed(src.vec0()), vec_signed(src.vec1())};
        */
}

#[inline] pub fn convert_i32(
    src: *const i32,
    dst: *mut f32,
    n:   i64)  {
    
    todo!();
        /*
            // i32 and float have same size
      i64 i;
      for (i = 0; i <= (n - Vectorized<float>::size()); i += Vectorized<float>::size()) {
        const i32* src_a = src + i;
        float* dst_a = dst + i;
        vint32 input_vec0 = vec_vsx_ld(offset0, reinterpret_cast<const vint32*>(src_a));
        vint32 input_vec1 =
            vec_vsx_ld(offset16, reinterpret_cast<const vint32*>(src_a));
        vfloat32 c0 = vec_float(input_vec0);
        vfloat32 c1 = vec_float(input_vec1);
        vec_vsx_st(c0, offset0, dst_a);
        vec_vsx_st(c1, offset16, dst_a);
      }

      for (; i < n; i++) {
        dst[i] = static_cast<float>(src[i]);
      }
        */
}

#[inline] pub fn convert_i64(
    src: *const i64,
    dst: *mut f64,
    n:   i64)  {

    todo!();
        /*
            i64 i;
      for (i = 0; i <= (n - Vectorized<double>::size()); i += Vectorized<double>::size()) {
        const i64* src_a = src + i;
        double* dst_a = dst + i;
        vint64 input_vec0 =
            vec_vsx_ld(offset0, reinterpret_cast<const vint64*>(src_a));
        vint64 input_vec1 =
            vec_vsx_ld(offset16, reinterpret_cast<const vint64*>(src_a));
        vfloat64 c0 = vec_double(input_vec0);
        vfloat64 c1 = vec_double(input_vec1);
        vec_vsx_st(c0, offset0, reinterpret_cast<double*>(dst_a));
        vec_vsx_st(c1, offset16, reinterpret_cast<double*>(dst_a));
      }
      for (; i < n; i++) {
        dst[i] = static_cast<double>(src[i]);
      }
        */
}

#[inline] pub fn interleave2_double(
    a: &Vectorized<f64>,
    b: &Vectorized<f64>) -> (Vectorized<f64>,Vectorized<f64>) {
    
    todo!();
        /*
            // inputs:
      //   a      = {a0, a1, a2, a3}
      //   b      = {b0, b1, b2, b3}

      vfloat64 ab00 = vec_xxpermdi(a.vec0(), b.vec0(), 0);
      vfloat64 ab11 = vec_xxpermdi(a.vec0(), b.vec0(), 3);
      vfloat64 ab2_00 = vec_xxpermdi(a.vec1(), b.vec1(), 0);
      vfloat64 ab2_11 = vec_xxpermdi(a.vec1(), b.vec1(), 3);
      //   return {a0, b0, a1, b1}
      //          {a2, b2, a3, b3}
      return make_pair(
          Vectorized<double>{ab00, ab11}, Vectorized<double>{ab2_00, ab2_11});
        */
}

#[inline] pub fn deinterleave2_double(
    a: &Vectorized<f64>,
    b: &Vectorized<f64>) -> (Vectorized<f64>,Vectorized<f64>) {
    
    todo!();
        /*
            // inputs:
      //   a = {a0, b0, a1, b1}
      //   b = {a2, b2, a3, b3}
      vfloat64 aa01 = vec_xxpermdi(a.vec0(), a.vec1(), 0);
      vfloat64 aa23 = vec_xxpermdi(b.vec0(), b.vec1(), 0);

      vfloat64 bb_01 = vec_xxpermdi(a.vec0(), a.vec1(), 3);
      vfloat64 bb_23 = vec_xxpermdi(b.vec0(), b.vec1(), 3);

      // swap lanes:
      //   return {a0, a1, a2, a3}
      //          {b0, b1, b2, b3}
      return make_pair(
          Vectorized<double>{aa01, aa23}, Vectorized<double>{bb_01, bb_23});
        */
}

#[inline] pub fn interleave2_float(
    a: &Vectorized<f32>,
    b: &Vectorized<f32>) -> (Vectorized<f32>,Vectorized<f32>) {
    
    todo!();
        /*
            // inputs:
      //   a = {a0, a1, a2, a3,, a4, a5, a6, a7}
      //   b = {b0, b1, b2, b3,, b4, b5, b6, b7}

      vfloat32 ab0011 = vec_mergeh(a.vec0(), b.vec0());
      vfloat32 ab2233 = vec_mergel(a.vec0(), b.vec0());

      vfloat32 ab2_0011 = vec_mergeh(a.vec1(), b.vec1());
      vfloat32 ab2_2233 = vec_mergel(a.vec1(), b.vec1());
      // group cols crossing lanes:
      //   return {a0, b0, a1, b1,, a2, b2, a3, b3}
      //          {a4, b4, a5, b5,, a6, b6, a7, b7}

      return make_pair(
          Vectorized<float>{ab0011, ab2233}, Vectorized<float>{ab2_0011, ab2_2233});
        */
}

#[inline] pub fn deinterleave2_float(
    a: &Vectorized<f32>,
    b: &Vectorized<f32>) -> (Vectorized<f32>,Vectorized<f32>) {
    
    todo!();
        /*
            // inputs:
      //   a = {a0, b0, a1, b1,, a2, b2, a3, b3}
      //   b = {a4, b4, a5, b5,, a6, b6, a7, b7}

      // {a0,a2,b0,b2} {a1,a3,b1,b3}
      vfloat32 a0a2b0b2 = vec_mergeh(a.vec0(), a.vec1());
      vfloat32 a1a3b1b3 = vec_mergel(a.vec0(), a.vec1());

      vfloat32 aa0123 = vec_mergeh(a0a2b0b2, a1a3b1b3);
      vfloat32 bb0123 = vec_mergel(a0a2b0b2, a1a3b1b3);

      vfloat32 a0a2b0b2_2 = vec_mergeh(b.vec0(), b.vec1());
      vfloat32 a1a3b1b3_2 = vec_mergel(b.vec0(), b.vec1());

      vfloat32 aa0123_2 = vec_mergeh(a0a2b0b2_2, a1a3b1b3_2);
      vfloat32 bb0123_2 = vec_mergel(a0a2b0b2_2, a1a3b1b3_2);

      // it could be done with vec_perm ,too
      // swap lanes:
      //   return {a0, a1, a2, a3,, a4, a5, a6, a7}
      //          {b0, b1, b2, b3,, b4, b5, b6, b7}

      return make_pair(
          Vectorized<float>{aa0123, aa0123_2}, Vectorized<float>{bb0123, bb0123_2});
        */
}
