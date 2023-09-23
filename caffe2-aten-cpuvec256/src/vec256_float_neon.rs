/*!
  | Sleef offers vectorized versions of some transcedentals
  | such as sin, cos, tan etc..
  |
  | However for now opting for STL, since we are not building
  | with Sleef for mobile yet.
  |
  | Right now contains only aarch64 implementation.
  |
  | Due to follow two reasons aarch32 is not currently supported.
  |
  | 1. Due to difference in ISA been aarch32 and aarch64, intrinsics
  |    that work for aarch64 dont work for aarch32.
  |
  | 2. Android NDK r21 has problems with compiling aarch32.
  |    Clang seg faults.
  |    https://github.com/android/ndk/issues/1248
  |    https://bugs.llvm.org/show_bug.cgi?id=45824
  |
  | Most likely we will do aarch32 support with inline asm.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/vec256_float_neon.h]

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
pub struct BlendRegs<const index: i32,const mask_val: bool> {

}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
impl<const index: i32,const mask_val: bool> BlendRegs<index,mask_val> {
    
    pub fn impl_mask_val_true(
        a:   &float32x4_t,
        b:   &float32x4_t,
        res: &mut float32x4_t) -> float32x4_t {
        
        todo!();
        /*
            return vsetq_lane_f32(vgetq_lane_f32(b, index), res, index);
        */
    }
    
    pub fn impl_mask_val_false(
        a:   &float32x4_t,
        b:   &float32x4_t,
        res: &mut float32x4_t) -> float32x4_t {
        
        todo!();
        /*
            return vsetq_lane_f32(vgetq_lane_f32(a, index), res, index);
        */
    }
}


#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
pub struct VectorizedFloat {
    values: float32x4x2_t,
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
pub mod vectorized_float {
    pub type ValueType = f32;
    pub type SizeType  = i32;
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
impl VectorizedFloat {
    
    pub fn size() -> SizeType {
        
        todo!();
        /*
            return 8;
        */
    }
    
    pub fn new(v: float32x4x2_t) -> Self {
    
        todo!();
        /*
        : values(v),

        
        */
    }
    
    pub fn new(val: f32) -> Self {
    
        todo!();
        /*


            : values{vdupq_n_f32(val), vdupq_n_f32(val) }
        */
    }
    
    pub fn new(
        val0: f32,
        val1: f32,
        val2: f32,
        val3: f32,
        val4: f32,
        val5: f32,
        val6: f32,
        val7: f32) -> Self {
    
        todo!();
        /*


            : values{val0, val1, val2, val3, val4, val5, val6, val7}
        */
    }
    
    pub fn new(
        val0: float32x4_t,
        val1: float32x4_t) -> Self {
    
        todo!();
        /*


            : values{val0, val1}
        */
    }
    
    pub fn operator_float_32x4x2_t(&self) -> float32x4x2_t {
        
        todo!();
        /*
            return values;
        */
    }
    
    pub fn blend<const mask: i64>(
        a: &Vectorized<f32>,
        b: &Vectorized<f32>) -> Vectorized<f32> {
    
        todo!();
        /*
            Vectorized<float> vec;
        // 0.
        vec.values.val[0] =
          BlendRegs<0, (mask & 0x01)!=0>::impl(
              a.values.val[0], b.values.val[0], vec.values.val[0]);
        vec.values.val[0] =
          BlendRegs<1, (mask & 0x02)!=0>::impl(
              a.values.val[0], b.values.val[0], vec.values.val[0]);
        vec.values.val[0] =
          BlendRegs<2, (mask & 0x04)!=0>::impl(
              a.values.val[0], b.values.val[0], vec.values.val[0]);
        vec.values.val[0] =
          BlendRegs<3, (mask & 0x08)!=0>::impl(
              a.values.val[0], b.values.val[0], vec.values.val[0]);
        // 1.
        vec.values.val[1] =
          BlendRegs<0, (mask & 0x10)!=0>::impl(
              a.values.val[1], b.values.val[1], vec.values.val[1]);
        vec.values.val[1] =
          BlendRegs<1, (mask & 0x20)!=0>::impl(
              a.values.val[1], b.values.val[1], vec.values.val[1]);
        vec.values.val[1] =
          BlendRegs<2, (mask & 0x40)!=0>::impl(
              a.values.val[1], b.values.val[1], vec.values.val[1]);
        vec.values.val[1] =
          BlendRegs<3, (mask & 0x80)!=0>::impl(
              a.values.val[1], b.values.val[1], vec.values.val[1]);
        return vec;
        */
    }
    
    pub fn blendv(
        a:    &Vectorized<f32>,
        b:    &Vectorized<f32>,
        mask: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            // TODO
        // NB: This requires that each value, i.e., each uint value,
        // of the mask either all be zeros or all be 1s.
        // We perhaps need some kind of an assert?
        // But that will affect performance.
        Vectorized<float> vec(mask.values);
        vec.values.val[0] = vbslq_f32(
            vreinterpretq_u32_f32(vec.values.val[0]),
            b.values.val[0],
            a.values.val[0]);
        vec.values.val[1] = vbslq_f32(
            vreinterpretq_u32_f32(vec.values.val[1]),
            b.values.val[1],
            a.values.val[1]);
        return vec;
        */
    }
    
    pub fn arange<step_t>(
        base: f32,
        step: Step) -> Vectorized<f32> {
        let base: f32 = base.unwrap_or(0.0);
        let step: Step = step.unwrap_or(1);
        todo!();
        /*
            const Vectorized<float> base_vec(base);
        const Vectorized<float> step_vec(step);
        const Vectorized<float> step_sizes(0, 1, 2, 3, 4, 5, 6, 7);
        return fmadd(step_sizes, step_vec, base_vec);
        */
    }
    
    pub fn set(
        a:     &Vectorized<f32>,
        b:     &Vectorized<f32>,
        count: i64) -> Vectorized<f32> {
        let count: i64 = count.unwrap_or(size);

        todo!();
        /*
            switch (count) {
          case 0:
            return a;
          case 1:
            {
              Vectorized<float> vec;
              static uint32x4_t mask_low = {0xFFFFFFFF, 0x0, 0x0, 0x0};
              vec.values.val[0] = vreinterpretq_f32_u32(mask_low);
              vec.values.val[1] = a.values.val[1];
              vec.values.val[0] = vbslq_f32(
                  vreinterpretq_u32_f32(vec.values.val[0]),
                  b.values.val[0],
                  a.values.val[0]);
              return vec;
            }
          case 2:
            {
              Vectorized<float> vec;
              static uint32x4_t mask_low = {0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0};
              vec.values.val[0] = vreinterpretq_f32_u32(mask_low);
              vec.values.val[1] = a.values.val[1];
              vec.values.val[0] = vbslq_f32(
                  vreinterpretq_u32_f32(vec.values.val[0]),
                  b.values.val[0],
                  a.values.val[0]);
              return vec;
            }
          case 3:
            {
              Vectorized<float> vec;
              static uint32x4_t mask_low = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0};
              vec.values.val[0] = vreinterpretq_f32_u32(mask_low);
              vec.values.val[1] = a.values.val[1];
              vec.values.val[0] = vbslq_f32(
                  vreinterpretq_u32_f32(vec.values.val[0]),
                  b.values.val[0],
                  a.values.val[0]);
              return vec;
            }
          case 4:
            return Vectorized<float>(b.values.val[0], a.values.val[1]);
          case 5:
            {
              Vectorized<float> vec;
              static uint32x4_t mask_high = {0xFFFFFFFF, 0x0, 0x0, 0x0};
              vec.values.val[0] = b.values.val[0];
              vec.values.val[1] = vreinterpretq_f32_u32(mask_high);
              vec.values.val[1] = vbslq_f32(
                  vreinterpretq_u32_f32(vec.values.val[1]),
                  b.values.val[1],
                  a.values.val[1]);
              return vec;
            }
          case 6:
            {
              Vectorized<float> vec;
              static uint32x4_t mask_high = {0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0};
              vec.values.val[0] = b.values.val[0];
              vec.values.val[1] = vreinterpretq_f32_u32(mask_high);
              vec.values.val[1] = vbslq_f32(
                  vreinterpretq_u32_f32(vec.values.val[1]),
                  b.values.val[1],
                  a.values.val[1]);
              return vec;
            }
          case 7:
            {
              Vectorized<float> vec;
              static uint32x4_t mask_high = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0};
              vec.values.val[0] = b.values.val[0];
              vec.values.val[1] = vreinterpretq_f32_u32(mask_high);
              vec.values.val[1] = vbslq_f32(
                  vreinterpretq_u32_f32(vec.values.val[1]),
                  b.values.val[1],
                  a.values.val[1]);
              return vec;
            }
        }
        return b;
        */
    }
    
    pub fn loadu(
        ptr:   *const void,
        count: i64) -> Vectorized<f32> {
        let count: i64 = count.unwrap_or(size);

        todo!();
        /*
            if (count == size()) {
          return vld1q_f32_x2(reinterpret_cast<const float*>(ptr));
        }
        else if (count == (size() >> 1)) {
          Vectorized<float> res;
          res.values.val[0] = vld1q_f32(reinterpret_cast<const float*>(ptr));
          res.values.val[1] = vdupq_n_f32(0.f);
          return res;
        }
        else {
          __at_align32__ float tmp_values[size()];
          for (auto i = 0; i < size(); ++i) {
            tmp_values[i] = 0.0;
          }
          memcpy(
              tmp_values,
              reinterpret_cast<const float*>(ptr),
              count * sizeof(float));
          return vld1q_f32_x2(reinterpret_cast<const float*>(tmp_values));
        }
        */
    }
    
    pub fn store(&self, 
        ptr:   *mut void,
        count: i64)  {
        let count: i64 = count.unwrap_or(size);

        todo!();
        /*
            if (count == size()) {
          vst1q_f32_x2(reinterpret_cast<float*>(ptr), values);
        }
        else if (count == (size() >> 1)) {
          vst1q_f32(reinterpret_cast<float*>(ptr), values.val[0]);
        }
        else {
          float tmp_values[size()];
          vst1q_f32_x2(reinterpret_cast<float*>(tmp_values), values);
          memcpy(ptr, tmp_values, count * sizeof(float));
        }
        */
    }
    
    #[inline] pub fn get_low(&self) -> &float32x4_t {
        
        todo!();
        /*
            return values.val[0];
        */
    }
    
    #[inline] pub fn get_low(&mut self) -> &mut float32x4_t {
        
        todo!();
        /*
            return values.val[0];
        */
    }
    
    #[inline] pub fn get_high(&self) -> &float32x4_t {
        
        todo!();
        /*
            return values.val[1];
        */
    }
    
    #[inline] pub fn get_high(&mut self) -> &mut float32x4_t {
        
        todo!();
        /*
            return values.val[1];
        */
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
impl Index<i32> for VectorizedFloat {

    type Output = f32;
    
    /**
      | Very slow implementation of indexing.
      |
      | Only required because vec256_qint refers to this.
      |
      | Once we specialize that implementation for ARM
      | this should be removed. TODO (kimishpatel)
      */
    #[inline] fn index(&self, idx: i32) -> &Self::Output {
        todo!();
        /*
            __at_align32__ float tmp[size()];
        store(tmp);
        return tmp[idx];
        */
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
impl IndexMut<i32> for VectorizedFloat {
    
    #[inline] fn index_mut(&mut self, idx: i32) -> &mut Self::Output {
        todo!();
        /*
            __at_align32__ float tmp[size()];
        store(tmp);
        return tmp[idx];
        */
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
impl VectorizedFloat {
    
    /**
      | For boolean version where we want to
      | if any 1/all zero etc. can be done faster
      | in a different way.
      |
      */
    pub fn zero_mask(&self) -> i32 {
        
        todo!();
        /*
            __at_align32__ float tmp[size()];
        store(tmp);
        int mask = 0;
        for (int i = 0; i < size(); ++ i) {
          if (tmp[i] == 0.f) {
            mask |= (1 << i);
          }
        }
        return mask;
        */
    }
    
    pub fn isnan(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            __at_align32__ float tmp[size()];
        __at_align32__ float res[size()];
        store(tmp);
        for (int i = 0; i < size(); i++) {
          if (_isnan(tmp[i])) {
            memset(static_cast<void*>(&res[i]), 0xFF, sizeof(float));
          } else {
            memset(static_cast<void*>(&res[i]), 0, sizeof(float));
          }
        }
        return loadu(res);
      }{
        */
    }
    
    pub fn map(&self, f: fn(_0: f32) -> f32) -> Vectorized<f32> {
        
        todo!();
        /*
            __at_align32__ float tmp[size()];
        store(tmp);
        for (i64 i = 0; i < size(); i++) {
          tmp[i] = f(tmp[i]);
        }
        return loadu(tmp);
        */
    }
    
    pub fn abs(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(vabsq_f32(values.val[0]), vabsq_f32(values.val[1]));
        */
    }
    
    pub fn angle(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            auto zero = Vectorized<float>(0);
        auto pi = Vectorized<float>(pi<float>);
        auto tmp = blendv(zero, pi, *this < zero);
        return blendv(tmp, *this, isnan());
        */
    }
    
    pub fn real(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return *this;
        */
    }
    
    pub fn imag(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(0.f);
        */
    }
    
    pub fn conj(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return *this;
        */
    }
    
    pub fn acos(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(acos);
        */
    }
    
    pub fn asin(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(asin);
        */
    }
    
    pub fn atan(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(atan);
        */
    }
    
    pub fn atan2(&self, exp: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            __at_align32__ float tmp[size()];
        __at_align32__ float tmp_exp[size()];
        store(tmp);
        exp.store(tmp_exp);
        for (i64 i = 0; i < size(); i++) {
          tmp[i] = atan2(tmp[i], tmp_exp[i]);
        }
        return loadu(tmp);
        */
    }
    
    pub fn copysign(&self, sign: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            __at_align32__ float tmp[size()];
        __at_align32__ float tmp_sign[size()];
        store(tmp);
        sign.store(tmp_sign);
        for (size_type i = 0; i < size(); i++) {
          tmp[i] = copysign(tmp[i], tmp_sign[i]);
        }
        return loadu(tmp);
        */
    }
    
    pub fn erf(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(erf);
        */
    }
    
    pub fn erfc(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(erfc);
        */
    }
    
    pub fn erfinv(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(calc_erfinv);
        */
    }
    
    pub fn exp(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(exp);
        */
    }
    
    pub fn expm1(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(expm1);
        */
    }
    
    pub fn fmod(&self, q: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            __at_align32__ float tmp[size()];
        __at_align32__ float tmp_q[size()];
        store(tmp);
        q.store(tmp_q);
        for (i64 i = 0; i < size(); i++) {
          tmp[i] = fmod(tmp[i], tmp_q[i]);
        }
        return loadu(tmp);
        */
    }
    
    pub fn hypot(&self, b: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            __at_align32__ float tmp[size()];
        __at_align32__ float tmp_b[size()];
        store(tmp);
        b.store(tmp_b);
        for (i64 i = 0; i < size(); i++) {
          tmp[i] = hypot(tmp[i], tmp_b[i]);
        }
        return loadu(tmp);
        */
    }
    
    pub fn i0(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(calc_i0);
        */
    }
    
    pub fn i0e(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(calc_i0e);
        */
    }
    
    pub fn igamma(&self, x: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            __at_align32__ float tmp[size()];
        __at_align32__ float tmp_x[size()];
        store(tmp);
        x.store(tmp_x);
        for (i64 i = 0; i < size(); i++) {
          tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
        }
        return loadu(tmp);
        */
    }
    
    pub fn igammac(&self, x: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            __at_align32__ float tmp[size()];
        __at_align32__ float tmp_x[size()];
        store(tmp);
        x.store(tmp_x);
        for (i64 i = 0; i < size(); i++) {
          tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
        }
        return loadu(tmp);
        */
    }
    
    pub fn log(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(log);
        */
    }
    
    pub fn log10(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(log10);
        */
    }
    
    pub fn log1p(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(log1p);
        */
    }
    
    pub fn log2(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(log2);
        */
    }
    
    pub fn nextafter(&self, b: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            __at_align32__ float tmp[size()];
        __at_align32__ float tmp_b[size()];
        store(tmp);
        b.store(tmp_b);
        for (i64 i = 0; i < size(); i++) {
          tmp[i] = nextafter(tmp[i], tmp_b[i]);
        }
        return loadu(tmp);
        */
    }
    
    pub fn sin(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(sin);
        */
    }
    
    pub fn sinh(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(sinh);
        */
    }
    
    pub fn cos(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(cos);
        */
    }
    
    pub fn cosh(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(cosh);
        */
    }
    
    pub fn ceil(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(native::ceil_impl);
        */
    }
    
    pub fn floor(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(native::floor_impl);
        */
    }
    
    pub fn neg(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(
            vnegq_f32(values.val[0]),
            vnegq_f32(values.val[1]));
        */
    }
    
    pub fn round(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            // We do not use round because we would like to round midway numbers to the nearest even integer.
        return map(native::round_impl);
        */
    }
    
    pub fn tan(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(tan);
        */
    }
    
    pub fn tanh(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(tanh);
        */
    }
    
    pub fn trunc(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            float32x4_t r0 = vrndq_f32(values.val[0]);
        float32x4_t r1 = vrndq_f32(values.val[1]);
        return Vectorized<float>(r0, r1);
        */
    }
    
    pub fn lgamma(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(lgamma);
        */
    }
    
    pub fn sqrt(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(
            vsqrtq_f32(values.val[0]),
            vsqrtq_f32(values.val[1]));
        */
    }
    
    pub fn reciprocal(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            auto r0 = vdivq_f32(vdupq_n_f32(1.0f), values.val[0]);
        auto r1 = vdivq_f32(vdupq_n_f32(1.0f), values.val[1]);
        return Vectorized<float>(r0, r1);
        */
    }
    
    pub fn rsqrt(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return this->sqrt().reciprocal();
        */
    }
    
    pub fn pow(&self, exp: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            __at_align32__ float tmp[size()];
        __at_align32__ float tmp_exp[size()];
        store(tmp);
        exp.store(tmp_exp);
        for (i64 i = 0; i < size(); i++) {
          tmp[i] = pow(tmp[i], tmp_exp[i]);
        }
        return loadu(tmp);
        */
    }
}


#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
impl PartialEq for VectorizedFloat {

    fn eq(&self, x: &Self) -> bool {

        todo!();

        /*
        float32x4_t r0 =
          vreinterpretq_f32_u32(vceqq_f32(values.val[0], other.values.val[0]));
        float32x4_t r1 =
          vreinterpretq_f32_u32(vceqq_f32(values.val[1], other.values.val[1]));
        return Vectorized<float>(r0, r1);
        */
    }

    fn ne(&self, x: &Self) -> bool {

        todo!();

        /*
        float32x4_t r0 = vreinterpretq_f32_u32(
            vmvnq_u32(vceqq_f32(values.val[0], other.values.val[0])));
        float32x4_t r1 = vreinterpretq_f32_u32(
            vmvnq_u32(vceqq_f32(values.val[1], other.values.val[1])));
        return Vectorized<float>(r0, r1);
        */
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
impl PartialCmp for VectorizedFloat {

    fn partial_cmp(&self, x: &Self) -> Option<Ordering> {

        todo!();

        /*
        float32x4_t r0 =
          vreinterpretq_f32_u32(vcltq_f32(values.val[0], other.values.val[0]));
        float32x4_t r1 =
          vreinterpretq_f32_u32(vcltq_f32(values.val[1], other.values.val[1]));
        return Vectorized<float>(r0, r1);
        */
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
impl Add<&Vectorized<f32>> for Vectorized<f32>  {

    type Output = Vectorized<f32>;

    fn add(self, other: &Vectorized<f32>) -> Self::Output {

        todo!();

        /*
        float32x4_t r0 = vaddq_f32(a.get_low(), b.get_low());
        float32x4_t r1 = vaddq_f32(a.get_high(), b.get_high());
        return Vectorized<float>(r0, r1);
        */
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
impl Sub<&Vectorized<f32>> for Vectorized<f32>  {

    type Output = Vectorized<f32>;
    
    fn sub(self, other: &Vectorized<f32>) -> Self::Output {
        todo!();
        /*
            float32x4_t r0 = vsubq_f32(a.get_low(), b.get_low());
      float32x4_t r1 = vsubq_f32(a.get_high(), b.get_high());
      return Vectorized<float>(r0, r1);
        */
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
impl Mul<&Vectorized<f32>> for Vectorized<f32>  {

    type Output = Vectorized<f32>;
    
    fn mul(self, other: &Vectorized<f32>) -> Self::Output {

        todo!();

        /*
          float32x4_t r0 = vmulq_f32(a.get_low(), b.get_low());
          float32x4_t r1 = vmulq_f32(a.get_high(), b.get_high());
          return Vectorized<float>(r0, r1);
        */
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
impl Div<&Vectorized<f32>> for Vectorized<f32>  {

    type Output = Vectorized<f32>;
    
    fn div(self, other: &Vectorized<f32>) -> Self::Output {

        todo!();

        /*
          float32x4_t r0 = vdivq_f32(a.get_low(), b.get_low());
          float32x4_t r1 = vdivq_f32(a.get_high(), b.get_high());
          return Vectorized<float>(r0, r1);
        */
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
impl VectorizedFloat {

    /// frac. Implement this here so we can use
    /// subtraction
    ///
    pub fn frac(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return *this - this->trunc();
        */
    }

    /// Implements the IEEE 754 201X `maximum`
    /// operation, which propagates NaN if either
    /// input is a NaN.
    ///
    #[inline] pub fn maximum(&mut self, 
        a: &Vectorized<f32>,
        b: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            float32x4_t r0 = vmaxq_f32(a.get_low(), b.get_low());
      float32x4_t r1 = vmaxq_f32(a.get_high(), b.get_high());
      return Vectorized<float>(r0, r1);
        */
    }

    /// Implements the IEEE 754 201X `minimum`
    /// operation, which propagates NaN if either
    /// input is a NaN.
    ///
    #[inline] pub fn minimum(&mut self, 
        a: &Vectorized<f32>,
        b: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            float32x4_t r0 = vminq_f32(a.get_low(), b.get_low());
      float32x4_t r1 = vminq_f32(a.get_high(), b.get_high());
      return Vectorized<float>(r0, r1);
        */
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[inline] pub fn clamp(
        a:   &Vectorized<f32>,
        min: &Vectorized<f32>,
        max: &Vectorized<f32>) -> Vectorized<f32> {
    
    todo!();
        /*
            return minimum(max, maximum(min, a));
        */
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[inline] pub fn clamp_max(
        a:   &Vectorized<f32>,
        max: &Vectorized<f32>) -> Vectorized<f32> {
    
    todo!();
        /*
            return minimum(max, a);
        */
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[inline] pub fn clamp_min(
        a:   &Vectorized<f32>,
        min: &Vectorized<f32>) -> Vectorized<f32> {
    
    todo!();
        /*
            return maximum(min, a);
        */
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
impl BitAnd<&Vectorized<f32>> for Vectorized<f32> {

    type Output = Vectorized<f32>;
    
    fn bitand(self, other: &Vectorized<f32>) -> Self::Output {
        todo!();
        /*
            float32x4_t r0 = vreinterpretq_f32_u32(vandq_u32(
          vreinterpretq_u32_f32(a.get_low()),
          vreinterpretq_u32_f32(b.get_low())));
      float32x4_t r1 = vreinterpretq_f32_u32(vandq_u32(
          vreinterpretq_u32_f32(a.get_high()),
          vreinterpretq_u32_f32(b.get_high())));
      return Vectorized<float>(r0, r1);
        */
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
impl BitOr<&Vectorized<f32>> for Vectorized<f32> {

    type Output = Vectorized<f32>;
    
    fn bitor(self, other: &Vectorized<f32>) -> Self::Output {
        todo!();
        /*
            float32x4_t r0 = vreinterpretq_f32_u32(vorrq_u32(
          vreinterpretq_u32_f32(a.get_low()),
          vreinterpretq_u32_f32(b.get_low())));
      float32x4_t r1 = vreinterpretq_f32_u32(vorrq_u32(
          vreinterpretq_u32_f32(a.get_high()),
          vreinterpretq_u32_f32(b.get_high())));
      return Vectorized<float>(r0, r1);
        */
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
impl BitXor<&Vectorized<f32>> for Vectorized<f32> {

    type Output = Vectorized<f32>;
    
    fn bitxor(self, other: &Vectorized<f32>) -> Self::Output {
        todo!();
        /*
            float32x4_t r0 = vreinterpretq_f32_u32(veorq_u32(
          vreinterpretq_u32_f32(a.get_low()),
          vreinterpretq_u32_f32(b.get_low())));
      float32x4_t r1 = vreinterpretq_f32_u32(veorq_u32(
          vreinterpretq_u32_f32(a.get_high()),
          vreinterpretq_u32_f32(b.get_high())));
      return Vectorized<float>(r0, r1);
        */
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
impl VectorizedFloat {
    
    pub fn eq(&self, other: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return (*this == other) & Vectorized<float>(1.0f);
        */
    }
    
    pub fn ne(&self, other: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return (*this != other) & Vectorized<float>(1.0f);
        */
    }
    
    pub fn gt(&self, other: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return (*this > other) & Vectorized<float>(1.0f);
        */
    }
    
    pub fn ge(&self, other: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return (*this >= other) & Vectorized<float>(1.0f);
        */
    }
    
    pub fn lt(&self, other: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return (*this < other) & Vectorized<float>(1.0f);
        */
    }
    
    pub fn le(&self, other: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return (*this <= other) & Vectorized<float>(1.0f);
        */
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[inline] pub fn convert(
        src: *const f32,
        dst: *mut i32,
        n:   i64)  {
    
    todo!();
        /*
            i64 i;
    #pragma unroll
      for (i = 0; i <= (n - Vectorized<float>::size()); i += Vectorized<float>::size()) {
        vst1q_s32(dst + i, vcvtq_s32_f32(vld1q_f32(src + i)));
        vst1q_s32(dst + i + 4, vcvtq_s32_f32(vld1q_f32(src + i + 4)));
      }
    #pragma unroll
      for (; i < n; i++) {
        dst[i] = static_cast<i32>(src[i]);
      }
        */
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[inline] pub fn convert(
        src: *const i32,
        dst: *mut f32,
        n:   i64)  {
    
    todo!();
        /*
            i64 i;
    #pragma unroll
      for (i = 0; i <= (n - Vectorized<float>::size()); i += Vectorized<float>::size()) {
        vst1q_f32(dst + i, vcvtq_f32_s32(vld1q_s32(src + i)));
        vst1q_f32(dst + i + 4, vcvtq_f32_s32(vld1q_s32(src + i + 4)));
      }
    #pragma unroll
      for (; i < n; i++) {
        dst[i] = static_cast<float>(src[i]);
      }
        */
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[inline] pub fn fmadd(
        a: &Vectorized<f32>,
        b: &Vectorized<f32>,
        c: &Vectorized<f32>) -> Vectorized<f32> {
    
    todo!();
        /*
            float32x4_t r0 = vfmaq_f32(c.get_low(), a.get_low(), b.get_low());
      float32x4_t r1 = vfmaq_f32(c.get_high(), a.get_high(), b.get_high());
      return Vectorized<float>(r0, r1);
        */
}
