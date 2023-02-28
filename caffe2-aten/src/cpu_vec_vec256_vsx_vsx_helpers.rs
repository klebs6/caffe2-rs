crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/vsx/vsx_helpers.h]

lazy_static!{
    /*
    using vbool8   =  __attribute__((altivec(vector__))) __attribute__((altivec(bool__))) char;
    using vbool16  =  __attribute__((altivec(vector__))) __attribute__((altivec(bool__))) short;
    using vbool32  =  __attribute__((altivec(vector__))) __attribute__((altivec(bool__))) int;
    using vbool64  =  __attribute__((altivec(vector__))) __attribute__((altivec(bool__))) long long;
    using vint8    =  __attribute__((altivec(vector__)))  signed char;
    using vint16   =  __attribute__((altivec(vector__)))  signed short;
    using vint32   =  __attribute__((altivec(vector__)))  signed int;
    using vint64   =  __attribute__((altivec(vector__)))  signed long long;
    using vuint8   =  __attribute__((altivec(vector__)))  unsigned char;
    using vuint16  =  __attribute__((altivec(vector__)))  unsigned short;
    using vuint32  =  __attribute__((altivec(vector__)))  unsigned  int;
    using vuint64  =  __attribute__((altivec(vector__)))  unsigned long long;
    using vfloat32 =  __attribute__((altivec(vector__)))  float;
    using vfloat64 =  __attribute__((altivec(vector__)))  double;
    */
}

#[inline(always)] 
#[cfg(not(vec_float))]
pub fn vec_float(vec_in: &vint32) -> Vfloat32 {
    
    todo!();
        /*
            vfloat32 vec_out;
      __asm__("xvcvsxwsp %x0,%x1" : "=wf"(vec_out) : "wa"(vec_in));
      return vec_out;
        */
}

#[inline(always)] 
#[cfg(not(vec_signed))]
pub fn vec_signed32(vec_in: &Vfloat32) -> vint32 {
    
    todo!();
        /*
            vint32 vec_out;
      __asm__("xvcvspsxws %x0,%x1" : "=wa"(vec_out) : "wf"(vec_in));
      return vec_out;
        */
}

#[inline(always)] 
#[cfg(not(vec_signed))]
pub fn vec_signed64(vec_in: &Vfloat64) -> Vint64 {
    
    todo!();
        /*
            vint64 vec_out;
      __asm__("xvcvdpsxds %x0,%x1" : "=wa"(vec_out) : "wd"(vec_in));
      return vec_out;
        */
}

#[cfg(not(vec_neg))]
#[inline(always)] 
pub fn vec_neg_f32(vec_in: &Vfloat32) -> Vfloat32 {
    
    todo!();
        /*
            vfloat32 vec_out;
      __asm__("xvnegsp %x0,%x1" : "=wf"(vec_out) : "wf"(vec_in));
      return vec_out;
        */
}

#[inline(always)] 
#[cfg(not(vec_neg))]
pub fn vec_neg_f64(vec_in: &Vfloat64) -> Vfloat64 {
    
    todo!();
        /*
            vfloat64 vec_out;
      __asm__("xvnegdp %x0,%x1" : "=wd"(vec_out) : "wd"(vec_in));
      return vec_out;
        */
}

#[inline(always)] 
#[cfg(not(vec_neg))]
pub fn vec_neg_i16(vec_in: &Vint16) -> Vint16 {
    
    todo!();
        /*
            vint16 vint0 = {0, 0, 0, 0 ,0, 0, 0, 0};
      return vec_vsubuhm(vint0, vec_in);
        */
}

#[inline(always)] 
#[cfg(not(vec_neg))]
pub fn vec_neg_i32(vec_in: &vint32) -> vint32 {
    
    todo!();
        /*
            vint32 vint0 = {0, 0, 0, 0};
      return vec_vsubuwm(vint0, vec_in);
        */
}

#[inline(always)] 
#[cfg(not(vec_neg))]
pub fn vec_neg_i64(vec_in: &Vint64) -> Vint64 {
    
    todo!();
        /*
            vint64 vint0 = {0, 0};
      return vec_vsubudm(vint0, vec_in);
        */
}

#[cfg(not(vec_sldw))]
#[inline(always)] 
pub fn vec_sldw_aux<const C: u32>(
    vec_in0: &Vfloat32,
    vec_in1: &Vfloat32) -> Vfloat32 {

    todo!();
        /*
            vfloat32 vec_out;
      __asm("xxsldwi %x0, %x1, %x2, %3 "
            : "=wa"(vec_out)
            : "wa"(vec_in0), "wa"(vec_in1), "I"(C));
      return vec_out;
        */
}

#[cfg(not(vec_sldw))]
#[macro_export] macro_rules! vec_sldw {
    ($a:ident, $b:ident, $c:ident) => {
        /*
                vec_sldw_aux<c>(a, b)
        */
    }
}

#[macro_export] macro_rules! vec_not {
    ($a:ident) => {
        /*
                vec_nor(a, a)
        */
    }
}

/**
  | Vectorized min/max which return a if
  | any operand is nan
  |
  */
#[inline(always)] 
pub fn vec_min_nan<T>(a: &T, b: &T) -> T {

    todo!();
        /*
            return vec_min(a, b);
        */
}

#[inline(always)] 
pub fn vec_max_nan<T>(a: &T, b: &T) -> T {

    todo!();
        /*
            return vec_max(a, b);
        */
}

/// Specializations for float/double taken from
/// Eigen
///
#[inline(always)] 
pub fn vec_min_nan_vfloat32(
        a: &Vfloat32,
        b: &Vfloat32) -> Vfloat32 {
    
    todo!();
        /*
            // NOTE: about 10% slower than vec_min, but consistent with min and SSE regarding NaN
      vfloat32 ret;
      __asm__ ("xvcmpgesp %x0,%x1,%x2\n\txxsel %x0,%x1,%x2,%x0" : "=&wa" (ret) : "wa" (a), "wa" (b));
      return ret;
        */
}


// Specializations for float/double taken from Eigen
#[inline(always)] 
pub fn vec_max_nan_vfloat32(
        a: &Vfloat32,
        b: &Vfloat32) -> Vfloat32 {
    
    todo!();
        /*
            // NOTE: about 10% slower than vec_max, but consistent with min and SSE regarding NaN
      vfloat32 ret;
       __asm__ ("xvcmpgtsp %x0,%x2,%x1\n\txxsel %x0,%x1,%x2,%x0" : "=&wa" (ret) : "wa" (a), "wa" (b));
      return ret;
        */
}

#[inline(always)] 
pub fn vec_min_nan_vfloat64(
        a: &Vfloat64,
        b: &Vfloat64) -> Vfloat64 {
    
    todo!();
        /*
            // NOTE: about 10% slower than vec_min, but consistent with min and SSE regarding NaN
      vfloat64 ret;
      __asm__ ("xvcmpgedp %x0,%x1,%x2\n\txxsel %x0,%x1,%x2,%x0" : "=&wa" (ret) : "wa" (a), "wa" (b));
      return ret;
        */
}

#[inline(always)] 
pub fn vec_max_nan_vfloat64(
    a: &Vfloat64,
    b: &Vfloat64) -> Vfloat64 {
    
    todo!();
        /*
            // NOTE: about 10% slower than vec_max, but consistent with max and SSE regarding NaN
      vfloat64 ret;
      __asm__ ("xvcmpgtdp %x0,%x2,%x1\n\txxsel %x0,%x1,%x2,%x0" : "=&wa" (ret) : "wa" (a), "wa" (b));
      return ret;
        */
}



// Vectorizes min/max function which returns nan if any side is nan
#[macro_export] macro_rules! c10_vsx_vec_nan_propag {
    ($name:ident, $type:ident, $btype:ident, $func:ident) => {
        /*
        
          #[inline(always)] type name(const type& a, const type& b) { 
            type tmp = func(a, b);                                    
            btype nan_a = vec_cmpne(a, a);                            
            btype nan_b = vec_cmpne(b, b);                            
            tmp = vec_sel(tmp, a, nan_a);                             
            return vec_sel(tmp, b, nan_b);                            
          }
        */
    }
}

c10_vsx_vec_nan_propag!{vec_min_nan2, vfloat32, vbool32, vec_min}
c10_vsx_vec_nan_propag!{vec_max_nan2, vfloat32, vbool32, vec_max}
c10_vsx_vec_nan_propag!{vec_min_nan2, vfloat64, vbool64, vec_min}
c10_vsx_vec_nan_propag!{vec_max_nan2, vfloat64, vbool64, vec_max}

#[macro_export] macro_rules! define_member_unary_op {
    ($op:ident, $op_type:ident, $func:ident) => {
        /*
        
          Vectorized<op_type> #[inline(always)] op() const {      
            return Vectorized<op_type>{func(_vec0), func(_vec1)}; 
          }
        */
    }
}

#[macro_export] macro_rules! define_member_op {
    ($op:ident, $op_type:ident, $func:ident) => {
        /*
        
          Vectorized<op_type> #[inline(always)] op(const Vectorized<op_type>& other) const { 
            return Vectorized<op_type>{                                                  
                func(_vec0, other._vec0), func(_vec1, other._vec1)};                 
          }
        */
    }
}

#[macro_export] macro_rules! define_member_bitwise_op {
    ($op:ident, $op_type:ident, $func:ident) => {
        /*
        
          Vectorized<op_type> #[inline(always)] op(const Vectorized<op_type>& other) const { 
            return Vectorized<op_type>{                                                  
                func(_vecb0, other._vecb0), func(_vecb1, other._vecb1)};             
          }
        */
    }
}

#[macro_export] macro_rules! define_member_ternary_op {
    ($op:ident, $op_type:ident, $func:ident) => {
        /*
        
          Vectorized<op_type> #[inline(always)] op(                                
              const Vectorized<op_type>& b, const Vectorized<op_type>& c) const {      
            return Vectorized<op_type>{                                            
                func(_vec0, b._vec0, c._vec0), func(_vec1, b._vec1, c._vec1)}; 
          }
        */
    }
}

#[macro_export] macro_rules! define_member_emulate_binary_op {
    ($op:ident, $op_type:ident, $binary_op:ident) => {
        /*
        
          Vectorized<op_type> #[inline(always)] op(const Vectorized<op_type>& b) const { 
            Vectorized<op_type>::vec_internal_type ret_0;                         
            Vectorized<op_type>::vec_internal_type ret_1;                         
            for (int i = 0; i < Vectorized<op_type>::size() / 2; i++) {           
              ret_0[i] = _vec0[i] binary_op b._vec0[i];                       
              ret_1[i] = _vec1[i] binary_op b._vec1[i];                       
            }                                                                 
            return Vectorized<op_type>{ret_0, ret_1};                             
          }
        */
    }
}

#[macro_export] macro_rules! define_member_op_and_one {
    ($op:ident, $op_type:ident, $func:ident) => {
        /*
        
          Vectorized<op_type> #[inline(always)] op(const Vectorized<op_type>& other) const { 
            using vvtype = Vectorized<op_type>::vec_internal_type;                       
            const vvtype v_one = vec_splats(static_cast<op_type>(1.0));              
            vvtype ret0 = (vvtype)func(_vec0, other._vec0);                          
            vvtype ret1 = (vvtype)func(_vec1, other._vec1);                          
            return Vectorized<op_type>{vec_and(ret0, v_one), vec_and(ret1, v_one)};      
          }
        */
    }
}

#[macro_export] macro_rules! define_clamp_funcs {
    ($operand_type:ident) => {
        /*
        
          template <>                                                                   
          Vectorized<operand_type> #[inline(always)] clamp(                             
              const Vectorized<operand_type>& a,                                        
              const Vectorized<operand_type>& min,                                      
              const Vectorized<operand_type>& max) {                                    
            return Vectorized<operand_type>{                                            
                vec_min_nan(vec_max_nan(a.vec0(), min.vec0()), max.vec0()),             
                vec_min_nan(vec_max_nan(a.vec1(), min.vec1()), max.vec1())};            
          }                                                                             
          template <>                                                                   
          Vectorized<operand_type> #[inline(always)] clamp_min(                         
              const Vectorized<operand_type>& a, const Vectorized<operand_type>& min) { 
            return Vectorized<operand_type>{                                            
                vec_max_nan(a.vec0(), min.vec0()),                                      
                vec_max_nan(a.vec1(), min.vec1())};                                     
          }                                                                             
          template <>                                                                   
          Vectorized<operand_type> #[inline(always)] clamp_max(                         
              const Vectorized<operand_type>& a, const Vectorized<operand_type>& max) { 
            return Vectorized<operand_type>{                                            
                vec_min_nan(a.vec0(), max.vec0()),                                      
                vec_min_nan(a.vec1(), max.vec1())};                                     
          }
        */
    }
}

#[macro_export] macro_rules! define_reinterpret_cast_funcs {
    () => {
        /*
                (                             
            first_type, cast_type, cast_inner_vector_type)                 
          template <>                                                      
          #[inline(always)] Vectorized<cast_type> cast<cast_type, first_type>( 
              const Vectorized<first_type>& src) {                                 
            return Vectorized<cast_type>{(cast_inner_vector_type)src.vec0(),       
                                     (cast_inner_vector_type)src.vec1()};      
          }
        */
    }
}

#[macro_export] macro_rules! define_reinterpret_cast_to_all_funcs {
    ($first_type:ident) => {
        /*
        
          DEFINE_REINTERPRET_CAST_FUNCS(first_type, double, vfloat64)    
          DEFINE_REINTERPRET_CAST_FUNCS(first_type, float, vfloat32)     
          DEFINE_REINTERPRET_CAST_FUNCS(first_type, i64, vint64) 
          DEFINE_REINTERPRET_CAST_FUNCS(first_type, i32, vint32)   
          DEFINE_REINTERPRET_CAST_FUNCS(first_type, i16, vint16)
        */
    }
}

/// it can be used to emulate blend faster
pub fn blend_choice(
        mask:  u32,
        half1: u32,
        half2: u32) -> i32 {

    let half1: u32 = half1.unwrap_or(0xF);
    let half2: u32 = half2.unwrap_or(0xF0);

    todo!();
        /*
            u32 none = 0;
      u32 both = half1 | half2;
      // clamp it between 0 and both
      mask = mask & both;
      // return  (a._vec0, a._vec1)
      if (mask == none) return 0;
      // return (b._vec0,b._vec1)
      else if (mask == both)
        return 1;
      // return  (b._vec0,a._vec1)
      else if (mask == half1)
        return 2;
      // return  (a._vec0,b._vec1)
      else if (mask == half2)
        return 3;
      // return  (*_vec0,a._vec1)
      else if (mask > 0 && mask < half1)
        return 4;
      // return  (*_vec0,b._vec1)
      else if ((mask & half2) == half2)
        return 5;
      // return (a._vec0,*_vec1)
      else if ((mask & half1) == 0 && mask > half1)
        return 6;
      // return (b._vec0,*_vec1)
      else if ((mask & half1) == half1 && mask > half1)
        return 7;
      // return (*_vec0,*_vec1)
      return 8;
        */
}

/// it can be used to emulate blend faster
pub fn blend_choice_dbl(mask: u32) -> i32 {
    
    todo!();
        /*
            // clamp it 0 and 0xF
      return blendChoice(mask, 0x3, 0xC);
        */
}

pub fn vsx_mask1(mask: u32) -> Vbool32 {
    
    todo!();
        /*
            u32 g0 = (mask & 1) * 0xffffffff;
      u32 g1 = ((mask & 2) >> 1) * 0xffffffff;
      u32 g2 = ((mask & 4) >> 2) * 0xffffffff;
      u32 g3 = ((mask & 8) >> 3) * 0xffffffff;
      return (vbool32){g0, g1, g2, g3};
        */
}

pub fn vsx_mask2(mask: u32) -> Vbool32 {
    
    todo!();
        /*
            u32 mask2 = (mask & 0xFF) >> 4;
      return VsxMask1(mask2);
        */
}

pub fn vsx_dbl_mask1(mask: u32) -> Vbool64 {
    
    todo!();
        /*
            u64 g0 = (mask & 1) * 0xffffffffffffffff;
      u64 g1 = ((mask & 2) >> 1) * 0xffffffffffffffff;
      return (vbool64){g0, g1};
        */
}


pub fn vsx_dbl_mask2(mask: u32) -> Vbool64 {
    
    todo!();
        /*
            u32 mask2 = (mask & 0xF) >> 2;
      return VsxDblMask1(mask2);
        */
}


pub fn mask_for_complex(mask: u32) -> i32 {
    
    todo!();
        /*
            mask = mask & 0xF;
      int complex_mask = 0;
      if (mask & 1) complex_mask |= 3;
      if (mask & 2) complex_mask |= (3 << 2);
      if (mask & 4) complex_mask |= (3 << 4);
      if (mask & 8) complex_mask |= (3 << 6);
      return complex_mask;
        */
}


pub fn mask_for_complex_dbl(mask: u32) -> i32 {
    
    todo!();
        /*
            mask = mask & 0x3;
      int complex_mask = 0;
      if (mask & 1) complex_mask |= 3;
      if (mask & 2) complex_mask |= (3 << 2);
      return complex_mask;
        */
}


pub fn blend_choice_complex(mask: u32) -> i32 {
    
    todo!();
        /*
            return blendChoice(maskForComplex(mask));
        */
}


pub fn blend_choice_complex_dbl(mask: u32) -> i32 {
    
    todo!();
        /*
            return blendChoiceDbl(maskForComplexDbl(mask));
        */
}


pub fn vsx_complex_mask1(mask: u32) -> Vbool32 {
    
    todo!();
        /*
            return VsxMask1(maskForComplex(mask));
        */
}


pub fn vsx_complex_mask2(mask: u32) -> Vbool32 {
    
    todo!();
        /*
            u32 mask2 = (mask & 0xF) >> 2;
      return VsxMask1(maskForComplex(mask2));
        */
}


pub fn vsx_complex_dbl_mask1(mask: u32) -> Vbool64 {
    
    todo!();
        /*
            return VsxDblMask1(mask);
        */
}


pub fn vsx_complex_dbl_mask2(mask: u32) -> Vbool64 {
    
    todo!();
        /*
            u32 mask2 = (mask & 0xF) >> 2;
      return VsxDblMask1(mask2);
        */
}

pub const OFFSET0:  i32 = 0;
pub const OFFSET16: i32 = 16;

lazy_static!{
    /*
    pub const MASK_ZERO_BITS:     Vuint8   = vuint8{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 96,  64,  32,  0};
    pub const SWAP_MASK:          Vuint8   = vuint8{4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11};
    pub const V0X7F:              vint32   = vec_splats(0x7f);
    pub const VI_0:               vint32   = vec_splats((int)(0));
    pub const VI_1:               vint32   = vec_splats((int)1);
    pub const VI_2:               vint32   = vec_splats((int)2);
    pub const VI_4:               vint32   = vec_splats((int)4);
    pub const VI_INV1:            vint32   = vec_splats((int)~1);
    pub const VU_29:              Vuint32  = vec_splats(29u);
    pub const VU_23:              Vuint32  = vec_splats(23u);
    pub const INV_MANT_MASK:      Vbool32  = (vbool32)vec_splats((unsigned int)~0xff800000);
    pub const SIGN_MASK:          Vbool32  = (vbool32)vec_splats((int)0x80000000);
    pub const REAL_MASK:          Vbool32  = vbool32{0xFFFFFFFF, 0x0, 0xFFFFFFFF, 0x0};
    pub const IMAG_MASK:          Vbool32  = vbool32{0x0, 0xFFFFFFFF, 0x0, 0xFFFFFFFF};
    pub const ISIGN_MASK:         Vbool32  = vbool32{0x0, 0x80000000, 0x0, 0x80000000};
    pub const RSIGN_MASK:         Vbool32  = vbool32{0x80000000, 0x0, 0x80000000, 0x0};
    pub const VD_IMAG_MASK:       Vbool64  = vbool64{0x0, 0xFFFFFFFFFFFFFFFF};
    pub const VD_REAL_MASK:       Vbool64  = vbool64{0xFFFFFFFFFFFFFFFF, 0x0};
    pub const VD_ISIGN_MASK:      Vbool64  = vbool64{0x0, 0x8000000000000000};
    pub const VD_RSIGN_MASK:      Vbool64  = vbool64{0x8000000000000000, 0x0};
    pub const ZERO:               Vfloat32 = vec_splats(0.f);
    pub const HALF:               Vfloat32 = vec_splats(0.5f);
    pub const ONE:                Vfloat32 = vec_splats(1.f);
    pub const TWO:                Vfloat32 = vec_splats(2.0f);
    pub const 4DIV_PI:            Vfloat32 = vec_splats(1.27323954473516f);
    pub const V_INF:              Vfloat32 = (vfloat32)vec_splats(0x7f800000u);
    pub const V_MINUS_INF:        Vfloat32 = vfloat32{ 0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u };
    pub const V_NAN:              Vfloat32 = (vfloat32)vec_splats(0x7fffffff);
    pub const LOG_10E_INV:        Vfloat32 = vec_splats(0.43429448190325176f);
    pub const LOG2E_INV:          Vfloat32 = vec_splats(1.4426950408889634f);
    pub const LOG2EB_INV:         Vfloat32 = vec_splats(1.442695036924675f);
    pub const CEPHES_SQRTHF:      Vfloat32 = vec_splats(0.707106781186547524f);
    pub const COSCOF_P0:          Vfloat32 = vec_splats(2.443315711809948E-005f);
    pub const COSCOF_P1:          Vfloat32 = vec_splats(-1.388731625493765E-003f);
    pub const COSCOF_P2:          Vfloat32 = vec_splats(4.166664568298827E-002f);
    pub const EXP_HI:             Vfloat32 = vec_splats(104.f);
    pub const EXP_LO:             Vfloat32 = vec_splats(-104.f);
    pub const EXP_P0:             Vfloat32 = vec_splats(0.000198527617612853646278381f);
    pub const EXP_P1:             Vfloat32 = vec_splats((0.00139304355252534151077271f));
    pub const EXP_P2:             Vfloat32 = vec_splats(0.00833336077630519866943359f);
    pub const EXP_P3:             Vfloat32 = vec_splats(0.0416664853692054748535156f);
    pub const EXP_P4:             Vfloat32 = vec_splats(0.166666671633720397949219f);
    pub const EXP_P5:             Vfloat32 = vec_splats(0.5f);
    pub const LOG_P0:             Vfloat32 = vec_splats(7.0376836292E-2f);
    pub const LOG_P1:             Vfloat32 = vec_splats(-1.1514610310E-1f);
    pub const LOG_P2:             Vfloat32 = vec_splats(1.1676998740E-1f);
    pub const LOG_P3:             Vfloat32 = vec_splats(-1.2420140846E-1f);
    pub const LOG_P4:             Vfloat32 = vec_splats(+1.4249322787E-1f);
    pub const LOG_P5:             Vfloat32 = vec_splats(-1.6668057665E-1f);
    pub const LOG_P6:             Vfloat32 = vec_splats(+2.0000714765E-1f);
    pub const LOG_P7:             Vfloat32 = vec_splats(-2.4999993993E-1f);
    pub const LOG_P8:             Vfloat32 = vec_splats(+3.3333331174E-1f);
    pub const LOG_Q1:             Vfloat32 = vec_splats(-2.12194440e-4f);
    pub const LOG_Q2:             Vfloat32 = vec_splats(0.693359375f);
    pub const MAX_LOGF:           Vfloat32 = vec_splats(88.02969187150841f);
    pub const MAX_NUMF:           Vfloat32 = vec_splats(1.7014117331926442990585209174225846272e38f);
    pub const MIN_INF:            Vfloat32 = (vfloat32)vec_splats(0xff800000u);
    pub const MIN_NORM_POS:       Vfloat32 = (vfloat32)vec_splats(0x0800000u);
    pub const MINUS_CEPHES_DP1:   Vfloat32 = vec_splats(-0.78515625f);
    pub const MINUS_CEPHES_DP2:   Vfloat32 = vec_splats(-2.4187564849853515625e-4f);
    pub const MINUS_CEPHES_DP3:   Vfloat32 = vec_splats(-3.77489497744594108e-8f);
    pub const NEGLN2F_HI:         Vfloat32 = vec_splats(-0.693145751953125f);
    pub const NEGLN2F_LO:         Vfloat32 = vec_splats(-1.428606765330187045e-06f);
    pub const P0:                 Vfloat32 = vec_splats(2.03721912945E-4f);
    pub const P1:                 Vfloat32 = vec_splats(8.33028376239E-3f);
    pub const P2:                 Vfloat32 = vec_splats(1.66667160211E-1f);
    pub const SINCOF_P0:          Vfloat32 = vec_splats(-1.9515295891E-4f);
    pub const SINCOF_P1:          Vfloat32 = vec_splats(8.3321608736E-3f);
    pub const SINCOF_P2:          Vfloat32 = vec_splats(-1.6666654611E-1f);
    pub const TANH_0P625:         Vfloat32 = vec_splats(0.625f);
    pub const TANH_HALF_MAX:      Vfloat32 = vec_splats(44.014845935754205f);
    pub const TANH_P0:            Vfloat32 = vec_splats(-5.70498872745E-3f);
    pub const TANH_P1:            Vfloat32 = vec_splats(2.06390887954E-2f);
    pub const TANH_P2:            Vfloat32 = vec_splats(-5.37397155531E-2f);
    pub const TANH_P3:            Vfloat32 = vec_splats(1.33314422036E-1f);
    pub const TANH_P4:            Vfloat32 = vec_splats(-3.33332819422E-1f);
    pub const VCHECK:             Vfloat32 = vec_splats((float)(1LL << 24));
    pub const IMAG_ONE:           Vfloat32 = vfloat32{0.f, 1.f, 0.f, 1.f};
    pub const IMAG_HALF:          Vfloat32 = vfloat32{0.f, 0.5f, 0.f, 0.5f};
    pub const SQRT2_2:            Vfloat32 = vfloat32{0.70710676908493042f, 0.70710676908493042, 0.70710676908493042, 0.70710676908493042};
    pub const PI_2:               Vfloat32 = vfloat32{M_PI / 2, 0.0, M_PI / 2, 0.0};
    pub const VF_89:              Vfloat32 = vfloat32{89.f, 89.f, 89.f, 89.f};
    pub const VD_ONE:             Vfloat64 = vec_splats(1.0);
    pub const VD_ZERO:            Vfloat64 = vec_splats(0.0);
    pub const VD_LOG_10E_INV:     Vfloat64 = vec_splats(0.43429448190325176);
    pub const VD_LOG2E_INV:       Vfloat64 = vec_splats(1.4426950408889634);
    pub const VD_IMAG_ONE:        Vfloat64 = vfloat64{0.0, 1.0};
    pub const VD_IMAG_HALF:       Vfloat64 = vfloat64{0.0, 0.5};
    pub const VD_SQRT2_2:         Vfloat64 = vfloat64{0.70710678118654757, 0.70710678118654757};
    pub const VD_PI_2:            Vfloat64 = vfloat64{M_PI / 2.0, 0.0};
    */
}

