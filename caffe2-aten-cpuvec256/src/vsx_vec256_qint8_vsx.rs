/*!
  | This file defines Vectorized<> for the
  | quantized types.
  |
  | Currently, we simply use these classes as
  | efficient converters between the quantized
  | types and Vectorized<float>, usually in
  | bandwidth-bound cases where doing the
  | arithmetic in full-precision is acceptable
  | (e.g. elementwise operators).
  |
  | Conversions are as follows:
  |  Vectorized<qint8> -> 4x Vectorized<float>
  |
  | The size of the returned float vector is
  | specified by the special constexpr function
  | float_num_vecs. The type of the value returned
  | from dequantize (and expected as an argument to
  | quantize) is specified by
  | float_vec_return_type.
  |
  | When writing kernels with these vectors, it is
  | expected that floating- point operations will
  | be carried out in a loop over
  | Vectorized<T>::float_num_vecs iterations.
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/vsx/vec256_qint8_vsx.h]

struct VectorizedQint8Alt0 {
    vec0: Vint8,
    vec1: Vint8,
}

struct VectorizedQint8Alt1 {
    vecb0: Vbool8,
    vecb1: Vbool8,
}

pub union VectorizedQint8 {
    u0: VectorizedQint8Alt0,
    u1: VectorizedQint8Alt1,
}

pub mod vectorized_qint8 {

    use super::*;

    pub type SizeType            = i32;
    pub type FloatVecReturnType  = Array<Vectorized<f32>,4>;
    pub type IntVecReturnType    = Array<Vectorized<qint32>,4>;
    pub type ValueType           = qint8::underlying;
    pub type VecInternalType     = Vint8;
    pub type VecInternalMaskType = Vbool8;

    lazy_static!{
        /*
        DEFINE_MEMBER_OP(operator==, qint8, vec_cmpeq)
          DEFINE_MEMBER_OP(operator!=, qint8, vec_cmpne)
          DEFINE_MEMBER_OP(operator<, qint8, vec_cmplt)
          DEFINE_MEMBER_OP(operator<=, qint8, vec_cmple)
          DEFINE_MEMBER_OP(operator>, qint8, vec_cmpgt)
          DEFINE_MEMBER_OP(operator>=, qint8, vec_cmpge)
          DEFINE_MEMBER_OP(operator+, qint8, vec_add)
          DEFINE_MEMBER_OP(operator-, qint8, vec_sub)
          DEFINE_MEMBER_OP(operator*, qint8, vec_mul)
          DEFINE_MEMBER_EMULATE_BINARY_OP(operator/, qint8, /)
          DEFINE_MEMBER_OP(maximum, qint8, vec_max)
          DEFINE_MEMBER_OP(minimum, qint8, vec_min)
          DEFINE_MEMBER_OP(operator&, qint8, vec_and)
          DEFINE_MEMBER_OP(operator|, qint8, vec_or)
          DEFINE_MEMBER_OP(operator^, qint8, vec_xor)
        */
    }
}

impl VectorizedQint8 {

    pub fn size() -> SizeType {
        
        todo!();
        /*
            return 32;
        */
    }
    
    pub fn float_num_vecs() -> usize {
        
        todo!();
        /*
            return 4;
        */
    }
    
    pub fn int_num_vecs() -> i32 {
        
        todo!();
        /*
            return 4;
        */
    }

    /// Broadcast constructor
    #[inline(always)] 
    pub fn new(val: &qint8) -> Self {
    
        todo!();
        /*


            : _vec0{vec_splats(val.val_)}, _vec1{vec_splats(val.val_)}
        */
    }

    #[inline(always)] 
    pub fn new(other: &Vectorized<qint8>) -> Self {
    
        todo!();
        /*


            : _vec0{other._vec0}, _vec1(other._vec1)
        */
    }

  #[inline(always)] 
    pub fn new(v: Vint8) -> Self {
    
        todo!();
        /*


            : _vec0{v}, _vec1{v}
        */
    }

  #[inline(always)] 
    pub fn new(vmask: Vbool8) -> Self {
    
        todo!();
        /*


            : _vecb0{vmask}, _vecb1{vmask}
        */
    }

  #[inline(always)] 
    pub fn new(
        v1: Vint8,
        v2: Vint8) -> Self {
    
        todo!();
        /*


            : _vec0{v1}, _vec1{v2}
        */
    }

  #[inline(always)] 
    pub fn new(
        v1: Vbool8,
        v2: Vbool8) -> Self {
    
        todo!();
        /*


            : _vecb0{v1}, _vecb1{v2}
        */
    }

  #[inline(always)] 
    pub fn vec0(&self) -> &VecInternalType {
        
        todo!();
        /*
            return _vec0;
        */
    }

  #[inline(always)] 
    pub fn vec1(&self) -> &VecInternalType {
        
        todo!();
        /*
            return _vec1;
        */
    }

#[inline(always)]
    pub fn loadu(
        ptr:   *const c_void,
        count: i32) -> Vectorized<qint8> {
        let count: i32 = count.unwrap_or(0);

        todo!();
        /*
            if (count == size()) {
          return {
              vec_vsx_ld(offset0, reinterpret_cast<const vint8*>(ptr)),
              vec_vsx_ld(offset16, reinterpret_cast<const vint8*>(ptr))};
        }
        __at_align32__ value_type tmp_values[size()];
        memcpy(tmp_values, ptr, min(count, size()) * sizeof(value_type));
        return {vec_vsx_ld(offset0, tmp_values), vec_vsx_ld(offset16, tmp_values)};
        */
    }

    #[inline(always)]
    pub fn store(&self, 
        ptr:   *mut c_void,
        count: i32)  {
        let count: i32 = count.unwrap_or(0);

        todo!();
        /*
            if (count == size()) {
          vec_vsx_st(_vec0, offset0, reinterpret_cast<value_type*>(ptr));
          vec_vsx_st(_vec1, offset16, reinterpret_cast<value_type*>(ptr));
        } else if (count > 0) {
          __at_align32__ value_type tmp_values[size()];
          vec_vsx_st(_vec0, offset0, tmp_values);
          vec_vsx_st(_vec1, offset16, tmp_values);
          memcpy(
              ptr, tmp_values, min(count, size()) * sizeof(value_type));
        }
        */
    }

     #[inline(always)]
    pub fn dequantize(&self, 
        scale:           Vectorized<f32>,
        zero_point:      Vectorized<f32>,
        scale_zp_premul: Vectorized<f32>) -> FloatVecReturnType {
        
        todo!();
        /*
            vint16 vecshi0 = vec_unpackh(_vec0);
        vint16 vecshi1 = vec_unpackl(_vec0);

        vint16 vecshi2 = vec_unpackh(_vec1);
        vint16 vecshi3 = vec_unpackl(_vec1);

        vint32 veci0 = vec_unpackh(vecshi0);
        vint32 veci1 = vec_unpackl(vecshi0);

        vint32 veci2 = vec_unpackh(vecshi1);
        vint32 veci3 = vec_unpackl(vecshi1);

        vint32 veci4 = vec_unpackh(vecshi2);
        vint32 veci5 = vec_unpackl(vecshi2);

        vint32 veci6 = vec_unpackh(vecshi3);
        vint32 veci7 = vec_unpackl(vecshi3);

        vfloat32 vecf0_0 = vec_float(veci0);
        vfloat32 vecf1_0 = vec_float(veci1);

        vfloat32 vecf0_1 = vec_float(veci2);
        vfloat32 vecf1_1 = vec_float(veci3);

        vfloat32 vecf0_2 = vec_float(veci4);
        vfloat32 vecf1_2 = vec_float(veci5);

        vfloat32 vecf0_3 = vec_float(veci6);
        vfloat32 vecf1_3 = vec_float(veci7);
        vfloat32 scale_vec0 = scale.vec0();
        vfloat32 scale_vec1 = scale.vec1();
        vfloat32 scale_zp_premul0 = scale_zp_premul.vec0();
        vfloat32 scale_zp_premul1 = scale_zp_premul.vec1();
        return {
            Vectorized<float>{
                vec_madd(scale_vec0, vecf0_0, scale_zp_premul0),
                vec_madd(scale_vec1, vecf1_0, scale_zp_premul1)},
            Vectorized<float>{
                vec_madd(scale_vec0, vecf0_1, scale_zp_premul0),
                vec_madd(scale_vec1, vecf1_1, scale_zp_premul1)},
            Vectorized<float>{
                vec_madd(scale_vec0, vecf0_2, scale_zp_premul0),
                vec_madd(scale_vec1, vecf1_2, scale_zp_premul1)},
            Vectorized<float>{
                vec_madd(scale_vec0, vecf0_3, scale_zp_premul0),
                vec_madd(scale_vec1, vecf1_3, scale_zp_premul1)}};
        */
    }
    
    pub fn quantize(
        rhs:           &FloatVecReturnType,
        scale:         f32,
        zero_point:    i32,
        inverse_scale: f32) -> Vectorized<qint8> {
        
        todo!();
        /*
            // constexpr i32 min_val = value_type::min;
        // constexpr i32 max_val = value_type::max;

        vfloat32 inverse_scale_v = vec_splats(inverse_scale);
        vfloat32 vec_zero_point = vec_splats((float)zero_point);
        // vint32 vmin = vec_splats(min_val);
        // vint32 vmax = vec_splats(max_val);

        Vectorized<float> vf0 = rhs[0];
        Vectorized<float> vf1 = rhs[1];
        Vectorized<float> vf2 = rhs[2];
        Vectorized<float> vf3 = rhs[3];
        vfloat32 vecf0 = vf0.vec0();
        vfloat32 vecf1 = vf0.vec1();
        vfloat32 vecf2 = vf1.vec0();
        vfloat32 vecf3 = vf1.vec1();

        vfloat32 vecf4 = vf2.vec0();
        vfloat32 vecf5 = vf2.vec1();
        vfloat32 vecf6 = vf3.vec0();
        vfloat32 vecf7 = vf3.vec1();

        vecf0 = vec_mul(vecf0, inverse_scale_v);
        vecf1 = vec_mul(vecf1, inverse_scale_v);
        vecf2 = vec_mul(vecf2, inverse_scale_v);
        vecf3 = vec_mul(vecf3, inverse_scale_v);

        vecf4 = vec_mul(vecf4, inverse_scale_v);
        vecf5 = vec_mul(vecf5, inverse_scale_v);
        vecf6 = vec_mul(vecf6, inverse_scale_v);
        vecf7 = vec_mul(vecf7, inverse_scale_v);

        vecf0 = vec_add(vec_rint(vecf0), vec_zero_point);
        vecf1 = vec_add(vec_rint(vecf1), vec_zero_point);
        vecf2 = vec_add(vec_rint(vecf2), vec_zero_point);
        vecf3 = vec_add(vec_rint(vecf3), vec_zero_point);

        vecf4 = vec_add(vec_rint(vecf4), vec_zero_point);
        vecf5 = vec_add(vec_rint(vecf5), vec_zero_point);
        vecf6 = vec_add(vec_rint(vecf6), vec_zero_point);
        vecf7 = vec_add(vec_rint(vecf7), vec_zero_point);

        vint32 veci0 = vec_signed(vecf0);
        vint32 veci1 = vec_signed(vecf1);
        vint32 veci2 = vec_signed(vecf2);
        vint32 veci3 = vec_signed(vecf3);

        vint32 veci4 = vec_signed(vecf4);
        vint32 veci5 = vec_signed(vecf5);
        vint32 veci6 = vec_signed(vecf6);
        vint32 veci7 = vec_signed(vecf7);

        // veci0 = vec_min(vmax, vec_max( vmin, vecf0)) ;
        // veci1 = vec_min(vmax, vec_max( vmin, vecf1)) ;
        // veci2 = vec_min(vmax, vec_max( vmin, vecf2)) ;
        // veci3 = vec_min(vmax, vec_max( vmin, vecf3)) ;

        // veci4 = vec_min(vmax, vec_max( vmin, vecf4)) ;
        // veci5 = vec_min(vmax, vec_max( vmin, vecf5)) ;
        // veci6 = vec_min(vmax, vec_max( vmin, vecf6)) ;
        // veci7 = vec_min(vmax, vec_max( vmin, vecf7)) ;
        // vec_packs CLAMP already
        vint16 vecshi0 = vec_packs(veci0, veci1);
        vint16 vecshi1 = vec_packs(veci2, veci3);
        vint16 vecshi2 = vec_packs(veci4, veci5);
        vint16 vecshi3 = vec_packs(veci6, veci7);

        vint8 vec0 = vec_packs(vecshi0, vecshi1);
        vint8 vec1 = vec_packs(vecshi2, vecshi3);

        return {vec0, vec1};
        */
    }

    #[inline(always)]
    pub fn relu(&self, zero_point: Vectorized<qint8>) -> Vectorized<qint8> {
        
        todo!();
        /*
            return {vec_max(_vec0, zero_point._vec0), vec_max(_vec1, zero_point._vec1)};
        */
    }

    #[inline(always)]
    pub fn relu6(&self, 
        zero_point: Vectorized<qint8>,
        q_six:      Vectorized<qint8>) -> Vectorized<qint8> {
        
        todo!();
        /*
            vint8 max0 = vec_max(_vec0, zero_point._vec0);
        vint8 max1 = vec_max(_vec1, zero_point._vec1);
        return {vec_min(max0, q_six._vec0), vec_min(max1, q_six._vec1)};
        */
    }
    
    pub fn widening_subtract(&self, b: Vectorized<qint8>) -> IntVecReturnType {
        
        todo!();
        /*
            vint16 vecshi0 = vec_unpackh(_vec0);
        vint16 vecBshi0 = vec_unpackh(b._vec0);
        vint16 vecshi1 = vec_unpackl(_vec0);
        vint16 vecBshi1 = vec_unpackl(b._vec0);

        vint16 vecshi2 = vec_unpackh(_vec1);
        vint16 vecBshi2 = vec_unpackh(b._vec1);
        vint16 vecshi3 = vec_unpackl(_vec1);
        vint16 vecBshi3 = vec_unpackl(b._vec1);

        vint32 veci0 = vec_unpackh(vecshi0);
        vint32 vecBi0 = vec_unpackh(vecBshi0);
        vint32 veci1 = vec_unpackl(vecshi0);
        vint32 vecBi1 = vec_unpackl(vecBshi0);

        vint32 veci2 = vec_unpackh(vecshi1);
        vint32 vecBi2 = vec_unpackh(vecBshi1);
        vint32 veci3 = vec_unpackl(vecshi1);
        vint32 vecBi3 = vec_unpackl(vecBshi1);

        vint32 veci4 = vec_unpackh(vecshi2);
        vint32 vecBi4 = vec_unpackh(vecBshi2);
        vint32 veci5 = vec_unpackl(vecshi2);
        vint32 vecBi5 = vec_unpackl(vecBshi2);

        vint32 veci6 = vec_unpackh(vecshi3);
        vint32 vecBi6 = vec_unpackh(vecBshi3);
        vint32 veci7 = vec_unpackl(vecshi3);
        vint32 vecBi7 = vec_unpackl(vecBshi3);

        return {
            Vectorized<qint32>(veci0 - vecBi0, veci1 - vecBi1),
            Vectorized<qint32>(veci2 - vecBi2, veci3 - vecBi3),
            Vectorized<qint32>(veci4 - vecBi4, veci5 - vecBi5),
            Vectorized<qint32>(veci6 - vecBi6, veci7 - vecBi7)};
        */
    }
    
    pub fn requantize_from_int(
        inp:        &IntVecReturnType,
        multiplier: f32,
        zero_point: i32) -> Vectorized<qint8> {
        
        todo!();
        /*
            vfloat32 vec_multiplier = vec_splats(multiplier);
        vint32 vec_zero_point = vec_splats(zero_point);

        Vectorized<qint32> vi0 = inp[0];
        Vectorized<qint32> vi1 = inp[1];
        Vectorized<qint32> vi2 = inp[2];
        Vectorized<qint32> vi3 = inp[3];

        vfloat32 vecf0 = vec_float(vi0.vec0());
        vfloat32 vecf1 = vec_float(vi0.vec1());
        vfloat32 vecf2 = vec_float(vi1.vec0());
        vfloat32 vecf3 = vec_float(vi1.vec1());

        vfloat32 vecf4 = vec_float(vi2.vec0());
        vfloat32 vecf5 = vec_float(vi2.vec1());
        vfloat32 vecf6 = vec_float(vi3.vec0());
        vfloat32 vecf7 = vec_float(vi3.vec1());

        vecf0 = vec_mul(vecf0, vec_multiplier);
        vecf1 = vec_mul(vecf1, vec_multiplier);
        vecf2 = vec_mul(vecf2, vec_multiplier);
        vecf3 = vec_mul(vecf3, vec_multiplier);

        vecf4 = vec_mul(vecf4, vec_multiplier);
        vecf5 = vec_mul(vecf5, vec_multiplier);
        vecf6 = vec_mul(vecf6, vec_multiplier);
        vecf7 = vec_mul(vecf7, vec_multiplier);

        vecf0 = vec_rint(vecf0);
        vecf1 = vec_rint(vecf1);
        vecf2 = vec_rint(vecf2);
        vecf3 = vec_rint(vecf3);

        vecf4 = vec_rint(vecf4);
        vecf5 = vec_rint(vecf5);
        vecf6 = vec_rint(vecf6);
        vecf7 = vec_rint(vecf7);

        vint32 veci0 = vec_signed(vecf0);
        vint32 veci1 = vec_signed(vecf1);
        vint32 veci2 = vec_signed(vecf2);
        vint32 veci3 = vec_signed(vecf3);

        vint32 veci4 = vec_signed(vecf4);
        vint32 veci5 = vec_signed(vecf5);
        vint32 veci6 = vec_signed(vecf6);
        vint32 veci7 = vec_signed(vecf7);

        veci0 = vec_add(veci0, vec_zero_point);
        veci1 = vec_add(veci1, vec_zero_point);
        veci2 = vec_add(veci2, vec_zero_point);
        veci3 = vec_add(veci3, vec_zero_point);

        veci4 = vec_add(veci4, vec_zero_point);
        veci5 = vec_add(veci5, vec_zero_point);
        veci6 = vec_add(veci6, vec_zero_point);
        veci7 = vec_add(veci7, vec_zero_point);

        vint16 vecshi0 = vec_packs(veci0, veci1);
        vint16 vecshi1 = vec_packs(veci2, veci3);
        vint16 vecshi2 = vec_packs(veci4, veci5);
        vint16 vecshi3 = vec_packs(veci6, veci7);

        vint8 vec0 = vec_packs(vecshi0, vecshi1);
        vint8 vec1 = vec_packs(vecshi2, vecshi3);

        return {vec0, vec1};
        */
    }
    
    pub fn dump(&self)  {
        
        todo!();
        /*
            value_type vals[size()];
        store((void*)vals);
        for (int i = 0; i < size(); ++i) {
          cout << (int)(vals[i]) << " ";
        }
        cout << endl;
        */
    }
}

#[inline] pub fn maximum(
    a: &Vectorized<qint8>,
    b: &Vectorized<qint8>) -> Vectorized<qint8> {
    
    todo!();
        /*
            return a.maximum(b);
        */
}

#[inline] pub fn minimum(
    a: &Vectorized<qint8>,
    b: &Vectorized<qint8>) -> Vectorized<qint8> {
    
    todo!();
        /*
            return a.minimum(b);
        */
}
