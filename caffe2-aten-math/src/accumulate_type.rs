crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/AccumulateType.h]

/**
  | Defines the accumulation type for a
  | scalar type.
  | 
  | Example: using accscalar_t = acc_type<Scalar,
  | true>;
  |
  */
pub trait AccumulateType<const is_cuda: bool> {
    type Type;
}

#[macro_export] macro_rules! map_accumulate_type {
    (cuda,$src:ty,$dst:ty) => {
        impl AccumulateType<true>  for $src { type Type = $dst; }
    };
    ($src:ty,$dst:ty) => {
        impl AccumulateType<false>  for $src { type Type = $dst; }
    }
}

#[cfg(any(feature = "cudacc", feature = "hipcc"))] 
map_accumulate_type!{cuda,half,f32}

map_accumulate_type!{cuda, bf16,         f32}
map_accumulate_type!{cuda, Complex<f32>, Complex<f32>}
map_accumulate_type!{cuda, Complex<f64>, Complex<f64>}
map_accumulate_type!{cuda, f16,          f32}
map_accumulate_type!{cuda, bool,         bool}
map_accumulate_type!{cuda, char,         i64}
map_accumulate_type!{cuda, f32,          f32}
map_accumulate_type!{cuda, f64,          f64}
map_accumulate_type!{cuda, i16,          i64}
map_accumulate_type!{cuda, i32,          i64}
map_accumulate_type!{cuda, i64,          i64}
map_accumulate_type!{cuda, i8,           i64}
map_accumulate_type!{cuda, u8,           i64}

map_accumulate_type!{bf16,         f32}
map_accumulate_type!{Complex<f32>, Complex<f64>}
map_accumulate_type!{Complex<f64>, Complex<f64>}
map_accumulate_type!{f16,          f32}
map_accumulate_type!{bool,         bool}
map_accumulate_type!{char,         i64}
map_accumulate_type!{f32,          f64}
map_accumulate_type!{f64,          f64}
map_accumulate_type!{i16,          i64}
map_accumulate_type!{i32,          i64}
map_accumulate_type!{i64,          i64}
map_accumulate_type!{i8,           i64}
map_accumulate_type!{u8,           i64}

pub type acc_type<T, const is_cuda: bool> = <T as AccumulateType<is_cuda>>::Type;

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/AccumulateType.cpp]
pub fn to_accumulate_type<ScalarType>(
    ty:      ScalarType,
    is_cuda: bool) -> ScalarType {
    
    todo!();
        /*
            switch (type) {
    #define DEFINE_CASE(Scalar, TypeNum)                                  \
        case ScalarType::TypeNum:                                           \
          return is_cuda ?                                                  \
              CppTypeToScalarType<at::acc_type<Scalar, true>>::value :    \
              CppTypeToScalarType<at::acc_type<Scalar, false>>::value;

        AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_CASE)
    #undef DEFINE_CASE

        default: TORCH_INTERNAL_ASSERT(false, "Unrecognized ScalarType: ", type);
      }
        */
}
