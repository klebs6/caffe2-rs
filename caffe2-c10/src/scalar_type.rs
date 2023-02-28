/*!
  | For the macros below:
  |
  | NB: If you want to macro some code for all
  | non-QInt scalar types (i.e. types with complete
  | information, you probably want one of the
  | AT_FORALL_SCALAR_TYPES
  | / AT_FORALL_SCALAR_TYPES_AND macros below,
  | which are designed to behave similarly to the
  | Dispatch macros with the same name.
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/ScalarType.h]

/**
  | NB: Order matters for this macro; it is relied
  | upon in _promoteTypesLookup and the
  | serialization format.
  |
  */
#[macro_export] macro_rules! at_forall_scalar_types_with_complex_and_qints {
    ($_:ident) => {
        /*
        
          _(uint8_t, Byte) /* 0 */                               
          _(int8_t, Char) /* 1 */                                
          _(int16_t, Short) /* 2 */                              
          _(int, Int) /* 3 */                                    
          _(int64_t, Long) /* 4 */                               
          _(f16, f16) /* 5 */                              
          _(float, Float) /* 6 */                                
          _(double, Double) /* 7 */                              
          _(complex<f16>, Complexf16) /* 8 */        
          _(complex<float>, ComplexFloat) /* 9 */           
          _(complex<double>, ComplexDouble) /* 10 */        
          _(bool, Bool) /* 11 */                                 
          _(qint8, QInt8) /* 12 */                          
          _(quint8, QUInt8) /* 13 */                        
          _(qint32, QInt32) /* 14 */                        
          _(BFloat16, BFloat16) /* 15 */                     
          _(quint4x2, QUInt4x2) /* 16 */
        */
    }
}

/**
  | If you want to support Complexf16 for real,
  | add Complexf16 into this macro (and change the
  | name).  But beware: convert() doesn't work for
  | all the conversions you need...
  |
  */
#[macro_export] macro_rules! at_forall_scalar_types_with_complex_except_complex_half {
    ($_:ident) => {
        /*
        
          _(uint8_t, Byte)                                                 
          _(int8_t, Char)                                                  
          _(int16_t, Short)                                                
          _(int, Int)                                                      
          _(int64_t, Long)                                                 
          _(f16, f16)                                                
          _(float, Float)                                                  
          _(double, Double)                                                
          _(complex<float>, ComplexFloat)                             
          _(complex<double>, ComplexDouble)                           
          _(bool, Bool)                                                    
          _(BFloat16, BFloat16)
        */
    }
}

#[repr(i8)]
pub enum ScalarType {

    /*
    macro_rules! DEFINE_ENUM {
        ($_1:ident, $n:ident) => {
            /*
                    n,
              AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ENUM)
            */
        }
    }
    */

    Undefined,
    NumOptions
}

pub const NUM_SCALAR_TYPES: u16 = ScalarType::NumOptions as u16;

/// These are used to map ScalarTypes to C++
/// types.
///
macro_rules! specialize_scalar_type_to_cpp_type {
    ($cpp_type:ident, $scalar_type:ident) => {
        /*
        
          template <>                                                                
          struct ScalarTypeToCPPType<ScalarType::scalar_type> {                 
            using type = cpp_type;                                                   
                                                                                     
            /* This is a workaround for the Cuda bug which prevents */               
            /* ::ScalarTypeToCType<T>::type being used directly due to */    
            /* ambiguous reference which can't to be resolved. For some reason it */ 
            /* cant pick between detail and detail. */                 
            /* For repro example, please see: */                                     
            /* https://gist.github.com/izdeby/952ae7cf256ddb740a73776d39a7e7ba */    
            /* TODO: remove once the bug is fixed. */                                
            static type t;                                                           
          };
        */
    }
}

at_forall_scalar_types_with_complex_and_qints!{SPECIALIZE_ScalarTypeToCPPType}

macro_rules! specialize_cpp_type_to_scalar_type {
    ($cpp_type:ident, $scalar_type:ident) => {
        /*
        
          template <>                                                                  
          struct CppTypeToScalarType<cpp_type>                                         
              :                                                                   
                    integral_constant<ScalarType, ScalarType::scalar_type> { 
          };
        */
    }
}

at_forall_scalar_types_with_complex_and_qints!{SPECIALIZE_CppTypeToScalarType}

#[macro_export] macro_rules! at_forall_int_types {
    ($_:ident) => {
        /*
        
          _(uint8_t, Byte)             
          _(int8_t, Char)              
          _(int16_t, Short)            
          _(int, Int)                  
          _(int64_t, Long)
        */
    }
}

#[macro_export] macro_rules! at_forall_scalar_types {
    ($_:ident) => {
        /*
        
          _(uint8_t, Byte)                
          _(int8_t, Char)                 
          _(int16_t, Short)               
          _(int, Int)                     
          _(int64_t, Long)                
          _(float, Float)                 
          _(double, Double)
        */
    }
}

#[macro_export] macro_rules! at_forall_scalar_types_and {
    ($SCALARTYPE:ident, $_:ident) => {
        /*
        
          _(uint8_t, Byte)                                
          _(int8_t, Char)                                 
          _(int16_t, Short)                               
          _(int, Int)                                     
          _(int64_t, Long)                                
          _(float, Float)                                 
          _(double, Double)                               
          _(decltype(::ScalarTypeToCPPType<    
                     ::ScalarType::SCALARTYPE>::t),  
            SCALARTYPE)
        */
    }
}

#[macro_export] macro_rules! at_forall_scalar_types_and2 {
    ($SCALARTYPE1:ident, $SCALARTYPE2:ident, $_:ident) => {
        /*
        
          _(uint8_t, Byte)                                               
          _(int8_t, Char)                                                
          _(int16_t, Short)                                              
          _(int, Int)                                                    
          _(int64_t, Long)                                               
          _(float, Float)                                                
          _(double, Double)                                              
          _(decltype(::ScalarTypeToCPPType<                   
                     ::ScalarType::SCALARTYPE1>::t),                
            SCALARTYPE1)                                                 
          _(decltype(::ScalarTypeToCPPType<                   
                     ::ScalarType::SCALARTYPE2>::t),                
            SCALARTYPE2)
        */
    }
}

#[macro_export] macro_rules! at_forall_scalar_types_and3 {
    ($SCALARTYPE1:ident, $SCALARTYPE2:ident, $SCALARTYPE3:ident, $_:ident) => {
        /*
        
          _(uint8_t, Byte)                                                            
          _(int8_t, Char)                                                             
          _(int16_t, Short)                                                           
          _(int, Int)                                                                 
          _(int64_t, Long)                                                            
          _(float, Float)                                                             
          _(double, Double)                                                           
          _(decltype(::ScalarTypeToCPPType<                                
                     ::ScalarType::SCALARTYPE1>::t),                             
            SCALARTYPE1)                                                              
          _(decltype(::ScalarTypeToCPPType<                                
                     ::ScalarType::SCALARTYPE2>::t),                             
            SCALARTYPE2)                                                              
          _(decltype(::ScalarTypeToCPPType<                                
                     ::ScalarType::SCALARTYPE3>::t),                             
            SCALARTYPE3)
        */
    }
}

#[macro_export] macro_rules! at_forall_qint_types {
    ($_:ident) => {
        /*
        
          _(qint8, QInt8)          
          _(quint8, QUInt8)        
          _(qint32, QInt32)        
          _(quint4x2, QUInt4x2)
        */
    }
}

#[macro_export] macro_rules! at_forall_complex_types {
    ($_:ident) => {
        /*
        
          _(complex<float>, ComplexFloat) 
          _(complex<double>, ComplexDouble)
        */
    }
}

#[macro_export] macro_rules! define_constant {
    ($_:ident, $name:ident) => {
        /*
        
          constexpr ScalarType k##name = ScalarType::name;
        */
    }
}

at_forall_scalar_types_with_complex_and_qints!{DEFINE_CONSTANT}

#[inline] pub fn to_string(t: ScalarType) -> *const u8 {
    
    todo!();
        /*
            #define DEFINE_CASE(_, name) \
      case ScalarType::name:     \
        return #name;

      switch (t) {
        AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CASE)
        default:
          return "UNKNOWN_SCALAR";
      }
    #undef DEFINE_CASE
        */
}

#[inline] pub fn element_size(t: ScalarType) -> usize {
    
    todo!();
        /*
            #define CASE_ELEMENTSIZE_CASE(ctype, name) \
      case ScalarType::name:                   \
        return sizeof(ctype);

      switch (t) {
        AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(CASE_ELEMENTSIZE_CASE)
        default:
          TORCH_CHECK(false, "Unknown ScalarType");
      }
    #undef CASE_ELEMENTSIZE_CASE
        */
}

#[deprecated = "isIntegralType is deprecated. Please use the overload with 'includeBool' parameter instead."]
#[inline] pub fn is_integral_type_a(t: ScalarType) -> bool {
    
    todo!();
        /*
            return (
          t == ScalarType::Byte || t == ScalarType::Char || t == ScalarType::Int ||
          t == ScalarType::Long || t == ScalarType::Short);
        */
}

#[inline] pub fn is_integral_type_b(
        t:            ScalarType,
        include_bool: bool) -> bool {
    
    todo!();
        /*
            bool isIntegral =
          (t == ScalarType::Byte || t == ScalarType::Char || t == ScalarType::Int ||
           t == ScalarType::Long || t == ScalarType::Short);

      return includeBool ? isIntegral || (t == ScalarType::Bool) : isIntegral;
        */
}

#[inline] pub fn is_floating_type(t: ScalarType) -> bool {
    
    todo!();
        /*
            return (
          t == ScalarType::Double || t == ScalarType::Float ||
          t == ScalarType::f16 || t == ScalarType::BFloat16);
        */
}

#[inline] pub fn is_complex_type(t: ScalarType) -> bool {
    
    todo!();
        /*
            return (
          t == ScalarType::Complexf16 || t == ScalarType::ComplexFloat ||
          t == ScalarType::ComplexDouble);
        */
}

#[inline] pub fn is_qint_type(t: ScalarType) -> bool {
    
    todo!();
        /*
            // Don't forget to extend this when adding new QInt types
      return t == ScalarType::QInt8 || t == ScalarType::QUInt8 ||
          t == ScalarType::QInt32 || t == ScalarType::QUInt4x2;
        */
}

#[inline] pub fn to_qint_type(t: ScalarType) -> ScalarType {
    
    todo!();
        /*
            switch (t) {
        case ScalarType::Byte:
          return ScalarType::QUInt8;
        case ScalarType::Char:
          return ScalarType::QInt8;
        case ScalarType::Int:
          return ScalarType::QInt32;
        default:
          return t;
      }
        */
}

#[inline] pub fn to_underlying(t: ScalarType) -> ScalarType {
    
    todo!();
        /*
            switch (t) {
        case ScalarType::QUInt8:
          return ScalarType::Byte;
        case ScalarType::QInt8:
          return ScalarType::Char;
        case ScalarType::QInt32:
          return ScalarType::Int;
        case ScalarType::QUInt4x2:
          return ScalarType::Byte;
        default:
          return t;
      }
        */
}

#[inline] pub fn is_signed_type(t: ScalarType) -> bool {
    
    todo!();
        /*
            TORCH_CHECK(!isQIntType(t), "isSignedType not supported for quantized types");
    #define CASE_SIGNED(ctype, name) \
      case ScalarType::name:         \
        return numeric_limits<ctype>::is_signed;

      switch (t) {
        case ScalarType::Complexf16:
        case ScalarType::ComplexFloat:
        case ScalarType::ComplexDouble:
          return true;
          AT_FORALL_SCALAR_TYPES_AND3(f16, Bool, BFloat16, CASE_SIGNED)
        default:
          TORCH_CHECK(false, "Unknown ScalarType");
      }
    #undef CASE_SIGNED
        */
}

#[inline] pub fn is_underlying(
        ty:    ScalarType,
        qtype: ScalarType) -> bool {
    
    todo!();
        /*
            return type == toUnderlying(qtype);
        */
}

#[inline] pub fn to_value_type(t: ScalarType) -> ScalarType {
    
    todo!();
        /*
            switch (t) {
        case ScalarType::Complexf16:
          return ScalarType::f16;
        case ScalarType::ComplexFloat:
          return ScalarType::Float;
        case ScalarType::ComplexDouble:
          return ScalarType::Double;
        default:
          return t;
      }
        */
}

#[inline] pub fn to_complex_type(t: ScalarType) -> ScalarType {
    
    todo!();
        /*
            switch (t) {
        case ScalarType::f16:
          return ScalarType::Complexf16;
        case ScalarType::Float:
          return ScalarType::ComplexFloat;
        case ScalarType::Double:
          return ScalarType::ComplexDouble;
        case ScalarType::Complexf16:
          return ScalarType::Complexf16;
        case ScalarType::ComplexFloat:
          return ScalarType::ComplexFloat;
        case ScalarType::ComplexDouble:
          return ScalarType::ComplexDouble;
        default:
          TORCH_CHECK(false, "Unknown Complex ScalarType for ", t);
      }
        */
}

/**
  | see tensor_attributes.rst for detailed
  | explanation and examples of casting
  | rules.
  |
  */
#[inline] pub fn can_cast(
        from: ScalarType,
        to:   ScalarType) -> bool {
    
    todo!();
        /*
            // We disallow complex -> non complex, e.g., float_tensor *= complex is
      // disallowed.
      if (isComplexType(from) && !isComplexType(to)) {
        return false;
      }
      // We disallow float -> integral, e.g., int_tensor *= float is disallowed.
      if (isFloatingType(from) && isIntegralType(to, false)) {
        return false;
      }

      // Treat bool as a distinct "category," to be consistent with type promotion
      // rules (e.g. `bool_tensor + 5 -> int64_tensor`). If `5` was in the same
      // category as `bool_tensor`, we would not promote. Differing categories
      // implies `bool_tensor += 5` is disallowed.
      //
      // NB: numpy distinguishes "unsigned" as a category to get the desired
      // `bool_tensor + 5 -> int64_tensor` behavior. We don't, because:
      // * We don't want the performance hit of checking the runtime sign of
      // Scalars.
      // * `uint8_tensor + 5 -> int64_tensor` would be undesirable.
      if (from != ScalarType::Bool && to == ScalarType::Bool) {
        return false;
      }
      return true;
        */
}

#[inline] pub fn promote_types(
    a: ScalarType,
    b: ScalarType) -> ScalarType {
    
    todo!();
        /*
            // This is generated according to NumPy's promote_types
      constexpr auto u1 = ScalarType::Byte;
      constexpr auto i1 = ScalarType::Char;
      constexpr auto i2 = ScalarType::Short;
      constexpr auto i4 = ScalarType::Int;
      constexpr auto i8 = ScalarType::Long;
      constexpr auto f2 = ScalarType::f16;
      constexpr auto f4 = ScalarType::Float;
      constexpr auto f8 = ScalarType::Double;
      constexpr auto c2 = ScalarType::Complexf16;
      constexpr auto c4 = ScalarType::ComplexFloat;
      constexpr auto c8 = ScalarType::ComplexDouble;
      constexpr auto b1 = ScalarType::Bool;
      constexpr auto bf = ScalarType::BFloat16;
      constexpr auto ud = ScalarType::Undefined;
      if (a == ud || b == ud) {
        return ScalarType::Undefined;
      }

      // For QInt types, we only allow exact match
      if (isQIntType(a) && a == b) {
        return a;
      }

      if (isQIntType(a) || isQIntType(b)) {
        TORCH_CHECK(
            false,
            "promoteTypes with quantized numbers is not handled yet; figure out what the correct rules should be, offending types: ",
            toString(a),
            " ",
            toString(b));
      }

      // this matrix has to be consistent with AT_FORALL_SCALAR_TYPES_WITH_COMPLEX
      // so that's why we have to add undefined as we are not sure what is the
      // corrent values for the type promotions in complex type cases.
      static constexpr ScalarType _promoteTypesLookup[static_cast<int>(
          ScalarType::NumOptions)][static_cast<int>(ScalarType::NumOptions)] = {
          /*        u1  i1  i2  i4  i8  f2  f4  f8  c2  c4  c8  b1  q1  q2  q3  bf*/
          /* u1 */ {u1, i2, i2, i4, i8, f2, f4, f8, ud, c4, c8, u1, ud, ud, ud, bf},
          /* i1 */ {i2, i1, i2, i4, i8, f2, f4, f8, ud, c4, c8, i1, ud, ud, ud, bf},
          /* i2 */ {i2, i2, i2, i4, i8, f2, f4, f8, ud, c4, c8, i2, ud, ud, ud, bf},
          /* i4 */ {i4, i4, i4, i4, i8, f2, f4, f8, ud, c4, c8, i4, ud, ud, ud, bf},
          /* i8 */ {i8, i8, i8, i8, i8, f2, f4, f8, ud, c4, c8, i8, ud, ud, ud, bf},
          /* f2 */ {f2, f2, f2, f2, f2, f2, f4, f8, ud, c4, c8, f2, ud, ud, ud, f4},
          /* f4 */ {f4, f4, f4, f4, f4, f4, f4, f8, ud, c4, c8, f4, ud, ud, ud, f4},
          /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, ud, c8, c8, f8, ud, ud, ud, f8},
          /* c2 */ {ud, ud, ud, ud, ud, ud, ud, ud, c2, c4, c8, ud, ud, ud, ud, ud},
          /* c4 */ {c4, c4, c4, c4, c4, c4, c4, c8, c4, c4, c8, c4, ud, ud, ud, c4},
          /* c8 */ {c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, ud, ud, ud, c8},
          /* b1 */ {u1, i1, i2, i4, i8, f2, f4, f8, ud, c4, c8, b1, ud, ud, ud, bf},
          /* q1 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
          /* q2 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
          /* q3 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
          /* bf */ {bf, bf, bf, bf, bf, f4, f4, f8, ud, c4, c8, bf, ud, ud, ud, bf},
      };
      return _promoteTypesLookup[static_cast<int>(a)][static_cast<int>(b)];
        */
}

impl fmt::Display for ScalarType {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            return stream << toString(scalar_type);
        */
    }
}

macro_rules! at_forautocast_scalar_types {
    ($_:ident) => {
        /*
        
          _(half, f16) /* 0 */                
          _(bfloat16, BFloat16) /* 1 */
        */
    }
}
