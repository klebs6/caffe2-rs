crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/Dispatch.h]

/**
  | The method should_include_kernel_dtype()
  | returns true/false based on whether
  | the switching code for a specific dtype
  | should be included based on build time
  | constants generated from tracing model
  | execution. This method will be implmeneted
  | via code-generation and included in
  | this file when code-gen is ready.
  |
  */
#[cfg(not(TEMPLATE_SELECTIVE_BUILD))]
#[inline] pub fn should_include_kernel_dtype(
        kernel_tag_str: *const u8,
        scalar_type:    ScalarType) -> bool {
    
    todo!();
        /*
            return true;
        */
}

/**
  | In the Facebook internal build (using
  | BUCK), this macro is enabled by passing
  | in -c pt.enable_record_kernel_dtype=1
  | when building the tracer binary.
  |
  */
#[cfg(ENABLE_RECORD_KERNEL_FUNCTION_DTYPE)]
#[macro_export] macro_rules! record_kernel_function_dtype {
    ($NAME:ident, $enum_type:ident) => {
        /*
        
          {RECORD_FUNCTION_WITH_SCOPE(                                             
            RecordScope::KERNEL_FUNCTION_DTYPE,                                
            string(NAME) + "$" + toString(enum_type),                         
            {});}
        */
    }
}

#[cfg(not(ENABLE_RECORD_KERNEL_FUNCTION_DTYPE))]
#[macro_export] macro_rules! record_kernel_function_dtype { ($NAME:ident, $enum_type:ident) => { } }

#[cfg(__cpp_if_constexpr)]
#[macro_export] macro_rules! at_private_case_type_using_hint {
    ($NAME:ident, $enum_type:ident, $type:ident, $HINT:ident, $($arg:ident),*) => {
        /*
        
          case enum_type: {                                                              
            if constexpr (!should_include_kernel_dtype(NAME, enum_type)) {           
              AT_ERROR("dtype '", toString(enum_type), "' not selected for kernel tag ", #NAME); 
            }                                                                            
            using HINT = type;                                                           
            return __VA_ARGS__();                                                        
          }
        */
    }
}

#[cfg(not(__cpp_if_constexpr))]
#[macro_export] macro_rules! at_private_case_type_using_hint {
    ($NAME:ident, $enum_type:ident, $type:ident, $HINT:ident, $($arg:ident),*) => {
        /*
        
          case enum_type: {                                                              
            if_constexpr<(!should_include_kernel_dtype(NAME, enum_type))>( 
              [] {                                                                       
                AT_ERROR("dtype '" #enum_type "' not selected for kernel tag " #NAME);   
              }                                                                          
            );                                                                           
            using HINT = type;                                                           
            return __VA_ARGS__();                                                        
          }
        */
    }
}

#[macro_export] macro_rules! at_private_case_type {
    ($NAME:ident, $enum_type:ident, $type:ident, $($arg:ident),*) => {
        /*
        
          AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, Scalar, __VA_ARGS__)
        */
    }
}

/**
  | Workaround for  because CUDA 10.1 and below
  | fails to handle unused attribute in the type
  | aliasing context.
  |
  | Keep name long and verbose to avoid macro
  | collisions.
  |
  */
lazy_static!{
    /*
    #if defined(__CUDACC__) && CUDA_VERSION <= 10100
    #define _DISPATCH_CUDA_WORKAROUND
    #else
    #define _DISPATCH_CUDA_WORKAROUND 
    #endif // defined(__CUDACC__) && CUDA_VERSION <= 10100
    */
}

#[macro_export] macro_rules! at_qint_private_case_type {
    () => {
        /*
                (                                           
            enum_type, type, underlying_enum, underlying_type, ...)                  
          case enum_type: {                                                          
            using Scalar = type;                                                   
            using underlying_t _DISPATCH_CUDA_WORKAROUND =                 
                Scalar::underlying;                                                
            const auto& SCALAR_TYPE _DISPATCH_CUDA_WORKAROUND = enum_type; 
            const auto& UNDERLYING_TYPE _DISPATCH_CUDA_WORKAROUND =        
                toUnderlying(enum_type);                                             
            (void)SCALAR_TYPE;  /* Suppress unused-var compiler warning */           
            /* TODO: Use [[maybe-unused]] when C++17 becomes the standard */         
            return __VA_ARGS__();                                                    
          }
        */
    }
}

#[macro_export] macro_rules! at_qint_sub_byte_private_case_type {
    () => {
        /*
                (                                       
            enum_type, type, underlying_type, bitwidth, qmin, qmax, ...)                  
          case enum_type: {                                                               
            using Scalar = type;                                                        
            using underlying_t _DISPATCH_CUDA_WORKAROUND =                      
                Scalar::underlying;                                                     
            const auto& SCALAR_TYPE _DISPATCH_CUDA_WORKAROUND = enum_type;      
            const auto& UNDERLYING_TYPE _DISPATCH_CUDA_WORKAROUND =             
                toUnderlying(enum_type);                                                  
            int bit_width = bitwidth;                                                     
            i64 quant_min = qmin;                                                     
            i64 quant_max = qmax;                                                     
            (void)bit_width; /* Suppress unused variable warning */                       
            (void)quant_min; /* Suppress unused variable warning */                       
            (void)quant_max; /* Suppress unused variable warning */                       
            return __VA_ARGS__();                                                         
          }
        */
    }
}

#[inline] pub fn scalar_type(s: ScalarType) -> ScalarType {
    
    todo!();
        /*
            return s;
        */
}

#[deprecated = "passing DeprecatedTypeProperties to an AT_DISPATCH macro is deprecated, pass an ScalarType instead"]
#[inline] pub fn scalar_type_with_deprecated_type_properties(t: &DeprecatedTypeProperties) -> ScalarType {
    
    todo!();
        /*
            return t.scalarType();
        */
}

#[deprecated = "AT_DISPATCH_ALL_TYPES_AND_HALF is deprecated, use AT_DISPATCH_ALL_TYPES_AND(ScalarType::Half, ...) instead"]
#[inline] pub fn deprecated_at_dispatch_all_types_and_half()  {
    
    todo!();
        /*
        
        */
}

#[deprecated = "AT_DISPATCH_ALL_TYPES_AND_HALF_AND_COMPLEX is deprecated, use AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(ScalarType::Half, ...) instead"]
#[inline] pub fn deprecated_at_dispatch_all_types_and_half_and_complex()  {
    
    todo!();
        /*
        
        */
}

/*
  | The AT_DISPATCH_* family of macros provides the
  | ability to conveniently generate
  | specializations of a kernel over all of the
  | dtypes we care about in PyTorch.  We call it
  | "dispatch" because we are "dispatching" to the
  | correct, dtype-specific kernel.
  |
  | A standard usage looks like:
  |
  |      AT_DISPATCH_ALL_TYPES(self.scalar_type(), "op_name", [&] {
  |          // Your code here, with 'Scalar' now defined to
  |          // be the dtype in question
  |      })
  |
  | There are many variations of this macro, so
  | it's important to understand exactly /which/
  | dtypes you want to get instantiated, as well as
  | what the "default" set is.
  |
  | The default set of dtypes that are instantiated
  | (e.g., by AT_DISPATCH_ALL_TYPES) are floating
  | point types (float, double), and integral types
  | (i32, i64, i16, i8, u8),
  | but NOT booleans (bool), half-precision floats
  | (Half) or complex number (complex<float>,
  | complex<double>).
  |
  | This "cut" is somewhat historical (the default
  | types are the ones that TH historically
  | supported), but it also reflects the fact that
  | the non-default types are "poorly" behaved
  | (booleans are NOT integers mod 2, half
  | precision operations ~essentially don't exist
  | on CPU, complex numbers are an experimental
  | application).
  |
  | Here are the questions you should generally ask
  | to decide which dispatch you want:
  |
  | 1. Is this an integral or floating point
  |    specific operation? (If so, you'll want one
  |    of the FLOATING or INTEGRAL macros.)
  |
  | 2. Should half be supported?  (If you're on
  |    CPU, the answer is almost definitely no.  If
  |    you do want support, use one of the AND_HALF
  |    macros)
  |
  | Much rarer situations:
  |
  | 3. Should bool be supported?  (You often have
  |    to write your kernel differently if
  |    arithmetic operations are involved.)  If so,
  |    Use AT_DISPATCH_ALL_TYPES_AND along with
  |    ScalarType::Bool
  |
  | 4. Should complex be supported?  The answer is
  |    almost always no, unless you are working on
  |    "generic" code that should work on all
  |    dtypes.
  |
  | Parameters: -----------
  |
  | 1. The NAME argument is a "tag" that is used to
  |    trace and then conditionally compile
  |    fragments of the case statements such that
  |    the kernel functions are specialized only
  |    for the dtypes that are needed. The NAME
  |    parameter *must* be a build time cons char*
  |    (can't be string, etc...)
  |
  | Please ensure that the NAME is unique for every
  | implementation or you run the risk of
  | over-including code for the kernel
  | functions. There is no risk of missing out on
  | any code, so it's mostly a risk of a Type-2
  | error, and not a Type-1 error.
  |
  */

/**
  | NB: the the_type variable is not used, but we
  | have kept it for backwards compatibility.  It's
  | probably not used by anyone though; but we're
  | just being safe (and it doesn't hurt.)  Note we
  | must use it to shut up warnings about unused
  | store.
  */
#[macro_export] macro_rules! at_dispatch_floating_types {
    ($TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                     
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op */ 
            ScalarType _st = ::scalar_type(the_type);                   
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                
            switch (_st) {                                                          
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Double, double, __VA_ARGS__)     
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Float, float, __VA_ARGS__)       
              default:                                                              
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      
            }                                                                       
          }()
        */
    }
}

#[macro_export] macro_rules! at_dispatch_floating_types_and_half {
    ($TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                        
            const auto& the_type = TYPE;                                               
            /* don't use TYPE again in case it is an expensive or side-effect op */    
            ScalarType _st = ::scalar_type(the_type);                      
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                   
            switch (_st) {                                                             
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Double, double, __VA_ARGS__)  
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Float, float, __VA_ARGS__)    
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Half, Half, __VA_ARGS__)  
              default:                                                                 
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");         
            }                                                                          
          }()
        */
    }
}

#[macro_export] macro_rules! at_dispatch_floating_types_and {
    ($SCALARTYPE:ident, $TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                     
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op */ 
            ScalarType _st = ::scalar_type(the_type);                   
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                
            switch (_st) {                                                          
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Double, double, __VA_ARGS__)  
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Float, float, __VA_ARGS__)    
              AT_PRIVATE_CASE_TYPE(NAME,                                               
                  SCALARTYPE,                                                       
                  decltype(ScalarTypeToCPPType<SCALARTYPE>::t),          
                  __VA_ARGS__)                                                      
              default:                                                              
                AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     
            }                                                                       
          }()
        */
    }
}

#[macro_export] macro_rules! at_dispatch_floating_types_and2 {
    ($SCALARTYPE1:ident, $SCALARTYPE2:ident, $TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                     
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op */ 
            ScalarType _st = ::scalar_type(the_type);                   
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                
            switch (_st) {                                                          
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Double, double, __VA_ARGS__)  
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Float, float, __VA_ARGS__)    
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  SCALARTYPE1,                                                      
                  decltype(ScalarTypeToCPPType<SCALARTYPE1>::t),         
                  __VA_ARGS__)                                                      
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  SCALARTYPE2,                                                      
                  decltype(ScalarTypeToCPPType<SCALARTYPE2>::t),         
                  __VA_ARGS__)                                                      
              default:                                                              
                AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     
            }                                                                       
          }()
        */
    }
}

#[macro_export] macro_rules! at_dispatch_floating_and_complex_types {
    ($TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                     
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op */ 
            ScalarType _st = ::scalar_type(the_type);                   
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                
            switch (_st) {                                                          
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Double, double, __VA_ARGS__)  
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Float, float, __VA_ARGS__)    
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  ScalarType::ComplexDouble,                                    
                  complex<double>,                                             
                  __VA_ARGS__)                                                      
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  ScalarType::ComplexFloat,                                     
                  complex<float>,                                              
                  __VA_ARGS__)                                                      
              default:                                                              
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      
            }                                                                       
          }()
        */
    }
}

#[macro_export] macro_rules! at_dispatch_floating_and_complex_types_and1 {
    ($SCALARTYPE:ident, $TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                     
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op */ 
            ScalarType _st = ::scalar_type(the_type);                   
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                
            switch (_st) {                                                          
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Double, double, __VA_ARGS__)  
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Float, float, __VA_ARGS__)    
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  ScalarType::ComplexDouble, complex<double>, __VA_ARGS__) 
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  ScalarType::ComplexFloat, complex<float>, __VA_ARGS__)   
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  SCALARTYPE,                                                       
                  decltype(ScalarTypeToCPPType<SCALARTYPE>::t),          
                  __VA_ARGS__)                                                      
              default:                                                              
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      
            }                                                                       
          }()
        */
    }
}

#[macro_export] macro_rules! at_dispatch_floating_and_complex_types_and2 {
    ($SCALARTYPE1:ident, $SCALARTYPE2:ident, $TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                     
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op */ 
            ScalarType _st = ::scalar_type(the_type);                       
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                    
            switch (_st) {                                                              
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Double, double, __VA_ARGS__)   
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Float, float, __VA_ARGS__)     
              AT_PRIVATE_CASE_TYPE(                                                     
                  NAME,                                                                 
                  ScalarType::ComplexDouble,                                        
                  complex<double>,                                                 
                  __VA_ARGS__)                                                          
              AT_PRIVATE_CASE_TYPE(                                                     
                  NAME,                                                                 
                  ScalarType::ComplexFloat,                                         
                  complex<float>,                                                  
                  __VA_ARGS__)                                                          
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  SCALARTYPE1,                                                      
                  decltype(ScalarTypeToCPPType<SCALARTYPE1>::t),         
                  __VA_ARGS__)                                                      
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  SCALARTYPE2,                                                      
                  decltype(ScalarTypeToCPPType<SCALARTYPE2>::t),         
                  __VA_ARGS__)                                                      
              default:                                                              
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      
            }                                                                       
          }()
        */
    }
}

#[macro_export] macro_rules! at_dispatch_integral_types {
    ($TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                     
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op */ 
            ScalarType _st = ::scalar_type(the_type);                   
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                
            switch (_st) {                                                          
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Byte, u8, __VA_ARGS__)      
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Char, i8, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Int, i32, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Long, i64, __VA_ARGS__)      
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Short, i16, __VA_ARGS__)     
              default:                                                              
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      
            }                                                                       
          }()
        */
    }
}

#[macro_export] macro_rules! at_dispatch_integral_types_and {
    ($SCALARTYPE:ident, $TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                     
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op */ 
            ScalarType _st = ::scalar_type(the_type);                   
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                
            switch (_st) {                                                      
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Byte, u8, __VA_ARGS__)  
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Char, i8, __VA_ARGS__)   
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Int, i32, __VA_ARGS__)   
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Long, i64, __VA_ARGS__)  
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Short, i16, __VA_ARGS__) 
              AT_PRIVATE_CASE_TYPE(NAME,                                              
                  SCALARTYPE,                                                   
                  decltype(ScalarTypeToCPPType<SCALARTYPE>::t),      
                  __VA_ARGS__)                                                  
              default:                                                          
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");  
            }                                                                   
          }()
        */
    }
}

#[macro_export] macro_rules! at_dispatch_all_types {
    ($TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                      
            const auto& the_type = TYPE;                                             
            /* don't use TYPE again in case it is an expensive or side-effect op  */ 
            ScalarType _st = ::scalar_type(the_type);                    
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                 
            switch (_st) {                                                              
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Byte, u8, __VA_ARGS__)    
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Char, i8, __VA_ARGS__)     
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Double, double, __VA_ARGS__)   
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Float, float, __VA_ARGS__)     
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Int, i32, __VA_ARGS__)     
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Long, i64, __VA_ARGS__)    
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Short, i16, __VA_ARGS__)   
              default:                                                                  
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");          
            }                                                                           
          }()
        */
    }
}

#[macro_export] macro_rules! at_dispatch_complex_types {
    ($TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                     
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op */ 
            ScalarType _st = ::scalar_type(the_type);                   
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                
            switch (_st) {                                                          
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  ScalarType::ComplexFloat,                                     
                  complex<float>,                                              
                  __VA_ARGS__)                                                      
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  ScalarType::ComplexDouble,                                    
                  complex<double>,                                             
                  __VA_ARGS__)                                                      
              default:                                                              
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      
            }                                                                       
          }()
        */
    }
}

#[macro_export] macro_rules! at_dispatch_qint_types {
    ($TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                     
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op */ 
            ScalarType _st = ::scalar_type(the_type);                   
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                
            switch (_st) {                                                          
              AT_QINT_PRIVATE_CASE_TYPE(                                            
                  kQInt8, qint8, kChar, i8, __VA_ARGS__)            
              AT_QINT_PRIVATE_CASE_TYPE(                                            
                  kQUInt8, quint8, kByte, u8, __VA_ARGS__)         
              AT_QINT_PRIVATE_CASE_TYPE(                                            
                  kQInt32, qint32, kInt, int, __VA_ARGS__)              
              default:                                                              
                AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     
            }                                                                       
          }()
        */
    }
}

#[macro_export] macro_rules! at_dispatch_qint_and_sub_byte_types {
    ($TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                                        
            const auto& the_type = TYPE;                                                               
            /* don't use TYPE again in case it is an expensive or side-effect op */                    
            ScalarType _st = ::scalar_type(the_type);                                      
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                                   
            switch (_st) {                                                                             
              AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                                      
                  kQInt8, qint8, i8, CHAR_BIT, SCHAR_MIN, SCHAR_MAX, __VA_ARGS__)          
              AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                                      
                  kQUInt8, quint8, u8, CHAR_BIT, 0, UCHAR_MAX, __VA_ARGS__)               
              AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                                      
                  kQInt32, qint32, int, CHAR_BIT * sizeof(int), INT_MIN, INT_MAX, __VA_ARGS__) 
              AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                                      
                  kQUInt4x2, quint4x2, u8, 4, 0, 15, __VA_ARGS__)                         
              default:                                                                                 
                AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");                        
            }                                                                                          
          }()
        */
    }
}

#[macro_export] macro_rules! at_dispatch_all_types_and_complex {
    ($TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                     
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op*/  
            ScalarType _st = ::scalar_type(the_type);                   
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                
            switch (_st) {                                                          
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Byte, u8, __VA_ARGS__)      
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Char, i8, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Double, double, __VA_ARGS__)     
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Float, float, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Int, i32, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Long, i64, __VA_ARGS__)      
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Short, i16, __VA_ARGS__)     
              AT_PRIVATE_CASE_TYPE(NAME,                                                  
                  ScalarType::ComplexFloat, complex<float>, __VA_ARGS__)   
              AT_PRIVATE_CASE_TYPE(NAME,                                                  
                  ScalarType::ComplexDouble, complex<double>, __VA_ARGS__) 
              default:                                                              
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      
            }                                                                       
          }()
        */
    }
}

#[macro_export] macro_rules! at_dispatch_all_types_and {
    ($SCALARTYPE:ident, $TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                 
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op*/  
            ScalarType _st = ::scalar_type(the_type);                   
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                
            switch (_st) {                                                          
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Byte, u8, __VA_ARGS__)  
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Char, i8, __VA_ARGS__)   
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Double, double, __VA_ARGS__) 
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Float, float, __VA_ARGS__)   
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Int, i32, __VA_ARGS__)   
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Long, i64, __VA_ARGS__)  
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Short, i16, __VA_ARGS__) 
              AT_PRIVATE_CASE_TYPE(                                             
                  NAME,                                                         
                  SCALARTYPE,                                                   
                  decltype(ScalarTypeToCPPType<SCALARTYPE>::t),      
                  __VA_ARGS__)                                                  
              default:                                                          
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");  
            }                                                                   
          }()
        */
    }
}

#[macro_export] macro_rules! at_dispatch_all_types_and_complex_and {
    ($SCALARTYPE:ident, $TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                     
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op*/  
            ScalarType _st = ::scalar_type(the_type);                   
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                
            switch (_st) {                                                         
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Byte, u8, __VA_ARGS__)      
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Char, i8, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Double, double, __VA_ARGS__)     
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Float, float, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Int, i32, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Long, i64, __VA_ARGS__)      
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Short, i16, __VA_ARGS__)     
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  ScalarType::ComplexFloat,                                     
                  complex<float>,                                              
                  __VA_ARGS__)                                                      
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  ScalarType::ComplexDouble,                                    
                  complex<double>,                                             
                  __VA_ARGS__)                                                      
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  SCALARTYPE,                                                       
                  decltype(ScalarTypeToCPPType<SCALARTYPE>::t),          
                  __VA_ARGS__)                                                      
              default:                                                              
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      
            }                                                                       
          }()
        */
    }
}

#[macro_export] macro_rules! at_dispatch_all_types_and2 {
    ($SCALARTYPE1:ident, $SCALARTYPE2:ident, $TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                       
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op*/  
            ScalarType _st = ::scalar_type(the_type);                   
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                
            switch (_st) {                                                           
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Byte, u8, __VA_ARGS__)        
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Char, i8, __VA_ARGS__)         
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Double, double, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Float, float, __VA_ARGS__)         
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Int, i32, __VA_ARGS__)         
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Long, i64, __VA_ARGS__)        
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Short, i16, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(                                                   
                  NAME,                                                               
                  SCALARTYPE1,                                                        
                  decltype(ScalarTypeToCPPType<SCALARTYPE1>::t),           
                  __VA_ARGS__)                                                        
              AT_PRIVATE_CASE_TYPE(                                                   
                  NAME,                                                               
                  SCALARTYPE2,                                                        
                  decltype(ScalarTypeToCPPType<SCALARTYPE2>::t),           
                  __VA_ARGS__)                                                        
              default:                                                                
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");        
            }                                                                         
          }()
        */
    }
}

#[macro_export] macro_rules! at_dispatch_all_types_and_complex_and2 {
    ($SCALARTYPE1:ident, $SCALARTYPE2:ident, $TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                     
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op*/  
            ScalarType _st = ::scalar_type(the_type);                   
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                
            switch (_st) {                                                          
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Byte, u8, __VA_ARGS__)      
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Char, i8, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Double, double, __VA_ARGS__)     
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Float, float, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Int, i32, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Long, i64, __VA_ARGS__)      
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Short, i16, __VA_ARGS__)     
              AT_PRIVATE_CASE_TYPE(                                                       
                  NAME, ScalarType::ComplexFloat, complex<float>, __VA_ARGS__)   
              AT_PRIVATE_CASE_TYPE(                                                       
                  NAME, ScalarType::ComplexDouble, complex<double>, __VA_ARGS__) 
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  SCALARTYPE1,                                                      
                  decltype(ScalarTypeToCPPType<SCALARTYPE1>::t),         
                  __VA_ARGS__)                                                      
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  SCALARTYPE2,                                                      
                  decltype(ScalarTypeToCPPType<SCALARTYPE2>::t),         
                  __VA_ARGS__)                                                      
              default:                                                              
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      
            }                                                                       
          }()
        */
    }
}


#[macro_export] macro_rules! at_dispatch_all_types_and3 {
    ($SCALARTYPE1:ident, $SCALARTYPE2:ident, $SCALARTYPE3:ident, $TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
            [&] {                                                                     
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op*/  
            ScalarType _st = ::scalar_type(the_type);                   
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                
            switch (_st) {                                                          
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Byte, u8, __VA_ARGS__)  
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Char, i8, __VA_ARGS__)   
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Double, double, __VA_ARGS__) 
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Float, float, __VA_ARGS__)   
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Int, i32, __VA_ARGS__)   
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Long, i64, __VA_ARGS__)  
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Short, i16, __VA_ARGS__) 
              AT_PRIVATE_CASE_TYPE(                                             
                  NAME,                                                         
                  SCALARTYPE1,                                                  
                  decltype(ScalarTypeToCPPType<SCALARTYPE1>::t),     
                  __VA_ARGS__)                                                  
              AT_PRIVATE_CASE_TYPE(                                             
                  NAME,                                                         
                  SCALARTYPE2,                                                  
                  decltype(ScalarTypeToCPPType<SCALARTYPE2>::t),     
                  __VA_ARGS__)                                                  
              AT_PRIVATE_CASE_TYPE(                                             
                  NAME,                                                         
                  SCALARTYPE3,                                                  
                  decltype(ScalarTypeToCPPType<SCALARTYPE3>::t),     
                  __VA_ARGS__)                                                  
              default:                                                          
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");  
            }                                                                   
          }()
        */
    }
}


#[macro_export] macro_rules! at_dispatch_all_types_and_complex_and3 {
    ($SCALARTYPE1:ident, $SCALARTYPE2:ident, $SCALARTYPE3:ident, $TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                     
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op*/  
            ScalarType _st = ::scalar_type(the_type);                   
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                
            switch (_st) {                                                          
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Byte, u8, __VA_ARGS__)      
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Char, i8, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Double, double, __VA_ARGS__)     
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Float, float, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Int, i32, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Long, i64, __VA_ARGS__)      
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Short, i16, __VA_ARGS__)     
              AT_PRIVATE_CASE_TYPE(                                                       
                  NAME, ScalarType::ComplexFloat, complex<float>, __VA_ARGS__)   
              AT_PRIVATE_CASE_TYPE(                                                       
                  NAME, ScalarType::ComplexDouble, complex<double>, __VA_ARGS__) 
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  SCALARTYPE1,                                                      
                  decltype(ScalarTypeToCPPType<SCALARTYPE1>::t),         
                  __VA_ARGS__)                                                      
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  SCALARTYPE2,                                                      
                  decltype(ScalarTypeToCPPType<SCALARTYPE2>::t),         
                  __VA_ARGS__)                                                      
              AT_PRIVATE_CASE_TYPE(                                                 
                  NAME,                                                             
                  SCALARTYPE3,                                                      
                  decltype(ScalarTypeToCPPType<SCALARTYPE3>::t),         
                  __VA_ARGS__)                                                      
              default:                                                              
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      
            }                                                                       
          }()
        */
    }
}


#[macro_export] macro_rules! at_dispatch_index_types {
    ($TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                     
            const auto& the_index_type = TYPE;                                      
            /* don't use TYPE again in case it is an expensive or side-effect op */ 
            ScalarType _it = ::scalar_type(the_index_type);             
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _it)                                 
            switch (_it) {                                                          
              AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, ScalarType::Int, i32, Index, __VA_ARGS__) 
              AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, ScalarType::Long, i64, Index, __VA_ARGS__)
              default:                                                              
                AT_ERROR(#NAME, " not implemented for '", toString(_it), "'");      
            }                                                                       
          }()
        */
    }
}


// ----------------------------------------------------------------------------
// DEPRECATED MACROS, DON'T USE THESE
// ----------------------------------------------------------------------------

#[macro_export] macro_rules! at_dispatch_all_types_and_half {
    ($TYPE:ident, $NAME:ident, $($arg:ident),*) => {
        /*
        
          [&] {                                                                     
            deprecated_AT_DISPATCH_ALL_TYPES_AND_HALF();                    
            const auto& the_type = TYPE;                                            
            /* don't use TYPE again in case it is an expensive or side-effect op */ 
            ScalarType _st = ::scalar_type(the_type);                   
            RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                
            switch (_st) {                                                          
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Byte, u8, __VA_ARGS__)      
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Char, i8, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Double, double, __VA_ARGS__)     
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Float, float, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Int, i32, __VA_ARGS__)       
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Long, i64, __VA_ARGS__)      
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Short, i16, __VA_ARGS__)     
              AT_PRIVATE_CASE_TYPE(NAME, ScalarType::Half, Half, __VA_ARGS__)     
              default:                                                              
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      
            }                                                                       
          }()
        */
    }
}
