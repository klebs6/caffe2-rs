crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/Scalar.h]

macro_rules! DEFINE_IMPLICIT_CTOR {
    ($type:ident, $name:ident) => {
        /*
        
          Scalar(type vv) : Scalar(vv, true) {}
        */
    }
}

lazy_static!{
    /*
    at_forall_scalar_types_and2!{f16, BFloat16, DEFINE_IMPLICIT_CTOR}
    at_forall_complex_types!{DEFINE_IMPLICIT_CTOR}
    */
}

/**
  | Scalar represents a 0-dimensional
  | tensor which contains a single element.
  | 
  | Unlike a tensor, numeric literals (in
  | C++) are implicitly convertible to
  | Scalar (which is why, for example, we
  | provide both add(Tensor) and add(Scalar)
  | overloads for many operations). 
  |
  | It may also be used in circumstances where
  | you statically know a tensor is 0-dim
  | and single size, but don't know its type.
  |
  */
pub struct Scalar {
    tag: ScalarTag,
    v:   ScalarValue,
}

impl Default for Scalar {
    
    fn default() -> Self {
        todo!();
        /*
        : scalar(int64_t(0)),

        
        */
    }
}

macro_rules! DEFINE_ACCESSOR {
    ($type:ident, $name:ident) => {
        /*
        
          type to##name() const {                                             
            if (Tag::HAS_d == tag) {                                          
              return checked_convert<type, double>(v.d, #type);               
            } else if (Tag::HAS_z == tag) {                                   
              return checked_convert<type, complex<double>>(v.z, #type); 
            }                                                                 
            if (Tag::HAS_b == tag) {                                          
              return checked_convert<type, bool>(v.i, #type);                 
            } else {                                                          
              return checked_convert<type, int64_t>(v.i, #type);              
            }                                                                 
          }
        */
    }
}

// TODO: Support Complexf16 accessor
at_forall_scalar_types_with_complex_except_complex_half!{DEFINE_ACCESSOR}

pub enum ScalarTag { 
    HAS_d, 
    HAS_i, 
    HAS_z, 
    HAS_b 
}

pub enum ScalarValue {

    /// d 
    Double(f64),           

    /// i
    Int(i64),

    /// c
    Complex(Complex<f64>),
}

impl Scalar {

    /**
     | Value* is both implicitly convertible to
     | SymbolicVariable and bool which causes
     | ambiguity error. Specialized constructor for
     | bool resolves this problem.
     |
     */
    //template < typename T, typename enable_if<is_same<T, bool>::value, bool>::type* = nullptr>
    pub fn new<T>(vv: T) -> Self {

        todo!();
        /*
        : tag(Tag::HAS_b),

            v.i = convert<int64_t, bool>(vv);
        */
    }
    
    pub fn is_floating_point(&self) -> bool {
        
        todo!();
        /*
            return Tag::HAS_d == tag;
        */
    }

    pub fn is_integral(&self, include_bool: bool) -> bool {
        
        todo!();
        /*
            return Tag::HAS_i == tag || (includeBool && isBoolean());
        */
    }
    
    pub fn is_complex(&self) -> bool {
        
        todo!();
        /*
            return Tag::HAS_z == tag;
        */
    }
    
    pub fn is_boolean(&self) -> bool {
        
        todo!();
        /*
            return Tag::HAS_b == tag;
        */
    }
    
    pub fn equal_not_complex<T: Real>(&self, num: T) -> bool {
        
        todo!();
        /*
            if (isComplex()) {
          auto val = v.z;
          return (val.real() == num) && (val.imag() == T());
        } else if (isFloatingPoint()) {
          return v.d == num;
        } else if (isIntegral(/*includeBool=*/false)) {
          return v.i == num;
        } else {
          // boolean scalar does not equal to a non boolean value
          return false;
        }
        */
    }

    pub fn equal_complex<T: ComplexFloat>(&self, num: T) -> bool {
        
        todo!();
        /*
            if (isComplex()) {
          return v.z == num;
        } else if (isFloatingPoint()) {
          return (v.d == num.real()) && (num.imag() == T());
        } else if (isIntegral(/*includeBool=*/false)) {
          return (v.i == num.real()) && (num.imag() == T());
        } else {
          // boolean scalar does not equal to a non boolean value
          return false;
        }
        */
    }
    
    pub fn equal(&self, num: bool) -> bool {
        
        todo!();
        /*
            if (isBoolean()) {
          return static_cast<bool>(v.i) == num;
        } else {
          return false;
        }
        */
    }
    
    pub fn ty(&self) -> ScalarType {
        
        todo!();
        /*
            if (isComplex()) {
          return ScalarType::ComplexDouble;
        } else if (isFloatingPoint()) {
          return ScalarType::Double;
        } else if (isIntegral(/*includeBool=*/false)) {
          return ScalarType::Long;
        } else if (isBoolean()) {
          return ScalarType::Bool;
        } else {
          throw runtime_error("Unknown scalar type.");
        }
        */
    }

    pub fn new_with_bool<T: PrimInt>(vv: T, _1: bool) -> Self {
    
        todo!();
        /*
        : tag(Tag::HAS_i),

            v.i = convert<decltype(v.i), T>(vv);
        */
    }

    pub fn new_not_int_not_complex<T>(vv: T, _1: bool) -> Self {
    
        todo!();
        /*
        : tag(Tag::HAS_d),

            v.d = convert<decltype(v.d), T>(vv);
        */
    }

    pub fn new_complex<T: ComplexFloat>(vv: T, _1: bool) -> Self {
    
        todo!();
        /*
        : tag(Tag::HAS_z),

            v.z = convert<decltype(v.z), T>(vv);
        */
    }

    // We can't set v in the initializer list using the
    // syntax v{ .member = ... } because it doesn't work on MSVC
}

/// define the scalar.to<int64_t>() specializations
macro_rules! DEFINE_TO {
    ($T:ident, $name:ident) => {
        /*
        
          template <>                      
          inline T Scalar::to<T>() const { 
            return to##name();             
          }
        */
    }
}

at_forall_scalar_types_with_complex_except_complex_half!{DEFINE_TO}

//-------------------------------------------[.cpp/pytorch/c10/core/Scalar.cpp]

impl Neg for Scalar {

    type Output = Self;
    
    #[inline] fn neg(self) -> Self::Output {
        todo!();
        /*
            TORCH_CHECK(
          !isBoolean(),
          "torch boolean negative, the `-` operator, is not supported.");
      if (isFloatingPoint()) {
        return Scalar(-v.d);
      } else if (isComplex()) {
        return Scalar(-v.z);
      } else {
        return Scalar(-v.i);
      }
        */
    }
}

impl Scalar {
    
    pub fn conj(&self) -> Scalar {
        
        todo!();
        /*
          if (isComplex()) {
            return Scalar(conj(v.z));
          } else {
            return *this;
          }
        */
    }

    pub fn log(&self) -> Scalar {
        
        todo!();
        /*
          if (isComplex()) {
            return log(v.z);
          } else if (isFloatingPoint()) {
            return log(v.d);
          } else {
            return log(v.i);
          }
        */
    }
}
