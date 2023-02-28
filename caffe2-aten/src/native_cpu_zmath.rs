/*!
  | Complex number math operations that
  | act as no-ops for other dtypes.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/zmath.h]

//-----------------------------------
pub trait ZAbs<Output> {

    #[inline] fn zabs(z: Self) -> Output {

        todo!();
            /*
                return z;
            */
    }
}

impl ZAbs<Complex<f32>> for Complex<f32> {

    #[inline] fn zabs(z: Complex<f32>) -> Complex<f32> {
        
        todo!();
            /*
                return complex<float>(abs(z));
            */
    }
}

impl ZAbs<f32> for Complex<f32> {

    #[inline] fn zabs(z: Complex<f32>) -> f32 {
        
        todo!();
            /*
                return abs(z);
            */
    }
}

impl ZAbs<Complex<f64>> for Complex<f64> {

    #[inline] fn zabs(z: Complex<f64>) -> Complex<f64> {
        
        todo!();
            /*
                return complex<double>(abs(z));
            */
    }
}

impl ZAbs<f64> for Complex<f64> {

    #[inline] fn zabs(z: Complex<f64>) -> f64 {
        
        todo!();
            /*
                return abs(z);
            */
    }
}

//-----------------------------------
pub trait AngleImpl<Output> {

    /**
      | This overload corresponds to non-complex
      | dtypes.
      |
      | The function is consistent with its NumPy
      | equivalent for non-complex dtypes where `pi` is
      | returned for negative real numbers and `0` is
      | returned for 0 or positive real numbers.
      |
      | Note: `nan` is propagated.
      |
      */
    #[inline] fn angle_impl(z: Self) -> Output {

        todo!();
            /*
                if (_isnan(z)) {
            return z;
          }
          return z < 0 ? pi<double> : 0;
            */
    }
}

impl AngleImpl<Complex<f32>> for Complex<f32> {

    #[inline] fn angle_impl(z: Complex<f32>) -> Complex<f32> {
        
        todo!();
            /*
                return complex<float>(arg(z), 0.0);
            */
    }
}

impl AngleImpl<f32> for Complex<f32> {

    #[inline] fn angle_impl(z: Complex<f32>) -> f32 {
        
        todo!();
            /*
                return arg(z);
            */
    }
}

impl AngleImpl<Complex<f64>> for Complex<f64> {

    #[inline] fn angle_impl(z: Complex<f64>) -> Complex<f64> {
        
        todo!();
            /*
                return complex<double>(arg(z), 0.0);
            */
    }
}

impl AngleImpl<f64> for Complex<f64> {

    #[inline] fn angle_impl(z: Complex<f64>) -> f64 {
        
        todo!();
            /*
                return arg(z);
            */
    }
}

//-----------------------------------
pub trait RealImpl<Output>: Scalar {

    fn real_impl(z: Self) -> Output {

        todo!();
            /*
                return z; //No-Op
            */
    }
}

impl RealImpl<Self> for Complex<f32> {

    fn real_impl(z: Complex<f32>) -> Complex<f32> {
        
        todo!();
            /*
                return complex<float>(z.real(), 0.0);
            */
    }
}

impl RealImpl<f32> for Complex<f32> {

    fn real_impl(z: Complex<f32>) -> f32 {
        
        todo!();
            /*
                return z.real();
            */
    }
}

impl RealImpl<Complex<f64>> for Complex<f64> {

    fn real_impl(z: Complex<f64>) -> Complex<f64> {
        
        todo!();
            /*
                return complex<double>(z.real(), 0.0);
            */
    }
}

impl RealImpl<f64> for Complex<f64> {

    fn real_impl(z: Complex<f64>) -> f64 {
        
        todo!();
            /*
                return z.real();
            */
    }
}

//-----------------------------------
pub trait ImagImpl<Output>: Scalar {

    fn imag_impl(z: Self) -> Output {

        todo!();

        /*
            return 0;
        */
    }
}

impl ImagImpl<Self> for Complex<f32> {

    fn imag_impl(z: Complex<f32>) -> Complex<f32> {
        
        todo!();
            /*
                return complex<float>(z.imag(), 0.0);
            */
    }
}

impl ImagImpl<f32> for Complex<f32> {

    fn imag_impl(z: Complex<f32>) -> f32 {
        
        todo!();
            /*
                return z.imag();
            */
    }
}

impl ImagImpl<Self> for Complex<f64> {

    fn imag_impl(z: Complex<f64>) -> Complex<f64> {
        
        todo!();
            /*
                return complex<double>(z.imag(), 0.0);
            */
    }
}

impl ImagImpl<f64> for Complex<f64> {

    fn imag_impl(z: Complex<f64>) -> f64 {
        
        todo!();
            /*
                return z.imag();
            */
    }
}

//-----------------------------------
pub trait ConjImpl {

    #[inline] fn conj_impl(z: Self) -> Self {

        todo!();
            /*
                return z; //No-Op
            */
    }
}

impl ConjImpl for Complex<f32> {

    #[inline] fn conj_impl(z: Complex<f32>) -> Complex<f32> {
        
        todo!();
            /*
                return complex<float>(z.real(), -z.imag());
            */
    }
}

impl ConjImpl for Complex<f64> {

    #[inline] fn conj_impl(z: Complex<f64>) -> Complex<f64> {
        
        todo!();
            /*
                return complex<double>(z.real(), -z.imag());
            */
    }
}

//-----------------------------------
pub trait CeilImpl {

    #[inline] fn ceil_impl(z: Self) -> Self {

        todo!();
            /*
                return ceil(z);
            */
    }
}

impl CeilImpl for Complex<f32> {

    #[inline] fn ceil_impl(z: Complex<f32>) -> Complex<f32> {
        
        todo!();
            /*
                return complex<float>(ceil(z.real()), ceil(z.imag()));
            */
    }
}

impl CeilImpl for Complex<f64> {

    #[inline] fn ceil_impl(z: Complex<f64>) -> Complex<f64> {
        
        todo!();
            /*
                return complex<double>(ceil(z.real()), ceil(z.imag()));
            */
    }
}

//-----------------------------------
pub trait SgnImpl {

    #[inline] fn sgn_impl<T>(z: Self) -> Self;
}

impl<T> SgnImpl for Complex<T> {

    #[inline] fn sgn_impl(z: Complex<T>) -> Complex<T> {

        todo!();
            /*
          if (z == complex<T>(0, 0)) {
            return complex<T>(0, 0);
          } else {
            return z / zabs(z);
          }
            */
    }
}

//-----------------------------------
pub trait FloorImpl {

    #[inline] fn floor_impl(z: Self) -> Self {

        todo!();
            /*
                return floor(z);
            */
    }
}

impl FloorImpl for Complex<f32> {

    #[inline] fn floor_impl(z: Complex<f32>) -> Complex<f32> {
        
        todo!();
            /*
                return complex<float>(floor(z.real()), floor(z.imag()));
            */
    }
}

impl FloorImpl for Complex<f64> {

    #[inline] fn floor_impl(z: Complex<f64>) -> Complex<f64> {
        
        todo!();
            /*
                return complex<double>(floor(z.real()), floor(z.imag()));
            */
    }
}

//-----------------------------------
pub trait RoundImpl {

    #[inline] fn round_impl(z: Self) -> Self {

        todo!();
            /*
                return nearbyint(z);
            */
    }
}

impl RoundImpl for Complex<f32> {

    #[inline] fn round_impl(z: Complex<f32>) -> Complex<f32> {
        
        todo!();
            /*
                return complex<float>(nearbyint(z.real()), nearbyint(z.imag()));
            */
    }
}

impl RoundImpl for Complex<f64> {

    #[inline] fn round_impl(z: Complex<f64>) -> Complex<f64> {
        
        todo!();
            /*
                return complex<double>(nearbyint(z.real()), nearbyint(z.imag()));
            */
    }
}

//-----------------------------------
pub trait TruncImpl {

    #[inline] fn trunc_impl(z: Self) -> Self {

        todo!();
            /*
                return trunc(z);
            */
    }
}


impl TruncImpl for Complex<f32> {

    #[inline] fn trunc_impl(z: Complex<f32>) -> Complex<f32> {
        
        todo!();
            /*
                return complex<float>(trunc(z.real()), trunc(z.imag()));
            */
    }
}

impl TruncImpl for Complex<f64> {

    #[inline] fn trunc_impl(z: Complex<f64>) -> Complex<f64> {
        
        todo!();
            /*
                return complex<double>(trunc(z.real()), trunc(z.imag()));
            */
    }
}

//-----------------------------------
pub trait MaxImpl {

    #[inline] fn max_impl(a: Self, b: Self) -> Self {
        
        todo!();
            /*
                if (_isnan<TYPE>(a) || _isnan<TYPE>(b)) {
            return numeric_limits<TYPE>::quiet_NaN();
          } else {
            return max(a, b);
          }
            */
    }
}

impl<T> MaxImpl for Complex<T> {

    #[inline] fn max_impl(a: Self, b: Self) -> Self {
        
        todo!();
            /*
          if (_isnan<TYPE>(a)) {
            return a;
          } else if (_isnan<TYPE>(b)) {
            return b;
          } else {
            return abs(a) > abs(b) ? a : b;
          }
            */
    }
}

//-----------------------------------
pub trait MinImpl {

    #[inline] fn min_impl(a: Self, b: Self) -> Self {
        
        todo!();
            /*
                if (_isnan<TYPE>(a) || _isnan<TYPE>(b)) {
            return numeric_limits<TYPE>::quiet_NaN();
          } else {
            return min(a, b);
          }
            */
    }
}

impl<T> MinImpl for Complex<T> {

    #[inline] fn min_impl(a: Self, b: Self) -> Self {
        
        todo!();
            /*
                if (_isnan<TYPE>(a)) {
            return a;
          } else if (_isnan<TYPE>(b)) {
            return b;
          } else {
            return abs(a) < abs(b) ? a : b;
          }
            */
    }
}
