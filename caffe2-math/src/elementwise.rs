crate::ix!();

/**
  |note that i probably fucked up the
  |caffe2_use_eigen_for_blas and caffe2_use_mkl
  |switches and the macro invocations should also
  |probably be covered by the switches
  */
#[inline] pub fn exp<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn log<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn log1p<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn sin<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn asin<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn cos<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn acos<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn tan<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn atan<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn sinh<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn cosh<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn sin_cos<T, Context>(
    n:       i32,
    x:       *const T,
    s:       *mut T,
    c:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn tanh<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn abs<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn sqr<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn sqrt<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn rsqrt<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn cube<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn cbrt<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn neg<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn sign<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn not<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn powx<T, Context>(
    n:       i32,
    a:       *const T,
    b:       T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn inv<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn erf<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn cdf_norm<T, Context>(
    n:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}


#[inline] pub fn set<T, Context>(
    n:       i64,
    alpha:   T,
    x:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn scale<TAlpha, TData, Context>(
    n:       i64,
    alpha:   TAlpha,
    x:       *const TData,
    y:       *mut TData,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

/**
  | Different from the Scale function above, if
  | alpha is passed in as a pointer, we will
  | assume that it lives on the Context device,
  | for example on GPU.
  */
#[inline] pub fn scale_with_alpha_from_pointer<TAlpha, TData, Context>(
    n:       i64,
    alpha:   *const TAlpha,
    x:       *const TData,
    y:       *mut TData,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}


#[inline] pub fn add<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn sub<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn mul<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn div<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}


#[inline] pub fn min<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn max<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}


#[inline] pub fn and<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn or<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn xor<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}


#[inline] pub fn bitwise_and<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn bitwise_or<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn bitwise_xor<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}


#[inline] pub fn eQ<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut bool,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn nE<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut bool,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn lT<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut bool,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn lE<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut bool,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn gT<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut bool,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn gE<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    c:       *mut bool,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn axpy<TAlpha, TData, Context>(
    n:       i64,
    alpha:   TAlpha,
    x:       *const TData,
    y:       *mut TData,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

/**
  | Different from the Axpy function above, if
  | alpha is passed in as a pointer, we will
  | assume that it lives on the Context device,
  | for example on GPU.
  */
#[inline] pub fn axpy_with_alpha_as_ptr<TAlpha, TData, Context>(
    n:       i64,
    alpha:   *const TAlpha,
    x:       *const TData,
    y:       *mut TData,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn axpby<TAlpha, TData, Context>(
    n:       i64,
    alpha:   *const TAlpha,
    x:       *const TData,
    beta:    *const TAlpha,
    y:       *mut TData,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[cfg(caffe2_use_mkl)]
#[macro_export] macro_rules! delegate_sincos {
    ($T:ty, $MKLFunc:ident) => {
        /*
        template <>                                                           
            C10_EXPORT void SinCos<T, CPUContext>(                                
                const int N, const T* X, T* S, T* C, CPUContext* /* context */) { 
                MKLFunc(N, X, S, C);                                                
            }
                */
    }
}

#[cfg(caffe2_use_mkl)] delegate_sincos!{f32,  vsSinCos}
#[cfg(caffe2_use_mkl)] delegate_sincos!{f64, vdSinCos}

#[cfg(caffe2_use_mkl)]
#[macro_export] macro_rules! delegate_powx {
    ($T:ty, $MKLFunc:ident) => {
        /*
        template <>                                                                
            C10_EXPORT void Powx<T, CPUContext>(                                       
                const int N, const T* A, const T b, T* Y, CPUContext* /* context */) { 
                MKLFunc(N, A, b, Y);                                                     
            }
                */
    }
}

#[cfg(caffe2_use_mkl)] delegate_powx!{f32,  vsPowx}
#[cfg(caffe2_use_mkl)] delegate_powx!{f64, vdPowx}

#[cfg(caffe2_use_mkl)]
#[macro_export] macro_rules! delegate_simple_binary_function {
    ($T:ty, $Func:ident, MKLFunc) => {
        /*
        template <>                                                                 
            C10_EXPORT void Func<T, CPUContext>(                                        
                const int N, const T* A, const T* B, T* C, CPUContext* /* context */) { 
                MKLFunc(N, A, B, C);                                                      
            }
                */
    }
}

#[cfg(caffe2_use_mkl)] delegate_simple_binary_function!{f32, Add, vsAdd}
#[cfg(caffe2_use_mkl)] delegate_simple_binary_function!{f64, Add, vdAdd}
#[cfg(caffe2_use_mkl)] delegate_simple_binary_function!{f32, Sub, vsSub}
#[cfg(caffe2_use_mkl)] delegate_simple_binary_function!{f64, Sub, vdSub}
#[cfg(caffe2_use_mkl)] delegate_simple_binary_function!{f32, Mul, vsMul}
#[cfg(caffe2_use_mkl)] delegate_simple_binary_function!{f64, Mul, vdMul}
#[cfg(caffe2_use_mkl)] delegate_simple_binary_function!{f32, Div, vsDiv}
#[cfg(caffe2_use_mkl)] delegate_simple_binary_function!{f64, Div, vdDiv}

#[cfg(caffe2_use_mkl)]
#[macro_export] macro_rules! delegate_axpby {
    ($TAlpha:ty, $TData:ty, $MKLFunc:ident) => {
        /*
        template <>                                                                  
            C10_EXPORT void Axpby<TAlpha, TData, CPUContext>(                            
                const i64 N,                                                    
                const TAlpha alpha,                                                      
                const TData* X,                                                          
                const TAlpha beta,                                                       
                TData* Y,                                                                
                CPUContext* /* context */) {                                             
                MKLFunc(                                                                   
                    N, static_cast<TData>(alpha), X, 1, static_cast<TData>(beta), Y, 1);   
            }                                                                            
        template <>                                                                  
            C10_EXPORT void Axpby<TAlpha, TData, CPUContext>(                            
                const i64 N,                                                    
                const TAlpha* alpha,                                                     
                const TData* X,                                                          
                const TAlpha* beta,                                                      
                TData* Y,                                                                
                CPUContext* /* context */) {                                             
                MKLFunc(                                                                   
                    N, static_cast<TData>(*alpha), X, 1, static_cast<TData>(*beta), Y, 1); 
            }
        */
    }
}

#[cfg(caffe2_use_mkl)] delegate_axpby!{f32, f32, cblas_saxpby}

#[macro_export] macro_rules! delegate_simple_unary_function { 
    ($T:ty, $Func:ident, $EigenFunc:ident) => {
        /*
        #[cfg(not(caffe2_use_mkl))]
        template <>                                                     
            C10_EXPORT void Func<T, CPUContext>(                            
                const int N, const T* X, T* Y, CPUContext* /* context */) { 
                EigenVectorArrayMap<T>(Y, N) =                                
                    ConstEigenVectorArrayMap<T>(X, N).EigenFunc();            
            }
                */
    };

    /*
      | MKL VML alternatives.
      |
      | Depending on whether we are using MKL, we will
      | delegate the Caffe2 math functions that are
      | VML-related to either the VML call or the Eigen
      | implementation. If you are setting the flags
      | (such as AVX) right for your CPU architecture,
      | usually Eigen will deliver a throughput as fast
      | as the VML functions.
      */
    ($T:ident, $Func:ident, $MKLFunc:ident, $($arg:expr),*) => {
        /*
        #[cfg(caffe2_use_mkl)]
        template <>                                                     
            C10_EXPORT void Func<T, CPUContext>(                            
                const int N, const T* X, T* Y, CPUContext* /* context */) { 
                MKLFunc(N, X, Y, ##__VA_ARGS__);                              
            }
                */
    };

    ($T:ty, $Func:ident, $EigenFunc:ident) => {
        /*
        template <>                                                     
            C10_EXPORT void Func<T, CPUContext>(                            
                const int N, const T* X, T* Y, CPUContext* /* context */) { 
                EigenVectorArrayMap<T>(Y, N) =                                
                    ConstEigenVectorArrayMap<T>(X, N).EigenFunc();            
            }
        // Eigen's Tanh implementation is faster than MKL, so use Eigen here.
        */
    }
}

delegate_simple_unary_function!{f32, Tanh, tanh}
delegate_simple_unary_function!{f64, Tanh, tanh}
delegate_simple_unary_function!{i32, Sign, sign}
delegate_simple_unary_function!{i64, Sign, sign}
delegate_simple_unary_function!{f32, Sign, sign}
delegate_simple_unary_function!{f64, Sign, sign}
delegate_simple_unary_function!{i32, Abs, abs}
delegate_simple_unary_function!{i64, Abs, abs}
delegate_simple_unary_function!{i32, Cube, cube}
delegate_simple_unary_function!{i64, Cube, cube}
delegate_simple_unary_function!{f32, Cube, cube}
delegate_simple_unary_function!{f64, Cube, cube}

#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Exp, vmsExp, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Exp, vmdExp, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Log, vsLn}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Log, vdLn}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Log1p, vsLog1p}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Log1p, vdLog1p}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Sin, vsSin}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Sin, vdSin}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Asin, vsAsin}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Asin, vdAsin}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Cos, vsCos}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Cos, vdCos}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Acos, vsAcos}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Acos, vdAcos}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Tan, vsTan}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Tan, vdTan}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Atan, vsAtan}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Atan, vdAtan}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Sinh, vsSinh}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Sinh, vdSinh}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Cosh, vsCosh}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Cosh, vdCosh}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Abs, vsAbs}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Abs, vdAbs}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Sqr, vsSqr}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Sqr, vdSqr}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Sqrt, vsSqrt}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Sqrt, vdSqrt}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Rsqrt, vsInvSqrt}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Rsqrt, vdInvSqrt}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Cbrt, vsCbrt}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Cbrt, vdCbrt}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Inv, vsInv}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Inv, vdInv}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, Erf, vsErf}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, Erf, vdErf}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f32, CdfNorm, vsCdfNorm}
#[cfg(caffe2_use_mkl)] delegate_simple_unary_function!{f64, CdfNorm, vdCdfNorm}

#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f32, Exp, exp}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f64, Exp, exp}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f32, Log, log}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f64, Log, log}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f32, Log1p, log1p}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f64, Log1p, log1p}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f32, Sin, sin}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f64, Sin, sin}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f32, Asin, asin}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f64, Asin, asin}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f32, Cos, cos}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f64, Cos, cos}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f32, Acos, acos}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f64, Acos, acos}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f32, Tan, tan}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f64, Tan, tan}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f32, Atan, atan}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f64, Atan, atan}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f32, Abs, abs}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f64, Abs, abs}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f32, Sqr, square}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f64, Sqr, square}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f32, Sqrt, sqrt}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f64, Sqrt, sqrt}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f32, Rsqrt, rsqrt}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f64, Rsqrt, rsqrt}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f32, Inv, inverse}
#[cfg(not(caffe2_use_mkl))] delegate_simple_unary_function!{f64, Inv, inverse}

#[cfg(not(caffe2_use_mkl))]
#[macro_export] macro_rules! caffe2_specialized_sinh {
    ($T:ty) => {
        /*
        template <>                                                             
            C10_EXPORT void Sinh<T, CPUContext>(                                    
                const int N, const T* X, T* Y, CPUContext* /* context */) {         
                ConstEigenVectorArrayMap<T> X_arr(X, N);                              
                    EigenVectorArrayMap<T>(Y, N) = (X_arr.exp() - (-X_arr).exp()) / T(2); 
            }
        */
    }
}

#[cfg(not(caffe2_use_mkl))] caffe2_specialized_sinh!{f32}
#[cfg(not(caffe2_use_mkl))] caffe2_specialized_sinh!{f64}

#[cfg(not(caffe2_use_mkl))]
#[macro_export] macro_rules! caffe2_specialized_cosh {
    ($T:ty) => {
        /*
        template <>                                                             
            C10_EXPORT void Cosh<T, CPUContext>(                                    
                const int N, const T* X, T* Y, CPUContext* /* context */) {         
                ConstEigenVectorArrayMap<T> X_arr(X, N);                              
                    EigenVectorArrayMap<T>(Y, N) = (X_arr.exp() + (-X_arr).exp()) / T(2); 
            }
        */
    }
}

#[cfg(not(caffe2_use_mkl))] caffe2_specialized_cosh!{f32}
#[cfg(not(caffe2_use_mkl))] caffe2_specialized_cosh!{f64}

#[cfg(not(caffe2_use_mkl))]
#[macro_export] macro_rules! caffe2_specialized_sincos {
    ($T:ty) => {
        /*
        template <>                                                               
            C10_EXPORT void SinCos<T, CPUContext>(                                    
                const int N, const T* X, T* S, T* C, CPUContext* /* context */) {     
                EigenVectorArrayMap<T>(S, N) = ConstEigenVectorArrayMap<T>(X, N).sin(); 
                    EigenVectorArrayMap<T>(C, N) = ConstEigenVectorArrayMap<T>(X, N).cos(); 
            }
        */
    }
}

#[cfg(not(caffe2_use_mkl))] caffe2_specialized_sincos!{f32}
#[cfg(not(caffe2_use_mkl))] caffe2_specialized_sincos!{f64}

#[cfg(not(caffe2_use_mkl))]
#[macro_export] macro_rules! caffe2_specialized_powx {
    ($T:ty) => {
        /*
        template <>                                                                
            C10_EXPORT void Powx<T, CPUContext>(                                       
                const int N, const T* A, const T b, T* Y, CPUContext* /* context */) { 
                EigenVectorArrayMap<T>(Y, N) = ConstEigenVectorArrayMap<T>(A, N).pow(b); 
            }
        */
    }
}

#[cfg(not(caffe2_use_mkl))] caffe2_specialized_powx!{f32}
#[cfg(not(caffe2_use_mkl))] caffe2_specialized_powx!{f64}

#[cfg(not(caffe2_use_mkl))]
#[macro_export] macro_rules! caffe2_specialized_cbrt {
    ($T:ty) => {
        /*
        template <>                                                       
            C10_EXPORT void Cbrt<T, CPUContext>(                              
                const int N, const T* X, T* Y, CPUContext* /* context */) {   
                std::transform(X, X + N, Y, [](const T x) { return cbrt(x); }); 
            }
        */
    }
}

#[cfg(not(caffe2_use_mkl))] caffe2_specialized_cbrt!{f32}
#[cfg(not(caffe2_use_mkl))] caffe2_specialized_cbrt!{f64}

#[cfg(not(caffe2_use_mkl))]
#[macro_export] macro_rules! caffe2_specialized_erf {
    ($T:ty) => {
        /*
        template <>                                                      
            C10_EXPORT void Erf<T, CPUContext>(                              
                const int N, const T* X, T* Y, CPUContext* /* context */) {  
                std::transform(X, X + N, Y, [](const T x) { return erf(x); }); 
            }
        */
    }
}

#[cfg(not(caffe2_use_mkl))] caffe2_specialized_erf!{f32}
#[cfg(not(caffe2_use_mkl))] caffe2_specialized_erf!{f64}

#[cfg(not(caffe2_use_mkl))]
#[macro_export] macro_rules! caffe2_specialized_cdf_norm {
    ($T:ty) => {
        /*
        template <>                                                     
            C10_EXPORT void CdfNorm<T, CPUContext>(                         
                const int N, const T* X, T* Y, CPUContext* /* context */) { 
                std::transform(X, X + N, Y, [](const T x) {                   
                    constexpr T kRsqrt2 = 0.7071067811865475;                   
                        return (T(1) + erf(x * kRsqrt2)) * static_cast<T>(0.5);     
                });                                                           
            }
        */
    }
}

#[cfg(not(caffe2_use_mkl))] caffe2_specialized_cdf_norm!{f32}
#[cfg(not(caffe2_use_mkl))] caffe2_specialized_cdf_norm!{f64}


#[cfg(not(caffe2_use_mkl))]
#[macro_export] macro_rules! caffe2_specialized_axpby {
    ($TAlpha:ty, $TData:ident) => {
        /*
        template <>                                                               
            C10_EXPORT void Axpby<TAlpha, TData, CPUContext>(                         
                const i64 N,                                                 
                const TAlpha alpha,                                                   
                const TData* X,                                                       
                const TAlpha beta,                                                    
                TData* Y,                                                             
                CPUContext* /* context */) {                                          
                EigenVectorArrayMap<TData> Y_arr(Y, N);                                 
                    Y_arr = Y_arr * static_cast<TData>(beta) +                              
                    ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(alpha);  
            }                                                                         
        template <>                                                               
            C10_EXPORT void Axpby<TAlpha, TData, CPUContext>(                         
                const i64 N,                                                 
                const TAlpha* alpha,                                                  
                const TData* X,                                                       
                const TAlpha* beta,                                                   
                TData* Y,                                                             
                CPUContext* /* context */) {                                          
                EigenVectorArrayMap<TData> Y_arr(Y, N);                                 
                    Y_arr = Y_arr * static_cast<TData>(*beta) +                             
                    ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(*alpha); 
            }
        */
    }
}

#[cfg(not(caffe2_use_mkl))] caffe2_specialized_axpby!{f32, f32}

/**
  | BLAS alternatives.
  |
  | Depending on whether we have specified an
  | external BLAS library or not, we will delegate
  | the Caffe math functions that are BLAS-related
  | to either the CBLAS call or the Eigen
  | implementation.
  */
#[cfg(caffe2_use_eigen_for_blas)]
#[macro_export] macro_rules! caffe2_specialized_scale {
    ($TAlpha:ty, $TData:ident) => {
        /*
        template <>                                                                 
            C10_EXPORT void Scale<TAlpha, TData, CPUContext>(                           
                const i64 N,                                                   
                const TAlpha alpha,                                                     
                const TData* X,                                                         
                TData* Y,                                                               
                CPUContext* /* context */) {                                            
                if (X == Y) {                                                             
                    EigenVectorArrayMap<TData>(Y, N) *= static_cast<TData>(alpha);          
                } else {                                                                  
                    EigenVectorArrayMap<TData>(Y, N) =                                      
                        ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(alpha);  
                }                                                                         
            }                                                                           
        template <>                                                                 
            C10_EXPORT void Scale<TAlpha, TData, CPUContext>(                           
                const i64 N,                                                   
                const TAlpha* alpha,                                                    
                const TData* X,                                                         
                TData* Y,                                                               
                CPUContext* /* context */) {                                            
                if (X == Y) {                                                             
                    EigenVectorArrayMap<TData>(Y, N) *= static_cast<TData>(*alpha);         
                } else {                                                                  
                    EigenVectorArrayMap<TData>(Y, N) =                                      
                        ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(*alpha); 
                }                                                                         
            }
        */
    }
}

#[cfg(caffe2_use_eigen_for_blas)] caffe2_specialized_scale!{f32, f32}
#[cfg(caffe2_use_eigen_for_blas)] caffe2_specialized_scale!{f64, f64}
#[cfg(caffe2_use_eigen_for_blas)] caffe2_specialized_scale!{f32, f64}

#[cfg(caffe2_use_eigen_for_blas)]
#[macro_export] macro_rules! caffe2_specialized_axpy {
    ($TAlpha:ty, $TData:ident) => {
        /*
        template <>                                                               
            C10_EXPORT void Axpy<TAlpha, TData, CPUContext>(                          
                const i64 N,                                                 
                const TAlpha alpha,                                                   
                const TData* X,                                                       
                TData* Y,                                                             
                CPUContext* /* context */) {                                          
                EigenVectorArrayMap<TData>(Y, N) +=                                     
                    ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(alpha);  
            }                                                                         
        template <>                                                               
            C10_EXPORT void Axpy<TAlpha, TData, CPUContext>(                          
                const i64 N,                                                 
                const TAlpha* alpha,                                                  
                const TData* X,                                                       
                TData* Y,                                                             
                CPUContext* /* context */) {                                          
                EigenVectorArrayMap<TData>(Y, N) +=                                     
                    ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(*alpha); 
            }
        */
    }
}

#[cfg(caffe2_use_eigen_for_blas)]
caffe2_specialized_axpy!{f32, f32}

#[cfg(all(not(caffe2_use_eigen_for_blas), caffe2_use_mkl))]
#[macro_export] macro_rules! delegate_scale {
    ($TAlpha:ty, $TData:ty, $MKLFunc1:ident, $MKLFunc2:ident) => {
        /*
        template <>                                                        
            C10_EXPORT void Scale<TAlpha, TData, CPUContext>(                  
                const i64 N,                                          
                const TAlpha alpha,                                            
                const TData* X,                                                
                TData* Y,                                                      
                CPUContext* /* context */) {                                   
                const int max_int = i32::max;         
                    int batch = N / max_int;                                         
                    int remainder = N % max_int;                                     
                    i64 offset = 0;                                         
                    for (int i = 0; i < batch; i ++) {                               
                        if (Y == X) {                                                  
                            MKLFunc1(max_int, static_cast<TData>(alpha), Y + offset, 1); 
                        } else {                                                       
                            MKLFunc2(max_int, static_cast<TData>(alpha), X + offset, 1, TData(0), Y + offset, 1);  
                        }                                                              
                        offset += max_int;                                             
                    }                                                                
                if (remainder != 0) {                                            
                    if (Y == X) {                                                  
                        MKLFunc1(remainder, static_cast<TData>(alpha), Y + offset, 1); 
                    } else {                                                       
                        MKLFunc2(remainder, static_cast<TData>(alpha), X + offset, 1, TData(0), Y + offset, 1);  
                    }                                                              
                }                                                                
            }                                                                  
        template <>                                                        
            C10_EXPORT void Scale<TAlpha, TData, CPUContext>(                  
                const i64 N,                                          
                const TAlpha* alpha,                                           
                const TData* X,                                                
                TData* Y,                                                      
                CPUContext* /* context */) {                                   
                const int max_int = i32::max;         
                    int batch = N / max_int;                                         
                    int remainder = N % max_int;                                     
                    i64 offset = 0;                                         
                    for (int i = 0; i < batch; i ++) {                               
                        if (Y == X) {                                                  
                            MKLFunc1(max_int, static_cast<TData>(*alpha), Y + offset, 1); 
                        } else {                                                       
                            MKLFunc2(max_int, static_cast<TData>(*alpha), X + offset, 1, TData(0), Y + offset, 1);  
                        }                                                              
                        offset += max_int;                                             
                    }                                                                
                if (remainder != 0) {                                            
                    if (Y == X) {                                                  
                        MKLFunc1(remainder, static_cast<TData>(*alpha), Y + offset, 1); 
                    } else {                                                       
                        MKLFunc2(remainder, static_cast<TData>(*alpha), X + offset, 1, TData(0), Y + offset, 1); 
                    }                                                              
                }                                                                
            }
        */
    }
}

#[cfg(all(not(caffe2_use_eigen_for_blas), caffe2_use_mkl))] delegate_scale!{f32, f32, cblas_sscal, cblas_saxpby}
#[cfg(all(not(caffe2_use_eigen_for_blas), caffe2_use_mkl))] delegate_scale!{f64, f64, cblas_dscal, cblas_daxpby}
#[cfg(all(not(caffe2_use_eigen_for_blas), caffe2_use_mkl))] delegate_scale!{f32, f64, cblas_dscal, cblas_daxpby}

#[cfg(all(not(caffe2_use_mkl), not(caffe2_use_eigen_for_blas)))]
#[macro_export] macro_rules! delegate_scale {
    ($TAlpha:ty, $TData:ty, $BLASFunc:ident) => {
        /*
        template <>                                                                 
            C10_EXPORT void Scale<TAlpha, TData, CPUContext>(                           
                const i64 N,                                                   
                const TAlpha alpha,                                                     
                const TData* X,                                                         
                TData* Y,                                                               
                CPUContext* /* context */) {                                            
                if (Y == X) {                                                             
                    BLASFunc(N, static_cast<TData>(alpha), Y, 1);                           
                } else {                                                                  
                    EigenVectorArrayMap<TData>(Y, N) =                                      
                        ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(alpha);  
                }                                                                         
            }                                                                           
        template <>                                                                 
            C10_EXPORT void Scale<TAlpha, TData, CPUContext>(                           
                const i64 N,                                                   
                const TAlpha* alpha,                                                    
                const TData* X,                                                         
                TData* Y,                                                               
                CPUContext* /* context */) {                                            
                if (Y == X) {                                                             
                    BLASFunc(N, static_cast<TData>(*alpha), Y, 1);                          
                } else {                                                                  
                    EigenVectorArrayMap<TData>(Y, N) =                                      
                        ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(*alpha); 
                }                                                                         
            }
        */
    }
}

#[cfg(all(not(caffe2_use_mkl), not(caffe2_use_eigen_for_blas)))] delegate_scale!{f32, f32, cblas_sscal}
#[cfg(all(not(caffe2_use_mkl), not(caffe2_use_eigen_for_blas)))] delegate_scale!{f64, f64, cblas_dscal}
#[cfg(all(not(caffe2_use_mkl), not(caffe2_use_eigen_for_blas)))] delegate_scale!{f32, f64, cblas_dscal}

#[cfg(not(caffe2_use_eigen_for_blas))]
#[macro_export] macro_rules! delegate_axpy {
    ($TAlpha:ty, $TData:ty, $BLASFunc:ident) => {
        /*
        template <>                                            
            C10_EXPORT void Axpy<TAlpha, TData, CPUContext>(       
                const i64 N,                              
                const TAlpha alpha,                                
                const TData* X,                                    
                TData* Y,                                          
                CPUContext* /* context */) {                       
                BLASFunc(N, static_cast<TData>(alpha), X, 1, Y, 1);  
            }                                                      
        template <>                                            
            C10_EXPORT void Axpy<TAlpha, TData, CPUContext>(       
                const i64 N,                              
                const TAlpha* alpha,                               
                const TData* X,                                    
                TData* Y,                                          
                CPUContext* /* context */) {                       
                BLASFunc(N, static_cast<TData>(*alpha), X, 1, Y, 1); 
            }
        */
    }
}

#[cfg(not(caffe2_use_eigen_for_blas))] delegate_axpy!{f32, f32, cblas_saxpy}

/**
  | Common math functions being used in Caffe that
  | do not have a BLAS or MKL equivalent. For all
  | these functions, we will simply implement them
  | either via Eigen or via custom code.
  */
#[macro_export] macro_rules! caffe2_specialized_set {
    ($T:ty) => {
        /*
        template <>                                                                 
            C10_EXPORT void Set<T, CPUContext>(                                         
                const i64 N, const T alpha, T* Y, CPUContext* /* context */) { 
                if (N == 0) {                                                             
                    return;                                                                 
                }                                                                         
                if (alpha == T(0)) {                                                      
                    std::memset(Y, 0, N * sizeof(T));                                       
                } else {                                                                  
                    EigenVectorArrayMap<T>(Y, N).setConstant(alpha);                        
                }                                                                         
            }
        */
    }
}

caffe2_specialized_set!{f32}
caffe2_specialized_set!{f64}
caffe2_specialized_set!{i32}
caffe2_specialized_set!{i8}
caffe2_specialized_set!{i16}
caffe2_specialized_set!{i64}
caffe2_specialized_set!{bool}
caffe2_specialized_set!{char}
caffe2_specialized_set!{u8}
caffe2_specialized_set!{u16}


#[macro_export] macro_rules! caffe2_specialized_neg {
    ($T:ty) => {
        /*
        template <>                                                          
            C10_EXPORT void Neg<T, CPUContext>(                                  
                const int N, const T* X, T* Y, CPUContext* /* context */) {      
                EigenVectorArrayMap<T>(Y, N) = -ConstEigenVectorArrayMap<T>(X, N); 
            }
        */
    }
}

caffe2_specialized_neg!{i32}
caffe2_specialized_neg!{i64}
caffe2_specialized_neg!{f32}
caffe2_specialized_neg!{f64}

#[macro_export] macro_rules! caffe2_specialized_scale {
    ($TAlpha:ty, $TData:ty) => {
        /*
        template <>                                                                 
            C10_EXPORT void Scale<TAlpha, TData, CPUContext>(                           
                const i64 N,                                                   
                const TAlpha alpha,                                                     
                const TData* X,                                                         
                TData* Y,                                                               
                CPUContext* /* context */) {                                            
                if (X == Y) {                                                             
                    EigenVectorArrayMap<TData>(Y, N) *= static_cast<TData>(alpha);          
                } else {                                                                  
                    EigenVectorArrayMap<TData>(Y, N) =                                      
                        ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(alpha);  
                }                                                                         
            }                                                                           
        template <>                                                                 
            C10_EXPORT void Scale<TAlpha, TData, CPUContext>(                           
                const i64 N,                                                   
                const TAlpha* alpha,                                                    
                const TData* X,                                                         
                TData* Y,                                                               
                CPUContext* /* context */) {                                            
                if (X == Y) {                                                             
                    EigenVectorArrayMap<TData>(Y, N) *= static_cast<TData>(*alpha);         
                } else {                                                                  
                    EigenVectorArrayMap<TData>(Y, N) =                                      
                        ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(*alpha); 
                }                                                                         
            }
        */
    }
}

caffe2_specialized_scale!{i32, i32}
caffe2_specialized_scale!{i64, i64}

#[macro_export] macro_rules! delegate_simple_binary_function_by_eigen_operator {
    ($T:ty, $Func:ident, $EigenOp:tt) => {
        /*
        template <>                                                                 
            C10_EXPORT void Func<T, CPUContext>(                                        
                const int N, const T* A, const T* B, T* C, CPUContext* /* context */) { 
                EigenVectorMap<T>(C, N) = ConstEigenVectorArrayMap<T>(A, N)               
                    EigenOp ConstEigenVectorArrayMap<T>(B, N);                            
            }
        */

        /*
        #[cfg(not(caffe2_use_mkl))]
        template <>                                                                 
            C10_EXPORT void Func<T, CPUContext>(                                        
                const int N, const T* A, const T* B, T* C, CPUContext* /* context */) { 
                EigenVectorMap<T>(C, N) = ConstEigenVectorArrayMap<T>(A, N)               
                    EigenOp ConstEigenVectorArrayMap<T>(B, N);                            
            }
        */
    };
}

delegate_simple_binary_function_by_eigen_operator!{i32, Add, +}
delegate_simple_binary_function_by_eigen_operator!{i64, Add, +}
delegate_simple_binary_function_by_eigen_operator!{i32, Sub, -}
delegate_simple_binary_function_by_eigen_operator!{i64, Sub, -}
delegate_simple_binary_function_by_eigen_operator!{i32, Mul, *}
delegate_simple_binary_function_by_eigen_operator!{i64, Mul, *}
delegate_simple_binary_function_by_eigen_operator!{i32, Div, /}
delegate_simple_binary_function_by_eigen_operator!{i64, Div, /}

#[cfg(not(caffe2_use_mkl))] delegate_simple_binary_function_by_eigen_operator!{f32, Add, +}
#[cfg(not(caffe2_use_mkl))] delegate_simple_binary_function_by_eigen_operator!{f64, Add, +}
#[cfg(not(caffe2_use_mkl))] delegate_simple_binary_function_by_eigen_operator!{f32, Sub, -}
#[cfg(not(caffe2_use_mkl))] delegate_simple_binary_function_by_eigen_operator!{f64, Sub, -}
#[cfg(not(caffe2_use_mkl))] delegate_simple_binary_function_by_eigen_operator!{f32, Mul, *}
#[cfg(not(caffe2_use_mkl))] delegate_simple_binary_function_by_eigen_operator!{f64, Mul, *}
#[cfg(not(caffe2_use_mkl))] delegate_simple_binary_function_by_eigen_operator!{f32, Div, /}
#[cfg(not(caffe2_use_mkl))] delegate_simple_binary_function_by_eigen_operator!{f64, Div, /}

#[macro_export] macro_rules! delegate_simple_binary_function_by_eigen_function {
    ($T:ty, $Func:ident, $EigenFunc:ident) => {
        /*
        template <>                                                                 
            C10_EXPORT void Func<T, CPUContext>(                                        
                const int N, const T* A, const T* B, T* C, CPUContext* /* context */) { 
                EigenVectorMap<T>(C, N) = ConstEigenVectorArrayMap<T>(A, N).EigenFunc(    
                    ConstEigenVectorArrayMap<T>(B, N));                                   
            }
        */
    }
}

delegate_simple_binary_function_by_eigen_function!{i32, Min, min}
delegate_simple_binary_function_by_eigen_function!{i64, Min, min}
delegate_simple_binary_function_by_eigen_function!{f32, Min, min}
delegate_simple_binary_function_by_eigen_function!{f64, Min, min}
delegate_simple_binary_function_by_eigen_function!{i32, Max, max}
delegate_simple_binary_function_by_eigen_function!{i64, Max, max}
delegate_simple_binary_function_by_eigen_function!{f32, Max, max}
delegate_simple_binary_function_by_eigen_function!{f64, Max, max}

#[macro_export] macro_rules! delegate_simple_binary_function_by_std_function {
    () => {

    };
    ($T:ty, $Func:ident, $StdFunc:ident) => {
        /*
        template <>                                                                 
            C10_EXPORT void Func<T, CPUContext>(                                        
                const int N, const T* A, const T* B, T* C, CPUContext* /* context */) { 
                std::transform(A, A + N, B, C, StdFunc);                                  
            }
        */
    }
}

delegate_simple_binary_function_by_std_function!{
    /*
    bool,
    And,
    std::logical_and<bool>()
    */
}

delegate_simple_binary_function_by_std_function!{
    /*
    bool,
    Or,
    std::logical_or<bool>()
    */
}

delegate_simple_binary_function_by_std_function!{
    /*
    bool, 
    Xor, 
    std::bit_xor<bool>()
    */
}

delegate_simple_binary_function_by_std_function!{
    /*
    bool,
    BitwiseAnd,
    std::bit_and<bool>()
    */
}

delegate_simple_binary_function_by_std_function!{
    /*
    i32,
    BitwiseAnd,
    std::bit_and<i32>()
    */
}

delegate_simple_binary_function_by_std_function!{
    /*
    i64,
    BitwiseAnd,
    std::bit_and<i64>()
    */
}

delegate_simple_binary_function_by_std_function!{
    /*
    bool,
    BitwiseOr,
    std::bit_or<bool>()
    */
}

delegate_simple_binary_function_by_std_function!{
    /*
    i32,
    BitwiseOr,
    std::bit_or<i32>()
    */
}

delegate_simple_binary_function_by_std_function!{
    /*
    i64,
    BitwiseOr,
    std::bit_or<i64>()
    */
}

delegate_simple_binary_function_by_std_function!{
    /*
    bool,
    BitwiseXor,
    std::bit_xor<bool>()
    */
}

delegate_simple_binary_function_by_std_function!{
    /*
    i32,
    BitwiseXor,
    std::bit_xor<i32>()
    */
}

delegate_simple_binary_function_by_std_function!{
    /*
    i64,
    BitwiseXor,
    std::bit_xor<i64>()
    */
}

#[macro_export] macro_rules! delegate_simple_compare_function_by_eigen_operator {
    ($T:ty, $Func:ident, $EigenOp:tt) => {
        /*
        template <>                                                                
            C10_EXPORT void Func<T, CPUContext>(                                       
                const int N,                                                           
                const T* A,                                                            
                const T* B,                                                            
                bool* C,                                                               
                CPUContext* /* context */) {                                           
                EigenVectorArrayMap<bool>(C, N) = ConstEigenVectorArrayMap<T>(A, N)      
                    EigenOp ConstEigenVectorArrayMap<T>(B, N);                           
            }
        */
    }
}

delegate_simple_compare_function_by_eigen_operator!{bool, EQ, ==}
delegate_simple_compare_function_by_eigen_operator!{i32,  EQ, ==}
delegate_simple_compare_function_by_eigen_operator!{i64,  EQ, ==}
delegate_simple_compare_function_by_eigen_operator!{f32,  EQ, ==}
delegate_simple_compare_function_by_eigen_operator!{f64,  EQ, ==}
delegate_simple_compare_function_by_eigen_operator!{bool, NE, !=}
delegate_simple_compare_function_by_eigen_operator!{i32,  NE, !=}
delegate_simple_compare_function_by_eigen_operator!{i64,  NE, !=}
delegate_simple_compare_function_by_eigen_operator!{f32,  NE, !=}
delegate_simple_compare_function_by_eigen_operator!{f64,  NE, !=}
delegate_simple_compare_function_by_eigen_operator!{bool, LT, <}
delegate_simple_compare_function_by_eigen_operator!{i32,  LT, <}
delegate_simple_compare_function_by_eigen_operator!{i64,  LT, <}
delegate_simple_compare_function_by_eigen_operator!{f32,  LT, <}
delegate_simple_compare_function_by_eigen_operator!{f64,  LT, <}
delegate_simple_compare_function_by_eigen_operator!{bool, LE, <=}
delegate_simple_compare_function_by_eigen_operator!{i32,  LE, <=}
delegate_simple_compare_function_by_eigen_operator!{i64,  LE, <=}
delegate_simple_compare_function_by_eigen_operator!{f32,  LE, <=}
delegate_simple_compare_function_by_eigen_operator!{f64,  LE, <=}
delegate_simple_compare_function_by_eigen_operator!{bool, GT, >}
delegate_simple_compare_function_by_eigen_operator!{i32,  GT, >}
delegate_simple_compare_function_by_eigen_operator!{i64,  GT, >}
delegate_simple_compare_function_by_eigen_operator!{f32,  GT, >}
delegate_simple_compare_function_by_eigen_operator!{f64,  GT, >}
delegate_simple_compare_function_by_eigen_operator!{bool, GE, >=}
delegate_simple_compare_function_by_eigen_operator!{i32,  GE, >=}
delegate_simple_compare_function_by_eigen_operator!{i64,  GE, >=}
delegate_simple_compare_function_by_eigen_operator!{f32,  GE, >=}
delegate_simple_compare_function_by_eigen_operator!{f64,  GE, >=}

