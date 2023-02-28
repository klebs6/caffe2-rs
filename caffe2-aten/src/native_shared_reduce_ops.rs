/*!
  | Please note that this file is used across
  | both CPU and GPU.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/SharedReduceOps.h]

#[cfg(not(any(feature = "cudacc", feature = "hipcc")))]
#[macro_export] macro_rules! device_sqrt {
    () => {
        /*
                sqrt
        */
    }
}

#[cfg(any(feature = "cudacc", feature = "hipcc"))]
#[cfg(target_arch = "cuda")]
#[inline] pub fn max_propagate_nan<Scalar>(
        a: Scalar,
        b: Scalar) -> Scalar {

    todo!();
        /*
            #if defined(__HIPCC__)
      // TODO: remove this special case for HIP when issue is fixed:
      //       https://github.com/ROCm-Developer-Tools/HIP/issues/2209
      Scalar max = _isnan(a) ? a : (_isnan(b) ? b : max(a, b));
    #else
      Scalar max = _isnan(b) ? b : max(a, b);
    #endif
      return max;
        */
}

#[cfg(any(feature = "cudacc", feature = "hipcc"))]
#[cfg(target_arch = "cuda")]
#[inline] pub fn min_propagate_nan<Scalar>(
        a: Scalar,
        b: Scalar) -> Scalar {

    todo!();
        /*
            #if defined(__HIPCC__)
      // TODO: remove this special case for HIP when issue is fixed:
      //       https://github.com/ROCm-Developer-Tools/HIP/issues/2209
      Scalar min = _isnan(a) ? a : (_isnan(b) ? b : min(a, b));
    #else
      Scalar min = _isnan(b) ? b : min(a, b);
    #endif
      return min;
        */
}

#[cfg(any(feature = "cudacc", feature = "hipcc"))]
#[macro_export] macro_rules! max {
    ($X:ident, $Y:ident) => {
        /*
                max_propagate_nan(X,Y)
        */
    }
}

#[cfg(any(feature = "cudacc", feature = "hipcc"))]
#[macro_export] macro_rules! min {
    ($X:ident, $Y:ident) => {
        /*
                min_propagate_nan(X,Y)
        */
    }
}

#[cfg(not(any(feature = "cudacc", feature = "hipcc")))]
#[macro_export] macro_rules! max {
    ($X:ident, $Y:ident) => {
        /*
                max_impl(X,Y)
        */
    }
}

#[cfg(not(any(feature = "cudacc", feature = "hipcc")))]
#[macro_export] macro_rules! min {
    ($X:ident, $Y:ident) => {
        /*
                min_impl(X,Y)
        */
    }
}

// ROCM hcc doesn't work well with using  in
// kernel functions
//
lazy_static!{
    /*
    #if defined(__CUDA_ARCH__)

    #define compat_pow compat::pow
    #elif defined(__HIPCC__)

    #define compat_pow hip::compat::pow
    #else
    #define compat_pow pow
    #endif
    */
}

pub struct WelfordData<Scalar,Index,combine_t> {
    mean: Scalar,
    m2:   Scalar,
    n:    Index,
    nf:   Combine,
}

impl Default for WelfordData {
    
    fn default() -> Self {
        todo!();
        /*
        : mean(0),
        : m2(0),
        : n(0),
        : nf(0),

        
        */
    }
}

impl WelfordData<Scalar,Index,combine_t> {
    
    pub fn new(
        mean: Scalar,
        m2:   Scalar,
        n:    Index,
        nf:   Combine) -> Self {
    
        todo!();
        /*
        : mean(mean),
        : m2(m2),
        : n(n),
        : nf(nf),

        
        */
    }
}

pub struct WelfordOps<Scalar,AccScalar,Index,combine_t,res_t> {
    correction: Index,
    take_sqrt:  bool,
}

pub mod welford_ops {
    pub type Acc = WelfordData<AccScalar,Index,Combine>;
}

impl WelfordOps<Scalar,AccScalar,Index,combine_t,res_t> {

    #[cfg(target_arch = "cuda")]
    #[inline] pub fn reduce(&self, 
        acc:  Acc,
        data: Scalar,
        idx:  Index) -> Acc {
        
        todo!();
        /*
            AccScalar delta = data - acc.mean;
        // using acc.nf(combine_t) here, as acc.n(Index) would still be converted
        // accumulation in reduce is done through index_T
        AccScalar new_mean = acc.mean + delta / (acc.nf + 1);
        AccScalar new_delta = data - new_mean;
        return {
          new_mean,
          acc.m2 + delta * new_delta,
          acc.n + 1,
          combine_t(acc.n + 1), // accumulate for combine_t uses Index
        };
        */
    }

    #[cfg(target_arch = "cuda")]
    #[inline] pub fn combine(&self, a: Acc, b: Acc) -> Acc {
        
        todo!();
        /*
            if (a.nf == 0) {
          return b;
        }
        if (b.nf == 0) {
          return a;
        }
        AccScalar delta = b.mean - a.mean;
        combine_t new_count = a.nf + b.nf;
        AccScalar nb_over_n = b.nf / new_count;
        return {
          a.mean + delta * nb_over_n,
          a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
          // setting acc.n as -1 since acc.n might not be able to represent the count
          // correctly within its range, setting it to -1 to avoid confusion
          -1,
          new_count
        };
        */
    }

    #[cfg(target_arch = "cuda")]
    #[__ubsan_ignore_float_divide_by_zero__]
    #[inline] pub fn project(&self, acc: Acc) -> Res {
        
        todo!();
        /*
            const auto mean = static_cast<Scalar>(acc.mean);
        const combine_t divisor = acc.nf > correction ? acc.nf - correction : 0;
        const auto var = acc.m2 / divisor;
        res_t results(take_sqrt ? device_sqrt(var) : var, mean);
        return results;
        */
    }

    #[cfg(target_arch = "cuda")]
    pub fn translate_idx(
        acc:      Acc,
        base_idx: i64) -> Acc {
        
        todo!();
        /*
            return acc;
        */
    }

    #[cfg(any(feature = "cudacc", feature = "hipcc"))]
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn warp_shfl_down(&self, 
        acc:    Acc,
        offset: i32) -> Acc {
        
        todo!();
        /*
            return {
          WARP_SHFL_DOWN(acc.mean, offset)
          , WARP_SHFL_DOWN(acc.m2, offset)
          , WARP_SHFL_DOWN(acc.n, offset)
          , WARP_SHFL_DOWN(acc.nf, offset)
        };
        */
    }
    
    pub fn new(
        correction: Index,
        take_sqrt:  bool) -> Self {
    
        todo!();
        /*
        : correction(correction),
        : take_sqrt(take_sqrt),

        
        */
    }
}

pub struct MeanOps<Acc,factor_t> {
    factor: Factor,
}

impl MeanOps<Acc,factor_t> {

    #[cfg(target_arch = "cuda")]
    #[inline] pub fn reduce(&self, 
        a:   Acc,
        b:   Acc,
        idx: i64) -> Acc {
        
        todo!();
        /*
            return combine(a, b);
        */
    }

    #[cfg(target_arch = "cuda")] 
    #[inline] pub fn combine(&self, a: Acc, b: Acc) -> Acc {
        
        todo!();
        /*
            return a + b;
        */
    }

    #[cfg(target_arch = "cuda")]
    #[inline] pub fn project(&self, a: Acc) -> Acc {
        
        todo!();
        /*
            return a * factor;
        */
    }

    #[cfg(target_arch = "cuda")]
    pub fn translate_idx(
        acc:      Acc,
        base_idx: i64) -> Acc {
        
        todo!();
        /*
            return acc;
        */
    }

    #[cfg(any(feature = "cudacc", feature = "hipcc"))]
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn warp_shfl_down(&self, 
        data:   Acc,
        offset: i32) -> Acc {
        
        todo!();
        /*
            return WARP_SHFL_DOWN(data, offset);
        */
    }
    
    pub fn new(factor: Factor) -> Self {
    
        todo!();
        /*
        : factor(factor),

        
        */
    }
}

/**
  | This accumulator template is used to calculate
  | the minimum absolute value of a set of numbers.
  |
  | `Scalar` is the type of the input and `Acc` is
  | the type of the accumulated value. These types
  | differ for complex number input support.
  |
  */
pub struct AbsMinOps<Scalar,Acc = Scalar> {

}

impl AbsMinOps<Scalar,Acc> {

    #[cfg(target_arch = "cuda")]
    #[inline] pub fn reduce(&self, 
        acc:  Acc,
        data: Scalar,
        idx:  i64) -> Acc {
        
        todo!();
        /*
            return MIN(acc, static_cast<Acc>(abs(data)));
        */
    }

    #[cfg(target_arch = "cuda")]
    #[inline] pub fn combine(&self, a: Acc, b: Acc) -> Acc {
        
        todo!();
        /*
            return MIN(a, b);
        */
    }

    #[cfg(target_arch = "cuda")]
    #[inline] pub fn project(&self, a: Acc) -> Acc {
        
        todo!();
        /*
            return a;
        */
    }

    #[cfg(target_arch = "cuda")]
    pub fn translate_idx(
        acc:      Acc,
        base_idx: i64) -> Acc {
        
        todo!();
        /*
            return acc;
        */
    }

    #[cfg(any(feature = "cudacc", feature = "hipcc"))]
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn warp_shfl_down(&self, 
        acc:    Acc,
        offset: i32) -> Acc {
        
        todo!();
        /*
            return WARP_SHFL_DOWN(acc, offset);
        */
    }
}

/**
  | This accumulator template is used to calculate
  | the maximum absolute value of a set of numbers.
  |
  | `Scalar` is the type of the input and `Acc` is
  | the type of the accumulated value. These types
  | differ for complex number input support.
  |
  */
pub struct AbsMaxOps<Scalar,Acc = Scalar> {

}

impl AbsMaxOps<Scalar,Acc> {

    #[cfg(target_arch = "cuda")]
    #[inline] pub fn reduce(&self, 
        acc:  Acc,
        data: Scalar,
        idx:  i64) -> Acc {
        
        todo!();
        /*
            return MAX(acc, static_cast<Acc>(abs(data)));
        */
    }

    #[cfg(target_arch = "cuda")]
    #[inline] pub fn combine(&self, a: Acc, b: Acc) -> Acc {
        
        todo!();
        /*
            return MAX(a, b);
        */
    }

    #[cfg(target_arch = "cuda")]
    #[inline] pub fn project(&self, a: Acc) -> Acc {
        
        todo!();
        /*
            return a;
        */
    }

    #[cfg(target_arch = "cuda")]
    pub fn translate_idx(
        acc:      Acc,
        base_idx: i64) -> Acc {
        
        todo!();
        /*
            return acc;
        */
    }

    #[cfg(any(feature = "cudacc", feature = "hipcc"))]
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn warp_shfl_down(&self, 
        acc:    Acc,
        offset: i32) -> Acc {
        
        todo!();
        /*
            return WARP_SHFL_DOWN(acc, offset);
        */
    }
}

/**
  | This accumulator template is used to calculate
  | the norm of the absolute value of a set of
  | numbers.
  |
  | `Scalar` is the type of the input and `Acc` is
  | the type of the accumulated value. These types
  | differ for complex number input support.
  */
pub struct NormOps<Scalar,Acc = Scalar> {
    norm: Acc,
}

impl NormOps<Scalar,Acc> {

    #[cfg(target_arch = "cuda")]
    #[inline] pub fn reduce(&self, 
        acc:  Acc,
        data: Scalar,
        idx:  i64) -> Acc {
        
        todo!();
        /*
            return acc + compat_pow(static_cast<Acc>(abs(data)), norm_);
        */
    }

    #[cfg(target_arch = "cuda")]
    #[inline] pub fn combine(&self, a: Acc, b: Acc) -> Acc {
        
        todo!();
        /*
            return a + b;
        */
    }

    #[cfg(target_arch = "cuda")]
    #[inline] pub fn project(&self, a: Acc) -> Acc {
        
        todo!();
        /*
            return compat_pow(a, static_cast<Acc>(1.0) / norm_);
        */
    }

    #[cfg(target_arch = "cuda")]
    pub fn translate_idx(
        acc:      Acc,
        base_idx: i64) -> Acc {
        
        todo!();
        /*
            return acc;
        */
    }

    #[cfg(any(feature = "cudacc", feature = "hipcc"))]
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn warp_shfl_down(&self, 
        acc:    Acc,
        offset: i32) -> Acc {
        
        todo!();
        /*
            return WARP_SHFL_DOWN(acc, offset);
        */
    }
    
    pub fn new(norm: Acc) -> Self {
    
        todo!();
        /*
        : norm(norm_),

        
        */
    }
}

/**
  | This accumulator template is used to calculate
  | the order zero norm of the absolute value of
  | a set of numbers.
  |
  | `Scalar` is the type of the input and `Acc` is
  | the type of the accumulated value. These types
  | differ for complex number input support.
  |
  */
pub struct NormZeroOps<Scalar,Acc = Scalar> {

}
impl NormZeroOps<Scalar,Acc> {

    #[cfg(target_arch = "cuda")]
    #[inline] pub fn reduce(&self, 
        acc:  Acc,
        data: Scalar,
        idx:  i64) -> Acc {
        
        todo!();
        /*
            return acc + (data == static_cast<Scalar>(0) ? static_cast<Acc>(0) : static_cast<Acc>(1));
        */
    }
    
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn combine(&self, a: Acc, b: Acc) -> Acc {
        
        todo!();
        /*
            return a + b;
        */
    }
    
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn project(&self, a: Acc) -> Acc {
        
        todo!();
        /*
            return a;
        */
    }
    
    #[cfg(target_arch = "cuda")]
    pub fn translate_idx(
        acc:      Acc,
        base_idx: i64) -> Acc {
        
        todo!();
        /*
            return acc;
        */
    }

    #[cfg(any(feature = "cudacc", feature = "hipcc"))]
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn warp_shfl_down(&self, 
        acc:    Acc,
        offset: i32) -> Acc {
        
        todo!();
        /*
            return WARP_SHFL_DOWN(acc, offset);
        */
    }
}

/**
  | This accumulator template is used to calculate
  | the order one norm of the absolute value of
  | a set of numbers.
  |
  | `Scalar` is the type of the input and `Acc` is
  | the type of the accumulated value. These types
  | differ for complex number input support.
  |
  */
pub struct NormOneOps<Scalar,Acc = Scalar> {

}

impl NormOneOps<Scalar,Acc> {

    #[cfg(target_arch = "cuda")]
    #[inline] pub fn reduce(&self, 
        acc:  Acc,
        data: Scalar,
        idx:  i64) -> Acc {
        
        todo!();
        /*
            return acc + static_cast<Acc>(abs(data));
        */
    }
    
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn combine(&self, a: Acc, b: Acc) -> Acc {
        
        todo!();
        /*
            return a + b;
        */
    }
    
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn project(&self, a: Acc) -> Acc {
        
        todo!();
        /*
            return a;
        */
    }
    
    #[cfg(target_arch = "cuda")]
    pub fn translate_idx(
        acc:      Acc,
        base_idx: i64) -> Acc {
        
        todo!();
        /*
            return acc;
        */
    }

    #[cfg(any(feature = "cudacc", feature = "hipcc"))]
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn warp_shfl_down(&self, 
        acc:    Acc,
        offset: i32) -> Acc {
        
        todo!();
        /*
            return WARP_SHFL_DOWN(acc, offset);
        */
    }
}


pub struct AbsSwitch<Acc> {

}

#[cfg(target_arch = "cuda")]
#[inline] pub fn abs_if_complex<Scalar,Acc>(
        data: Scalar,
        s:    AbsSwitch<Acc>) -> Acc {
    
    todo!();
        /*
            return static_cast<Acc>(data);
        */
}

#[cfg(target_arch = "cuda")]
#[inline] pub fn abs_if_complex<Scalar,Acc>(
        data: Complex<Scalar>,
        s:    AbsSwitch<Acc>) -> Acc {
    
    todo!();
        /*
            return static_cast<Acc>(abs(data));
        */
}

#[cfg(target_arch = "cuda")]
#[inline] pub fn abs_if_complex<Scalar,Acc>(
        data: Complex<Scalar>,
        s:    AbsSwitch<Acc>) -> Acc {
    
    todo!();
        /*
            return static_cast<Acc>(abs(data));
        */
}

/**
  | This accumulator template is used to calculate
  | the order two norm of the absolute value of
  | a set of numbers.
  |
  | `Scalar` is the type of the input and `Acc` is
  | the type of the accumulated value. These types
  | differ for complex number input support.
  |
  */
pub struct NormTwoOps<Scalar,Acc = Scalar> {

}

impl NormTwoOps<Scalar,Acc> {
    
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn reduce(&self, 
        acc:  Acc,
        data: Scalar,
        idx:  i64) -> Acc {
        
        todo!();
        /*
            Acc data_ = abs_if_complex(data, AbsSwitch<Acc>());
        return acc + data_ * data_;
        */
    }
    
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn combine(&self, a: Acc, b: Acc) -> Acc {
        
        todo!();
        /*
            return a + b;
        */
    }
    
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn project(&self, a: Acc) -> Acc {
        
        todo!();
        /*
            return device_sqrt(a);
        */
    }
    
    #[cfg(target_arch = "cuda")]
    pub fn translate_idx(
        acc:      Acc,
        base_idx: i64) -> Acc {
        
        todo!();
        /*
            return acc;
        */
    }

    #[cfg(any(feature = "cudacc", feature = "hipcc"))]
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn warp_shfl_down(&self, 
        acc:    Acc,
        offset: i32) -> Acc {
        
        todo!();
        /*
            return WARP_SHFL_DOWN(acc, offset);
        */
    }
}

pub struct NanSumOps<Acc,Data> {

}

impl NanSumOps<Acc,Data> {
    
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn reduce(&self, 
        a:   Acc,
        b:   Data,
        idx: i64) -> Acc {
        
        todo!();
        /*
            return a + (_isnan(b) ? Acc{0.} : Acc{b});
        */
    }
    
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn combine(&self, a: Acc, b: Acc) -> Acc {
        
        todo!();
        /*
            return  a + b;
        */
    }
    
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn project(&self, a: Acc) -> Data {
        
        todo!();
        /*
            return Data{a};
        */
    }
    
    #[cfg(target_arch = "cuda")]
    pub fn translate_idx(
        acc:      Acc,
        base_idx: i64) -> Acc {
        
        todo!();
        /*
            return acc;
        */
    }

    #[cfg(any(feature = "cudacc", feature = "hipcc"))]
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn warp_shfl_down(&self, 
        data:   Acc,
        offset: i32) -> Acc {
        
        todo!();
        /*
            return WARP_SHFL_DOWN(data, offset);
        */
    }
}

pub struct LessOrNan<Scalar> {

}

impl LessOrNan<Scalar> {
    
    #[cfg(target_arch = "cuda")]
    pub fn invoke(&self, 
        a:     Scalar,
        b:     Scalar,
        idx_a: i64,
        idx_b: i64) -> bool {
        
        todo!();
        /*
            // If (a == b), then choose the one with lower idx, else min(a, b)
        if (_isnan(a)) {
          if (_isnan(b)) {
            return idx_a < idx_b;
          }
          return true;
        }
        return (a == b) ? idx_a < idx_b : (a < b);
        */
    }
}

pub struct GreaterOrNan<Scalar> {

}

impl GreaterOrNan<Scalar> {
    
    #[cfg(target_arch = "cuda")]
    pub fn invoke(&self, 
        a:     Scalar,
        b:     Scalar,
        idx_a: i64,
        idx_b: i64) -> bool {
        
        todo!();
        /*
            // If (a == b), then choose the one with lower idx, else max(a, b)
        if (_isnan(a)) {
          if (_isnan(b)) {
            return idx_a < idx_b;
          }
          return true;
        }
        return (a == b) ? idx_a < idx_b : (a > b);
        */
    }
}


pub struct MinMaxReductionOps<comp_t> {

}

pub mod minmax_reduction_ops {

    use super::*;

    lazy_static!{
        /*
        using Scalar = typename binary_function_traits<comp_t>::arg1_t;
          using Index = i64;
          using arg_t = pair<Scalar, Index>;
        */
    }
}

impl MinMaxReductionOps<comp_t> {
    
    #[cfg(target_arch = "cuda")]
    pub fn project(arg: Arg) -> Arg {
        
        todo!();
        /*
            return arg;
        */
    }
    
    #[cfg(target_arch = "cuda")]
    pub fn reduce(
        arg: Arg,
        val: Scalar,
        idx: i64) -> Arg {
        
        todo!();
        /*
            return comp_t{}(arg.first, val, arg.second, idx) ? arg : arg_t(val, idx);
        */
    }
    
    #[cfg(target_arch = "cuda")]
    pub fn combine(a: Arg, b: Arg) -> Arg {
        
        todo!();
        /*
            return comp_t{}(a.first, b.first, a.second, b.second) ? a : b;
        */
    }
    
    #[cfg(target_arch = "cuda")]
    pub fn translate_idx(
        a:        Arg,
        base_idx: i64) -> Arg {
        
        todo!();
        /*
            return {a.first, a.second + base_idx};
        */
    }

    #[cfg(any(feature = "cudacc", feature = "hipcc"))]
    #[cfg(target_arch = "cuda")]
    pub fn warp_shfl_down(
        arg:    Arg,
        offset: i32) -> Arg {
        
        todo!();
        /*
            return arg_t(WARP_SHFL_DOWN(arg.first, offset),
                     WARP_SHFL_DOWN(arg.second, offset));
        */
    }
}

pub struct ArgReductionOps<comp_t> {
    base: MinMaxReductionOps<Comp>,
}

pub mod arg_reduction_ops {

    use super::*;

    lazy_static!{
        /*
        using typename MinMaxReductionOps<comp_t>::Scalar;
          using typename MinMaxReductionOps<comp_t>::Index;
          using typename MinMaxReductionOps<comp_t>::arg_t;
        */
    }
}

impl ArgReductionOps<comp_t> {
    
    #[cfg(target_arch = "cuda")]
    pub fn project(arg: Arg) -> Index {
        
        todo!();
        /*
            return arg.second;
        */
    }
}

pub struct ArgMaxOps<Scalar> {
    base: ArgReductionOps<GreaterOrNan<Scalar>>,
}

pub struct ArgMinOps<Scalar> {
    base: ArgReductionOps<LessOrNan<Scalar>>,
}

pub struct MinOps<Scalar> {
    base: MinMaxReductionOps<LessOrNan<Scalar>>,
}

pub struct MaxOps<Scalar> {
    base: MinMaxReductionOps<GreaterOrNan<Scalar>>,
}

pub struct MinMaxOps<Scalar,AccScalar,Index> {

}

pub mod minmax_ops {

    use super::*;

    pub type Acc = (AccScalar,AccScalar);
}

impl MinMaxOps<Scalar,AccScalar,Index> {
    
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn reduce(&self, 
        acc:  Acc,
        data: Scalar,
        idx:  Index) -> Acc {
        
        todo!();
        /*
            return combine(acc, {data, data});
        */
    }
    
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn combine(&self, a: Acc, b: Acc) -> Acc {
        
        todo!();
        /*
            auto min_val = (_isnan(a.first) || a.first < b.first) ? a.first : b.first;
        auto max_val = (_isnan(a.second) || a.second > b.second) ? a.second : b.second;

        return {min_val, max_val};
        */
    }
    
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn project(&self, acc: Acc) -> Acc {
        
        todo!();
        /*
            return acc;
        */
    }
    
    #[cfg(target_arch = "cuda")]
    pub fn translate_idx(
        acc:      Acc,
        base_idx: i64) -> Acc {
        
        todo!();
        /*
            return acc;
        */
    }

    #[cfg(any(feature = "cudacc", feature = "hipcc"))]
    #[cfg(target_arch = "cuda")]
    #[inline] pub fn warp_shfl_down(&self, 
        acc:    Acc,
        offset: i32) -> Acc {
        
        todo!();
        /*
            return {
          WARP_SHFL_DOWN(acc.first, offset), WARP_SHFL_DOWN(acc.second, offset)
        };
        */
    }
}
