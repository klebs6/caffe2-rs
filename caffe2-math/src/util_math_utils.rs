crate::ix!();

#[inline] pub fn not<T>(x: T) -> T {
    todo!();
    /*
        return !x;
    */
}

#[inline] pub fn sign<T>(x: T) -> T {
    todo!();
    /*
        return x > 0 ? T(1) : (x < 0 ? T(-1) : T(0));
    */
}

#[inline] pub fn negate<T>(x: T) -> T {
    todo!();
    /*
        return -x;
    */
}

#[inline] pub fn inv<T>(x: T) -> T {
    todo!();
    /*
        return T(1) / x;
    */
}

#[inline] pub fn square<T>(x: T) -> T {
    todo!();
    /*
        return x * x;
    */
}

#[inline] pub fn cube<T>(x: T) -> T {
    todo!();
    /*
        return x * x * x;
    */
}

/**
  | Function uses casting from int to unsigned
  | to compare if value of parameter a is
  | greater or equal to zero and lower than
  | value of parameter b.
  | 
  | The b parameter is of type signed and
  | is always positive,
  | 
  | therefore its value is always lower
  | than 0x800... where casting negative
  | value of a parameter converts it to value
  | higher than 0x800...
  | 
  | The casting allows to use one condition
  | instead of two.
  |
  */
#[inline] pub fn is_age_zero_and_altB(a: i32, b: i32) -> bool {

    todo!();
    /*
        return static_cast<unsigned int>(a) < static_cast<unsigned int>(b);
    */
}

/**
  | Calculates ceil(a / b). User must be
  | careful to ensure that there is no overflow
  | or underflow in the calculation.
  |
  */
#[inline] pub fn div_up<T>(a: T, b: T) -> T {
    todo!();
    /*
        return (a + b - T(1)) / b;
    */
}

/**
  | Rounds a up to the next highest multiple of
  | b. User must be careful to ensure that there
  | is no overflow or underflow in the calculation
  | of divUp.
  */
#[inline] pub fn round_up<T>(a: T, b: T) -> T {
    todo!();
    /*
        return DivUp<T>(a, b) * b;
    */
}

/// Returns log2(n) for a positive integer type
#[inline] pub fn integer_log2<T>(n: T, p: i32) -> i32 {
    todo!();
    /*
        return (n <= 1) ? p : IntegerLog2(n / 2, p + 1);
    */
}

/**
  | Returns the next highest power-of-2
  | for an integer type
  |
  */
#[inline] pub fn integer_next_highest_power_of2<T>(v: T) -> T {
    todo!();
    /*
        return (IntegerIsPowerOf2(v) ? T(2) * v : (T(1) << (IntegerLog2(v) + 1)));
    */
}

/**
  | Increase the index digits by one based
  | on dims.
  |
  */
#[macro_export] macro_rules! caffe2_specialized_increase_index_in_dims {
    ($TIndex:ty) => {
        /*
        template <>                                              
            C10_EXPORT void IncreaseIndexInDims<TIndex>(             
                const int ndim, const TIndex* dims, TIndex* index) { 
                for (int i = ndim - 1; i >= 0; --i) {                  
                    ++index[i];                                          
                    if (index[i] >= dims[i]) {                           
                        index[i] -= dims[i];                               
                    } else {                                             
                        break;                                             
                    }                                                    
                }                                                      
            }
        */
    }
}

caffe2_specialized_increase_index_in_dims!{i32}
caffe2_specialized_increase_index_in_dims!{i64}

/// Get index value from dims and index digits.
#[macro_export] macro_rules! caffe2_specialized_get_index_from_dims {
    ($TIndex:ty) => {
        /*
        template <>                                                 
            C10_EXPORT TIndex GetIndexFromDims(                         
                const int n, const TIndex* dims, const TIndex* index) { 
                TIndex sum = 0;                                           
                for (int i = 0; i < n; ++i) {                             
                    if (dims[i] > 1) {                                      
                        sum = sum * dims[i] + index[i];                       
                    }                                                       
                }                                                         
                return sum;                                               
            }
        */
    }
}

caffe2_specialized_get_index_from_dims!{i32}
caffe2_specialized_get_index_from_dims!{i64}

/**
  | Checks if the input permutation is an
  | identity permutation;
  |
  */
#[inline] pub fn is_identity_permutation(n: i32, perm: *const i32) -> bool {
    
    todo!();
    /*
        for (int i = 0; i < n; ++i) {
        if (perm[i] != i) {
          return false;
        }
      }
      return true;
    */
}

#[inline] pub fn check_reduce_dims(
    ndim:   i32,
    x_dims: *const i32,
    y_dims: *const i32) -> bool {
    
    todo!();
    /*
        for (int i = 0; i < ndim; ++i) {
        if (X_dims[i] != Y_dims[i] && Y_dims[i] != 1) {
          return false;
        }
      }
      return true;
    */
}

#[inline] pub fn is_rowwise_reduce(
    ndim:   i32,
    a_dims: *const i32,
    b_dims: *const i32,
    rows:   *mut i32,
    cols:   *mut i32) -> bool {
    
    todo!();
    /*
        *cols = 1;
      int pivot = ndim - 1;
      for (; pivot >= 0 && B_dims[pivot] == 1; --pivot) {
        *cols *= A_dims[pivot];
      }
      *rows = 1;
      for (int i = pivot; i >= 0; --i) {
        if (A_dims[i] != B_dims[i]) {
          return false;
        }
        *rows *= A_dims[i];
      }
      return true;
    */
}

#[inline] pub fn is_colwise_reduce(
    ndim:   i32,
    a_dims: *const i32,
    b_dims: *const i32,
    rows:   *mut i32,
    cols:   *mut i32) -> bool {
    
    todo!();
    /*
        *rows = 1;
      int pivot = 0;
      for (; pivot < ndim && B_dims[pivot] == 1; ++pivot) {
        *rows *= A_dims[pivot];
      }
      *cols = 1;
      for (int i = pivot; i < ndim; ++i) {
        if (A_dims[i] != B_dims[i]) {
          return false;
        }
        *cols *= A_dims[i];
      }
      return true;
    */
}

#[inline] pub fn is_both_ends_reduce(
    ndim:   i32,
    a_dims: *const i32,
    b_dims: *const i32,
    pre:    *mut i32,
    mid:    *mut i32,
    nxt:    *mut i32) -> bool {
    
    todo!();
    /*
        *nxt = 1;
      int r = ndim - 1;
      for (; r >= 0 && B_dims[r] == 1; --r) {
        *nxt *= A_dims[r];
      }
      *pre = 1;
      int l = 0;
      for (; l <= r && B_dims[l] == 1; ++l) {
        *pre *= A_dims[l];
      }
      *mid = 1;
      for (int i = l; i <= r; ++i) {
        if (A_dims[i] != B_dims[i]) {
          return false;
        }
        *mid *= A_dims[i];
      }
      return true;
    */
}

/// Computest the broadcast binary operation dims.
#[macro_export] macro_rules! caffe2_specialized_compute_broadcast_binary_op_dims {
    ($TIndex:ty) => {
        /*
        template <>                                                             
            C10_EXPORT void ComputeBroadcastBinaryOpDims(                           
                const int A_ndim,                                                   
                const TIndex* A_dims,                                               
                const int B_ndim,                                                   
                const TIndex* B_dims,                                               
                TIndex* A_broadcast_dims,                                           
                TIndex* B_broadcast_dims,                                           
                TIndex* C_broadcast_dims) {                                         
                const int ndim = std::max(A_ndim, B_ndim);                            
                std::fill(A_broadcast_dims, A_broadcast_dims + ndim - A_ndim, 1);     
                std::fill(B_broadcast_dims, B_broadcast_dims + ndim - B_ndim, 1);     
                std::copy(A_dims, A_dims + A_ndim, A_broadcast_dims + ndim - A_ndim); 
                std::copy(B_dims, B_dims + B_ndim, B_broadcast_dims + ndim - B_ndim); 
                for (int i = 0; i < ndim; ++i) {                                      
                    CAFFE_ENFORCE(                                                      
                        A_broadcast_dims[i] == B_broadcast_dims[i] ||                   
                        A_broadcast_dims[i] <= 1 || B_broadcast_dims[i] <= 1);          
                    if (A_broadcast_dims[i] == 0 || B_broadcast_dims[i] == 0) {         
                        C_broadcast_dims[i] = 0;                                          
                    } else {                                                            
                        C_broadcast_dims[i] =                                             
                            std::max(A_broadcast_dims[i], B_broadcast_dims[i]);           
                    }                                                                   
                }                                                                     
            }
        */
    }
}

caffe2_specialized_compute_broadcast_binary_op_dims!{i32}
caffe2_specialized_compute_broadcast_binary_op_dims!{i64}

#[inline] pub fn is_rowwise_broadcast_binary_op(
    ndim:          i32,
    a_dims:        *const i32,
    b_dims:        *const i32,
    rows:          *mut i32,
    cols:          *mut i32,
    broadcast_1st: *mut bool) -> bool {
    
    todo!();
    /*
        if (ndim == 0) {
        return false;
      }
      int A_pivot = 0;
      for (; A_pivot < ndim && A_dims[A_pivot] == 1; ++A_pivot)
        ;
      int B_pivot = 0;
      for (; B_pivot < ndim && B_dims[B_pivot] == 1; ++B_pivot)
        ;
      if (A_pivot == B_pivot) {
        return false;
      }
      const int pivot = std::max(A_pivot, B_pivot);
      if (A_pivot > B_pivot) {
        *rows = c10::multiply_integers(B_dims + B_pivot, B_dims + pivot);
        *broadcast_1st = true;
      } else {
        *rows = c10::multiply_integers(A_dims + A_pivot, A_dims + pivot);
        *broadcast_1st = false;
      }
      *cols = 1;
      for (int i = pivot; i < ndim; ++i) {
        if (A_dims[i] != B_dims[i]) {
          return false;
        }
        *cols *= A_dims[i];
      }
      return true;
    */
}

#[inline] pub fn is_colwise_broadcast_binary_op(
    ndim:          i32,
    a_dims:        *const i32,
    b_dims:        *const i32,
    rows:          *mut i32,
    cols:          *mut i32,
    broadcast_1st: *mut bool) -> bool {
    
    todo!();
    /*
        if (ndim == 0) {
        return false;
      }
      int A_pivot = ndim - 1;
      for (; A_pivot >= 0 && A_dims[A_pivot] == 1; --A_pivot)
        ;
      int B_pivot = ndim - 1;
      for (; B_pivot >= 0 && B_dims[B_pivot] == 1; --B_pivot)
        ;
      if (A_pivot == B_pivot) {
        return false;
      }
      ++A_pivot;
      ++B_pivot;
      const int pivot = std::min(A_pivot, B_pivot);
      if (A_pivot < B_pivot) {
        *cols = c10::multiply_integers(B_dims + pivot, B_dims + B_pivot);
        *broadcast_1st = true;
      } else {
        *cols = c10::multiply_integers(A_dims + pivot, A_dims + A_pivot);
        *broadcast_1st = false;
      }
      *rows = 1;
      for (int i = 0; i < pivot; ++i) {
        if (A_dims[i] != B_dims[i]) {
          return false;
        }
        *rows *= A_dims[i];
      }
      return true;
    */
}

#[inline] pub fn is_both_ends_broadcast_binary_op(
    ndim:          i32,
    a_dims:        *const i32,
    b_dims:        *const i32,
    pre:           *mut i32,
    mid:           *mut i32,
    nxt:           *mut i32,
    broadcast_1st: *mut bool) -> bool {
    
    todo!();
    /*
        if (ndim == 0) {
        return false;
      }
      int A_pre = 0;
      for (; A_pre < ndim && A_dims[A_pre] == 1; ++A_pre)
        ;
      int B_pre = 0;
      for (; B_pre < ndim && B_dims[B_pre] == 1; ++B_pre)
        ;
      int A_nxt = ndim - 1;
      for (; A_nxt >= 0 && A_dims[A_nxt] == 1; --A_nxt)
        ;
      int B_nxt = ndim - 1;
      for (; B_nxt >= 0 && B_dims[B_nxt] == 1; --B_nxt)
        ;
      ++A_nxt;
      ++B_nxt;
      if (A_pre == B_pre || A_nxt == B_nxt) {
        return false;
      }
      if (A_pre > B_pre && A_nxt < B_nxt) {
        *pre = c10::multiply_integers(B_dims + B_pre, B_dims + A_pre);
        *nxt = c10::multiply_integers(B_dims + A_nxt, B_dims + B_nxt);
        *broadcast_1st = true;
      } else if (A_pre < B_pre && A_nxt > B_nxt) {
        *pre = c10::multiply_integers(A_dims + A_pre, A_dims + B_pre);
        *nxt = c10::multiply_integers(A_dims + B_nxt, A_dims + A_nxt);
        *broadcast_1st = false;
      } else {
        return false;
      }
      const int l = std::max(A_pre, B_pre);
      const int r = std::min(A_nxt, B_nxt);
      *mid = 1;
      for (int i = l; i < r; ++i) {
        if (A_dims[i] != B_dims[i]) {
          return false;
        }
        *mid *= A_dims[i];
      }
      return true;
    */
}

#[inline] pub fn is_batch_transpose2D(ndim: i32, axes: *const i32) -> bool {
    
    todo!();
    /*
        if (ndim < 2) {
        return false;
      }
      for (int i = 0; i < ndim - 2; ++i) {
        if (axes[i] != i) {
          return false;
        }
      }
      return axes[ndim - 2] == ndim - 1 && axes[ndim - 1] == ndim - 2;
    */
}

#[inline] pub fn compute_transpose_axes_for_reduce_op_with_reduce_axes(
    num_dims:        i32,
    num_reduce_axes: i32,
    reduce_axes:     *const i32,
    transpose_axes:  *mut i32)  {
    
    todo!();
    /*
        const int d = num_dims - num_reduce_axes;
      std::copy_n(reduce_axes, num_reduce_axes, transpose_axes + d);
      std::sort(transpose_axes + d, transpose_axes + num_dims);
      int p = 0;
      int q = d;
      for (int i = 0; i < num_dims; ++i) {
        if (q < num_dims && i == transpose_axes[q]) {
          ++q;
        } else {
          transpose_axes[p++] = i;
        }
      }
    */
}

#[inline] pub fn compute_transpose_axes_for_reduce_op(
    ndim: i32,
    dims: *const i32,
    axes: *mut i32)  {
    
    todo!();
    /*
        const int d = ndim - std::count(dims, dims + ndim, 1);
      int p = 0;
      int q = d;
      for (int i = 0; i < ndim; ++i) {
        if (dims[i] == 1) {
          axes[q++] = i;
        } else {
          axes[p++] = i;
        }
      }
    */
}

#[macro_export] macro_rules! caffe2_specialized_compute_transposed_strides {
    ($TIndex:ty) => {
        /*
        template <>                                                                 
            C10_EXPORT void ComputeTransposedStrides<TIndex>(                           
                const int ndim, const TIndex* dims, const int* axes, TIndex* strides) { 
                std::vector<TIndex> buff(ndim);                                           
                TIndex cur_stride = 1;                                                    
                for (int i = ndim - 1; i >= 0; --i) {                                     
                    buff[i] = cur_stride;                                                   
                    cur_stride *= dims[i];                                                  
                }                                                                         
                for (int i = 0; i < ndim; ++i) {                                          
                    strides[i] = buff[axes[i]];                                             
                }                                                                         
        }
        */
    }
}

caffe2_specialized_compute_transposed_strides!{i32}
caffe2_specialized_compute_transposed_strides!{i64}
