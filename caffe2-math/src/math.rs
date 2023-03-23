crate::ix!();

//TODO: this is unknown
pub struct TypeMetaCopy { }

/**
  | An empty class as a placeholder for a
  | math function that has no specific engine
  | specified.
  |
  */
pub struct DefaultEngine {}

#[macro_export] macro_rules! declare_compare_op {
    ($Comp:ident) => {
        /*
  template <typename T, class Context, bool kBroadcast1st = false> 
  void Rowwise##Comp(                                              
      const int rows,                                              
      const int cols,                                              
      const T* A,                                                  
      const T* B,                                                  
      bool* C,                                                     
      Context* context);                                           
                                                                   
  template <typename T, class Context, bool kBroadcast1st = false> 
  void Colwise##Comp(                                              
      const int rows,                                              
      const int cols,                                              
      const T* A,                                                  
      const T* B,                                                  
      bool* C,                                                     
      Context* context);                                           
                                                                   
  template <typename T, class Context>                             
  void Comp(                                                       
      const int A_ndim,                                            
      const int* A_dims,                                           
      const int B_ndim,                                            
      const int* B_dims,                                           
      const T* A,                                                  
      const T* B,                                                  
      bool* C,                                                     
      Context* context);
  */
    }
}

declare_compare_op!{EQ}
declare_compare_op!{NE}
declare_compare_op!{LT}
declare_compare_op!{LE}
declare_compare_op!{GT}
declare_compare_op!{GE}

#[macro_export] macro_rules! declare_binary_op {
    ($Func:ident) => {
        /*
          template <typename T, class Context, bool kBroadcast1st = false> 
          void Rowwise##Func(                                              
              const int rows,                                              
              const int cols,                                              
              const T* A,                                                  
              const T* B,                                                  
              T* C,                                                        
              Context* context);                                           
                                                                           
          template <typename T, class Context, bool kBroadcast1st = false> 
          void Colwise##Func(                                              
              const int rows,                                              
              const int cols,                                              
              const T* A,                                                  
              const T* B,                                                  
              T* C,                                                        
              Context* context);                                           
                                                                           
          template <typename T, class Context>                             
          void Func(                                                       
              const int A_ndim,                                            
              const int* A_dims,                                           
              const int B_ndim,                                            
              const int* B_dims,                                           
              const T* A,                                                  
              const T* B,                                                  
              T* C,                                                        
              Context* context);
          */
    }
}

declare_binary_op!{Add}
declare_binary_op!{Sub}
declare_binary_op!{Mul}
declare_binary_op!{Div}

declare_binary_op!{And}
declare_binary_op!{Or}
declare_binary_op!{Xor}

declare_binary_op!{BitwiseAnd}
declare_binary_op!{BitwiseOr}
declare_binary_op!{BitwiseXor}

// Broadcasts X with X_dims to Y with Y_dims.
#[inline] pub fn broadcast<T, Context>(
    x_ndim:  i32,
    x_dims:  *const i32,
    y_ndim:  i32,
    y_dims:  *const i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

// Computes inv_std from variance.
#[inline] pub fn inv_std<T, Context>(
    n:       i32,
    epsilon: T,
    var:     *const T,
    inv_std: *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

/**
  | Adds batch sub-tensors elementwise
  | to output. Stripe is the stripe length
  | and N is the number of elements to add
  | (size of Y).
  |
  */
#[inline] pub fn add_striped_batch<T, Context>(
    n:       i32,
    first:   *const T,
    y:       *mut T,
    stripe:  i32,
    batch:   i32,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

/**
  | Compute the row-wise max of a N*D matrix
  | X, and write it to a N dimensional vector
  | y.
  |
  */
#[inline] pub fn rowwise_max<T, Context>(
    n:       i32,
    d:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

/**
  | Compute the column-wise max of a N*D
  | matrix X, and write it to a D dimensional
  | vector y.
  |
  */
#[inline] pub fn colwise_max<T, Context>(
    n:       i32,
    d:       i32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

/**
  | Elemwise maximum of vector x and scalar
  | alpha. y[i] = max(x[i], alpha)
  |
  */
#[inline] pub fn maximum<T, Context>(
    n:       i32,
    alpha:   f32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

/**
  | Decaf gemm provides a simpler interface
  | to the gemm functions, with the limitation
  | that the data has to be contiguous in
  | memory.
  |
  */
#[inline] pub fn gemm<T, Context, Engine>(
    trans_A:   cblas_sys::CBLAS_TRANSPOSE,
    trans_B:   cblas_sys::CBLAS_TRANSPOSE,
    m:         i32,
    n:         i32,
    k:         i32,
    alpha:     f32,
    a:         *const T,
    b:         *const T,
    beta:      f32,
    c:         *mut T,
    context:   *mut Context,
    math_type: TensorProto_DataType)  {
    todo!();
    /*
    
    */
}

/**
  | We also provide a gemm that has explicit lda,
  | ldb and ldc specified.
  |
  | In most cases you probably want to use the
  | function above, though.
  */
#[inline] pub fn gemm_ex<T, Context, Engine>(
    trans_A: cblas_sys::CBLAS_TRANSPOSE,
    trans_B: cblas_sys::CBLAS_TRANSPOSE,
    m:       i32,
    n:       i32,
    k:       i32,
    alpha:   T,
    a:       *const T,
    lda:     i32,
    b:       *const T,
    ldb:     i32,
    beta:    T,
    c:       *mut T,
    ldc:     i32,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

/**
  | GemmBatched provides a simple abstraction
  | into library routines
  |
  */
#[inline] pub fn gemm_batched<T, Context, Engine>(
    trans_A:    cblas_sys::CBLAS_TRANSPOSE,
    trans_B:    cblas_sys::CBLAS_TRANSPOSE,
    batch_size: i32,
    m:          i32,
    n:          i32,
    k:          i32,
    alpha:      f32,
    a:          *const *const T,
    b:          *const *const T,
    beta:       f32,
    c:          *mut *mut T,
    context:    *mut Context,
    math_type:  TensorProto_DataType)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn gemm_strided_batched<T, Context, Engine>(
    trans_A:    cblas_sys::CBLAS_TRANSPOSE,
    trans_B:    cblas_sys::CBLAS_TRANSPOSE,
    batch_size: i32,
    m:          i32,
    n:          i32,
    k:          i32,
    alpha:      f32,
    a:          *const T,
    a_stride:   i32,
    b:          *const T,
    b_stride:   i32,
    beta:       f32,
    c:          *mut T,
    c_stride:   i32,
    context:    *mut Context,
    math_type:  TensorProto_DataType)
{
    todo!();
    /*
    
    */
}

/**
  | Gemv always takes in a M*N matrix A, and
  | depending on whether we set TransA to Trans,
  | the output is:
  |
  | CblasNoTrans: x is an N dim vector and y is an M dim vector.
  | CblasTrans:   x is an M dim vector and y is an N dim vector.
  */
#[inline] pub fn gemv<T, Context, Engine>(
    trans_A:   cblas_sys::CBLAS_TRANSPOSE,
    m:         i32,
    n:         i32,
    alpha:     f32,
    a:         *const T,
    x:         *const T,
    beta:      f32,
    y:         *mut T,
    context:   *mut Context,
    math_type: TensorProto_DataType)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn rand_uniform<T, Context>(
    n:       usize,
    a:       T,
    b:       T,
    r:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

/**
  | Generate n values that sum up to a fixed
  | sum and subject to a restriction a <=
  | x <= b for each x generated
  |
  */
#[inline] pub fn rand_fixed_sum<T, Context>(
    n:       usize,
    a:       T,
    b:       T,
    sum:     T,
    r:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn rand_uniform_unique<T, Context>(
    n:       usize,
    a:       T,
    b:       T,
    r:       *mut T,
    m:       usize,
    avoid:   *const T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

/**
  | Generate n values from synthetic data
  | distribution, define by unique accesses
  | and stack distances
  |
  */
#[inline] pub fn rand_synthetic_data<T, Context>(
    n:       usize,
    a:       T,
    b:       T,
    r:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn rand_gaussian<T, Context>(
    n:       usize,
    mean:    T,
    std:     T,
    r:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

/**
  | Dot matrix of vector a and b, and writes
  | the result to a single value y.
  |
  */
#[inline] pub fn dot<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

/**
  | Sum of vector x, and writes the result
  | to a single value y.
  |
  */
#[inline] pub fn sum<T, Context>(
    n:           i32,
    x:           *const T,
    y:           *mut T,
    context:     *mut Context,
    scratch_ptr: *mut Tensor)  {
    todo!();
    /*
    
    */
}

/**
  | Sum of squares of vector x, and writes
  | the result to a single value y.
  |
  */
#[inline] pub fn sum_sqr<T, Context>(
    n:           i32,
    x:           *const T,
    y:           *mut T,
    context:     *mut Context,
    scratch_ptr: *mut Tensor)  {
    todo!();
    /*
    
    */
}

/**
  | Select does index selection of the rows
  | a N*D matrix x, and gives the N dimensional
  | vector y that contains the selected
  | data.
  |
  */
#[inline] pub fn select<T, Context>(
    n:       i32,
    d:       i32,
    x:       *const T,
    idx:     *const i32,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

/**
  | groups must be 1 for GPU
  |
  | For NHWC order with groups > 1, the result
  | will be layout in NHW G RS C/G order to make
  | data within the same group to be contiguous.
  |
  | For NCHW order, groups doesn't make any
  | difference because we're doing Im2Col for each
  | N and C is the slowest moving dimension among
  | CHW.
  */
#[inline] pub fn im_2col<T, Context>(
    kOrder:     StorageOrder,
    channels:   i32,
    height:     i32,
    width:      i32,
    kernel_h:   i32,
    kernel_w:   i32,
    dilation_h: i32,
    dilation_w: i32,
    pad_t:      i32,
    pad_l:      i32,
    pad_b:      i32,
    pad_r:      i32,
    stride_h:   i32,
    stride_w:   i32,
    img_data:   *const T,
    col_data:   *mut T,
    context:    *mut Context,
    groups:     i32)  {
    todo!();
    /*
    
    */
}

/// groups must be 1 for GPU
#[inline] pub fn im_2col_nd<T, Context>(
    kOrder:       StorageOrder,
    n:            i32,
    img_size:     i32,
    col_size:     i32,
    img_shape:    *const i32,
    col_shape:    *const i32,
    kernel_shape: *const i32,
    stride:       *const i32,
    dilation:     *const i32,
    pad:          *const i32,
    img_data:     *const T,
    col_data:     *mut T,
    context:      *mut Context,
    groups:       i32)  {
    todo!();
    /*
    
    */
}

/**
  | groups must be 1 for GPU
  |
  | For NHWC order with groups > 1, the result
  | will be layout in NHW G RS C/G order to make
  | data within the same group to be contiguous.
  |
  | For NCHW order, groups doesn't make any
  | difference because we're doing Im2Col for each
  | N and C is the slowest moving dimension among
  | CHW.
  */
#[inline] pub fn col_2im<T, Context>(
    kOrder:     StorageOrder,
    channels:   i32,
    height:     i32,
    width:      i32,
    patch_h:    i32,
    patch_w:    i32,
    dilation_h: i32,
    dilation_w: i32,
    pad_t:      i32,
    pad_l:      i32,
    pad_b:      i32,
    pad_r:      i32,
    stride_h:   i32,
    stride_w:   i32,
    col_data:   *const T,
    img_data:   *mut T,
    context:    *mut Context,
    groups:     i32)  {
    todo!();
    /*
    
    */
}

/**
  | groups must be 1 for GPU
  |
  | For NHWC order with groups > 1, the result
  | will be layout in NHW G RS C/G order to make
  | data within the same group to be contiguous.
  |
  | For NCHW order, groups doesn't make any
  | difference because we're doing Im2Col for each
  | N and C is the slowest moving dimension among
  | CHW.
  */
#[inline] pub fn col_2im_nd<T, Context>(
    kOrder:       StorageOrder,
    n:            i32,
    img_size:     i32,
    col_size:     i32,
    img_shape:    *const i32,
    col_shape:    *const i32,
    kernel_shape: *const i32,
    stride:       *const i32,
    dilation:     *const i32,
    pad:          *const i32,
    col_data:     *const T,
    img_data:     *mut T,
    context:      *mut Context,
    groups:       i32)  {
    todo!();
    /*
    
    */
}

/**
  | Applies a per-channel bias value to
  | each channel of the input image. image_size
  | is H * W
  |
  */
#[inline] pub fn biasCHW<T, Context>(
    bias:            *const T,
    bias_multiplier: *const T,
    bias_channels:   i32,
    image_size:      i32,
    image:           *mut T,
    context:         *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn copy_matrix_with_item_size<Context>(
    item_size: usize,
    m:         i32,
    n:         i32,
    a:         *const c_void,
    lda:       i32,
    b:         *mut c_void,
    ldb:       i32,
    context:   *mut Context,
    copy:      TypeMetaCopy)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn copy_matrix<T, Context>(
    m:       i32,
    n:       i32,
    a:       *const T,
    lda:     i32,
    b:       *mut T,
    ldb:     i32,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn copy_matrix_strided<T, Context>(
    m:              i32,
    n:              i32,
    a:              *const T,
    a_outer_stride: i32,
    a_inner_stride: i32,
    b:              *mut T,
    b_outer_stride: i32,
    b_inner_stride: i32,
    context:        *mut Context)  {
    todo!();
    /*
    
    */
}


#[inline] pub fn copy_vector<T, Context>(
    n:       i32,
    a:       *const T,
    b:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}
