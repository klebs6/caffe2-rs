crate::ix!();

#[inline] pub fn reduce_min<T, Context>(
    n:           i32,
    x:           *const T,
    y:           *mut T,
    scratch_ptr: *mut Tensor,
    context:     *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn reduce_max<T, Context>(
    n:           i32,
    x:           *const T,
    y:           *mut T,
    scratch_ptr: *mut Tensor,
    context:     *mut Context)  {
    todo!();
    /*
    
    */
}

/**
  | In all of the reduce functions, X_dims and
  | Y_dims should have ndim elements.
  |
  | Each dimension of Y_dims must match the
  | corresponding dimension of X_dims or must be
  | equal to 1. The dimensions equal to 1 indicate
  | the dimensions of X to be reduced.
  */

/// Y = alpha * ReduceMin(X)
#[inline] pub fn reduce_min_with_alpha<T, Context>(
    ndim:    i32,
    x_dims:  *const i32,
    y_dims:  *const i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

/// Y = alpha * ReduceMax(X)
#[inline] pub fn reduce_max_with_alpha<T, Context>(
    ndim:    i32,
    x_dims:  *const i32,
    y_dims:  *const i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}


/// Y = alpha * ReduceSum(X)
#[inline] pub fn reduce_sum<T, Context>(
    ndim:    i32,
    x_dims:  *const i32,
    y_dims:  *const i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}


/// Y = alpha * ReduceMean(X)
#[inline] pub fn reduce_mean<T, Context>(
    ndim:    i32,
    x_dims:  *const i32,
    y_dims:  *const i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}


/// Y = alpha * ReduceL1(X)
#[inline] pub fn reduceL1<T, Context>(
    ndim:    i32,
    x_dims:  *const i32,
    y_dims:  *const i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}


/// Y = alpha * ReduceL2(X)
#[inline] pub fn reduceL2<T, Context>(
    ndim:    i32,
    x_dims:  *const i32,
    y_dims:  *const i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}


/// Computes mean and variance over axes.
#[inline] pub fn moments<T, Context>(
    ndims:   i32,
    x_dims:  *const i32,
    y_dims:  *const i32,
    x:       *const T,
    mean:    *mut T,
    var:     *mut T,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[cfg(caffe2_use_eigen_for_blas)]
#[macro_export] macro_rules! delegate_rowwise_reduce_function {
    () => {
        //TODO eliminate branch
    };
    ($Func:ident, $EigenFunc:ident) => {
        /*

        template <typename T>                                                
            void Rowwise##Func(                                                  
                const int rows,                                                  
                const int cols,                                                  
                const T alpha,                                                   
                const T* X,                                                      
                T* Y,                                                            
                CPUContext* /* context */) {                                     
                EigenVectorMap<T>(Y, rows) = ConstEigenMatrixMap<T>(X, cols, rows) 
                    .colwise()                        
                    .EigenFunc()                      
                    .transpose() *                    
                    alpha;                                                         
            }
        */
    }
}

#[cfg(caffe2_use_eigen_for_blas)] delegate_rowwise_reduce_function!{/*ReduceMin, minCoeff*/}
#[cfg(caffe2_use_eigen_for_blas)] delegate_rowwise_reduce_function!{/*ReduceMax, maxCoeff*/}
#[cfg(caffe2_use_eigen_for_blas)] delegate_rowwise_reduce_function!{/*ReduceSum, sum*/}
#[cfg(caffe2_use_eigen_for_blas)] delegate_rowwise_reduce_function!{/*ReduceMean, mean*/}
#[cfg(caffe2_use_eigen_for_blas)] delegate_rowwise_reduce_function!{/*ReduceL1, template lpNorm<1>*/}
#[cfg(caffe2_use_eigen_for_blas)] delegate_rowwise_reduce_function!{/*ReduceL2, norm*/}

#[cfg(not(caffe2_use_eigen_for_blas))]
#[macro_export] macro_rules! delegate_rowwise_reduce_function {
    ($T:ty, $Func:ident, $BLASFunc:tt) => {
        /*

        template <>                                               
            void Rowwise##Func(                                       
                const int rows,                                       
                const int cols,                                       
                const T alpha,                                        
                const T* X,                                           
                T* Y,                                                 
                CPUContext* /* context */) {                          
                for (int i = 0; i < rows; ++i) {                        
                    Y[i] = BLASFunc(cols, X + i * cols, 1) * alpha;       
                }                                                       
            }
        */
    }
}

#[cfg(not(caffe2_use_eigen_for_blas))] delegate_rowwise_reduce_function!{f32, ReduceL1, cblas_sasum}
#[cfg(not(caffe2_use_eigen_for_blas))] delegate_rowwise_reduce_function!{f64, ReduceL1, cblas_dasum}
#[cfg(not(caffe2_use_eigen_for_blas))] delegate_rowwise_reduce_function!{f32, ReduceL2, cblas_snrm2}
#[cfg(not(caffe2_use_eigen_for_blas))] delegate_rowwise_reduce_function!{f64, ReduceL2, cblas_dnrm2}

#[macro_export] macro_rules! delegate_colwise_reduce_function {
    ($Func:tt, $MathFunc:tt) => {
        /*

        template <typename T>                                           
            void Colwise##Func(                                             
                const int rows,                                             
                const int cols,                                             
                const T alpha,                                              
                const T* X,                                                 
                T* Y,                                                       
                CPUContext* context) {                                      
                std::memcpy(Y, X, sizeof(T) * cols);                          
                    for (int i = 1; i < rows; ++i) {                              
                        MathFunc<T, CPUContext>(cols, Y, X + i * cols, Y, context); 
                    }                                                             
                Scale<T, T, CPUContext>(cols, alpha, Y, Y, context);          
            }
        */
    }
}

delegate_colwise_reduce_function!{ReduceMin, Min}
delegate_colwise_reduce_function!{ReduceMax, Max}
delegate_colwise_reduce_function!{ReduceSum, Add}

#[inline] pub fn colwise_reduce_mean<T>(
    rows:    i32,
    cols:    i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        ColwiseReduceSum<T>(rows, cols, alpha / static_cast<T>(rows), X, Y, context);
    */
}


#[inline] pub fn colwise_reduceL1<T>(
    rows:    i32,
    cols:    i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        ConstEigenArrayMap<T> X_arr(X, cols, rows);
      EigenVectorArrayMap<T> Y_arr(Y, cols);
      Y_arr = X_arr.col(0).abs();
      for (int i = 1; i < rows; ++i) {
        Y_arr += X_arr.col(i).abs();
      }
      Scale<T, T, CPUContext>(cols, alpha, Y, Y, context);
    */
}

#[inline] pub fn colwise_reduceL2<T>(
    rows:    i32,
    cols:    i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        ConstEigenArrayMap<T> X_arr(X, cols, rows);
      EigenVectorArrayMap<T> Y_arr(Y, cols);
      Y_arr = X_arr.col(0).square();
      for (int i = 1; i < rows; ++i) {
        Y_arr += X_arr.col(i).square();
      }
      Y_arr = Y_arr.sqrt() * alpha;
    */
}

#[inline] pub fn both_ends_reduce_min<T>(
    m:       i32,
    n:       i32,
    k:       i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        EigenVectorArrayMap<T> Y_arr(Y, N);
      Y_arr = ConstEigenArrayMap<T>(X, K, N).colwise().minCoeff();
      for (int i = 1; i < M; ++i) {
        ConstEigenArrayMap<T> X_arr(X + i * N * K, K, N);
        for (int j = 0; j < N; ++j) {
          Y[j] = std::min(Y[j], X_arr.col(j).minCoeff());
        }
      }
      Scale<T, T, CPUContext>(N, alpha, Y, Y, context);
    */
}

#[inline] pub fn both_ends_reduce_max<T>(
    m:       i32,
    n:       i32,
    k:       i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        EigenVectorArrayMap<T> Y_arr(Y, N);
      Y_arr = ConstEigenArrayMap<T>(X, K, N).colwise().maxCoeff();
      for (int i = 1; i < M; ++i) {
        ConstEigenArrayMap<T> X_arr(X + i * N * K, K, N);
        for (int j = 0; j < N; ++j) {
          Y[j] = std::max(Y[j], X_arr.col(j).maxCoeff());
        }
      }
      Scale<T, T, CPUContext>(N, alpha, Y, Y, context);
    */
}

#[inline] pub fn both_ends_reduce_sum<T>(
    m:       i32,
    n:       i32,
    k:       i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        EigenVectorArrayMap<T> Y_arr(Y, N);
      Y_arr = ConstEigenArrayMap<T>(X, K, N).colwise().sum();
      for (int i = 1; i < M; ++i) {
        Y_arr +=
            ConstEigenArrayMap<T>(X + i * N * K, K, N).colwise().sum().transpose();
      }
      Scale<T, T, CPUContext>(N, alpha, Y, Y, context);
    */
}

#[inline] pub fn both_ends_reduce_mean<T>(
    m:       i32,
    n:       i32,
    k:       i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        EigenVectorArrayMap<T> Y_arr(Y, N);
      Y_arr = ConstEigenArrayMap<T>(X, K, N).colwise().sum();
      for (int i = 1; i < M; ++i) {
        Y_arr +=
            ConstEigenArrayMap<T>(X + i * N * K, K, N).colwise().sum().transpose();
      }
      Scale<T, T, CPUContext>(N, alpha / static_cast<T>(M * K), Y, Y, context);
    */
}

#[inline] pub fn both_ends_reduceL1<T>(
    m:       i32,
    n:       i32,
    k:       i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        EigenVectorMap<T> Y_vec(Y, N);
      Y_vec = ConstEigenMatrixMap<T>(X, K, N).colwise().template lpNorm<1>();
      for (int i = 1; i < M; ++i) {
        Y_vec += ConstEigenMatrixMap<T>(X + i * N * K, K, N)
                     .colwise()
                     .template lpNorm<1>()
                     .transpose();
      }
      Scale<T, T, CPUContext>(N, alpha, Y, Y, context);
    */
}

#[inline] pub fn both_ends_reduceL2<T>(
    m:       i32,
    n:       i32,
    k:       i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        ConstEigenArrayMap<T> X0_arr(X, K, N);
      EigenVectorArrayMap<T> Y_arr(Y, N);
      for (int i = 0; i < N; ++i) {
        Y_arr(i) = X0_arr.col(i).square().sum();
      }
      for (int i = 1; i < M; ++i) {
        ConstEigenArrayMap<T> Xi_arr(X + i * N * K, K, N);
        for (int j = 0; j < N; ++j) {
          Y_arr(j) += Xi_arr.col(j).square().sum();
        }
      }
      Y_arr = Y_arr.sqrt() * alpha;
    */
}

#[inline] pub fn reduce_tensor_impl<T, Reducer>(
    ndim:    i32,
    x_dims:  *const i32,
    y_dims:  *const i32,
    reducer: &Reducer,
    init:    T,
    x:       *const T,
    y:       *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        const auto X_size = c10::multiply_integers(X_dims, X_dims + ndim);
      const auto Y_size = c10::multiply_integers(Y_dims, Y_dims + ndim);
      Set<T, CPUContext>(Y_size, init, Y, context);
      std::vector<int> index(ndim, 0);
      for (int X_index = 0; X_index < X_size; ++X_index) {
        const int Y_index = utils::GetIndexFromDims(ndim, Y_dims, index.data());
        Y[Y_index] = reducer(Y[Y_index], X[X_index]);
        utils::IncreaseIndexInDims(ndim, X_dims, index.data());
      }
    */
}

#[inline] pub fn reduce_min_impl<T>(
    ndim:    i32,
    x_dims:  *const i32,
    y_dims:  *const i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        ReduceTensorImpl(
          ndim,
          X_dims,
          Y_dims,
          [](const T a, const T b) { return std::min(a, b); },
          T::max,
          X,
          Y,
          context);
      const auto Y_size = c10::multiply_integers(Y_dims, Y_dims + ndim);
      Scale<T, T, CPUContext>(Y_size, alpha, Y, Y, context);
    */
}

#[inline] pub fn reduce_max_impl<T>(
    ndim:    i32,
    x_dims:  *const i32,
    y_dims:  *const i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        ReduceTensorImpl(
          ndim,
          X_dims,
          Y_dims,
          [](const T a, const T b) { return std::max(a, b); },
          std::numeric_limits<T>::lowest(),
          X,
          Y,
          context);
      const auto Y_size = c10::multiply_integers(Y_dims, Y_dims + ndim);
      Scale<T, T, CPUContext>(Y_size, alpha, Y, Y, context);
    */
}

#[inline] pub fn reduce_sum_impl<T>(
    ndim:    i32,
    x_dims:  *const i32,
    y_dims:  *const i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        ReduceTensorImpl(ndim, X_dims, Y_dims, std::plus<T>(), T(0), X, Y, context);
      const auto Y_size = c10::multiply_integers(Y_dims, Y_dims + ndim);
      Scale<T, T, CPUContext>(Y_size, alpha, Y, Y, context);
    */
}

#[inline] pub fn reduce_mean_impl<T>(
    ndim:    i32,
    x_dims:  *const i32,
    y_dims:  *const i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        ReduceTensorImpl(ndim, X_dims, Y_dims, std::plus<T>(), T(0), X, Y, context);
      const auto X_size = c10::multiply_integers(X_dims, X_dims + ndim);
      const auto Y_size = c10::multiply_integers(Y_dims, Y_dims + ndim);
      Scale<T, T, CPUContext>(
          Y_size,
          alpha * static_cast<T>(Y_size) / static_cast<T>(X_size),
          Y,
          Y,
          context);
    */
}

#[inline] pub fn reduce_l1impl<T>(
    ndim:    i32,
    x_dims:  *const i32,
    y_dims:  *const i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        ReduceTensorImpl(
          ndim,
          X_dims,
          Y_dims,
          [](const T a, const T b) { return a + std::abs(b); },
          T(0),
          X,
          Y,
          context);
      const auto Y_size = c10::multiply_integers(Y_dims, Y_dims + ndim);
      Scale<T, T, CPUContext>(Y_size, alpha, Y, Y, context);
    */
}

#[inline] pub fn reduce_l2impl<T>(
    ndim:    i32,
    x_dims:  *const i32,
    y_dims:  *const i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        ReduceTensorImpl(
          ndim,
          X_dims,
          Y_dims,
          [](const T a, const T b) { return a + b * b; },
          T(0),
          X,
          Y,
          context);
      const auto Y_size = c10::multiply_integers(Y_dims, Y_dims + ndim);
      EigenVectorArrayMap<T> Y_arr(Y, Y_size);
      Y_arr = Y_arr.sqrt() * alpha;
    */
}

#[inline] pub fn rowwise_moments<T>(
    rows: i32,
    cols: i32,
    x:    *const T,
    mean: *mut T,
    var:  *mut T)  {
    todo!();
    /*
        ConstEigenArrayMap<T> X_arr(X, cols, rows);
      for (int i = 0; i < rows; ++i) {
        mean[i] = X_arr.col(i).mean();
        var[i] = X_arr.col(i).square().mean() - mean[i] * mean[i];
      }
    */
}

#[inline] pub fn colwise_moments<T>(
    rows: i32,
    cols: i32,
    x:    *const T,
    mean: *mut T,
    var:  *mut T)  {
    todo!();
    /*
        ConstEigenArrayMap<T> X_arr(X, cols, rows);
      EigenVectorArrayMap<T> mean_arr(mean, cols);
      EigenVectorArrayMap<T> var_arr(var, cols);
      mean_arr = X_arr.col(0);
      var_arr = X_arr.col(0).square();
      for (int i = 1; i < rows; ++i) {
        mean_arr += X_arr.col(i);
        var_arr += X_arr.col(i).square();
      }
      const T scale = T(1) / static_cast<T>(rows);
      mean_arr *= scale;
      var_arr = var_arr * scale - mean_arr.square();
    */
}

#[inline] pub fn both_ends_moments<T>(
    m:    i32,
    n:    i32,
    k:    i32,
    x:    *const T,
    mean: *mut T,
    var:  *mut T)  {
    todo!();
    /*
        ConstEigenArrayMap<T> X_arr(X, K, M * N);
      EigenVectorArrayMap<T> mean_arr(mean, N);
      EigenVectorArrayMap<T> var_arr(var, N);
      for (int i = 0; i < N; ++i) {
        mean_arr(i) = X_arr.col(i).sum();
        var_arr(i) = X_arr.col(i).square().sum();
      }
      for (int i = 1; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          const int c = i * N + j;
          mean_arr(j) += X_arr.col(c).sum();
          var_arr(j) += X_arr.col(c).square().sum();
        }
      }
      const T scale = T(1) / static_cast<T>(M * K);
      mean_arr *= scale;
      var_arr = var_arr * scale - mean_arr.square();
    */
}

#[inline] pub fn moments_impl<T>(
    ndim:    i32,
    x_dims:  *const i32,
    y_dims:  *const i32,
    x:       *const T,
    mean:    *mut T,
    var:     *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        const auto X_size = c10::multiply_integers(X_dims, X_dims + ndim);
      const auto Y_size = c10::multiply_integers(Y_dims, Y_dims + ndim);
      if (X_size == 0) {
        std::memset(mean, 0, sizeof(T) * Y_size);
        std::memset(var, 0, sizeof(T) * Y_size);
        return;
      }
      if (std::equal(X_dims, X_dims + ndim, Y_dims)) {
        std::memcpy(mean, X, sizeof(T) * Y_size);
        std::memset(var, 0, sizeof(T) * Y_size);
        return;
      }
      int rows;
      int cols;
      if (utils::IsRowwiseReduce(ndim, X_dims, Y_dims, &rows, &cols)) {
        RowwiseMoments<T>(rows, cols, X, mean, var);
        return;
      }
      if (utils::IsColwiseReduce(ndim, X_dims, Y_dims, &rows, &cols)) {
        ColwiseMoments<T>(rows, cols, X, mean, var);
        return;
      }
      int pre;
      int mid;
      int nxt;
      if (utils::IsBothEndsReduce(ndim, X_dims, Y_dims, &pre, &mid, &nxt)) {
        BothEndsMoments<T>(pre, mid, nxt, X, mean, var);
        return;
      }
      std::memset(mean, 0, sizeof(T) * Y_size);
      std::memset(var, 0, sizeof(T) * Y_size);
      std::vector<int> index(ndim, 0);
      for (int X_index = 0; X_index < X_size; ++X_index) {
        const int Y_index = utils::GetIndexFromDims(ndim, Y_dims, index.data());
        mean[Y_index] += X[X_index];
        var[Y_index] += X[X_index] * X[X_index];
        utils::IncreaseIndexInDims(ndim, X_dims, index.data());
      }
      const T scale = static_cast<T>(Y_size) / static_cast<T>(X_size);
      EigenVectorArrayMap<T> mean_arr(mean, Y_size);
      EigenVectorArrayMap<T> var_arr(var, Y_size);
      mean_arr *= scale;
      var_arr = var_arr * scale - mean_arr.square();
    */
}

#[macro_export] macro_rules! delegate_global_reduce_function {
    ($T:ty, $Func:ident, $EigenFunc:ident) => {
        /*

        template <>                                               
            C10_EXPORT void Func<T, CPUContext>(                      
                const int N,                                          
                const T* X,                                           
                T* Y,                                                 
                Tensor* /* scratch_ptr */,                            
                CPUContext* /* context */) {                          
                *Y = ConstEigenVectorArrayMap<T>(X, N).EigenFunc();     
      }
        */
    }
}

delegate_global_reduce_function!{f32, ReduceMin, minCoeff}
delegate_global_reduce_function!{i32, ReduceMin, minCoeff}
delegate_global_reduce_function!{i64, ReduceMin, minCoeff}
delegate_global_reduce_function!{f32, ReduceMax, maxCoeff}
delegate_global_reduce_function!{i32, ReduceMax, maxCoeff}
delegate_global_reduce_function!{i64, ReduceMax, maxCoeff}

#[macro_export] macro_rules! delegate_reduce_function {
    () => {
        //TODO eliminate branch
    };
    ($T:ty, $Func:ident, $kInit:ident, $kIsNorm:ident) => {
        /*
        template <>                                                              
            C10_EXPORT void Func<T, CPUContext>(                                     
                const int ndim,                                                      
                const int* X_dims,                                                   
                const int* Y_dims,                                                   
                const T alpha,                                                       
                const T* X,                                                          
                T* Y,                                                                
                CPUContext* context) {                                               
                const int X_size =                                                     
                    std::accumulate(X_dims, X_dims + ndim, 1, std::multiplies<int>()); 
                    const int Y_size =                                                     
                    std::accumulate(Y_dims, Y_dims + ndim, 1, std::multiplies<int>()); 
                    if (X_size == 0) {                                                     
                        Set<T, CPUContext>(Y_size, alpha * kInit, Y, context);               
                            return;                                                              
                    }                                                                      
                if (alpha == T(0)) {                                                   
                    std::memset(Y, 0, sizeof(T) * Y_size);                               
                        return;                                                              
                }                                                                      
                if (std::equal(X_dims, X_dims + ndim, Y_dims)) {                       
                    if (kIsNorm) {                                                       
                        EigenVectorArrayMap<T>(Y, Y_size) =                                
                            ConstEigenVectorArrayMap<T>(X, X_size).abs() * alpha;          
                    } else {                                                             
                        Scale<T, T, CPUContext>(Y_size, alpha, X, Y, context);             
                    }                                                                    
                    return;                                                              
                }                                                                      
                int rows;                                                              
                    int cols;                                                              
                    if (utils::IsRowwiseReduce(ndim, X_dims, Y_dims, &rows, &cols)) {      
                        Rowwise##Func<T>(rows, cols, alpha, X, Y, context);                  
                            return;                                                              
                    }                                                                      
                if (utils::IsColwiseReduce(ndim, X_dims, Y_dims, &rows, &cols)) {      
                    Colwise##Func<T>(rows, cols, alpha, X, Y, context);                  
                        return;                                                              
                }                                                                      
                int M;                                                                 
                    int N;                                                                 
                    int K;                                                                 
                    if (utils::IsBothEndsReduce(ndim, X_dims, Y_dims, &M, &N, &K)) {       
                        BothEnds##Func<T>(M, N, K, alpha, X, Y, context);                    
                            return;                                                              
                    }                                                                      
                Func##Impl<T>(ndim, X_dims, Y_dims, alpha, X, Y, context);             
            }
        */
    }
}

delegate_reduce_function![/*f32, ReduceMin,  f32::MAX, false*/];
delegate_reduce_function![/*f64, ReduceMin,  f64::MAX, false*/];
delegate_reduce_function![/*i32, ReduceMin,  i32::MAX, false*/];
delegate_reduce_function![/*i64, ReduceMin,  i64::MAX, false*/];
delegate_reduce_function![/*f32, ReduceMax,  f32::MIN, false*/];
delegate_reduce_function![/*f64, ReduceMax,  f64::MIN, false*/];
delegate_reduce_function![/*i32, ReduceMax,  i32::MIN, false*/];
delegate_reduce_function![/*i64, ReduceMax,  i64::MIN, false*/];
delegate_reduce_function![/*f32, ReduceSum,  0.0,      false*/];
delegate_reduce_function![/*f64, ReduceSum,  0.0,      false*/];
delegate_reduce_function![/*i32, ReduceSum,  0,        false*/];
delegate_reduce_function![/*i64, ReduceSum,  0,        false*/];
delegate_reduce_function![/*f32, ReduceMean, 0.0,      false*/];
delegate_reduce_function![/*f64, ReduceMean, 0.0,      false*/];
delegate_reduce_function![/*f32, ReduceL1,   0.0,      true*/];
delegate_reduce_function![/*f64, ReduceL1,   0.0,      true*/];
delegate_reduce_function![/*i32, ReduceL1,   0,        true*/];
delegate_reduce_function![/*i64, ReduceL1,   0,        true*/];
delegate_reduce_function![/*f32, ReduceL2,   0.0,      true*/];
delegate_reduce_function![/*f64, ReduceL2,   0.0,      true*/];

#[macro_export] macro_rules! caffe2_specialized_moments {
    ($T:ty) => {
        /*

        template <>                                                    
            C10_EXPORT void Moments<T, CPUContext>(                        
                const int ndim,                                            
                const int* X_dims,                                         
                const int* Y_dims,                                         
                const T* X,                                                
                T* mean,                                                   
                T* var,                                                    
                CPUContext* context) {                                     
                MomentsImpl<T>(ndim, X_dims, Y_dims, X, mean, var, context); 
            }
        */
    }
}

caffe2_specialized_moments!{f32}
caffe2_specialized_moments!{f64}
