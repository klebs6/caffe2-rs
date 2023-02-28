crate::ix!();

/**
  | Transpose tensor X with dims by axes
  | and write the result to tensor Y.
  |
  */
#[inline] pub fn transpose<TIndex, TData, Context>(
    ndim:    i32,
    dims:    *const TIndex,
    axes:    *const i32,
    x:       *const TData,
    y:       *mut TData,
    context: *mut Context)  {
    todo!();
    /*
    
    */
}

#[inline] pub fn transpose2D<TIndex, TData>(
    rows: TIndex,
    cols: TIndex,
    x:    *const TData,
    y:    *mut TData)  {
    todo!();
    /*
        EigenMatrixMap<TData>(Y, rows, cols) =
          ConstEigenMatrixMap<TData>(X, cols, rows).transpose();
    */
}

#[cfg(caffe2_use_mkl)]
#[macro_export] macro_rules! delegate_transpose_2d {
    ($TIndex:ty, $TData:ty, $MKLFunc:ident) => {
        /*
        template <>                                                           
            void Transpose2D<TIndex, TData>(                                      
                const TIndex rows, const TIndex cols, const TData* X, TData* Y) { 
                MKLFunc('R', 'T', rows, cols, TData(1), X, cols, Y, rows);          
            }
        */
    }
}

#[cfg(caffe2_use_mkl)] delegate_transpose_2d!{i32, f32, mkl_somatcopy}
#[cfg(caffe2_use_mkl)] delegate_transpose_2d!{i64, f32, mkl_somatcopy}
#[cfg(caffe2_use_mkl)] delegate_transpose_2d!{i32, f64, mkl_domatcopy}
#[cfg(caffe2_use_mkl)] delegate_transpose_2d!{i64, f64, mkl_domatcopy}

#[cfg(caffe2_use_hptt)]
#[inline] pub fn transpose_byHPTT<TIndex, TData>(
    ndim: i32,
    dims: *const TIndex,
    axes: *const i32,
    x:    *const TData,
    y:    *mut TData) -> bool {
    todo!();
    /*
        for (int i = 0; i < ndim; ++i) {
        if (dims[i] <= 0 || dims[i] > int::max) {
          return false;
        }
      }

      std::vector<int> axes_cm(ndim);
      std::vector<int> dims_cm(ndim);
      // Convert row-major index to column-major.
      const auto cm_fn = [ndim](const int i) { return ndim - i - 1; };
      for (int i = 0; i < ndim; ++i) {
        axes_cm[i] = cm_fn(axes[cm_fn(i)]);
        dims_cm[i] = dims[cm_fn(i)];
      }
      auto plan = hptt::create_plan(
          axes_cm.data(),
          ndim,
          TData(1),
          X,
          dims_cm.data(),
          nullptr,
          TData(0),
          Y,
          nullptr,
          hptt::ESTIMATE,
          1 /* num_threads */);
      if (plan == nullptr) {
        return false;
      }
      plan->execute();
      return true;
    */
}

#[inline] pub fn transposeND<TIndex, TData>(
    ndim: i32,
    dims: *const TIndex,
    axes: *const i32,
    x:    *const TData,
    y:    *mut TData)  {
    todo!();
    /*
        std::vector<TIndex> Y_dims(ndim);
      for (int i = 0; i < ndim; ++i) {
        Y_dims[i] = dims[axes[i]];
      }
      // Measure amount of contiguous data we can copy at once
      int pivot = ndim - 1;
      TIndex block_size = 1;
      for (; pivot >= 0 && axes[pivot] == pivot; --pivot) {
        block_size *= Y_dims[pivot];
      }
      ++pivot;
      const TIndex num_blocks = std::accumulate(
          Y_dims.cbegin(),
          Y_dims.cbegin() + pivot,
          TIndex(1),
          std::multiplies<TIndex>());
      std::vector<TIndex> X_strides(pivot);
      utils::ComputeTransposedStrides<TIndex>(pivot, dims, axes, X_strides.data());
      std::vector<TIndex> index(pivot, 0);
      for (TIndex Y_index = 0; Y_index < num_blocks; ++Y_index) {
        const TIndex X_index = std::inner_product(
            X_strides.cbegin(), X_strides.cend(), index.cbegin(), TIndex(0));
        if (block_size == 1) {
          Y[Y_index] = X[X_index];
        } else {
          std::memcpy(
              Y + block_size * Y_index,
              X + block_size * X_index,
              block_size * sizeof(TData));
        }
        utils::IncreaseIndexInDims<TIndex>(pivot, Y_dims.data(), index.data());
      }
    */
}

#[inline] pub fn transpose_impl<TIndex, TData>(
    ndim: i32,
    dims: *const TIndex,
    axes: *const i32,
    x:    *const TData,
    y:    *mut TData)  {
    todo!();
    /*
        const TIndex size =
          std::accumulate(dims, dims + ndim, TIndex(1), std::multiplies<TIndex>());
      if (size == 0) {
        return;
      }
      if (utils::IsIdentityPermutation(ndim, axes)) {
        std::memcpy(Y, X, size * sizeof(TData));
        return;
      }
      if (utils::IsBatchTranspose2D(ndim, axes)) {
        const TIndex H = dims[ndim - 2];
        const TIndex W = dims[ndim - 1];
        const TIndex N = size / (H * W);
        for (TIndex i = 0; i < N; ++i) {
          Transpose2D<TIndex, TData>(H, W, X + i * H * W, Y + i * H * W);
        }
        return;
      }
      TransposeND<TIndex, TData>(ndim, dims, axes, X, Y);
    */
}

#[cfg(caffe2_use_hptt)]
#[macro_export] macro_rules! caffe2_specialized_transpose_impl {
    ($TIndex:ty, $TData:ty) => {
        /*
           template<>
            void TransposeImpl<TIndex, TData>(                                    
                const int ndim,                                                   
                const TIndex* dims,                                               
                const int* axes,                                                  
                const TData* X,                                                   
                TData* T) {                                                       
                const TIndex size = std::accumulate(                                
                    dims, dims + ndim, TIndex(1), std::multiplies<TIndex>());       
                if (size == 0) {                                                    
                    return;                                                           
                }                                                                   
                if (utils::IsIdentityPermutation(ndim, axes)) {                     
                    std::memcpy(Y, X, size * sizeof(TData));                          
                    return;                                                           
                }                                                                   
                if (TransposeByHPTT(ndim, dims, axes, X, Y)) {                      
                    return;                                                           
                }                                                                   
                if (utils::IsBatchTranspose2D(ndim, axes)) {                        
                    const TIndex H = dims[ndim - 2];                                  
                    const TIndex W = dims[ndim - 1];                                  
                    const TIndex N = size / (H * W);                                  
                    for (TIndex i = 0; i < N; ++i) {                                  
                        Transpose2D<TIndex, TData>(H, W, X + i * H * W, Y + i * H * W); 
                    }                                                                 
                    return;                                                           
                }                                                                   
                TransposeND<TIndex, TData>(ndim, dims, axes, X, Y);                 
            }
        */
    }
}

#[cfg(caffe2_use_hptt)] caffe2_specialized_transpose_impl!{i32, f32}
#[cfg(caffe2_use_hptt)] caffe2_specialized_transpose_impl!{i64, f32}
#[cfg(caffe2_use_hptt)] caffe2_specialized_transpose_impl!{i32, f64}
#[cfg(caffe2_use_hptt)] caffe2_specialized_transpose_impl!{i64, f64}

#[macro_export] macro_rules! caffe2_specialized_transpose {
    ($TIndex:ty, $TData:ty) => {
        /*
        template <>                                             
            C10_EXPORT void Transpose<TIndex, TData, CPUContext>(   
                const int ndim,                                     
                const TIndex* dims,                                 
                const int* axes,                                    
                const TData* X,                                     
                TData* Y,                                           
                CPUContext* /* context */) {                        
                TransposeImpl<TIndex, TData>(ndim, dims, axes, X, Y); 
            }
                */
    }
}

caffe2_specialized_transpose!{i32, f32}
caffe2_specialized_transpose!{i64, f32}
caffe2_specialized_transpose!{i32, f64}
caffe2_specialized_transpose!{i64, f64}
caffe2_specialized_transpose!{i32, i32}
caffe2_specialized_transpose!{i64, i32}
caffe2_specialized_transpose!{i32, i64}
caffe2_specialized_transpose!{i64, i64}
caffe2_specialized_transpose!{i32, u8}
caffe2_specialized_transpose!{i64, u8}
caffe2_specialized_transpose!{i32, u16}
caffe2_specialized_transpose!{i64, u16}

#[macro_export] macro_rules! caffe2_specialized_nchw2nhwc {
    ($T:ty) => {
        /*
        template <>                                              
            C10_EXPORT void NCHW2NHWC<T, CPUContext>(                
                const int N,                                         
                const int C,                                         
                const int HxW,                                       
                const T* X,                                          
                T* Y,                                                
                CPUContext* /* context */) {                         
                const int stride = C * HxW;                            
                for (int i = 0; i < N; ++i) {                          
                    Transpose2D(C, HxW, X + i * stride, Y + i * stride); 
                }                                                      
            }
                */
    }
}

caffe2_specialized_nchw2nhwc!{f32}

#[macro_export] macro_rules! caffe2_specialized_nhwc2nchw {
    ($T:ty) => {
        /*
        template <>                                              
            C10_EXPORT void NHWC2NCHW<T, CPUContext>(                
                const int N,                                         
                const int C,                                         
                const int HxW,                                       
                const T* X,                                          
                T* Y,                                                
                CPUContext* /* context */) {                         
                const int stride = HxW * C;                            
                for (int i = 0; i < N; ++i) {                          
                    Transpose2D(HxW, C, X + i * stride, Y + i * stride); 
                }                                                      
            }
                */
    }
}

caffe2_specialized_nhwc2nchw!{f32}
