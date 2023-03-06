crate::ix!();

pub struct CudnnMaxPoolFunctor {
    max_pool_functor: MaxPoolFunctor<CUDAContext>,
    deterministic: bool,
}

register_cudnn_operator!{MaxPool,           CudnnPoolOp<CudnnMaxPoolFunctor>}
register_cudnn_operator!{MaxPoolGradient,   CudnnPoolGradientOp<CudnnMaxPoolFunctor>}

register_cudnn_operator!{MaxPool1D,         CudnnPoolOp<CudnnMaxPoolFunctor>}
register_cudnn_operator!{MaxPool1DGradient, CudnnPoolGradientOp<CudnnMaxPoolFunctor>}

register_cudnn_operator!{MaxPool2D,         CudnnPoolOp<CudnnMaxPoolFunctor>}
register_cudnn_operator!{MaxPool2DGradient, CudnnPoolGradientOp<CudnnMaxPoolFunctor>}

register_cudnn_operator!{MaxPool3D,         CudnnPoolOp<CudnnMaxPoolFunctor>}
register_cudnn_operator!{MaxPool3DGradient, CudnnPoolGradientOp<CudnnMaxPoolFunctor>}

impl CudnnMaxPoolFunctor {
    
    pub fn new(op: &OperatorStorage) -> Self {
        todo!();
        /*
            : max_pool_functor(op),
            deterministic(op.GetSingleArgument<bool>("deterministic", false))
        */
    }
    
    #[inline] pub fn get_pooling_mode(&self) -> CudnnPoolingMode {
        
        todo!();
        /*
            #if CUDNN_VERSION_MIN(6, 0, 0)
        return deterministic ? CUDNN_POOLING_MAX_DETERMINISTIC : CUDNN_POOLING_MAX;
    #else
        return CUDNN_POOLING_MAX;
    #endif
        */
    }
    
    #[inline] pub fn global_pooling_forward<T, const kOrder: StorageOrder>(
        &mut self, n: i32,
        c: i32,
        hxW: i32,
        x: *const T,
        y: *mut T,
        context: *mut CUDAContext) -> bool 
    {
        todo!();
        /*
            return max_pool_functor.GlobalPoolingForward<T, kOrder>(
              N, C, HxW, X, Y, context);
        */
    }
    
    #[inline] pub fn forward<T, const kOrder: StorageOrder>(
        &mut self, 
        n:         i32,
        c:         i32,
        x_dims:    &Vec<i32>,
        y_dims:    &Vec<i32>,
        kernel:    &Vec<i32>,
        dilation:  &Vec<i32>,
        stride:    &Vec<i32>,
        pads:      &Vec<i32>,
        x:         *const T,
        y:         *mut T,
        context:   *mut CUDAContext) -> bool 
    {
        todo!();
        /*
            return max_pool_functor.Forward<T, kOrder>(
              N, C, X_dims, Y_dims, kernel, dilation, stride, pads, X, Y, context);
        */
    }
    
    #[inline] pub fn global_pooling_backward<T, const kOrder: StorageOrder>(
        &mut self, 
        n:        i32,
        c:        i32,
        hxW:      i32,
        dY:       *const T,
        x:        *const T,
        y:        *const T,
        dX:       *mut T,
        context:  *mut CUDAContext) -> bool 
    {
        todo!();
        /*
            return max_pool_functor.GlobalPoolingBackward<T, kOrder>(
              N, C, HxW, dY, X, Y, dX, context);
        */
    }
    
    #[inline] pub fn backward<T, const kOrder: StorageOrder>(
        &mut self, 
        n:        i32,
        c:        i32,
        x_dims:   &Vec<i32>,
        y_dims:   &Vec<i32>,
        kernel:   &Vec<i32>,
        dilation: &Vec<i32>,
        stride:   &Vec<i32>,
        pads:     &Vec<i32>,
        dY:       *const T,
        x:        *const T,
        y:        *const T,
        dX:       *mut T,
        context:  *mut CUDAContext) -> bool 
    {
        todo!();
        /*
            return max_pool_functor.Backward<T, kOrder>(
              N,
              C,
              X_dims,
              Y_dims,
              kernel,
              dilation,
              stride,
              pads,
              dY,
              X,
              Y,
              dX,
              context);
        */
    }
}

