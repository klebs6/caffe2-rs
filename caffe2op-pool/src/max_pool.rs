crate::ix!();

/**
  | consumes an input blob and applies max
  | pooling across the the blob according
  | to kernel sizes, stride sizes, pad lengths
  | and dilation. Max pooling consists
  | of taking the maximum value of a subset
  | of the input tensor according to the
  | kernel size and downsampling the data
  | into the output blob for further processing.
  | The `brew` module has a wrapper for this
  | operator for use in a `ModelHelper`
  | object.
  | 
  | Pooling layers reduce the spatial dimensionality
  | of the input blob. Each of the output
  | blob's dimensions will reduce according
  | to:
  | 
  | $$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h
  |
  */
pub struct MaxPoolFunctor<Context> {
    phantom: PhantomData<Context>,
}

impl<Context> MaxPoolFunctor<Context> {
    
    pub fn new(op: &OperatorStorage) -> Self {
    
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn global_pooling_forward<T, const kOrder: StorageOrder>(
        &self, 
        n:       i32,
        c:       i32,
        hxw:     i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool {
    
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn forward<T, const kOrder: StorageOrder>(
        &self, 
        n:        i32,
        c:        i32,
        x_dims:   &Vec<i32>,
        y_dims:   &Vec<i32>,
        kernel:   &Vec<i32>,
        dilation: &Vec<i32>,
        stride:   &Vec<i32>,
        pads:     &Vec<i32>,
        x:        *const T,
        y:        *mut T,
        context:  *mut Context) -> bool {

        todo!();
        /*

        */
    }
    
    #[inline] pub fn global_pooling_backward<T, const kOrder: StorageOrder>(
        &self, 
        n:       i32,
        c:       i32,
        hxw:     i32,
        dy:      *const T,
        x:       *const T,
        y:       *const T,
        dx:      *mut T,
        context: *mut Context) -> bool {
    
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn backward<T, const kOrder: StorageOrder>(
        &self, 
        n:        i32,
        c:        i32,
        x_dims:   &Vec<i32>,
        y_dims:   &Vec<i32>,
        kernel:   &Vec<i32>,
        dilation: &Vec<i32>,
        stride:   &Vec<i32>,
        pads:     &Vec<i32>,
        dy:       *const T,
        x:        *const T,
        y:        *const T,
        dx:       *mut T,
        context:  *mut Context) -> bool {

        todo!();
        /*
        
        */
    }
}
