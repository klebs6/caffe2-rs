crate::ix!();

///--------------------------------------------------
#[USE_CONV_POOL_BASE_FUNCTIONS(Context)]
pub struct DeformConvOpBase<T, Context> {
    base: ConvPoolOpBase<Context>,

    deformable_group: i32,

    phantom: PhantomData<T>,
}

#[macro_export] macro_rules! use_deformable_conv_base_functions {
  ($T:ident, $Context:ident) => {
      todo!();
      /*
      USE_CONV_POOL_BASE_FUNCTIONS(Context);                 
      using DeformConvOpBase<T, Context>::deformable_group_; 
      using DeformConvOpBase<T, Context>::DeformableIm2col;  
      using DeformConvOpBase<T, Context>::DeformableCol2im;  
      using DeformConvOpBase<T, Context>::DeformableCol2imCoord
      */
  }
}

impl<T,Context> DeformConvOpBase<T,Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<Context>(operator_def, ws),
            deformable_group_(
                this->template GetSingleArgument<int>("deformable_group", 1))
        */
    }
    
    #[inline] pub fn deformable_im2col(
        &mut self, 
        data_im:       *const T,
        data_offset:   *const T,
        im_shape:      &[i32],
        col_shape:     &[i32],
        data_col:      *mut T)  
    {
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn deformable_col2im(
        &mut self, 
        data_col:      *const T,
        data_offset:   *const T,
        im_shape:      &[i32],
        col_shape:     &[i32],
        grad_im:       *mut T)  
    {
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn deformable_col2im_coord(
        &mut self, 
        data_col:      *const T,
        data_im:       *const T,
        data_offset:   *const T,
        im_shape:      &[i32],
        col_shape:     &[i32],
        grad_offset:   *mut T)  
    {
        todo!();
        /*
        
        */
    }
}

