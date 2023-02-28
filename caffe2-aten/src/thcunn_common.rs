crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/THCUNN/common.h]

#[macro_export] macro_rules! thcunn_assert_same_gpu {
    (, $($arg:ident),*) => {
        /*
                THAssertMsg(THCTensor_(checkGPU)(__VA_ARGS__), 
          "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one.")
        */
    }
}

/**
  | Use 1024 threads per block, which requires
  | cuda sm_2x or above
  |
  */
pub const CUDA_NUM_THREADS: i32 = 1024;

/// CUDA: number of blocks for threads.
///
#[inline] pub fn GET_BLOCKS(N: i64) -> i32 {
    
    todo!();
        /*
            // Round up division for positive number
      auto block_num = N / CUDA_NUM_THREADS + (N % CUDA_NUM_THREADS == 0 ? 0 : 1);

      constexpr i64 max_int = int::max;
      THAssertMsg(block_num <= max_int, "Can't schedule too many blocks on CUDA device");

      return static_cast<int>(block_num);
        */
}

#[macro_export] macro_rules! thcunn_resize_as_indices {
    ($STATE:ident, $I1:ident, $I2:ident) => {
        /*
        
          if (!I1->sizes().equals(I2->sizes()))                     
          { 
            THCudaLongTensor_resizeAs(STATE, I1, I2);               
          }
        */
    }
}


#[macro_export] macro_rules! thcunn_check_shape {
    ($STATE:ident, $I1:ident, $I2:ident) => {
        /*
        
          if (I1 != NULL && I2 != NULL && !THCTensor_(isSameSizeAs)(STATE, I1, I2))        
          { 
               THCDescBuff s1 = THCTensor_(sizeDesc)(STATE, I1);  
               THCDescBuff s2 = THCTensor_(sizeDesc)(STATE, I2);  
               THError(#I1 " and " #I2 " shapes do not match: "   
                       #I1 " %s, " #I2 " %s", s1.str, s2.str);    
          }
        */
    }
}


#[macro_export] macro_rules! thcunn_check_shape_indices {
    ($STATE:ident, $I1:ident, $I2:ident) => {
        /*
        
          if (!I1->sizes().equals(I2->sizes()))                        
          { 
               THCDescBuff s1 = THCIndexTensor_(sizeDesc)(STATE, I1);  
               THCDescBuff s2 = THCTensor_(sizeDesc)(STATE, I2);       
               THError(#I1 " and " #I2 " shapes do not match: "        
                       #I1 " %s, " #I2 " %s", s1.str, s2.str);         
          }
        */
    }
}


#[macro_export] macro_rules! thcunn_check_n_element {
    ($STATE:ident, $I1:ident, $I2:ident) => {
        /*
        
          if (I1 != NULL && I2 != NULL ) {                          
            ptrdiff_t n1 = THCTensor_(nElement)(STATE, I1);              
            ptrdiff_t n2 = THCTensor_(nElement)(STATE, I2);              
            if (n1 != n2)                                           
            {        
              THCDescBuff s1 = THCTensor_(sizeDesc)(state, I1);     
              THCDescBuff s2 = THCTensor_(sizeDesc)(state, I2);     
              THError(#I1 " and " #I2 " have different number of elements: "        
                      #I1 "%s has %ld elements, while "             
                      #I2 "%s has %ld elements", s1.str, n1, s2.str, n2); 
            }        
          }
        */
    }
}


#[macro_export] macro_rules! thcunn_check_dim_size {
    ($STATE:ident, $T:ident, $DIM:ident, $DIM_SIZE:ident, $SIZE:ident) => {
        /*
        
          if (THCTensor_(nDimensionLegacyNoScalars)(STATE, T) != DIM ||             
              THCTensor_(sizeLegacyNoScalars)(STATE, T, DIM_SIZE) != SIZE) {        
              THCDescBuff s1 = THCTensor_(sizeDesc)(state, T);       
              THError("Need " #T " of dimension %d and " #T ".size[%d] == %d"        
                      " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); 
          }
        */
    }
}


#[macro_export] macro_rules! thcunn_check_dim_size_indices {
    ($STATE:ident, $T:ident, $DIM:ident, $DIM_SIZE:ident, $SIZE:ident) => {
        /*
        
          if (THCIndexTensor_(nDimensionLegacyNoScalars)(STATE, T) != DIM ||                 
              THCIndexTensor_(sizeLegacyNoScalars)(STATE, T, DIM_SIZE) != SIZE) {            
              THCDescBuff s1 = THCIndexTensor_(sizeDesc)(state, T);           
              THError("Need " #T " of dimension %d and " #T ".size[%d] == %d" 
                      " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); 
          }
        */
    }
}


#[macro_export] macro_rules! thcunn_arg_check {
    ($STATE:ident, $COND:ident, $ARG:ident, $T:ident, $FORMAT:ident) => {
        /*
        
          if (!(COND)) { 
            THCDescBuff s1 = THCTensor_(sizeDesc)(state, T); 
            THArgCheck(COND, ARG, FORMAT, s1.str);           
          }
        */
    }
}
