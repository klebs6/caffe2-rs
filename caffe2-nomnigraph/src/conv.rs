crate::ix!();

#[NOMNIGRAPH_DEFINE_NN_RTTI("Conv")]
pub struct Conv {
    base:          NeuralNetOperator,
    kernel_shape:  Vec<i32>,
    pads:          Vec<i32>,
    strides:       Vec<i32>,
    group:         i32,
    dilations:     Vec<i32>,
}

impl Conv {
    pub fn new(
        kernel_shape:    Vec<i32>, 
        pads:            Option<Vec<i32>>,
        strides:         Option<Vec<i32>>,
        group:           Option<i32>,
        dilations:       Option<Vec<i32>>) -> Self 
    {
        let pads      = pads.unwrap_or(vec![0, 0]);
        let strides   = strides.unwrap_or(vec![1, 1]);
        let group     = group.unwrap_or(1);
        let dilations = dilations.unwrap_or(vec![1, 1]);

        todo!();
        /*
      : NeuralNetOperator(NNKind::Conv),
        kernelShape_(kernelShape),
        pads_(pads),
        strides_(strides),
        group_(group),
        dilations_(dilations) 
        */
    }
    
    #[inline] pub fn get_kernel_shape(&self) -> Vec<i32> {
        
        todo!();
        /*
            return kernelShape_;
        */
    }
    
    #[inline] pub fn get_pads(&self) -> Vec<i32> {
        
        todo!();
        /*
            return pads_;
        */
    }
    
    #[inline] pub fn get_strides(&self) -> Vec<i32> {
        
        todo!();
        /*
            return strides_;
        */
    }
    
    #[inline] pub fn get_group(&self) -> i32 {
        
        todo!();
        /*
            return group_;
        */
    }
    
    #[inline] pub fn get_dilations(&self) -> Vec<i32> {
        
        todo!();
        /*
            return dilations_;
        */
    }
    
    #[inline] pub fn set_kernel_shape(&mut self, kernel_shape: Vec<i32>)  {
        
        todo!();
        /*
            kernelShape_ = std::move(kernelShape);
        */
    }
    
    #[inline] pub fn set_pads(&mut self, pads: Vec<i32>)  {
        
        todo!();
        /*
            pads_ = std::move(pads);
        */
    }
    
    #[inline] pub fn set_strides(&mut self, strides: Vec<i32>)  {
        
        todo!();
        /*
            strides_ = std::move(strides);
        */
    }
    
    #[inline] pub fn set_group(&mut self, group: i32)  {
        
        todo!();
        /*
            group_ = group;
        */
    }
    
    #[inline] pub fn set_dilations(&mut self, dilations: Vec<i32>)  {
        
        todo!();
        /*
            dilations_ = std::move(dilations);
        */
    }
}

