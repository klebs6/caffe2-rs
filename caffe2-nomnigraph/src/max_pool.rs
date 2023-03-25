crate::ix!();

pub struct MaxPool {
    //NOMNIGRAPH_DEFINE_NN_RTTI(MaxPool);
    base: NeuralNetOperator,

    kernel_shape: Vec<i32>,
    pads:         Vec<i32>,
    strides:      Vec<i32>,
}

impl MaxPool {

    pub fn new(
        kernel_shape:    Vec<i32>, 
        pads:            Option<Vec<i32>>,
        strides:         Option<Vec<i32>>) -> Self
    {
        let pads      = pads.unwrap_or(vec![0, 0]);
        let strides   = strides.unwrap_or(vec![1, 1]);

        todo!();
        /*
          : NeuralNetOperator(NNKind::MaxPool),
            kernelShape_(kernelShape),
            pads_(pads),
            strides_(strides) 
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
}