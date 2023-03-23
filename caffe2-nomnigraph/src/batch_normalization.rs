crate::ix!();

pub struct BatchNormalization {
    //NOMNIGRAPH_DEFINE_NN_RTTI(BatchNormalization);
    base: NeuralNetOperator,

    epsilon:  f32,
    momentum: f32,
    spatial:  bool,
    is_test:  bool,
}

impl BatchNormalization {
    
    pub fn new(
        epsilon:  Option<f32>,
        momentum: Option<f32>,
        spatial:  Option<bool>,
        is_test:  Option<bool>) -> Self {

        let epsilon:  f32 = epsilon.unwrap_or(1e-5);
        let momentum: f32 = momentum.unwrap_or(0.9);
        let spatial: bool = spatial.unwrap_or(true);
        let is_test: bool = is_test.unwrap_or(false);

        todo!();
        /*
            : NeuralNetOperator(NNKind::BatchNormalization),
            epsilon_(epsilon),
            momentum_(momentum),
            spatial_(spatial),
            isTest_(isTest)
        */
    }
    
    #[inline] pub fn get_epsilon(&self) -> f32 {
        
        todo!();
        /*
            return epsilon_;
        */
    }
    
    #[inline] pub fn get_momentum(&self) -> f32 {
        
        todo!();
        /*
            return momentum_;
        */
    }
    
    #[inline] pub fn get_spatial(&self) -> bool {
        
        todo!();
        /*
            return spatial_;
        */
    }
    
    #[inline] pub fn get_is_test(&self) -> bool {
        
        todo!();
        /*
            return isTest_;
        */
    }
    
    #[inline] pub fn set_epsilon(&mut self, epsilon: f32)  {
        
        todo!();
        /*
            epsilon_ = epsilon;
        */
    }
    
    #[inline] pub fn set_momentum(&mut self, momentum: f32)  {
        
        todo!();
        /*
            momentum_ = momentum;
        */
    }
    
    #[inline] pub fn set_spatial(&mut self, spatial: bool)  {
        
        todo!();
        /*
            spatial_ = spatial;
        */
    }
    
    #[inline] pub fn set_is_test(&mut self, is_test: bool)  {
        
        todo!();
        /*
            isTest_ = isTest;
        */
    }
}

