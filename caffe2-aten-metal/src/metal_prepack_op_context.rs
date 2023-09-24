crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/metal/MetalPrepackOpContext.h]

pub type SerializationTypeConv2dPrePack = (
    Tensor,
    Option<Tensor>,
    Vec<i64>,
    Vec<i64>,
    Vec<i64>,
    i64,
    Option<Scalar>,
    Option<Scalar>
);

pub struct Conv2dOpContext {
    base:             CustomClassHolder,

    weight:           Tensor,
    bias:             Option<Tensor>,
    stride:           Vec<i64>,
    padding:          Vec<i64>,
    dilation:         Vec<i64>,
    groups:           i64,
    output_min:       Option<Scalar>,
    output_max:       Option<Scalar>,

    /**
      | reserved to hold MPSCNNConv2dOp objects
      |
      */
    conv2d_op:        *mut void, // default = nullptr

    release_callback: fn(_0: *mut void) -> (), // default = nullptr
}

impl Conv2dOpContext {
    
    pub fn pack(&mut self) -> SerializationTypeConv2dPrePack {
        
        todo!();
        /*
            return make_tuple(
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            output_min,
            output_max);
        */
    }
    
    pub fn new(
        weight:     Tensor,
        bias:       Option<Tensor>,
        stride:     &Vec<i64>,
        padding:    &Vec<i64>,
        dilation:   &Vec<i64>,
        groups:     i64,
        output_min: &Option<Scalar>,
        output_max: &Option<Scalar>) -> Self {
    
        todo!();
        /*
        : weight(move(weight)),
        : bias(move(bias)),
        : stride(stride),
        : padding(padding),
        : dilation(dilation),
        : groups(groups),
        : output_min(output_min),
        : output_max(output_max),

        
        */
    }
    
    pub fn release_resources(&mut self)  {
        
        todo!();
        /*
            if (releaseCallback) {
          releaseCallback(conv2dOp);
          conv2dOp = nullptr;
        }
        */
    }
}

pub type SerializationTypeLinearPrePack = (
    Tensor,
    Option<Tensor>,
    Option<Scalar>,
    Option<Scalar>
);

pub struct LinearOpContext {
    base:             CustomClassHolder,
    weight:           Tensor,
    bias:             Option<Tensor>,
    output_min:       Option<Scalar>,
    output_max:       Option<Scalar>,

    /**
      | reserved to hold MPSCNNFullyConnected
      | objects
      |
      */
    opaque_op_ptr:    *mut void, // default = nullptr

    release_callback: fn(_0: *mut void) -> (), // default = nullptr
}

impl LinearOpContext {
    
    pub fn pack(&mut self) -> SerializationTypeLinearPrePack {
        
        todo!();
        /*
            return make_tuple(weight_, bias_, output_min_, output_max_);
        */
    }
    
    pub fn new(
        weight:     Tensor,
        bias:       Option<Tensor>,
        output_min: &Option<Scalar>,
        output_max: &Option<Scalar>) -> Self {
    
        todo!();
        /*
        : weight(move(weight)),
        : bias(move(bias)),
        : output_min(output_min),
        : output_max(output_max),

        
        */
    }
    
    pub fn release_resources(&mut self)  {
        
        todo!();
        /*
            if (releaseCallback_) {
          releaseCallback_(opaqueOpPtr_);
          opaqueOpPtr_ = nullptr;
        }
        */
    }
    
    pub fn get_weight(&self) -> &Tensor {
        
        todo!();
        /*
            return weight_;
        */
    }
    
    pub fn get_bias(&self) -> &Option<Tensor> {
        
        todo!();
        /*
            return bias_;
        */
    }
    
    pub fn get_output_min(&self) -> &Option<Scalar> {
        
        todo!();
        /*
            return output_min_;
        */
    }
    
    pub fn get_output_max(&self) -> &Option<Scalar> {
        
        todo!();
        /*
            return output_max_;
        */
    }
    
    pub fn set_opaque_op_ptr(&mut self, ptr: *mut void)  {
        
        todo!();
        /*
            opaqueOpPtr_ = ptr;
        */
    }
    
    pub fn get_opaque_op_ptr(&self)  {
        
        todo!();
        /*
            return opaqueOpPtr_;
        */
    }
    
    pub fn set_release_callback(&mut self, func: &fn(_0: *mut void) -> ())  {
        
        todo!();
        /*
            releaseCallback_ = func;
        */
    }
    
    pub fn get_release_callback(&mut self) -> &mut fn(_0: *mut void) -> () {
        
        todo!();
        /*
            return releaseCallback_;
        */
    }
}
