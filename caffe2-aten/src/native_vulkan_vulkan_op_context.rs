crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/VulkanOpContext.h]

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
    base:        TorchJitCustomClassHolder,
    orig_weight: Tensor,
    orig_bias:   Option<Tensor>,
    stride:      Vec<i64>,
    padding:     Vec<i64>,
    dilation:    Vec<i64>,
    groups:      i64,
    output_min:  Option<Scalar>,
    output_max:  Option<Scalar>,
}

impl Conv2dOpContext {
    
    pub fn unpack(&mut self) -> SerializationTypeConv2dPrePack {
        
        todo!();
        /*
            return make_tuple(
            orig_weight_,
            orig_bias_,
            stride_,
            padding_,
            dilation_,
            groups_,
            output_min_,
            output_max_);
        */
    }
}

pub trait Conv2dOpContextInterface: RunTensor {}

pub trait RunTensor {

    fn run(&mut self, input: &Tensor) -> Tensor;
}

pub struct VulkanConv2dOpContext {
    base:       Conv2dOpContext,
    op_context: ContextConv2D,
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/VulkanOpContext.cpp]
impl VulkanConv2dOpContext {
    
    pub fn new(
        weight:     Tensor,
        bias:       Option<Tensor>,
        padding:    Vec<i64>,
        stride:     Vec<i64>,
        dilation:   Vec<i64>,
        groups:     u64,
        min:        &Option<Scalar>,
        max:        &Option<Scalar>,
        op_context: ContextConv2D) -> Self {
    
        todo!();
        /*


            : op_context_(move(op_context)) 

        orig_weight_ = move(weight);
        orig_bias_ = move(bias);
        padding_ = move(padding);
        stride_ = move(stride);
        dilation_ = move(dilation);
        groups_ = groups;
        output_min_ = min;
        output_max_ = max;
        */
    }
    
    pub fn create_context(
        weight:     Tensor,
        bias:       Option<Tensor>,
        padding:    Vec<i64>,
        stride:     Vec<i64>,
        dilation:   Vec<i64>,
        groups:     i64,
        output_min: &Option<Scalar>,
        output_max: &Option<Scalar>) -> IntrusivePtr<Conv2dOpContext> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn create_context(&mut self, 
        weight:     Tensor,
        bias:       Option<Tensor>,
        padding:    Vec<i64>,
        stride:     Vec<i64>,
        dilation:   Vec<i64>,
        groups:     i64,
        output_min: &Option<Scalar>,
        output_max: &Option<Scalar>) -> IntrusivePtr<Conv2dOpContext> {
        
        todo!();
        /*
            auto op_context = vulkan::convolution2d::create(
          weight,
          bias,
          padding,
          stride,
          dilation,
          groups,
          output_min ? output_min->to<float>() : vulkan::ContextConv2D::kMin,
          output_max ? output_max->to<float>() : vulkan::ContextConv2D::kMax);
      return make_intrusive<VulkanConv2dOpContext>(
          move(weight),
          move(bias),
          move(padding),
          move(stride),
          move(dilation),
          groups,
          output_min,
          output_max,
          move(op_context));
        */
    }
    
    pub fn run(&mut self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            return vulkan::convolution2d::run(op_context_, input);
        */
    }
}
