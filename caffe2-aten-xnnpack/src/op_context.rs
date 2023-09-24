crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/xnnpack/OpContext.h]

pub trait LinearOpContextInterface:
Run
+ FreeOrigWeightAndBias {}

pub trait FreeOrigWeightAndBias {

    fn free_orig_weight_and_bias(&mut self);
}

pub trait Conv2dOpContextInterface:
Run
+ FreeOrigWeightAndBias {}

pub trait Run {

    fn run(&mut self, input: &Tensor) -> Tensor;
}

pub trait TransposeConv2dOpContextInterface:
Run
+ FreeOrigWeightAndBias {}

pub type SerializationTypeLinearPrePack = (
    Tensor,
    Option<Tensor>,
    Option<Scalar>,
    Option<Scalar>
);

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

pub type SerializationTypeTransposeConv2dPrePack = (
    Tensor,
    Option<Tensor>,
    Vec<i64>,
    Vec<i64>,
    Vec<i64>,
    Vec<i64>,
    i64,
    Option<Scalar>,
    Option<Scalar>
);

pub struct LinearOpContext {
    base:                       CustomClassHolder,
    orig_weight:                Tensor,
    orig_bias:                  Option<Tensor>,
    output_min:                 Option<Scalar>,
    output_max:                 Option<Scalar>,
    orig_weight_and_bias_freed: bool,
}

impl LinearOpContext {
    
    pub fn unpack(&mut self) -> SerializationTypeLinearPrePack {
        
        todo!();
        /*
            TORCH_CHECK(!orig_weight_and_bias_freed_, "Original weight and bias have been freed");
        return make_tuple(orig_weight_, orig_bias_, output_min_, output_max_);
        */
    }
}

pub struct XNNPackLinearOpContext {
    base:       LinearOpContext,
    op_context: ContextLinear,
}

impl XNNPackLinearOpContext {
    
    pub fn new(
        weight:     Tensor,
        bias:       Option<Tensor>,
        min:        &Option<Scalar>,
        max:        &Option<Scalar>,
        op_context: ContextLinear) -> Self {
    
        todo!();
        /*
        : op_context(move(op_context)),

            orig_weight_ = move(weight);
        orig_bias_ = move(bias);
        output_min_ = min;
        output_max_ = max;
        orig_weight_and_bias_freed_ = false;
        */
    }
    
    pub fn run(&mut self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
        
        */
    }
    
    pub fn free_orig_weight_and_bias(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn create_context(
        weight:     Tensor,
        bias:       Option<Tensor>,
        output_min: &Option<Scalar>,
        output_max: &Option<Scalar>) -> IntrusivePtr<LinearOpContext> {
        
        todo!();
        /*
        
        */
    }
}

pub struct Conv2dOpContext {
    base:                       CustomClassHolder,
    orig_weight:                Tensor,
    orig_bias:                  Option<Tensor>,
    stride:                     Vec<i64>,
    padding:                    Vec<i64>,
    dilation:                   Vec<i64>,
    groups:                     i64,
    output_min:                 Option<Scalar>,
    output_max:                 Option<Scalar>,
    orig_weight_and_bias_freed: bool,
}

impl Conv2dOpContext {
    
    pub fn unpack(&mut self) -> SerializationTypeConv2dPrePack {
        
        todo!();
        /*
            TORCH_CHECK(!orig_weight_and_bias_freed_, "Original weight and bias have been freed");
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

pub struct TransposeConv2dOpContext {
    base:                       CustomClassHolder,
    orig_weight:                Tensor,
    orig_bias:                  Option<Tensor>,
    stride:                     Vec<i64>,
    padding:                    Vec<i64>,
    output_padding:             Vec<i64>,
    dilation:                   Vec<i64>,
    groups:                     i64,
    output_min:                 Option<Scalar>,
    output_max:                 Option<Scalar>,
    orig_weight_and_bias_freed: bool,
}

impl TransposeConv2dOpContext {
    
    pub fn unpack(&mut self) -> SerializationTypeTransposeConv2dPrePack {
        
        todo!();
        /*
            TORCH_CHECK(!orig_weight_and_bias_freed_, "Original weight and bias have been freed");
        return make_tuple(
            orig_weight_,
            orig_bias_,
            stride_,
            padding_,
            output_padding_,
            dilation_,
            groups_,
            output_min_,
            output_max_);
        */
    }
}



pub struct XNNPackConv2dOpContext {
    base:       Conv2dOpContext,
    op_context: ContextConv2D,
}

impl XNNPackConv2dOpContext {
    
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
        : op_context(move(op_context)),

            orig_weight_ = move(weight);
        orig_bias_ = move(bias);
        padding_ = move(padding);
        stride_ = move(stride);
        dilation_ = move(dilation);
        groups_ = groups;
        output_min_ = min;
        output_max_ = max;
        orig_weight_and_bias_freed_ = false;
        */
    }
}

pub struct XNNPackTransposeConv2dOpContext {
    base:       TransposeConv2dOpContext,
    op_context: ContextConv2D,
}

impl XNNPackTransposeConv2dOpContext {
    
    pub fn new(
        weight:         Tensor,
        bias:           Option<Tensor>,
        padding:        Vec<i64>,
        output_padding: Vec<i64>,
        stride:         Vec<i64>,
        dilation:       Vec<i64>,
        groups:         u64,
        min:            &Option<Scalar>,
        max:            &Option<Scalar>,
        op_context:     ContextConv2D) -> Self {
    
        todo!();
        /*
        : op_context(move(op_context)),

            orig_weight_ = move(weight);
        orig_bias_ = move(bias);
        padding_ = move(padding);
        output_padding_ = move(output_padding);
        stride_ = move(stride);
        dilation_ = move(dilation);
        groups_ = groups;
        output_min_ = min;
        output_max_ = max;
        orig_weight_and_bias_freed_ = false;
        */
    }
    
    pub fn run(&mut self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
        
        */
    }
    
    pub fn free_orig_weight_and_bias(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn create_context(
        weight:         Tensor,
        bias:           Option<Tensor>,
        padding:        Vec<i64>,
        output_padding: Vec<i64>,
        stride:         Vec<i64>,
        dilation:       Vec<i64>,
        groups:         i64,
        output_min:     &Option<Scalar>,
        output_max:     &Option<Scalar>) -> IntrusivePtr<TransposeConv2dOpContext> {
        
        todo!();
        /*
        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/xnnpack/OpContext.cpp]

impl XNNPackLinearOpContext {
    
    pub fn create_context(&mut self, 
        weight:     Tensor,
        bias:       Option<Tensor>,
        output_min: &Option<Scalar>,
        output_max: &Option<Scalar>) -> IntrusivePtr<LinearOpContext> {
        
        todo!();
        /*
            auto linear_op_context =
          make_intrusive<XNNPackLinearOpContext>(
              move(weight),
              move(bias),
              output_min,
              output_max,
              xnnpack::internal::linear::create(
                  weight,
                  bias,
                  output_min ? output_min->to<float>()
                             : xnnpack::ContextLinear::kMin,
                  output_max ? output_max->to<float>()
                             : xnnpack::ContextLinear::kMax)
              );
      if (globalContext().releaseWeightsWhenPrepacking()) {
        linear_op_context->free_orig_weight_and_bias();
      }

      return linear_op_context;
        */
    }
    
    pub fn free_orig_weight_and_bias(&mut self)  {
        
        todo!();
        /*
            orig_weight_and_bias_freed_ = true;
      orig_weight_.reset();
      orig_bias_.reset();
        */
    }
    
    pub fn run(&mut self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            return xnnpack::internal::linear::run(op_context_, input);
        */
    }
}

impl XNNPackConv2dOpContext {
    
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
            auto op_context =
          xnnpack::internal::convolution2d::create(
              weight,
              bias,
              padding,
              {0, 0}, // output_padding
              stride,
              dilation,
              groups,
              false,  // transposed
              output_min ? output_min->to<float>()
                         : xnnpack::ContextConv2D::kMin,
              output_max ? output_max->to<float>()
                         : xnnpack::ContextConv2D::kMax);

      auto conv2d_op_context =
          make_intrusive<XNNPackConv2dOpContext>(
              move(weight),
              move(bias),
              move(padding),
              move(stride),
              move(dilation),
              groups,
              output_min,
              output_max,
              move(op_context));

      if (globalContext().releaseWeightsWhenPrepacking()) {
        conv2d_op_context->free_orig_weight_and_bias();
      }

      return conv2d_op_context;
        */
    }
}

impl XNNPackTransposeConv2dOpContext {
    
    pub fn create_context(&mut self, 
        weight:         Tensor,
        bias:           Option<Tensor>,
        padding:        Vec<i64>,
        output_padding: Vec<i64>,
        stride:         Vec<i64>,
        dilation:       Vec<i64>,
        groups:         i64,
        output_min:     &Option<Scalar>,
        output_max:     &Option<Scalar>) -> IntrusivePtr<TransposeConv2dOpContext> {
        
        todo!();
        /*
            auto op_context =
          xnnpack::internal::convolution2d::create(
              weight,
              bias,
              padding,
              output_padding,
              stride,
              dilation,
              groups,
              true, // transposed
              output_min ? output_min->to<float>()
                         : xnnpack::ContextConv2D::kMin,
              output_max ? output_max->to<float>()
                         : xnnpack::ContextConv2D::kMax);

      auto conv2d_op_context =
          make_intrusive<XNNPackTransposeConv2dOpContext>(
              move(weight),
              move(bias),
              move(padding),
              move(output_padding),
              move(stride),
              move(dilation),
              groups,
              output_min,
              output_max,
              move(op_context));

      if (globalContext().releaseWeightsWhenPrepacking()) {
        conv2d_op_context->free_orig_weight_and_bias();
      }

      return conv2d_op_context;
        */
    }
}

impl XNNPackConv2dOpContext {
    
    pub fn run(&mut self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            return xnnpack::internal::convolution2d::run(op_context_, input);
        */
    }
}

impl XNNPackTransposeConv2dOpContext {
    
    pub fn run(&mut self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            return xnnpack::internal::convolution2d::run(op_context_, input);
        */
    }
}

impl XNNPackConv2dOpContext {
    
    pub fn free_orig_weight_and_bias(&mut self)  {
        
        todo!();
        /*
            orig_weight_and_bias_freed_ = true;
      orig_weight_.reset();
      orig_bias_.reset();
        */
    }
}

impl XNNPackTransposeConv2dOpContext {
    
    pub fn free_orig_weight_and_bias(&mut self)  {
        
        todo!();
        /*
            orig_weight_and_bias_freed_ = true;
      orig_weight_.reset();
      orig_bias_.reset();
        */
    }
}
