crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/VulkanConvolution.h]

pub struct Conv2DParams {

    /**
      | batch size
      |
      */
    N:    i64,

    /**
      | channels
      |
      */
    C:    i64,

    /**
      | input height
      |
      */
    H:    i64,

    /**
      | input width
      |
      */
    W:    i64,

    /**
      | output channels
      |
      */
    OC:   i64,

    /**
      | kernel height
      |
      */
    KH:   i64,

    /**
      | kernel width
      |
      */
    KW:   i64,

    /**
      | stride y (height)
      |
      */
    SY:   i64,

    /**
      | stride x (width)
      |
      */
    SX:   i64,

    /**
      | padding y (height)
      |
      */
    PY:   i64,

    /**
      | padding x (width)
      |
      */
    PX:   i64,

    /**
      | dilation y (height)
      |
      */
    DY:   i64,

    /**
      | dilation x (width)
      |
      */
    DX:   i64,

    /**
      | groups
      |
      */
    G:    i64,

    /**
      | output width
      |
      */
    OW:   i64,

    /**
      | output height
      |
      */
    OH:   i64,

    OC_4: i64,
    C_4:  i64,
}

impl Conv2DParams {
    
    pub fn new(
        input_sizes: &[i32],
        OC:          i64,
        KH:          i64,
        KW:          i64,
        SY:          i64,
        SX:          i64,
        PY:          i64,
        PX:          i64,
        DY:          i64,
        DX:          i64,
        G:           i64) -> Self {
    
        todo!();
        /*


            // TODO: What if inputSizes is not of the expected dimensionality?
        // Should check prior to indexing.
          : N(inputSizes[0]),
            C(inputSizes[1]),
            H(inputSizes[2]),
            W(inputSizes[3]),
            OC(OC),
            KH(KH),
            KW(KW),
            SY(SY),
            SX(SX),
            PY(PY),
            PX(PX),
            DY(DY),
            DX(DX),
            G(G) 
        OC_4 = UP_DIV(OC, 4);
        C_4 = UP_DIV(C, 4);
        const i64 KWE = (KW - 1) * DX + 1;
        const i64 KHE = (KH - 1) * DY + 1;
        OW = ((W - KWE + 2 * PX) / SX) + 1;
        OH = ((H - KHE + 2 * PY) / SY) + 1;
        */
    }
    
    pub fn new(
        input_sizes:  &[i32],
        weight_sizes: &[i32],
        padding:      &[i32],
        stride:       &[i32],
        dilation:     &[i32],
        groups:       i64) -> Self {
    
        todo!();
        /*


            // TODO: What if these parameters are not of the correct dimensionality?
        // Should check prior to indexing.
          : Conv2DParams(
                inputSizes,
                weightSizes[0],
                weightSizes[2],
                weightSizes[3],
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                dilation[0],
                dilation[1],
                groups)
        */
    }
    
    pub fn output_sizes(&self) -> Vec<i64> {
        
        todo!();
        /*
            return {N, OC, OH, OW};
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/VulkanConvolution.cpp]

/// TODO: This function is not used.
pub fn available(
        weight:     &Tensor,
        bias:       &Option<Tensor>,
        padding:    &[i32],
        stride:     &[i32],
        dilation:   &[i32],
        groups:     i64,
        output_min: f32,
        output_max: f32) -> bool {
    
    todo!();
        /*
            return native::is_vulkan_available() && (4 == weight.ndimension()) &&
          (Backend::CPU == weight.options().backend()) &&
          (kFloat == weight.scalar_type());
        */
}

pub fn create_conv2d_clamp_pre_pack_op_context(
        weight:     Tensor,
        bias:       Option<Tensor>,
        stride:     Vec<i64>,
        padding:    Vec<i64>,
        dilation:   Vec<i64>,
        groups:     i64,
        output_min: &Option<Scalar>,
        output_max: &Option<Scalar>) -> IntrusivePtr<Conv2dOpContext> {
    
    todo!();
        /*
            return vulkan::VulkanConv2dOpContext::create_context(
          move(weight),
          move(bias),
          move(padding),
          move(stride),
          move(dilation),
          groups,
          output_min,
          output_max);
        */
}

pub fn conv2d_clamp_run(
        input:      &Tensor,
        op_context: &IntrusivePtr<Conv2dOpContext>) -> Tensor {
    
    todo!();
        /*
            return op_context->run(input);
        */
}

pub fn create(
        weight:     &Tensor,
        bias:       &Option<Tensor>,
        padding:    &[i32],
        stride:     &[i32],
        dilation:   &[i32],
        groups:     i64,
        output_min: f32,
        output_max: f32) -> ContextConv2D {
    
    todo!();
        /*
            const auto padding_expanded = expand_param_if_needed(padding, "padding", 2);
      const auto stride_expanded = expand_param_if_needed(stride, "stride", 2);
      const auto dilation_expanded =
          expand_param_if_needed(dilation, "dilation", 2);
      const Tensor weight_nchw = weight.contiguous();
      const auto ws = weight_nchw.sizes();
      return ContextConv2D{
          groups == 1 ? native::vulkan::convolution_prepack_weights(weight_nchw)
                      : weight_nchw.vulkan(),
          bias.has_value() ? make_optional((*bias).vulkan()) : nullopt,
          // TODO: Are we sure these tensors will always come into this fucntion with the
          // the dimensions expected below? What if they don't?  This may trigger a segfault.
          // TODO: If we need TORCH_CHECK(available()) calls here as a sanity check, add it.
          {{ws[0], ws[1], ws[2], ws[3]}},
          {padding_expanded[0], padding_expanded[1]},
          {stride_expanded[0], stride_expanded[1]},
          {dilation_expanded[0], dilation_expanded[1]},
          groups,
          output_min,
          output_max};
        */
}

pub fn run(
        context: &ContextConv2D,
        input:   &Tensor) -> Tensor {
    
    todo!();
        /*
            return native::vulkan::convolution_prepacked(
          input,
          context.weight_size_,
          context.weight_prepacked_vulkan_,
          context.bias_vulkan_,
          context.padding_,
          context.stride_,
          context.dilation_,
          context.groups_,
          context.output_min_,
          context.output_max_);
        */
}
