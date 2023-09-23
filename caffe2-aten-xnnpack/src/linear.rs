crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/xnnpack/Linear.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/xnnpack/Linear.cpp]

/**
  | Supports NHWC and NCHW FP32 linear operators.
  |
  | TODO: Decouple and improve error handling
  | and messages.
  |
  */
pub fn available(
    weight:     &Tensor,
    bias:       &Option<Tensor>,
    output_min: f32,
    output_max: f32) -> bool {

    todo!();
        /*
            // XNNPACK
      return xnnpack::internal::available() &&
              // Weight
              (2 == weight.ndimension()) &&
              (weight.device().is_cpu()) &&
              (kFloat == weight.scalar_type()) &&
              !weight.requires_grad() &&
              // Bias
              ((bias && bias->defined()) ? ((1 == bias->ndimension()) &&
                                           (bias->device().is_cpu()) &&
                                           (kFloat == bias->scalar_type()) &&
                                           (weight.size(Layout::Filter::output)) == bias->size(0) &&
                                           !bias->requires_grad())
                                         : true) &&
              // Output Min / Max
              (output_max > output_min) &&
              true;
        */
}

/**
  | TODO: Decouple and improve error handling
  | and messages.
  |
  */
pub fn usable(input: &Tensor) -> bool {
    
    todo!();
        /*
            // Input
      return (1 <= input.ndimension()) &&
             (input.device().is_cpu()) &&
             (kFloat == input.scalar_type()) &&
             !input.requires_grad() &&
             true;
        */
}

pub fn create_and_run(
    input:      &Tensor,
    weight:     &Tensor,
    bias:       &Tensor,
    output_min: f32,
    output_max: f32) -> Tensor {

    todo!();
        /*
            return run(
          create(
              weight,
              bias,
              output_min,
              output_max),
          input);
        */
}

pub fn create(
    weight:     &Tensor,
    bias:       &Option<Tensor>,
    output_min: f32,
    output_max: f32) -> ContextLinear {

    todo!();
        /*
            const Tensor weight_contig = weight.contiguous();

      TORCH_CHECK(
            available(
              weight_contig,
              bias,
              output_min,
              output_max),
          "XNNPACK Linear not available! "
          "Reason: The provided (weight, bias, output_min, output_max) parameters are "
          "either invalid individually or their combination is not supported by XNNPACK.");

      xnn_operator_t linear_op{};

      const xnn_status create_status = xnn_create_fully_connected_nc_f32(
          weight_contig.size(Layout::Filter::input),                        // input_channels
          weight_contig.size(Layout::Filter::output),                       // output_channels
          weight_contig.size(Layout::Filter::input),                        // input_pixel_stride
          weight_contig.size(Layout::Filter::output),                       // output_pixel_stride
          weight_contig.data_ptr<float>(),                                  // kernel
          (bias && bias->defined()) ?
              bias->contiguous().data_ptr<float>() :
              nullptr,                                                      // bias
          output_min,                                                     // output_min
          output_max,                                                     // output_max
          0u,                                                             // flags
          &linear_op);                                                    // operator

      TORCH_CHECK(
          xnn_status_success == create_status,
          "xnn_create_fully_connected_nc_f32 failed!");

      return ContextLinear(
        Operator(linear_op),
        weight_contig.size(Layout::Filter::output)
      );
        */
}

pub fn run(
    context: &ContextLinear,
    input:   &Tensor) -> Tensor {

    todo!();
        /*
            using namespace internal;

      // For compatibility with linear
      auto ip = input;
      if (input.ndimension() == 1) {
        ip = input.unsqueeze(0);
      }

      const Tensor padded_input = mobile::allocate_padded_contiguous_if_needed(
          ip, ip.suggest_memory_format());

      TORCH_CHECK(
          usable(padded_input),
          "XNNPACK Linear not usable! "
          "Reason: The provided input tensor is either invalid or unsupported by XNNPACK.");

      const IntArrayRef input_size = padded_input.sizes();
      vector<i64> output_size(input_size.cbegin(), input_size.cend());
      output_size.back() = context.output_channels;

      Tensor output = mobile::empty_with_tail_padding(
          output_size,
          padded_input.options().dtype(),
          padded_input.suggest_memory_format(),
          padded_input.names());

      const xnn_status setup_status = xnn_setup_fully_connected_nc_f32(
          context.op.get(),                                   // operator
          Layout::ActivationND::batch(padded_input.sizes()),  // Batch,
          padded_input.data_ptr<float>(),                     // input
          output.data_ptr<float>(),                           // output
          pthreadpool_());                            // threadpool

      TORCH_CHECK(
          xnn_status_success == setup_status,
          "xnn_setup_fully_connected_nc_f32 failed!");

      const xnn_status run_status = xnn_run_operator(
          context.op.get(),         // operator
          pthreadpool_());  // threadpool

      TORCH_INTERNAL_ASSERT(
          xnn_status_success == run_status,
          "xnn_run_operator failed!");

      // For compatibility with linear
      if (input.ndimension() == 1) {
          output.squeeze_(0);
      }

      return output;
        */
}

pub fn create_linear_clamp_pre_pack_op_context(
    weight:     Tensor,
    bias:       Option<Tensor>,
    output_min: &Option<Scalar>,
    output_max: &Option<Scalar>) -> IntrusivePtr<LinearOpContext> {
    
    todo!();
        /*
            return xnnpack::XNNPackLinearOpContext::create_context(
          move(weight), move(bias), output_min, output_max);
        */
}

pub fn linear_clamp_run(
    input:      &Tensor,
    op_context: &IntrusivePtr<LinearOpContext>) -> Tensor {
    
    todo!();
        /*
            return op_context->run(input);
        */
}

pub fn use_linear(
    input:  &Tensor,
    weight: &Tensor,
    bias:   &Tensor) -> bool {
    
    todo!();
        /*
            return internal::linear::available(
                weight,
                bias,
                ContextLinear::kMin,
                ContextLinear::kMax) &&
             internal::linear::usable(input);
          internal::linear::usable(input);
        */
}

pub fn linear(
    input:  &Tensor,
    weight: &Tensor,
    bias:   &Tensor) -> Tensor {
    
    todo!();
        /*
            return internal::linear::create_and_run(
          input,
          weight,
          bias,
          ContextLinear::kMin,
          ContextLinear::kMax);
        */
}
