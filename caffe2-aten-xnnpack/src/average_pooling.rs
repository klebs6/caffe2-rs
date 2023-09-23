// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/xnnpack/AveragePooling.cpp]

pub fn use_global_average_pool(input: &Tensor) -> bool {
    
    todo!();
        /*
            return xnnpack::internal::available() &&
              (1 <= input.ndimension()) &&
              (input.device().is_cpu()) &&
              (kFloat == input.scalar_type()) &&
              !input.requires_grad() &&
               true;
        */
}


pub fn global_average_pool(input: &Tensor) -> Tensor {
    
    todo!();
        /*
            using namespace internal;

      const Tensor input_padded_contig_nhwc =
          mobile::allocate_padded_contiguous_if_needed(
              input, MemoryFormat::ChannelsLast);

      Tensor output = mobile::empty_with_tail_padding(
          {
            input_padded_contig_nhwc.size(Layout::Activation4D::batch),
            input_padded_contig_nhwc.size(Layout::Activation4D::channels),
            1,
            1,
          },
          input_padded_contig_nhwc.options().dtype(),
          MemoryFormat::ChannelsLast,
          input_padded_contig_nhwc.names());

      xnn_operator_t global_average_pooling_op{};
      const xnn_status create_status = xnn_create_global_average_pooling_nwc_f32(
        input_padded_contig_nhwc.size(Layout::Activation4D::channels), // channels
        input_padded_contig_nhwc.size(
            Layout::Activation4D::channels), // input stride
        input_padded_contig_nhwc.size(
            Layout::Activation4D::channels), // output stride
        -numeric_limits<float>::infinity(),
        numeric_limits<float>::infinity(),
        0 /* flags */,
        &global_average_pooling_op);

      TORCH_CHECK(
        xnn_status_success == create_status,
        "xnn_create_global_average_pooling_nwc_f32 failed!");

      Operator global_avg_pool_scoped_op(global_average_pooling_op);

      const xnn_status setup_status = xnn_setup_global_average_pooling_nwc_f32(
          global_average_pooling_op,
          input_padded_contig_nhwc.size(Layout::Activation4D::batch), // batch_size
          input_padded_contig_nhwc.size(Layout::Activation4D::width) *
              input_padded_contig_nhwc.size(Layout::Activation4D::height), // width
          input_padded_contig_nhwc.data_ptr<float>(),
          output.data_ptr<float>(),
          pthreadpool_());

      TORCH_CHECK(
        xnn_status_success == setup_status,
        "xnn_setup_global_average_pooling_nwc_f32 failed!");

      const xnn_status run_status = xnn_run_operator(
        global_average_pooling_op,
        pthreadpool_());

      TORCH_CHECK(
        xnn_status_success == run_status,
        "xnn_setup_global_average_pooling_nwc_f32 failed!");

      return output.to(input.suggest_memory_format());
        */
}
