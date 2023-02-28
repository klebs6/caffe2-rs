crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/q_avgpool.cpp]

define_dispatch!{qavg_pool2d_nhwc_stub}

pub fn avg_pool2d_out_frame<Scalar>(
        input:             &Tensor,
        output:            &mut Tensor,
        b:                 i64,
        n_input_plane:     i64,
        input_width:       i64,
        input_height:      i64,
        output_width:      i64,
        output_height:     i64,
        kw:                i32,
        kh:                i32,
        dw:                i32,
        dh:                i32,
        padw:              i32,
        padh:              i32,
        count_include_pad: bool,
        divisor_override:  Option<i64>)  {

    todo!();
        /*
            parallel_for(0, nInputPlane, 0, [&](i64 start, i64 end) {
        for (auto k = start; k < end; k++) {
          i64 xx, yy;
          /* For all output pixels... */
          Tensor input_contig = input.contiguous();
          auto input_data = input_contig.data_ptr<Scalar>();
          auto output_data = output.data_ptr<Scalar>();
          Scalar* ptr_output = output_data +
              b * nInputPlane * outputWidth * outputHeight +
              k * outputWidth * outputHeight;
          const Scalar* ptr_input = input_data +
              b * nInputPlane * inputWidth * inputHeight +
              k * inputWidth * inputHeight;
          auto minimum =
              numeric_limits<typename Scalar::underlying>::lowest();
          auto maximum = numeric_limits<typename Scalar::underlying>::max();

          for (yy = 0; yy < outputHeight; yy++) {
            for (xx = 0; xx < outputWidth; xx++) {
              /* Compute the mean of the input image... */
              i64 hstart = yy * dH - padH;
              i64 wstart = xx * dW - padW;
              i64 hend = min(hstart + kH, inputHeight + padH);
              i64 wend = min(wstart + kW, inputWidth + padW);
              i64 pool_size = (hend - hstart) * (wend - wstart);
              hstart = max(hstart, (i64)0);
              wstart = max(wstart, (i64)0);
              hend = min(hend, inputHeight);
              wend = min(wend, inputWidth);

              int sum_int = 0;
              ptr_output->val_ = 0;

              i64 divide_factor;
              i64 size = (hend - hstart) * (wend - wstart);
              if (divisor_override.has_value()) {
                divide_factor = divisor_override.value();
              } else {
                if (count_include_pad) {
                  divide_factor = pool_size;
                } else {
                  divide_factor = (hend - hstart) * (wend - wstart);
                }
              }

              i64 kx, ky;
              for (ky = hstart; ky < hend; ky++) {
                for (kx = wstart; kx < wend; kx++)
                  sum_int += (ptr_input + ky * inputWidth + kx)->val_;
              }
              float multiplier = input.q_scale() / output.q_scale() / divide_factor;

              sum_int -= size * input.q_zero_point();
              float sum = sum_int * 1.0;
              /* Update output by requantizing the result */
              ptr_output->val_ =
                  static_cast<typename Scalar::underlying>(min<i32>(
                      max<i32>(
                          nearbyint(sum * multiplier + output.q_zero_point()),
                          minimum),
                      maximum));
              ptr_output++;
            }
          }
        }
      });
        */
}

#[inline] pub fn get_kernel(kernel_size: &[i32]) -> (i32,i32) {
    
    todo!();
        /*
            TORCH_CHECK(
          kernel_size.size() == 1 || kernel_size.size() == 2,
          "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
      const int kH = safe_downcast<int, i64>(kernel_size[0]);
      const int kW = kernel_size.size() == 1
          ? kH
          : safe_downcast<int, i64>(kernel_size[1]);
      return make_pair(kW, kH);
        */
}

#[inline] pub fn get_stride(
        stride: &[i32],
        kw:     i32,
        kh:     i32) -> (i32,i32) {
    
    todo!();
        /*
            TORCH_CHECK(
          stride.empty() || stride.size() == 1 || stride.size() == 2,
          "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
      const int dH = stride.empty() ? kH : safe_downcast<int, i64>(stride[0]);
      const int dW = stride.empty()
          ? kW
          : stride.size() == 1 ? dH : safe_downcast<int, i64>(stride[1]);
      return make_pair(dW, dH);
        */
}

#[inline] pub fn get_padding(padding: &[i32]) -> (i32,i32) {
    
    todo!();
        /*
            TORCH_CHECK(
          padding.size() == 1 || padding.size() == 2,
          "avg_pool2d: padding must either be a single int, or a tuple of two ints");
      const int padH = safe_downcast<int, i64>(padding[0]);
      const int padW =
          padding.size() == 1 ? padH : safe_downcast<int, i64>(padding[1]);
      return make_pair(padW, padH);
        */
}

pub fn get_output_shape(
        input:     &Tensor,
        kw:        i32,
        kh:        i32,
        dw:        i32,
        dh:        i32,
        padw:      i32,
        padh:      i32,
        ceil_mode: bool) -> Vec<i64> {
    
    todo!();
        /*
            const i64 nbatch = input_.ndimension() == 4 ? input_.size(-4) : 1;
      const i64 nInputPlane = input_.size(-3);
      const i64 inputHeight = input_.size(-2);
      const i64 inputWidth = input_.size(-1);
      const i64 outputHeight =
          pooling_output_shape<i64>(inputHeight, kH, padH, dH, 1, ceil_mode);
      const i64 outputWidth =
          pooling_output_shape<i64>(inputWidth, kW, padW, dW, 1, ceil_mode);
      if (input_.ndimension() == 3) {
        return {nInputPlane, outputHeight, outputWidth};
      }
      return {nbatch, nInputPlane, outputHeight, outputWidth};
        */
}

pub fn q_avg_pool2d<Scalar>(
        input:             &Tensor,
        kernel_size:       &[i32],
        stride:            &[i32],
        padding:           &[i32],
        ceil_mode:         bool,
        count_include_pad: bool,
        divisor_override:  Option<i64>) -> Tensor {

    todo!();
        /*
      int kW, kH, dW, dH, padW, padH;
      tie(kW, kH) = get_kernel(kernel_size);
      tie(dW, dH) = get_stride(stride, kW, kH);
      tie(padW, padH) = get_padding(padding);

      const i64 nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
      const i64 nInputPlane = input.size(-3);
      const i64 inputHeight = input.size(-2);
      const i64 inputWidth = input.size(-1);

      TORCH_CHECK(
          !divisor_override.has_value() || divisor_override.value() != 0,
          "divisor must be not zero");

      auto output_shape =
          get_output_shape(input, kW, kH, dW, dH, padW, padH, ceil_mode);
      const i64 outputHeight = output_shape[output_shape.size() - 2];
      const i64 outputWidth = output_shape[output_shape.size() - 1];
      if (input.is_contiguous(MemoryFormat::ChannelsLast)) {
        auto output = _empty_affine_quantized(
            output_shape,
            input.options().memory_format(input.suggest_memory_format()),
            input.q_scale(),
            input.q_zero_point(),
            nullopt);
        // fast path for channel last: qavg_pool_2d_nhwc_stub
        if (output_shape.size() == 3) {
          qavg_pool2d_nhwc_stub(
              input.device().type(),
              input,
              output,
              0,
              nInputPlane,
              inputWidth,
              inputHeight,
              outputWidth,
              outputHeight,
              kW,
              kH,
              dW,
              dH,
              padW,
              padH,
              count_include_pad,
              divisor_override);
        } else {
          parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
            for (auto b = start; b < end; b++) {
              qavg_pool2d_nhwc_stub(
                  input.device().type(),
                  input,
                  output,
                  b,
                  nInputPlane,
                  inputWidth,
                  inputHeight,
                  outputWidth,
                  outputHeight,
                  kW,
                  kH,
                  dW,
                  dH,
                  padW,
                  padH,
                  count_include_pad,
                  divisor_override);
            }
          });
        }
        return output;
      } else {
        auto output = _empty_affine_quantized(
            output_shape, input.options(), input.q_scale(), input.q_zero_point());
        if (output_shape.size() == 3) {
          avg_pool2d_out_frame<Scalar>(
              input,
              output,
              0,
              nInputPlane,
              inputWidth,
              inputHeight,
              outputWidth,
              outputHeight,
              kW,
              kH,
              dW,
              dH,
              padW,
              padH,
              count_include_pad,
              divisor_override);
        } else {
          parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
            for (auto b = start; b < end; b++) {
              avg_pool2d_out_frame<Scalar>(
                  input,
                  output,
                  b,
                  nInputPlane,
                  inputWidth,
                  inputHeight,
                  outputWidth,
                  outputHeight,
                  kW,
                  kH,
                  dW,
                  dH,
                  padW,
                  padH,
                  count_include_pad,
                  divisor_override);
            }
          });
        }
        return output;
      }
        */
}

#[cfg(USE_PYTORCH_QNNPACK)]
pub fn qnnpack_avg_pool2d(
        input:             Tensor,
        kernel_size:       &[i32],
        stride:            &[i32],
        padding:           &[i32],
        ceil_mode:         bool,
        count_include_pad: bool,
        divisor_override:  Option<i64>) -> Tensor {
    
    todo!();
        /*
            Tensor output;
      int kW, kH, dW, dH, padW, padH;
      tie(kW, kH) = get_kernel(kernel_size);
      tie(dW, dH) = get_stride(stride, kW, kH);
      tie(padW, padH) = get_padding(padding);
      TORCH_CHECK(
          input.ndimension() == 4,
          "qnnpack_avg_pool2d(): Expected input to be 4-dimensional: got ",
          input.ndimension());

      i64 batch_size = input.size(0);
      i64 inC = input.size(1);
      i64 inH = input.size(2);
      i64 inW = input.size(3);
      auto output_shape =
          get_output_shape(input, kW, kH, dW, dH, padW, padH, ceil_mode);
      const i64 oH = output_shape[output_shape.size() - 2];
      const i64 oW = output_shape[output_shape.size() - 1];
      const auto outC = inC;

      Tensor input_contig = input.contiguous(MemoryFormat::ChannelsLast);

      initQNNPACK();
      const auto scale = input_contig.q_scale();
      const auto zero_point = input_contig.q_zero_point();

      TORCH_CHECK(
          oH > 0 && oW > 0,
          "qnnpack_avg_pool2d(): the resulting output Tensor size should be >= 0");
      // NHWC output
      output = _empty_affine_quantized(
          output_shape,
          device(kCPU).dtype(kQUInt8),
          scale,
          zero_point,
          MemoryFormat::ChannelsLast);

      pytorch_qnnp_operator_t qnnpack_operator{nullptr};
      const pytorch_qnnp_status createStatus =
          pytorch_qnnp_create_average_pooling2d_nhwc_q8(
              padH /* input_padding_top */,
              padW /* input_padding_right */,
              padH /* input_padding_bottom */,
              padW /* input_padding_left */,
              kH /* kernel height */,
              kW /* kernel width */,
              dH /* stride height */,
              dW /* stride width */,
              inC /* input channels */,
              zero_point /* input zero_point */,
              scale /* input scale */,
              zero_point /* output zero_point */,
              scale /* output scale */,
              u8::min /* output min */,
              u8::max /* output max */,
              0 /* flags */,
              &qnnpack_operator);
      CAFFE_ENFORCE(
          createStatus == pytorch_qnnp_status_success,
          "failed to create QNNPACK Average Pooling operator");
      unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
          qnnpack_uniq_ptr(qnnpack_operator);

      const pytorch_qnnp_status setupStatus =
          pytorch_qnnp_setup_average_pooling2d_nhwc_q8(
              qnnpack_operator,
              batch_size,
              inH,
              inW,
              (u8*)input_contig.data_ptr<quint8>() /* input data */,
              inC,
              (u8*)output.data_ptr<quint8>() /* output data */,
              outC,
              nullptr /* thread pool */);
      CAFFE_ENFORCE(
          setupStatus == pytorch_qnnp_status_success,
          "failed to setup QNNPACK Average Pooling operator");
      pthreadpool_t threadpool = pthreadpool_();
      const pytorch_qnnp_status runStatus =
          pytorch_qnnp_run_operator(qnnpack_operator, threadpool);
      TORCH_INTERNAL_ASSERT(
          runStatus == pytorch_qnnp_status_success,
          "failed to run QNNPACK Average Pool operator");
      return output.contiguous(input.suggest_memory_format());
        */
}

pub fn avg_pool2d_quantized_cpu(
        input:             &Tensor,
        kernel_size:       &[i32],
        stride:            &[i32],
        padding:           &[i32],
        ceil_mode:         bool,
        count_include_pad: bool,
        divisor_override:  Option<i64>) -> Tensor {
    
    todo!();
        /*
            Tensor output;
    #ifdef USE_PYTORCH_QNNPACK
      if (globalContext().qEngine() == QEngine::QNNPACK &&
          input.scalar_type() == kQUInt8) {
        return native::qnnp_avgpool_helper::qnnpack_avg_pool2d(
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override);
      }
    #endif
      AT_DISPATCH_QINT_TYPES(input.scalar_type(), "avg_pool2d_quantized_cpu", [&]() {
        output = q_avg_pool2d<Scalar>(
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override);
      });
      return output;
        */
}
