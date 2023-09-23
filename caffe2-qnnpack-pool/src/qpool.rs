crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qpool.cpp]

define_dispatch!{qmaxpool_2d_nhwc_stub}

/**
  | Computes the spatial 2D max pooling
  | with dilation.
  | 
  | Argument description in the argument
  | list.
  |
  */
pub fn spatial_dilated_max_pooling<T>(
        i_data: *const T,

        // input/output channels
        ic:     i64,

        // input sizes
        ih:     i64,
        iw:     i64,

        // output sizes
        oh:     i64,
        ow:     i64,

        // kernel size
        kh:     i64,
        kw:     i64,

        // strides
        sh:     i64,
        sw:     i64,

        // padding
        ph:     i64,
        pw:     i64,

        // dilation
        dh:     i64,
        dw:     i64,

        o_data: *mut T)  {

    todo!();
        /*
            // output arrays (data and max-index)
      parallel_for(0, iC, 0, [&](i64 start, i64 end) {
        for (auto p = start; p < end; ++p) {
          i64 row, col;
          const T* i_p = iData + p * iW * iH;
          for (row = 0; row < oH; ++row) {
            for (col = 0; col < oW; ++col) {
              i64 h_start = row * sH - pH;
              i64 w_start = col * sW - pW;
              i64 h_end = min(h_start + (kH - 1) * dH + 1, iH);
              i64 w_end = min(w_start + (kW - 1) * dW + 1, iW);
              while (h_start < 0)
                h_start += dH;
              while (w_start < 0)
                w_start += dW;

              // local pointers
              T* o_p = oData + p * oW * oH + row * oW + col;

              // local max
              auto max_val = numeric_limits<typename T::underlying>::lowest();
              i64 tcntr = 0; // center point
              i64 x, y;
              for (y = h_start; y < h_end; y += dH) {
                for (x = w_start; x < w_end; x += dW) {
                  tcntr = y * iW + x;
                  auto val = (i_p + tcntr)->val_;
                  if (val > max_val) {
                    max_val = val;
                  }
                }
              }
              *o_p = T(max_val); // Output.
            }
          }
        }
      });
        */
}

pub fn q_maxpool_2d<Q>(

    // Input Tensor (Quantized)
    qx:        Tensor,

    // kernel size
    kh:        i64,
    kw:        i64,

    // strides
    sh:        i64,
    sw:        i64,

    // padding
    ph:        i64,
    pw:        i64,

    dh:        i64,
    dw:        i64,
    ceil_mode: bool) -> Tensor {

    todo!();
        /*
            // dilation
      // Check input dimensions.
      TORCH_CHECK(kH > 0 && kW > 0, "kernel_size should be greater than zero.");
      TORCH_CHECK(sH > 0 && sW > 0, "strides should be greater than zero.");
      TORCH_CHECK(
          dH > 0 && dW > 0,
          "dilation should be greater than zero. "
          "Got (",
          dH,
          ", ",
          dW,
          ")");

      int ndim = qx.dim();
      TORCH_CHECK(
          ndim == 3 || ndim == 4, "Expecting the input tensor of rank 3 or 4.");
      int dimc = 0;
      int dimh = 1;
      int dimw = 2;
      int nbatch = 1;
      if (ndim == 4) { // Includes batches
        ++dimc;
        ++dimh;
        ++dimw;
        nbatch = qx.size(0);
      }

      // Check if inputs are valid.
      i64 iC = qx.size(dimc);
      i64 iH = qx.size(dimh);
      i64 iW = qx.size(dimw);
      TORCH_CHECK(iC > 0 && iH > 0 && iW > 0, "input dimensions must be non-zero.");
      TORCH_CHECK(
          (ndim == 3 || ndim == 4),
          "non-empty 3D or 4D input tensor is expected.");
      TORCH_CHECK(
          kH / 2 >= pH && kW / 2 >= pW,
          "padding should be smaller than half of kernel_size.");

      // Check output dimensions.
      i64 oC = iC;
      i64 oH = pooling_output_shape(iH, kH, pH, sH, dH, ceil_mode);
      i64 oW = pooling_output_shape(iW, kW, pW, sW, dW, ceil_mode);
      TORCH_CHECK(oH > 0 && oW > 0,
                  "Given input size: (",
                  iC, "x", iH, "x", iW,
                  "). Calculated output size: (",
                  oC, "x", oH, "x", oW,
                  "). Output size is too small.");

      vector<i64> oSizes;
      if (ndim == 3) {
        oSizes = {oC, oH, oW};
      } else {
        oSizes = {nbatch, oC, oH, oW};
      }

      if (qx.is_contiguous(MemoryFormat::ChannelsLast)) {
        // Fast path case for channels-last case.
        // In this case, we can preserve the data layout in memory
        // as well as use a loop nest that is more amenable to
        // vectorization.
        Tensor qy = _empty_affine_quantized(
            oSizes,
            qx.options()
              .dtype(toQIntType(qx.scalar_type()))
              .memory_format(qx.suggest_memory_format()),
            qx.q_scale(),
            qx.q_zero_point(),
            nullopt);
        qmaxpool_2d_nhwc_stub(qx.device().type(), qx, iC, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, qy);
        return qy;
      } else {
        Tensor qy = _empty_affine_quantized(
            oSizes,
            qx.options().dtype(toQIntType(qx.scalar_type())),
            qx.q_scale(),
            qx.q_zero_point());
        auto qx_contig = qx.contiguous();
        auto qxd = qx_contig.data_ptr<Q>();
        auto qyd = qy.data_ptr<Q>();
        if (ndim == 3 || nbatch == 1) {
          auto* iData = qxd;
          auto* oData = qyd;
          spatial_dilated_max_pooling<Q>(
              iData,
              iC,
              iH,
              iW,
              oH,
              oW,
              kH,
              kW,
              sH,
              sW,
              pH,
              pW,
              dH,
              dW,
              oData);
        } else {
          parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
            for (auto p = start; p < end; ++p) {
              auto* iData = qxd + p * iC * iW * iH;
              auto* oData = qyd + p * oC * oW * oH;
              spatial_dilated_max_pooling<Q>(
                  iData,
                  iC,
                  iH,
                  iW,
                  oH,
                  oW,
                  kH,
                  kW,
                  sH,
                  sW,
                  pH,
                  pW,
                  dH,
                  dW,
                  oData);
            }
          });
        }
        return qy;
      }
        */
}

pub fn check_maxpool2d_params(
    kernel_size: &[i32],
    stride:      &[i32],
    padding:     &[i32],
    dilation:    &[i32])  {
    
    todo!();
        /*
            TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
                  "Expected 1d or 2d kernel size, got ", kernel_size.size());
      TORCH_CHECK(stride.empty() || stride.size() == 2,
                  "Expected no strides or 2d strides, got", stride.size());
      TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
                  "Expected 1d or 2d padding, got ", padding.size());
      TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
                  "Expected 1d or 2d dilation, got ", dilation.size());
        */
}


#[cfg(USE_PYTORCH_QNNPACK)]
pub fn qnnpack_maxpool2d(
    input:       Tensor,
    kernel_size: &[i32],
    stride:      &[i32],
    padding:     &[i32],
    dilation:    &[i32],
    ceil_mode:   bool) -> Tensor {

    todo!();
        /*
            Tensor qy;

       TORCH_CHECK(
           input.ndimension() == 4,
           "qnnpack_maxpool2d(): Expected input to be 4-dimensional: got ",
           input.ndimension());
       TORCH_CHECK(
           kernel_size.size() == 2,
           "qnnpack_maxpool2d(): Expected kernel_size to be 2-dimensional: got ",
           kernel_size.size());
       TORCH_CHECK(
           stride.size() == 2,
           "qnnpack_maxpool2d(): Expected stride to be 2-dimensional: got ",
           stride.size());
       TORCH_CHECK(
           dilation.size() == 2,
           "qnnpack_maxpool2d(): Expected dilation to be 2-dimensional: got ",
           dilation.size());
       TORCH_CHECK(
           padding.size() == 2,
           "qnnpack_maxpool2d(): Expected padding to be 2-dimensional: got ",
           padding.size());

       i64 batch_size = input.size(0);
       i64 inC = input.size(1);
       i64 inH = input.size(2);
       i64 inW = input.size(3);
       Tensor input_contig = input.contiguous(MemoryFormat::ChannelsLast);

       initQNNPACK();
       const auto scale = input_contig.q_scale();
       const auto zero_point = input_contig.q_zero_point();
       pytorch_qnnp_operator_t qnnpack_operator{nullptr};

       i64 padH = padding[0];
       i64 padW = padding[1];
       i64 kH = kernel_size[0];
       i64 kW = kernel_size[1];
       i64 strideH = stride[0];
       i64 strideW = stride[1];
       i64 dilationH = dilation[0];
       i64 dilationW = dilation[1];

       TORCH_CHECK(
           kH > 0 && kW > 0,
           "qnnpack_maxpool2d(): kernel_size should be greater than zero.");
       TORCH_CHECK(
           strideH > 0 && strideW > 0,
           "qnnpack_maxpool2d(): strides should be greater than zero.");

       const pytorch_qnnp_status createStatus =
           pytorch_qnnp_create_max_pooling2d_nhwc_u8(
               padH /* input_padding_top */,
               padW /* input_padding_right */,
               padH /* input_padding_bottom */,
               padW /* input_padding_left */,
               kH /* pooling height */,
               kW /* pooling width */,
               strideH /* stride height */,
               strideW /* stride width */,
               dilationH /* dilation height */,
               dilationW /* dilation width */,
               inC /* input channels */,
               u8::min /* output min */,
               u8::max /* output max */,
               0 /* flags */,
               &qnnpack_operator);
       TORCH_INTERNAL_ASSERT(
           createStatus == pytorch_qnnp_status_success,
           "failed to create QNNPACK MaxPool operator");

       i64 outC = inC;
       i64 outH =
           pooling_output_shape(inH, kH, padH, strideH, dilationH, ceil_mode);
       i64 outW =
           pooling_output_shape(inW, kW, padW, strideW, dilationW, ceil_mode);

       TORCH_CHECK(
           outH > 0 && outW > 0,
           "qnnpack_maxpool2d(): the resulting output Tensor size should be >= 0");

       unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
           qnnpack_uniq_ptr(qnnpack_operator);

       // NHWC output
       qy = _empty_affine_quantized(
           {batch_size, outC, outH, outW},
           device(kCPU).dtype(kQUInt8),
           scale,
           zero_point,
           MemoryFormat::ChannelsLast);

       const pytorch_qnnp_status setupStatus =
           pytorch_qnnp_setup_max_pooling2d_nhwc_u8(
               qnnpack_operator /* max pooling */,
               batch_size /* batch size */,
               inH /* input height */,
               inW /* input width */,
               (u8*)input_contig.data_ptr<quint8>() /* input */,
               inC /* input_pixel_stride */,
               (u8*)qy.data_ptr<quint8>() /* output data */,
               outC /* output_pixel_stride */,
               nullptr /* thread pool */);
       TORCH_INTERNAL_ASSERT(
           setupStatus == pytorch_qnnp_status_success,
           "failed to setup QNNPACK MaxPool operator");

       pthreadpool_t threadpool = pthreadpool_();
       const pytorch_qnnp_status runStatus =
           pytorch_qnnp_run_operator(qnnpack_operator, threadpool);
       TORCH_INTERNAL_ASSERT(
           runStatus == pytorch_qnnp_status_success,
           "failed to run QNNPACK MaxPool operator");
       return qy.contiguous(input.suggest_memory_format());
        */
}

/**
  | native functions for the native_functions.yaml
  |
  */
pub fn quantized_max_pool2d(
        qx:          &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        ceil_mode:   bool) -> Tensor {
    
    todo!();
        /*
            check_maxpool2d_params(
          kernel_size,
          stride,
          padding,
          dilation);
      if (stride.empty()) {
        stride = kernel_size;
      }
    #ifdef USE_PYTORCH_QNNPACK
      if (globalContext().qEngine() == QEngine::QNNPACK && qx.scalar_type() == kQUInt8 && !ceil_mode) {
        return qnnpack_maxpool2d(qx, kernel_size, stride, padding, dilation, ceil_mode);
      }
    #endif
      Tensor qy;
      AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "max_pool2d", [&]() {
        qy = q_maxpool_2d<Scalar>(
            qx,
            kernel_size[0],
            kernel_size[1],
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
            ceil_mode);
      });
      return qy;
        */
}

/**
  | Quantized max_pool1d is a special case of the
  | max_pool2d, with one of the dimensions and
  | kernels removed.
  |
  */
pub fn quantized_max_pool1d(
    qx:          &Tensor,
    kernel_size: &[i32],
    stride:      &[i32],
    padding:     &[i32],
    dilation:    &[i32],
    ceil_mode:   bool) -> Tensor {
    
    todo!();
        /*
            // (C, L) -> (C, 1, L) => kSqueezeDim = 1
      // (N, C, L) -> (N, C, 1, L) => kSqueezeDim = 2
      const i32 kSqueezeDim = qx.dim() - 1;
      const auto qx_unsqueeze = qx.unsqueeze(kSqueezeDim);
      if (stride.empty()) {
        stride = kernel_size;
      }
      auto qy = quantized_max_pool2d(
        qx.unsqueeze(kSqueezeDim),
        {1, kernel_size[0]},
        {1, stride[0]},
        {0, padding[0]},
        {1, dilation[0]},
        ceil_mode);
      qy = qy.squeeze(kSqueezeDim);
      return qy;
        */
}

/**
  | Keep the registry in the anonymous namespace.
  |
  */
pub struct QMaxPool_arr_args<const kSpatialDim: u32> {

}

impl<const kSpatialDim: u32> QMaxPool_arr_args<kSpatialDim> {
    
    pub fn run(
        qx:          Tensor,
        kernel_size: Vec<i64>,
        stride:      Vec<i64>,
        padding:     Vec<i64>,
        dilation:    Vec<i64>,
        ceil_mode:   bool) -> Tensor {
        
        todo!();
        /*
            if (kSpatialDim == 1) {
          return quantized_max_pool1d(qx, kernel_size, stride, padding,
                                          dilation, ceil_mode);
        } else if (kSpatialDim == 2) {
          return quantized_max_pool2d(qx, kernel_size, stride, padding,
                                          dilation, ceil_mode);
        }
        TORCH_CHECK(false, "MaxPool", kSpatialDim, "D is not supported.");
        */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::max_pool1d"), TORCH_FN(QMaxPool_arr_args<1>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::max_pool2d"), TORCH_FN(QMaxPool_arr_args<2>::run));
    }
    */
}
