crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/q_adaavgpool.cpp]

define_dispatch!{qadaptive_avg_pool2d_nhwc_stub}
define_dispatch!{qadaptive_avg_pool3d_ndhwc_stub}

#[inline] pub fn start_index(
        out_idx: i32,
        out_len: i32,
        in_len:  i32) -> i32 {
    
    todo!();
        /*
            /*
       * out_idx: the current index of output matrix
       * out_len: the dimension_size of output matrix
       * in_len: the dimension_size of input matrix
       * Basically, in_len / out_len gives the number of
       * elements in each average computation.
       * This function computes the start index on input matrix.
       */
      return (int)floor((float)(out_idx * in_len) / out_len);
        */
}

#[inline] pub fn end_index(
        out_idx: i32,
        out_len: i32,
        in_len:  i32) -> i32 {
    
    todo!();
        /*
            /*
       * Parameter definition is the same as start_index.
       * This function computes the end index on input matrix.
       */
      return (int)ceil((float)((out_idx + 1) * in_len) / out_len);
        */
}

// adaptive avg pool for 2D and 3D inputs
//
pub fn adaptive_avg_pool_single_out_frame<Scalar>(
        input_p:  *mut Scalar,
        output_p: *mut Scalar,
        sizec:    i64,

        // Set to 1 for 2D
        isized:   i64,
        isizeh:   i64,
        isizew:   i64,

        // Set to 1 for 2D
        osized:   i64,
        osizeh:   i64,
        osizew:   i64,
        istridec: i64,

        // Set to 1 for 2D
        istrided: i64,
        istrideh: i64,
        istridew: i64)  {

    todo!();
        /*
            parallel_for(0, sizeC, 0, [&](i64 start, i64 end) {
        for (auto c = start; c < end; c++) {
          /* loop over output */
          i64 od, oh, ow;
          for (od = 0; od < osizeD; od++) {
            int istartD = start_index(od, osizeD, isizeD);
            int iendD = end_index(od, osizeD, isizeD);
            int kD = iendD - istartD;
            float kDr = 1.0 / kD;
            for (oh = 0; oh < osizeH; oh++) {
              int istartH = start_index(oh, osizeH, isizeH);
              int iendH = end_index(oh, osizeH, isizeH);
              int kH = iendH - istartH;
              float kDHr = kDr / kH;

              for (ow = 0; ow < osizeW; ow++) {
                int istartW = start_index(ow, osizeW, isizeW);
                int iendW = end_index(ow, osizeW, isizeW);
                int kW = iendW - istartW;
                float kDHWr = kDHr / kW;

                /* local pointers */
                Scalar* ip = input_p +
                               c * istrideC +
                               istartD * istrideD +
                               istartH * istrideH +
                               istartW * istrideW;
                Scalar* op = output_p +
                               c * osizeD * osizeH * osizeW +
                               od * osizeH * osizeW +
                               oh * osizeW +
                               ow;

                /* compute local average: */
                i64 sum = 0;
                int id, ih, iw;
                for (id = 0; id < kD; id++) {
                  for (ih = 0; ih < kH; ih++) {
                    for (iw = 0; iw < kW; iw++) {
                      i64 val = (ip +
                                     id * istrideD +
                                     ih * istrideH +
                                     iw * istrideW)->val_;
                      sum += val;
                    }
                  }
                }

                /* set output to local average */
                // TODO: add the max/min clip
                op->val_ = static_cast<typename Scalar::underlying>(
                    nearbyint(sum * kDHWr));
              } // ow
            } // oh
          } // od
        }
      });
        */
}

pub fn get_output_shape<const DIM: i64>(
        input:       &Tensor,
        output_size: &[i32]) -> Vec<i64> {

    todo!();
        /*
            for (i64 i = 1; i < input.dim(); i++) {
        // Allow for empty batch.
        TORCH_CHECK(
            input.size(i) > 0,
            "adaptive_avg_pooling", DIM, "d(): ",
            "expected input to have non-empty spatial "
            "dimensions, but input has sizes ",
            input.sizes(),
            " with dimension ",
            i,
            " being empty");
      }

      TORCH_CHECK(
          (input.dim() == DIM + 1 || input.dim() == DIM + 2),
          "non-empty ",
          DIM + 1,
          "D or ",
          DIM + 2,
          "D (batch mode) tensor expected for input");

      /* Channels */
      const i64 sizeC = input.size(-(DIM+1));

      vector<i64> output_shape;
      output_shape.reserve(input.dim());
      if (input.dim() == DIM + 2) {
        // Include Batch
        output_shape.push_back(input.size(0));
      }
      output_shape.push_back(sizeC);
      for (const auto size : output_size) {
        output_shape.push_back(size);
      }
      return output_shape;
        */
}

pub fn adaptive_avg_pool<const kSpatialDim: i32, Scalar>(
        input:       &Tensor,
        output_size: &[i32],
        output:      &mut Tensor) -> Tensor {

    todo!();
        /*
            const auto output_shape = get_output_shape<kSpatialDim>(input, output_size);
      /* sizes */
      i64 sizeC = input.size(-(kSpatialDim + 1));
      i64 isizeD = kSpatialDim == 2 ? 1 : input.size(-3);
      i64 isizeH = input.size(-2);
      i64 isizeW = input.size(-1);
      /* strides */
      i64 istrideC = input.stride(-(kSpatialDim + 1));
      i64 istrideD = kSpatialDim == 2 ? 1 : input.stride(-3);
      i64 istrideH = input.stride(-2);
      i64 istrideW = input.stride(-1);

      auto osizeD = kSpatialDim == 2 ? 1 : output_shape[output_shape.size() - 3];
      auto osizeH = output_shape[output_shape.size() - 2];
      auto osizeW = output_shape[output_shape.size() - 1];

      i64 sizeB = output_shape.size() ==(kSpatialDim + 1) ? 0 : output_shape[0];
      if (input.is_contiguous(MemoryFormat::ChannelsLast) ||
          input.is_contiguous(MemoryFormat::ChannelsLast3d)) {
        // Fast path for NDHWC
        output = _empty_affine_quantized(
            output_shape,
            input.options().memory_format(input.suggest_memory_format()),
            input.q_scale(),
            input.q_zero_point(),
            nullopt);
        if (input.dim() == (kSpatialDim + 1) || input.size(0) == 1) {
          qadaptive_avg_pool3d_ndhwc_stub(
              input.device().type(),
              input,
              output,
              0,
              sizeC,
              isizeD,
              isizeH,
              isizeW,
              osizeD,
              osizeH,
              osizeW,
              0,
              istrideC,
              istrideD,
              istrideH,
              istrideW);
        } else {
          i64 istrideB = input.stride(-(kSpatialDim + 2));
          parallel_for(0, sizeB, 0, [&](i64 start, i64 end) {
            for (auto b = start; b < end; b++) {
              qadaptive_avg_pool3d_ndhwc_stub(
                  input.device().type(),
                  input,
                  output,
                  b,
                  sizeC,
                  isizeD,
                  isizeH,
                  isizeW,
                  osizeD,
                  osizeH,
                  osizeW,
                  istrideB,
                  istrideC,
                  istrideD,
                  istrideH,
                  istrideW);
            }
          });
        }
        return output;
      } else {
        output = _empty_affine_quantized(
            output_shape, input.options(), input.q_scale(), input.q_zero_point());
        auto input_contig = input.contiguous();
        auto input_data = input_contig.data_ptr<Scalar>();
        auto output_data = output.data_ptr<Scalar>();

        if (input.dim() ==(kSpatialDim + 1) || input.size(0) == 1) {
          adaptive_avg_pool_single_out_frame<Scalar>(
              input_data,
              output_data,
              sizeC,
              isizeD,
              isizeH,
              isizeW,
              osizeD,
              osizeH,
              osizeW,
              istrideC,
              istrideD,
              istrideH,
              istrideW);
        } else {
          i64 istrideB = input.stride(-(kSpatialDim + 2));
          parallel_for(0, sizeB, 0, [&](i64 start, i64 end) {
            for (auto b = start; b < end; b++) {
              adaptive_avg_pool_single_out_frame<Scalar>(
                  input_data + b * istrideB,
                  output_data + b * sizeC * osizeD * osizeH * osizeW,
                  sizeC,
                  isizeD,
                  isizeH,
                  isizeW,
                  osizeD,
                  osizeH,
                  osizeW,
                  istrideC,
                  istrideD,
                  istrideH,
                  istrideW);
            }
          });
        }
        return output;
      }
        */
}

pub fn q_adaptive_avg_pool2d<Scalar>(
    input:       &Tensor,
    output_size: &[i32]) -> Tensor {

    todo!();
        /*
            Tensor output;
      return _adaptive_avg_pool<2, Scalar>(input, output_size, output);
        */
}

pub fn q_adaptive_avg_pool3d<Scalar>(
    output:      &mut Tensor,
    input:       &Tensor,
    output_size: &[i32]) -> Tensor {

    todo!();
        /*
            return _adaptive_avg_pool<3, Scalar>(input, output_size, output);
        */
}

#[cfg(USE_PYTORCH_QNNPACK)]
pub fn qnnpack_adaptive_avg_pool2d(
    input:       &Tensor,
    output_size: &[i32]) -> Tensor {
    
    todo!();
        /*
      array<i64, 2> kernel_size;
      array<i64, 2> stride;
      array<i64, 2> padding{0, 0};
      bool ceil_mode{false};
      bool count_include_pad{false};

      const auto output_shape = get_output_shape<2>(input, output_size);
      auto output_height = output_shape[output_shape.size() - 2];
      auto output_width = output_shape[output_shape.size() - 1];
      auto input_height = input.sizes()[input.dim() - 2];
      auto input_width = input.sizes()[input.dim() - 1];
      stride[0] = input_height / output_height;
      stride[1] = input_width / output_width;
      // Given the constraint that input_height/width % output_height/width == 0
      // stride and kernel size are same.
      kernel_size[0] = stride[0];
      kernel_size[1] = stride[1];

      return native::qnnp_avgpool_helper::qnnpack_avg_pool2d(
          input,
          kernel_size,
          stride,
          padding,
          ceil_mode,
          count_include_pad,
          nullopt);
        */
}

#[cfg(USE_PYTORCH_QNNPACK)]
pub fn enable_qnnpack_for_ada_avgpool(
    input:       &Tensor,
    output_size: &[i32]) -> bool {
    
    todo!();
        /*
            const auto output_shape = get_output_shape<2>(input, output_size);
      auto output_height = output_shape[output_shape.size() - 2];
      auto output_width = output_shape[output_shape.size() - 1];
      auto input_height = input.sizes()[input.dim() - 2];
      auto input_width = input.sizes()[input.dim() - 1];

      return !(input_width == output_width && input_height == output_height) &&
          (input_height % output_height == 0) && (input_width % output_width == 0);
        */
}

pub fn adaptive_avg_pool2d_quantized_cpu(
    input:       &Tensor,
    output_size: &[i32]) -> Tensor {
    
    todo!();
        /*
            #ifdef USE_PYTORCH_QNNPACK
      if (globalContext().qEngine() == QEngine::QNNPACK &&
          input.scalar_type() == kQUInt8 &&
          enable_qnnpack_for_ada_avgpool(input, output_size)) {
        return qnnpack_adaptive_avg_pool2d(input, output_size);
      }
    #endif
      Tensor output;
      AT_DISPATCH_QINT_TYPES(
          input.scalar_type(), "adaptive_avg_pool2d_quantized_cpu", [&]() {
            output = q_adaptive_avg_pool2d<Scalar>(input, output_size);
          });
      return output;
        */
}

pub fn adaptive_avg_pool3d_out_quantized_cpu<'a>(
    input:       &Tensor,
    output_size: &[i32],
    output:      &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            #ifdef USE_PYTORCH_QNNPACK
      if (globalContext().qEngine() == QEngine::QNNPACK) {
        TORCH_WARN("Quantized Adaptive Average Pool 3D is not implemented for ",
                   "QNNPACK. Falling back to default implementation.");
      }
    #endif
      AT_DISPATCH_QINT_TYPES(
          input.scalar_type(), "adaptive_avg_pool3d_quantized_cpu", [&]() {
            output = q_adaptive_avg_pool3d<Scalar>(output, input, output_size);
          });
      return output;
        */
}

pub fn adaptive_avg_pool3d_quantized_cpu(
    input:       &Tensor,
    output_size: &[i32]) -> Tensor {
    
    todo!();
        /*
            Tensor output;
      return native::adaptive_avg_pool3d_out_quantized_cpu(input, output_size, output);
        */
}
