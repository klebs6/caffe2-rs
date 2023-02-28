crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/AdaptiveAveragePooling3d.cpp]

#[inline] pub fn start_index(
        a: i32,
        b: i32,
        c: i32) -> i32 {
    
    todo!();
        /*
      return (int)std::floor((float)(a * c) / b);
        */
}

#[inline] pub fn end_index(
        a: i32,
        b: i32,
        c: i32) -> i32 {
    
    todo!();
        /*
      return (int)std::ceil((float)((a + 1) * c) / b);
        */
}

pub fn adaptive_avg_pool3d_out_frame<Scalar>(
    input_p:  *mut Scalar,
    output_p: *mut Scalar,
    sized:    i64,
    isizet:   i64,
    isizeh:   i64,
    isizew:   i64,
    osizet:   i64,
    osizeh:   i64,
    osizew:   i64,
    istrided: i64,
    istridet: i64,
    istrideh: i64,
    istridew: i64)  {

    todo!();
        /*
            at::parallel_for(0, sizeD, 1, [&](i64 start, i64 end) {
        for (i64 d = start; d < end; d++) {
          /* loop over output */
          for (i64 ot = 0; ot < osizeT; ot++) {
            int istartT = start_index(ot, osizeT, isizeT);
            int iendT = end_index(ot, osizeT, isizeT);
            int kT = iendT - istartT;

            for (i64 oh = 0; oh < osizeH; oh++) {
              int istartH = start_index(oh, osizeH, isizeH);
              int iendH = end_index(oh, osizeH, isizeH);
              int kH = iendH - istartH;

              for (i64 ow = 0; ow < osizeW; ow++) {
                int istartW = start_index(ow, osizeW, isizeW);
                int iendW = end_index(ow, osizeW, isizeW);
                int kW = iendW - istartW;

                /* local pointers */
                Scalar* ip = input_p + d * istrideD + istartT * istrideT +
                    istartH * istrideH + istartW * istrideW;
                Scalar* op = output_p + d * osizeT * osizeH * osizeW +
                    ot * osizeH * osizeW + oh * osizeW + ow;

                /* compute local average: */
                Scalar sum = 0;
                for (int it = 0; it < kT; it++) {
                  for (int ih = 0; ih < kH; ih++) {
                    for (int iw = 0; iw < kW; iw++) {
                      Scalar val =
                          *(ip + it * istrideT + ih * istrideH + iw * istrideW);
                      sum += val;
                    }
                  }
                }

                /* set output to local average */
                *op = sum / kT / kH / kW;
              }
            }
          }
        }
      });
        */
}

pub fn adaptive_avg_pool3d_out_cpu_template(
    output:      &mut Tensor,
    input:       &Tensor,
    output_size: &[i32])  {
    
    todo!();
        /*
            TORCH_CHECK(output_size.size() == 3, "adaptive_avg_pool3d: output_size must be 3");

      for (i64 i = 0; i < input.ndimension(); i++) {
        TORCH_CHECK(
            input.size(i) > 0,
            "adaptive_avg_pool3d(): expected input to have non-empty spatial dimensions, "
            "but input has sizes ",
            input.sizes(),
            " with dimension ",
            i,
            " being "
            "empty");
      }

      TORCH_CHECK(
          (input.ndimension() == 4 || input.ndimension() == 5),
          "non-empty 4D or 5D (batch mode) tensor expected for input");
      TORCH_CHECK(input.dtype() == output.dtype(),
          "expected dtype ", input.dtype(), " for `output` but got dtype ", output.dtype());

      /* sizes */
      i64 sizeD = input.size(-4);
      i64 isizeT = input.size(-3);
      i64 isizeH = input.size(-2);
      i64 isizeW = input.size(-1);
      /* strides */
      i64 istrideD = input.stride(-4);
      i64 istrideT = input.stride(-3);
      i64 istrideH = input.stride(-2);
      i64 istrideW = input.stride(-1);
      /* output sizes */
      auto osizeT = output_size[0];
      auto osizeH = output_size[1];
      auto osizeW = output_size[2];

      if (input.ndimension() == 4) {
        output.resize_({sizeD, osizeT, osizeH, osizeW});

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "adaptive_avg_pool3d_cpu", [&] {
              auto input_data = input.data_ptr<Scalar>();
              auto output_data = output.data_ptr<Scalar>();
              adaptive_avg_pool3d_out_frame<Scalar>(
                  input_data,
                  output_data,
                  sizeD,
                  isizeT,
                  isizeH,
                  isizeW,
                  osizeT,
                  osizeH,
                  osizeW,
                  istrideD,
                  istrideT,
                  istrideH,
                  istrideW);
            });
      } else {
        output.resize_({input.size(-5), sizeD, osizeT, osizeH, osizeW});
        i64 n = input.size(0);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "adaptive_avg_pool3d_cpu", [&] {
              auto input_data = input.data_ptr<Scalar>();
              auto output_data = output.data_ptr<Scalar>();
              at::parallel_for(0, n, 1, [&](i64 start, i64 end) {
                for (i64 b = start; b < end; ++b) {
                  adaptive_avg_pool3d_out_frame<Scalar>(
                      input_data + b * input.stride(0),
                      output_data + b * sizeD * osizeT * osizeH * osizeW,
                      sizeD,
                      isizeT,
                      isizeH,
                      isizeW,
                      osizeT,
                      osizeH,
                      osizeW,
                      istrideD,
                      istrideT,
                      istrideH,
                      istrideW);
                }
              });
        });
      }
        */
}

pub fn adaptive_avg_pool3d_backward_out_frame<Scalar>(
    grad_input_p:  *mut Scalar,
    grad_output_p: *mut Scalar,
    sized:         i64,
    isizet:        i64,
    isizeh:        i64,
    isizew:        i64,
    osizet:        i64,
    osizeh:        i64,
    osizew:        i64)  {

    todo!();
        /*
            at::parallel_for(0, sizeD, 1, [&](i64 start, i64 end) {
        for (i64 d = start; d < end; d++) {
          Scalar* gradInput_p_d = gradInput_p + d * isizeT * isizeW * isizeH;
          Scalar* gradOutput_p_d = gradOutput_p + d * osizeT * osizeW * osizeH;

          /* calculate average */
          for (i64 ot = 0; ot < osizeT; ot++) {
            int istartT = start_index(ot, osizeT, isizeT);
            int iendT = end_index(ot, osizeT, isizeT);
            int kT = iendT - istartT;

            for (i64 oh = 0; oh < osizeH; oh++) {
              int istartH = start_index(oh, osizeH, isizeH);
              int iendH = end_index(oh, osizeH, isizeH);
              int kH = iendH - istartH;

              for (i64 ow = 0; ow < osizeW; ow++) {
                int istartW = start_index(ow, osizeW, isizeW);
                int iendW = end_index(ow, osizeW, isizeW);
                int kW = iendW - istartW;

                Scalar grad_delta =
                    gradOutput_p_d[ot * osizeH * osizeW + oh * osizeW + ow] / kT /
                    kH / kW;

                for (int it = istartT; it < iendT; it++) {
                  for (int ih = istartH; ih < iendH; ih++) {
                    for (int iw = istartW; iw < iendW; iw++) {
                      /* update gradient */
                      gradInput_p_d[it * isizeH * isizeW + ih * isizeW + iw] +=
                          grad_delta;
                    }
                  }
                }
              }
            }
          }
        }
      });
        */
}

pub fn adaptive_avg_pool3d_backward_out_cpu_template(
    grad_input:  &mut Tensor,
    grad_output: &Tensor,
    input:       &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            /* get contiguous gradOutput */
      auto gradOutput = gradOutput_.contiguous();

      /* sizes */
      i64 sizeD = input.size(-4);
      i64 isizeT = input.size(-3);
      i64 isizeH = input.size(-2);
      i64 isizeW = input.size(-1);
      i64 osizeT = gradOutput.size(-3);
      i64 osizeH = gradOutput.size(-2);
      i64 osizeW = gradOutput.size(-1);

      /* backprop */
      if (input.ndimension() == 4) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "adaptive_avg_pool3d_backward_cpu", [&] {
              /* get raw pointers */
              Scalar* gradInput_data = gradInput.data_ptr<Scalar>();
              Scalar* gradOutput_data = gradOutput.data_ptr<Scalar>();

              adaptive_avg_pool3d_backward_out_frame<Scalar>(
                  gradInput_data,
                  gradOutput_data,
                  sizeD,
                  isizeT,
                  isizeH,
                  isizeW,
                  osizeT,
                  osizeH,
                  osizeW);
            });
      } else {
        i64 n = input.size(0);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "adaptive_avg_pool3d_backward_cpu", [&] {
              /* get raw pointers */
              Scalar* gradInput_data = gradInput.data_ptr<Scalar>();
              Scalar* gradOutput_data = gradOutput.data_ptr<Scalar>();
              at::parallel_for(0, n, 1, [&](i64 start, i64 end) {
                for (i64 b = start; b < end; b++) {
                  adaptive_avg_pool3d_backward_out_frame<Scalar>(
                      gradInput_data + b * sizeD * isizeT * isizeH * isizeW,
                      gradOutput_data + b * sizeD * osizeT * osizeH * osizeW,
                      sizeD,
                      isizeT,
                      isizeH,
                      isizeW,
                      osizeT,
                      osizeH,
                      osizeW);
                }
              });
        });
      }
      return gradInput;
        */
}

pub fn adaptive_avg_pool3d_out_cpu(
    input:       &Tensor,
    output_size: &[i32],
    output:      &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            adaptive_avg_pool3d_out_cpu_template(output, input, output_size);
      return output;
        */
}

pub fn adaptive_avg_pool3d_cpu(
    input:       &Tensor,
    output_size: &[i32]) -> Tensor {

    todo!();
        /*
            auto output = at::empty({0}, input.options());
      adaptive_avg_pool3d_out_cpu_template(output, input, output_size);
      return output;
        */
}

pub fn adaptive_avg_pool3d(
        input:       &Tensor,
        output_size: &[i32]) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(output_size.size() == 3, "adaptive_avg_pool3d: output_size must be 3");

      if (output_size[0] == 1 && output_size[1] == 1 && output_size[2] == 1) {
        // in this case, adaptive pooling is just computing mean over hw
        // dimensions, which can be done more efficiently
        Tensor out = input.mean({-1, -2, -3}, /* keepdim = */ true);
        return out;
      } else {
        return _adaptive_avg_pool3d(input, output_size);
      }
        */
}

pub fn adaptive_avg_pool3d_backward_out_cpu(
        grad_output: &Tensor,
        input:       &Tensor,
        grad_input:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            gradInput.resize_as_(input).zero_();
      adaptive_avg_pool3d_backward_out_cpu_template(gradInput, gradOutput_, input);
      return gradInput;
        */
}

pub fn adaptive_avg_pool3d_backward_cpu(
        grad_output: &Tensor,
        input:       &Tensor) -> Tensor {
    
    todo!();
        /*
            auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      adaptive_avg_pool3d_backward_out_cpu_template(gradInput, gradOutput_, input);
      return gradInput;
        */
}
