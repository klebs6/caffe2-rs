// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/AveragePool3d.cpp]

lazy_static!{
    /*
    TORCH_META_FUNC(avg_pool3d) (
      const Tensor& input,
      IntArrayRef kernel_size,
      IntArrayRef stride,
      IntArrayRef padding,
      bool ceil_mode,
      bool count_include_pad,
      optional<i64> divisor_override
    ) {
      // #20866, #22032: Guarantee this for the official C++ API?
      TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
        "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
      const int kT = safe_downcast<int, i64>(kernel_size[0]);
      const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, i64>(kernel_size[1]);
      const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, i64>(kernel_size[2]);

      TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 3,
        "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints");
      const int dT = stride.empty() ? kT : safe_downcast<int, i64>(stride[0]);
      const int dH = stride.empty() ? kH :
                     stride.size() == 1 ? dT : safe_downcast<int, i64>(stride[1]);
      const int dW = stride.empty() ? kW :
                     stride.size() == 1 ? dT : safe_downcast<int, i64>(stride[2]);

      TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
        "avg_pool3d: padding must be a single int, or a tuple of three ints");
      const int padT = safe_downcast<int, i64>(padding[0]);
      const int padH = padding.size() == 1 ? padT : safe_downcast<int, i64>(padding[1]);
      const int padW = padding.size() == 1 ? padT : safe_downcast<int, i64>(padding[2]);

      TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
        "non-empty 4D or 5D (batch mode) tensor expected for input");

      TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
        "divisor must be not zero");

      /* sizes */
      const i64 nbatch = input.size(0);
      const i64 nslices = input.size(-4);
      const i64 itime = input.size(-3);
      const i64 iheight = input.size(-2);
      const i64 iwidth = input.size(-1);

      const i64 otime = pooling_output_shape<i64>(itime, kT, padT, dT, 1, ceil_mode);
      const i64 oheight = pooling_output_shape<i64>(iheight, kH, padH, dH, 1, ceil_mode);
      const i64 owidth = pooling_output_shape<i64>(iwidth, kW, padW, dW, 1, ceil_mode);

      pool3d_shape_check(
        input,
        nslices,
        kT, kH, kW,
        dT, dH, dW,
        padT, padH, padW,
        1, 1, 1,
        itime, iheight, iwidth,
        otime, oheight, owidth,
        /*check_input_size=*/ true);

      /* resize output */
      if (input.ndimension() == 4) {
        set_output(0, {nslices, otime, oheight, owidth}, input.options());
      }
      else {
        set_output(0, {nbatch, nslices, otime, oheight, owidth}, input.options());
      }
    }
    */
}

pub fn avg_pool3d_out_frame<Scalar>(
        input_p:           *mut Scalar,
        output_p:          *mut Scalar,
        nslices:           i64,
        itime:             i64,
        iwidth:            i64,
        iheight:           i64,
        otime:             i64,
        owidth:            i64,
        oheight:           i64,
        kt:                i32,
        kw:                i32,
        kh:                i32,
        dt:                i32,
        dw:                i32,
        dh:                i32,
        padt:              i32,
        padw:              i32,
        padh:              i32,
        count_include_pad: bool,
        divisor_override:  Option<i64>)  {

    todo!();
        /*
            parallel_for(0, nslices, 0, [&](i64 start, i64 end) {
        for (auto k = start; k < end; k++)
        {
          i64 i, j, ti;

          /* local pointers. */
          Scalar *ip = input_p + k * itime * iwidth * iheight;
          Scalar *op = output_p + k * otime * owidth * oheight;
          for (i = 0; i < otime * oheight * owidth; ++i)
            *(op + i) = 0;

          /* loop over output */
          for (ti = 0; ti < otime; ti++)
          {
            for (i = 0; i < oheight; i++)
            {
              for (j = 0; j < owidth; j++)
              {
                /* compute pool range. */
                i64 tstart = ti * dT - padT;
                i64 hstart = i  * dH - padH;
                i64 wstart = j  * dW - padW;
                i64 tend = min(tstart + kT, itime + padT);
                i64 hend = min(hstart + kH, iheight + padH);
                i64 wend = min(wstart + kW, iwidth + padW);
                i64 pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);
                tstart = max(tstart, (i64) 0);
                hstart = max(hstart, (i64) 0);
                wstart = max(wstart, (i64) 0);
                tend = min(tend, itime);
                hend = min(hend, iheight);
                wend = min(wend, iwidth);

                if (tstart >= tend || hstart >= hend || wstart >= wend) {
                  ++op;
                  continue;
                }

                int divide_factor;
                if (divisor_override.has_value()) {
                  divide_factor = divisor_override.value();
                } else {
                  if(count_include_pad) {
                    divide_factor = pool_size;
                  } else {
                    divide_factor = (tend - tstart) * (hend - hstart) * (wend - wstart);
                  }
                }

                /* compute local sum: */
                Scalar sum = 0.0;
                i64 x, y, z;

                for (z = tstart; z < tend; z++)
                {
                  for (y = hstart; y < hend; y++)
                  {
                    for (x = wstart; x < wend; x++)
                    {
                      sum +=  *(ip + z * iwidth * iheight + y * iwidth + x);
                    }
                  }
                }

                /* set output to local max */
                *op++ += sum / divide_factor;
              }
            }
          }
        }
      });
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(avg_pool3d_out_cpu) (
      const Tensor& input_,
      IntArrayRef kernel_size,
      IntArrayRef stride,
      IntArrayRef padding,
      bool ceil_mode,
      bool count_include_pad,
      optional<i64> divisor_override,
      const Tensor& output
    ) {
      const int kT = safe_downcast<int, i64>(kernel_size[0]);
      const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, i64>(kernel_size[1]);
      const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, i64>(kernel_size[2]);

      const int dT = stride.empty() ? kT : safe_downcast<int, i64>(stride[0]);
      const int dH = stride.empty() ? kH :
                     stride.size() == 1 ? dT : safe_downcast<int, i64>(stride[1]);
      const int dW = stride.empty() ? kW :
                     stride.size() == 1 ? dT : safe_downcast<int, i64>(stride[2]);

      const int padT = safe_downcast<int, i64>(padding[0]);
      const int padH = padding.size() == 1 ? padT : safe_downcast<int, i64>(padding[1]);
      const int padW = padding.size() == 1 ? padT : safe_downcast<int, i64>(padding[2]);

      const i64 nslices = input_.size(-4);
      const i64 itime = input_.size(-3);
      const i64 iheight = input_.size(-2);
      const i64 iwidth = input_.size(-1);

      const i64 otime = pooling_output_shape<i64>(itime, kT, padT, dT, 1, ceil_mode);
      const i64 oheight = pooling_output_shape<i64>(iheight, kH, padH, dH, 1, ceil_mode);
      const i64 owidth = pooling_output_shape<i64>(iwidth, kW, padW, dW, 1, ceil_mode);

      /* get contiguous input */
      Tensor input = input_.contiguous();

      if (input.ndimension() == 4) /* non-batch mode */
      {
        AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Long, input.scalar_type(),
          "avg_pool3d_out_frame",
          [&] {
            Scalar *input_data = input.data_ptr<Scalar>();
            Scalar *output_data = output.data_ptr<Scalar>();

            avg_pool3d_out_frame(
              input_data, output_data, nslices,
              itime, iwidth, iheight,
              otime, owidth, oheight,
              kT, kW, kH,
              dT, dW, dH,
              padT, padW, padH,
              count_include_pad,
              divisor_override);
        });
      }
      else  /* batch mode */
      {
        const i64 nbatch = input.size(0);
        const i64 istride = nslices * itime * iwidth * iheight;
        const i64 ostride = nslices * otime * owidth * oheight;

        AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Long, input.scalar_type(),
          "avg_pool3d_out_frame",
          [&] {
            Scalar *input_data = input.data_ptr<Scalar>();
            Scalar *output_data = output.data_ptr<Scalar>();

            parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
              for (auto p = start; p < end; p++) {
                avg_pool3d_out_frame(
                  input_data + p * istride, output_data + p * ostride, nslices,
                  itime, iwidth, iheight,
                  otime, owidth, oheight,
                  kT, kW, kH,
                  dT, dW, dH,
                  padT, padW, padH,
                  count_include_pad,
                  divisor_override
                );
              }
            });
        });
      }
    }
    */
}

pub fn avg_pool3d_backward_out_frame<Scalar>(
        grad_input_p:      *mut Scalar,
        grad_output_p:     *mut Scalar,
        nslices:           i64,
        itime:             i64,
        iwidth:            i64,
        iheight:           i64,
        otime:             i64,
        owidth:            i64,
        oheight:           i64,
        kt:                i32,
        kw:                i32,
        kh:                i32,
        dt:                i32,
        dw:                i32,
        dh:                i32,
        padt:              i32,
        padw:              i32,
        padh:              i32,
        count_include_pad: bool,
        divisor_override:  Option<i64>)  {

    todo!();
        /*
            parallel_for(0, nslices, 0, [&](i64 start, i64 end) {
        for (auto k = start; k < end; k++)
        {
          i64 i, j, ti;

          /* local pointers */
          Scalar *ip = gradInput_p + k * itime * iwidth * iheight;
          Scalar *op = gradOutput_p + k * otime * owidth * oheight;
          for (i = 0; i < itime*iwidth*iheight; i++)
            *(ip + i) = 0;

          /* loop over output */
          for (ti = 0; ti < otime; ti++)
          {
            for (i = 0; i < oheight; i++)
            {
              for (j = 0; j < owidth; j++)
              {
                i64 tstart = ti * dT - padT;
                i64 hstart = i  * dH - padH;
                i64 wstart = j  * dW - padW;
                i64 tend = min(tstart + kT, itime + padT);
                i64 hend = min(hstart + kH, iheight + padH);
                i64 wend = min(wstart + kW, iwidth + padW);
                i64 pool_size = (tend -tstart) * (hend - hstart) * (wend - wstart);
                tstart = max(tstart, (i64) 0);
                hstart = max(hstart, (i64) 0);
                wstart = max(wstart, (i64) 0);
                tend = min(tend, itime);
                hend = min(hend, iheight);
                wend = min(wend, iwidth);

                int divide_factor;
                if (divisor_override.has_value()) {
                  divide_factor = divisor_override.value();
                } else {
                  if(count_include_pad) {
                    divide_factor = pool_size;
                  } else {
                    divide_factor = (tend - tstart) * (hend - hstart) * (wend - wstart);
                  }
                }

                /* scatter gradients out to footprint: */
                Scalar val  = *op++;

                i64 x,y,z;
                for (z = tstart; z < tend; z++)
                {
                  for (y = hstart; y < hend; y++)
                  {
                    for (x = wstart; x < wend; x++)
                    {
                      *(ip + z * iheight * iwidth + y * iwidth + x) += val / divide_factor;
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

pub fn avg_pool3d_backward_out_cpu_template(
        grad_input:        &mut Tensor,
        grad_output:       &Tensor,
        input:             &Tensor,
        kernel_size:       &[i32],
        stride:            &[i32],
        padding:           &[i32],
        ceil_mode:         bool,
        count_include_pad: bool,
        divisor_override:  Option<i64>) -> &mut Tensor {
    
    todo!();
        /*
            // #20866, #22032: Guarantee this for the official C++ API?
      TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
        "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
      const int kT = safe_downcast<int, i64>(kernel_size[0]);
      const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, i64>(kernel_size[1]);
      const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, i64>(kernel_size[2]);

      TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 3,
        "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints");
      const int dT = stride.empty() ? kT : safe_downcast<int, i64>(stride[0]);
      const int dH = stride.empty() ? kH :
                     stride.size() == 1 ? dT : safe_downcast<int, i64>(stride[1]);
      const int dW = stride.empty() ? kW :
                     stride.size() == 1 ? dT : safe_downcast<int, i64>(stride[2]);

      TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
        "avg_pool3d: padding must be a single int, or a tuple of three ints");
      const int padT = safe_downcast<int, i64>(padding[0]);
      const int padH = padding.size() == 1 ? padT : safe_downcast<int, i64>(padding[1]);
      const int padW = padding.size() == 1 ? padT : safe_downcast<int, i64>(padding[2]);

      TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
        "non-empty 4D or 5D (batch mode) tensor expected for input");

      TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0, "divisor must be not zero");

      const i64 nslices = input.size(-4);
      const i64 itime = input.size(-3);
      const i64 iheight = input.size(-2);
      const i64 iwidth = input.size(-1);

      /* get contiguous gradOutput */
      Tensor gradOutput = gradOutput_.contiguous();

      const i64 otime = gradOutput.size(-3);
      const i64 oheight = gradOutput.size(-2);
      const i64 owidth = gradOutput.size(-1);

      /* XXX shape check behavior from TH */
      const i64 otime_for_shape_check = pooling_output_shape<i64>(itime, kT, padT, dT, 1, ceil_mode);
      const i64 oheight_for_shape_check = pooling_output_shape<i64>(iheight, kH, padH, dH, 1, ceil_mode);
      const i64 owidth_for_shape_check = pooling_output_shape<i64>(iwidth, kW, padW, dW, 1, ceil_mode);

      avg_pool3d_backward_shape_check(
        input,
        gradOutput_,
        nslices,
        kT, kH, kW,
        dT, dH, dW,
        padT, padH, padW,
        itime, iheight, iwidth,
        otime_for_shape_check, oheight_for_shape_check, owidth_for_shape_check);

      /* resize */
      gradInput.resize_as_(input);
      gradInput.zero_();

      /* backprop */
      if (input.ndimension() == 4) /* non-batch mode*/
      {
        AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Long, input.scalar_type(),
          "avg_pool3d_backward_out_frame",
          [&] {
           Scalar *gradInput_data = gradInput.data_ptr<Scalar>();
           Scalar *gradOutput_data = gradOutput.data_ptr<Scalar>();

           avg_pool3d_backward_out_frame(
             gradInput_data, gradOutput_data,
             nslices,
             itime, iwidth, iheight,
             otime, owidth, oheight,
             kT, kW, kH,
             dT, dW, dH,
             padT, padW, padH,
             count_include_pad,
             divisor_override);
        });
      }
      else /* batch mode */
      {
        const i64 nbatch = input.size(0);
        const i64 istride = nslices * itime * iwidth * iheight;
        const i64 ostride = nslices * otime * owidth * oheight;

        AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Long, input.scalar_type(),
          "avg_pool3d_backward_out_frame",
          [&] {
            Scalar *gradInput_data = gradInput.data_ptr<Scalar>();
            Scalar *gradOutput_data = gradOutput.data_ptr<Scalar>();

            parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
              for (auto p = start; p < end; p++)
              {
                avg_pool3d_backward_out_frame(
                  gradInput_data  + p * istride, gradOutput_data + p * ostride, nslices,
                  itime, iwidth, iheight,
                  otime, owidth, oheight,
                  kT, kW, kH,
                  dT, dW, dH,
                  padT, padW, padH,
                  count_include_pad,
                  divisor_override
                );
              }
            });
        });
      }

      return gradInput;
        */
}

pub fn avg_pool3d_backward_out_cpu(
        grad_output:       &Tensor,
        input:             &Tensor,
        kernel_size:       &[i32],
        stride:            &[i32],
        padding:           &[i32],
        ceil_mode:         bool,
        count_include_pad: bool,
        divisor_override:  Option<i64>,
        grad_input:        &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            avg_pool3d_backward_out_cpu_template(
        gradInput,
        gradOutput_,
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
      return gradInput;
        */
}

pub fn avg_pool3d_backward_cpu(
        grad_output:       &Tensor,
        input:             &Tensor,
        kernel_size:       &[i32],
        stride:            &[i32],
        padding:           &[i32],
        ceil_mode:         bool,
        count_include_pad: bool,
        divisor_override:  Option<i64>) -> Tensor {
    
    todo!();
        /*
            auto gradInput = zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      avg_pool3d_backward_out_cpu_template(
        gradInput,
        gradOutput_,
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
      return gradInput;
        */
}
