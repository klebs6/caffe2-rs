crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/DilatedMaxPool3d.cpp]

pub fn max_pool3d_with_indices_single_out_frame<Scalar>(
        input_p:   *mut Scalar,
        output_p:  *mut Scalar,
        indz_p:    *mut i64,
        nslices:   i64,
        itime:     i64,
        iwidth:    i64,
        iheight:   i64,
        otime:     i64,
        owidth:    i64,
        oheight:   i64,
        kt:        i32,
        kw:        i32,
        kh:        i32,
        dt:        i32,
        dw:        i32,
        dh:        i32,
        pt:        i32,
        pw:        i32,
        ph:        i32,
        dilationt: i32,
        dilationw: i32,
        dilationh: i32)  {

    todo!();
        /*
            parallel_for(0, nslices, 0, [&](i64 start, i64 end) {
        for (auto k = start; k < end; k++)
        {
          /* loop over output */
          i64 i, j, ti;
          Scalar *ip = input_p + k * itime * iwidth * iheight;
          for (ti = 0; ti < otime; ti++)
          {
            for (i = 0; i < oheight; i++)
            {
              for (j = 0; j < owidth; j++)
              {
                /* local pointers */

                i64 start_t = ti * dT - pT;
                i64 start_h = i * dH - pH;
                i64 start_w = j * dW - pW;

                i64 end_t = min(start_t + (kT - 1) * dilationT + 1, itime);
                i64 end_h = min(start_h + (kH - 1) * dilationH + 1, iheight);
                i64 end_w = min(start_w + (kW - 1) * dilationW + 1, iwidth);

                while(start_t < 0)
                  start_t += dilationT;
                while(start_h < 0)
                  start_h += dilationH;
                while(start_w < 0)
                  start_w += dilationW;

                Scalar *op = output_p + k * otime * owidth * oheight
                  + ti * owidth * oheight + i * owidth + j;
                i64 *indzp = indz_p + k * otime * owidth * oheight
                  + ti * owidth * oheight + i * owidth + j;

                /* compute local max: */
                i64 maxindex = start_t * iwidth * iheight + start_h * iwidth + start_w;
                Scalar maxval = -numeric_limits<Scalar>::infinity();

                for (i64 z = start_t; z < end_t; z += dilationT)
                {
                  for (i64 y = start_h; y < end_h; y += dilationH)
                  {
                    for (i64 x = start_w; x < end_w; x += dilationW)
                    {
                      i64 index = z * iwidth * iheight + y * iwidth + x;
                      Scalar val = ip[index];
                      if ((val > maxval) || isnan(val))
                      {
                        maxval = val;
                        maxindex = index;
                      }
                    }
                  }
                }

                // store location of max
                *indzp = maxindex;

                /* set output to local max */
                *op = maxval;
              }
            }
          }
        }
      });
        */
}

pub fn max_pool3d_with_indices_out_frame<Scalar>(
        input_data:   *mut Scalar,
        output_data:  *mut Scalar,
        indices_data: *mut i64,
        nbatch:       i64,
        nslices:      i64,
        istride:      i64,
        ostride:      i64,
        itime:        i64,
        iwidth:       i64,
        iheight:      i64,
        otime:        i64,
        owidth:       i64,
        oheight:      i64,
        kt:           i32,
        kw:           i32,
        kh:           i32,
        dt:           i32,
        dw:           i32,
        dh:           i32,
        pt:           i32,
        pw:           i32,
        ph:           i32,
        dilationt:    i32,
        dilationw:    i32,
        dilationh:    i32)  {

    todo!();
        /*
            parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
        for (auto p = start; p < end; p++)
        {
          max_pool3d_with_indices_single_out_frame(
            input_data   + p * istride,
            output_data  + p * ostride,
            indices_data + p * ostride,
            nslices,
            itime, iwidth, iheight,
            otime, owidth, oheight,
            kT, kW, kH,
            dT, dW, dH,
            pT, pW, pH,
            dilationT, dilationW, dilationH
          );
        }
      });
        */
}

pub fn max_pool3d_with_indices_out_cpu_template(
        output:      &mut Tensor,
        indices:     &mut Tensor,
        input:       &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        ceil_mode:   bool)  {
    
    todo!();
        /*
            // #20866, #22032: Guarantee this for the official C++ API?
      TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
        "max_pool3d: kernel_size must either be a single int, or a tuple of three ints")
      const int kT = safe_downcast<int, i64>(kernel_size[0]);
      const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, i64>(kernel_size[1]);
      const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, i64>(kernel_size[2]);

      TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
        "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints")
      const int dT = stride.empty() ? kT : safe_downcast<int, i64>(stride[0]);
      const int dH = stride.empty() ? kH :
                     stride.size() == 1 ? dT : safe_downcast<int, i64>(stride[1]);
      const int dW = stride.empty() ? kW :
                     stride.size() == 1 ? dT : safe_downcast<int, i64>(stride[2]);

      TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
        "max_pool3d: padding must be either be a single int, or a tuple of three ints");
      const int pT = safe_downcast<int, i64>(padding[0]);
      const int pH = padding.size() == 1 ? pT : safe_downcast<int, i64>(padding[1]);
      const int pW = padding.size() == 1 ? pT : safe_downcast<int, i64>(padding[2]);

      TORCH_CHECK(dilation.size() == 1 || dilation.size() == 3,
        "max_pool3d: dilation must be either a single int, or a tuple of three ints");
      const int dilationT = safe_downcast<int, i64>(dilation[0]);
      const int dilationH = dilation.size() == 1 ? dilationT : safe_downcast<int, i64>(dilation[1]);
      const int dilationW = dilation.size() == 1 ? dilationT : safe_downcast<int, i64>(dilation[2]);

      TORCH_CHECK((input_.ndimension() == 4 || input_.ndimension() == 5),
        "non-empty 4D or 5D (batch mode) tensor expected for input");

      const i64 nslices = input_.size(-4);
      const i64 itime = input_.size(-3);
      const i64 iheight = input_.size(-2);
      const i64 iwidth = input_.size(-1);

      const i64 otime = pooling_output_shape<i64>(itime, kT, pT, dT, dilationT, ceil_mode);
      const i64 oheight = pooling_output_shape<i64>(iheight, kH, pH, dH, dilationH, ceil_mode);
      const i64 owidth = pooling_output_shape<i64>(iwidth, kW, pW, dW, dilationW, ceil_mode);

      pool3d_shape_check(
        input_,
        nslices,
        kT, kH, kW,
        dT, dH, dW,
        pT, pH, pW,
        dilationT, dilationH, dilationW,
        itime, iheight, iwidth,
        otime, oheight, owidth);

      /* get contiguous input */
      Tensor input = input_.contiguous();

      if (input.dim() == 4) { /* non-batch mode */
        /* resize output */
        output.resize_({nslices, otime, oheight, owidth});
        /* indices will contain ti,i,j locations for each output point */
        indices.resize_({nslices, otime, oheight, owidth});

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
          "max_pool3d_with_indices_cpu",
          [&] {
            Scalar *input_data = input.data_ptr<Scalar>();
            Scalar *output_data = output.data_ptr<Scalar>();
            i64 *indices_data = indices.data_ptr<i64>();

            max_pool3d_with_indices_single_out_frame(
              input_data, output_data,
              indices_data,
              nslices,
              itime, iwidth, iheight,
              otime, owidth, oheight,
              kT, kW, kH,
              dT, dW, dH,
              pT, pW, pH,
              dilationT, dilationW, dilationH);
          }
        );
      }
      else { /* batch mode */
        const i64 nbatch = input.size(0);
        const i64 istride = nslices * itime * iwidth * iheight;
        const i64 ostride = nslices * otime * owidth * oheight;

        /* resize output */
        output.resize_({nbatch, nslices, otime, oheight, owidth});
        /* indices will contain ti,i,j locations for each output point */
        indices.resize_({nbatch, nslices, otime, oheight, owidth});

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
          "max_pool3d_with_indices_cpu",
          [&] {
            Scalar *input_data = input.data_ptr<Scalar>();
            Scalar *output_data = output.data_ptr<Scalar>();
            i64 *indices_data = indices.data_ptr<i64>();

            max_pool3d_with_indices_out_frame(
              input_data,
              output_data,
              indices_data,
              nbatch,
              nslices,
              istride, ostride,
              itime, iwidth, iheight,
              otime, owidth, oheight,
              kT, kW, kH,
              dT, dW, dH,
              pT, pW, pH,
              dilationT, dilationW, dilationH);
         }
       );
      }
        */
}

pub fn max_pool3d_with_indices_backward_single_out_frame<Scalar>(
        grad_input_p:  *mut Scalar,
        grad_output_p: *mut Scalar,
        indz_p:        *mut i64,
        nslices:       i64,
        itime:         i64,
        iwidth:        i64,
        iheight:       i64,
        otime:         i64,
        owidth:        i64,
        oheight:       i64,
        dt:            i32,
        dw:            i32,
        dh:            i32,
        pt:            i32,
        pw:            i32,
        ph:            i32,
        dilationt:     i32,
        dilationw:     i32,
        dilationh:     i32)  {

    todo!();
        /*
            parallel_for(0, nslices, 0, [&](i64 start, i64 end) {
        for (auto k = start; k < end; k++)
        {
          Scalar *gradInput_p_k  = gradInput_p  + k * itime * iwidth * iheight;
          Scalar *gradOutput_p_k = gradOutput_p + k * otime * owidth * oheight;
          i64 *indz_p_k = indz_p + k * otime * owidth * oheight;

          /* calculate max points */
          i64 ti, i, j;
          for (ti = 0; ti < otime; ti++)
          {
            for (i = 0; i < oheight; i++)
            {
              for (j = 0; j < owidth; j++)
              {
                /* retrieve position of max */
                i64 index = ti * oheight * owidth + i * owidth + j;
                i64 maxp = indz_p_k[index];

                if (maxp != -1) {
                  /* update gradient */
                  gradInput_p_k[maxp] += gradOutput_p_k[index];
                }
              }
            }
          }
        }
      });
        */
}

pub fn max_pool3d_with_indices_backward_out_frame<Scalar>(
        grad_input_data:  *mut Scalar,
        grad_output_data: *mut Scalar,
        indices_data:     *mut i64,
        nbatch:           i64,
        nslices:          i64,
        istride:          i64,
        ostride:          i64,
        itime:            i64,
        iwidth:           i64,
        iheight:          i64,
        otime:            i64,
        owidth:           i64,
        oheight:          i64,
        dt:               i32,
        dw:               i32,
        dh:               i32,
        pt:               i32,
        pw:               i32,
        ph:               i32,
        dilationt:        i32,
        dilationw:        i32,
        dilationh:        i32)  {

    todo!();
        /*
            parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
        for (auto p = start; p < end; p++)
        {
          max_pool3d_with_indices_backward_single_out_frame<Scalar>(
            gradInput_data + p * istride,
            gradOutput_data + p * ostride,
            indices_data + p * ostride,
            nslices,
            itime, iwidth, iheight,
            otime, owidth, oheight,
            dT, dW, dH,
            pT, pW, pH,
            dilationT, dilationW, dilationH
          );
        }
      });
        */
}

pub fn max_pool3d_with_indices_backward_out_cpu_template(
        grad_input:  &mut Tensor,
        grad_output: &Tensor,
        input:       &Tensor,
        indices:     &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        ceil_mode:   bool) -> &mut Tensor {
    
    todo!();
        /*
            // #20866, #22032: Guarantee this for the official C++ API?
      TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
        "max_pool3d: kernel_size must either be a single int, or a tuple of three ints")
      const int kT = safe_downcast<int, i64>(kernel_size[0]);
      const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, i64>(kernel_size[1]);
      const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, i64>(kernel_size[2]);

      TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
        "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints")
      const int dT = stride.empty() ? kT : safe_downcast<int, i64>(stride[0]);
      const int dH = stride.empty() ? kH :
                     stride.size() == 1 ? dT : safe_downcast<int, i64>(stride[1]);
      const int dW = stride.empty() ? kW :
                     stride.size() == 1 ? dT : safe_downcast<int, i64>(stride[2]);

      TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
        "max_pool3d: padding must be either be a single int, or a tuple of three ints");
      const int pT = safe_downcast<int, i64>(padding[0]);
      const int pH = padding.size() == 1 ? pT : safe_downcast<int, i64>(padding[1]);
      const int pW = padding.size() == 1 ? pT : safe_downcast<int, i64>(padding[2]);

      TORCH_CHECK(dilation.size() == 1 || dilation.size() == 3,
        "max_pool3d: dilation must be either a single int, or a tuple of three ints");
      const int dilationT = safe_downcast<int, i64>(dilation[0]);
      const int dilationH = dilation.size() == 1 ? dilationT : safe_downcast<int, i64>(dilation[1]);
      const int dilationW = dilation.size() == 1 ? dilationT : safe_downcast<int, i64>(dilation[2]);

      TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
        "non-empty 4D or 5D (batch mode) tensor expected for input");

      const i64 nslices = input.size(-4);
      const i64 itime = input.size(-3);
      const i64 iheight = input.size(-2);
      const i64 iwidth = input.size(-1);

      /* get contiguous gradOutput */
      Tensor gradOutput = gradOutput_.contiguous();

      /* resize */
      gradInput.resize_as_(input);
      gradInput.zero_();

      const i64 otime = gradOutput.size(-3);
      const i64 oheight = gradOutput.size(-2);
      const i64 owidth = gradOutput.size(-1);

      max_pool3d_backward_shape_check(
        input,
        gradOutput,
        indices,
        nslices,
        kT, kH, kW,
        dT, dH, dW,
        pT, pH, pW,
        dilationT, dilationH, dilationW,
        itime, iheight, iwidth,
        otime, oheight, owidth);

      /* backprop */
      if (input.ndimension() == 4) /* non-batch mode*/
      {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
          "max_pool3d_with_indices_backward",
          [&] {
            /* get raw pointers */
            Scalar *gradInput_data = gradInput.data_ptr<Scalar>();
            Scalar *gradOutput_data = gradOutput.data_ptr<Scalar>();
            i64 *indices_data = indices.data_ptr<i64>();

            max_pool3d_with_indices_backward_single_out_frame(
              gradInput_data, gradOutput_data,
              indices_data,
              nslices,
              itime, iwidth, iheight,
              otime, owidth, oheight,
              dT, dW, dH,
              pT, pW, pH,
              dilationT, dilationW, dilationH);
          }
        );
      }
      else /* batch mode */
      {
        const i64 nbatch = input.size(0);
        const i64 istride = nslices * itime * iwidth * iheight;
        const i64 ostride = nslices * otime * owidth * oheight;

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
          "max_pool3d_with_indices_backward",
          [&] {
            Scalar *gradInput_data = gradInput.data_ptr<Scalar>();
            Scalar *gradOutput_data = gradOutput.data_ptr<Scalar>();
            i64 *indices_data = indices.data_ptr<i64>();

            max_pool3d_with_indices_backward_out_frame<Scalar>(
              gradInput_data,
              gradOutput_data,
              indices_data,
              nbatch,
              nslices,
              istride, ostride,
              itime, iwidth, iheight,
              otime, owidth, oheight,
              dT, dW, dH,
              pT, pW, pH,
              dilationT, dilationW, dilationH);
          }
        );
      }

      return gradInput;
        */
}

pub fn max_pool3d_with_indices_out_cpu(
        input:       &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        ceil_mode:   bool,
        output:      &mut Tensor,
        indices:     &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            max_pool3d_with_indices_out_cpu_template(
        output,
        indices,
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode);
      return tuple<Tensor&, Tensor&>(output, indices);
        */
}

pub fn max_pool3d_with_indices_cpu(
        input:       &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        ceil_mode:   bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            NoNamesGuard guard;

      Tensor output = empty({0}, input.options());
      Tensor indices = empty({0}, input.options().dtype(kLong));
      max_pool3d_with_indices_out_cpu_template(
        output,
        indices,
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode);

      guard.reset();
      namedinference::propagate_names(output, input);
      namedinference::propagate_names(indices, input);

      return tuple<Tensor, Tensor>(output, indices);
        */
}

pub fn max_pool3d_with_indices_backward_out_cpu(
        grad_output: &Tensor,
        input:       &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        ceil_mode:   bool,
        indices:     &Tensor,
        grad_input:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            max_pool3d_with_indices_backward_out_cpu_template(
        gradInput,
        gradOutput_,
        input,
        indices,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode);
      return gradInput;
        */
}

pub fn max_pool3d_with_indices_backward_cpu(
        grad_output: &Tensor,
        input:       &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        ceil_mode:   bool,
        indices:     &Tensor) -> Tensor {
    
    todo!();
        /*
            auto gradInput = zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      max_pool3d_with_indices_backward_out_cpu_template(
        gradInput,
        gradOutput_,
        input,
        indices,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode);
      return gradInput;
        */
}
