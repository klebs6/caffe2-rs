// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ReflectionPad.cpp]

lazy_static!{
    /*
    TORCH_META_FUNC(reflection_pad1d)(const Tensor& input, IntArrayRef padding) {
      i64 dim_plane = 0;
      i64 dim_w = 1;
      i64 nbatch = 1;

      // allow dim=0 only in the batch dimension.
      TORCH_CHECK(
          (input.ndimension() == 2 && input.size(1) != 0) ||
              (input.ndimension() == 3 && input.size(1) != 0 && input.size(2) != 0),
          "2D or 3D (batch mode) tensor expected for input, but got: ",
          input);

      if (input.ndimension() == 3) {
        nbatch = input.size(0);
        dim_w++;
        dim_plane++;
      }

      /* sizes */
      auto pad_l = padding[0];
      auto pad_r = padding[1];

      i64 nplane = input.size(dim_plane);
      i64 input_w = input.size(dim_w);
      i64 output_w = input_w + pad_l + pad_r;

      TORCH_CHECK(
          pad_l < input_w && pad_r < input_w,
          "Argument #4: Padding size "
          "should be less than the corresponding input dimension, but got: padding (",
          pad_l,
          ", ",
          pad_r,
          ") at dimension ",
          dim_w,
          " of input ",
          input.sizes());

      TORCH_CHECK(
          output_w >= 1,
          2,
          "input (W: ",
          input_w,
          ")is too small. Calculated output W: ",
          output_w);

      if (input.ndimension() == 2) {
        set_output({nplane, output_w}, input.options());
      } else {
        set_output({nbatch, nplane, output_w}, input.options());
      }
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(reflection_pad1d_backward)(const Tensor& grad_output,
        const Tensor& input,
        IntArrayRef padding) {
      i64 dim_plane = 0;
      i64 dim_w = 1;
      i64 nbatch = 1;

      if (input.ndimension() == 3) {
        nbatch = input.size(0);
        dim_w++;
        dim_plane++;
      }

      /* sizes */
      auto pad_l = padding[0];
      auto pad_r = padding[1];
      i64 input_w = input.size(dim_w);
      i64 output_w  = input_w + pad_l + pad_r;

      TORCH_CHECK(
          pad_l < input_w && pad_r < input_w,
          "Argument #4: Padding size "
          "should be less than the corresponding input dimension, but got: padding (",
          pad_l,
          ", ",
          pad_r,
          ") at dimension ",
          dim_w,
          " of input ",
          input.sizes());

      TORCH_CHECK(output_w == grad_output.size(dim_w), "grad_output width unexpected."
        " Expected: ", output_w, ", Got: ", grad_output.size(dim_w));

      set_output(input.sizes(), input.options());
    }
    */
}

pub fn reflection_pad1d_out_frame<Scalar>(
    input_p:  *mut Scalar,
    output_p: *mut Scalar,
    nplane:   i64,
    input_w:  i64,
    output_w: i64,
    pad_l:    i64)  {

    todo!();
        /*
            i64 i_start_x = max(i64(0), -pad_l);
      i64 o_start_x = max(i64(0), pad_l);

      parallel_for(0, nplane, 0, [&](i64 start, i64 end) {
        i64 ip_x;
        for (auto k = start; k < end; k++) {
          for (i64 j = 0; j < output_w; j++) {
            if (j < pad_l) {
              ip_x = pad_l * 2 - j;
            } else if (j >= pad_l && j < input_w + pad_l) {
              ip_x = j;
            } else {
              ip_x = (input_w + pad_l - 1) * 2 - j;
            }
            ip_x = ip_x - o_start_x + i_start_x;

            Scalar *dest_p = output_p + k*output_w + j;
            Scalar *src_p = input_p + k*input_w + ip_x;
            *dest_p = *src_p;
          }
        }
      });
        */
}

#[inline] pub fn reflection_pad1d_out_loop<Scalar>(
    input_p:  *mut Scalar,
    output_p: *mut Scalar,
    nbatch:   i64,
    nplane:   i64,
    input_w:  i64,
    output_w: i64,
    pad_l:    i64)  {

    todo!();
        /*
            parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
        for (auto p = start; p < end; p++) {
          reflection_pad1d_out_frame<Scalar>(
            input_p + p * nplane * input_w,
            output_p + p * nplane * output_w,
            nplane,
            input_w, output_w,
            pad_l);
        }
      });
        */
}

pub fn reflection_pad1d_out_template(
    output:  &Tensor,
    input:   &Tensor,
    padding: &[i32])  {
    
    todo!();
        /*
            i64 dim_plane = 0;
      i64 dim_w = 1;
      i64 nbatch = 1;
      // allow dim=0 only in the batch dimension.
      TORCH_CHECK(
          (input_.ndimension() == 2 && input_.size(1) != 0) ||
          (input_.ndimension() == 3 && input_.size(1) != 0 && input_.size(2) != 0),
          "2D or 3D (batch mode) tensor expected for input, but got: ", input_);

      if (input_.ndimension() == 3) {
        nbatch = input_.size(0);
        dim_w++;
        dim_plane++;
      }

      /* sizes */
      auto pad_l = padding[0];
      auto pad_r = padding[1];

      i64 nplane = input_.size(dim_plane);
      i64 input_w = input_.size(dim_w);
      i64 output_w  = input_w + pad_l + pad_r;

      TORCH_CHECK(pad_l < input_w && pad_r < input_w, "Argument #4: Padding size "
        "should be less than the corresponding input dimension, but got: padding (",
        pad_l, ", ", pad_r, ") at dimension ", dim_w, " of input ", input_.sizes());

      TORCH_CHECK(output_w >= 1 , 2,
        "input (W: ", input_w, ")is too small. Calculated output W: ", output_w);

      /* get contiguous input */
      Tensor input = input_.contiguous();

      /* resize output */
      if (input.ndimension() == 2) {
        output.resize_({nplane, output_w});
        if (input.is_quantized()) {
          AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qreflection_pad1d", [&]() {
            reflection_pad1d_out_frame<Scalar>(
              input.data_ptr<Scalar>(), output.data_ptr<Scalar>(),
              nplane,
              input_w, output_w,
              pad_l);
          });
        } else {
          AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "reflection_pad1d", [&] {
            reflection_pad1d_out_frame<Scalar>(
              input.data_ptr<Scalar>(), output.data_ptr<Scalar>(),
              nplane,
              input_w, output_w,
              pad_l);
          });
        }
      } else {
        output.resize_({nbatch, nplane, output_w});
        if (input.is_quantized()) {
          AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qreflection_pad1d", [&]() {
            reflection_pad1d_out_loop<Scalar>(
              input.data_ptr<Scalar>(), output.data_ptr<Scalar>(),
              nbatch, nplane,
              input_w, output_w,
              pad_l);
          });
        } else {
          AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "reflection_pad1d", [&] {
            reflection_pad1d_out_loop<Scalar>(
              input.data_ptr<Scalar>(), output.data_ptr<Scalar>(),
              nbatch, nplane,
              input_w, output_w,
              pad_l);
          });
        }
      }
        */
}

pub fn reflection_pad1d_backward_out_frame<Scalar>(
    grad_input:  *mut Scalar,
    grad_output: *mut Scalar,
    nplane:      i64,
    input_w:     i64,
    output_w:    i64,
    pad_l:       i64)  {

    todo!();
        /*
            i64 i_start_x = max(i64(0), -pad_l);
      i64 o_start_x = max(i64(0), pad_l);

      parallel_for(0, nplane, 0, [&](i64 start, i64 end) {
        i64 ip_x;
        for (auto k = start; k < end; k++) {
          for (i64 j = 0; j < output_w; j++) {
            if (j < pad_l) {
              ip_x = pad_l * 2 - j;
            } else if (j >= pad_l && j < input_w + pad_l) {
              ip_x = j;
            } else {
              ip_x = (input_w + pad_l - 1) * 2 - j;
            }
            ip_x = ip_x - o_start_x + i_start_x;

            Scalar *src_p = grad_output + k*output_w + j;
            Scalar *dest_p = grad_input + k*input_w + ip_x;
            *dest_p += *src_p;
          }
        }
      });
        */
}

#[inline] pub fn reflection_pad1d_backward_out_loop<Scalar>(
    grad_input:  *mut Scalar,
    grad_output: *mut Scalar,
    nbatch:      i64,
    nplane:      i64,
    input_w:     i64,
    output_w:    i64,
    pad_l:       i64)  {

    todo!();
        /*
            parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
        for (auto p = start; p < end; p++) {
          reflection_pad1d_backward_out_frame<Scalar>(
            grad_input + p * nplane * input_w,
            grad_output + p * nplane * output_w,
            nplane,
            input_w, output_w,
            pad_l);
        }
      });
        */
}

pub fn reflection_pad2d_out_frame<Scalar>(
    input_p:  *mut Scalar,
    output_p: *mut Scalar,
    nplane:   i64,
    input_w:  i64,
    input_h:  i64,
    output_w: i64,
    output_h: i64,
    pad_l:    i64,
    pad_t:    i64)  {

    todo!();
        /*
            auto i_start_x = max(i64(0), -pad_l);
      auto i_start_y = max(i64(0), -pad_t);
      auto o_start_x = max(i64(0), pad_l);
      auto o_start_y = max(i64(0), pad_t);

      parallel_for(0, nplane, 0, [&](i64 start, i64 end) {
        i64 ip_x, ip_y;
        for (auto k = start; k < end; k++) {
          for (i64 i = 0; i < output_h; i++) {
            for (i64 j = 0; j < output_w; j++) {
              if (j < pad_l) {
                ip_x = pad_l * 2 - j;
              } else if (j >= pad_l && j < input_w + pad_l) {
                ip_x = j;
              } else {
                ip_x = (input_w + pad_l - 1) * 2 - j;
              }
              ip_x = ip_x - o_start_x + i_start_x;

              if (i < pad_t) {
                ip_y = pad_t * 2 - i;
              } else if (i >= pad_t && i < input_h + pad_t) {
                ip_y = i;
              } else {
                ip_y = (input_h + pad_t - 1) * 2 - i;
              }
              ip_y = ip_y - o_start_y + i_start_y;

              Scalar *dest_p = output_p + k*output_w*output_h + i * output_w + j;
              Scalar *src_p = input_p + k*input_w*input_h + ip_y * input_w + ip_x;
              *dest_p = *src_p;
            }
          }
        }
      });
        */
}


#[inline] pub fn reflection_pad2d_out_loop<Scalar>(
    input_p:  *mut Scalar,
    output_p: *mut Scalar,
    nbatch:   i64,
    nplane:   i64,
    input_w:  i64,
    input_h:  i64,
    output_w: i64,
    output_h: i64,
    pad_l:    i64,
    pad_t:    i64)  {

    todo!();
        /*
            parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
        for (auto p = start; p < end; p++) {
          reflection_pad2d_out_frame(
            input_p + p * nplane * input_w * input_h,
            output_p + p * nplane * output_w * output_h,
            nplane,
            input_w, input_h, output_w, output_h,
            pad_l, pad_t);
        }
      });
        */
}


pub fn reflection_pad2d_out_template(
        output:  &mut Tensor,
        input:   &Tensor,
        padding: &[i32])  {
    
    todo!();
        /*
            int dim_w = 2;
      int dim_h = 1;
      int dim_slices = 0;
      i64 nbatch = 1;

      bool valid_dims = input_.size(1) != 0 && input_.size(2) != 0;
      TORCH_CHECK(
          (input_.ndimension() == 3 && valid_dims) ||
          (input_.ndimension() == 4 && valid_dims && input_.size(3) != 0),
          "3D or 4D (batch mode) tensor expected for input, but got: ", input_);

      if (input_.ndimension() == 4) {
        nbatch = input_.size(0);
        dim_w++;
        dim_h++;
        dim_slices++;
      }

      /* sizes */
      i64 pad_l = padding[0];
      i64 pad_r = padding[1];
      i64 pad_t = padding[2];
      i64 pad_b = padding[3];

      i64 nplane = input_.size(dim_slices);
      i64 input_h = input_.size(dim_h);
      i64 input_w = input_.size(dim_w);
      i64 output_h = input_h + pad_t + pad_b;
      i64 output_w  = input_w + pad_l + pad_r;

      TORCH_CHECK(pad_l < input_w && pad_r < input_w,
        "Argument #4: Padding size should be less than the corresponding "
        "input dimension, but got: padding (", pad_l, ", ", pad_r,
        ") at dimension ", dim_w, " of input ", input_.ndimension());

      TORCH_CHECK(pad_t < input_h && pad_b < input_h,
        "Argument #6: Padding size should be less than the corresponding "
        "input dimension, but got: padding (", pad_t, ", ", pad_b,
        ") at dimension ", dim_h, " of input ", input_.ndimension());

      TORCH_CHECK(output_w >= 1 || output_h >= 1,
        "input (H: ", input_h, ", W: ", input_w, ")is too small. Calculated "
        "output H: ", output_h, " W: ", output_w);

      /* get contiguous input */
      Tensor input = input_.contiguous();

      if (input.ndimension() == 3) {
        /* resize output */
        output.resize_({nplane, output_h, output_w});
        if (input.is_quantized()) {
          AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qreflection_pad2d", [&] {
            reflection_pad2d_out_frame(
              input.data_ptr<Scalar>(), output.data_ptr<Scalar>(),
              nplane,
              input_w, input_h, output_w, output_h,
              pad_l, pad_t);
          });
        } else {
          AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "reflection_pad2d", [&] {
            reflection_pad2d_out_frame(
              input.data_ptr<Scalar>(), output.data_ptr<Scalar>(),
              nplane,
              input_w, input_h, output_w, output_h,
              pad_l, pad_t);
          });
        }
      } else {
        /* resize output */
        output.resize_({nbatch, nplane, output_h, output_w});
        if (input.is_quantized()) {
          AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qreflection_pad2d", [&] {
            reflection_pad2d_out_loop(
              input.data_ptr<Scalar>(), output.data_ptr<Scalar>(),
              nbatch, nplane,
              input_w, input_h, output_w, output_h,
              pad_l, pad_t);
          });
        } else {
          AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "reflection_pad2d", [&] {
            reflection_pad2d_out_loop(
              input.data_ptr<Scalar>(), output.data_ptr<Scalar>(),
              nbatch, nplane,
              input_w, input_h, output_w, output_h,
              pad_l, pad_t);
          });
        }
      }
        */
}



pub fn reflection_pad2d_backward_out_frame<Scalar>(
        grad_input:  *mut Scalar,
        grad_output: *mut Scalar,
        nplane:      i64,
        input_w:     i64,
        input_h:     i64,
        output_w:    i64,
        output_h:    i64,
        pad_l:       i64,
        pad_t:       i64)  {

    todo!();
        /*
            auto i_start_x = max(i64(0), -pad_l);
      auto i_start_y = max(i64(0), -pad_t);
      auto o_start_x = max(i64(0), pad_l);
      auto o_start_y = max(i64(0), pad_t);

      parallel_for(0, nplane, 0, [&](i64 start, i64 end) {
        i64 ip_x, ip_y;
        for (auto k = start; k < end; k++) {
          for (i64 i = 0; i < output_h; i++) {
            for (i64 j = 0; j < output_w; j++) {
              if (j < pad_l) {
                ip_x = pad_l * 2 - j;
              } else if (j >= pad_l && j < input_w + pad_l) {
                ip_x = j;
              } else {
                ip_x = (input_w + pad_l - 1) * 2 - j;
              }
              ip_x = ip_x - o_start_x + i_start_x;

              if (i < pad_t) {
                ip_y = pad_t * 2 - i;
              } else if (i >= pad_t && i < input_h + pad_t) {
                ip_y = i;
              } else {
                ip_y = (input_h + pad_t - 1) * 2 - i;
              }
              ip_y = ip_y - o_start_y + i_start_y;

              Scalar *src_p =
                grad_output + k * output_w * output_h + i * output_w + j;
              Scalar *dest_p =
                grad_input + k * input_w * input_h + ip_y * input_w + ip_x;
              *dest_p += *src_p;
            }
          }
        }
      });
        */
}



#[inline] pub fn reflection_pad2d_backward_out_loop<Scalar>(
        grad_input:  *mut Scalar,
        grad_output: *mut Scalar,
        nbatch:      i64,
        nplane:      i64,
        input_w:     i64,
        input_h:     i64,
        output_w:    i64,
        output_h:    i64,
        pad_l:       i64,
        pad_t:       i64)  {

    todo!();
        /*
            parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
        for (auto p = start; p < end; p++) {
          reflection_pad2d_backward_out_frame(
            grad_input + p * nplane * input_h * input_w,
            grad_output + p * nplane * output_h * output_w,
            nplane,
            input_w, input_h, output_w, output_h,
            pad_l, pad_t);
        }
      });
        */
}


pub fn reflection_pad2d_backward_out_template(
        grad_input:  &mut Tensor,
        grad_output: &Tensor,
        input:       &Tensor,
        padding:     &[i32])  {
    
    todo!();
        /*
            int dim_w = 2;
      int dim_h = 1;
      int dim_plane = 0;
      i64 nbatch = 1;

      if (input.ndimension() == 4) {
        nbatch = input.size(0);
        dim_w++;
        dim_h++;
        dim_plane++;
      }

      /* sizes */
      i64 pad_l = padding[0];
      i64 pad_r = padding[1];
      i64 pad_t = padding[2];
      i64 pad_b = padding[3];

      i64 nplane = input.size(dim_plane);
      i64 input_h = input.size(dim_h);
      i64 input_w = input.size(dim_w);
      i64 output_h = input_h + pad_t + pad_b;
      i64 output_w  = input_w + pad_l + pad_r;

      TORCH_CHECK(output_w == grad_output_.size(dim_w),
        "gradOutput width unexpected. Expected: ", output_w, ", Got: ",
        grad_output_.size(dim_w));

      TORCH_CHECK(output_h == grad_output_.size(dim_h),
        "gradOutput height unexpected. Expected: ", output_h, ", Got: ",
        grad_output_.size(dim_h));

      /* get contiguous gradOutput */
      Tensor grad_output = grad_output_.contiguous();

      /* backprop */
      if (input.ndimension() == 3) {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
          grad_output.scalar_type(), "reflection_pad2d_backward", [&] {
            reflection_pad2d_backward_out_frame(
              grad_input.data_ptr<Scalar>(), grad_output.data_ptr<Scalar>(),
              nplane,
              input_w, input_h, output_w, output_h,
              pad_l, pad_t);
          }
        );
      } else {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
          grad_output.scalar_type(), "reflection_pad2d_backward", [&] {
            reflection_pad2d_backward_out_loop(
              grad_input.data_ptr<Scalar>(), grad_output.data_ptr<Scalar>(),
              nbatch, nplane,
              input_w, input_h, output_w, output_h,
              pad_l, pad_t);
          }
        );
      }
        */
}


pub fn reflection_pad1d_out_cpu<'a>(
        input:   &Tensor,
        padding: &[i32],
        output:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            reflection_pad1d_out_template(output, input, padding);
      return output;
        */
}



pub fn reflection_pad1d_cpu(
        input:   &Tensor,
        padding: &[i32]) -> Tensor {
    
    todo!();
        /*
            Tensor output;
      if (input.is_quantized()) {
        if (input.qscheme() == kPerTensorAffine) {
          output = _empty_affine_quantized({0}, input.options(),
                                               input.q_scale(),
                                               input.q_zero_point());
        } else {
          TORCH_CHECK(false, "Only per tensor quantization is supported");
        }
      } else {
        output = empty({0}, input.options());
      }
      reflection_pad1d_out_template(output, input, padding);
      return output;
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(reflection_pad1d_out_cpu)
    (const Tensor& input, IntArrayRef padding, const Tensor& output) {
      reflection_pad1d_out_template(output, input, padding);
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(reflection_pad1d_backward_out_cpu)(const Tensor& grad_output_,
        const Tensor& input,
        IntArrayRef padding,
        const Tensor& grad_input) {
      grad_input.zero_();

      i64 dim_plane = 0;
      i64 dim_w = 1;
      i64 nbatch = 1;

      if (input.ndimension() == 3) {
        nbatch = input.size(0);
        dim_w++;
        dim_plane++;
      }

      /* sizes */
      auto pad_l = padding[0];
      auto pad_r = padding[1];
      i64 nplane = input.size(dim_plane);
      i64 input_w = input.size(dim_w);
      i64 output_w  = input_w + pad_l + pad_r;

      /* get contiguous grad_output */
      Tensor grad_output = grad_output_.contiguous();

      /* backprop */
      if (input.ndimension() == 2) {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
          grad_input.scalar_type(), "reflection_pad1d_backward_cpu", [&] {
            reflection_pad1d_backward_out_frame(
              grad_input.data_ptr<Scalar>(), grad_output.data_ptr<Scalar>(),
              nplane,
              input_w, output_w,
              pad_l);
            }
        );
      } else {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
          grad_input.scalar_type(), "reflection_pad1d_backward_cpu", [&] {
            reflection_pad1d_backward_out_loop(
              grad_input.data_ptr<Scalar>(),
              grad_output.data_ptr<Scalar>(),
              nbatch, nplane,
              input_w, output_w,
              pad_l);
          }
        );
      }
    }
    */
}


pub fn reflection_pad2d_out_cpu<'a>(
        input:   &Tensor,
        padding: &[i32],
        output:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            reflection_pad2d_out_template(output, input, padding);
      return output;
        */
}


pub fn reflection_pad2d_cpu(
        input:   &Tensor,
        padding: &[i32]) -> Tensor {
    
    todo!();
        /*
            Tensor output;
      if (input.is_quantized()) {
        if (input.qscheme() == kPerTensorAffine) {
          output = _empty_affine_quantized({0}, input.options(),
                                               input.q_scale(),
                                               input.q_zero_point());
        } else {
          TORCH_CHECK(false, "Only per tensor quantization is supported");
        }
      } else {
        output = empty({0}, input.options());
      }
      reflection_pad2d_out_template(output, input, padding);
      return output;
        */
}


pub fn reflection_pad2d_backward_out_cpu<'a>(
        grad_output: &Tensor,
        input:       &Tensor,
        padding:     &[i32],
        grad_input:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            grad_input.resize_as_(input);
      grad_input.zero_();
      reflection_pad2d_backward_out_template(
        grad_input, grad_output, input, padding);
      return grad_input;
        */
}


pub fn reflection_pad2d_backward_cpu(
        grad_output: &Tensor,
        input:       &Tensor,
        padding:     &[i32]) -> Tensor {
    
    todo!();
        /*
            auto grad_input = zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      reflection_pad2d_backward_out_template(
        grad_input, grad_output, input, padding);
      return grad_input;
        */
}
