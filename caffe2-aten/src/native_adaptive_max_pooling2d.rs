crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/AdaptiveMaxPooling2d.cpp]

lazy_static!{
    /*
    TORCH_META_FUNC(adaptive_max_pool2d) (const Tensor& input, IntArrayRef output_size) {
      for (i64 i = 0; i < input.ndimension(); i++) {
        TORCH_CHECK(
            input.size(i) > 0,
            "adaptive_max_pool2d: expected input to have non-empty spatial dimensions, "
            "but input has sizes ",
            input.sizes(),
            " with dimension ",
            i,
            " being "
            "empty");
      }

      TORCH_CHECK(
          (input.ndimension() == 3 || input.ndimension() == 4),
          "non-empty 3D or 4D (batch mode) tensor expected for input");

      TORCH_CHECK(
          output_size.size() == 2,
          "adaptive_max_pool2d: internal error: output_size.size() must be 2");

      int dimH = 1;
      i64 sizeB = 1;
      i64 sizeD = 0;

      if (input.ndimension() == 4) {
        sizeB = input.size(0);
        dimH++;
      }

      sizeD = input.size(dimH - 1);

      i64 osizeH = output_size[0];
      i64 osizeW = output_size[1];

      /* resize output */
      if (input.ndimension() == 3) {
        set_output(0, {sizeD, osizeH, osizeW}, input.options());
        /* indices will contain i,j locations for each output point */
        set_output(1, {sizeD, osizeH, osizeW}, input.options().dtype(kLong));
      } else {
        set_output(0, {sizeB, sizeD, osizeH, osizeW}, input.options());
        /* indices will contain i,j locations for each output point */
        set_output(1, {sizeB, sizeD, osizeH, osizeW}, input.options().dtype(kLong));
      }
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(adaptive_max_pool2d_backward)
    (const Tensor& gradOutput, const Tensor& input, const Tensor& indices) {
      set_output(0, input.sizes(), input.options());
    }
    */
}

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

// #define START_IND(a,b,c) a * c / b
// #define END_IND(a,b,c)  (a + 1) * c / b + ((a + 1) * c % b > 0)?1:0

// 4d tensor B x D x H x W
pub fn adaptive_max_pool2d_single_out_frame<Scalar>(
    input_p:  *mut Scalar,
    output_p: *mut Scalar,
    ind_p:    *mut i64,
    sized:    i64,
    isizeh:   i64,
    isizew:   i64,
    osizeh:   i64,
    osizew:   i64,
    istrided: i64,
    istrideh: i64,
    istridew: i64)  {

    todo!();
        /*
            at::parallel_for(0, sizeD, 0, [&](i64 start, i64 end) {
        for (auto d = start; d < end; d++)
        {
          /* loop over output */
          i64 oh, ow;
          for(oh = 0; oh < osizeH; oh++)
          {
            int istartH = start_index(oh, osizeH, isizeH);
            int iendH   = end_index(oh, osizeH, isizeH);
            int kH = iendH - istartH;

            for(ow = 0; ow < osizeW; ow++)
            {
              int istartW = start_index(ow, osizeW, isizeW);
              int iendW   = end_index(ow, osizeW, isizeW);
              int kW = iendW - istartW;

              /* local pointers */
              Scalar *ip = input_p   + d*istrideD + istartH*istrideH + istartW*istrideW;
              Scalar *op = output_p  + d*osizeH*osizeW + oh*osizeW + ow;
              i64 *indp = ind_p   + d*osizeH*osizeW + oh*osizeW + ow;

              /* compute local max: */
              int ih=0, iw=0;
              i64 maxindex = (ih+istartH)*isizeW + (iw+istartW);
              Scalar maxval = -std::numeric_limits<Scalar>::infinity();
              for(ih=0; ih < kH; ih++)
              {
                for(iw=0; iw < kW; iw++)
                {
                  Scalar val = *(ip + ih*istrideH + iw*istrideW);
                  if ((val > maxval) || std::isnan(val))
                  {
                    maxval = val;
                    maxindex = (ih+istartH)*isizeW + (iw+istartW);
                  }
                }
              }

              /* set output to local max */
              *op = maxval;

              /* store location of max */
              *indp = maxindex;
            }
          }
        }
      });
        */
}

pub fn adaptive_max_pool2d_out_frame<Scalar>(
    input_data:   *mut Scalar,
    output_data:  *mut Scalar,
    indices_data: *mut i64,
    sizeb:        i64,
    sized:        i64,
    isizeh:       i64,
    isizew:       i64,
    osizeh:       i64,
    osizew:       i64,
    istrideb:     i64,
    istrided:     i64,
    istrideh:     i64,
    istridew:     i64)  {

    todo!();
        /*
            at::parallel_for(0, sizeB, 0, [&](i64 start, i64 end) {
        for (auto b = start; b < end; b++) {
          adaptive_max_pool2d_single_out_frame<Scalar>(input_data+b*istrideB, output_data+b*sizeD*osizeH*osizeW,
                                                         indices_data+b*sizeD*osizeH*osizeW,
                                                         sizeD,
                                                         isizeH, isizeW,
                                                         osizeH, osizeW,
                                                         istrideD,
                                                         istrideH, istrideW);
        }
      });
        */
}

pub fn adaptive_max_pool2d_backward_single_out_frame<Scalar>(
    grad_input_p:  *mut Scalar,
    grad_output_p: *mut Scalar,
    indices:       *mut i64,
    sized:         i64,
    isizeh:        i64,
    isizew:        i64,
    osizeh:        i64,
    osizew:        i64)  {

    todo!();
        /*
            at::parallel_for(0, sizeD, 0, [&](i64 start, i64 end) {
        for (auto d = start; d < end; d++)
        {
          Scalar *gradInput_p_d = gradInput_p + d*isizeH*isizeW;
          Scalar *gradOutput_p_d = gradOutput_p + d*osizeH*osizeW;
          i64 *ind_p_d = indices + d*osizeH*osizeW;

          /* calculate max points */
          i64 oh, ow;
          for(oh = 0; oh < osizeH; oh++)
          {
            for(ow = 0; ow < osizeW; ow++)
            {
              /* retrieve position of max */
              i64 maxp = ind_p_d[oh*osizeW + ow];

              /* update gradient */
              gradInput_p_d[maxp] += gradOutput_p_d[oh*osizeW + ow];
            }
          }
        }
      });
        */
}

pub fn adaptive_max_pool2d_backward_out_frame<Scalar>(
    grad_input_data:  *mut Scalar,
    grad_output_data: *mut Scalar,
    indices_data:     *mut i64,
    sizeb:            i64,
    sized:            i64,
    isizeh:           i64,
    isizew:           i64,
    osizeh:           i64,
    osizew:           i64)  {

    todo!();
        /*
            at::parallel_for(0, sizeB, 0, [&](i64 start, i64 end) {
        for (auto b = start; b < end; b++) {
          adaptive_max_pool2d_backward_single_out_frame<Scalar>(gradInput_data+b*sizeD*isizeH*isizeW,
                                                                  gradOutput_data+b*sizeD*osizeH*osizeW,
                                                                  indices_data+b*sizeD*osizeH*osizeW,
                                                                  sizeD,
                                                                  isizeH, isizeW,
                                                                  osizeH, osizeW);
        }
      });
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(adaptive_max_pool2d_out_cpu)
    (const Tensor& input, IntArrayRef output_size, const Tensor& output, const Tensor& indices) {
      int dimW = 2;
      int dimH = 1;
      i64 sizeB = 1;
      i64 sizeD = 0;
      i64 isizeH = 0;
      i64 isizeW = 0;

      i64 istrideD = 0;
      i64 istrideH = 0;
      i64 istrideW = 0;
      i64 istrideB = 0;

      if (input.ndimension() == 4) {
        istrideB = input.stride(0);
        sizeB = input.size(0);
        dimW++;
        dimH++;
      }

      /* sizes */
      sizeD = input.size(dimH - 1);
      isizeH = input.size(dimH);
      isizeW = input.size(dimW);
      /* strides */
      istrideD = input.stride(dimH - 1);
      istrideH = input.stride(dimH);
      istrideW = input.stride(dimW);

      i64 osizeH = output_size[0];
      i64 osizeW = output_size[1];

      /* resize output */
      if (input.ndimension() == 3) {
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "adaptive_max_pool2d_cpu", [&] {
              auto input_data = input.data_ptr<Scalar>();
              auto output_data = output.data_ptr<Scalar>();
              auto indices_data = indices.data_ptr<i64>();

              adaptive_max_pool2d_single_out_frame<Scalar>(
                  input_data,
                  output_data,
                  indices_data,
                  sizeD,
                  isizeH,
                  isizeW,
                  osizeH,
                  osizeW,
                  istrideD,
                  istrideH,
                  istrideW);
            });
      } else {
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "adaptive_max_pool2d_cpu", [&] {
              auto input_data = input.data_ptr<Scalar>();
              auto output_data = output.data_ptr<Scalar>();
              auto indices_data = indices.data_ptr<i64>();

              adaptive_max_pool2d_out_frame<Scalar>(
                  input_data,
                  output_data,
                  indices_data,
                  sizeB,
                  sizeD,
                  isizeH,
                  isizeW,
                  osizeH,
                  osizeW,
                  istrideB,
                  istrideD,
                  istrideH,
                  istrideW);
            });
      }
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(adaptive_max_pool2d_backward_out_cpu)
    (const Tensor& gradOutput,
     const Tensor& input,
     const Tensor& indices,
     const Tensor& gradInput) {
      int dimW = 2;
      int dimH = 1;
      i64 sizeB = 1;
      int sizeD;
      int isizeH;
      int isizeW;
      int osizeH;
      int osizeW;

      /* get contiguous gradOutput */
      auto gradOutput_ = gradOutput.contiguous();

      /* zero */
      gradInput.zero_();

      if (input.ndimension() == 4) {
        sizeB = input.size(0);
        dimW++;
        dimH++;
      }

      sizeD = input.size(dimH - 1);
      isizeH = input.size(dimH);
      isizeW = input.size(dimW);
      osizeH = gradOutput_.size(dimH);
      osizeW = gradOutput_.size(dimW);

      /* backprop */
      if (input.ndimension() == 3) {
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "adaptive_max_pool2d_backward", [&] {
              /* get raw pointers */
              Scalar* gradInput_data = gradInput.data_ptr<Scalar>();
              Scalar* gradOutput_data = gradOutput_.data_ptr<Scalar>();
              i64* indices_data = indices.data_ptr<i64>();

              adaptive_max_pool2d_backward_single_out_frame<Scalar>(
                  gradInput_data,
                  gradOutput_data,
                  indices_data,
                  sizeD,
                  isizeH,
                  isizeW,
                  osizeH,
                  osizeW);
            });
      } else {
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "adaptive_max_pool2d_backward", [&] {
              /* get raw pointers */
              Scalar* gradInput_data = gradInput.data_ptr<Scalar>();
              Scalar* gradOutput_data = gradOutput_.data_ptr<Scalar>();
              i64* indices_data = indices.data_ptr<i64>();

              adaptive_max_pool2d_backward_out_frame<Scalar>(
                  gradInput_data,
                  gradOutput_data,
                  indices_data,
                  sizeB,
                  sizeD,
                  isizeH,
                  isizeW,
                  osizeH,
                  osizeW);
            });
      }
     }
    */
}

