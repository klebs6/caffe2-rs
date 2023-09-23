crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/AdaptiveMaxPooling3d.cpp]

lazy_static!{
    /*
    TORCH_META_FUNC(adaptive_max_pool3d) (const Tensor& input, IntArrayRef output_size) {
      for (i64 i = 0; i < input.ndimension(); i++) {
        TORCH_CHECK(
            input.size(i) > 0,
            "adaptive_max_pool3d: expected input to have non-empty spatial dimensions, "
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

      TORCH_CHECK(
          output_size.size() == 3,
          "adaptive_max_pool3d: internal error: output_size.size() must be 3");

      int dimD = 0;
      i64 sizeB = 1;
      i64 sizeD = 0;

      if (input.ndimension() == 5) {
        sizeB = input.size(0);
        dimD++;
      }

      /* sizes */
      sizeD = input.size(dimD);

      i64 osizeT = output_size[0];
      i64 osizeH = output_size[1];
      i64 osizeW = output_size[2];

      /* resize output */
      if (input.ndimension() == 4) {
        set_output(0, {sizeD, osizeT, osizeH, osizeW}, input.options());
        /* indices will contain max input locations for each output point */
        set_output(1, {sizeD, osizeT, osizeH, osizeW}, input.options().dtype(kLong));
      } else {
        set_output(0, {sizeB, sizeD, osizeT, osizeH, osizeW}, input.options());
        /* indices will contain max input locations for each output point */
        set_output(1, {sizeB, sizeD, osizeT, osizeH, osizeW}, input.options().dtype(kLong));
      }
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(adaptive_max_pool3d_backward)
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
            return (int)floor((float)(a * c) / b);
        */
}

#[inline] pub fn end_index(
        a: i32,
        b: i32,
        c: i32) -> i32 {
    
    todo!();
        /*
            return (int)ceil((float)((a + 1) * c) / b);
        */
}

// #define START_IND(a,b,c) a * c / b
// #define END_IND(a,b,c)  (a + 1) * c / b + ((a + 1) * c % b > 0)?1:0

// 5d tensor B x D x T x H x W
pub fn adaptive_max_pool3d_single_out_frame<Scalar>(
        input_p:  *mut Scalar,
        output_p: *mut Scalar,
        ind_p:    *mut i64,
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
            parallel_for(0, sizeD, 0, [&](i64 start, i64 end) {
        for (auto d = start; d < end; d++)
        {
          /* loop over output */
          i64 ot, oh, ow;
          for(ot = 0; ot < osizeT; ot++)
          {
            i64 istartT = start_index(ot, osizeT, isizeT);
            i64 iendT   = end_index(ot, osizeT, isizeT);
            i64 kT = iendT - istartT;

            for(oh = 0; oh < osizeH; oh++)
            {
              i64 istartH = start_index(oh, osizeH, isizeH);
              i64 iendH   = end_index(oh, osizeH, isizeH);
              i64 kH = iendH - istartH;

              for(ow = 0; ow < osizeW; ow++)
              {

                i64 istartW = start_index(ow, osizeW, isizeW);
                i64 iendW   = end_index(ow, osizeW, isizeW);
                i64 kW = iendW - istartW;

                /* local pointers */
                Scalar *ip = input_p   + d*istrideD + istartT *istrideT + istartH*istrideH + istartW*istrideW;
                Scalar *op = output_p  + d*osizeT*osizeH*osizeW + ot*osizeH*osizeW + oh*osizeW + ow;
                i64 *indp = ind_p   + d*osizeT*osizeH*osizeW + ot*osizeH*osizeW + oh*osizeW + ow;

                /* compute local max: */
                i64 it = 0, ih = 0, iw = 0;
                i64 maxindex = (it+istartT)*isizeH*isizeW + (ih+istartH)*isizeW + (iw+istartW);
                Scalar maxval = -numeric_limits<Scalar>::infinity();
                for(it = 0; it < kT; it++)
                {
                  for(ih = 0; ih < kH; ih++)
                  {
                    for(iw = 0; iw < kW; iw++)
                    {
                      Scalar val = *(ip + it*istrideT + ih*istrideH + iw*istrideW);
                      if ((val > maxval) || isnan(val))
                      {
                        maxval = val;
                        maxindex = (it+istartT)*isizeH*isizeW + (ih+istartH)*isizeW + (iw+istartW);
                      }
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
        }
      });
        */
}

pub fn adaptive_max_pool3d_out_frame<Scalar>(
        input_data:   *mut Scalar,
        output_data:  *mut Scalar,
        indices_data: *mut i64,
        sizeb:        i64,
        sized:        i64,
        isizet:       i64,
        isizeh:       i64,
        isizew:       i64,
        osizet:       i64,
        osizeh:       i64,
        osizew:       i64,
        istrideb:     i64,
        istrided:     i64,
        istridet:     i64,
        istrideh:     i64,
        istridew:     i64)  {

    todo!();
        /*
            parallel_for(0, sizeB, 0, [&](i64 start, i64 end) {
        for (auto b = start; b < end; b++)
        {
          adaptive_max_pool3d_single_out_frame<Scalar>(input_data+b*istrideB, output_data+b*sizeD*osizeT*osizeH*osizeW,
                                                         indices_data+b*sizeD*osizeT*osizeH*osizeW,
                                                         sizeD,
                                                         isizeT, isizeH, isizeW,
                                                         osizeT, osizeH, osizeW,
                                                         istrideD, istrideT,
                                                         istrideH, istrideW);
        }
      });
        */
}

pub fn adaptive_max_pool3d_backward_single_out_frame<Scalar>(
        grad_input_p:  *mut Scalar,
        grad_output_p: *mut Scalar,
        ind_p:         *mut i64,
        sized:         i64,
        isizet:        i64,
        isizeh:        i64,
        isizew:        i64,
        osizet:        i64,
        osizeh:        i64,
        osizew:        i64)  {

    todo!();
        /*
            parallel_for(0, sizeD, 0, [&](i64 start, i64 end) {
        for (auto d = start; d < end; d++)
        {
          Scalar *gradInput_p_d = gradInput_p + d*isizeT*isizeH*isizeW;
          Scalar *gradOutput_p_d = gradOutput_p + d*osizeT*osizeH*osizeW;
          i64 *ind_p_d = ind_p + d*osizeT*osizeH*osizeW;

          /* calculate max points */
          i64 ot, oh, ow;
          for(ot = 0; ot < osizeT; ot++)
          {
            for(oh = 0; oh < osizeH; oh++)
            {
              for(ow = 0; ow < osizeW; ow++)
              {
                /* retrieve position of max */
                i64 maxp = ind_p_d[ot*osizeH*osizeW + oh*osizeW + ow];

                /* update gradient */
                gradInput_p_d[maxp] += gradOutput_p_d[ot*osizeH*osizeW + oh*osizeW + ow];
              }
            }
          }
        }
      });
        */
}

pub fn adaptive_max_pool3d_backward_out_frame<Scalar>(
        grad_input_data:  *mut Scalar,
        grad_output_data: *mut Scalar,
        indices_data:     *mut i64,
        sizeb:            i64,
        sized:            i64,
        isizet:           i64,
        isizeh:           i64,
        isizew:           i64,
        osizet:           i64,
        osizeh:           i64,
        osizew:           i64)  {

    todo!();
        /*
            parallel_for(0, sizeB, 0, [&](i64 start, i64 end) {
        for (auto b = start; b < end; b++)
        {
          adaptive_max_pool3d_backward_single_out_frame<Scalar>(gradInput_data+b*sizeD*isizeT*isizeH*isizeW, gradOutput_data+b*sizeD*osizeT*osizeH*osizeW,
                                                                  indices_data+b*sizeD*osizeT*osizeH*osizeW,
                                                                  sizeD,
                                                                  isizeT, isizeH, isizeW,
                                                                  osizeT, osizeH, osizeW);
        }
      });
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(adaptive_max_pool3d_out_cpu)
    (const Tensor& input, IntArrayRef output_size, const Tensor& output, const Tensor& indices) {
      int dimD = 0;
      int dimT = 1;
      int dimH = 2;
      int dimW = 3;
      i64 sizeB = 1;
      i64 sizeD = 0;
      i64 isizeT = 0;
      i64 isizeH = 0;
      i64 isizeW = 0;

      i64 istrideB = 0;
      i64 istrideD = 0;
      i64 istrideT = 0;
      i64 istrideH = 0;
      i64 istrideW = 0;

      if (input.ndimension() == 5) {
        istrideB = input.stride(0);
        sizeB = input.size(0);
        dimD++;
        dimT++;
        dimH++;
        dimW++;
      }

      /* sizes */
      sizeD = input.size(dimD);
      isizeT = input.size(dimT);
      isizeH = input.size(dimH);
      isizeW = input.size(dimW);
      /* strides */
      istrideD = input.stride(dimD);
      istrideT = input.stride(dimT);
      istrideH = input.stride(dimH);
      istrideW = input.stride(dimW);

      i64 osizeT = output_size[0];
      i64 osizeH = output_size[1];
      i64 osizeW = output_size[2];

      if (input.ndimension() == 4) {
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "adaptive_max_pool3d_cpu", [&] {
              auto input_data = input.data_ptr<Scalar>();
              auto output_data = output.data_ptr<Scalar>();
              auto indices_data = indices.data_ptr<i64>();

              adaptive_max_pool3d_single_out_frame<Scalar>(
                  input_data,
                  output_data,
                  indices_data,
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
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "adaptive_max_pool3d_cpu", [&] {
              auto input_data = input.data_ptr<Scalar>();
              auto output_data = output.data_ptr<Scalar>();
              auto indices_data = indices.data_ptr<i64>();

              adaptive_max_pool3d_out_frame<Scalar>(
                  input_data,
                  output_data,
                  indices_data,
                  sizeB,
                  sizeD,
                  isizeT,
                  isizeH,
                  isizeW,
                  osizeT,
                  osizeH,
                  osizeW,
                  istrideB,
                  istrideD,
                  istrideT,
                  istrideH,
                  istrideW);
            });
      }
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(adaptive_max_pool3d_backward_out_cpu)
    (const Tensor& gradOutput,
     const Tensor& input,
     const Tensor& indices,
     const Tensor& gradInput) {
      int dimD = 0;
      int dimT = 1;
      int dimH = 2;
      int dimW = 3;
      i64 sizeB = 1;
      i64 sizeD;
      i64 isizeT;
      i64 isizeH;
      i64 isizeW;
      i64 osizeT;
      i64 osizeH;
      i64 osizeW;

      /* get contiguous gradOutput */
      auto gradOutput_ = gradOutput.contiguous();

      /* resize */
      gradInput.zero_();

      if (input.ndimension() == 5) {
        sizeB = input.size(0);
        dimD++;
        dimT++;
        dimH++;
        dimW++;
      }

      /* sizes */
      sizeD = input.size(dimD);
      isizeT = input.size(dimT);
      isizeH = input.size(dimH);
      isizeW = input.size(dimW);
      osizeT = gradOutput_.size(dimT);
      osizeH = gradOutput_.size(dimH);
      osizeW = gradOutput_.size(dimW);

      /* backprop */
      if (input.ndimension() == 4) {
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "adaptive_max_pool3d_backward", [&] {
              /* get raw pointers */
              Scalar* gradInput_data = gradInput.data_ptr<Scalar>();
              Scalar* gradOutput_data = gradOutput_.data_ptr<Scalar>();
              i64* indices_data = indices.data_ptr<i64>();

              adaptive_max_pool3d_backward_single_out_frame<Scalar>(
                  gradInput_data,
                  gradOutput_data,
                  indices_data,
                  sizeD,
                  isizeT,
                  isizeH,
                  isizeW,
                  osizeT,
                  osizeH,
                  osizeW);
            });
      } else {
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "adaptive_max_pool3d_backward", [&] {
              /* get raw pointers */
              Scalar* gradInput_data = gradInput.data_ptr<Scalar>();
              Scalar* gradOutput_data = gradOutput_.data_ptr<Scalar>();
              i64* indices_data = indices.data_ptr<i64>();

              adaptive_max_pool3d_backward_out_frame<Scalar>(
                  gradInput_data,
                  gradOutput_data,
                  indices_data,
                  sizeB,
                  sizeD,
                  isizeT,
                  isizeH,
                  isizeW,
                  osizeT,
                  osizeH,
                  osizeW);
            });
      }
    }
    */
}
