crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/Unfold2d.cpp]

#[inline] pub fn cadd<Scalar>(
        z: *mut Scalar,
        x: *const Scalar,
        y: *const Scalar,
        n: i64)  {

    todo!();
        /*
            using Vec = vec::Vectorized<Scalar>;
      char* ptrs[] = {reinterpret_cast<char*>(z),
                      reinterpret_cast<char*>(const_cast<Scalar*>(x)),
                      reinterpret_cast<char*>(const_cast<Scalar*>(y))};
      vectorized_loop(
          ptrs,
          n,
          -1,
          [](Scalar x, Scalar y) -> Scalar { return x + y; },
          [](Vec x, Vec y) -> Vec { return x + y; });
        */
}

pub fn unfolded2d_acc<Scalar>(
    finput_data:   *mut Scalar,
    input_data:    *mut Scalar,
    kh:            i64,
    kw:            i64,
    dh:            i64,
    dw:            i64,
    padh:          i64,
    padw:          i64,
    n_input_plane: i64,
    input_height:  i64,
    input_width:   i64,
    output_height: i64,
    output_width:  i64)  {

    todo!();
        /*
            at::parallel_for(0, n_input_plane, 0, [&](i64 start, i64 end) {
        for (auto nip = start; nip < end; nip++) {
          i64 kw, kh, y, x;
          i64 ix, iy;
          for (kh = 0; kh < kH; kh++) {
            for (kw = 0; kw < kW; kw++) {
              Scalar* src = finput_data +
                  nip * ((usize)kH * kW * output_height * output_width) +
                  kh * ((usize)kW * output_height * output_width) +
                  kw * ((usize)output_height * output_width);
              Scalar* dst =
                  input_data + nip * ((usize)input_height * input_width);
              if (padW > 0 || padH > 0) {
                i64 lpad, rpad;
                for (y = 0; y < output_height; y++) {
                  iy = (i64)y * dH - padH + kh;
                  if (iy < 0 || iy >= input_height) {
                  } else {
                    if (dW == 1) {
                      ix = 0 - padW + kw;
                      lpad = std::max<i64>(0, padW - kw);
                      rpad = std::max<i64>(0, padW - (kW - kw - 1));
                      Scalar* dst_slice =
                          dst + (usize)iy * input_width + ix + lpad;
                      cadd(
                          dst_slice,
                          dst_slice,
                          src + (usize)y * output_width + lpad,
                          output_width - lpad - rpad);
                    } else {
                      for (x = 0; x < output_width; x++) {
                        ix = (i64)x * dW - padW + kw;
                        if (ix < 0 || ix >= input_width) {
                        } else {
                          Scalar* dst_slice = dst + (usize)iy * input_width + ix;
                          *dst_slice = *dst_slice + src[(usize)y * output_width + x];
                        }
                      }
                    }
                  }
                }
              } else {
                for (y = 0; y < output_height; y++) {
                  iy = (i64)y * dH + kh;
                  ix = 0 + kw;
                  if (dW == 1) {
                    Scalar* dst_slice = dst + (usize)iy * input_width + ix;
                    cadd(
                        dst_slice,
                        dst_slice,
                        src + (usize)y * output_width,
                        output_width);
                  } else {
                    for (x = 0; x < output_width; x++) {
                      Scalar* dst_slice =
                          dst + (usize)iy * input_width + ix + x * dW;
                      *dst_slice = *dst_slice + src[(usize)y * output_width + x];
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

/**
  | note: due to write issues, this one cannot
  | be parallelized as well as unfolded2d_copy
  |
  */
pub fn unfolded2d_acc_kernel(
    finput:        &mut Tensor,
    input:         &mut Tensor,
    kh:            i64,
    kw:            i64,
    dh:            i64,
    dw:            i64,
    padh:          i64,
    padw:          i64,
    n_input_plane: i64,
    input_height:  i64,
    input_width:   i64,
    output_height: i64,
    output_width:  i64)  {
    
    todo!();
        /*
            // This function assumes that
      // output_height*dH does not overflow a i64
      // output_width*dW does not overflow a i64

      AT_DISPATCH_FLOATING_TYPES_AND(
          at::ScalarType::BFloat16, input.scalar_type(), "unfolded2d_acc", [&] {
            Scalar* finput_data = finput.data_ptr<Scalar>();
            Scalar* input_data = input.data_ptr<Scalar>();

            unfolded2d_acc(
                finput_data,
                input_data,
                kH,
                kW,
                dH,
                dW,
                padH,
                padW,
                n_input_plane,
                input_height,
                input_width,
                output_height,
                output_width);
          });
        */
}

pub fn unfolded2d_copy<Scalar>(
    input_data:    *mut Scalar,
    finput_data:   *mut Scalar,
    kh:            i64,
    kw:            i64,
    dh:            i64,
    dw:            i64,
    padh:          i64,
    padw:          i64,
    n_input_plane: i64,
    input_height:  i64,
    input_width:   i64,
    output_height: i64,
    output_width:  i64)  {

    todo!();
        /*
            at::parallel_for(
          0, (i64)n_input_plane * kH * kW, 0, [&](i64 start, i64 end) {
            for (auto k = start; k < end; k++) {
              i64 nip = k / (kH * kW);
              i64 rest = k % (kH * kW);
              i64 kh = rest / kW;
              i64 kw = rest % kW;
              i64 x, y;
              i64 ix, iy;
              Scalar* dst = finput_data +
                  nip * ((usize)kH * kW * output_height * output_width) +
                  kh * ((usize)kW * output_height * output_width) +
                  kw * ((usize)output_height * output_width);
              Scalar* src =
                  input_data + nip * ((usize)input_height * input_width);
              if (padW > 0 || padH > 0) {
                i64 lpad, rpad;
                for (y = 0; y < output_height; y++) {
                  iy = (i64)y * dH - padH + kh;
                  if (iy < 0 || iy >= input_height) {
                    memset(
                        dst + (usize)y * output_width,
                        0,
                        sizeof(Scalar) * output_width);
                  } else {
                    if (dW == 1) {
                      ix = 0 - padW + kw;
                      lpad = std::max<i64>(0, padW - kw);
                      rpad = std::max<i64>(0, padW - (kW - kw - 1));
                      if (output_width - rpad - lpad <= 0) {
                        memset(
                            dst + (usize)y * output_width,
                            0,
                            sizeof(Scalar) * output_width);
                      } else {
                        if (lpad > 0)
                          memset(
                              dst + (usize)y * output_width,
                              0,
                              sizeof(Scalar) * lpad);
                        memcpy(
                            dst + (usize)y * output_width + lpad,
                            src + (usize)iy * input_width + ix + lpad,
                            sizeof(Scalar) * (output_width - rpad - lpad));
                        if (rpad > 0)
                          memset(
                              dst + (usize)y * output_width + output_width - rpad,
                              0,
                              sizeof(Scalar) * rpad);
                      }
                    } else {
                      for (x = 0; x < output_width; x++) {
                        ix = (i64)x * dW - padW + kw;
                        if (ix < 0 || ix >= input_width)
                          memset(
                              dst + (usize)y * output_width + x,
                              0,
                              sizeof(Scalar) * 1);
                        else
                          memcpy(
                              dst + (usize)y * output_width + x,
                              src + (usize)iy * input_width + ix,
                              sizeof(Scalar) * (1));
                      }
                    }
                  }
                }
              } else {
                for (y = 0; y < output_height; y++) {
                  iy = (i64)y * dH + kh;
                  ix = 0 + kw;
                  if (dW == 1)
                    memcpy(
                        dst + (usize)y * output_width,
                        src + (usize)iy * input_width + ix,
                        sizeof(Scalar) * output_width);
                  else {
                    for (x = 0; x < output_width; x++)
                      memcpy(
                          dst + (usize)y * output_width + x,
                          src + (usize)iy * input_width + ix + (i64)x * dW,
                          sizeof(Scalar) * (1));
                  }
                }
              }
            }
          });
        */
}

pub fn unfolded2d_copy_kernel(
    finput:        &mut Tensor,
    input:         &mut Tensor,
    kh:            i64,
    kw:            i64,
    dh:            i64,
    dw:            i64,
    padh:          i64,
    padw:          i64,
    n_input_plane: i64,
    input_height:  i64,
    input_width:   i64,
    output_height: i64,
    output_width:  i64)  {

    todo!();
        /*
            // This function assumes that
      // kH*kW does not overflow an int
      // n_input_plane*kH*kW does not overflow a i64
      // output_height*dH does not overflow a i64
      // output_width*dW does not overflow a i64

      AT_DISPATCH_ALL_TYPES_AND(
          at::ScalarType::BFloat16, input.scalar_type(), "unfolded2d_copy", [&] {
            Scalar* input_data = input.data_ptr<Scalar>();
            Scalar* finput_data = finput.data_ptr<Scalar>();

            unfolded2d_copy(
                input_data,
                finput_data,
                kH,
                kW,
                dH,
                dW,
                padH,
                padW,
                n_input_plane,
                input_height,
                input_width,
                output_height,
                output_width);
          });
        */
}

register_dispatch!{
    unfolded2d_copy_stub, 
    &unfolded2d_copy_kernel
}

register_dispatch!{
    unfolded2d_acc_stub, 
    &unfolded2d_acc_kernel
}
