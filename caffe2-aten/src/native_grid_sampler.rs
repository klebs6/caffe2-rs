crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/GridSampler.h]

pub enum GridSamplerInterpolation {
    Bilinear, 
    Nearest, 
    Bicubic
}

pub enum GridSamplerPadding {
    Zeros, 
    Border, 
    Reflection
}

/**
  | Unnormalizes a coordinate from the -1 to +1
  | scale to its pixel index value, where we view
  | each pixel as an area between (idx - 0.5) and
  | (idx + 0.5).
  |
  | if align_corners: -1 and +1 get sent to the
  | centers of the corner pixels
  |
  |     -1 --> 0
  |     +1 --> (size - 1)
  |     scale_factor = (size - 1) / 2
  |
  | if not align_corners: -1 and +1 get sent to the
  | image edges
  |
  |     -1 --> -0.5
  |     +1 --> (size - 1) + 0.5 == size - 0.5
  |     scale_factor = size / 2
  */
#[inline] pub fn grid_sampler_unnormalize<Scalar>(
        coord:         Scalar,
        size:          i64,
        align_corners: bool) -> Scalar {

    todo!();
        /*
            if (align_corners) {
        // unnormalize coord from [-1, 1] to [0, size - 1]
        return ((coord + 1) / 2) * (size - 1);
      } else {
        // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
        return ((coord + 1) * size - 1) / 2;
      }
        */
}

/**
  | grid_sampler_unnormalize_set_grad works the
  | same as grid_sampler_unnormalize except that it
  | also returns the `d output / d input` via
  | pointer argument `grad_in`.
  |
  | This is useful in the backward pass of
  | grid_sampler.
  |
  */
#[inline] pub fn grid_sampler_unnormalize_set_grad<Scalar>(
        coord:         Scalar,
        size:          i64,
        align_corners: bool,
        grad_in:       *mut Scalar) -> Scalar {

    todo!();
        /*
            if (align_corners) {
        // unnormalize coord from [-1, 1] to [0, size - 1]
        *grad_in = static_cast<Scalar>(size - 1) / 2;
        return ((coord + 1) / 2) * (size - 1);
      } else {
        // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
        *grad_in = static_cast<Scalar>(size) / 2;
        return ((coord + 1) * size - 1) / 2;
      }
        */
}

/**
  | Clips coordinates to between 0 and clip_limit
  | - 1
  |
  */
#[inline] pub fn clip_coordinates<Scalar>(
        in_:        Scalar,
        clip_limit: i64) -> Scalar {

    todo!();
        /*
            return min(static_cast<Scalar>(clip_limit - 1), max(in, static_cast<Scalar>(0)));
        */
}

/**
  | clip_coordinates_set_grad works similarly to
  | clip_coordinates except that it also returns
  | the `d output / d input` via pointer argument
  | `grad_in`.
  |
  | This is useful in the backward pass of
  | grid_sampler.
  */
#[inline] pub fn clip_coordinates_set_grad<Scalar>(
        in_:        Scalar,
        clip_limit: i64,
        grad_in:    *mut Scalar) -> Scalar {

    todo!();
        /*
            // Note that it is important for the gradient calculation that borders
      // are considered out of bounds.
      if (in <= static_cast<Scalar>(0)) {
        *grad_in = static_cast<Scalar>(0);
        return static_cast<Scalar>(0);
      } else {
        Scalar max = static_cast<Scalar>(clip_limit - 1);
        if (in >= max) {
          *grad_in = static_cast<Scalar>(0);
          return max;
        } else {
          *grad_in = static_cast<Scalar>(1);
          return in;
        }
      }
        */
}

/**
  | Reflects coordinates until they fall between
  | low and high (inclusive).
  |
  | The bounds are passed as twice their value so
  | that half-integer values can be represented as
  | ints.
  |
  */
#[inline] pub fn reflect_coordinates<Scalar>(
        in_:        Scalar,
        twice_low:  i64,
        twice_high: i64) -> Scalar {

    todo!();
        /*
            if (twice_low == twice_high) {
        return static_cast<Scalar>(0);
      }
      Scalar min = static_cast<Scalar>(twice_low) / 2;
      Scalar span = static_cast<Scalar>(twice_high - twice_low) / 2;
      in = fabs(in - min);
      // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
      Scalar extra = fmod(in, span);
      int flips = static_cast<int>(floor(in / span));
      if (flips % 2 == 0) {
        return extra + min;
      } else {
        return span - extra + min;
      }
        */
}

/**
  | reflect_coordinates_set_grad works similarly to
  | reflect_coordinates except that it also returns
  | the `d output / d input` via pointer argument
  | `grad_in`.
  |
  | This is useful in the backward pass of
  | grid_sampler.
  |
  */
#[inline] pub fn reflect_coordinates_set_grad<Scalar>(
        in_:        Scalar,
        twice_low:  i64,
        twice_high: i64,
        grad_in:    *mut Scalar) -> Scalar {

    todo!();
        /*
            if (twice_low == twice_high) {
        *grad_in = static_cast<Scalar>(0);
        return static_cast<Scalar>(0);
      }
      int grad_in_mult_;
      Scalar min = static_cast<Scalar>(twice_low) / 2;
      Scalar span = static_cast<Scalar>(twice_high - twice_low) / 2;
      in = in - min;
      if (in < static_cast<Scalar>(0)) {
        grad_in_mult_ = -1;
        in = -in;
      } else {
        grad_in_mult_ = 1;
      }
      // `fmod` returns same sign as `in`, which is positive after the `if` above.
      Scalar extra = fmod(in, span);
      int flips = static_cast<int>(floor(in / span));
      if (flips % 2 == 0) {
        *grad_in = static_cast<Scalar>(grad_in_mult_);
        return extra + min;
      } else {
        *grad_in = static_cast<Scalar>(-grad_in_mult_);
        return span - extra + min;
      }
        */
}

/**
  | Mapping the out-of-boundary points back into
  | boundary
  |
  | This would only affect padding_mode=border or
  | reflection
  |
  */
#[inline] pub fn compute_coordinates<Scalar>(
        coord:         Scalar,
        size:          i64,
        padding_mode:  GridSamplerPadding,
        align_corners: bool) -> Scalar {

    todo!();
        /*
            if (padding_mode == GridSamplerPadding::Border) {
        // clip coordinates to image borders
        coord = clip_coordinates(coord, size);
      } else if (padding_mode == GridSamplerPadding::Reflection) {
        // reflect coordinates by image borders
        if (align_corners) {
          coord = reflect_coordinates(coord, 0, 2*(size - 1));
        } else {
          coord = reflect_coordinates(coord, -1, 2*size - 1);
        }
        // clip coordinates to image borders
        coord = clip_coordinates(coord, size);
      }
      return coord;
        */
}

/**
  | Computes the pixel source index value
  | for a grid coordinate
  |
  */
#[inline] pub fn grid_sampler_compute_source_index<Scalar>(
        coord:         Scalar,
        size:          i64,
        padding_mode:  GridSamplerPadding,
        align_corners: bool) -> Scalar {

    todo!();
        /*
            coord = grid_sampler_unnormalize(coord, size, align_corners);
      coord = compute_coordinates(coord, size, padding_mode, align_corners);
      return coord;
        */
}

/**
  | grid_sampler_compute_source_index_set_grad
  | works similarly to
  | grid_sampler_compute_source_index except that
  | it also returns the `d output / d input` via
  | pointer argument `grad_in`.
  |
  | This is useful in the backward pass of
  | grid_sampler.
  |
  */
#[inline] pub fn grid_sampler_compute_source_index_set_grad<Scalar>(
        coord:         Scalar,
        size:          i64,
        padding_mode:  GridSamplerPadding,
        align_corners: bool,
        grad_in:       *mut Scalar) -> Scalar {

    todo!();
        /*
            Scalar grad_clip, grad_refl;
      coord = grid_sampler_unnormalize_set_grad(coord, size, align_corners, grad_in);
      if (padding_mode == GridSamplerPadding::Border) {
        // clip coordinates to image borders
        coord = clip_coordinates_set_grad(coord, size, &grad_clip);
        *grad_in = (*grad_in) * grad_clip;
      } else if (padding_mode == GridSamplerPadding::Reflection) {
        // reflect coordinates by image borders
        if (align_corners) {
          coord = reflect_coordinates_set_grad(coord, 0, 2*(size - 1), &grad_refl);
        } else {
          coord = reflect_coordinates_set_grad(coord, -1, 2*size - 1, &grad_refl);
        }
        // clip coordinates to image borders
        coord = clip_coordinates_set_grad(coord, size, &grad_clip);
        *grad_in = (*grad_in) * grad_refl * grad_clip;
      }
      return coord;
        */
}

#[inline] pub fn within_bounds_2d(
        h: i64,
        w: i64,
        H: i64,
        W: i64) -> bool {
    
    todo!();
        /*
            return h >= 0 && h < H && w >= 0 && w < W;
        */
}


#[inline] pub fn within_bounds_3d(
        d: i64,
        h: i64,
        w: i64,
        D: i64,
        H: i64,
        W: i64) -> bool {
    
    todo!();
        /*
            return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
        */
}

#[inline] pub fn get_value_bounded<Scalar>(
        data:          *mut Scalar,
        x:             Scalar,
        y:             Scalar,
        W:             i64,
        H:             i64,
        sw:            i64,
        sh:            i64,
        padding_mode:  GridSamplerPadding,
        align_corners: bool) -> Scalar {

    todo!();
        /*
            x = compute_coordinates(x, W, padding_mode, align_corners);
      y = compute_coordinates(y, H, padding_mode, align_corners);

      i64 ix = static_cast<i64>(x);
      i64 iy = static_cast<i64>(y);

      if (within_bounds_2d(iy, ix, H, W)) {
        return data[iy * sH + ix * sW];
      }
      return static_cast<Scalar>(0);
        */
}

#[inline] pub fn safe_add_2d<Scalar>(
        data:  *mut Scalar,
        h:     i64,
        w:     i64,
        sh:    i64,
        sw:    i64,
        H:     i64,
        W:     i64,
        delta: Scalar)  {

    todo!();
        /*
            if (within_bounds_2d(h, w, H, W)) {
        data[h * sH + w * sW] += delta;
      }
        */
}

#[inline] pub fn safe_add_3d<Scalar>(
        data:  *mut Scalar,
        d:     i64,
        h:     i64,
        w:     i64,
        sd:    i64,
        sh:    i64,
        sw:    i64,
        D:     i64,
        H:     i64,
        W:     i64,
        delta: Scalar)  {

    todo!();
        /*
            if (within_bounds_3d(d, h, w, D, H, W)) {
        data[d * sD + h * sH + w * sW] += delta;
      }
        */
}

#[inline] pub fn add_value_bounded<Scalar>(
        data:          *mut Scalar,
        x:             Scalar,
        y:             Scalar,
        W:             i64,
        H:             i64,
        sw:            i64,
        sh:            i64,
        delta:         Scalar,
        padding_mode:  GridSamplerPadding,
        align_corners: bool)  {

    todo!();
        /*
            x = compute_coordinates(x, W, padding_mode, align_corners);
      y = compute_coordinates(y, H, padding_mode, align_corners);

      i64 ix = static_cast<i64>(x);
      i64 iy = static_cast<i64>(y);

      safe_add_2d(data, iy, ix, sH, sW, H, W, delta);
        */
}

/**
  | Calculate the differential of the cubic
  | convolution, i.e. `d coeff / d x`
  |
  */
#[inline] pub fn get_cubic_coefficients_grad<Scalar>(
        coeffs: [Scalar; 4],
        t:      Scalar)  {

    todo!();
        /*
            // Must be the same as forward calculation in
      // aten/src/ATen/native/UpSample.h:get_cubic_upsample_coefficients
      Scalar A = -0.75;

      Scalar x;
      x = -1 - t; // 1 < x = |-1 - tx| < 2
      coeffs[0] = (-3 * A * x - 10 * A ) * x - 8 * A;
      x = -t;     // x = |0 - tx| <= 1
      coeffs[1] = (-3 * (A + 2) * x - 2 * (A + 3)) * x;
      x = 1 - t;  // x = |1 - tx| <= 1
      coeffs[2] = (3 * (A + 2) * x - 2 * (A + 3)) * x;
      x = 2 - t;  // 1 < x = |2 - tx| < 2
      coeffs[3] = (3 * A * x - 10 * A) * x + 8 * A;
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/GridSampler.cpp]

pub fn grid_sampler_3d_cpu_impl<Scalar>(
        input:              &Tensor,
        grid:               &Tensor,
        interpolation_mode: GridSamplerInterpolation,
        padding_mode:       GridSamplerPadding,
        align_corners:      bool) -> Tensor {

    todo!();
        /*
            i64 N = input.size(0);
        i64 C = input.size(1);
        i64 inp_D = input.size(2);
        i64 inp_H = input.size(3);
        i64 inp_W = input.size(4);
        i64 out_D = grid.size(1);
        i64 out_H = grid.size(2);
        i64 out_W = grid.size(3);
        auto output = empty({N, C, out_D, out_H, out_W}, input.options());
        i64 inp_sN = input.stride(0);
        i64 inp_sC = input.stride(1);
        i64 inp_sD = input.stride(2);
        i64 inp_sH = input.stride(3);
        i64 inp_sW = input.stride(4);
        i64 grid_sN = grid.stride(0);
        i64 grid_sD = grid.stride(1);
        i64 grid_sH = grid.stride(2);
        i64 grid_sW = grid.stride(3);
        i64 grid_sCoor = grid.stride(4);
        i64 out_sN = output.stride(0);
        i64 out_sC = output.stride(1);
        i64 out_sD = output.stride(2);
        i64 out_sH = output.stride(3);
        i64 out_sW = output.stride(4);
        Scalar *inp_ptr = input.data_ptr<Scalar>();
        Scalar *out_ptr = output.data_ptr<Scalar>();
        Scalar *grid_ptr = grid.data_ptr<Scalar>();
        // loop over each output pixel
        parallel_for(0, N, 0, [&](i64 start, i64 end) {
          for (i64 n = start; n < end; ++n) {
            Scalar *grid_ptr_N = grid_ptr + n * grid_sN;
            Scalar *inp_ptr_N = inp_ptr + n * inp_sN;
            for (i64 d = 0; d < out_D; ++d) {
              for (i64 h = 0; h < out_H; ++h) {
                for (i64 w = 0; w < out_W; ++w) {
                  // get the corresponding input x, y, z co-ordinates from grid
                  Scalar *grid_ptr_NDHW = grid_ptr_N + d * grid_sD + h * grid_sH + w * grid_sW;
                  Scalar ix = *grid_ptr_NDHW;
                  Scalar iy = grid_ptr_NDHW[grid_sCoor];
                  Scalar iz = grid_ptr_NDHW[2 * grid_sCoor];

                  ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
                  iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);
                  iz = grid_sampler_compute_source_index(iz, inp_D, padding_mode, align_corners);

                  if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                    // get corner pixel values from (x, y, z)
                    // for 4d, we used north-east-south-west
                    // for 5d, we add top-bottom
                    i64 ix_tnw = static_cast<i64>(floor(ix));
                    i64 iy_tnw = static_cast<i64>(floor(iy));
                    i64 iz_tnw = static_cast<i64>(floor(iz));

                    i64 ix_tne = ix_tnw + 1;
                    i64 iy_tne = iy_tnw;
                    i64 iz_tne = iz_tnw;

                    i64 ix_tsw = ix_tnw;
                    i64 iy_tsw = iy_tnw + 1;
                    i64 iz_tsw = iz_tnw;

                    i64 ix_tse = ix_tnw + 1;
                    i64 iy_tse = iy_tnw + 1;
                    i64 iz_tse = iz_tnw;

                    i64 ix_bnw = ix_tnw;
                    i64 iy_bnw = iy_tnw;
                    i64 iz_bnw = iz_tnw + 1;

                    i64 ix_bne = ix_tnw + 1;
                    i64 iy_bne = iy_tnw;
                    i64 iz_bne = iz_tnw + 1;

                    i64 ix_bsw = ix_tnw;
                    i64 iy_bsw = iy_tnw + 1;
                    i64 iz_bsw = iz_tnw + 1;

                    i64 ix_bse = ix_tnw + 1;
                    i64 iy_bse = iy_tnw + 1;
                    i64 iz_bse = iz_tnw + 1;

                    // get surfaces to each neighbor:
                    Scalar tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
                    Scalar tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
                    Scalar tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
                    Scalar tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
                    Scalar bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
                    Scalar bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
                    Scalar bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
                    Scalar bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

                    // calculate bilinear weighted pixel value and set output pixel
                    Scalar *out_ptr_NCDHW = out_ptr + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
                    Scalar *inp_ptr_NC = inp_ptr_N;
                    for (i64 c = 0; c < C; ++c, out_ptr_NCDHW += out_sC, inp_ptr_NC += inp_sC) {
                      //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
                      // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
                      // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
                      // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
                      *out_ptr_NCDHW = static_cast<Scalar>(0);
                      if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
                        *out_ptr_NCDHW += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
                      }
                      if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
                        *out_ptr_NCDHW += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
                      }
                      if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
                        *out_ptr_NCDHW += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
                      }
                      if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
                        *out_ptr_NCDHW += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
                      }
                      if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
                        *out_ptr_NCDHW += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
                      }
                      if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
                        *out_ptr_NCDHW += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
                      }
                      if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
                        *out_ptr_NCDHW += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
                      }
                      if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
                        *out_ptr_NCDHW += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;
                      }
                    }
                  } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
                    i64 ix_nearest = static_cast<i64>(round(ix));
                    i64 iy_nearest = static_cast<i64>(round(iy));
                    i64 iz_nearest = static_cast<i64>(round(iz));

                    // assign nearest neighor pixel value to output pixel
                    Scalar *out_ptr_NCDHW = out_ptr + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
                    Scalar *inp_ptr_NC = inp_ptr_N;
                    for (i64 c = 0; c < C; ++c, out_ptr_NCDHW += out_sC, inp_ptr_NC += inp_sC) {
                      if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, inp_D, inp_H, inp_W)) {
                        *out_ptr_NCDHW = inp_ptr_NC[iz_nearest * inp_sD + iy_nearest * inp_sH + ix_nearest * inp_sW];
                      } else {
                        *out_ptr_NCDHW = static_cast<Scalar>(0);
                      }
                    }
                  }
                }
              }
            }
          }
        });
        return output;
        */
}

pub fn grid_sampler_3d_backward_cpu_impl<Scalar>(
        grad_output:        &Tensor,
        input:              &Tensor,
        grid:               &Tensor,
        interpolation_mode: GridSamplerInterpolation,
        padding_mode:       GridSamplerPadding,
        align_corners:      bool) -> (Tensor,Tensor) {

    todo!();
        /*
            auto grad_input = zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        auto grad_grid = empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        // If interpolation mode is Nearest, then grad_grid is not filled in the
        // loop below.
        if (interpolation_mode == GridSamplerInterpolation::Nearest) {
          grad_grid.zero_();
        }
        i64 N = input.size(0);
        i64 C = input.size(1);
        i64 inp_D = input.size(2);
        i64 inp_H = input.size(3);
        i64 inp_W = input.size(4);
        i64 out_D = grid.size(1);
        i64 out_H = grid.size(2);
        i64 out_W = grid.size(3);
        i64 inp_sN = input.stride(0);
        i64 inp_sC = input.stride(1);
        i64 inp_sD = input.stride(2);
        i64 inp_sH = input.stride(3);
        i64 inp_sW = input.stride(4);
        i64 grid_sN = grid.stride(0);
        i64 grid_sD = grid.stride(1);
        i64 grid_sH = grid.stride(2);
        i64 grid_sW = grid.stride(3);
        i64 grid_sCoor = grid.stride(4);
        i64 gOut_sN = grad_output.stride(0);
        i64 gOut_sC = grad_output.stride(1);
        i64 gOut_sD = grad_output.stride(2);
        i64 gOut_sH = grad_output.stride(3);
        i64 gOut_sW = grad_output.stride(4);
        i64 gInp_sN = grad_input.stride(0);
        i64 gInp_sC = grad_input.stride(1);
        i64 gInp_sD = grad_input.stride(2);
        i64 gInp_sH = grad_input.stride(3);
        i64 gInp_sW = grad_input.stride(4);
        i64 gGrid_sN = grad_grid.stride(0);
        i64 gGrid_sW = grad_grid.stride(3);
        Scalar *inp_ptr = input.data_ptr<Scalar>();
        Scalar *grid_ptr = grid.data_ptr<Scalar>();
        Scalar *gOut_ptr = grad_output.data_ptr<Scalar>();
        Scalar *gInp_ptr = grad_input.data_ptr<Scalar>();
        Scalar *gGrid_ptr = grad_grid.data_ptr<Scalar>();
        // loop over each output pixel
        parallel_for(0, N, 0, [&](i64 start, i64 end) {
          for (i64 n = start; n < end; ++n) {
            Scalar *grid_ptr_N = grid_ptr + n * grid_sN;
            Scalar *inp_ptr_N = inp_ptr + n * inp_sN;
            Scalar *gGrid_ptr_NDHW = gGrid_ptr + n * gGrid_sN;
            for (i64 d = 0; d < out_D; ++d) {
              for (i64 h = 0; h < out_H; ++h) {
                for (i64 w = 0; w < out_W; ++w, gGrid_ptr_NDHW += gGrid_sW /* grad_grid is contiguous */ ) {
                  // get the corresponding input x, y, z co-ordinates from grid
                  Scalar *grid_ptr_NDHW = grid_ptr_N + d * grid_sD + h * grid_sH + w * grid_sW;
                  Scalar ix = *grid_ptr_NDHW;
                  Scalar iy = grid_ptr_NDHW[grid_sCoor];
                  Scalar iz = grid_ptr_NDHW[2 * grid_sCoor];

                  // multipliers for gradients on ix, iy, and iz
                  Scalar gix_mult, giy_mult, giz_mult;
                  ix = grid_sampler_compute_source_index_set_grad(ix, inp_W, padding_mode, align_corners, &gix_mult);
                  iy = grid_sampler_compute_source_index_set_grad(iy, inp_H, padding_mode, align_corners, &giy_mult);
                  iz = grid_sampler_compute_source_index_set_grad(iz, inp_D, padding_mode, align_corners, &giz_mult);

                  if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                    // get corner pixel values from (x, y, z)
                    // for 4d, we used north-east-south-west
                    // for 5d, we add top-bottom
                    i64 ix_tnw = static_cast<i64>(floor(ix));
                    i64 iy_tnw = static_cast<i64>(floor(iy));
                    i64 iz_tnw = static_cast<i64>(floor(iz));

                    i64 ix_tne = ix_tnw + 1;
                    i64 iy_tne = iy_tnw;
                    i64 iz_tne = iz_tnw;

                    i64 ix_tsw = ix_tnw;
                    i64 iy_tsw = iy_tnw + 1;
                    i64 iz_tsw = iz_tnw;

                    i64 ix_tse = ix_tnw + 1;
                    i64 iy_tse = iy_tnw + 1;
                    i64 iz_tse = iz_tnw;

                    i64 ix_bnw = ix_tnw;
                    i64 iy_bnw = iy_tnw;
                    i64 iz_bnw = iz_tnw + 1;

                    i64 ix_bne = ix_tnw + 1;
                    i64 iy_bne = iy_tnw;
                    i64 iz_bne = iz_tnw + 1;

                    i64 ix_bsw = ix_tnw;
                    i64 iy_bsw = iy_tnw + 1;
                    i64 iz_bsw = iz_tnw + 1;

                    i64 ix_bse = ix_tnw + 1;
                    i64 iy_bse = iy_tnw + 1;
                    i64 iz_bse = iz_tnw + 1;

                    // get surfaces to each neighbor:
                    Scalar tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
                    Scalar tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
                    Scalar tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
                    Scalar tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
                    Scalar bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
                    Scalar bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
                    Scalar bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
                    Scalar bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

                    Scalar gix = static_cast<Scalar>(0), giy = static_cast<Scalar>(0), giz = static_cast<Scalar>(0);
                    Scalar *gOut_ptr_NCDHW = gOut_ptr + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
                    Scalar *gInp_ptr_NC = gInp_ptr + n * gInp_sN;
                    Scalar *inp_ptr_NC = inp_ptr_N;
                    // calculate bilinear weighted pixel value and set output pixel
                    for (i64 c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC += inp_sC) {
                      Scalar gOut = *gOut_ptr_NCDHW;

                      // calculate and set grad_input
                      safe_add_3d(gInp_ptr_NC, iz_tnw, iy_tnw, ix_tnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tnw * gOut);
                      safe_add_3d(gInp_ptr_NC, iz_tne, iy_tne, ix_tne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tne * gOut);
                      safe_add_3d(gInp_ptr_NC, iz_tsw, iy_tsw, ix_tsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tsw * gOut);
                      safe_add_3d(gInp_ptr_NC, iz_tse, iy_tse, ix_tse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tse * gOut);
                      safe_add_3d(gInp_ptr_NC, iz_bnw, iy_bnw, ix_bnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bnw * gOut);
                      safe_add_3d(gInp_ptr_NC, iz_bne, iy_bne, ix_bne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bne * gOut);
                      safe_add_3d(gInp_ptr_NC, iz_bsw, iy_bsw, ix_bsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bsw * gOut);
                      safe_add_3d(gInp_ptr_NC, iz_bse, iy_bse, ix_bse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bse * gOut);

                      // calculate grad_grid
                      if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
                        Scalar tnw_val = inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW];
                        gix -= tnw_val * (iy_bse - iy)    * (iz_bse - iz)    * gOut;
                        giy -= tnw_val * (ix_bse - ix)    * (iz_bse - iz)    * gOut;
                        giz -= tnw_val * (ix_bse - ix)    * (iy_bse - iy)    * gOut;
                      }
                      if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
                        Scalar tne_val = inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW];
                        gix += tne_val * (iy_bsw - iy)    * (iz_bsw - iz)    * gOut;
                        giy -= tne_val * (ix    - ix_bsw) * (iz_bsw - iz)    * gOut;
                        giz -= tne_val * (ix    - ix_bsw) * (iy_bsw - iy)    * gOut;
                      }
                      if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
                        Scalar tsw_val = inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW];
                        gix -= tsw_val * (iy - iy_bne)    * (iz_bne - iz)    * gOut;
                        giy += tsw_val * (ix_bne - ix)    * (iz_bne - iz)    * gOut;
                        giz -= tsw_val * (ix_bne - ix)    * (iy    - iy_bne) * gOut;
                      }
                      if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
                        Scalar tse_val = inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW];
                        gix += tse_val * (iy - iy_bnw)    * (iz_bnw - iz)    * gOut;
                        giy += tse_val * (ix    - ix_bnw) * (iz_bnw - iz)    * gOut;
                        giz -= tse_val * (ix    - ix_bnw) * (iy    - iy_bnw) * gOut;
                      }
                      if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
                        Scalar bnw_val = inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW];
                        gix -= bnw_val * (iy_tse - iy)    * (iz - iz_tse)    * gOut;
                        giy -= bnw_val * (ix_tse - ix)    * (iz - iz_tse)    * gOut;
                        giz += bnw_val * (ix_tse - ix)    * (iy_tse - iy)    * gOut;
                      }
                      if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
                        Scalar bne_val = inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW];
                        gix += bne_val * (iy_tsw - iy)    * (iz - iz_tsw)    * gOut;
                        giy -= bne_val * (ix    - ix_tsw) * (iz - iz_tsw)    * gOut;
                        giz += bne_val * (ix    - ix_tsw) * (iy_tsw - iy)    * gOut;
                      }
                      if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
                        Scalar bsw_val = inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW];
                        gix -= bsw_val * (iy - iy_tne)    * (iz - iz_tne)    * gOut;
                        giy += bsw_val * (ix_tne - ix)    * (iz - iz_tne)    * gOut;
                        giz += bsw_val * (ix_tne - ix)    * (iy    - iy_tne) * gOut;
                      }
                      if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
                        Scalar bse_val = inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW];
                        gix += bse_val * (iy - iy_tnw)    * (iz - iz_tnw)    * gOut;
                        giy += bse_val * (ix    - ix_tnw) * (iz - iz_tnw)    * gOut;
                        giz += bse_val * (ix    - ix_tnw) * (iy    - iy_tnw) * gOut;
                      }
                    }

                    // assuming grad_grid is contiguous
                    gGrid_ptr_NDHW[0] = gix_mult * gix;
                    gGrid_ptr_NDHW[1] = giy_mult * giy;
                    gGrid_ptr_NDHW[2] = giz_mult * giz;
                  } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
                    i64 ix_nearest = static_cast<i64>(round(ix));
                    i64 iy_nearest = static_cast<i64>(round(iy));
                    i64 iz_nearest = static_cast<i64>(round(iz));

                    // assign nearest neighor pixel value to output pixel
                    Scalar *gOut_ptr_NCDHW = gOut_ptr + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
                    Scalar *gInp_ptr_NC = gInp_ptr + n * gInp_sN;
                    for (i64 c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC) {
                      // calculate and set grad_input
                      safe_add_3d(gInp_ptr_NC, iz_nearest, iy_nearest, ix_nearest,
                                  gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, *gOut_ptr_NCDHW);
                    }
                  }
                }
              }
            }
          }
        });
        return make_tuple(grad_input, grad_grid);
        */
}

pub fn grid_sampler_2d_cpu_fallback(
        input:              &Tensor,
        grid:               &Tensor,
        interpolation_mode: i64,
        padding_mode:       i64,
        align_corners:      bool) -> Tensor {
    
    todo!();
        /*
            auto interpolation_mode = static_cast<GridSamplerInterpolation>(interpolation_mode_);
      auto padding_mode = static_cast<GridSamplerPadding>(padding_mode_);
      using Scalar = float;

      i64 N = input.size(0);
      i64 C = input.size(1);
      i64 inp_H = input.size(2);
      i64 inp_W = input.size(3);
      i64 out_H = grid.size(1);
      i64 out_W = grid.size(2);
      auto output = empty({N, C, out_H, out_W}, input.options());
      i64 inp_sN = input.stride(0);
      i64 inp_sC = input.stride(1);
      i64 inp_sH = input.stride(2);
      i64 inp_sW = input.stride(3);
      i64 grid_sN = grid.stride(0);
      i64 grid_sH = grid.stride(1);
      i64 grid_sW = grid.stride(2);
      i64 grid_sCoor = grid.stride(3);
      i64 out_sN = output.stride(0);
      i64 out_sC = output.stride(1);
      i64 out_sH = output.stride(2);
      i64 out_sW = output.stride(3);
      Scalar *inp_ptr = input.data_ptr<Scalar>();
      Scalar *out_ptr = output.data_ptr<Scalar>();
      Scalar *grid_ptr = grid.data_ptr<Scalar>();
      // loop over each output pixel
      parallel_for(0, N, 0, [&](i64 start, i64 end) {
        for (i64 n = start; n < end; ++n) {
          Scalar *grid_ptr_N = grid_ptr + n * grid_sN;
          Scalar *inp_ptr_N = inp_ptr + n * inp_sN;
          for (i64 h = 0; h < out_H; ++h) {
            for (i64 w = 0; w < out_W; ++w) {
              // get the corresponding input x, y, z co-ordinates from grid
              Scalar *grid_ptr_NHW = grid_ptr_N + h * grid_sH + w * grid_sW;
              Scalar x = *grid_ptr_NHW;
              Scalar y = grid_ptr_NHW[grid_sCoor];

              Scalar ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
              Scalar iy = grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);

              if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                // get corner pixel values from (x, y)
                // for 4d, we use north-east-south-west
                i64 ix_nw = static_cast<i64>(floor(ix));
                i64 iy_nw = static_cast<i64>(floor(iy));

                i64 ix_ne = ix_nw + 1;
                i64 iy_ne = iy_nw;

                i64 ix_sw = ix_nw;
                i64 iy_sw = iy_nw + 1;

                i64 ix_se = ix_nw + 1;
                i64 iy_se = iy_nw + 1;

                // get surfaces to each neighbor:
                Scalar nw = (ix_se - ix)    * (iy_se - iy);
                Scalar ne = (ix    - ix_sw) * (iy_sw - iy);
                Scalar sw = (ix_ne - ix)    * (iy    - iy_ne);
                Scalar se = (ix    - ix_nw) * (iy    - iy_nw);

                // calculate bilinear weighted pixel value and set output pixel
                Scalar *inp_ptr_NC = inp_ptr_N;
                Scalar *out_ptr_NCHW = out_ptr + n * out_sN + h * out_sH + w * out_sW;
                for (i64 c = 0; c < C; ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
                  auto res = static_cast<Scalar>(0);
                  if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                    res += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
                  }
                  if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                    res += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
                  }
                  if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                    res += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
                  }
                  if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                    res += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
                  }
                  *out_ptr_NCHW = res;
                }
              } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
                i64 ix_nearest = static_cast<i64>(nearbyint(ix));
                i64 iy_nearest = static_cast<i64>(nearbyint(iy));

                // assign nearest neighor pixel value to output pixel
                Scalar *out_ptr_NCHW = out_ptr + n * out_sN + h * out_sH + w * out_sW;
                Scalar *inp_ptr_NC = inp_ptr_N;
                for (i64 c = 0; c < C; ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
                  if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
                    *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
                  } else {
                    *out_ptr_NCHW = static_cast<Scalar>(0);
                  }
                }
              } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
                // grid_sampler_compute_source_index will "clip the value" of idx depends on the padding,
                // which would cause calculation to be wrong,
                // for example x = -0.1 -> ix = 0 for zero padding, but in bicubic ix = floor(x) = -1
                // There would be more problem in reflection padding, since the -1 and +1 direction is not fixed in boundary condition
                ix = grid_sampler_unnormalize(x, inp_W, align_corners);
                iy = grid_sampler_unnormalize(y, inp_H, align_corners);

                Scalar ix_nw = floor(ix);
                Scalar iy_nw = floor(iy);

                const Scalar tx = ix - ix_nw;
                const Scalar ty = iy - iy_nw;

                Scalar *inp_ptr_NC = inp_ptr_N;
                Scalar *out_ptr_NCHW = out_ptr + n * out_sN + h * out_sH + w * out_sW;
                for (i64 c = 0; c < C; ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
                  Scalar coefficients[4];

                  // Interpolate 4 values in the x directon
                  for (i64 i = 0; i < 4; ++i) {
                    coefficients[i] = cubic_interp1d<Scalar>(
                      get_value_bounded<Scalar>(inp_ptr_NC, ix_nw - 1, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
                      get_value_bounded<Scalar>(inp_ptr_NC, ix_nw + 0, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
                      get_value_bounded<Scalar>(inp_ptr_NC, ix_nw + 1, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
                      get_value_bounded<Scalar>(inp_ptr_NC, ix_nw + 2, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
                      tx);
                  }

                  // Interpolate in the y direction
                  *out_ptr_NCHW = cubic_interp1d<Scalar>(
                    coefficients[0],
                    coefficients[1],
                    coefficients[2],
                    coefficients[3],
                    ty);
                }
              }
            }
          }
        }
      });
      return output;
        */
}

pub fn grid_sampler_2d_cpu_fallback_backward(
        grad_output:        &Tensor,
        input:              &Tensor,
        grid:               &Tensor,
        interpolation_mode: i64,
        padding_mode:       i64,
        align_corners:      bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            const auto interpolation_mode = static_cast<GridSamplerInterpolation>(interpolation_mode_);
      const auto padding_mode = static_cast<GridSamplerPadding>(padding_mode_);
      using Scalar = float;

      auto grad_input = zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto grad_grid = empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      // If interpolation mode is Nearest, then grad_grid is not filled in the
      // loop below.
      if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        grad_grid.zero_();
      }
      i64 N = input.size(0);
      i64 C = input.size(1);
      i64 inp_H = input.size(2);
      i64 inp_W = input.size(3);
      i64 out_H = grid.size(1);
      i64 out_W = grid.size(2);
      i64 inp_sN = input.stride(0);
      i64 inp_sC = input.stride(1);
      i64 inp_sH = input.stride(2);
      i64 inp_sW = input.stride(3);
      i64 grid_sN = grid.stride(0);
      i64 grid_sH = grid.stride(1);
      i64 grid_sW = grid.stride(2);
      i64 grid_sCoor = grid.stride(3);
      i64 gOut_sN = grad_output.stride(0);
      i64 gOut_sC = grad_output.stride(1);
      i64 gOut_sH = grad_output.stride(2);
      i64 gOut_sW = grad_output.stride(3);
      i64 gInp_sN = grad_input.stride(0);
      i64 gInp_sC = grad_input.stride(1);
      i64 gInp_sH = grad_input.stride(2);
      i64 gInp_sW = grad_input.stride(3);
      i64 gGrid_sN = grad_grid.stride(0);
      i64 gGrid_sW = grad_grid.stride(2);
      Scalar *inp_ptr = input.data_ptr<Scalar>();
      Scalar *grid_ptr = grid.data_ptr<Scalar>();
      Scalar *gOut_ptr = grad_output.data_ptr<Scalar>();
      Scalar *gInp_ptr = grad_input.data_ptr<Scalar>();
      Scalar *gGrid_ptr = grad_grid.data_ptr<Scalar>();
      // loop over each output pixel
      parallel_for(0, N, 0, [&](i64 start, i64 end) {
        for (i64 n = start; n < end; ++n) {
          Scalar *grid_ptr_N = grid_ptr + n * grid_sN;
          Scalar *inp_ptr_N = inp_ptr + n * inp_sN;
          Scalar *gGrid_ptr_NHW = gGrid_ptr + n * gGrid_sN;
          for (i64 h = 0; h < out_H; ++h) {
            for (i64 w = 0; w < out_W; ++w, gGrid_ptr_NHW += gGrid_sW /* grad_grid is contiguous */ ) {
              // get the corresponding input x, y co-ordinates from grid
              Scalar *grid_ptr_NHW = grid_ptr_N + h * grid_sH + w * grid_sW;
              Scalar x = *grid_ptr_NHW;
              Scalar y = grid_ptr_NHW[grid_sCoor];

              // multipliers for gradients on ix, iy
              Scalar gix_mult, giy_mult;
              Scalar ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &gix_mult);
              Scalar iy = grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &giy_mult);

              if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                // get corner pixel values from (x, y)
                // for 4d, we use north-east-south-west
                i64 ix_nw = static_cast<i64>(floor(ix));
                i64 iy_nw = static_cast<i64>(floor(iy));

                i64 ix_ne = ix_nw + 1;
                i64 iy_ne = iy_nw;

                i64 ix_sw = ix_nw;
                i64 iy_sw = iy_nw + 1;

                i64 ix_se = ix_nw + 1;
                i64 iy_se = iy_nw + 1;

                // get surfaces to each neighbor:
                Scalar nw = (ix_se - ix)    * (iy_se - iy);
                Scalar ne = (ix    - ix_sw) * (iy_sw - iy);
                Scalar sw = (ix_ne - ix)    * (iy    - iy_ne);
                Scalar se = (ix    - ix_nw) * (iy    - iy_nw);

                Scalar gix = static_cast<Scalar>(0), giy = static_cast<Scalar>(0);
                Scalar *gOut_ptr_NCHW = gOut_ptr + n * gOut_sN + h * gOut_sH + w * gOut_sW;
                Scalar *gInp_ptr_NC = gInp_ptr + n * gInp_sN;
                Scalar *inp_ptr_NC = inp_ptr_N;
                // calculate bilinear weighted pixel value and set output pixel
                for (i64 c = 0; c < C; ++c, gOut_ptr_NCHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC += inp_sC) {
                  Scalar gOut = *gOut_ptr_NCHW;

                  // calculate and set grad_input
                  safe_add_2d(gInp_ptr_NC, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut);
                  safe_add_2d(gInp_ptr_NC, iy_ne, ix_ne, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut);
                  safe_add_2d(gInp_ptr_NC, iy_sw, ix_sw, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut);
                  safe_add_2d(gInp_ptr_NC, iy_se, ix_se, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut);

                  // calculate grad_grid
                  if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                    Scalar nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
                    gix -= nw_val * (iy_se - iy) * gOut;
                    giy -= nw_val * (ix_se - ix) * gOut;
                  }
                  if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                    Scalar ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
                    gix += ne_val * (iy_sw - iy) * gOut;
                    giy -= ne_val * (ix - ix_sw) * gOut;
                  }
                  if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                    Scalar sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
                    gix -= sw_val * (iy - iy_ne) * gOut;
                    giy += sw_val * (ix_ne - ix) * gOut;
                  }
                  if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                    Scalar se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
                    gix += se_val * (iy - iy_nw) * gOut;
                    giy += se_val * (ix - ix_nw) * gOut;
                  }
                }

                // assuming grad_grid is contiguous
                gGrid_ptr_NHW[0] = gix_mult * gix;
                gGrid_ptr_NHW[1] = giy_mult * giy;
              } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
                i64 ix_nearest = static_cast<i64>(nearbyint(ix));
                i64 iy_nearest = static_cast<i64>(nearbyint(iy));

                // assign nearest neighor pixel value to output pixel
                Scalar *gOut_ptr_NCHW = gOut_ptr + n * gOut_sN + h * gOut_sH + w * gOut_sW;
                Scalar *gInp_ptr_NC = gInp_ptr + n * gInp_sN;
                for (i64 c = 0; c < C; ++c, gOut_ptr_NCHW += gOut_sC, gInp_ptr_NC += gInp_sC) {
                  // calculate and set grad_input
                  safe_add_2d(gInp_ptr_NC, iy_nearest, ix_nearest, gInp_sH, gInp_sW,
                              inp_H, inp_W, *gOut_ptr_NCHW);
                }
              } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {

                ix = grid_sampler_unnormalize_set_grad(x, inp_W, align_corners, &gix_mult);
                iy = grid_sampler_unnormalize_set_grad(y, inp_H, align_corners, &giy_mult);

                Scalar ix_nw = floor(ix);
                Scalar iy_nw = floor(iy);

                const Scalar tx = ix - ix_nw;
                const Scalar ty = iy - iy_nw;

                Scalar x_coeffs[4];
                Scalar y_coeffs[4];
                Scalar x_coeffs_grad[4];
                Scalar y_coeffs_grad[4];

                get_cubic_upsample_coefficients<Scalar>(x_coeffs, tx);
                get_cubic_upsample_coefficients<Scalar>(y_coeffs, ty);
                get_cubic_coefficients_grad<Scalar>(x_coeffs_grad, tx);
                get_cubic_coefficients_grad<Scalar>(y_coeffs_grad, ty);

                Scalar gix = static_cast<Scalar>(0);
                Scalar giy = static_cast<Scalar>(0);

                Scalar *gOut_ptr_NCHW = gOut_ptr + n * gOut_sN + h * gOut_sH + w * gOut_sW;
                Scalar *gInp_ptr_NC = gInp_ptr + n * gInp_sN;
                Scalar *inp_ptr_NC = inp_ptr_N;

                for (i64 c = 0; c < C; ++c, gOut_ptr_NCHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC+= inp_sC) {
                  Scalar gOut = *gOut_ptr_NCHW;

                  for (i64 i = 0; i < 4; ++i) {
                    for (i64 j = 0; j < 4; ++j) {

                      // set input gradient
                      add_value_bounded<Scalar>(gInp_ptr_NC, ix_nw - 1 + i, iy_nw - 1 + j,
                        inp_W, inp_H, gInp_sW, gInp_sH, gOut * x_coeffs[i] * y_coeffs[j], padding_mode, align_corners);

                      // set grid gradient
                      Scalar val = get_value_bounded<Scalar>(inp_ptr_NC, ix_nw - 1 + i, iy_nw - 1 + j,
                        inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners);

                      gix -= val * x_coeffs_grad[i] * y_coeffs[j] * gOut;
                      giy -= val * y_coeffs_grad[j] * x_coeffs[i] * gOut;
                    }
                  }
                }
                gGrid_ptr_NHW[0] = gix_mult * gix;
                gGrid_ptr_NHW[1] = giy_mult * giy;
              }
            }
          }
        }
      });
      return make_tuple(grad_input, grad_grid);
        */
}

/**
  | No shape checking needed here. See #
  | NOTE [ grid_sampler Native Functions
  | ].
  |
  */
pub fn grid_sampler_2d_cpu(
        input:              &Tensor,
        grid:               &Tensor,
        interpolation_mode: i64,
        padding_mode:       i64,
        align_corners:      bool) -> Tensor {
    
    todo!();
        /*
            // AVX gather instructions use signed 32-bit offsets to gather float values.
      // Check for possible overflow and fallback to scalar implementation
      if (input.scalar_type() != kDouble) {
        TORCH_CHECK(input.scalar_type() == kFloat,
                    "grid_sampler_2d_cpu not implemented for ", input.scalar_type());
        auto sizes = input.sizes();
        auto strides = input.strides();
        const auto grid_sW = grid.strides()[2];
        // NOTE: Gather offsets are only used for the input H, W dimensions
        //       or only for strided access to the grid tensor
        auto max_gather_offset = max(
          (sizes[2] - 1) * strides[2] + (sizes[3] - 1) * strides[3],
          grid_sW * (vec::Vectorized<float>::size() - 1));

        if (max_gather_offset > i32::max) {
          return native::_grid_sampler_2d_cpu_fallback(
            input, grid, interpolation_mode, padding_mode, align_corners);
        }
      }

      return grid_sampler_2d_cpu_kernel(
        kCPU, input, grid, interpolation_mode, padding_mode, align_corners);
        */
}

define_dispatch!{grid_sampler_2d_cpu_kernel}

/**
  | No shape checking needed here. See #
  | NOTE [ grid_sampler Native Functions
  | ].
  |
  */
pub fn grid_sampler_3d_cpu(
        input:              &Tensor,
        grid:               &Tensor,
        interpolation_mode: i64,
        padding_mode:       i64,
        align_corners:      bool) -> Tensor {
    
    todo!();
        /*
            return AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_sampler3d_cpu", [&] {
        return grid_sampler_3d_cpu_impl<Scalar>(
          input, grid, static_cast<GridSamplerInterpolation>(interpolation_mode),
          static_cast<GridSamplerPadding>(padding_mode), align_corners);
      });
        */
}

/**
  | No shape checking needed here. See #
  | NOTE [ grid_sampler Native Functions ].
  |
  */
pub fn grid_sampler_2d_backward_cpu(
        grad_output:        &Tensor,
        input:              &Tensor,
        grid:               &Tensor,
        interpolation_mode: i64,
        padding_mode:       i64,
        align_corners:      bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            // AVX gather instructions use signed 32-bit offsets to gather float values.
      // Check for possible overflow and fallback to scalar implementation
      if (input.scalar_type() != kDouble) {
        TORCH_CHECK(input.scalar_type() == kFloat,
                    "grid_sampler_2d_backward_cpu not implemented for ", input.scalar_type());
        auto isizes = input.sizes();
        auto istrides = input.strides();
        auto gsizes = grad_output.sizes();
        auto gstrides = grad_output.strides();
        const auto grid_sW = grid.strides()[2];
        // NOTE: Gather offsets are only used for the height and width dimensions
        auto max_gather_offset = max(
          max(
            (isizes[2] - 1) * istrides[2] + (isizes[3] - 1) * istrides[3],
            (gsizes[2] - 1) * gstrides[2] + (gsizes[3] - 1) * gstrides[3]),
          grid_sW * (vec::Vectorized<float>::size() - 1));

        if (max_gather_offset > i32::max) {
          return native::_grid_sampler_2d_cpu_fallback_backward(
            grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
        }
      }

      return grid_sampler_2d_backward_cpu_kernel(
        kCPU, grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
        */
}

define_dispatch!{grid_sampler_2d_backward_cpu_kernel}

/**
  | No shape checking needed here. See #
  | NOTE [ grid_sampler Native Functions ].
  |
  */
pub fn grid_sampler_3d_backward_cpu(
        grad_output:        &Tensor,
        input:              &Tensor,
        grid:               &Tensor,
        interpolation_mode: i64,
        padding_mode:       i64,
        align_corners:      bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_sampler_3d_backward_cpu", [&] {
        return grid_sampler_3d_backward_cpu_impl<Scalar>(
          grad_output, input, grid,
          static_cast<GridSamplerInterpolation>(interpolation_mode),
          static_cast<GridSamplerPadding>(padding_mode), align_corners);
      });
        */
}

pub fn grid_sampler(
        input:              &Tensor,
        grid:               &Tensor,
        interpolation_mode: i64,
        padding_mode:       i64,
        align_corners:      bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
        input.defined() && grid.defined(),
        "grid_sampler(): expected input and grid to not be undefined, but input "
        "is ", input, " and grid is ", grid);
      auto input_opt = input.options();
      auto grid_opt = grid.options();
      TORCH_CHECK(
        input_opt.device() == grid_opt.device(),
        "grid_sampler(): expected input and grid to be on same device, but input "
        "is on ", input_opt.device(), " and grid is on ", grid_opt.device());
      TORCH_CHECK(
        input_opt.dtype() == grid_opt.dtype(),
        "grid_sampler(): expected input and grid to have same dtype, but input "
        "has ", input_opt.dtype(), " and grid has ", grid_opt.dtype());
      TORCH_CHECK(
        input_opt.layout() == kStrided && grid_opt.layout() == kStrided,
        "grid_sampler(): expected input and grid to have torch.strided layout, but "
        "input has ", input_opt.layout(), " and grid has ", grid_opt.layout());
      TORCH_CHECK(
        (input.dim() == 4 || input.dim() == 5) && input.dim() == grid.dim(),
        "grid_sampler(): expected 4D or 5D input and grid with same number of "
        "dimensions, but got input with sizes ", input.sizes(),
        " and grid with sizes ", grid.sizes());
      TORCH_CHECK(
        input.size(0) == grid.size(0),
        "grid_sampler(): expected grid and input to have same batch size, but got "
        "input with sizes ", input.sizes(), " and grid with sizes ", grid.sizes());
      TORCH_CHECK(
        grid.size(-1) == input.dim() - 2,
        "grid_sampler(): expected grid to have size ", input.dim() - 2, " in last "
        "dimension, but got grid with sizes ", grid.sizes());
      TORCH_CHECK(
        !(input.dim() == 5 && static_cast<GridSamplerInterpolation>(interpolation_mode) == GridSamplerInterpolation::Bicubic),
        "grid_sampler(): bicubic interpolation only supports 4D input"
      );
      for (i64 i = 2; i < input.dim(); i++) {
        TORCH_CHECK(input.size(i) > 0,
          "grid_sampler(): expected input to have non-empty spatial dimensions, "
          "but input has sizes ", input.sizes(), " with dimension ", i, " being "
          "empty");
      }
      // cudnn does not support inputs larger than 1024
      if (native::cudnn_is_acceptable(input) &&
          native::cudnn_is_acceptable(grid) &&
          native::canUse32BitIndexMath(input) &&
          native::canUse32BitIndexMath(grid) &&
          static_cast<GridSamplerInterpolation>(interpolation_mode) == GridSamplerInterpolation::Bilinear &&
          static_cast<GridSamplerPadding>(padding_mode) == GridSamplerPadding::Zeros &&
          align_corners &&
          input.dim() == 4 &&
          input.size(1) <= 1024) {
        return cudnn_grid_sampler(input, grid);
      }
      if (input.dim() == 4) {
        return grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners);
      } else {
        return grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners);
      }
        */
}
