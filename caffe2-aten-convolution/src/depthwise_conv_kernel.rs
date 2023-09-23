/**
  | Depthwise 3x3 Winograd convolution
  | operator
  |
  */
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/DepthwiseConvKernel.h]

lazy_static!{
    /*
    using convolution_depthwise3x3_winograd_fn =
        Tensor (*)(const Tensor &, const Tensor &, const Tensor &,IntArrayRef, IntArrayRef, i64);
    */
}

declare_dispatch!{convolution_depthwise3x3_winograd_fn, convolution_depthwise3x3_winograd_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/DepthwiseConvKernel.cpp]

pub struct Arguments  {

    /**
      | Input layer dimensions
      |
      */
    batch:    i64,
    in_rows:  i64,
    in_cols:  i64,
    stride:   i64,
    pad_rows: i64,
    pad_cols: i64,

    /**
      | Output layer dimensions
      |
      */
    out_rows: i64,

    out_cols: i64,
}

#[inline] pub fn calculate_conv_output_size(
        input_size:  &[i32],
        weight_size: &[i32],
        stride:      &[i32],
        padding:     &[i32]) -> Vec<i64> {
    
    todo!();
        /*
            const auto calc_output_dimension = [](
        const i64 input, const i64 kernel, const i64 stride, const i64 padding) {
        return 1 + (input - kernel + 2 * padding) / stride;
      };

      return vector<i64> {
        input_size[0],
        weight_size[0],
        calc_output_dimension(input_size[2], weight_size[2], stride[0], padding[0]),
        calc_output_dimension(input_size[3], weight_size[3], stride[1], padding[1]),
      };
        */
}

#[cfg(__ARM_NEON__)]
#[inline] pub fn winograd_f2k3_input_transform_inplace_neon(
        d0: *mut float32x4_t,
        d1: *mut float32x4_t,
        d2: *mut float32x4_t,
        d3: *mut float32x4_t)  {
    
    todo!();
        /*
            const float32x4_t wd0 = *d0 - *d2;
      const float32x4_t wd1 = *d1 + *d2;
      const float32x4_t wd2 = -*d1 + *d2;
      const float32x4_t wd3 = *d1 - *d3;
      *d0 = wd0;
      *d1 = wd1;
      *d2 = wd2;
      *d3 = wd3;
        */
}

#[cfg(__ARM_NEON__)]
#[inline] pub fn winograd_f2k3_output_transform_inplace_neon(
        m0: *mut float32x4_t,
        m1: *mut float32x4_t,
        m2: *const float32x4_t,
        m3: *const float32x4_t)  {
    
    todo!();
        /*
            *m0 = *m0 + *m1 + *m2;
      *m1 = *m1 - *m2 - *m3;
        */
}

#[cfg(__ARM_NEON__)]
#[inline] pub fn vmuladdq_f32(
        c: float32x4_t,
        a: float32x4_t,
        b: float32x4_t) -> float32x4_t {
    
    todo!();
        /*
            #if defined(__aarch64__)
      return vfmaq_f32(c, a, b);
    #else
      return vmlaq_f32(c, a, b);
    #endif
        */
}

#[cfg(__ARM_NEON__)]
#[inline] pub fn vmulsubq_f32(
        c: float32x4_t,
        a: float32x4_t,
        b: float32x4_t) -> float32x4_t {
    
    todo!();
        /*
            #if defined(__aarch64__)
      return vfmsq_f32(c, a, b);
    #else
      return vmlsq_f32(c, a, b);
    #endif
        */
}

#[cfg(__ARM_NEON__)]
#[inline] pub fn winograd_f2k3_kernel_transform_neon(
        g0:         float32x4_t,
        g1:         float32x4_t,
        g2:         float32x4_t,
        transform0: *mut float32x4_t,
        transform1: *mut float32x4_t,
        transform2: *mut float32x4_t,
        transform3: *mut float32x4_t)  {
    
    todo!();
        /*
            const float32x4_t const_half = vdupq_n_f32(0.5f);
      float32x4_t half_g0_plus_g2 = const_half * (g0 + g2);
      *transform0 = g0;
      *transform1 = vmuladdq_f32(half_g0_plus_g2, const_half, g1);
      *transform2 = vmulsubq_f32(half_g0_plus_g2, const_half, g1);
      *transform3 = g2;
        */
}

#[cfg(__ARM_NEON__)]
#[inline] pub fn v4f_transpose4x4_neon(m: float32x4x4_t) -> float32x4x4_t {
    
    todo!();
        /*
            float32x4x4_t ret;
      vst4q_f32((float*)(&ret), m);
      return ret;
        */
}

#[cfg(__ARM_NEON__)]
pub fn convolution_depthwise3x3_winograd_impl(
        args:   &Arguments,
        input:  *const f32,
        kernel: *const f32,
        bias:   *const f32,
        output: *mut f32)  {
    
    todo!();
        /*
            const float32x4_t vbias = vsetq_lane_f32(*bias, vdupq_n_f32(0.0), 1);
      float32x4x4_t kernel_tile;

      {
        const float32x4_t g0 = vld1q_f32(kernel);
        const float32x4_t g1 = vld1q_f32(kernel + 3);
        // g2[3] is junk
        const float32x4_t g2 =
            vextq_f32(vld1q_f32(kernel + 5), vld1q_f32(kernel + 5), 1);
        float32x4x4_t w;
        winograd_f2k3_kernel_transform__neon(
            g0, g1, g2, &w.val[0], &w.val[1], &w.val[2], &w.val[3]);
        w = v4f_transpose4x4__neon(w);

        winograd_f2k3_kernel_transform__neon(
            w.val[0],
            w.val[1],
            w.val[2],
            &kernel_tile.val[0],
            &kernel_tile.val[1],
            &kernel_tile.val[2],
            &kernel_tile.val[3]);
      }

    #define TILE                                                  \
      winograd_f2k3_input_transform_inplace__neon(                \
          &input_tile.val[0],                                     \
          &input_tile.val[1],                                     \
          &input_tile.val[2],                                     \
          &input_tile.val[3]);                                    \
      input_tile = v4f_transpose4x4__neon(input_tile);            \
      winograd_f2k3_input_transform_inplace__neon(                \
          &input_tile.val[0],                                     \
          &input_tile.val[1],                                     \
          &input_tile.val[2],                                     \
          &input_tile.val[3]);                                    \
                                                                  \
      for (i64 row = 0; row < 4; ++row) {                         \
        input_tile.val[row] =                                     \
            vmulq_f32(input_tile.val[row], kernel_tile.val[row]); \
      }                                                           \
                                                                  \
      input_tile.val[1] = input_tile.val[1] + vbias;              \
      winograd_f2k3_output_transform_inplace__neon(               \
          &input_tile.val[0],                                     \
          &input_tile.val[1],                                     \
          &input_tile.val[2],                                     \
          &input_tile.val[3]);                                    \
      input_tile = v4f_transpose4x4__neon(input_tile);            \
      winograd_f2k3_output_transform_inplace__neon(               \
          &input_tile.val[0],                                     \
          &input_tile.val[1],                                     \
          &input_tile.val[2],                                     \
          &input_tile.val[3])

      // Non-padded regime.

      // Iterate over non-padded output tiles.
      // TODO: avoid spilling W by breaking out the non-padded vs padded case.
      for (i64 oth = 0; oth < (args.out_rows + 1) / 2; ++oth) {
        for (i64 otw = 0; otw < (args.out_cols + 1) / 2; ++otw) {
          // load input tile for [oth, otw];
          i64 ih = oth * 2 - args.pad_rows;
          i64 iw = otw * 2 - args.pad_cols;
          // fast-path, all accesses in-bounds
          if (C10_LIKELY(
                  ih >= 0 && iw >= 0 && ih + 3 < args.in_rows &&
                      iw + 3 < args.in_cols && 2 * oth + 1 < args.out_rows &&
                      2 * otw + 1 < args.out_cols
                  )) {
            float32x4x4_t input_tile;
            for (i64 row = 0; row < 4; ++row) {
              input_tile.val[row] =
                  vld1q_f32(input + (ih + row) * args.in_cols + iw);
            }

            TILE;

            for (usize row = 0; row < 2; ++row) {
              vst1_f32(
                  output + (oth * 2 + row) * args.out_cols + otw * 2,
                  vget_low_f32(input_tile.val[row]));
            }
          } else {
            float block[4][4];
            for (i64 row = 0; row < 4; ++row) {
              for (i64 col = 0; col < 4; ++col) {
                if (ih + row >= 0 && iw + col >= 0 && ih + row < args.in_rows &&
                    iw + col < args.in_cols) {
                  block[row][col] = input[(ih + row) * args.in_cols + iw + col];
                } else {
                  block[row][col] = 0.0;
                }
              }
            }

            float32x4x4_t input_tile;
            for (i64 row = 0; row < 4; ++row) {
              input_tile.val[row] = vld1q_f32(&block[row][0]);
            }

            TILE;

            float oblock[2][2];
            for (i64 row = 0; row < 2; ++row) {
              vst1_f32(&oblock[row][0], vget_low_f32(input_tile.val[row]));
            }
            for (i64 row = 0; row < 2; ++row) {
              for (i64 col = 0; col < 2; ++col) {
                if (2 * oth + row < args.out_rows &&
                    2 * otw + col < args.out_cols) {
                  output[(2 * oth + row) * args.out_cols + 2 * otw + col] =
                      oblock[row][col];
                }
              }
            }
          }
        }
      }
        */
}

#[cfg(not(__ARM_NEON__))]
pub fn convolution_depthwise3x3_winograd_impl(
    args:   &Arguments,
    input:  *const f32,
    kernel: *const f32,
    bias:   *const f32,
    output: *mut f32)  {

    todo!();
        /*
        
        */
}

pub fn convolution_depthwise3x3_winograd(
        input:                      &Tensor,
        kernel:                     &Tensor,
        bias_potentially_undefined: &Tensor,
        stride:                     &[i32],
        padding:                    &[i32],
        groups:                     i64) -> Tensor {
    
    todo!();
        /*
            const IntArrayRef input_sizes = input.sizes();
      const IntArrayRef kernel_sizes = kernel.sizes();

      Tensor output = empty(
        calculate_conv_output_size(input_sizes, kernel_sizes, stride, padding),
        input.options());

      const IntArrayRef output_sizes = output.sizes();

      const Arguments args {
          input_sizes[0],     // Input N
          input_sizes[2],     // Input H
          input_sizes[3],     // Input W
          stride[0],          // Stride
          padding[0],         // Padding Rows
          padding[1],         // Padding Columns
          output_sizes[2],    // Output H
          output_sizes[3],    // Output W
      };

      const i64 input_hxw = args.in_rows * args.in_cols;
      const i64 output_hxw = args.out_rows * args.out_cols;

      const Tensor bias = bias_potentially_undefined.defined() ?
                          bias_potentially_undefined :
                          zeros({kernel_sizes[0]}, input.options());

      parallel_for(0, args.batch * groups, 0, [&](i64 start, i64 end) {
        for (i64 k = start; k < end; ++k) {
          const i64 g = k % groups;
          convolution_depthwise3x3_winograd_impl(
              args,
              input.data_ptr<float>() + k * input_hxw,
              kernel.data_ptr<float>() + g * 3 * 3,
              bias.data_ptr<float>() + g,
              output.data_ptr<float>() + k * output_hxw);
        }
      });

      return output;
        */
}

register_dispatch!{convolution_depthwise3x3_winograd_stub, &_convolution_depthwise3x3_winograd}
