crate::ix!();

#[inline] pub fn resize_nearest_nchw2x(
    batch_size:   i32,
    num_channels: i32,
    input_height: i32,
    input_width:  i32,
    input:        *const f32,
    output:       *mut f32)  {
    
    todo!();
    /*
        const int output_height = input_height * 2;
      const int output_width = input_width * 2;
      for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < num_channels; ++c) {
          for (int y = 0; y < output_height; ++y) {
            const int in_y = y / 2;

    #if defined(__ARM_NEON__) || defined(__ARM_NEON)
            int vecW = (input_width / 4) * 4; // round down
            int x = 0;
            for (; x < vecW; x += 4) {
              // load 0 1 2 3
              float32x4_t v = vld1q_f32(input + in_y * input_width + x);
              const int oidx = output_width * y + x * 2;
              float32x4x2_t v2 = {{v, v}};
              // store 00 11 22 33
              vst2q_f32(output + oidx + 0, v2);
            }

            // handle remainder
            for (; x < input_width; ++x) {
              const float v = input[in_y * input_width + x];
              const int oidx = output_width * y + x * 2;
              output[oidx + 0] = v;
              output[oidx + 1] = v;
            }
    #else
            for (int x = 0; x < input_width; ++x) {
              const float v = input[in_y * input_width + x];
              const int oidx = output_width * y + x * 2;
              output[oidx + 0] = v;
              output[oidx + 1] = v;
            }
    #endif
          }
          input += input_height * input_width;
          output += output_height * output_width;
        }
      }
    */
}

