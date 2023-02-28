crate::ix!();

#[inline] pub fn resize_nearest_3dnchw2x(
    batch_size:     i32,
    num_channels:   i32,
    temporal_scale: i32,
    input_frames:   i32,
    input_height:   i32,
    input_width:    i32,
    input:          *const f32,
    output:         *mut f32)  {
    
    todo!();
    /*
        const int output_frames = input_frames * temporal_scale;
      const int output_height = input_height * 2;
      const int output_width = input_width * 2;
      for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < num_channels; ++c) {
          for (int f = 0; f < output_frames; ++f ) {
            const int in_f = f / temporal_scale;
            for (int y = 0; y < output_height; ++y) {
              const int in_y = y / 2;

              for (int x = 0; x < input_width; ++x) {
                const float v =
                  input[((in_f * input_height) + in_y) * input_width + x];
                const int oidx = y * output_width + x * 2;
                output[oidx + 0] = v;
                output[oidx + 1] = v;
              }
            }
            output += output_height * output_width;
          }
          input += input_frames * input_height * input_width;
        }
      }
    */
}
