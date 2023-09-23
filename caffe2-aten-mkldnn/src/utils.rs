crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkldnn/Utils.h]

#[inline] pub fn mkldnn_bf16_device_check() -> bool {
    
    todo!();
        /*
            return cpuinfo_initialize() && cpuinfo_has_x86_avx512bw()
          && cpuinfo_has_x86_avx512vl() && cpuinfo_has_x86_avx512dq();
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkldnn/Utils.cpp]

pub fn pool_output_sizes(
        input_size:  &[i32],
        kernel_size: &[i32],
        stride:      &[i32],
        padding_l:   &[i32],
        padding_r:   &[i32],
        dilation:    &[i32],
        ceil_mode:   bool) -> Vec<i64> {
    
    todo!();
        /*
            vector<i64> output_size(input_size.size());
      // copy N and C
      output_size[0] = input_size[0];
      output_size[1] = input_size[1];

      for (usize i = 2; i < input_size.size(); ++i) {
        output_size[i] = pooling_output_shape_pad_lr<i64>(
          input_size[i],
          kernel_size[i - 2],
          padding_l[i - 2],
          padding_r[i - 2],
          stride[i - 2],
          dilation[i - 2],
          ceil_mode
        );
      }

       return output_size;
        */
}
