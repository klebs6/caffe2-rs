// # vim: ft=none
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8dwconv.cc]



#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm {

    use super::*;

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .test(pytorch_q8dwconv_ukernel_up8x9__neon);

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_eq_8_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .qmin(128)
              .test(pytorch_q8dwconv_ukernel_up8x9__neon);

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_eq_8_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .qmax(128)
              .test(pytorch_q8dwconv_ukernel_up8x9__neon);

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_eq_8_with_input_zero_point_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(255)
              .kernelZeroPoint(0)
              .test(pytorch_q8dwconv_ukernel_up8x9__neon);

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_eq_8_with_kernel_zero_point_only() {
        todo!();
        /*
        
              TEST_REQUIRES_ARM_NEON;
              DWConvMicrokernelTester()
                  .kernelHeight(3)
                  .kernelWidth(3)
                  .cr(8)
                  .channels(8)
                  .width(1)
                  .inputZeroPoint(0)
                  .kernelZeroPoint(255)
                  .test(pytorch_q8dwconv_ukernel_up8x9__neon);
            
        */
    }


    #[test] fn q8dwconv_up8x9_neon_multi_output_channels_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_up8x9__neon);

        */
    }

    #[test] fn q8dwconv_up8x9_neon_multi_output_channels_eq_8_with_subsampling() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .subsampling(2)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_up8x9__neon);

        */
    }

    #[test] fn q8dwconv_up8x9_neon_multi_output_channels_eq_8_with_input_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .inputStride(17)
              .test(pytorch_q8dwconv_ukernel_up8x9__neon);

        */
    }

    #[test] fn q8dwconv_up8x9_neon_multi_output_channels_eq_8_with_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .outputStride(19)
              .test(pytorch_q8dwconv_ukernel_up8x9__neon);

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_up8x9__neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_multi_output_channels_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_up8x9__neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_multi_output_channels_div_8_with_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(171)
                .test(pytorch_q8dwconv_ukernel_up8x9__neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_up8x9__neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_gt_8_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmin(128)
                .test(pytorch_q8dwconv_ukernel_up8x9__neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_gt_8_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmax(128)
                .test(pytorch_q8dwconv_ukernel_up8x9__neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_gt_8_with_input_zero_point_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .inputZeroPoint(255)
                .kernelZeroPoint(0)
                .test(pytorch_q8dwconv_ukernel_up8x9__neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_gt_8_with_kernel_zero_point_only() {
        todo!();
        /*
        
              TEST_REQUIRES_ARM_NEON;
              for (u32 channels = 9; channels < 16; channels++) {
                DWConvMicrokernelTester()
                    .kernelHeight(3)
                    .kernelWidth(3)
                    .cr(8)
                    .channels(channels)
                    .width(1)
                    .inputZeroPoint(0)
                    .kernelZeroPoint(255)
                    .test(pytorch_q8dwconv_ukernel_up8x9__neon);
              }
            
        */
    }

    #[test] fn q8dwconv_up8x9_neon_multi_output_channels_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_up8x9__neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_multi_output_channels_gt_8_with_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(17)
                .test(pytorch_q8dwconv_ukernel_up8x9__neon);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .test(pytorch_q8dwconv_ukernel_mp8x25__neon);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_multi_output_channels_eq_8_with_subsampling() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .subsampling(2)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_mp8x25__neon);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_multi_output_channels_eq_8_with_input_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(5)
              .inputStride(17)
              .test(pytorch_q8dwconv_ukernel_mp8x25__neon);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_multi_output_channels_eq_8_with_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(5)
              .outputStride(19)
              .test(pytorch_q8dwconv_ukernel_mp8x25__neon);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_eq_8_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .qmin(128)
              .test(pytorch_q8dwconv_ukernel_mp8x25__neon);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_eq_8_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .qmax(128)
              .test(pytorch_q8dwconv_ukernel_mp8x25__neon);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_eq_8_with_input_zero_point_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(255)
              .kernelZeroPoint(0)
              .test(pytorch_q8dwconv_ukernel_mp8x25__neon);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_eq_8_with_kernel_zero_point_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(0)
              .kernelZeroPoint(255)
              .test(pytorch_q8dwconv_ukernel_mp8x25__neon);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_multi_output_channels_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(3)
              .test(pytorch_q8dwconv_ukernel_mp8x25__neon);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_multi_output_channels_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_multi_output_channels_div_8_with_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(171)
                .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_gt_8_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmin(128)
                .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_gt_8_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmax(128)
                .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_multi_output_channels_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_multi_output_channels_gt_8_with_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(17)
                .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_eq_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_eq_8_with_qmin_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .qmin(128)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_eq_8_with_qmax_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .qmax(128)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_eq_8_with_input_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(255)
              .kernelZeroPoint(0)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_eq_8_with_kernel_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(0)
              .kernelZeroPoint(255)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_neon_multi_output_channels_eq_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_neon_multi_output_channels_eq_8_with_subsampling_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .subsampling(2)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_neon_multi_output_channels_eq_8_with_input_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .inputStride(17)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_neon_multi_output_channels_eq_8_with_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .outputStride(19)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_div_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_multi_output_channels_div_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_multi_output_channels_div_8_with_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(171)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_gt_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_gt_8_with_qmin_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmin(128)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_gt_8_with_qmax_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmax(128)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_gt_8_with_input_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .inputZeroPoint(255)
                .kernelZeroPoint(0)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_single_output_channels_gt_8_with_kernel_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .inputZeroPoint(0)
                .kernelZeroPoint(255)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_multi_output_channels_gt_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_neon_multi_output_channels_gt_8_with_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(17)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__neon, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_eq_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_multi_output_channels_eq_8_with_subsampling_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .subsampling(2)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_multi_output_channels_eq_8_with_input_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(5)
              .inputStride(17)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_multi_output_channels_eq_8_with_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(5)
              .outputStride(19)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_eq_8_with_qmin_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .qmin(128)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_eq_8_with_qmax_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .qmax(128)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_eq_8_with_input_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(255)
              .kernelZeroPoint(0)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_eq_8_with_kernel_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(0)
              .kernelZeroPoint(255)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_multi_output_channels_eq_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(3)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_div_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_multi_output_channels_div_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_multi_output_channels_div_8_with_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(171)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_gt_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_gt_8_with_qmin_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmin(128)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_single_output_channels_gt_8_with_qmax_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmax(128)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_multi_output_channels_gt_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_neon_multi_output_channels_gt_8_with_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(17)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon, true);
          }

        */
    }
}

#[cfg(CPUINFO_ARCH_ARM)]
mod arm {
    use super::*;

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_eq_8_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .qmin(128)
              .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_eq_8_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .qmax(128)
              .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_eq_8_with_input_zero_point_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(255)
              .kernelZeroPoint(0)
              .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_eq_8_with_kernel_zero_point_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(0)
              .kernelZeroPoint(255)
              .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_multi_output_channels_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_multi_output_channels_eq_8_with_subsampling() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .subsampling(2)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_multi_output_channels_eq_8_with_input_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .inputStride(17)
              .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_multi_output_channels_eq_8_with_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .outputStride(19)
              .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_multi_output_channels_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_multi_output_channels_div_8_with_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(171)
                .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_gt_8_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmin(128)
                .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_gt_8_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmax(128)
                .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_gt_8_with_input_zero_point_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .inputZeroPoint(255)
                .kernelZeroPoint(0)
                .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_gt_8_with_kernel_zero_point_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .inputZeroPoint(0)
                .kernelZeroPoint(255)
                .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_multi_output_channels_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_multi_output_channels_gt_8_with_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(17)
                .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_eq_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_eq_8_with_qmin_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .qmin(128)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_eq_8_with_qmax_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .qmax(128)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_eq_8_with_input_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(255)
              .kernelZeroPoint(0)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_eq_8_with_kernel_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(0)
              .kernelZeroPoint(255)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_multi_output_channels_eq_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_multi_output_channels_eq_8_with_subsampling_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .subsampling(2)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_multi_output_channels_eq_8_with_input_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .inputStride(17)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_multi_output_channels_eq_8_with_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .outputStride(19)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_div_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_multi_output_channels_div_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_multi_output_channels_div_8_with_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(171)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_gt_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_gt_8_with_qmin_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmin(128)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_gt_8_with_qmax_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmax(128)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_gt_8_with_input_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .inputZeroPoint(255)
                .kernelZeroPoint(0)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_single_output_channels_gt_8_with_kernel_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .inputZeroPoint(0)
                .kernelZeroPoint(255)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_multi_output_channels_gt_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_aarch32_neon_multi_output_channels_gt_8_with_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(17)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon, true);
          }

        */
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {

    use super::*;

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .test(pytorch_q8dwconv_ukernel_up8x9__sse2);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_eq_8_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .qmin(128)
              .test(pytorch_q8dwconv_ukernel_up8x9__sse2);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_eq_8_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .qmax(128)
              .test(pytorch_q8dwconv_ukernel_up8x9__sse2);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_eq_8_with_input_zero_point_only() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(255)
              .kernelZeroPoint(0)
              .test(pytorch_q8dwconv_ukernel_up8x9__sse2);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_eq_8_with_kernel_zero_point_only() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(0)
              .kernelZeroPoint(255)
              .test(pytorch_q8dwconv_ukernel_up8x9__sse2);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_multi_output_channels_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_up8x9__sse2);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_multi_output_channels_eq_8_with_subsampling() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .subsampling(2)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_up8x9__sse2);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_multi_output_channels_eq_8_with_input_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .inputStride(17)
              .test(pytorch_q8dwconv_ukernel_up8x9__sse2);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_multi_output_channels_eq_8_with_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .outputStride(19)
              .test(pytorch_q8dwconv_ukernel_up8x9__sse2);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_multi_output_channels_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_multi_output_channels_div_8_with_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(171)
                .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_gt_8_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmin(128)
                .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_gt_8_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmax(128)
                .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_gt_8_with_input_zero_point_only() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .inputZeroPoint(255)
                .kernelZeroPoint(0)
                .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_gt_8_with_kernel_zero_point_only() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .inputZeroPoint(0)
                .kernelZeroPoint(255)
                .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_multi_output_channels_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_multi_output_channels_gt_8_with_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(17)
                .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_eq_8_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .qmin(128)
              .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_eq_8_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .qmax(128)
              .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_eq_8_with_input_zero_point_only() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(255)
              .kernelZeroPoint(0)
              .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_eq_8_with_kernel_zero_point_only() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(0)
              .kernelZeroPoint(255)
              .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_multi_output_channels_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_multi_output_channels_eq_8_with_subsampling() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .subsampling(2)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_multi_output_channels_eq_8_with_input_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(5)
              .inputStride(17)
              .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_multi_output_channels_eq_8_with_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(5)
              .outputStride(19)
              .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_multi_output_channels_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_multi_output_channels_div_8_with_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(171)
                .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_gt_8_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmin(128)
                .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_gt_8_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmax(128)
                .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_gt_8_with_input_zero_point_only() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .inputZeroPoint(255)
                .kernelZeroPoint(0)
                .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_gt_8_with_kernel_zero_point_only() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .inputZeroPoint(0)
                .kernelZeroPoint(255)
                .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_multi_output_channels_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_multi_output_channels_gt_8_with_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(17)
                .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_eq_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_eq_8_with_qmin_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .qmin(128)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_eq_8_with_qmax_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .qmax(128)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_eq_8_with_input_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(255)
              .kernelZeroPoint(0)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_eq_8_with_kernel_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(0)
              .kernelZeroPoint(255)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_multi_output_channels_eq_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_multi_output_channels_eq_8_with_subsampling_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .subsampling(2)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_multi_output_channels_eq_8_with_input_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .inputStride(17)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_multi_output_channels_eq_8_with_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(3)
              .kernelWidth(3)
              .cr(8)
              .channels(8)
              .width(5)
              .outputStride(19)
              .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_div_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_multi_output_channels_div_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_multi_output_channels_div_8_with_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(171)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_gt_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_gt_8_with_qmin_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmin(128)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_gt_8_with_qmax_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmax(128)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_gt_8_with_input_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .inputZeroPoint(255)
                .kernelZeroPoint(0)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_single_output_channels_gt_8_with_kernel_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(1)
                .inputZeroPoint(0)
                .kernelZeroPoint(255)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_multi_output_channels_gt_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_up8x9_sse2_multi_output_channels_gt_8_with_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(3)
                .kernelWidth(3)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(17)
                .test(pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_eq_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_eq_8_with_qmin_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .qmin(128)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_eq_8_with_qmax_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .qmax(128)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_eq_8_with_input_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(255)
              .kernelZeroPoint(0)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_eq_8_with_kernel_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(1)
              .inputZeroPoint(0)
              .kernelZeroPoint(255)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_multi_output_channels_eq_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_multi_output_channels_eq_8_with_subsampling_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .subsampling(2)
              .cr(8)
              .channels(8)
              .width(5)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_multi_output_channels_eq_8_with_input_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(5)
              .inputStride(17)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_multi_output_channels_eq_8_with_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          DWConvMicrokernelTester()
              .kernelHeight(5)
              .kernelWidth(5)
              .cr(8)
              .channels(8)
              .width(5)
              .outputStride(19)
              .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_div_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_multi_output_channels_div_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_multi_output_channels_div_8_with_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 16; channels < 128; channels += 24) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(171)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_gt_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_gt_8_with_qmin_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmin(128)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_gt_8_with_qmax_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .qmax(128)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_gt_8_with_input_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .inputZeroPoint(255)
                .kernelZeroPoint(0)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_single_output_channels_gt_8_with_kernel_zero_point_only_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(1)
                .inputZeroPoint(0)
                .kernelZeroPoint(255)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_multi_output_channels_gt_8_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(5)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);
          }

        */
    }

    #[test] fn q8dwconv_mp8x25_sse2_multi_output_channels_gt_8_with_stride_per_channel() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (u32 channels = 9; channels < 16; channels++) {
            DWConvMicrokernelTester()
                .kernelHeight(5)
                .kernelWidth(5)
                .cr(8)
                .channels(channels)
                .width(5)
                .outputStride(17)
                .test(pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2, true);
          }

        */
    }
}
