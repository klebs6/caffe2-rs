// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gavgpool.cc]


#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm {
    use super::*;

    #[test] fn q8gavgpool_up8x7_neon_n_eq_8_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GAvgPoolMicrokernelTester().m(7).n(8).test(pytorch_q8gavgpool_ukernel_up8x7__neon);

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_eq_8_few_m() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize m = 1; m < 7; m++) {
            GAvgPoolMicrokernelTester().m(m).n(8).test(pytorch_q8gavgpool_ukernel_up8x7__neon);
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_eq_8_all_m_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GAvgPoolMicrokernelTester().m(7).n(8).xStride(11).test(
              pytorch_q8gavgpool_ukernel_up8x7__neon);

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_eq_8_all_m_with_x_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
            GAvgPoolMicrokernelTester().m(7).n(8).xScale(xScale).test(
                pytorch_q8gavgpool_ukernel_up8x7__neon);
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_eq_8_all_m_with_x_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (i32 xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
            GAvgPoolMicrokernelTester()
                .m(7)
                .n(8)
                .xZeroPoint(xZeroPoint)
                .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_eq_8_all_m_with_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
            GAvgPoolMicrokernelTester().m(7).n(8).yScale(yScale).test(
                pytorch_q8gavgpool_ukernel_up8x7__neon);
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_eq_8_all_m_with_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
            GAvgPoolMicrokernelTester()
                .m(7)
                .n(8)
                .yZeroPoint(yZeroPoint)
                .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_eq_8_all_m_with_y_max() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GAvgPoolMicrokernelTester()
              .m(7)
              .n(8)
              .xZeroPoint(128)
              .yZeroPoint(128)
              .xScale(1.0f)
              .yScale(1.0f)
              .yMax(128)
              .test(pytorch_q8gavgpool_ukernel_up8x7__neon);

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_eq_8_all_m_with_y_min() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GAvgPoolMicrokernelTester()
              .m(7)
              .n(8)
              .xZeroPoint(128)
              .yZeroPoint(128)
              .xScale(1.0f)
              .yScale(1.0f)
              .yMin(128)
              .test(pytorch_q8gavgpool_ukernel_up8x7__neon);

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_div_8_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 8; n < 128; n += 24) {
            GAvgPoolMicrokernelTester().m(7).n(n).test(pytorch_q8gavgpool_ukernel_up8x7__neon);
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_div_8_few_m() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 8; n < 128; n += 24) {
            for (usize m = 1; m < 7; m++) {
              GAvgPoolMicrokernelTester().m(m).n(n).test(
                  pytorch_q8gavgpool_ukernel_up8x7__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_gt_8_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            GAvgPoolMicrokernelTester().m(7).n(n).test(pytorch_q8gavgpool_ukernel_up8x7__neon);
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_gt_8_few_m() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            for (usize m = 1; m < 7; m++) {
              GAvgPoolMicrokernelTester().m(m).n(n).test(
                  pytorch_q8gavgpool_ukernel_up8x7__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_gt_8_all_m_with_x_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
              GAvgPoolMicrokernelTester().m(7).n(n).xScale(xScale).test(
                  pytorch_q8gavgpool_ukernel_up8x7__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_gt_8_all_m_with_x_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            for (i32 xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
              GAvgPoolMicrokernelTester()
                  .m(7)
                  .n(n)
                  .xZeroPoint(xZeroPoint)
                  .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_gt_8_all_m_with_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
              GAvgPoolMicrokernelTester().m(7).n(n).yScale(yScale).test(
                  pytorch_q8gavgpool_ukernel_up8x7__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_gt_8_all_m_with_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
              GAvgPoolMicrokernelTester()
                  .m(7)
                  .n(n)
                  .yZeroPoint(yZeroPoint)
                  .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_gt_8_all_m_with_y_max() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            GAvgPoolMicrokernelTester()
                .m(7)
                .n(n)
                .xZeroPoint(128)
                .yZeroPoint(128)
                .xScale(1.0f)
                .yScale(1.0f)
                .yMax(128)
                .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_neon_n_gt_8_all_m_with_y_min() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            GAvgPoolMicrokernelTester()
                .m(7)
                .n(n)
                .xZeroPoint(128)
                .yZeroPoint(128)
                .xScale(1.0f)
                .yScale(1.0f)
                .yMin(128)
                .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_eq_8_2pass_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GAvgPoolMicrokernelTester().m(14).n(8).nr(8).test(
              pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_eq_8_2pass_all_m_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GAvgPoolMicrokernelTester().m(14).n(8).nr(8).xStride(11).test(
              pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_eq_8_2pass_all_m_with_x_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
            GAvgPoolMicrokernelTester().m(14).n(8).nr(8).xScale(xScale).test(
                pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_eq_8_2pass_all_m_with_x_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (i32 xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
            GAvgPoolMicrokernelTester()
                .m(14)
                .n(8)
                .nr(8)
                .xZeroPoint(xZeroPoint)
                .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_eq_8_2pass_all_m_with_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
            GAvgPoolMicrokernelTester().m(14).n(8).nr(8).yScale(yScale).test(
                pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_eq_8_2pass_all_m_with_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
            GAvgPoolMicrokernelTester()
                .m(14)
                .n(8)
                .nr(8)
                .yZeroPoint(yZeroPoint)
                .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_eq_8_2pass_all_m_with_y_max() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GAvgPoolMicrokernelTester()
              .m(14)
              .n(8)
              .nr(8)
              .xZeroPoint(128)
              .yZeroPoint(128)
              .xScale(1.0f)
              .yScale(1.0f)
              .yMax(128)
              .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_eq_8_2pass_all_m_with_y_min() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GAvgPoolMicrokernelTester()
              .m(14)
              .n(8)
              .nr(8)
              .xZeroPoint(128)
              .yZeroPoint(128)
              .xScale(1.0f)
              .yScale(1.0f)
              .yMin(128)
              .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_eq_8_2pass_few_m() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize m = 1; m < 7; m++) {
            GAvgPoolMicrokernelTester().m(7 + m).n(8).nr(8).test(
                pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_eq_8_2pass_few_m_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize m = 1; m < 7; m++) {
            GAvgPoolMicrokernelTester().m(7 + m).n(8).nr(8).xStride(11).test(
                pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_eq_8_multipass_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize m = 14; m <= 35; m += 7) {
            GAvgPoolMicrokernelTester().m(m).n(8).nr(8).test(
                pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_eq_8_multipass_all_m_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize m = 14; m <= 35; m += 7) {
            GAvgPoolMicrokernelTester().m(m).n(8).nr(8).test(
                pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_div_8_2pass_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 8; n < 128; n += 24) {
            GAvgPoolMicrokernelTester().m(14).n(n).nr(8).test(
                pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_div_8_2pass_few_m() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 8; n < 128; n += 24) {
            for (usize m = 1; m < 7; m++) {
              GAvgPoolMicrokernelTester().m(7 + m).n(n).nr(8).test(
                  pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_div_8_multipass_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 8; n < 128; n += 24) {
            for (usize m = 14; m <= 35; m += 7) {
              GAvgPoolMicrokernelTester().m(m).n(n).nr(8).nr(8).test(
                  pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_div_8_multipass_all_m_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 8; n < 128; n += 24) {
            for (usize m = 14; m <= 35; m += 7) {
              GAvgPoolMicrokernelTester().m(m).n(n).nr(8).nr(8).xStride(131).test(
                  pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_gt_8_2pass_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            GAvgPoolMicrokernelTester().m(14).n(n).nr(8).test(
                pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_gt_8_2pass_all_m_with_x_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
            for (usize n = 9; n < 16; n++) {
              GAvgPoolMicrokernelTester().m(14).n(n).nr(8).xScale(xScale).test(
                  pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_gt_8_2pass_all_m_with_x_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (i32 xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
            for (usize n = 9; n < 16; n++) {
              GAvgPoolMicrokernelTester()
                  .m(14)
                  .n(n)
                  .nr(8)
                  .xZeroPoint(xZeroPoint)
                  .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_gt_8_2pass_all_m_with_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
            for (usize n = 9; n < 16; n++) {
              GAvgPoolMicrokernelTester().m(14).n(n).nr(8).yScale(yScale).test(
                  pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_gt_8_2pass_all_m_with_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
            for (usize n = 9; n < 16; n++) {
              GAvgPoolMicrokernelTester()
                  .m(14)
                  .n(n)
                  .nr(8)
                  .yZeroPoint(yZeroPoint)
                  .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_gt_8_2pass_all_m_with_y_max() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            GAvgPoolMicrokernelTester()
                .m(14)
                .n(n)
                .nr(8)
                .xZeroPoint(128)
                .yZeroPoint(128)
                .xScale(1.0f)
                .yScale(1.0f)
                .yMax(128)
                .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_gt_8_2pass_all_m_with_y_min() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            GAvgPoolMicrokernelTester()
                .m(14)
                .n(n)
                .nr(8)
                .xZeroPoint(128)
                .yZeroPoint(128)
                .xScale(1.0f)
                .yScale(1.0f)
                .yMin(128)
                .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_gt_8_2pass_few_m() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            for (usize m = 1; m < 7; m++) {
              GAvgPoolMicrokernelTester().m(7 + m).n(n).nr(8).test(
                  pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_gt_8_multipass_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            for (usize m = 14; m <= 35; m += 7) {
              GAvgPoolMicrokernelTester().m(m).n(n).nr(8).test(
                  pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_neon_n_gt_8_multipass_all_m_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            for (usize m = 14; m <= 35; m += 7) {
              GAvgPoolMicrokernelTester().m(m).n(n).nr(8).xStride(23).test(
                  pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8xm_neon_n_lt_8_small_m() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 8; n++) {
            for (usize m = 1; m < 8; m++) {
              GAvgPoolMicrokernelTester().m(m).n(n).test(
                  pytorch_q8gavgpool_ukernel_up8xm__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8xm_neon_n_lt_8_large_m() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 8; n++) {
            for (usize m = 8; m < 16; m++) {
              GAvgPoolMicrokernelTester().m(m).n(n).test(
                  pytorch_q8gavgpool_ukernel_up8xm__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8xm_neon_n_lt_8_with_x_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 8; n++) {
            for (usize m = 1; m < 16; m += 5) {
              for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
                GAvgPoolMicrokernelTester().m(m).n(n).xScale(xScale).test(
                    pytorch_q8gavgpool_ukernel_up8xm__neon);
              }
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8xm_neon_n_lt_8_with_x_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 8; n++) {
            for (usize m = 1; m < 16; m += 5) {
              for (i32 xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
                GAvgPoolMicrokernelTester()
                    .m(m)
                    .n(n)
                    .xZeroPoint(xZeroPoint)
                    .test(pytorch_q8gavgpool_ukernel_up8xm__neon);
              }
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8xm_neon_n_lt_8_with_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 8; n++) {
            for (usize m = 1; m < 16; m += 5) {
              for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
                GAvgPoolMicrokernelTester().m(m).n(n).yScale(yScale).test(
                    pytorch_q8gavgpool_ukernel_up8xm__neon);
              }
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8xm_neon_n_lt_8_with_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 8; n++) {
            for (usize m = 1; m < 16; m += 5) {
              for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
                GAvgPoolMicrokernelTester()
                    .m(m)
                    .n(n)
                    .yZeroPoint(yZeroPoint)
                    .test(pytorch_q8gavgpool_ukernel_up8xm__neon);
              }
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8xm_neon_n_lt_8_with_y_max() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 8; n++) {
            for (usize m = 1; m < 16; m += 5) {
              GAvgPoolMicrokernelTester()
                  .m(m)
                  .n(n)
                  .xZeroPoint(128)
                  .yZeroPoint(128)
                  .xScale(1.0f)
                  .yScale(1.0f)
                  .yMax(128)
                  .test(pytorch_q8gavgpool_ukernel_up8xm__neon);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8xm_neon_n_lt_8_with_y_min() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 8; n++) {
            for (usize m = 1; m < 16; m += 5) {
              GAvgPoolMicrokernelTester()
                  .m(m)
                  .n(n)
                  .xZeroPoint(128)
                  .yZeroPoint(128)
                  .xScale(1.0f)
                  .yScale(1.0f)
                  .yMin(128)
                  .test(pytorch_q8gavgpool_ukernel_up8xm__neon);
            }
          }

        */
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;

    #[test] fn q8gavgpool_up8x7_sse2_n_eq_8_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GAvgPoolMicrokernelTester().m(7).n(8).test(pytorch_q8gavgpool_ukernel_up8x7__sse2);

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_eq_8_few_m() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize m = 1; m < 7; m++) {
            GAvgPoolMicrokernelTester().m(m).n(8).test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_eq_8_all_m_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GAvgPoolMicrokernelTester().m(7).n(8).xStride(11).test(
              pytorch_q8gavgpool_ukernel_up8x7__sse2);

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_eq_8_all_m_with_x_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
            GAvgPoolMicrokernelTester().m(7).n(8).xScale(xScale).test(
                pytorch_q8gavgpool_ukernel_up8x7__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_eq_8_all_m_with_x_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (i32 xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
            GAvgPoolMicrokernelTester()
                .m(7)
                .n(8)
                .xZeroPoint(xZeroPoint)
                .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_eq_8_all_m_with_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
            GAvgPoolMicrokernelTester().m(7).n(8).yScale(yScale).test(
                pytorch_q8gavgpool_ukernel_up8x7__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_eq_8_all_m_with_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
            GAvgPoolMicrokernelTester()
                .m(7)
                .n(8)
                .yZeroPoint(yZeroPoint)
                .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_eq_8_all_m_with_y_max() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GAvgPoolMicrokernelTester()
              .m(7)
              .n(8)
              .xZeroPoint(128)
              .yZeroPoint(128)
              .xScale(1.0f)
              .yScale(1.0f)
              .yMax(128)
              .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_eq_8_all_m_with_y_min() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GAvgPoolMicrokernelTester()
              .m(7)
              .n(8)
              .xZeroPoint(128)
              .yZeroPoint(128)
              .xScale(1.0f)
              .yScale(1.0f)
              .yMin(128)
              .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_div_8_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 8; n < 128; n += 24) {
            GAvgPoolMicrokernelTester().m(7).n(n).test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_div_8_few_m() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 8; n < 128; n += 24) {
            for (usize m = 1; m < 7; m++) {
              GAvgPoolMicrokernelTester().m(m).n(n).test(
                  pytorch_q8gavgpool_ukernel_up8x7__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_gt_8_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 9; n < 16; n++) {
            GAvgPoolMicrokernelTester().m(7).n(n).test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_gt_8_few_m() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 9; n < 16; n++) {
            for (usize m = 1; m < 7; m++) {
              GAvgPoolMicrokernelTester().m(m).n(n).test(
                  pytorch_q8gavgpool_ukernel_up8x7__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_gt_8_all_m_with_x_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 9; n < 16; n++) {
            for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
              GAvgPoolMicrokernelTester().m(7).n(n).xScale(xScale).test(
                  pytorch_q8gavgpool_ukernel_up8x7__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_gt_8_all_m_with_x_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 9; n < 16; n++) {
            for (i32 xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
              GAvgPoolMicrokernelTester()
                  .m(7)
                  .n(n)
                  .xZeroPoint(xZeroPoint)
                  .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_gt_8_all_m_with_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 9; n < 16; n++) {
            for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
              GAvgPoolMicrokernelTester().m(7).n(n).yScale(yScale).test(
                  pytorch_q8gavgpool_ukernel_up8x7__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_gt_8_all_m_with_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 9; n < 16; n++) {
            for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
              GAvgPoolMicrokernelTester()
                  .m(7)
                  .n(n)
                  .yZeroPoint(yZeroPoint)
                  .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_gt_8_all_m_with_y_max() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 9; n < 16; n++) {
            GAvgPoolMicrokernelTester()
                .m(7)
                .n(n)
                .xZeroPoint(128)
                .yZeroPoint(128)
                .xScale(1.0f)
                .yScale(1.0f)
                .yMax(128)
                .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_up8x7_sse2_n_gt_8_all_m_with_y_min() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 9; n < 16; n++) {
            GAvgPoolMicrokernelTester()
                .m(7)
                .n(n)
                .xZeroPoint(128)
                .yZeroPoint(128)
                .xScale(1.0f)
                .yScale(1.0f)
                .yMin(128)
                .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_eq_8_2pass_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GAvgPoolMicrokernelTester().m(14).n(8).nr(8).test(
              pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_eq_8_2pass_all_m_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GAvgPoolMicrokernelTester().m(14).n(8).nr(8).xStride(11).test(
              pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_eq_8_2pass_all_m_with_x_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
            GAvgPoolMicrokernelTester().m(14).n(8).nr(8).xScale(xScale).test(
                pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_eq_8_2pass_all_m_with_x_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (i32 xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
            GAvgPoolMicrokernelTester()
                .m(14)
                .n(8)
                .nr(8)
                .xZeroPoint(xZeroPoint)
                .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_eq_8_2pass_all_m_with_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
            GAvgPoolMicrokernelTester().m(14).n(8).nr(8).yScale(yScale).test(
                pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_eq_8_2pass_all_m_with_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
            GAvgPoolMicrokernelTester()
                .m(14)
                .n(8)
                .nr(8)
                .yZeroPoint(yZeroPoint)
                .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_eq_8_2pass_all_m_with_y_max() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GAvgPoolMicrokernelTester()
              .m(14)
              .n(8)
              .nr(8)
              .xZeroPoint(128)
              .yZeroPoint(128)
              .xScale(1.0f)
              .yScale(1.0f)
              .yMax(128)
              .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_eq_8_2pass_all_m_with_y_min() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GAvgPoolMicrokernelTester()
              .m(14)
              .n(8)
              .nr(8)
              .xZeroPoint(128)
              .yZeroPoint(128)
              .xScale(1.0f)
              .yScale(1.0f)
              .yMin(128)
              .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_eq_8_2pass_few_m() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize m = 1; m < 7; m++) {
            GAvgPoolMicrokernelTester().m(7 + m).n(8).nr(8).test(
                pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_eq_8_2pass_few_m_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize m = 1; m < 7; m++) {
            GAvgPoolMicrokernelTester().m(7 + m).n(8).nr(8).xStride(11).test(
                pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_eq_8_multipass_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize m = 14; m <= 35; m += 7) {
            GAvgPoolMicrokernelTester().m(m).n(8).nr(8).test(
                pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_eq_8_multipass_all_m_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize m = 14; m <= 35; m += 7) {
            GAvgPoolMicrokernelTester().m(m).n(8).nr(8).test(
                pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_div_8_2pass_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 8; n < 128; n += 24) {
            GAvgPoolMicrokernelTester().m(14).n(n).nr(8).test(
                pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_div_8_2pass_few_m() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 8; n < 128; n += 24) {
            for (usize m = 1; m < 7; m++) {
              GAvgPoolMicrokernelTester().m(7 + m).n(n).nr(8).test(
                  pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_div_8_multipass_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 8; n < 128; n += 24) {
            for (usize m = 14; m <= 35; m += 7) {
              GAvgPoolMicrokernelTester().m(m).n(n).nr(8).nr(8).test(
                  pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_div_8_multipass_all_m_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 8; n < 128; n += 24) {
            for (usize m = 14; m <= 35; m += 7) {
              GAvgPoolMicrokernelTester().m(m).n(n).nr(8).nr(8).xStride(131).test(
                  pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_gt_8_2pass_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 9; n < 16; n++) {
            GAvgPoolMicrokernelTester().m(14).n(n).nr(8).test(
                pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_gt_8_2pass_all_m_with_x_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
            for (usize n = 9; n < 16; n++) {
              GAvgPoolMicrokernelTester().m(14).n(n).nr(8).xScale(xScale).test(
                  pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_gt_8_2pass_all_m_with_x_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (i32 xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
            for (usize n = 9; n < 16; n++) {
              GAvgPoolMicrokernelTester()
                  .m(14)
                  .n(n)
                  .nr(8)
                  .xZeroPoint(xZeroPoint)
                  .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_gt_8_2pass_all_m_with_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
            for (usize n = 9; n < 16; n++) {
              GAvgPoolMicrokernelTester().m(14).n(n).nr(8).yScale(yScale).test(
                  pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_gt_8_2pass_all_m_with_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
            for (usize n = 9; n < 16; n++) {
              GAvgPoolMicrokernelTester()
                  .m(14)
                  .n(n)
                  .nr(8)
                  .yZeroPoint(yZeroPoint)
                  .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_gt_8_2pass_all_m_with_y_max() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 9; n < 16; n++) {
            GAvgPoolMicrokernelTester()
                .m(14)
                .n(n)
                .nr(8)
                .xZeroPoint(128)
                .yZeroPoint(128)
                .xScale(1.0f)
                .yScale(1.0f)
                .yMax(128)
                .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_gt_8_2pass_all_m_with_y_min() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 9; n < 16; n++) {
            GAvgPoolMicrokernelTester()
                .m(14)
                .n(n)
                .nr(8)
                .xZeroPoint(128)
                .yZeroPoint(128)
                .xScale(1.0f)
                .yScale(1.0f)
                .yMin(128)
                .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_gt_8_2pass_few_m() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 9; n < 16; n++) {
            for (usize m = 1; m < 7; m++) {
              GAvgPoolMicrokernelTester().m(7 + m).n(n).nr(8).test(
                  pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_gt_8_multipass_all_m() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 9; n < 16; n++) {
            for (usize m = 14; m <= 35; m += 7) {
              GAvgPoolMicrokernelTester().m(m).n(n).nr(8).test(
                  pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_mp8x7p7q_sse2_n_gt_8_multipass_all_m_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 9; n < 16; n++) {
            for (usize m = 14; m <= 35; m += 7) {
              GAvgPoolMicrokernelTester().m(m).n(n).nr(8).xStride(23).test(
                  pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8xm_sse2_n_lt_8_small_m() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 8; n++) {
            for (usize m = 1; m < 8; m++) {
              GAvgPoolMicrokernelTester().m(m).n(n).test(
                  pytorch_q8gavgpool_ukernel_up8xm__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8xm_sse2_n_lt_8_large_m() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 8; n++) {
            for (usize m = 8; m < 16; m++) {
              GAvgPoolMicrokernelTester().m(m).n(n).test(
                  pytorch_q8gavgpool_ukernel_up8xm__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8xm_sse2_n_lt_8_with_x_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 8; n++) {
            for (usize m = 1; m < 16; m += 5) {
              for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
                GAvgPoolMicrokernelTester().m(m).n(n).xScale(xScale).test(
                    pytorch_q8gavgpool_ukernel_up8xm__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8xm_sse2_n_lt_8_with_x_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 8; n++) {
            for (usize m = 1; m < 16; m += 5) {
              for (i32 xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
                GAvgPoolMicrokernelTester()
                    .m(m)
                    .n(n)
                    .xZeroPoint(xZeroPoint)
                    .test(pytorch_q8gavgpool_ukernel_up8xm__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8xm_sse2_n_lt_8_with_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 8; n++) {
            for (usize m = 1; m < 16; m += 5) {
              for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
                GAvgPoolMicrokernelTester().m(m).n(n).yScale(yScale).test(
                    pytorch_q8gavgpool_ukernel_up8xm__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8xm_sse2_n_lt_8_with_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 8; n++) {
            for (usize m = 1; m < 16; m += 5) {
              for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
                GAvgPoolMicrokernelTester()
                    .m(m)
                    .n(n)
                    .yZeroPoint(yZeroPoint)
                    .test(pytorch_q8gavgpool_ukernel_up8xm__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8xm_sse2_n_lt_8_with_y_max() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 8; n++) {
            for (usize m = 1; m < 16; m += 5) {
              GAvgPoolMicrokernelTester()
                  .m(m)
                  .n(n)
                  .xZeroPoint(128)
                  .yZeroPoint(128)
                  .xScale(1.0f)
                  .yScale(1.0f)
                  .yMax(128)
                  .test(pytorch_q8gavgpool_ukernel_up8xm__sse2);
            }
          }

        */
    }

    #[test] fn q8gavgpool_up8xm_sse2_n_lt_8_with_y_min() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 8; n++) {
            for (usize m = 1; m < 16; m += 5) {
              GAvgPoolMicrokernelTester()
                  .m(m)
                  .n(n)
                  .xZeroPoint(128)
                  .yZeroPoint(128)
                  .xScale(1.0f)
                  .yScale(1.0f)
                  .yMin(128)
                  .test(pytorch_q8gavgpool_ukernel_up8xm__sse2);
            }
          }

        */
    }
}
