// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8avgpool.cc]

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm {

    use super::*;

    #[test] fn q8avgpool_up8xm_neon_kc_lt_8_small_ks() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize kc = 1; kc < 8; kc++) {
            for (usize ks = 1; ks < 8; ks++) {
              for (usize kh = 1; kh <= ks; kh++) {
                for (usize kw = 1; kw <= ks; kw++) {
                  if (kh * kw == ks) {
                    AvgPoolMicrokernelTester().kr(8).kh(kh).kw(kw).kc(kc).test(
                        pytorch_q8avgpool_ukernel_up8xm__neon);
                  }
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_neon_kc_lt_8_large_ks() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize kc = 1; kc < 8; kc++) {
            for (usize ks = 8; ks < 16; ks++) {
              AvgPoolMicrokernelTester().kr(8).kh(ks).kw(1).kc(kc).test(
                  pytorch_q8avgpool_ukernel_up8xm__neon);
              AvgPoolMicrokernelTester().kr(8).kh(1).kw(ks).kc(kc).test(
                  pytorch_q8avgpool_ukernel_up8xm__neon);
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_neon_kc_lt_8_with_x_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 3; n += 2) {
            for (usize kc = 1; kc < 8; kc++) {
              for (usize ks : vector<usize>{{2, 3, 5}}) {
                for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
                  AvgPoolMicrokernelTester()
                      .kr(8)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .xScale(xScale)
                      .iterations(1)
                      .test(pytorch_q8avgpool_ukernel_up8xm__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_neon_kc_lt_8_with_x_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 3; n += 2) {
            for (usize kc = 1; kc < 8; kc++) {
              for (usize ks : vector<usize>{{2, 3, 5}}) {
                for (i32 xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
                  AvgPoolMicrokernelTester()
                      .kr(8)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .xZeroPoint(u8(xZeroPoint))
                      .iterations(1)
                      .test(pytorch_q8avgpool_ukernel_up8xm__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_neon_kc_lt_8_with_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 3; n += 2) {
            for (usize kc = 1; kc < 8; kc++) {
              for (usize ks : vector<usize>{{2, 3, 5}}) {
                for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
                  AvgPoolMicrokernelTester()
                      .kr(8)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .yScale(yScale)
                      .iterations(1)
                      .test(pytorch_q8avgpool_ukernel_up8xm__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_neon_kc_lt_8_with_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 3; n += 2) {
            for (usize kc = 1; kc < 8; kc++) {
              for (usize ks : vector<usize>{{2, 3, 5}}) {
                for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
                  AvgPoolMicrokernelTester()
                      .kr(8)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .yZeroPoint(u8(yZeroPoint))
                      .iterations(1)
                      .test(pytorch_q8avgpool_ukernel_up8xm__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_neon_kc_lt_8_with_y_max() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 3; n += 2) {
            for (usize kc = 1; kc < 8; kc++) {
              for (usize ks : vector<usize>{{2, 3, 5}}) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .xZeroPoint(128)
                    .yZeroPoint(128)
                    .xScale(1.0f)
                    .yScale(1.0f)
                    .yMax(128)
                    .iterations(3)
                    .test(pytorch_q8avgpool_ukernel_up8xm__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_neon_kc_lt_8_with_y_min() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 3; n += 2) {
            for (usize kc = 1; kc < 8; kc++) {
              for (usize ks : vector<usize>{{2, 3, 5}}) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .xZeroPoint(128)
                    .yZeroPoint(128)
                    .xScale(1.0f)
                    .yScale(1.0f)
                    .yMin(128)
                    .iterations(3)
                    .test(pytorch_q8avgpool_ukernel_up8xm__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_neon_small_n() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize kc = 1; kc < 8; kc++) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .iterations(3)
                    .test(pytorch_q8avgpool_ukernel_up8xm__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_neon_small_n_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize kc = 1; kc < 8; kc++) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .xStride(11)
                    .iterations(3)
                    .test(pytorch_q8avgpool_ukernel_up8xm__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_neon_small_n_with_y_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize kc = 1; kc < 8; kc++) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .yStride(13)
                    .iterations(3)
                    .test(pytorch_q8avgpool_ukernel_up8xm__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_neon_small_n_with_s() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize s = 2; s <= 5; s++) {
                for (usize kc = 1; kc < 8; kc++) {
                  AvgPoolMicrokernelTester()
                      .kr(8)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .s(s)
                      .iterations(1)
                      .test(pytorch_q8avgpool_ukernel_up8xm__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_kc_eq_8_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).kc(8);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                tester.kh(kh).kw(kw).test(pytorch_q8avgpool_ukernel_up8x9__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_kc_eq_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).kc(8);
          for (usize ks = 2; ks < tester.mr(); ks++) {
            for (usize kh = 1; kh <= ks; kh++) {
              for (usize kw = 1; kw <= ks; kw++) {
                if (kh * kw == ks) {
                  tester.kh(kh).kw(kw).test(pytorch_q8avgpool_ukernel_up8x9__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_kc_div_8_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 8; kc < 128; kc += 24) {
                  tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_up8x9__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_kc_div_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).iterations(3);
          for (usize ks = 2; ks < tester.mr(); ks++) {
            for (usize kh = 1; kh <= ks; kh++) {
              for (usize kw = 1; kw <= ks; kw++) {
                if (kh * kw == ks) {
                  for (usize kc = 8; kc < 128; kc += 24) {
                    tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_up8x9__neon);
                  }
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_kc_div_8_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).iterations(3);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 8; kc < 128; kc += 24) {
                  tester.kh(kh).kw(kw).kc(kc).xStride(131).test(
                      pytorch_q8avgpool_ukernel_up8x9__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_kc_gt_8_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 9; kc < 16; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_up8x9__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_kc_gt_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).iterations(3);
          for (usize ks = 2; ks < tester.mr(); ks++) {
            for (usize kh = 1; kh <= ks; kh++) {
              for (usize kw = 1; kw <= ks; kw++) {
                if (kh * kw == ks) {
                  for (usize kc = 9; kc < 16; kc++) {
                    tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_up8x9__neon);
                  }
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_kc_gt_8_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).iterations(3);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 9; kc < 16; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).xStride(23).test(
                      pytorch_q8avgpool_ukernel_up8x9__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_kc_div_8_with_x_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .n(n)
                    .kh(3)
                    .kw(3)
                    .kc(kc)
                    .xScale(xScale)
                    .iterations(2)
                    .test(pytorch_q8avgpool_ukernel_up8x9__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_kc_div_8_with_x_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              for (i32 xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .n(n)
                    .kh(3)
                    .kw(3)
                    .kc(kc)
                    .xZeroPoint(u8(xZeroPoint))
                    .iterations(3)
                    .test(pytorch_q8avgpool_ukernel_up8x9__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_kc_div_8_with_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .n(n)
                    .kh(3)
                    .kw(3)
                    .kc(kc)
                    .yScale(yScale)
                    .iterations(2)
                    .test(pytorch_q8avgpool_ukernel_up8x9__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_kc_div_8_with_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .n(n)
                    .kh(3)
                    .kw(3)
                    .kc(kc)
                    .yZeroPoint(u8(yZeroPoint))
                    .iterations(3)
                    .test(pytorch_q8avgpool_ukernel_up8x9__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_kc_div_8_with_y_max() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              AvgPoolMicrokernelTester()
                  .kr(8)
                  .mr(9)
                  .n(n)
                  .kh(3)
                  .kw(3)
                  .kc(kc)
                  .xZeroPoint(128)
                  .yZeroPoint(128)
                  .xScale(1.0f)
                  .yScale(1.0f)
                  .yMax(128)
                  .test(pytorch_q8avgpool_ukernel_up8x9__neon);
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_kc_div_8_with_y_min() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              AvgPoolMicrokernelTester()
                  .kr(8)
                  .mr(9)
                  .n(n)
                  .kh(3)
                  .kw(3)
                  .kc(kc)
                  .xZeroPoint(128)
                  .yZeroPoint(128)
                  .xScale(1.0f)
                  .yScale(1.0f)
                  .yMin(128)
                  .test(pytorch_q8avgpool_ukernel_up8x9__neon);
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_small_n() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3}}) {
              for (usize kc = 8; kc < 25; kc += 5) {
                AvgPoolMicrokernelTester().kr(8).mr(9).n(n).kh(ks).kw(ks).kc(kc).test(
                    pytorch_q8avgpool_ukernel_up8x9__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_small_n_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3}}) {
              for (usize kc = 8; kc < 25; kc += 5) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .xStride(29)
                    .test(pytorch_q8avgpool_ukernel_up8x9__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_small_n_with_y_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3}}) {
              for (usize kc = 8; kc < 25; kc += 5) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .yStride(31)
                    .test(pytorch_q8avgpool_ukernel_up8x9__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_neon_small_n_with_s() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3}}) {
              for (usize kc = 8; kc < 25; kc += 5) {
                for (usize s = 2; s <= ks; s++) {
                  AvgPoolMicrokernelTester()
                      .kr(8)
                      .mr(9)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .s(s)
                      .test(pytorch_q8avgpool_ukernel_up8x9__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_eq_8_twopass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).kc(8);
          const usize ks = tester.mr() + tester.qr();
          for (usize kh = 1; kh <= ks; kh++) {
            for (usize kw = 1; kw <= ks; kw++) {
              if (kh * kw == ks) {
                tester.kh(kh).kw(kw).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_eq_8_twopass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).kc(8);
          for (usize ks = 10; ks < tester.mr() + tester.qr(); ks++) {
            tester.kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
            tester.kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_eq_8_multipass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize ks : vector<usize>{{25, 49}}) {
            auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).kc(8);
            for (usize kh = 1; kh <= ks; kh++) {
              for (usize kw = 1; kw <= ks; kw++) {
                if (kh * kw == ks) {
                  tester.kh(kh).kw(kw).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_eq_8_multipass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize ksMax : vector<usize>{{25, 49}}) {
            auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).kc(8);
            for (usize ks = ksMax - tester.qr() + 1; ks < ksMax; ks++) {
              tester.kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
              tester.kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_div_8_twopass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
          const usize ks = 17;
          for (usize kc = 8; kc < 128; kc += 24) {
            tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
            tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_div_8_twopass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
          for (usize ks = 10; ks < tester.mr() + tester.qr(); ks++) {
            for (usize kc = 8; kc < 128; kc += 24) {
              tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
              tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_div_8_twopass_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
          const usize ks = tester.mr() + tester.qr();
          for (usize kh = 1; kh <= ks; kh++) {
            for (usize kw = 1; kw <= ks; kw++) {
              if (kh * kw == ks) {
                for (usize kc = 8; kc < 128; kc += 24) {
                  tester.kh(kh).kw(kw).kc(kc).xStride(131).test(
                      pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_div_8_multipass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize ks : vector<usize>{{25, 49}}) {
            auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
            for (usize kh = 1; kh <= ks; kh++) {
              for (usize kw = 1; kw <= ks; kw++) {
                if (kh * kw == ks) {
                  for (usize kc = 8; kc < 128; kc += 24) {
                    tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
                  }
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_div_8_multipass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize ksMax : vector<usize>{{25, 49}}) {
            auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
            for (usize ks = ksMax - tester.qr() + 1; ks < ksMax; ks++) {
              for (usize kc = 8; kc < 128; kc += 24) {
                tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
                tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_div_8_multipass_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize ks : vector<usize>{{25, 49}}) {
            auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
            for (usize kh = 1; kh <= ks; kh++) {
              for (usize kw = 1; kw <= ks; kw++) {
                if (kh * kw == ks) {
                  for (usize kc = 8; kc < 128; kc += 24) {
                    tester.kh(kh).kw(kw).kc(kc).xStride(131).test(
                        pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
                  }
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_gt_8_twopass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
          const usize ks = tester.mr() + tester.qr();
          for (usize kh = 1; kh <= ks; kh++) {
            for (usize kw = 1; kw <= ks; kw++) {
              if (kh * kw == ks) {
                for (usize kc = 9; kc < 16; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_gt_8_twopass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
          for (usize ks = 10; ks < tester.mr() + tester.qr(); ks++) {
            for (usize kc = 9; kc < 16; kc++) {
              tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
              tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_gt_8_twopass_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
          const usize ks = tester.mr() + tester.qr();
          for (usize kh = 1; kh <= ks; kh++) {
            for (usize kw = 1; kw <= ks; kw++) {
              if (kh * kw == ks) {
                for (usize kc = 9; kc < 16; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).xStride(23).test(
                      pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_gt_8_multipass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize ks : vector<usize>{{25, 49}}) {
            auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
            for (usize kh = 1; kh <= ks; kh++) {
              for (usize kw = 1; kw <= ks; kw++) {
                if (kh * kw == ks) {
                  for (usize kc = 9; kc < 16; kc++) {
                    tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
                  }
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_gt_8_multipass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize ksMax : vector<usize>{{25, 49}}) {
            auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
            for (usize ks = ksMax - tester.qr() + 1; ks < ksMax; ks++) {
              for (usize kc = 9; kc < 16; kc++) {
                tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
                tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_gt_8_multipass_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize ks : vector<usize>{{25, 49}}) {
            auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
            for (usize kh = 1; kh <= ks; kh++) {
              for (usize kw = 1; kw <= ks; kw++) {
                if (kh * kw == ks) {
                  for (usize kc = 9; kc < 16; kc++) {
                    tester.kh(kh).kw(kw).kc(kc).xStride(23).test(
                        pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
                  }
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_div_8_with_x_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(5)
                    .kw(5)
                    .kc(kc)
                    .xScale(xScale)
                    .iterations(1)
                    .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_div_8_with_x_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              for (i32 xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(5)
                    .kw(5)
                    .kc(kc)
                    .xZeroPoint(u8(xZeroPoint))
                    .iterations(1)
                    .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_div_8_with_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(5)
                    .kw(5)
                    .kc(kc)
                    .yScale(yScale)
                    .iterations(1)
                    .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_div_8_with_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(5)
                    .kw(5)
                    .kc(kc)
                    .yZeroPoint(u8(yZeroPoint))
                    .iterations(1)
                    .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_div_8_with_y_max() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              AvgPoolMicrokernelTester()
                  .kr(8)
                  .mr(9)
                  .qr(8)
                  .n(n)
                  .kh(5)
                  .kw(5)
                  .kc(kc)
                  .xZeroPoint(128)
                  .yZeroPoint(128)
                  .xScale(1.0f)
                  .yScale(1.0f)
                  .yMax(128)
                  .iterations(3)
                  .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_kc_div_8_with_y_min() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              AvgPoolMicrokernelTester()
                  .kr(8)
                  .mr(9)
                  .qr(8)
                  .n(n)
                  .kh(5)
                  .kw(5)
                  .kc(kc)
                  .xZeroPoint(128)
                  .yZeroPoint(128)
                  .xScale(1.0f)
                  .yScale(1.0f)
                  .yMin(128)
                  .iterations(3)
                  .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_small_n() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{5, 7}}) {
              for (usize kc = 8; kc < 25; kc += 5) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_small_n_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{5, 7}}) {
              for (usize kc = 8; kc < 25; kc += 5) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .xStride(29)
                    .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_small_n_with_y_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{5, 7}}) {
              for (usize kc = 8; kc < 25; kc += 5) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .yStride(31)
                    .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_neon_small_n_with_s() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{5, 7}}) {
              for (usize s = 2; s <= 5; s++) {
                for (usize kc = 8; kc < 25; kc += 5) {
                  AvgPoolMicrokernelTester()
                      .kr(8)
                      .mr(9)
                      .qr(8)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .s(s)
                      .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
                }
              }
            }
          }

        */
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {

    use super::*;

    #[test] fn q8avgpool_up8xm_sse2_kc_lt_8_small_ks() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize kc = 1; kc < 8; kc++) {
            for (usize ks = 1; ks < 8; ks++) {
              for (usize kh = 1; kh <= ks; kh++) {
                for (usize kw = 1; kw <= ks; kw++) {
                  if (kh * kw == ks) {
                    AvgPoolMicrokernelTester().kr(8).kh(kh).kw(kw).kc(kc).test(
                        pytorch_q8avgpool_ukernel_up8xm__sse2);
                  }
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_sse2_kc_lt_8_large_ks() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize kc = 1; kc < 8; kc++) {
            for (usize ks = 8; ks < 16; ks++) {
              AvgPoolMicrokernelTester().kr(8).kh(ks).kw(1).kc(kc).test(
                  pytorch_q8avgpool_ukernel_up8xm__sse2);
              AvgPoolMicrokernelTester().kr(8).kh(1).kw(ks).kc(kc).test(
                  pytorch_q8avgpool_ukernel_up8xm__sse2);
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_sse2_kc_lt_8_with_x_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 3; n += 2) {
            for (usize kc = 1; kc < 8; kc++) {
              for (usize ks : vector<usize>{{2, 3, 5}}) {
                for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
                  AvgPoolMicrokernelTester()
                      .kr(8)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .xScale(xScale)
                      .iterations(1)
                      .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_sse2_kc_lt_8_with_x_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 3; n += 2) {
            for (usize kc = 1; kc < 8; kc++) {
              for (usize ks : vector<usize>{{2, 3, 5}}) {
                for (i32 xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
                  AvgPoolMicrokernelTester()
                      .kr(8)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .xZeroPoint(u8(xZeroPoint))
                      .iterations(1)
                      .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_sse2_kc_lt_8_with_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 3; n += 2) {
            for (usize kc = 1; kc < 8; kc++) {
              for (usize ks : vector<usize>{{2, 3, 5}}) {
                for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
                  AvgPoolMicrokernelTester()
                      .kr(8)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .yScale(yScale)
                      .iterations(1)
                      .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_sse2_kc_lt_8_with_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 3; n += 2) {
            for (usize kc = 1; kc < 8; kc++) {
              for (usize ks : vector<usize>{{2, 3, 5}}) {
                for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
                  AvgPoolMicrokernelTester()
                      .kr(8)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .yZeroPoint(u8(yZeroPoint))
                      .iterations(1)
                      .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_sse2_kc_lt_8_with_y_max() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 3; n += 2) {
            for (usize kc = 1; kc < 8; kc++) {
              for (usize ks : vector<usize>{{2, 3, 5}}) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .xZeroPoint(128)
                    .yZeroPoint(128)
                    .xScale(1.0f)
                    .yScale(1.0f)
                    .yMax(128)
                    .iterations(3)
                    .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_sse2_kc_lt_8_with_y_min() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 3; n += 2) {
            for (usize kc = 1; kc < 8; kc++) {
              for (usize ks : vector<usize>{{2, 3, 5}}) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .xZeroPoint(128)
                    .yZeroPoint(128)
                    .xScale(1.0f)
                    .yScale(1.0f)
                    .yMin(128)
                    .iterations(3)
                    .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_sse2_small_n() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize kc = 1; kc < 8; kc++) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .iterations(3)
                    .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_sse2_small_n_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize kc = 1; kc < 8; kc++) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .xStride(11)
                    .iterations(3)
                    .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_sse2_small_n_with_y_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize kc = 1; kc < 8; kc++) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .yStride(13)
                    .iterations(3)
                    .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8xm_sse2_small_n_with_s() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize s = 2; s <= 5; s++) {
                for (usize kc = 1; kc < 8; kc++) {
                  AvgPoolMicrokernelTester()
                      .kr(8)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .s(s)
                      .iterations(1)
                      .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_kc_eq_8_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).kc(8);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                tester.kh(kh).kw(kw).test(pytorch_q8avgpool_ukernel_up8x9__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_kc_eq_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).kc(8);
          for (usize ks = 2; ks < tester.mr(); ks++) {
            for (usize kh = 1; kh <= ks; kh++) {
              for (usize kw = 1; kw <= ks; kw++) {
                if (kh * kw == ks) {
                  tester.kh(kh).kw(kw).test(pytorch_q8avgpool_ukernel_up8x9__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_kc_div_8_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 8; kc < 128; kc += 24) {
                  tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_up8x9__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_kc_div_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).iterations(3);
          for (usize ks = 2; ks < tester.mr(); ks++) {
            for (usize kh = 1; kh <= ks; kh++) {
              for (usize kw = 1; kw <= ks; kw++) {
                if (kh * kw == ks) {
                  for (usize kc = 8; kc < 128; kc += 24) {
                    tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_up8x9__sse2);
                  }
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_kc_div_8_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).iterations(3);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 8; kc < 128; kc += 24) {
                  tester.kh(kh).kw(kw).kc(kc).xStride(131).test(
                      pytorch_q8avgpool_ukernel_up8x9__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_kc_gt_8_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 9; kc < 16; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_up8x9__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_kc_gt_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).iterations(3);
          for (usize ks = 2; ks < tester.mr(); ks++) {
            for (usize kh = 1; kh <= ks; kh++) {
              for (usize kw = 1; kw <= ks; kw++) {
                if (kh * kw == ks) {
                  for (usize kc = 9; kc < 16; kc++) {
                    tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_up8x9__sse2);
                  }
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_kc_gt_8_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).iterations(3);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 9; kc < 16; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).xStride(23).test(
                      pytorch_q8avgpool_ukernel_up8x9__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_kc_div_8_with_x_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .n(n)
                    .kh(3)
                    .kw(3)
                    .kc(kc)
                    .xScale(xScale)
                    .iterations(2)
                    .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_kc_div_8_with_x_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              for (i32 xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .n(n)
                    .kh(3)
                    .kw(3)
                    .kc(kc)
                    .xZeroPoint(u8(xZeroPoint))
                    .iterations(3)
                    .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_kc_div_8_with_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .n(n)
                    .kh(3)
                    .kw(3)
                    .kc(kc)
                    .yScale(yScale)
                    .iterations(2)
                    .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_kc_div_8_with_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .n(n)
                    .kh(3)
                    .kw(3)
                    .kc(kc)
                    .yZeroPoint(u8(yZeroPoint))
                    .iterations(3)
                    .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_kc_div_8_with_y_max() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              AvgPoolMicrokernelTester()
                  .kr(8)
                  .mr(9)
                  .n(n)
                  .kh(3)
                  .kw(3)
                  .kc(kc)
                  .xZeroPoint(128)
                  .yZeroPoint(128)
                  .xScale(1.0f)
                  .yScale(1.0f)
                  .yMax(128)
                  .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_kc_div_8_with_y_min() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              AvgPoolMicrokernelTester()
                  .kr(8)
                  .mr(9)
                  .n(n)
                  .kh(3)
                  .kw(3)
                  .kc(kc)
                  .xZeroPoint(128)
                  .yZeroPoint(128)
                  .xScale(1.0f)
                  .yScale(1.0f)
                  .yMin(128)
                  .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_small_n() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3}}) {
              for (usize kc = 8; kc < 25; kc += 5) {
                AvgPoolMicrokernelTester().kr(8).mr(9).n(n).kh(ks).kw(ks).kc(kc).test(
                    pytorch_q8avgpool_ukernel_up8x9__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_small_n_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3}}) {
              for (usize kc = 8; kc < 25; kc += 5) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .xStride(29)
                    .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_small_n_with_y_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3}}) {
              for (usize kc = 8; kc < 25; kc += 5) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .yStride(31)
                    .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_up8x9_sse2_small_n_with_s() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3}}) {
              for (usize kc = 8; kc < 25; kc += 5) {
                for (usize s = 2; s <= ks; s++) {
                  AvgPoolMicrokernelTester()
                      .kr(8)
                      .mr(9)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .s(s)
                      .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_eq_8_twopass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).kc(8);
          const usize ks = tester.mr() + tester.qr();
          for (usize kh = 1; kh <= ks; kh++) {
            for (usize kw = 1; kw <= ks; kw++) {
              if (kh * kw == ks) {
                tester.kh(kh).kw(kw).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_eq_8_twopass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).kc(8);
          for (usize ks = 10; ks < tester.mr() + tester.qr(); ks++) {
            tester.kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
            tester.kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_eq_8_multipass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize ks : vector<usize>{{25, 49}}) {
            auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).kc(8);
            for (usize kh = 1; kh <= ks; kh++) {
              for (usize kw = 1; kw <= ks; kw++) {
                if (kh * kw == ks) {
                  tester.kh(kh).kw(kw).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_eq_8_multipass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize ksMax : vector<usize>{{25, 49}}) {
            auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).kc(8);
            for (usize ks = ksMax - tester.qr() + 1; ks < ksMax; ks++) {
              tester.kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
              tester.kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_div_8_twopass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
          const usize ks = 17;
          for (usize kc = 8; kc < 128; kc += 24) {
            tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
            tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_div_8_twopass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
          for (usize ks = 10; ks < tester.mr() + tester.qr(); ks++) {
            for (usize kc = 8; kc < 128; kc += 24) {
              tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
              tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_div_8_twopass_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
          const usize ks = tester.mr() + tester.qr();
          for (usize kh = 1; kh <= ks; kh++) {
            for (usize kw = 1; kw <= ks; kw++) {
              if (kh * kw == ks) {
                for (usize kc = 8; kc < 128; kc += 24) {
                  tester.kh(kh).kw(kw).kc(kc).xStride(131).test(
                      pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_div_8_multipass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize ks : vector<usize>{{25, 49}}) {
            auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
            for (usize kh = 1; kh <= ks; kh++) {
              for (usize kw = 1; kw <= ks; kw++) {
                if (kh * kw == ks) {
                  for (usize kc = 8; kc < 128; kc += 24) {
                    tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
                  }
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_div_8_multipass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize ksMax : vector<usize>{{25, 49}}) {
            auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
            for (usize ks = ksMax - tester.qr() + 1; ks < ksMax; ks++) {
              for (usize kc = 8; kc < 128; kc += 24) {
                tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
                tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_div_8_multipass_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize ks : vector<usize>{{25, 49}}) {
            auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
            for (usize kh = 1; kh <= ks; kh++) {
              for (usize kw = 1; kw <= ks; kw++) {
                if (kh * kw == ks) {
                  for (usize kc = 8; kc < 128; kc += 24) {
                    tester.kh(kh).kw(kw).kc(kc).xStride(131).test(
                        pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
                  }
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_gt_8_twopass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
          const usize ks = tester.mr() + tester.qr();
          for (usize kh = 1; kh <= ks; kh++) {
            for (usize kw = 1; kw <= ks; kw++) {
              if (kh * kw == ks) {
                for (usize kc = 9; kc < 16; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_gt_8_twopass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
          for (usize ks = 10; ks < tester.mr() + tester.qr(); ks++) {
            for (usize kc = 9; kc < 16; kc++) {
              tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
              tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_gt_8_twopass_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
          const usize ks = tester.mr() + tester.qr();
          for (usize kh = 1; kh <= ks; kh++) {
            for (usize kw = 1; kw <= ks; kw++) {
              if (kh * kw == ks) {
                for (usize kc = 9; kc < 16; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).xStride(23).test(
                      pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_gt_8_multipass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize ks : vector<usize>{{25, 49}}) {
            auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
            for (usize kh = 1; kh <= ks; kh++) {
              for (usize kw = 1; kw <= ks; kw++) {
                if (kh * kw == ks) {
                  for (usize kc = 9; kc < 16; kc++) {
                    tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
                  }
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_gt_8_multipass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize ksMax : vector<usize>{{25, 49}}) {
            auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
            for (usize ks = ksMax - tester.qr() + 1; ks < ksMax; ks++) {
              for (usize kc = 9; kc < 16; kc++) {
                tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
                tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_gt_8_multipass_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize ks : vector<usize>{{25, 49}}) {
            auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
            for (usize kh = 1; kh <= ks; kh++) {
              for (usize kw = 1; kw <= ks; kw++) {
                if (kh * kw == ks) {
                  for (usize kc = 9; kc < 16; kc++) {
                    tester.kh(kh).kw(kw).kc(kc).xStride(23).test(
                        pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
                  }
                }
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_div_8_with_x_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(5)
                    .kw(5)
                    .kc(kc)
                    .xScale(xScale)
                    .iterations(1)
                    .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_div_8_with_x_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              for (i32 xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(5)
                    .kw(5)
                    .kc(kc)
                    .xZeroPoint(u8(xZeroPoint))
                    .iterations(1)
                    .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_div_8_with_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(5)
                    .kw(5)
                    .kc(kc)
                    .yScale(yScale)
                    .iterations(1)
                    .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_div_8_with_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(5)
                    .kw(5)
                    .kc(kc)
                    .yZeroPoint(u8(yZeroPoint))
                    .iterations(1)
                    .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_div_8_with_y_max() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              AvgPoolMicrokernelTester()
                  .kr(8)
                  .mr(9)
                  .qr(8)
                  .n(n)
                  .kh(5)
                  .kw(5)
                  .kc(kc)
                  .xZeroPoint(128)
                  .yZeroPoint(128)
                  .xScale(1.0f)
                  .yScale(1.0f)
                  .yMax(128)
                  .iterations(3)
                  .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_kc_div_8_with_y_min() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n <= 5; n += 2) {
            for (usize kc = 8; kc < 128; kc += 24) {
              AvgPoolMicrokernelTester()
                  .kr(8)
                  .mr(9)
                  .qr(8)
                  .n(n)
                  .kh(5)
                  .kw(5)
                  .kc(kc)
                  .xZeroPoint(128)
                  .yZeroPoint(128)
                  .xScale(1.0f)
                  .yScale(1.0f)
                  .yMin(128)
                  .iterations(3)
                  .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_small_n() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{5, 7}}) {
              for (usize kc = 8; kc < 25; kc += 5) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_small_n_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{5, 7}}) {
              for (usize kc = 8; kc < 25; kc += 5) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .xStride(29)
                    .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_small_n_with_y_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{5, 7}}) {
              for (usize kc = 8; kc < 25; kc += 5) {
                AvgPoolMicrokernelTester()
                    .kr(8)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .yStride(31)
                    .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8avgpool_mp8x9p8q_sse2_small_n_with_s() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{5, 7}}) {
              for (usize s = 2; s <= 5; s++) {
                for (usize kc = 8; kc < 25; kc += 5) {
                  AvgPoolMicrokernelTester()
                      .kr(8)
                      .mr(9)
                      .qr(8)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .s(s)
                      .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
                }
              }
            }
          }

        */
    }
}
