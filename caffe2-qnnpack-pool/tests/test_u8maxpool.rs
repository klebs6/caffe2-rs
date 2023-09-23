crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/u8maxpool.cc]

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arch_arm {

    use super::*;

    #[test] fn u8maxpool_sub16_neon_kc_lt_16_mx1_pool() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize kc = 1; kc < 16; kc++) {
            for (usize ks = 2; ks < 16; ks++) {
              MaxPoolMicrokernelTester().kr(16).kh(ks).kw(1).kc(kc).test(
                  pytorch_u8maxpool_ukernel_sub16__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_neon_kc_lt_16_mx1_pool_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize kc = 1; kc < 16; kc++) {
            for (usize ks = 2; ks < 16; ks++) {
              MaxPoolMicrokernelTester().kr(16).kh(ks).kw(1).kc(kc).qmin(192).test(
                  pytorch_u8maxpool_ukernel_sub16__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_neon_kc_lt_16_mx1_pool_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize kc = 1; kc < 16; kc++) {
            for (usize ks = 2; ks < 16; ks++) {
              MaxPoolMicrokernelTester().kr(16).kh(ks).kw(1).kc(kc).qmax(192).test(
                  pytorch_u8maxpool_ukernel_sub16__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_neon_kc_lt_16_1xm_pool() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize kc = 1; kc < 16; kc++) {
            for (usize ks = 2; ks < 16; ks++) {
              MaxPoolMicrokernelTester().kr(16).kh(1).kw(ks).kc(kc).test(
                  pytorch_u8maxpool_ukernel_sub16__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_neon_kc_lt_16_1xm_pool_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize kc = 1; kc < 16; kc++) {
            for (usize ks = 2; ks < 16; ks++) {
              MaxPoolMicrokernelTester().kr(16).kh(1).kw(ks).kc(kc).qmin(192).test(
                  pytorch_u8maxpool_ukernel_sub16__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_neon_kc_lt_16_1xm_pool_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize kc = 1; kc < 16; kc++) {
            for (usize ks = 2; ks < 16; ks++) {
              MaxPoolMicrokernelTester().kr(16).kh(1).kw(ks).kc(kc).qmax(192).test(
                  pytorch_u8maxpool_ukernel_sub16__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_neon_kc_lt_16_small_n() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize kc = 1; kc < 16; kc++) {
                MaxPoolMicrokernelTester()
                    .kr(16)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .iterations(3)
                    .test(pytorch_u8maxpool_ukernel_sub16__neon);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_neon_kc_lt_16_small_n_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize kc = 1; kc < 16; kc++) {
                MaxPoolMicrokernelTester()
                    .kr(16)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .xStride(17)
                    .iterations(3)
                    .test(pytorch_u8maxpool_ukernel_sub16__neon);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_neon_kc_lt_16_small_n_with_s() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize s = 2; s <= 5; s++) {
                for (usize kc = 1; kc < 16; kc++) {
                  MaxPoolMicrokernelTester()
                      .kr(16)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .s(s)
                      .iterations(1)
                      .test(pytorch_u8maxpool_ukernel_sub16__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_neon_kc_lt_16_small_n_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize kc = 1; kc < 16; kc++) {
                MaxPoolMicrokernelTester()
                    .kr(16)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .qmin(192)
                    .iterations(3)
                    .test(pytorch_u8maxpool_ukernel_sub16__neon);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_neon_kc_lt_16_small_n_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize kc = 1; kc < 16; kc++) {
                MaxPoolMicrokernelTester()
                    .kr(16)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .qmax(192)
                    .iterations(3)
                    .test(pytorch_u8maxpool_ukernel_sub16__neon);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_eq_16_unipass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).kc(16);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                tester.kh(kh).kw(kw).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_eq_16_unipass_fulltile_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).kc(16);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                tester.kh(kh).kw(kw).qmin(192).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_eq_16_unipass_fulltile_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).kc(16);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                tester.kh(kh).kw(kw).qmax(192).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_eq_16_unipass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).kc(16);
          for (usize ks = 2; ks < tester.mr(); ks++) {
            tester.kh(ks).kw(1).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
            tester.kh(1).kw(ks).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_div_16_unipass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 16; kc < 256; kc += 48) {
                  tester.kh(kh).kw(kw).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_div_16_unipass_fulltile_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 16; kc < 256; kc += 48) {
                  tester.kh(kh).kw(kw).kc(kc).qmin(192).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_div_16_unipass_fulltile_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 16; kc < 256; kc += 48) {
                  tester.kh(kh).kw(kw).kc(kc).qmax(192).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_div_16_unipass_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).iterations(3);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 16; kc < 256; kc += 48) {
                  tester.kh(kh).kw(kw).kc(kc).xStride(257).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_div_16_unipass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).iterations(3);
          for (usize ks = 2; ks < tester.mr(); ks++) {
            for (usize kc = 16; kc < 256; kc += 48) {
              tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
              tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_gt_16_unipass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 17; kc < 32; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_gt_16_unipass_fulltile_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 17; kc < 32; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).qmin(192).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_gt_16_unipass_fulltile_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 17; kc < 32; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).qmax(192).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_gt_16_unipass_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).iterations(3);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 17; kc < 32; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).xStride(257).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_gt_16_unipass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).iterations(3);
          for (usize ks = 2; ks < tester.mr(); ks++) {
            for (usize kc = 17; kc < 32; kc++) {
              tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
              tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_eq_16_twopass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                tester.kh(kh).kw(kw).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_eq_16_twopass_fulltile_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                tester.kh(kh).kw(kw).qmin(192).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_eq_16_twopass_fulltile_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                tester.kh(kh).kw(kw).qmax(192).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_eq_16_twopass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
          for (usize ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
            tester.kh(ks).kw(1).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
            tester.kh(1).kw(ks).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_div_16_twopass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                for (usize kc = 16; kc < 256; kc += 48) {
                  tester.kh(kh).kw(kw).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_div_16_twopass_fulltile_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                for (usize kc = 16; kc < 256; kc += 48) {
                  tester.kh(kh).kw(kw).kc(kc).qmin(192).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_div_16_twopass_fulltile_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                for (usize kc = 16; kc < 256; kc += 48) {
                  tester.kh(kh).kw(kw).kc(kc).qmax(192).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_div_16_twopass_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                for (usize kc = 16; kc < 256; kc += 48) {
                  tester.kh(kh).kw(kw).kc(kc).xStride(257).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_div_16_twopass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
            for (usize kc = 16; kc < 256; kc += 48) {
              tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
              tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_gt_16_twopass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                for (usize kc = 17; kc < 32; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_gt_16_twopass_fulltile_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                for (usize kc = 17; kc < 32; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).qmin(192).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_gt_16_twopass_fulltile_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                for (usize kc = 17; kc < 32; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).qmax(192).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_gt_16_twopass_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                for (usize kc = 17; kc < 32; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).xStride(257).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__neon);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_gt_16_twopass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
            for (usize kc = 17; kc < 32; kc++) {
              tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
              tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_eq_16_multipass() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            tester.kh(ks).kw(1).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
            tester.kh(1).kw(ks).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_eq_16_multipass_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            tester.kh(ks).kw(1).qmin(192).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
            tester.kh(1).kw(ks).qmin(192).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_eq_16_multipass_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            tester.kh(ks).kw(1).qmax(192).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
            tester.kh(1).kw(ks).qmax(192).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_div_16_multipass() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            for (usize kc = 16; kc < 256; kc += 48) {
              tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
              tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_div_16_multipass_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            for (usize kc = 16; kc < 256; kc += 48) {
              tester.kh(ks).kw(1).kc(kc).qmin(192).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__neon);
              tester.kh(1).kw(ks).kc(kc).qmin(192).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_div_16_multipass_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            for (usize kc = 16; kc < 256; kc += 48) {
              tester.kh(ks).kw(1).kc(kc).qmax(192).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__neon);
              tester.kh(1).kw(ks).kc(kc).qmax(192).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_div_16_multipass_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            for (usize kc = 16; kc < 256; kc += 48) {
              tester.kh(ks).kw(1).kc(kc).xStride(257).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__neon);
              tester.kh(1).kw(ks).kc(kc).xStride(257).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_gt_16_multipass() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            for (usize kc = 17; kc < 32; kc++) {
              tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
              tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_gt_16_multipass_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            for (usize kc = 17; kc < 32; kc++) {
              tester.kh(ks).kw(1).kc(kc).qmin(192).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__neon);
              tester.kh(1).kw(ks).kc(kc).qmin(192).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_gt_16_multipass_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            for (usize kc = 17; kc < 32; kc++) {
              tester.kh(ks).kw(1).kc(kc).qmax(192).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__neon);
              tester.kh(1).kw(ks).kc(kc).qmax(192).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_kc_gt_16_multipass_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            for (usize kc = 17; kc < 32; kc++) {
              tester.kh(ks).kw(1).kc(kc).xStride(257).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__neon);
              tester.kh(1).kw(ks).kc(kc).xStride(257).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__neon);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_small_n() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5, 10}}) {
              for (usize kc = 16; kc < 51; kc += 5) {
                MaxPoolMicrokernelTester()
                    .kr(16)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .iterations(3)
                    .test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_small_n_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5, 10}}) {
              for (usize kc = 16; kc < 51; kc += 5) {
                MaxPoolMicrokernelTester()
                    .kr(16)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .xStride(101)
                    .iterations(1)
                    .test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_small_n_with_y_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5, 10}}) {
              for (usize kc = 16; kc < 51; kc += 5) {
                MaxPoolMicrokernelTester()
                    .kr(16)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .yStride(103)
                    .iterations(1)
                    .test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_neon_small_n_with_s() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize kc = 16; kc < 51; kc += 5) {
                for (usize s = 2; s <= ks; s++) {
                  MaxPoolMicrokernelTester()
                      .kr(16)
                      .mr(9)
                      .qr(8)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .s(s)
                      .iterations(1)
                      .test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
                }
              }
            }
          }

        */
    }

}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod arch_x86 {

    use super::*;

    #[test] fn u8maxpool_sub16_sse2_kc_lt_16_mx1_pool() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize kc = 1; kc < 16; kc++) {
            for (usize ks = 2; ks < 16; ks++) {
              MaxPoolMicrokernelTester().kr(16).kh(ks).kw(1).kc(kc).test(
                  pytorch_u8maxpool_ukernel_sub16__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_sse2_kc_lt_16_mx1_pool_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize kc = 1; kc < 16; kc++) {
            for (usize ks = 2; ks < 16; ks++) {
              MaxPoolMicrokernelTester().kr(16).kh(ks).kw(1).kc(kc).qmin(192).test(
                  pytorch_u8maxpool_ukernel_sub16__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_sse2_kc_lt_16_mx1_pool_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize kc = 1; kc < 16; kc++) {
            for (usize ks = 2; ks < 16; ks++) {
              MaxPoolMicrokernelTester().kr(16).kh(ks).kw(1).kc(kc).qmax(192).test(
                  pytorch_u8maxpool_ukernel_sub16__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_sse2_kc_lt_16_1xm_pool() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize kc = 1; kc < 16; kc++) {
            for (usize ks = 2; ks < 16; ks++) {
              MaxPoolMicrokernelTester().kr(16).kh(1).kw(ks).kc(kc).test(
                  pytorch_u8maxpool_ukernel_sub16__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_sse2_kc_lt_16_1xm_pool_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize kc = 1; kc < 16; kc++) {
            for (usize ks = 2; ks < 16; ks++) {
              MaxPoolMicrokernelTester().kr(16).kh(1).kw(ks).kc(kc).qmin(192).test(
                  pytorch_u8maxpool_ukernel_sub16__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_sse2_kc_lt_16_1xm_pool_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize kc = 1; kc < 16; kc++) {
            for (usize ks = 2; ks < 16; ks++) {
              MaxPoolMicrokernelTester().kr(16).kh(1).kw(ks).kc(kc).qmax(192).test(
                  pytorch_u8maxpool_ukernel_sub16__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_sse2_kc_lt_16_small_n() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize kc = 1; kc < 16; kc++) {
                MaxPoolMicrokernelTester()
                    .kr(16)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .iterations(3)
                    .test(pytorch_u8maxpool_ukernel_sub16__sse2);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_sse2_kc_lt_16_small_n_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize kc = 1; kc < 16; kc++) {
                MaxPoolMicrokernelTester()
                    .kr(16)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .xStride(17)
                    .iterations(3)
                    .test(pytorch_u8maxpool_ukernel_sub16__sse2);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_sse2_kc_lt_16_small_n_with_s() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize s = 2; s <= 5; s++) {
                for (usize kc = 1; kc < 16; kc++) {
                  MaxPoolMicrokernelTester()
                      .kr(16)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .s(s)
                      .iterations(1)
                      .test(pytorch_u8maxpool_ukernel_sub16__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_sse2_kc_lt_16_small_n_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize kc = 1; kc < 16; kc++) {
                MaxPoolMicrokernelTester()
                    .kr(16)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .qmin(192)
                    .iterations(3)
                    .test(pytorch_u8maxpool_ukernel_sub16__sse2);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_sub16_sse2_kc_lt_16_small_n_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize kc = 1; kc < 16; kc++) {
                MaxPoolMicrokernelTester()
                    .kr(16)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .qmax(192)
                    .iterations(3)
                    .test(pytorch_u8maxpool_ukernel_sub16__sse2);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_eq_16_unipass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).kc(16);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                tester.kh(kh).kw(kw).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_eq_16_unipass_fulltile_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).kc(16);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                tester.kh(kh).kw(kw).qmin(192).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_eq_16_unipass_fulltile_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).kc(16);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                tester.kh(kh).kw(kw).qmax(192).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_eq_16_unipass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).kc(16);
          for (usize ks = 2; ks < tester.mr(); ks++) {
            tester.kh(ks).kw(1).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
            tester.kh(1).kw(ks).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_div_16_unipass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 16; kc < 256; kc += 48) {
                  tester.kh(kh).kw(kw).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_div_16_unipass_fulltile_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 16; kc < 256; kc += 48) {
                  tester.kh(kh).kw(kw).kc(kc).qmin(192).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_div_16_unipass_fulltile_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 16; kc < 256; kc += 48) {
                  tester.kh(kh).kw(kw).kc(kc).qmax(192).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_div_16_unipass_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).iterations(3);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 16; kc < 256; kc += 48) {
                  tester.kh(kh).kw(kw).kc(kc).xStride(257).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_div_16_unipass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).iterations(3);
          for (usize ks = 2; ks < tester.mr(); ks++) {
            for (usize kc = 16; kc < 256; kc += 48) {
              tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_gt_16_unipass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 17; kc < 32; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_gt_16_unipass_fulltile_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 17; kc < 32; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).qmin(192).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_gt_16_unipass_fulltile_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 17; kc < 32; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).qmax(192).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_gt_16_unipass_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).iterations(3);
          for (usize kh = 1; kh <= tester.mr(); kh++) {
            for (usize kw = 1; kw <= tester.mr(); kw++) {
              if (kh * kw == tester.mr()) {
                for (usize kc = 17; kc < 32; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).xStride(257).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_gt_16_unipass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).iterations(3);
          for (usize ks = 2; ks < tester.mr(); ks++) {
            for (usize kc = 17; kc < 32; kc++) {
              tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_eq_16_twopass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                tester.kh(kh).kw(kw).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_eq_16_twopass_fulltile_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                tester.kh(kh).kw(kw).qmin(192).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_eq_16_twopass_fulltile_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                tester.kh(kh).kw(kw).qmax(192).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_eq_16_twopass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
          for (usize ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
            tester.kh(ks).kw(1).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
            tester.kh(1).kw(ks).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_div_16_twopass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                for (usize kc = 16; kc < 256; kc += 48) {
                  tester.kh(kh).kw(kw).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_div_16_twopass_fulltile_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                for (usize kc = 16; kc < 256; kc += 48) {
                  tester.kh(kh).kw(kw).kc(kc).qmin(192).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_div_16_twopass_fulltile_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                for (usize kc = 16; kc < 256; kc += 48) {
                  tester.kh(kh).kw(kw).kc(kc).qmax(192).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_div_16_twopass_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                for (usize kc = 16; kc < 256; kc += 48) {
                  tester.kh(kh).kw(kw).kc(kc).xStride(257).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_div_16_twopass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
            for (usize kc = 16; kc < 256; kc += 48) {
              tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_gt_16_twopass_fulltile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                for (usize kc = 17; kc < 32; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_gt_16_twopass_fulltile_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                for (usize kc = 17; kc < 32; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).qmin(192).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_gt_16_twopass_fulltile_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                for (usize kc = 17; kc < 32; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).qmax(192).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_gt_16_twopass_fulltile_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
            for (usize kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
              if (kh * kw == tester.mr() + tester.qr()) {
                for (usize kc = 17; kc < 32; kc++) {
                  tester.kh(kh).kw(kw).kc(kc).xStride(257).test(
                      pytorch_u8maxpool_ukernel_16x9p8q__sse2);
                }
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_gt_16_twopass_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
            for (usize kc = 17; kc < 32; kc++) {
              tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_eq_16_multipass() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            tester.kh(ks).kw(1).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
            tester.kh(1).kw(ks).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_eq_16_multipass_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            tester.kh(ks).kw(1).qmin(192).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
            tester.kh(1).kw(ks).qmin(192).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_eq_16_multipass_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            tester.kh(ks).kw(1).qmax(192).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
            tester.kh(1).kw(ks).qmax(192).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_div_16_multipass() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            for (usize kc = 16; kc < 256; kc += 48) {
              tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_div_16_multipass_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            for (usize kc = 16; kc < 256; kc += 48) {
              tester.kh(ks).kw(1).kc(kc).qmin(192).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              tester.kh(1).kw(ks).kc(kc).qmin(192).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_div_16_multipass_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            for (usize kc = 16; kc < 256; kc += 48) {
              tester.kh(ks).kw(1).kc(kc).qmax(192).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              tester.kh(1).kw(ks).kc(kc).qmax(192).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_div_16_multipass_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            for (usize kc = 16; kc < 256; kc += 48) {
              tester.kh(ks).kw(1).kc(kc).xStride(257).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              tester.kh(1).kw(ks).kc(kc).xStride(257).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_gt_16_multipass() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            for (usize kc = 17; kc < 32; kc++) {
              tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_gt_16_multipass_with_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            for (usize kc = 17; kc < 32; kc++) {
              tester.kh(ks).kw(1).kc(kc).qmin(192).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              tester.kh(1).kw(ks).kc(kc).qmin(192).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_gt_16_multipass_with_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            for (usize kc = 17; kc < 32; kc++) {
              tester.kh(ks).kw(1).kc(kc).qmax(192).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              tester.kh(1).kw(ks).kc(kc).qmax(192).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_kc_gt_16_multipass_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
          for (usize ks = tester.mr() + tester.qr() + 1;
               ks < tester.mr() + 3 * tester.qr();
               ks += 3) {
            for (usize kc = 17; kc < 32; kc++) {
              tester.kh(ks).kw(1).kc(kc).xStride(257).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              tester.kh(1).kw(ks).kc(kc).xStride(257).test(
                  pytorch_u8maxpool_ukernel_16x9p8q__sse2);
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_small_n() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5, 10}}) {
              for (usize kc = 16; kc < 51; kc += 5) {
                MaxPoolMicrokernelTester()
                    .kr(16)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .iterations(3)
                    .test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_small_n_with_x_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5, 10}}) {
              for (usize kc = 16; kc < 51; kc += 5) {
                MaxPoolMicrokernelTester()
                    .kr(16)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .xStride(101)
                    .iterations(1)
                    .test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_small_n_with_y_stride() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5, 10}}) {
              for (usize kc = 16; kc < 51; kc += 5) {
                MaxPoolMicrokernelTester()
                    .kr(16)
                    .mr(9)
                    .qr(8)
                    .n(n)
                    .kh(ks)
                    .kw(ks)
                    .kc(kc)
                    .yStride(103)
                    .iterations(1)
                    .test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
              }
            }
          }

        */
    }

    #[test] fn u8maxpool_16x9p8q_sse2_small_n_with_s() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 2; n < 5; n++) {
            for (usize ks : vector<usize>{{2, 3, 5}}) {
              for (usize kc = 16; kc < 51; kc += 5) {
                for (usize s = 2; s <= ks; s++) {
                  MaxPoolMicrokernelTester()
                      .kr(16)
                      .mr(9)
                      .qr(8)
                      .n(n)
                      .kh(ks)
                      .kw(ks)
                      .kc(kc)
                      .s(s)
                      .iterations(1)
                      .test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
                }
              }
            }
          }

        */
    }
}
