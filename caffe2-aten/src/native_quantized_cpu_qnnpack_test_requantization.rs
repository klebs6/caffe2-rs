// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/requantization.cc]

/**
  | Precise scalar implementation using
  | unsigned 32-bit arithmetics.
  |
  */
#[test] fn PRECISE__SCALAR_UNSIGNED32_exact_divide_by_po2() {
    todo!();
    /*
    
      for (u32 s = 1; s < 32; s++) {
        RequantizationTester().s(s).testExactDivideByPO2(
            pytorch_qnnp_requantize_precise__scalar_unsigned32);
      }

    */
}

#[test] fn PRECISE__SCALAR_UNSIGNED32_exact_divide_by_po2_with_zero_point() {
    todo!();
    /*
    
      for (i32 zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
              pytorch_qnnp_requantize_precise__scalar_unsigned32);
        }
      }

    */
}

#[test] fn PRECISE__SCALAR_UNSIGNED32_divide_by_po2_with_rounding_up() {
    todo!();
    /*
    
      for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester()
              .zeroPoint(zeroPoint)
              .s(s)
              .testDivideByPO2WithRoundingUp(
                  pytorch_qnnp_requantize_precise__scalar_unsigned32);
        }
      }

    */
}

#[test] fn PRECISE__SCALAR_UNSIGNED32_divide_by_po2_with_rounding_down() {
    todo!();
    /*
    
      for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester()
              .zeroPoint(zeroPoint)
              .s(s)
              .testDivideByPO2WithRoundingDown(
                  pytorch_qnnp_requantize_precise__scalar_unsigned32);
        }
      }

    */
}

#[test] fn PRECISE__SCALAR_UNSIGNED32_divide_by_po2_with_rounding_away() {
    todo!();
    /*
    
      for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester()
              .zeroPoint(zeroPoint)
              .s(s)
              .testDivideByPO2WithRoundingAway(
                  pytorch_qnnp_requantize_precise__scalar_unsigned32);
        }
      }

    */
}

#[test] fn PRECISE__SCALAR_UNSIGNED32_special_cases() {
    todo!();
    /*
    
      RequantizationTester().testSpecialCases(
          pytorch_qnnp_requantize_precise__scalar_unsigned32);

    */
}

#[test] fn PRECISE__SCALAR_UNSIGNED32_random_cases() {
    todo!();
    /*
    
      RequantizationTester().iterations(100).testRandomCasesPrecise(
          pytorch_qnnp_requantize_precise__scalar_unsigned32);

    */
}

/**
  | Precise scalar implementation using
  | unsigned 64-bit arithmetics.
  |
  */
#[test] fn PRECISE__SCALAR_UNSIGNED64_exact_divide_by_po2() {
    todo!();
    /*
    
      for (u32 s = 1; s < 32; s++) {
        RequantizationTester().s(s).testExactDivideByPO2(
            pytorch_qnnp_requantize_precise__scalar_unsigned64);
      }

    */
}

#[test] fn PRECISE__SCALAR_UNSIGNED64_exact_divide_by_po2_with_zero_point() {
    todo!();
    /*
    
      for (i32 zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
              pytorch_qnnp_requantize_precise__scalar_unsigned64);
        }
      }

    */
}

#[test] fn PRECISE__SCALAR_UNSIGNED64_divide_by_po2_with_rounding_up() {
    todo!();
    /*
    
      for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester()
              .zeroPoint(zeroPoint)
              .s(s)
              .testDivideByPO2WithRoundingUp(
                  pytorch_qnnp_requantize_precise__scalar_unsigned64);
        }
      }

    */
}

#[test] fn PRECISE__SCALAR_UNSIGNED64_divide_by_po2_with_rounding_down() {
    todo!();
    /*
    
      for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester()
              .zeroPoint(zeroPoint)
              .s(s)
              .testDivideByPO2WithRoundingDown(
                  pytorch_qnnp_requantize_precise__scalar_unsigned64);
        }
      }

    */
}

#[test] fn PRECISE__SCALAR_UNSIGNED64_divide_by_po2_with_rounding_away() {
    todo!();
    /*
    
      for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester()
              .zeroPoint(zeroPoint)
              .s(s)
              .testDivideByPO2WithRoundingAway(
                  pytorch_qnnp_requantize_precise__scalar_unsigned64);
        }
      }

    */
}

#[test] fn PRECISE__SCALAR_UNSIGNED64_special_cases() {
    todo!();
    /*
    
      RequantizationTester().testSpecialCases(
          pytorch_qnnp_requantize_precise__scalar_unsigned64);

    */
}

#[test] fn PRECISE__SCALAR_UNSIGNED64_random_cases() {
    todo!();
    /*
    
      RequantizationTester().iterations(100).testRandomCasesPrecise(
          pytorch_qnnp_requantize_precise__scalar_unsigned64);

    */
}

/*
 * Precise scalar implementation using signed 64-bit arithmetics.
 */

#[test] fn PRECISE__SCALAR_SIGNED64_exact_divide_by_po2() {
    todo!();
    /*
    
      for (u32 s = 1; s < 32; s++) {
        RequantizationTester().s(s).testExactDivideByPO2(
            pytorch_qnnp_requantize_precise__scalar_signed64);
      }

    */
}

#[test] fn PRECISE__SCALAR_SIGNED64_exact_divide_by_po2_with_zero_point() {
    todo!();
    /*
    
      for (i32 zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
              pytorch_qnnp_requantize_precise__scalar_signed64);
        }
      }

    */
}

#[test] fn PRECISE__SCALAR_SIGNED64_divide_by_po2_with_rounding_up() {
    todo!();
    /*
    
      for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester()
              .zeroPoint(zeroPoint)
              .s(s)
              .testDivideByPO2WithRoundingUp(
                  pytorch_qnnp_requantize_precise__scalar_signed64);
        }
      }

    */
}

#[test] fn PRECISE__SCALAR_SIGNED64_divide_by_po2_with_rounding_down() {
    todo!();
    /*
    
      for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester()
              .zeroPoint(zeroPoint)
              .s(s)
              .testDivideByPO2WithRoundingDown(
                  pytorch_qnnp_requantize_precise__scalar_signed64);
        }
      }

    */
}

#[test] fn PRECISE__SCALAR_SIGNED64_divide_by_po2_with_rounding_away() {
    todo!();
    /*
    
      for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester()
              .zeroPoint(zeroPoint)
              .s(s)
              .testDivideByPO2WithRoundingAway(
                  pytorch_qnnp_requantize_precise__scalar_signed64);
        }
      }

    */
}

#[test] fn PRECISE__SCALAR_SIGNED64_special_cases() {
    todo!();
    /*
    
      RequantizationTester().testSpecialCases(
          pytorch_qnnp_requantize_precise__scalar_signed64);

    */
}

#[test] fn PRECISE__SCALAR_SIGNED64_random_cases() {
    todo!();
    /*
    
      RequantizationTester().iterations(100).testRandomCasesPrecise(
          pytorch_qnnp_requantize_precise__scalar_signed64);

    */
}

/**
  | FP32-based scalar implementation
  | using lrintf function.
  |
  */
#[test] fn FP32__SCALAR_LRINTF_random_cases() {
    todo!();
    /*
    
      RequantizationTester().iterations(1000).testRandomCasesApproximate(
          pytorch_qnnp_requantize_fp32__scalar_lrintf);

    */
}

/**
  | FP32-based scalar implementation
  | using magic trick for FP32->INT32 conversion.
  |
  */
#[test] fn FP32__SCALAR_MAGIC_random_cases() {
    todo!();
    /*
    
      RequantizationTester().iterations(1000).testRandomCasesApproximate(
          pytorch_qnnp_requantize_fp32__scalar_magic);

    */
}

/**
  | Q31-based scalar implementation.
  |
  */
#[test] fn Q31__SCALAR_exact_divide_by_po2() {
    todo!();
    /*
    
      for (u32 s = 1; s < 32; s++) {
        RequantizationTester().s(s).testExactDivideByPO2(
            pytorch_qnnp_requantize_q31__scalar);
      }

    */
}

#[test] fn Q31__SCALAR_exact_divide_by_po2_with_zero_point() {
    todo!();
    /*
    
      for (i32 zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
              pytorch_qnnp_requantize_q31__scalar);
        }
      }

    */
}

#[test] fn Q31__SCALAR_divide_by_po2_with_rounding_up() {
    todo!();
    /*
    
      for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester()
              .zeroPoint(zeroPoint)
              .s(s)
              .testDivideByPO2WithRoundingUp(pytorch_qnnp_requantize_q31__scalar);
        }
      }

    */
}

/**
  | No rounding down test - it fails because
  | of upward bias in multiplication
  |
  */
#[test] fn Q31__SCALAR_divide_by_po2_with_rounding_away() {
    todo!();
    /*
    
      for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester()
              .zeroPoint(zeroPoint)
              .s(s)
              .testDivideByPO2WithRoundingAway(pytorch_qnnp_requantize_q31__scalar);
        }
      }

    */
}

#[test] fn Q31__SCALAR_special_cases() {
    todo!();
    /*
    
      RequantizationTester().testSpecialCases(pytorch_qnnp_requantize_q31__scalar);

    */
}

#[test] fn Q31__SCALAR_random_cases() {
    todo!();
    /*
    
      RequantizationTester().iterations(100).testRandomCasesApproximate(
          pytorch_qnnp_requantize_q31__scalar);

    */
}

#[test] fn Q31__SCALAR_random_match_gemmlowp() {
    todo!();
    /*
    
      RequantizationTester().iterations(100).testRandomCasesAgainstReference(
          pytorch_qnnp_requantize_q31__scalar,
          pytorch_qnnp_requantize_gemmlowp__scalar);

    */
}

/**
  | Scalar implementation from gemmlowp.
  |
  */
#[test] fn GEMMLOWP__SCALAR_random_cases() {
    todo!();
    /*
    
      RequantizationTester().iterations(100).testRandomCasesApproximate(
          pytorch_qnnp_requantize_gemmlowp__scalar);

    */
}

/**
  | Precise PSIMD implementation using
  | unsigned 32-bit arithmetics.
  |
  */
#[test] fn PRECISE__PSIMD_exact_divide_by_po2() {
    todo!();
    /*
    
      for (u32 s = 1; s < 32; s++) {
        RequantizationTester().s(s).testExactDivideByPO2(
            pytorch_qnnp_requantize_precise__psimd);
      }

    */
}

#[test] fn PRECISE__PSIMD_exact_divide_by_po2_with_zero_point() {
    todo!();
    /*
    
      for (i32 zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
              pytorch_qnnp_requantize_precise__psimd);
        }
      }

    */
}

#[test] fn PRECISE__PSIMD_divide_by_po2_with_rounding_up() {
    todo!();
    /*
    
      for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester()
              .zeroPoint(zeroPoint)
              .s(s)
              .testDivideByPO2WithRoundingUp(
                  pytorch_qnnp_requantize_precise__psimd);
        }
      }

    */
}

#[test] fn PRECISE__PSIMD_divide_by_po2_with_rounding_down() {
    todo!();
    /*
    
      for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester()
              .zeroPoint(zeroPoint)
              .s(s)
              .testDivideByPO2WithRoundingDown(
                  pytorch_qnnp_requantize_precise__psimd);
        }
      }

    */
}

#[test] fn PRECISE__PSIMD_divide_by_po2_with_rounding_away() {
    todo!();
    /*
    
      for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
        for (u32 s = 1; s < 32; s++) {
          RequantizationTester()
              .zeroPoint(zeroPoint)
              .s(s)
              .testDivideByPO2WithRoundingAway(
                  pytorch_qnnp_requantize_precise__psimd);
        }
      }

    */
}

#[test] fn PRECISE__PSIMD_special_cases() {
    todo!();
    /*
    
      RequantizationTester().testSpecialCases(
          pytorch_qnnp_requantize_precise__psimd);

    */
}

#[test] fn PRECISE__PSIMD_random_cases() {
    todo!();
    /*
    
      RequantizationTester().iterations(100).testRandomCasesPrecise(
          pytorch_qnnp_requantize_precise__psimd);

    */
}

/**
  | FP32-based PSIMD implementation using
  | magic trick for FP32->INT32 conversion.
  |
  */
#[test] fn FP32__PSIMD_random_cases() {
    todo!();
    /*
    
      RequantizationTester().iterations(1000).testRandomCasesApproximate(
          pytorch_qnnp_requantize_fp32__psimd);

    */
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {

    use super::*;

    /**
      | Precise SSE2 implementation using
      | floating-point shuffle.
      |
      */
    #[test] fn PRECISE__SSE2_exact_divide_by_po2() {
        todo!();
        /*
        
          for (u32 s = 1; s < 32; s++) {
            RequantizationTester().s(s).testExactDivideByPO2(
                pytorch_qnnp_requantize_precise__sse2);
          }

        */
    }

    #[test] fn PRECISE__SSE2_exact_divide_by_po2_with_zero_point() {
        todo!();
        /*
        
          for (i32 zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
                  pytorch_qnnp_requantize_precise__sse2);
            }
          }

        */
    }

    #[test] fn PRECISE__SSE2_divide_by_po2_with_rounding_up() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingUp(pytorch_qnnp_requantize_precise__sse2);
            }
          }

        */
    }

    #[test] fn PRECISE__SSE2_divide_by_po2_with_rounding_down() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingDown(
                      pytorch_qnnp_requantize_precise__sse2);
            }
          }

        */
    }

    #[test] fn PRECISE__SSE2_divide_by_po2_with_rounding_away() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingAway(
                      pytorch_qnnp_requantize_precise__sse2);
            }
          }

        */
    }

    #[test] fn PRECISE__SSE2_special_cases() {
        todo!();
        /*
        
          RequantizationTester().testSpecialCases(
              pytorch_qnnp_requantize_precise__sse2);

        */
    }

    #[test] fn PRECISE__SSE2_random_cases() {
        todo!();
        /*
        
          RequantizationTester().iterations(100).testRandomCasesPrecise(
              pytorch_qnnp_requantize_precise__sse2);

        */
    }

    /**
      | Precise SSSE3 implementation using
      | floating-point shuffle.
      |
      */
    #[test] fn PRECISE__SSSE3_exact_divide_by_po2() {
        todo!();
        /*
        
          for (u32 s = 1; s < 32; s++) {
            RequantizationTester().s(s).testExactDivideByPO2(
                pytorch_qnnp_requantize_precise__ssse3);
          }

        */
    }

    #[test] fn PRECISE__SSSE3_exact_divide_by_po2_with_zero_point() {
        todo!();
        /*
        
          for (i32 zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
                  pytorch_qnnp_requantize_precise__ssse3);
            }
          }

        */
    }

    #[test] fn PRECISE__SSSE3_divide_by_po2_with_rounding_up() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingUp(
                      pytorch_qnnp_requantize_precise__ssse3);
            }
          }

        */
    }

    #[test] fn PRECISE__SSSE3_divide_by_po2_with_rounding_down() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingDown(
                      pytorch_qnnp_requantize_precise__ssse3);
            }
          }

        */
    }

    #[test] fn PRECISE__SSSE3_divide_by_po2_with_rounding_away() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingAway(
                      pytorch_qnnp_requantize_precise__ssse3);
            }
          }

        */
    }

    #[test] fn PRECISE__SSSE3_special_cases() {
        todo!();
        /*
        
          RequantizationTester().testSpecialCases(
              pytorch_qnnp_requantize_precise__ssse3);

        */
    }

    #[test] fn PRECISE__SSSE3_random_cases() {
        todo!();
        /*
        
          RequantizationTester().iterations(100).testRandomCasesPrecise(
              pytorch_qnnp_requantize_precise__ssse3);

        */
    }

    /**
      | Precise SSE4.1 implementation using
      | static blend instruction.
      |
      */
    #[test] fn PRECISE__SSE4_exact_divide_by_po2() {
        todo!();
        /*
        
          for (u32 s = 1; s < 32; s++) {
            RequantizationTester().s(s).testExactDivideByPO2(
                pytorch_qnnp_requantize_precise__sse4);
          }

        */
    }

    #[test] fn PRECISE__SSE4_exact_divide_by_po2_with_zero_point() {
        todo!();
        /*
        
          for (i32 zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
                  pytorch_qnnp_requantize_precise__sse4);
            }
          }

        */
    }

    #[test] fn PRECISE__SSE4_divide_by_po2_with_rounding_up() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingUp(pytorch_qnnp_requantize_precise__sse4);
            }
          }

        */
    }

    #[test] fn PRECISE__SSE4_divide_by_po2_with_rounding_down() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingDown(
                      pytorch_qnnp_requantize_precise__sse4);
            }
          }

        */
    }

    #[test] fn PRECISE__SSE4_divide_by_po2_with_rounding_away() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingAway(
                      pytorch_qnnp_requantize_precise__sse4);
            }
          }

        */
    }

    #[test] fn PRECISE__SSE4_special_cases() {
        todo!();
        /*
        
          RequantizationTester().testSpecialCases(
              pytorch_qnnp_requantize_precise__sse4);

        */
    }

    #[test] fn PRECISE__SSE4_random_cases() {
        todo!();
        /*
        
          RequantizationTester().iterations(100).testRandomCasesPrecise(
              pytorch_qnnp_requantize_precise__sse4);

        */
    }

    /**
      | FP32-based x86 SSE2 implementation.
      |
      */
    #[test] fn FP32__SSE2_random_cases() {
        todo!();
        /*
        
          RequantizationTester().iterations(1000).testRandomCasesApproximate(
              pytorch_qnnp_requantize_fp32__sse2);

        */
    }

    /**
      | Q31-based x86 SSE2 implementation.
      |
      */
    #[test] fn Q31__SSE2_exact_divide_by_po2() {
        todo!();
        /*
        
          for (u32 s = 1; s < 32; s++) {
            RequantizationTester().s(s).testExactDivideByPO2(
                pytorch_qnnp_requantize_q31__sse2);
          }

        */
    }

    #[test] fn Q31__SSE2_exact_divide_by_po2_with_zero_point() {
        todo!();
        /*
        
          for (i32 zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
                  pytorch_qnnp_requantize_q31__sse2);
            }
          }

        */
    }

    #[test] fn Q31__SSE2_divide_by_po2_with_rounding_up() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingUp(pytorch_qnnp_requantize_q31__sse2);
            }
          }

        */
    }

    /**
      | No rounding down test - it fails because
      | of upward bias in multiplication
      |
      */
    #[test] fn Q31__SSE2_divide_by_po2_with_rounding_away() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingAway(pytorch_qnnp_requantize_q31__sse2);
            }
          }

        */
    }

    #[test] fn Q31__SSE2_special_cases() {
        todo!();
        /*
        
          RequantizationTester().testSpecialCases(pytorch_qnnp_requantize_q31__sse2);

        */
    }

    #[test] fn Q31__SSE2_random_cases() {
        todo!();
        /*
        
          RequantizationTester().iterations(100).testRandomCasesApproximate(
              pytorch_qnnp_requantize_q31__sse2);

        */
    }

    #[test] fn Q31__SSE2_random_match_gemmlowp() {
        todo!();
        /*
        
          RequantizationTester().iterations(100).testRandomCasesAgainstReference(
              pytorch_qnnp_requantize_q31__sse2,
              pytorch_qnnp_requantize_gemmlowp__sse2);

        */
    }

    /**
      | Q31-based x86 SSSE3 implementation.
      |
      */
    #[test] fn Q31__SSSE3_exact_divide_by_po2() {
        todo!();
        /*
        
          for (u32 s = 1; s < 32; s++) {
            RequantizationTester().s(s).testExactDivideByPO2(
                pytorch_qnnp_requantize_q31__ssse3);
          }

        */
    }

    #[test] fn Q31__SSSE3_exact_divide_by_po2_with_zero_point() {
        todo!();
        /*
        
          for (i32 zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
                  pytorch_qnnp_requantize_q31__ssse3);
            }
          }

        */
    }

    #[test] fn Q31__SSSE3_divide_by_po2_with_rounding_up() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingUp(pytorch_qnnp_requantize_q31__ssse3);
            }
          }

        */
    }

    /**
      | No rounding down test - it fails because
      | of upward bias in multiplication
      |
      */
    #[test] fn Q31__SSSE3_divide_by_po2_with_rounding_away() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingAway(pytorch_qnnp_requantize_q31__ssse3);
            }
          }

        */
    }

    #[test] fn Q31__SSSE3_special_cases() {
        todo!();
        /*
        
          RequantizationTester().testSpecialCases(pytorch_qnnp_requantize_q31__ssse3);

        */
    }

    #[test] fn Q31__SSSE3_random_cases() {
        todo!();
        /*
        
          RequantizationTester().iterations(100).testRandomCasesApproximate(
              pytorch_qnnp_requantize_q31__ssse3);

        */
    }

    #[test] fn Q31__SSSE3_random_match_gemmlowp() {
        todo!();
        /*
        
          RequantizationTester().iterations(100).testRandomCasesAgainstReference(
              pytorch_qnnp_requantize_q31__ssse3,
              pytorch_qnnp_requantize_gemmlowp__ssse3);

        */
    }

    /**
      | Q31-based x86 SSE4 implementation.
      |
      */
    #[test] fn Q31__SSE4_exact_divide_by_po2() {
        todo!();
        /*
        
          for (u32 s = 1; s < 32; s++) {
            RequantizationTester().s(s).testExactDivideByPO2(
                pytorch_qnnp_requantize_q31__sse4);
          }

        */
    }

    #[test] fn Q31__SSE4_exact_divide_by_po2_with_zero_point() {
        todo!();
        /*
        
          for (i32 zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
                  pytorch_qnnp_requantize_q31__sse4);
            }
          }

        */
    }

    #[test] fn Q31__SSE4_divide_by_po2_with_rounding_up() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingUp(pytorch_qnnp_requantize_q31__sse4);
            }
          }

        */
    }

    /**
      | No rounding down test - it fails because
      | of upward bias in multiplication
      |
      */
    #[test] fn Q31__SSE4_divide_by_po2_with_rounding_away() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingAway(pytorch_qnnp_requantize_q31__sse4);
            }
          }

        */
    }

    #[test] fn Q31__SSE4_special_cases() {
        todo!();
        /*
        
          RequantizationTester().testSpecialCases(pytorch_qnnp_requantize_q31__sse4);

        */
    }

    #[test] fn Q31__SSE4_random_cases() {
        todo!();
        /*
        
          RequantizationTester().iterations(100).testRandomCasesApproximate(
              pytorch_qnnp_requantize_q31__sse4);

        */
    }

    #[test] fn Q31__SSE4_random_match_gemmlowp() {
        todo!();
        /*
        
          RequantizationTester().iterations(100).testRandomCasesAgainstReference(
              pytorch_qnnp_requantize_q31__sse4,
              pytorch_qnnp_requantize_gemmlowp__sse4);

        */
    }

    /**
      | x86 SSE2 implementation from gemmlowp.
      |
      */
    #[test] fn GEMMLOWP__SSE2_exact_divide_by_po2() {
        todo!();
        /*
        
          for (u32 s = 1; s < 32; s++) {
            RequantizationTester().s(s).testExactDivideByPO2(
                pytorch_qnnp_requantize_gemmlowp__sse2);
          }

        */
    }

    #[test] fn GEMMLOWP__SSE2_exact_divide_by_po2_with_zero_point() {
        todo!();
        /*
        
          for (i32 zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
                  pytorch_qnnp_requantize_gemmlowp__sse2);
            }
          }

        */
    }

    #[test] fn GEMMLOWP__SSE2_divide_by_po2_with_rounding_up() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingUp(
                      pytorch_qnnp_requantize_gemmlowp__sse2);
            }
          }

        */
    }

    /**
      | No rounding down test - it fails because
      | of upward bias in multiplication
      |
      */
    #[test] fn GEMMLOWP__SSE2_divide_by_po2_with_rounding_away() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingAway(
                      pytorch_qnnp_requantize_gemmlowp__sse2);
            }
          }

        */
    }

    #[test] fn GEMMLOWP__SSE2_special_cases() {
        todo!();
        /*
        
          RequantizationTester().testSpecialCases(
              pytorch_qnnp_requantize_gemmlowp__sse2);

        */
    }

    #[test] fn GEMMLOWP__SSE2_random_cases() {
        todo!();
        /*
        
          RequantizationTester().iterations(100).testRandomCasesApproximate(
              pytorch_qnnp_requantize_gemmlowp__sse2);

        */
    }

    /**
      | x86 SSSE3 implementation from gemmlowp.
      |
      */
    #[test] fn GEMMLOWP__SSSE3_exact_divide_by_po2() {
        todo!();
        /*
        
          for (u32 s = 1; s < 32; s++) {
            RequantizationTester().s(s).testExactDivideByPO2(
                pytorch_qnnp_requantize_gemmlowp__ssse3);
          }

        */
    }

    #[test] fn GEMMLOWP__SSSE3_exact_divide_by_po2_with_zero_point() {
        todo!();
        /*
        
          for (i32 zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
                  pytorch_qnnp_requantize_gemmlowp__ssse3);
            }
          }

        */
    }

    #[test] fn GEMMLOWP__SSSE3_divide_by_po2_with_rounding_up() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingUp(
                      pytorch_qnnp_requantize_gemmlowp__ssse3);
            }
          }

        */
    }

    /**
      | No rounding down test - it fails because
      | of upward bias in multiplication
      |
      */
    #[test] fn GEMMLOWP__SSSE3_divide_by_po2_with_rounding_away() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingAway(
                      pytorch_qnnp_requantize_gemmlowp__ssse3);
            }
          }

        */
    }

    #[test] fn GEMMLOWP__SSSE3_special_cases() {
        todo!();
        /*
        
          RequantizationTester().testSpecialCases(
              pytorch_qnnp_requantize_gemmlowp__ssse3);

        */
    }

    #[test] fn GEMMLOWP__SSSE3_random_cases() {
        todo!();
        /*
        
          RequantizationTester().iterations(100).testRandomCasesApproximate(
              pytorch_qnnp_requantize_gemmlowp__ssse3);

        */
    }

    /**
      | x86 SSE4 implementation from gemmlowp.
      |
      */
    #[test] fn GEMMLOWP__SSE4_exact_divide_by_po2() {
        todo!();
        /*
        
          for (u32 s = 1; s < 32; s++) {
            RequantizationTester().s(s).testExactDivideByPO2(
                pytorch_qnnp_requantize_gemmlowp__sse4);
          }

        */
    }

    #[test] fn GEMMLOWP__SSE4_exact_divide_by_po2_with_zero_point() {
        todo!();
        /*
        
          for (i32 zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
                  pytorch_qnnp_requantize_gemmlowp__sse4);
            }
          }

        */
    }

    #[test] fn GEMMLOWP__SSE4_divide_by_po2_with_rounding_up() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingUp(
                      pytorch_qnnp_requantize_gemmlowp__sse4);
            }
          }

        */
    }

    /**
      | No rounding down test - it fails because
      | of upward bias in multiplication
      |
      */
    #[test] fn GEMMLOWP__SSE4_divide_by_po2_with_rounding_away() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingAway(
                      pytorch_qnnp_requantize_gemmlowp__sse4);
            }
          }

        */
    }

    #[test] fn GEMMLOWP__SSE4_special_cases() {
        todo!();
        /*
        
          RequantizationTester().testSpecialCases(
              pytorch_qnnp_requantize_gemmlowp__sse4);

        */
    }

    #[test] fn GEMMLOWP__SSE4_random_cases() {
        todo!();
        /*
        
          RequantizationTester().iterations(100).testRandomCasesApproximate(
              pytorch_qnnp_requantize_gemmlowp__sse4);

        */
    }
}

#[cfg(test)]
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm {

    use super::*;

    /**
      | Precise ARM NEON implementation.
      |
      */
    #[test] fn PRECISE__NEON_exact_divide_by_po2() {
        todo!();
        /*
        
          for (u32 s = 1; s < 32; s++) {
            RequantizationTester().s(s).testExactDivideByPO2(
                pytorch_qnnp_requantize_precise__neon);
          }

        */
    }

    #[test] fn PRECISE__NEON_exact_divide_by_po2_with_zero_point() {
        todo!();
        /*
        
          for (i32 zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
                  pytorch_qnnp_requantize_precise__neon);
            }
          }

        */
    }

    #[test] fn PRECISE__NEON_divide_by_po2_with_rounding_up() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingUp(pytorch_qnnp_requantize_precise__neon);
            }
          }

        */
    }

    #[test] fn PRECISE__NEON_divide_by_po2_with_rounding_down() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingDown(
                      pytorch_qnnp_requantize_precise__neon);
            }
          }

        */
    }

    #[test] fn PRECISE__NEON_divide_by_po2_with_rounding_away() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingAway(
                      pytorch_qnnp_requantize_precise__neon);
            }
          }

        */
    }

    #[test] fn PRECISE__NEON_special_cases() {
        todo!();
        /*
        
          RequantizationTester().testSpecialCases(
              pytorch_qnnp_requantize_precise__neon);

        */
    }

    #[test] fn PRECISE__NEON_random_cases() {
        todo!();
        /*
        
          RequantizationTester().iterations(100).testRandomCasesPrecise(
              pytorch_qnnp_requantize_precise__neon);

        */
    }

    /**
      | FP32-based ARM NEON implementation.
      |
      */
    #[test] fn FP32__NEON_random_cases() {
        todo!();
        /*
        
          RequantizationTester().iterations(1000).testRandomCasesApproximate(
              pytorch_qnnp_requantize_fp32__neon);

        */
    }

    /**
      | Q31-based ARM NEON implementation.
      |
      */
    #[test] fn Q31__NEON_exact_divide_by_po2() {
        todo!();
        /*
        
          for (u32 s = 1; s < 32; s++) {
            RequantizationTester().s(s).testExactDivideByPO2(
                pytorch_qnnp_requantize_q31__neon);
          }

        */
    }

    #[test] fn Q31__NEON_exact_divide_by_po2_with_zero_point() {
        todo!();
        /*
        
          for (i32 zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
                  pytorch_qnnp_requantize_q31__neon);
            }
          }

        */
    }

    #[test] fn Q31__NEON_divide_by_po2_with_rounding_up() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingUp(pytorch_qnnp_requantize_q31__neon);
            }
          }

        */
    }

    /**
      | No rounding down test - it fails because
      | of upward bias in multiplication
      |
      */
    #[test] fn Q31__NEON_divide_by_po2_with_rounding_away() {
        todo!();
        /*
        
          for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
            for (u32 s = 1; s < 32; s++) {
              RequantizationTester()
                  .zeroPoint(zeroPoint)
                  .s(s)
                  .testDivideByPO2WithRoundingAway(pytorch_qnnp_requantize_q31__neon);
            }
          }

        */
    }

    #[test] fn Q31__NEON_special_cases() {
        todo!();
        /*
        
          RequantizationTester().testSpecialCases(pytorch_qnnp_requantize_q31__neon);

        */
    }

    #[test] fn Q31__NEON_random_cases() {
        todo!();
        /*
        
          RequantizationTester().iterations(100).testRandomCasesApproximate(
              pytorch_qnnp_requantize_q31__neon);

        */
    }

    #[test] fn Q31__NEON_random_match_gemmlowp() {
        todo!();
        /*
        
          RequantizationTester().iterations(100).testRandomCasesAgainstReference(
              pytorch_qnnp_requantize_q31__neon,
              pytorch_qnnp_requantize_gemmlowp__neon);

        */
    }

    /**
      | ARM NEON implementation from gemmlowp.
      |
      */
    #[test] fn GEMMLOWP__NEON_random_cases() {
        todo!();
        /*
        
          RequantizationTester().iterations(100).testRandomCasesApproximate(
              pytorch_qnnp_requantize_gemmlowp__neon);

        */
    }
}
