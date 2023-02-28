// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/isa-checks.h]

#[macro_export] macro_rules! test_requires_x86_sse2 {
    () => {
        /*
        
          do {                                                      
            if (!cpuinfo_initialize() || !cpuinfo_has_x86_sse2()) { 
              return;                                               
            }                                                       
          } while (0)
        */
    }
}

#[macro_export] macro_rules! test_requires_arm_neon {
    () => {
        /*
        
          do {                                                      
            if (!cpuinfo_initialize() || !cpuinfo_has_arm_neon()) { 
              return;                                               
            }                                                       
          } while (0)
        */
    }
}

#[macro_export] macro_rules! test_requires_arm_neon_fp16_arith {
    () => {
        /*
        
          do {                                                                 
            if (!cpuinfo_initialize() || !cpuinfo_has_arm_neon_fp16_arith()) { 
              return;                                                          
            }                                                                  
          } while (0)
        */
    }
}
