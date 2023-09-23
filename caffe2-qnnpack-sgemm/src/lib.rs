#[macro_use] mod imports; use imports::*;

x!{sgemm_5x8_neon}
x!{sgemm_6x8_psimd}
x!{sgemm_6x8_neon}
x!{sgemm}