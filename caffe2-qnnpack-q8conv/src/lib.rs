#[macro_use] mod imports; use imports::*;

x!{q8conv_4x4c2_sse2}
x!{q8conv_8x8_aarch64_neon}
x!{q8conv_8x8_neon}
x!{q8conv_4x8_neon}
x!{q8conv_4x8_aarch32_neon}
x!{q8conv}