/*!
  | Flush-To-Zero and Denormals-Are-Zero mode
  |
  | Flush-To-Zero (FTZ) and Denormals-Are-Zero
  | (DAZ) are modes that bypass IEEE 754 methods
  | of dealing with denormal floating-point
  | numbers on x86-64 and some x86 CPUs. They
  | result in reduced precision for values near
  | zero, but increased performance.
  |
  | See
  | https://software.intel.com/en-us/articles/x87-and-sse-floating-point-assists-in-ia-32-flush-to-zero-ftz-and-denormals-are-zero-daz
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/FlushDenormal.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/FlushDenormal.cpp]

pub const CPU_DENORMALS_ZERO: u32 = 0x0040;
pub const CPU_FLUSH_ZERO:     u32 = 0x8000;

pub fn cpu_set_flush_denormal(on: bool) -> bool {
    
    todo!();
        /*
            // Compile if we have SSE support (GCC), x86-64 (MSVC), or x86 with SSE (MSVC)
        #if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
          // Denormals-Are-Zero is supported by most SSE2 processors, with the exception
          // of some early Pentium 4 processors. We guard it with a runtime check.
          // Flush-To-Zero (FTZ) only requires SSE.
          if (cpuinfo_has_x86_daz()) {
            unsigned int csr = _mm_getcsr();
            csr &= ~DENORMALS_ZERO;
            csr &= ~FLUSH_ZERO;
            if (on) {
              csr |= DENORMALS_ZERO;
              csr |= FLUSH_ZERO;
            }
            _mm_setcsr(csr);
            return true;
          }
        #endif
          return false;
        */
}
