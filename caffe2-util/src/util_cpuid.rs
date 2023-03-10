/*!
  | Implementation of CpuId that is borrowed from
  | folly.
  |
  | TODO: It might be good to use cpuinfo
  | third-party dependency instead for consistency
  | sake.
  */

crate::ix!();

/**
  | Identification of an Intel CPU.
  | 
  | Supports CPUID feature flags (EAX=1)
  | and extended features (EAX=7, ECX=0).
  | 
  | Values from http://www.intel.com/content/www/us/en/processors/processor-identification-cpuid-instruction-note.html
  |
  */
pub struct CpuId { }

#[inline] pub fn get_cpu_id<'a>() -> &'a CpuId {
    
    todo!();
    /*
        static CpuId cpuid_singleton;
      return cpuid_singleton;
    */
}

static f1c_: AtomicU32 = AtomicU32::new(0);
static f1d_: AtomicU32 = AtomicU32::new(0);
static f7b_: AtomicU32 = AtomicU32::new(0);
static f7c_: AtomicU32 = AtomicU32::new(0);

macro_rules! X {
    ($name:ident, $r:ident, $bit:tt) => {

        #[inline] pub fn $name() -> bool {
            todo!();
            /*
            (($r) & (1 << $bit)) != 0
            */
        }

    }
}

// cpuid(1): Processor Info and Feature Bits.
macro_rules! C {
    ($name:ident, $bit:tt) => {
        X!{$name, f1c_, $bit}
    }
}

macro_rules! D {
    ($name:ident, $bit:tt) => { 
        X!{$name, f1d_, $bit}
    }
}

// cpuid(7): Extended Features.
macro_rules! B {
    ($name:ident, $bit:tt) => {
        X!{$name, f7b_, $bit}
    }
}

macro_rules! E {
    ($name:ident, $bit:tt) => { 
        X!{$name, f7c_, $bit}
    }
}

impl CpuId {
    
    pub fn new() -> Self {
        todo!();
        /*
            #ifdef _MSC_VER
      int reg[4];
      __cpuid(static_cast<int*>(reg), 0);
      const int n = reg[0];
      if (n >= 1) {
        __cpuid(static_cast<int*>(reg), 1);
        f1c_ = uint32_t(reg[2]);
        f1d_ = uint32_t(reg[3]);
      }
      if (n >= 7) {
        __cpuidex(static_cast<int*>(reg), 7, 0);
        f7b_ = uint32_t(reg[1]);
        f7c_ = uint32_t(reg[2]);
      }
    #elif defined(__i386__) && defined(__PIC__) && !defined(__clang__) && \
        defined(__GNUC__)
      // The following block like the normal cpuid branch below, but gcc
      // reserves ebx for use of its pic register so we must specially
      // handle the save and restore to avoid clobbering the register
      uint32_t n;
      __asm__(
          "pushl %%ebx\n\t"
          "cpuid\n\t"
          "popl %%ebx\n\t"
          : "=a"(n)
          : "a"(0)
          : "ecx", "edx");
      if (n >= 1) {
        uint32_t f1a;
        __asm__(
            "pushl %%ebx\n\t"
            "cpuid\n\t"
            "popl %%ebx\n\t"
            : "=a"(f1a), "=c"(f1c_), "=d"(f1d_)
            : "a"(1)
            :);
      }
      if (n >= 7) {
        __asm__(
            "pushl %%ebx\n\t"
            "cpuid\n\t"
            "movl %%ebx, %%eax\n\r"
            "popl %%ebx"
            : "=a"(f7b_), "=c"(f7c_)
            : "a"(7), "c"(0)
            : "edx");
      }
    #elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
      uint32_t n;
      __asm__("cpuid" : "=a"(n) : "a"(0) : "ebx", "ecx", "edx");
      if (n >= 1) {
        uint32_t f1a;
        __asm__("cpuid" : "=a"(f1a), "=c"(f1c_), "=d"(f1d_) : "a"(1) : "ebx");
      }
      if (n >= 7) {
        uint32_t f7a;
        __asm__("cpuid"
                : "=a"(f7a), "=b"(f7b_), "=c"(f7c_)
                : "a"(7), "c"(0)
                : "edx");
      }
    #endif
        */
    }

    C![sse3,           0];
    C![pclmuldq,       1];
    C![dtes64,         2];
    C![monitor,        3];
    C![dscpl,          4];
    C![vmx,            5];
    C![smx,            6];
    C![eist,           7];
    C![tm2,            8];
    C![ssse3,          9];
    C![cnxtid,         10];
    C![fma,            12];
    C![cx16,           13];
    C![xtpr,           14];
    C![pdcm,           15];
    C![pcid,           17];
    C![dca,            18];
    C![sse41,          19];
    C![sse42,          20];
    C![x2apic,         21];
    C![movbe,          22];
    C![popcnt,         23];
    C![tscdeadline,    24];
    C![aes,            25];
    C![xsave,          26];
    C![osxsave,        27];
    C![avx,            28];
    C![f16c,           29];
    C![rdrand,         30];

    D![fpu,            0];
    D![vme,            1];
    D![de,             2];
    D![pse,            3];
    D![tsc,            4];
    D![msr,            5];
    D![pae,            6];
    D![mce,            7];
    D![cx8,            8];
    D![apic,           9];
    D![sep,            11];
    D![mtrr,           12];
    D![pge,            13];
    D![mca,            14];
    D![cmov,           15];
    D![pat,            16];
    D![pse36,          17];
    D![psn,            18];
    D![clfsh,          19];
    D![ds,             21];
    D![acpi,           22];
    D![mmx,            23];
    D![fxsr,           24];
    D![sse,            25];
    D![sse2,           26];
    D![ss,             27];
    D![htt,            28];
    D![tm,             29];
    D![pbe,            31];

    B![bmi1,           3];
    B![hle,            4];
    B![avx2,           5];
    B![smep,           7];
    B![bmi2,           8];
    B![erms,           9];
    B![invpcid,        10];
    B![rtm,            11];
    B![mpx,            14];
    B![avx512f,        16];
    B![avx512dq,       17];
    B![rdseed,         18];
    B![adx,            19];
    B![smap,           20];
    B![avx512ifma,     21];
    B![pcommit,        22];
    B![clflushopt,     23];
    B![clwb,           24];
    B![avx512pf,       26];
    B![avx512er,       27];
    B![avx512cd,       28];
    B![sha,            29];
    B![avx512bw,       30];
    B![avx512vl,       31];

    E![prefetchwt1,    0];
    E![avx512vbmi,     1];
}
