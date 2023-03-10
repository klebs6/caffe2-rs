crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/Version.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/Version.cpp]

pub fn get_mkl_version() -> String {
    
    todo!();
        /*
            string version;
      #if AT_MKL_ENABLED()
        {
          // Magic buffer number is from MKL documentation
          // https://software.intel.com/en-us/mkl-developer-reference-c-mkl-get-version-string
          char buf[198];
          mkl_get_version_string(buf, 198);
          version = buf;
        }
      #else
        version = "MKL not found";
      #endif
      return version;
        */
}

pub fn get_mkldnn_version() -> String {
    
    todo!();
        /*
            ostringstream ss;
      #if AT_MKLDNN_ENABLED()
        // Cribbed from mkl-dnn/src/common/verbose.cpp
        // Too bad: can't get ISA info conveniently :(
        // Apparently no way to get ideep version?
        // https://github.com/intel/ideep/issues/29
        {
          const mkldnn_version_t* ver = mkldnn_version();
          ss << "Intel(R) MKL-DNN v" << ver->major << "." << ver->minor << "." << ver->patch
             << " (Git Hash " << ver->hash << ")";
        }
      #else
        ss << "MKLDNN not found";
      #endif
      return ss.str();
        */
}

pub fn get_openmp_version() -> String {
    
    todo!();
        /*
            ostringstream ss;
      #ifdef _OPENMP
        {
          ss << "OpenMP " << _OPENMP;
          // Reference:
          // https://stackoverflow.com/questions/1304363/how-to-check-the-version-of-openmp-on-linux
          const char* ver_str = nullptr;
          switch (_OPENMP) {
            case 200505:
              ver_str = "2.5";
              break;
            case 200805:
              ver_str = "3.0";
              break;
            case 201107:
              ver_str = "3.1";
              break;
            case 201307:
              ver_str = "4.0";
              break;
            case 201511:
              ver_str = "4.5";
              break;
            default:
              ver_str = nullptr;
              break;
          }
          if (ver_str) {
            ss << " (a.k.a. OpenMP " << ver_str << ")";
          }
        }
      #else
        ss << "OpenMP not found";
      #endif
      return ss.str();
        */
}

pub fn used_cpu_capability() -> String {
    
    todo!();
        /*
            // It is possible that we override the cpu_capability with
      // environment variable
      ostringstream ss;
      ss << "CPU capability usage: ";
      auto capability = native::get_cpu_capability();
      switch (capability) {
    #ifdef HAVE_VSX_CPU_DEFINITION
        case native::CPUCapability::DEFAULT:
          ss << "DEFAULT";
          break;
        case native::CPUCapability::VSX:
          ss << "VSX";
          break;
    #else
        case native::CPUCapability::DEFAULT:
          ss << "NO AVX";
          break;
        case native::CPUCapability::AVX:
          ss << "AVX";
          break;
        case native::CPUCapability::AVX2:
          ss << "AVX2";
          break;
    #endif
        default:
          break;
      }
      return ss.str();
        */
}

/**
  | Returns a detailed string describing
  | the configuration PyTorch.
  |
  */
pub fn show_config() -> String {
    
    todo!();
        /*
            ostringstream ss;
      ss << "PyTorch built with:\n";

      // Reference:
      // https://blog.kowalczyk.info/article/j/guide-to-predefined-macros-in-c-compilers-gcc-clang-msvc-etc..html

    #if defined(__GNUC__)
      {
        ss << "  - GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "\n";
      }
    #endif

    #if defined(__cplusplus)
      {
        ss << "  - C++ Version: " << __cplusplus << "\n";
      }
    #endif

    #if defined(__clang_major__)
      {
        ss << "  - clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__ << "\n";
      }
    #endif

    #if defined(_MSC_VER)
      {
        ss << "  - MSVC " << _MSC_FULL_VER << "\n";
      }
    #endif

    #if AT_MKL_ENABLED()
      ss << "  - " << get_mkl_version() << "\n";
    #endif

    #if AT_MKLDNN_ENABLED()
      ss << "  - " << get_mkldnn_version() << "\n";
    #endif

    #ifdef _OPENMP
      ss << "  - " << get_openmp_version() << "\n";
    #endif

    #ifdef USE_LAPACK
      // TODO: Actually record which one we actually picked
      ss << "  - LAPACK is enabled (usually provided by MKL)\n";
    #endif

    #if AT_NNPACK_ENABLED()
      // TODO: No version; c.f. https://github.com/Maratyszcza/NNPACK/issues/165
      ss << "  - NNPACK is enabled\n";
    #endif

    #ifdef CROSS_COMPILING_MACOSX
      ss << "  - Cross compiling on MacOSX\n";
    #endif

      ss << "  - "<< used_cpu_capability() << "\n";

      if (hasCUDA()) {
        ss << getCUDAHooks().showConfig();
      }

      ss << "  - Build settings: ";
      for (const auto& pair : GetBuildOptions()) {
        if (!pair.second.empty()) {
          ss << pair.first << "=" << pair.second << ", ";
        }
      }
      ss << "\n";

      // TODO: do HIP
      // TODO: do XLA
      // TODO: do MLC

      return ss.str();
        */
}

pub fn get_cxx_flags() -> String {
    
    todo!();
        /*
            #if defined(FBCODE_CAFFE2)
      TORCH_CHECK(
        false,
        "Buck does not populate the `CXX_FLAGS` field of Caffe2 build options. "
        "As a result, `get_cxx_flags` is OSS only."
      );
      #endif
      return GetBuildOptions().at("CXX_FLAGS");
        */
}
