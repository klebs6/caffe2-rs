crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/Type.h]

/// Returns the printable name of the type.
#[inline] pub fn demangle_type<T>() -> *const u8 {

    todo!();
        /*
            #ifdef __GXX_RTTI
      static const auto& name = *(new string(demangle(typeid(T).name())));
      return name.c_str();
    #else // __GXX_RTTI
      return "(RTTI disabled, cannot show name)";
    #endif // __GXX_RTTI
        */
}

//-------------------------------------------[.cpp/pytorch/c10/util/Type.cpp]

#[cfg(any(__ANDROID__,_WIN32,__EMSCRIPTEN__))]
pub const HAS_DEMANGLE: usize = 0;

#[cfg(all(__APPLE__,any(TARGET_IPHONE_SIMULATOR,TARGET_OS_SIMULATOR,TARGET_OS_IPHONE)))]
pub const HAS_DEMANGLE: usize = 0;

#[cfg(not(any(
            any(__ANDROID__,_WIN32,__EMSCRIPTEN__),
            all(__APPLE__,any(TARGET_IPHONE_SIMULATOR,TARGET_OS_SIMULATOR,TARGET_OS_IPHONE))
)))]
pub const HAS_DEMANGLE: usize = 1;

/// Utility to demangle a C++ symbol name.
#[cfg(HAS_DEMANGLE)]
pub fn demangle(name: *const u8) -> String {
    
    todo!();
        /*
            int status = -1;

      // This function will demangle the mangled function name into a more human
      // readable format, e.g. _Z1gv -> g().
      // More information:
      // https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/libsupc%2B%2B/cxxabi.h
      // NOTE: `__cxa_demangle` returns a malloc'd string that we have to free
      // ourselves.
      unique_ptr<char, function<void(char*)>> demangled(
          abi::__cxa_demangle(
              name,
              /*__output_buffer=*/nullptr,
              // NOLINTNEXTLINE(modernize-use-nullptr)
              /*__length=*/0,
              &status),
          /*deleter=*/free);

      // Demangling may fail, for example when the name does not follow the
      // standard C++ (Itanium ABI) mangling scheme. This is the case for `main`
      // or `clone` for example, so the mangled name is a fine default.
      if (status == 0) {
        return demangled.get();
      } else {
        return name;
      }
        */
}

/// Utility to demangle a C++ symbol name.
#[cfg(not(HAS_DEMANGLE))]
pub fn demangle(name: *const u8) -> String {
    
    todo!();
        /*
            return string(name);
        */
}
