crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/DynamicLibrary.h]

pub struct DynamicLibrary {
    handle: *mut void, // default = nullptr
}

impl DynamicLibrary {
    
    pub fn new(
        name:     *const u8,
        alt_name: *const u8) -> Self {
        let alt_name: *const u8 = alt_name.unwrap_or(nullptr);
        todo!();
        /*


        
        */
    }
    
    pub fn sym(&mut self, name: *const u8)  {
        
        todo!();
        /*
        
        */
    }
}


//-------------------------------------------[.cpp/pytorch/aten/src/ATen/DynamicLibrary.cpp]
#[cfg(not(C10_MOBILE))]
#[cfg(not(_WIN32))]
pub fn checkdl(x: *mut void)  {
    
    todo!();
        /*
            if (!x) {
        AT_ERROR("Error in dlopen or dlsym: ", dlerror());
      }

      return x;
        */
}


#[cfg(not(C10_MOBILE))]
#[cfg(not(_WIN32))]
impl DynamicLibrary {
    
    pub fn new(
        name:     *const u8,
        alt_name: *const u8) -> Self {
    
        todo!();
        /*


      handle = dlopen(name, RTLD_LOCAL | RTLD_NOW);
      if (!handle) {
        if (alt_name) {
          handle = dlopen(alt_name, RTLD_LOCAL | RTLD_NOW);
          if (!handle) {
            AT_ERROR("Error in dlopen for library ", name, "and ", alt_name);
          }
        } else {
          AT_ERROR("Error in dlopen: ", dlerror());
        }
      }
        */
    }
    
    pub fn sym(&mut self, name: *const u8)  {
        
        todo!();
        /*
            AT_ASSERT(handle);
      return checkDL(dlsym(handle, name));
        */
    }
}

#[cfg(not(C10_MOBILE))]
#[cfg(not(_WIN32))]
impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        todo!();
        /*
            if (!handle)
        return;
      dlclose(handle);
        */
    }
}

#[cfg(not(C10_MOBILE))]
#[cfg(_WIN32)]
impl DynamicLibrary {
    
    pub fn new(
        name:     *const u8,
        alt_name: *const u8) -> Self {
    
        todo!();
        /*


      HMODULE theModule;
      bool reload = true;
      auto wname = u8u16(name);
      // Check if LOAD_LIBRARY_SEARCH_DEFAULT_DIRS is supported
      if (GetProcAddress(GetModuleHandleW(L"KERNEL32.DLL"), "AddDllDirectory") != NULL) {
        theModule = LoadLibraryExW(
            wname.c_str(),
            NULL,
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (theModule != NULL || (GetLastError() != ERROR_MOD_NOT_FOUND)) {
          reload = false;
        }
      }

      if (reload) {
        theModule = LoadLibraryW(wname.c_str());
      }

      if (theModule) {
        handle = theModule;
      } else {
        char buf[256];
        DWORD dw = GetLastError();
        FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                      NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                      buf, (sizeof(buf) / sizeof(char)), NULL);
        AT_ERROR("error in LoadLibrary for ", name, ". WinError ", dw, ": ", buf);
      }
        */
    }
    
    pub fn sym(&mut self, name: *const u8)  {
        
        todo!();
        /*
            AT_ASSERT(handle);
      FARPROC procAddress = GetProcAddress((HMODULE)handle, name);
      if (!procAddress) {
        AT_ERROR("error in GetProcAddress");
      }
      return (void*)procAddress;
        */
    }
}

#[cfg(not(C10_MOBILE))]
#[cfg(_WIN32)]
impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        todo!();
        /*
            if (!handle) {
        return;
      }
      FreeLibrary((HMODULE)handle);
        */
    }
}
