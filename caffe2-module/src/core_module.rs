/*!
  | A global dictionary that holds information
  | about what Caffe2 modules have been
  | loaded in the current runtime, and also
  | utility functions to load modules.
  |
  */

crate::ix!();

/**
  | A module schema that can be used to store
  | specific information about different
  | modules. Currently, we only store the
  | name and a simple description of what
  | this module does.
  |
  */
pub struct ModuleSchema {

}

#[macro_export] macro_rules! caffe2_module {
    ($name:ident, $description:expr) => {
        /*
          extern "C" {                                                              \
            bool gCaffe2ModuleSanityCheck##name() { return true; }                  \
          }                                                                         \
          namespace {                                                               \
            static ::caffe2::ModuleSchema module_schema_##name(#name, description); \
          }
        */
    }
}

#[inline] pub fn g_module_change_mutex<'a>() -> &'a mut parking_lot::RawMutex {
    
    todo!();
    /*
        static std::mutex m_;
      return m_;
    */
}

#[inline] pub fn mutable_current_modules<'a>() 
-> &'a mut HashMap<String, *const ModuleSchema> 
{
    todo!();
    /*
        static CaffeMap<string, const ModuleSchema*> module_schema_map_;
      return module_schema_map_;
    */
}

/**
  | Note(jiayq): I am not sure whether the module
  | handles are going to be used as C2 uses
  | modules via registration, but let's keep the
  | handles at least.
  */
#[inline] pub fn current_module_handles() -> HashMap<String, *mut c_void> {
    
    todo!();
    /*
        static CaffeMap<string, void*> module_handle_map_;
      return module_handle_map_;
    */
}

/**
  | @brief
  | 
  | Current Modules present in the Caffe2
  | runtime.
  | 
  | Returns:
  | 
  | map: a map of modules and (optionally)
  | their description. The key is the module
  | name, and the value is the description
  | for that module.
  | 
  | The module name is recommended to be
  | the part that constitutes the trunk
  | of the dynamic library: for example,
  | a module called libcaffe2_db_rocksdb.so
  | should have the name "caffe2_db_rocksdb".
  | 
  | The reason we do not use "lib" is because
  | it's somewhat redundant, and the reason
  | we do not include ".so" is for cross-platform
  | compatibility on platforms like mac
  | os.
  |
  */
#[inline] pub fn current_modules<'a>() -> &'a HashMap<String, *const ModuleSchema> {
    
    todo!();
    /*
        return MutableCurrentModules();
    */
}

impl ModuleSchema {
    
    pub fn new(name: *const u8, description: *const u8) -> Self {
        todo!();
        /*
            std::lock_guard<std::mutex> guard(gModuleChangeMutex());
      MutableCurrentModules().emplace(name, this);
        */
    }
}

/**
  | @brief
  | 
  | Checks whether a module is already present
  | in the current binary.
  |
  */
#[inline] pub fn has_module(name: &String) -> bool {
    
    todo!();
    /*
        auto& modules = CurrentModules();
      return (modules.find(name) != modules.end());
    */
}

/**
  | @brief
  | 
  | Load a module.
  | 
  | Inputs:
  | 
  | name: a module name or a path name.
  | 
  | It is recommended that you use the name
  | of the module, and leave the full path
  | option to only experimental modules.
  | 
  | filename: (optional) a filename that
  | serves as a hint to load the module.
  |
  */
#[inline] pub fn load_module(name: &String, filename: &String)  {
    
    todo!();
    /*
        CAFFE_ENFORCE(
          name.size() > 0 || filename.size() > 0,
          "You must provide at least one of name and filename.");
      if (name.size() && HasModule(name)) {
        VLOG(1) << "Module " << name << " already present. Skip loading.";
        return;
      }

    #ifdef _WIN32
      CAFFE_ENFORCE(
          !HasModule(name),
          "On Windows, LoadModule is currently not supported yet and you should "
          "use static linking for any module that you intend to use.");
    #else
      void* handle = nullptr;
      if (filename.size()) {
        handle = dlopen(filename.c_str(), RTLD_NOW | RTLD_GLOBAL);
        CAFFE_ENFORCE(
            handle != nullptr,
            "Cannot load module ",
            name,
            " (with given filename ",
            filename,
            "), are you sure it is correct?");
      } else {
        string inferred_name = string("lib") + name + ".so";
        handle = dlopen(inferred_name.c_str(), RTLD_NOW | RTLD_GLOBAL);
    #ifdef __APPLE__
        // For apple, we will also try the dylib extension.
        if (!handle) {
          string inferred_name = string("lib") + name + ".dylib";
          handle =
              dlopen(inferred_name.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
        }
    #endif
        CAFFE_ENFORCE(
            handle != nullptr,
            "Cannot load module ",
            name,
            " (with inferred filename ",
            inferred_name,
            "), are you sure it is in the dynamic linker search path?");
      }
      // After the module is loaded, we should check if it actually has the
      // intended module name. If not, it might be that the module file name
      // and the module name are inconsistent.
      if (name.size()) {
        string module_name_check = "gCaffe2ModuleSanityCheck" + name;
        CAFFE_ENFORCE(
            dlsym(handle, module_name_check.c_str()),
            "The loaded module ",
            name,
            " did not pass the module name sanity check. Is it built with the "
            "right configs? Make sure the file name and the CAFFE2_MODULE name "
            "are consistent.");
        // After it passes the dlopen and dlsym check, we should add it to the
        // current handles.
        std::lock_guard<std::mutex> guard(gModuleChangeMutex());
        CurrentModuleHandles()[name] = handle;
      } else {
        // If not, we issue a warning that one is recommended to use explicit
        // module name.
        LOG(WARNING)
            << "Module file " << filename
            << " was loaded without a proper module name. It is recommended "
               "that one load a model with an explicit module name in addition "
               "to the filename.";
        // As a contingency, we will store the current module handle with the
        // filename.
        std::lock_guard<std::mutex> guard(gModuleChangeMutex());
        CurrentModuleHandles()[filename] = handle;
      }
    #endif // _WIN32
    */
}
