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

impl ModuleSchema {
    
    pub fn new(name: *const u8, description: *const u8) -> Self {
        todo!();
        /*
            std::lock_guard<std::mutex> guard(gModuleChangeMutex());
      MutableCurrentModules().emplace(name, this);
        */
    }
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

