crate::ix!();

pub struct OnnxifiOptionHelper {
    
    /// Pointer to loaded onnxifi library
    lib:  *mut OnnxifiLibrary, // default = nullptr
}

impl OnnxifiOptionHelper {
    
    pub fn new() -> Self {
    
        todo!();
        /*
            lib_ = OnnxinitOnnxifiLibrary();
      CAFFE_ENFORCE(lib_, "Cannot initialize ONNXIFI library");
        */
    }
    
    /// Set Onnxifi option
    #[inline] pub fn set_onnxifi_option(&mut self, option: &String, value: &String) -> bool {
        
        todo!();
        /*
            #ifdef ONNXIFI_ENABLE_EXT
      onnxStatus (*onnxSetOptionFunctionPointer)(
          const char* optionName, const char* optionValue) = nullptr;
      union {
        onnxExtensionFunctionPointer p;
        decltype(onnxSetOptionFunctionPointer) set;
      } u{};
      onnxBackendID backend_id = nullptr;
      if (lib_->onnxGetExtensionFunctionAddress(
              backend_id, "onnxSetOptionFunction", &u.p) !=
          ONNXIFI_STATUS_SUCCESS) {
        LOG(ERROR) << "Cannot find onnxSetOptionFunction";
        return false;
      } else {
        onnxSetOptionFunctionPointer = u.set;
      }
      if (onnxSetOptionFunctionPointer != nullptr &&
          (*onnxSetOptionFunctionPointer)(option.c_str(), value.c_str()) ==
              ONNXIFI_STATUS_SUCCESS) {
        return true;
      }
    #endif
      return false;
        */
    }
    
    ///  Get Onnxifi option
    #[inline] pub fn get_onnxifi_option(&mut self, option: &String) -> String {
        
        todo!();
        /*
            #ifdef ONNXIFI_ENABLE_EXT
      onnxStatus (*onnxGetOptionFunctionPointer)(
          const char* optionName, char* optionValue, size_t* optionValueLength) =
          nullptr;
      union {
        onnxExtensionFunctionPointer p;
        decltype(onnxGetOptionFunctionPointer) get;
      } u{};
      onnxBackendID backend_id = nullptr;
      if (lib_->onnxGetExtensionFunctionAddress(
              backend_id, "onnxGetOptionFunction", &u.p) !=
          ONNXIFI_STATUS_SUCCESS) {
        LOG(ERROR) << "Cannot find onnxGetOptionFunction";
        return "";
      } else {
        onnxGetOptionFunctionPointer = u.get;
      }

      constexpr size_t ll = 1024;
      char buf[ll];
      size_t len = ll;
      if (onnxGetOptionFunctionPointer != nullptr &&
          (*onnxGetOptionFunctionPointer)(option.c_str(), buf, &len) ==
              ONNXIFI_STATUS_SUCCESS) {
        return std::string(buf, len);
      }
    #endif

      return "";
        */
    }
}


