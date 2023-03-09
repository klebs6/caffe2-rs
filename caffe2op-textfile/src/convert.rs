crate::ix!();

#[inline] pub fn convert(
    dst_type:  TensorProto_DataType,
    src_start: *const u8,
    src_end:   *const u8,
    dst:       *mut c_void)  {
    
    todo!();
    /*
        switch (dst_type) {
        case TensorProto_DataType_STRING: {
          static_cast<std::string*>(dst)->assign(src_start, src_end);
        } break;
        case TensorProto_DataType_FLOAT: {
          // TODO(azzolini): avoid copy, use faster conversion
          std::string str_copy(src_start, src_end);
          const char* src_copy = str_copy.c_str();
          char* src_copy_end;
          float val = strtof(src_copy, &src_copy_end);
          if (src_copy == src_copy_end) {
            throw std::runtime_error("Invalid float: " + str_copy);
          }
          *static_cast<float*>(dst) = val;
        } break;
        default:
          throw std::runtime_error("Unsupported type.");
      }
    */
}
