crate::ix!();

#[inline] pub fn extract_links(
    op:            *mut OperatorStorage,
    internal_arg:  &String,
    external_arg:  &String,
    offset_arg:    &String,
    window_arg:    &String,
    links:         *mut Vec<Link>)  
{
    todo!();
    /*
        const auto& internal = op->GetRepeatedArgument<std::string>(internalArg);
      const auto& external = op->GetRepeatedArgument<std::string>(externalArg);
      const auto& offset = op->GetRepeatedArgument<int32_t>(offsetArg);
      const auto& window = op->GetRepeatedArgument<int32_t>(
          windowArg, vector<int32_t>(offset.size(), 1));
      CAFFE_ENFORCE_EQ(
          internal.size(),
          offset.size(),
          "internal/offset mismatch: ",
          internalArg,
          " ",
          externalArg);
      CAFFE_ENFORCE_EQ(
          external.size(),
          offset.size(),
          "external/offset mismatch: ",
          externalArg,
          " ",
          offsetArg);
      CAFFE_ENFORCE_EQ(
          external.size(),
          window.size(),
          "external/window mismatch: ",
          externalArg,
          " ",
          windowArg);
      for (auto i = 0; i < internal.size(); ++i) {
        detail::Link l;
        l.internal = internal[i];
        l.external = external[i];
        l.offset = offset[i];
        l.window = window[i];
        links->push_back(l);
      }
    */
}
