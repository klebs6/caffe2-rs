crate::ix!();

#[inline] pub fn get_blob_options<'a>(
    blob_name:       &str,
    options_list:    &SerializationOptions,
    default_options: &'a BlobSerializationOptions) -> &'a BlobSerializationOptions 
{
    todo!();
    /*
        for (const auto& options : options_list.options()) {
        const auto& name_regex = options.blob_name_regex();
        if (name_regex.empty()) {
          return options;
        }

    #if CAFFE2_HAVE_RE2
        // If we have re2, prefer it over std::regex.
        re2::RE2 regex(name_regex);
        if (re2::RE2::FullMatch(
            re2::StringPiece(blob_name.data(), blob_name.size()), regex)) {
          return options;
        }
    #else
        // std::regex should be avoided if at all possible, but use it as a fallback
        // if we don't have re2 (e.g., for some issues with it see
        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61582)
        if (std::regex_match(
                blob_name.begin(), blob_name.end(), std::regex(name_regex))) {
          return options;
        }
    #endif
      }
      return default_options;
    */
}
