crate::ix!();

#[inline] pub fn format_string<Ts>( pattern: &String, values: Ts) -> String 
{
    todo!();
    /*
        // Start with an initial buffer size that is probably enough most of the time.
      std::string buffer(256, '\0');
      auto bytes_written =
          snprintf(&buffer[0], buffer.size(), pattern.c_str(), values...);
      if (bytes_written < 0) {
        throw std::runtime_error("FormatString failed");
      }
      if (bytes_written > buffer.size()) {
        // Our initial buffer size wasn't enough, resize and run again.
        buffer.resize(bytes_written + 1);
        bytes_written =
            snprintf(&buffer[0], buffer.size(), pattern.c_str(), values...);
        if (bytes_written < 0) {
          throw std::runtime_error("FormatString failed");
        }
      }
      // Truncate the string to the correct size to trim off the nul terminator.
      buffer.resize(bytes_written);
      return buffer;
    */
}
