crate::ix!();

pub mod emulator {

    /**
      | Replace a @substring in a given @line
      | with @target
      |
      */
    #[inline] pub fn replace(
        line:      String,
        substring: &String,
        target:    &String) -> String {
        
        todo!();
        /*
            size_t index = 0;
          while (true) {
            index = line.find(substring, index);
            if (index == std::string::npos) {
              break;
            }
            line.replace(index, substring.length(), target);
            index += substring.length();
          }
          return line;
        */
    }

    /**
      | Split given @str into a vector of strings
      | delimited by @delim
      |
      */
    #[inline] pub fn split(str: &String, delim: &String) -> Vec<String> {
        
        todo!();
        /*
            std::vector<std::string> tokens;
          size_t prev = 0, pos = 0;
          do {
            pos = str.find(delim, prev);
            if (pos == std::string::npos) {
              pos = str.length();
            }
            std::string token = str.substr(prev, pos - prev);
            if (!token.empty()) {
              tokens.push_back(token);
            }
            prev = pos + delim.length();
          } while (pos < str.length() && prev < str.length());
          return tokens;
        */
    }

    /**
      | Check if the given @path is valid.
      | 
      | Remove the file/folder if @remove is
      | specified
      |
      */
    #[inline] pub fn check_path_valid(path: String, remove: Option<bool>) -> bool {
        let remove: bool = remove.unwrap_or(true);

        todo!();
        /*
            CAFFE_ENFORCE(!path.empty());
          std::ifstream file(path.c_str());
          // The file should exist or the path is valid
          if (!file.good() && !static_cast<bool>(std::ofstream(path).put('t'))) {
            return false;
          }
          file.close();
          if (remove) {
            std::remove(path.c_str());
          }
          return true;
        */
    }
}
