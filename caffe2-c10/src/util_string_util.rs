crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/StringUtil.h]

pub struct CompileTimeEmptyString {

}

impl Into<String> for CompileTimeEmptyString {
    
    #[inline] fn into(self) -> String {
        todo!();
        /*
            static const string empty_string_literal;
        return empty_string_literal;
        */
    }
}

impl Into<u8> for CompileTimeEmptyString {
    
    #[inline] fn into(self) -> u8 {
        todo!();
        /*
            return "";
        */
    }
}

lazy_static!{
    /*
    template <typename T>
    struct CanonicalizeStrTypes {
      using type = const T&;
    };

    template <size_t N>
    struct CanonicalizeStrTypes<char[N]> {
      using type = const char*;
    };

    inline ostream& _str(ostream& ss) {
      return ss;
    }

    template <typename T>
    inline ostream& _str(ostream& ss, const T& t) {
      // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
      ss << t;
      return ss;
    }

    template <>
    inline ostream& _str<CompileTimeEmptyString>(
        ostream& ss,
        const CompileTimeEmptyString&) {
      return ss;
    }

    template <typename T, typename... Args>
    inline ostream& _str(ostream& ss, const T& t, const Args&... args) {
      return _str(_str(ss, t), args...);
    }

    template <typename... Args>
    struct _str_wrapper final {
      static string call(const Args&... args) {
        ostringstream ss;
        _str(ss, args...);
        return ss.str();
      }
    };

    // Specializations for already-a-string types.
    template <>
    struct _str_wrapper<string> final {
      // return by reference to avoid the binary size of a string copy
      static const string& call(const string& str) {
        return str;
      }
    };

    template <>
    struct _str_wrapper<const char*> final {
      static const char* call(const char* str) {
        return str;
      }
    };

    // For str() with an empty argument list (which is common in our assert
    // macros), we don't want to pay the binary size for constructing and
    // destructing a stringstream or even constructing a string.
    template <>
    struct _str_wrapper<> final {
      static CompileTimeEmptyString call() {
        return CompileTimeEmptyString();
      }
    };


    // Convert a list of string-like arguments into a single string.
    template <typename... Args>
    inline decltype(auto) str(const Args&... args) {
      return _str_wrapper<
          typename CanonicalizeStrTypes<Args>::type...>::call(args...);
    }
    */
}

#[inline] pub fn join<Container>(
    delimiter: &String,
    v:         &Container) -> String {

    todo!();
        /*
            stringstream s;
      int cnt = static_cast<int64_t>(v.size()) - 1;
      for (auto i = v.begin(); i != v.end(); ++i, --cnt) {
        s << (*i) << (cnt ? delimiter : "");
      }
      return s.str();
        */
}

/// Represents a location in source code (for
/// debugging).
///
#[derive(Debug)]
pub struct  SourceLocation {
    function: *const u8,
    file:     *const u8,
    line:     u32,
}

/// unix isprint but insensitive to locale
///
#[inline] pub fn is_print(s: u8) -> bool {
    
    todo!();
        /*
            return s > 0x1f && s < 0x7f;
        */
}

#[inline] pub fn print_quoted_string<W: Write>(
    stmt: W,
    str_: &str)  {
    
    todo!();
        /*
            stmt << "\"";
      for (auto s : str) {
        switch (s) {
          case '\\':
            stmt << "\\\\";
            break;
          case '\'':
            stmt << "\\'";
            break;
          case '\"':
            stmt << "\\\"";
            break;
          case '\a':
            stmt << "\\a";
            break;
          case '\b':
            stmt << "\\b";
            break;
          case '\f':
            stmt << "\\f";
            break;
          case '\n':
            stmt << "\\n";
            break;
          case '\r':
            stmt << "\\r";
            break;
          case '\t':
            stmt << "\\t";
            break;
          case '\v':
            stmt << "\\v";
            break;
          default:
            if (isPrint(s)) {
              stmt << s;
            } else {
              // C++ io has stateful formatting settings. Messing with
              // them is probably worse than doing this manually.
              char buf[4] = "000";
              buf[2] += s % 8;
              s /= 8;
              buf[1] += s % 8;
              s /= 8;
              buf[0] += s;
              stmt << "\\" << buf;
            }
            break;
        }
      }
      stmt << "\"";
        */
}

//-------------------------------------------[.cpp/pytorch/c10/util/StringUtil.cpp]

/// Obtains the base name from a full path.
pub fn strip_basename(full_path: &String) -> String {
    
    todo!();
        /*
            const char kSeparator = '/';
      size_t pos = full_path.rfind(kSeparator);
      if (pos != string::npos) {
        return full_path.substr(pos + 1, string::npos);
      } else {
        return full_path;
      }
        */
}

pub fn exclude_file_extension(file_name: &String) -> String {
    
    todo!();
        /*
            const char sep = '.';
      auto end_index = file_name.find_last_of(sep) == string::npos
          ? -1
          : file_name.find_last_of(sep);
      return file_name.substr(0, end_index);
        */
}

impl fmt::Display for SourceLocation {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            out << loc.function << " at " << loc.file << ":" << loc.line;
      return out;
        */
    }
}

/**
  | Replace all occurrences of "from" substring to
  | "to" string.
  |
  | Returns number of replacements
  |
  */
pub fn replace_all(
        s:    &mut String,
        from: *const u8,
        to:   *const u8) -> usize {
    
    todo!();
        /*
            TORCH_CHECK(from && *from, "");
      TORCH_CHECK(to, "");

      size_t numReplaced = 0;
      string::size_type lenFrom = strlen(from);
      string::size_type lenTo = strlen(to);
      for (auto pos = s.find(from); pos != string::npos;
           pos = s.find(from, pos + lenTo)) {
        s.replace(pos, lenFrom, to);
        numReplaced++;
      }
      return numReplaced;
        */
}


