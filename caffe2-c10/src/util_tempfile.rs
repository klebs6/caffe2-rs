crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/tempfile.h]

/**
  | Creates the filename pattern passed to and
  | completed by `mkstemp`.
  |
  | Returns vector<char> because `mkstemp` needs
  | a (non-const) `char*` and `string` only
  | provides `const char*` before C++17.
  |
  */
#[cfg(not(_WIN32))]
#[inline] pub fn make_filename(name_prefix: String) -> Vec<u8> {
    
    todo!();
        /*
            // The filename argument to `mkstemp` needs "XXXXXX" at the end according to
      // http://pubs.opengroup.org/onlinepubs/009695399/functions/mkstemp.html
      static const string kRandomPattern = "XXXXXX";

      // We see if any of these environment variables is set and use their value, or
      // else default the temporary directory to `/tmp`.
      static const char* env_variables[] = {"TMPDIR", "TMP", "TEMP", "TEMPDIR"};

      string tmp_directory = "/tmp";
      for (const char* variable : env_variables) {
        if (const char* path = getenv(variable)) {
          tmp_directory = path;
          break;
        }
      }

      vector<char> filename;
      filename.reserve(
          tmp_directory.size() + name_prefix.size() + kRandomPattern.size() + 2);

      filename.insert(filename.end(), tmp_directory.begin(), tmp_directory.end());
      filename.push_back('/');
      filename.insert(filename.end(), name_prefix.begin(), name_prefix.end());
      filename.insert(filename.end(), kRandomPattern.begin(), kRandomPattern.end());
      filename.push_back('\0');

      return filename;
        */
}

pub struct TempFile {

    #[cfg(not(_WIN32))]
    fd:   i32,

    name: String,
}

#[cfg(not(_WIN32))]
impl Default for TempFile {
    
    fn default() -> Self {
        todo!();
        /*
        : fd(-1),

        
        */
    }
}

#[cfg(not(_WIN32))]
impl Drop for TempFile {

    fn drop(&mut self) {
        todo!();
        /*
            if (fd >= 0) {
          unlink(name.c_str());
          close(fd);
        }
        */
    }
}

#[cfg(not(_WIN32))]
impl TempFile {
    
    pub fn new(
        name: String,
        fd:   i32) -> Self {
    
        todo!();
        /*
        : fd(fd),
        : name(move(name)),

        
        */
    }
    
    pub fn new_from_other(other: TempFile) -> Self {
    
        todo!();
        /*


            : fd(other.fd), name(move(other.name)) 

        other.fd = -1;
        other.name.clear();
        */
    }
    
    pub fn assign_from(&mut self, other: TempFile) -> &mut TempFile {
        
        todo!();
        /*
            fd = other.fd;
        name = move(other.name);
        other.fd = -1;
        other.name.clear();
        return *this;
        */
    }
}

pub struct TempDir {
    name: String,
}

impl Drop for TempDir {

    fn drop(&mut self) {
        todo!();
        /*
            if (!name.empty()) {
    #if !defined(_WIN32)
          rmdir(name.c_str());
    #else // defined(_WIN32)
          RemoveDirectoryA(name.c_str());
    #endif // defined(_WIN32)
        }
        */
    }
}

impl TempDir {

    pub fn new(name: &str) -> Self {
    
        todo!();
        /*
        : name(move(name)),

        
        */
    }
    
    pub fn new_from_other(other: TempDir) -> Self {
    
        todo!();
        /*
        : name(move(other.name)),

            other.name.clear();
        */
    }
    
    pub fn assign_from(&mut self, other: TempDir) -> &mut TempDir {
        
        todo!();
        /*
            name = move(other.name);
        other.name.clear();
        return *this;
        */
    }
}

/**
  | Attempts to return a temporary file or returns
  | `nullopt` if an error occurred.
  |
  | The file returned follows the pattern
  | `<tmp-dir>/<name-prefix><random-pattern>`,
  | where `<tmp-dir>` is the value of the
  | `"TMPDIR"`, `"TMP"`, `"TEMP"` or `"TEMPDIR"`
  | environment variable if any is set, or
  | otherwise `/tmp`; `<name-prefix>` is the value
  | supplied to this function, and
  | `<random-pattern>` is a random sequence of
  | numbers.
  |
  | On Windows, `name_prefix` is ignored and
  | `tmpnam` is used.
  |
  */
#[inline] pub fn try_make_tempfile(name_prefix: Option<&str>) -> Option<TempFile> {

    let name_prefix = name_prefix.unwrap_or("torch-file-");

    todo!();
        /*
            #if defined(_WIN32)
      return TempFile{tmpnam(nullptr)};
    #else
      vector<char> filename = make_filename(move(name_prefix));
      const int fd = mkstemp(filename.data());
      if (fd == -1) {
        return nullopt;
      }
      // Don't make the string from string(filename.begin(), filename.end(), or
      // there will be a trailing '\0' at the end.
      return TempFile(filename.data(), fd);
    #endif // defined(_WIN32)
        */
}

/**
  | Like `try_make_tempfile`, but throws an
  | exception if a temporary file could not be
  | returned.
  |
  */
#[inline] pub fn make_tempfile(name_prefix: Option<&str>) -> TempFile {

    let name_prefix = name_prefix.unwrap_or("torch-file-");

    todo!();
        /*
            if (auto tempfile = try_make_tempfile(move(name_prefix))) {
        return move(*tempfile);
      }
      TORCH_CHECK(false, "Error generating temporary file: ", strerror(errno));
        */
}

/**
  | Attempts to return a temporary directory or
  | returns `nullopt` if an error occurred.
  |
  | The directory returned follows the pattern
  | `<tmp-dir>/<name-prefix><random-pattern>/`,
  | where `<tmp-dir>` is the value of the
  | `"TMPDIR"`, `"TMP"`, `"TEMP"` or `"TEMPDIR"`
  | environment variable if any is set, or
  | otherwise `/tmp`; `<name-prefix>` is the value
  | supplied to this function, and
  | `<random-pattern>` is a random sequence of
  | numbers.
  |
  | On Windows, `name_prefix` is ignored and
  | `tmpnam` is used.
  |
  */
#[inline] pub fn try_make_tempdir(name_prefix: Option<&str>) -> Option<TempDir> {

    let name_prefix = name_prefix.unwrap_or("torch-dir-");

    todo!();
        /*
            #if defined(_WIN32)
      while (true) {
        const char* dirname = tmpnam(nullptr);
        if (!dirname) {
          return nullopt;
        }
        if (CreateDirectoryA(dirname, NULL)) {
          return TempDir(dirname);
        }
        if (GetLastError() != ERROR_ALREADY_EXISTS) {
          return nullopt;
        }
      }
      return nullopt;
    #else
      vector<char> filename = make_filename(move(name_prefix));
      const char* dirname = mkdtemp(filename.data());
      if (!dirname) {
        return nullopt;
      }
      return TempDir(dirname);
    #endif // defined(_WIN32)
        */
}

/**
  | Like `try_make_tempdir`, but throws
  | an exception if a temporary directory
  | could not be returned.
  |
  */
#[inline] pub fn make_tempdir(name_prefix: Option<&str>) -> TempDir {

    let name_prefix = name_prefix.unwrap_or("torch-dir-");

    todo!();
        /*
            if (auto tempdir = try_make_tempdir(move(name_prefix))) {
        return move(*tempdir);
      }
      TORCH_CHECK(
          false, "Error generating temporary directory: ", strerror(errno));
        */
}
