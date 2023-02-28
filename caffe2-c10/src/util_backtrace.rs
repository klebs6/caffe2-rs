crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/Backtrace.h]

//-------------------------------------------[.cpp/pytorch/c10/util/Backtrace.cpp]

#[cfg(SUPPORTS_BACKTRACE)]
pub struct FrameInformation {

    /**
    | If available, the demangled name of
    | the function at this frame, else whatever
    | (possibly mangled) name we got from
    | `backtrace()`.
    |
    */
    function_name:        String,

    /**
      | This is a number in hexadecimal form
      | (e.g. "0xdead") representing the offset
      | into the function's machine code at
      | which the function's body starts, i.e.
      | skipping the "prologue" that handles
      | stack manipulation and other calling
      | convention things.
      |
      */
    offset_into_function: String,


    /**
      | -----------
      | @note
      | 
      | In debugger parlance, the "object file"
      | refers to the ELF file that the symbol
      | originates from, i.e. either an executable
      | or a library.
      |
      */
    object_file:          String,
}

#[cfg(SUPPORTS_BACKTRACE)]
pub fn is_python_frame(frame: &FrameInformation) -> bool {
    
    todo!();
        /*
            return frame.object_file == "python" || frame.object_file == "python3" ||
          (frame.object_file.find("libpython") != string::npos);
        */
}

#[cfg(SUPPORTS_BACKTRACE)]
pub fn parse_frame_information(frame_string: &String) -> Option<FrameInformation> {
    
    todo!();
        /*
            FrameInformation frame;

      // This is the function name in the CXX ABI mangled format, e.g. something
      // like _Z1gv. Reference:
      // https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling
      string mangled_function_name;

    #if defined(__GLIBCXX__)
      // In GLIBCXX, `frame_string` follows the pattern
      // `<object-file>(<mangled-function-name>+<offset-into-function>)
      // [<return-address>]`

      auto function_name_start = frame_string.find("(");
      if (function_name_start == string::npos) {
        return nullopt;
      }
      function_name_start += 1;

      auto offset_start = frame_string.find('+', function_name_start);
      if (offset_start == string::npos) {
        return nullopt;
      }
      offset_start += 1;

      const auto offset_end = frame_string.find(')', offset_start);
      if (offset_end == string::npos) {
        return nullopt;
      }

      frame.object_file = frame_string.substr(0, function_name_start - 1);
      frame.offset_into_function =
          frame_string.substr(offset_start, offset_end - offset_start);

      // NOTE: We don't need to parse the return address because
      // we already have it from the call to `backtrace()`.

      mangled_function_name = frame_string.substr(
          function_name_start, (offset_start - 1) - function_name_start);
    #elif defined(_LIBCPP_VERSION)
      // In LIBCXX, The pattern is
      // `<frame number> <object-file> <return-address> <mangled-function-name> +
      // <offset-into-function>`
      string skip;
      istringstream input_stream(frame_string);
      // operator>>() does not fail -- if the input stream is corrupted, the
      // strings will simply be empty.
      input_stream >> skip >> frame.object_file >> skip >> mangled_function_name >>
          skip >> frame.offset_into_function;
    #else
    #warning Unknown standard library, backtraces may have incomplete debug information
      return nullopt;
    #endif // defined(__GLIBCXX__)

      // Some system-level functions don't have sufficient debug information, so
      // we'll display them as "<unknown function>". They'll still have a return
      // address and other pieces of information.
      if (mangled_function_name.empty()) {
        frame.function_name = "<unknown function>";
        return frame;
      }

      frame.function_name = demangle(mangled_function_name.c_str());
      return frame;
        */
}

#[cfg(not(SUPPORTS_BACKTRACE))]
#[cfg(_MSC_VER)]
pub const MAX_NAME_LEN: i32 = 256;

#[cfg(not(SUPPORTS_BACKTRACE))]
#[cfg(_MSC_VER)]
pub fn get_module_base_name(addr: *mut c_void) -> String {
    
    todo!();
        /*
            HMODULE h_module;
      char module[max_name_len];
      strcpy(module, "");
      GetModuleHandleEx(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          (LPCTSTR)addr,
          &h_module);
      if (h_module != NULL) {
        GetModuleFileNameA(h_module, module, max_name_len);
      }
      char* last_slash_pos = strrchr(module, '\\');
      if (last_slash_pos) {
        string module_base_name(last_slash_pos + 1);
        return module_base_name;
      } else {
        string module_base_name(module);
        return module_base_name;
      }
        */
}

#[cfg(not(SUPPORTS_BACKTRACE))]
#[cfg(_MSC_VER)]
pub struct SymbolHelper {
    inited:  bool, // default = false
    process: HANDLE,
}

#[cfg(not(SUPPORTS_BACKTRACE))]
#[cfg(_MSC_VER)]
impl SymbolHelper {
    
    pub fn get_instance() -> &mut SymbolHelper {
        
        todo!();
        /*
            static SymbolHelper instance;
        return instance;
        */
    }
}

#[cfg(not(SUPPORTS_BACKTRACE))]
#[cfg(_MSC_VER)]
impl Default for SymbolHelper {
    
    fn default() -> Self {
        todo!();
        /*


            process = GetCurrentProcess();
        DWORD flags = SymGetOptions();
        SymSetOptions(flags | SYMOPT_DEFERRED_LOADS);
        inited = SymInitialize(process, NULL, TRUE);
        */
    }
}

#[cfg(not(SUPPORTS_BACKTRACE))]
#[cfg(_MSC_VER)]
impl Drop for SymbolHelper {

    fn drop(&mut self) {
        todo!();
        /*
            if (inited) {
          SymCleanup(process);
        }
        */
    }
}

pub fn get_backtrace(
    frames_to_skip:           Option<usize>,
    maximum_number_of_frames: Option<usize>,
    skip_python_frames:       Option<bool>) -> String {

    let frames_to_skip:           usize = frames_to_skip.unwrap_or(0);
    let maximum_number_of_frames: usize = maximum_number_of_frames.unwrap_or(64);
    let skip_python_frames:        bool = skip_python_frames.unwrap_or(true);

    todo!();
        /*
            #ifdef FBCODE_CAFFE2
      // For some reason, the stacktrace implementation in fbcode is
      // better than ours, see  https://github.com/pytorch/pytorch/issues/56399
      // When it's available, just use that.
      facebook::process::StackTrace st;
      return st.toString();

    #elif SUPPORTS_BACKTRACE

      // We always skip this frame (backtrace).
      frames_to_skip += 1;

      vector<void*> callstack(
          frames_to_skip + maximum_number_of_frames, nullptr);
      // backtrace() gives us a list of return addresses in the current call stack.
      // NOTE: As per man (3) backtrace it can never fail
      // (http://man7.org/linux/man-pages/man3/backtrace.3.html).
      auto number_of_frames =
          ::backtrace(callstack.data(), static_cast<int>(callstack.size()));

      // Skip as many frames as requested. This is not efficient, but the sizes here
      // are small and it makes the code nicer and safer.
      for (; frames_to_skip > 0 && number_of_frames > 0;
           --frames_to_skip, --number_of_frames) {
        callstack.erase(callstack.begin());
      }

      // `number_of_frames` is strictly less than the current capacity of
      // `callstack`, so this is just a pointer subtraction and makes the subsequent
      // code safer.
      callstack.resize(static_cast<size_t>(number_of_frames));

      // `backtrace_symbols` takes the return addresses obtained from `backtrace()`
      // and fetches string representations of each stack. Unfortunately it doesn't
      // return a struct of individual pieces of information but a concatenated
      // string, so we'll have to parse the string after. NOTE: The array returned
      // by `backtrace_symbols` is malloc'd and must be manually freed, but not the
      // strings inside the array.
      unique_ptr<char*, function<void(char**)>> raw_symbols(
          ::backtrace_symbols(callstack.data(), static_cast<int>(callstack.size())),
          /*deleter=*/free);
      const vector<string> symbols(
          raw_symbols.get(), raw_symbols.get() + callstack.size());

      // The backtrace string goes into here.
      ostringstream stream;

      // Toggles to true after the first skipped python frame.
      bool has_skipped_python_frames = false;

      for (size_t frame_number = 0; frame_number < callstack.size();
           ++frame_number) {
        const auto frame = parse_frame_information(symbols[frame_number]);

        if (skip_python_frames && frame && is_python_frame(*frame)) {
          if (!has_skipped_python_frames) {
            stream << "<omitting python frames>\n";
            has_skipped_python_frames = true;
          }
          continue;
        }

        // frame #<number>:
        stream << "frame #" << frame_number << ": ";

        if (frame) {
          // <function_name> + <offset> (<return-address> in <object-file>)
          stream << frame->function_name << " + " << frame->offset_into_function
                 << " (" << callstack[frame_number] << " in " << frame->object_file
                 << ")\n";
        } else {
          // In the edge-case where we couldn't parse the frame string, we can
          // just use it directly (it may have a different format).
          stream << symbols[frame_number] << "\n";
        }
      }

      return stream.str();
    #elif defined(_MSC_VER) // !SUPPORTS_BACKTRACE
      // This backtrace retrieval is implemented on Windows via the Windows
      // API using `CaptureStackBackTrace`, `SymFromAddr` and
      // `SymGetLineFromAddr64`.
      // https://stackoverflow.com/questions/5693192/win32-backtrace-from-c-code
      // https://stackoverflow.com/questions/26398064/counterpart-to-glibcs-backtrace-and-backtrace-symbols-on-windows
      // https://docs.microsoft.com/en-us/windows/win32/debug/capturestackbacktrace
      // https://docs.microsoft.com/en-us/windows/win32/api/dbghelp/nf-dbghelp-symfromaddr
      // https://docs.microsoft.com/en-us/windows/win32/api/dbghelp/nf-dbghelp-symgetlinefromaddr64
      // TODO: Support skipping python frames

      // We always skip this frame (backtrace).
      frames_to_skip += 1;

      DWORD64 displacement;
      DWORD disp;
      unique_ptr<IMAGEHLP_LINE64> line;

      char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
      PSYMBOL_INFO p_symbol = (PSYMBOL_INFO)buffer;

      unique_ptr<void*[]> back_trace(new void*[maximum_number_of_frames]);
      bool with_symbol = false;
      bool with_line = false;

      // The backtrace string goes into here.
      ostringstream stream;

      // Get the frames
      const USHORT n_frame = CaptureStackBackTrace(
          static_cast<DWORD>(frames_to_skip),
          static_cast<DWORD>(maximum_number_of_frames),
          back_trace.get(),
          NULL);

      // Initialize symbols if necessary
      SymbolHelper& sh = SymbolHelper::getInstance();

      for (USHORT i_frame = 0; i_frame < n_frame; ++i_frame) {
        // Get the address and the name of the symbol
        if (sh.inited) {
          p_symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
          p_symbol->MaxNameLen = MAX_SYM_NAME;
          with_symbol = SymFromAddr(
              sh.process, (ULONG64)back_trace[i_frame], &displacement, p_symbol);
        }

        // Get the line number and the module
        if (sh.inited) {
          line.reset(new IMAGEHLP_LINE64());
          line->SizeOfStruct = sizeof(IMAGEHLP_LINE64);
          with_line = SymGetLineFromAddr64(
              sh.process, (ULONG64)back_trace[i_frame], &disp, line.get());
        }

        // Get the module basename
        string module = get_module_base_name(back_trace[i_frame]);

        // The pattern on Windows is
        // `<return-address> <symbol-address>
        // <module-name>!<demangled-function-name> [<file-name> @ <line-number>]
        stream << setfill('0') << setw(16) << uppercase << hex
               << back_trace[i_frame] << dec;
        if (with_symbol) {
          stream << setfill('0') << setw(16) << uppercase << hex
                 << p_symbol->Address << dec << " " << module << "!"
                 << p_symbol->Name;
        } else {
          stream << " <unknown symbol address> " << module << "!<unknown symbol>";
        }
        stream << " [";
        if (with_line) {
          stream << line->FileName << " @ " << line->LineNumber;
        } else {
          stream << "<unknown file> @ <unknown line number>";
        }
        stream << "]" << endl;
      }

      return stream.str();
    #else // !SUPPORTS_BACKTRACE && !_WIN32
      return "(no backtrace available)";
    #endif // SUPPORTS_BACKTRACE
        */
}
