crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/deps/clog/include/clog.h]

pub const CLOG_NONE:    usize = 0;
pub const CLOG_FATAL:   usize = 1;
pub const CLOG_ERROR:   usize = 2;
pub const CLOG_WARNING: usize = 3;
pub const CLOG_INFO:    usize = 4;
pub const CLOG_DEBUG:   usize = 5;

#[cfg(not(CLOG_VISIBILITY))]
lazy_static!{
    /*
    #if defined(__ELF__)
    #define CLOG_VISIBILITY __attribute__((__visibility__("internal")))
    #elif defined(__MACH__)
    #define CLOG_VISIBILITY __attribute__((__visibility__("hidden")))
    #else
    #define CLOG_VISIBILITY
    #endif
    */
}

#[cfg(not(CLOG_ARGUMENTS_FORMAT))]
lazy_static!{
    /*
    #if defined(__GNUC__)
    #define CLOG_ARGUMENTS_FORMAT __attribute__((__format__(__printf__, 1, 2)))
    #else
    #define CLOG_ARGUMENTS_FORMAT
    #endif
    */
}

#[macro_export] macro_rules! clog_define_log_debug {
    ($log_debug_function_name:ident, $module:expr, $level:ident) => {
        /*
        
          CLOG_ARGUMENTS_FORMAT                                                 
          inline static void log_debug_function_name(const char* format, ...) { 
            if (level >= CLOG_DEBUG) {                                          
              va_list args;                                                     
              va_start(args, format);                                           
              clog_vlog_debug(module, format, args);                            
              va_end(args);                                                     
            }                                                                   
          }
        */
    }
}


#[macro_export] macro_rules! clog_define_log_info {
    ($log_info_function_name:ident, $module:expr, $level:ident) => {
        /*
        
          CLOG_ARGUMENTS_FORMAT                                                
          inline static void log_info_function_name(const char* format, ...) { 
            if (level >= CLOG_INFO) {                                          
              va_list args;                                                    
              va_start(args, format);                                          
              clog_vlog_info(module, format, args);                            
              va_end(args);                                                    
            }                                                                  
          }
        */
    }
}

#[macro_export] macro_rules! clog_define_log_warning {
    ($log_warning_function_name:ident, $module:expr, $level:ident) => {
        /*
        
          CLOG_ARGUMENTS_FORMAT                                                   
          inline static void log_warning_function_name(const char* format, ...) { 
            if (level >= CLOG_WARNING) {                                          
              va_list args;                                                       
              va_start(args, format);                                             
              clog_vlog_warning(module, format, args);                            
              va_end(args);                                                       
            }                                                                     
          }
        */
    }
}

#[macro_export] macro_rules! clog_define_log_error {
    ($log_error_function_name:ident, $module:expr, $level:ident) => {
        /*
        
          CLOG_ARGUMENTS_FORMAT                                                 
          inline static void log_error_function_name(const char* format, ...) { 
            if (level >= CLOG_ERROR) {                                          
              va_list args;                                                     
              va_start(args, format);                                           
              clog_vlog_error(module, format, args);                            
              va_end(args);                                                     
            }                                                                   
          }
        */
    }
}

#[macro_export] macro_rules! clog_define_log_fatal {
    ($log_fatal_function_name:ident, $module:expr, $level:expr) => {
        /*
        
          CLOG_ARGUMENTS_FORMAT                                                 
          inline static void log_fatal_function_name(const char* format, ...) { 
            if (level >= CLOG_FATAL) {                                          
              va_list args;                                                     
              va_start(args, format);                                           
              clog_vlog_fatal(module, format, args);                            
              va_end(args);                                                     
            }                                                                   
            abort();                                                            
          }
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/deps/clog/src/clog.c]

#[cfg(not(CLOG_LOG_TO_STDIO))]
lazy_static!{
    /*
    #ifdef __ANDROID__
    #define CLOG_LOG_TO_STDIO 0
    #else
    #define CLOG_LOG_TO_STDIO 1
    #endif
    */
}

/**
  | Messages up to this size are formatted
  | entirely on-stack, and don't allocate
  | heap memory
  |
  */
pub const CLOG_STACK_BUFFER_SIZE: usize = 1024;

pub const CLOG_FATAL_PREFIX:           &'static str = "Fatal error: ";
pub const CLOG_FATAL_PREFIX_LENGTH:    usize = 13;
pub const CLOG_FATAL_PREFIX_FORMAT:    &'static str = "Fatal error in %s: ";
pub const CLOG_ERROR_PREFIX:           &'static str = "Error: ";
pub const CLOG_ERROR_PREFIX_LENGTH:    usize = 7;
pub const CLOG_ERROR_PREFIX_FORMAT:    &'static str = "Error in %s: ";
pub const CLOG_WARNING_PREFIX:         &'static str = "Warning: ";
pub const CLOG_WARNING_PREFIX_LENGTH:  usize = 9;
pub const CLOG_WARNING_PREFIX_FORMAT:  &'static str = "Warning in %s: ";
pub const CLOG_INFO_PREFIX:            &'static str = "Note: ";
pub const CLOG_INFO_PREFIX_LENGTH:     usize = 6;
pub const CLOG_INFO_PREFIX_FORMAT:     &'static str = "Note (%s): ";
pub const CLOG_DEBUG_PREFIX:           &'static str = "Debug: ";
pub const CLOG_DEBUG_PREFIX_LENGTH:    usize = 7;
pub const CLOG_DEBUG_PREFIX_FORMAT:    &'static str = "Debug (%s): ";
pub const CLOG_SUFFIX_LENGTH:          usize = 1;

pub fn clog_vlog_fatal(
    module: *const u8,
    format: *const u8,
    args:   &[Args])  {

    todo!();
        /*
            #if defined(__ANDROID__) && !CLOG_LOG_TO_STDIO
      __android_log_vprint(ANDROID_LOG_FATAL, module, format, args);
    #else
      char stack_buffer[CLOG_STACK_BUFFER_SIZE];
      char* heap_buffer = NULL;
      char* out_buffer = &stack_buffer[0];

      /* The first call to vsnprintf will clobber args, thus need a copy in case a
       * second vsnprintf call is needed */
      va_list args_copy;
      va_copy(args_copy, args);

      int prefix_chars = CLOG_FATAL_PREFIX_LENGTH;
      if (module == NULL) {
        memcpy(stack_buffer, CLOG_FATAL_PREFIX, CLOG_FATAL_PREFIX_LENGTH);
      } else {
        prefix_chars = snprintf(
            stack_buffer, CLOG_STACK_BUFFER_SIZE, CLOG_FATAL_PREFIX_FORMAT, module);
        if (prefix_chars < 0) {
          /* Format error in prefix (possible if prefix is modified): skip prefix
           * and continue as if nothing happened. */
          prefix_chars = 0;
        }
      }

      int format_chars;
      if (prefix_chars + CLOG_SUFFIX_LENGTH >= CLOG_STACK_BUFFER_SIZE) {
        /*
         * Prefix + suffix alone would overflow the on-stack buffer, thus need to
         * use on-heap buffer. Do not even try to format the string into on-stack
         * buffer.
         */
        format_chars = vsnprintf(NULL, 0, format, args);
      } else {
        format_chars = vsnprintf(
            &stack_buffer[prefix_chars],
            CLOG_STACK_BUFFER_SIZE - prefix_chars - CLOG_SUFFIX_LENGTH,
            format,
            args);
      }
      if (format_chars < 0) {
        /* Format error in the message: silently ignore this particular message. */
        goto cleanup;
      }
      if (prefix_chars + format_chars + CLOG_SUFFIX_LENGTH >
          CLOG_STACK_BUFFER_SIZE) {
        /* Allocate a buffer on heap, and vsnprintf to this buffer */
        heap_buffer = malloc(prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
        if (heap_buffer == NULL) {
          goto cleanup;
        }

        if (prefix_chars > CLOG_STACK_BUFFER_SIZE) {
          /* Prefix didn't fit into on-stack buffer, re-format it again to on-heap
           * buffer */
          snprintf(
              heap_buffer,
              prefix_chars + 1 /* for '\0'-terminator */,
              CLOG_FATAL_PREFIX_FORMAT,
              module);
        } else {
          /* Copy pre-formatted prefix from on-stack buffer to on-heap buffer */
          memcpy(heap_buffer, stack_buffer, prefix_chars);
        }
        vsnprintf(
            heap_buffer + prefix_chars,
            format_chars + CLOG_SUFFIX_LENGTH,
            format,
            args_copy);
        out_buffer = heap_buffer;
      }
      out_buffer[prefix_chars + format_chars] = '\n';
    #ifdef _WIN32
      DWORD bytes_written;
      WriteFile(
          GetStdHandle(STD_ERROR_HANDLE),
          out_buffer,
          prefix_chars + format_chars + CLOG_SUFFIX_LENGTH,
          &bytes_written,
          NULL);
    #else
      write(
          STDERR_FILENO,
          out_buffer,
          prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
    #endif

    cleanup:
      free(heap_buffer);
      va_end(args_copy);
    #endif
        */
}

pub fn clog_vlog_error(
    module: *const u8,
    format: *const u8,
    args:   &[Args])  {

    todo!();
        /*
            #if defined(__ANDROID__) && !CLOG_LOG_TO_STDIO
      __android_log_vprint(ANDROID_LOG_ERROR, module, format, args);
    #else
      char stack_buffer[CLOG_STACK_BUFFER_SIZE];
      char* heap_buffer = NULL;
      char* out_buffer = &stack_buffer[0];

      /* The first call to vsnprintf will clobber args, thus need a copy in case a
       * second vsnprintf call is needed */
      va_list args_copy;
      va_copy(args_copy, args);

      int prefix_chars = CLOG_ERROR_PREFIX_LENGTH;
      if (module == NULL) {
        memcpy(stack_buffer, CLOG_ERROR_PREFIX, CLOG_ERROR_PREFIX_LENGTH);
      } else {
        prefix_chars = snprintf(
            stack_buffer, CLOG_STACK_BUFFER_SIZE, CLOG_ERROR_PREFIX_FORMAT, module);
        if (prefix_chars < 0) {
          /* Format error in prefix (possible if prefix is modified): skip prefix
           * and continue as if nothing happened. */
          prefix_chars = 0;
        }
      }

      int format_chars;
      if (prefix_chars + CLOG_SUFFIX_LENGTH >= CLOG_STACK_BUFFER_SIZE) {
        /*
         * Prefix + suffix alone would overflow the on-stack buffer, thus need to
         * use on-heap buffer. Do not even try to format the string into on-stack
         * buffer.
         */
        format_chars = vsnprintf(NULL, 0, format, args);
      } else {
        format_chars = vsnprintf(
            &stack_buffer[prefix_chars],
            CLOG_STACK_BUFFER_SIZE - prefix_chars - CLOG_SUFFIX_LENGTH,
            format,
            args);
      }
      if (format_chars < 0) {
        /* Format error in the message: silently ignore this particular message. */
        goto cleanup;
      }
      if (prefix_chars + format_chars + CLOG_SUFFIX_LENGTH >
          CLOG_STACK_BUFFER_SIZE) {
        /* Allocate a buffer on heap, and vsnprintf to this buffer */
        heap_buffer = malloc(prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
        if (heap_buffer == NULL) {
          goto cleanup;
        }

        if (prefix_chars > CLOG_STACK_BUFFER_SIZE) {
          /* Prefix didn't fit into on-stack buffer, re-format it again to on-heap
           * buffer */
          snprintf(
              heap_buffer,
              prefix_chars + 1 /* for '\0'-terminator */,
              CLOG_ERROR_PREFIX_FORMAT,
              module);
        } else {
          /* Copy pre-formatted prefix from on-stack buffer to on-heap buffer */
          memcpy(heap_buffer, stack_buffer, prefix_chars);
        }
        vsnprintf(
            heap_buffer + prefix_chars,
            format_chars + CLOG_SUFFIX_LENGTH,
            format,
            args_copy);
        out_buffer = heap_buffer;
      }
      out_buffer[prefix_chars + format_chars] = '\n';
    #ifdef _WIN32
      DWORD bytes_written;
      WriteFile(
          GetStdHandle(STD_ERROR_HANDLE),
          out_buffer,
          prefix_chars + format_chars + CLOG_SUFFIX_LENGTH,
          &bytes_written,
          NULL);
    #else
      write(
          STDERR_FILENO,
          out_buffer,
          prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
    #endif

    cleanup:
      free(heap_buffer);
      va_end(args_copy);
    #endif
        */
}

pub fn clog_vlog_warning(
        module: *const u8,
        format: *const u8,
        args:   &[Args])  {
    
    todo!();
        /*
            #if defined(__ANDROID__) && !CLOG_LOG_TO_STDIO
      __android_log_vprint(ANDROID_LOG_WARN, module, format, args);
    #else
      char stack_buffer[CLOG_STACK_BUFFER_SIZE];
      char* heap_buffer = NULL;
      char* out_buffer = &stack_buffer[0];

      /* The first call to vsnprintf will clobber args, thus need a copy in case a
       * second vsnprintf call is needed */
      va_list args_copy;
      va_copy(args_copy, args);

      int prefix_chars = CLOG_WARNING_PREFIX_LENGTH;
      if (module == NULL) {
        memcpy(stack_buffer, CLOG_WARNING_PREFIX, CLOG_WARNING_PREFIX_LENGTH);
      } else {
        prefix_chars = snprintf(
            stack_buffer,
            CLOG_STACK_BUFFER_SIZE,
            CLOG_WARNING_PREFIX_FORMAT,
            module);
        if (prefix_chars < 0) {
          /* Format error in prefix (possible if prefix is modified): skip prefix
           * and continue as if nothing happened. */
          prefix_chars = 0;
        }
      }

      int format_chars;
      if (prefix_chars + CLOG_SUFFIX_LENGTH >= CLOG_STACK_BUFFER_SIZE) {
        /*
         * Prefix + suffix alone would overflow the on-stack buffer, thus need to
         * use on-heap buffer. Do not even try to format the string into on-stack
         * buffer.
         */
        format_chars = vsnprintf(NULL, 0, format, args);
      } else {
        format_chars = vsnprintf(
            &stack_buffer[prefix_chars],
            CLOG_STACK_BUFFER_SIZE - prefix_chars - CLOG_SUFFIX_LENGTH,
            format,
            args);
      }
      if (format_chars < 0) {
        /* Format error in the message: silently ignore this particular message. */
        goto cleanup;
      }
      if (prefix_chars + format_chars + CLOG_SUFFIX_LENGTH >
          CLOG_STACK_BUFFER_SIZE) {
        /* Allocate a buffer on heap, and vsnprintf to this buffer */
        heap_buffer = malloc(prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
        if (heap_buffer == NULL) {
          goto cleanup;
        }

        if (prefix_chars > CLOG_STACK_BUFFER_SIZE) {
          /* Prefix didn't fit into on-stack buffer, re-format it again to on-heap
           * buffer */
          snprintf(
              heap_buffer,
              prefix_chars + 1 /* for '\0'-terminator */,
              CLOG_WARNING_PREFIX_FORMAT,
              module);
        } else {
          /* Copy pre-formatted prefix from on-stack buffer to on-heap buffer */
          memcpy(heap_buffer, stack_buffer, prefix_chars);
        }
        vsnprintf(
            heap_buffer + prefix_chars,
            format_chars + CLOG_SUFFIX_LENGTH,
            format,
            args_copy);
        out_buffer = heap_buffer;
      }
      out_buffer[prefix_chars + format_chars] = '\n';
    #ifdef _WIN32
      DWORD bytes_written;
      WriteFile(
          GetStdHandle(STD_ERROR_HANDLE),
          out_buffer,
          prefix_chars + format_chars + CLOG_SUFFIX_LENGTH,
          &bytes_written,
          NULL);
    #else
      write(
          STDERR_FILENO,
          out_buffer,
          prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
    #endif

    cleanup:
      free(heap_buffer);
      va_end(args_copy);
    #endif
        */
}


pub fn clog_vlog_info(
        module: *const u8,
        format: *const u8,
        args:   &[Args])  {
    
    todo!();
        /*
            #if defined(__ANDROID__) && !CLOG_LOG_TO_STDIO
      __android_log_vprint(ANDROID_LOG_INFO, module, format, args);
    #else
      char stack_buffer[CLOG_STACK_BUFFER_SIZE];
      char* heap_buffer = NULL;
      char* out_buffer = &stack_buffer[0];

      /* The first call to vsnprintf will clobber args, thus need a copy in case a
       * second vsnprintf call is needed */
      va_list args_copy;
      va_copy(args_copy, args);

      int prefix_chars = CLOG_INFO_PREFIX_LENGTH;
      if (module == NULL) {
        memcpy(stack_buffer, CLOG_INFO_PREFIX, CLOG_INFO_PREFIX_LENGTH);
      } else {
        prefix_chars = snprintf(
            stack_buffer, CLOG_STACK_BUFFER_SIZE, CLOG_INFO_PREFIX_FORMAT, module);
        if (prefix_chars < 0) {
          /* Format error in prefix (possible if prefix is modified): skip prefix
           * and continue as if nothing happened. */
          prefix_chars = 0;
        }
      }

      int format_chars;
      if (prefix_chars + CLOG_SUFFIX_LENGTH >= CLOG_STACK_BUFFER_SIZE) {
        /*
         * Prefix + suffix alone would overflow the on-stack buffer, thus need to
         * use on-heap buffer. Do not even try to format the string into on-stack
         * buffer.
         */
        format_chars = vsnprintf(NULL, 0, format, args);
      } else {
        format_chars = vsnprintf(
            &stack_buffer[prefix_chars],
            CLOG_STACK_BUFFER_SIZE - prefix_chars - CLOG_SUFFIX_LENGTH,
            format,
            args);
      }
      if (format_chars < 0) {
        /* Format error in the message: silently ignore this particular message. */
        goto cleanup;
      }
      if (prefix_chars + format_chars + CLOG_SUFFIX_LENGTH >
          CLOG_STACK_BUFFER_SIZE) {
        /* Allocate a buffer on heap, and vsnprintf to this buffer */
        heap_buffer = malloc(prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
        if (heap_buffer == NULL) {
          goto cleanup;
        }

        if (prefix_chars > CLOG_STACK_BUFFER_SIZE) {
          /* Prefix didn't fit into on-stack buffer, re-format it again to on-heap
           * buffer */
          snprintf(
              heap_buffer,
              prefix_chars + 1 /* for '\0'-terminator */,
              CLOG_INFO_PREFIX_FORMAT,
              module);
        } else {
          /* Copy pre-formatted prefix from on-stack buffer to on-heap buffer */
          memcpy(heap_buffer, stack_buffer, prefix_chars);
        }
        vsnprintf(
            heap_buffer + prefix_chars,
            format_chars + CLOG_SUFFIX_LENGTH,
            format,
            args_copy);
        out_buffer = heap_buffer;
      }
      out_buffer[prefix_chars + format_chars] = '\n';
    #ifdef _WIN32
      DWORD bytes_written;
      WriteFile(
          GetStdHandle(STD_OUTPUT_HANDLE),
          out_buffer,
          prefix_chars + format_chars + CLOG_SUFFIX_LENGTH,
          &bytes_written,
          NULL);
    #else
      write(
          STDOUT_FILENO,
          out_buffer,
          prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
    #endif

    cleanup:
      free(heap_buffer);
      va_end(args_copy);
    #endif
        */
}


pub fn clog_vlog_debug(
        module: *const u8,
        format: *const u8,
        args:   &[Args])  {
    
    todo!();
        /*
            #if defined(__ANDROID__) && !CLOG_LOG_TO_STDIO
      __android_log_vprint(ANDROID_LOG_DEBUG, module, format, args);
    #else
      char stack_buffer[CLOG_STACK_BUFFER_SIZE];
      char* heap_buffer = NULL;
      char* out_buffer = &stack_buffer[0];

      /* The first call to vsnprintf will clobber args, thus need a copy in case a
       * second vsnprintf call is needed */
      va_list args_copy;
      va_copy(args_copy, args);

      int prefix_chars = CLOG_DEBUG_PREFIX_LENGTH;
      if (module == NULL) {
        memcpy(stack_buffer, CLOG_DEBUG_PREFIX, CLOG_DEBUG_PREFIX_LENGTH);
      } else {
        prefix_chars = snprintf(
            stack_buffer, CLOG_STACK_BUFFER_SIZE, CLOG_DEBUG_PREFIX_FORMAT, module);
        if (prefix_chars < 0) {
          /* Format error in prefix (possible if prefix is modified): skip prefix
           * and continue as if nothing happened. */
          prefix_chars = 0;
        }
      }

      int format_chars;
      if (prefix_chars + CLOG_SUFFIX_LENGTH >= CLOG_STACK_BUFFER_SIZE) {
        /*
         * Prefix + suffix alone would overflow the on-stack buffer, thus need to
         * use on-heap buffer. Do not even try to format the string into on-stack
         * buffer.
         */
        format_chars = vsnprintf(NULL, 0, format, args);
      } else {
        format_chars = vsnprintf(
            &stack_buffer[prefix_chars],
            CLOG_STACK_BUFFER_SIZE - prefix_chars - CLOG_SUFFIX_LENGTH,
            format,
            args);
      }
      if (format_chars < 0) {
        /* Format error in the message: silently ignore this particular message. */
        goto cleanup;
      }
      if (prefix_chars + format_chars + CLOG_SUFFIX_LENGTH >
          CLOG_STACK_BUFFER_SIZE) {
        /* Allocate a buffer on heap, and vsnprintf to this buffer */
        heap_buffer = malloc(prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
        if (heap_buffer == NULL) {
          goto cleanup;
        }

        if (prefix_chars > CLOG_STACK_BUFFER_SIZE) {
          /* Prefix didn't fit into on-stack buffer, re-format it again to on-heap
           * buffer */
          snprintf(
              heap_buffer,
              prefix_chars + 1 /* for '\0'-terminator */,
              CLOG_DEBUG_PREFIX_FORMAT,
              module);
        } else {
          /* Copy pre-formatted prefix from on-stack buffer to on-heap buffer */
          memcpy(heap_buffer, stack_buffer, prefix_chars);
        }
        vsnprintf(
            heap_buffer + prefix_chars,
            format_chars + CLOG_SUFFIX_LENGTH,
            format,
            args_copy);
        out_buffer = heap_buffer;
      }
      out_buffer[prefix_chars + format_chars] = '\n';
    #ifdef _WIN32
      DWORD bytes_written;
      WriteFile(
          GetStdHandle(STD_OUTPUT_HANDLE),
          out_buffer,
          prefix_chars + format_chars + CLOG_SUFFIX_LENGTH,
          &bytes_written,
          NULL);
    #else
      write(
          STDOUT_FILENO,
          out_buffer,
          prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
    #endif

    cleanup:
      free(heap_buffer);
      va_end(args_copy);
    #endif
        */
}
