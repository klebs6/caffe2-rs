// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/TH/THGeneral.h.in]

lazy_static!{
    /*
    #ifndef TH_GENERAL_INC
    #define TH_GENERAL_INC


    # define TH_EXTERNC extern "C"

    // Note(jiayq): copied from ATen/core/Macros.h. Because internal build of TH
    // and ATen are not unified yet, we need to duplicate code for now. Long term
    // we should merge macros.
    #ifdef _WIN32
    #if !defined(AT_CORE_STATIC_WINDOWS)
    // TODO: unify the controlling macros.
    #if defined(CAFFE2_BUILD_MAIN_LIBS) || defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)
    #define TH_CPP_API __declspec(dllexport)
    #else // defined(CAFFE2_BUILD_MAIN_LIBS) || defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)
    #define TH_CPP_API __declspec(dllimport)
    #endif // defined(CAFFE2_BUILD_MAIN_LIBS) || defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)
    #else // !defined(AT_CORE_STATIC_WINDOWS)
    #define TH_CPP_API
    #endif // !defined(AT_CORE_STATIC_WINDOWS)
    #else  // _WIN32
    #if defined(__GNUC__)
    #define TH_CPP_API __attribute__((__visibility__("default")))
    #endif // defined(__GNUC__)
    #endif  // _WIN32

    #ifdef NO_EXPORT
    #undef TH_CPP_API
    #define TH_CPP_API
    #endif

    #define TH_API TH_CPP_API

    #ifdef _WIN32
    # define TH_NO_RETURN __declspec(noreturn)
    # define TH_UNREACHABLE
    #else
    # define TH_NO_RETURN __attribute__((noreturn))
    # define TH_UNREACHABLE __builtin_unreachable();
    #endif

    #if defined(__GNUC__) && ((__GNUC__ > 2) || (__GNUC__ == 2 && __GNUC_MINOR__ > 4))
    # define TH_UNUSED __attribute__((unused))
    #else
    # define TH_UNUSED
    #endif

    typedef void (*THErrorHandlerFunction)(const char *msg, void *data);
    typedef void (*THArgErrorHandlerFunction)(int argNumber, const char *msg, void *data);

    #define TH_DESC_BUFF_LEN 64
    typedef struct {
        char str[TH_DESC_BUFF_LEN];
    } THDescBuff;

    TH_API THDescBuff _THSizeDesc(const i64 *size, const i64 ndim);
    TH_API TH_NO_RETURN void _THError(const char *file, const int line, const char *fmt, ...);
    TH_API void _THAssertionFailed(const char *file, const int line, const char *exp, const char *fmt, ...);
    TH_API void THSetErrorHandler(THErrorHandlerFunction new_handler, void *data);
    TH_API void THSetDefaultErrorHandler(THErrorHandlerFunction new_handler, void *data);
    TH_API void _THArgCheck(const char *file, int line, int condition, int argNumber, const char *fmt, ...);
    TH_API void THSetArgErrorHandler(THArgErrorHandlerFunction new_handler, void *data);
    TH_API void THSetDefaultArgErrorHandler(THArgErrorHandlerFunction new_handler, void *data);
    TH_API void* THAlloc(ptrdiff_t size);
    TH_API void* THRealloc(void *ptr, ptrdiff_t size);
    TH_API void THFree(void *ptr);
    TH_API void THSetGCHandler( void (*torchGCHandlerFunction)(void *data), void *data );
    // this hook should only be called by custom allocator functions
    TH_API void THHeapUpdate(ptrdiff_t size);

    #define THError(...) _THError(__FILE__, __LINE__, __VA_ARGS__)

    #define THCleanup(...) __VA_ARGS__

    #define THArgCheck(...)                                               \
    do {                                                                  \
      _THArgCheck(__FILE__, __LINE__, __VA_ARGS__);                       \
    } while(0)

    #define THArgError(...)                                               \
    do {                                                                  \
      _THArgCheck(__FILE__, __LINE__, false, __VA_ARGS__);                \
      TH_UNREACHABLE                                                      \
    } while(0)

    #define THArgCheckWithCleanup(condition, cleanup, ...)                \
    do if (!(condition)) {                                                \
      cleanup                                                             \
      _THArgCheck(__FILE__, __LINE__, 0, __VA_ARGS__);                    \
    } while(0)

    #define THAssert(exp)                                                 \
    do {                                                                  \
      if (!(exp)) {                                                       \
        _THAssertionFailed(__FILE__, __LINE__, #exp, "");                 \
      }                                                                   \
    } while(0)

    #define THAssertMsg(exp, ...)                                         \
    do {                                                                  \
      if (!(exp)) {                                                       \
        _THAssertionFailed(__FILE__, __LINE__, #exp, __VA_ARGS__);        \
      }                                                                   \
    } while(0)

    #define TH_CONCAT_STRING_2(x,y) TH_CONCAT_STRING_2_EXPAND(x,y)
    #define TH_CONCAT_STRING_2_EXPAND(x,y) #x #y

    #define TH_CONCAT_STRING_3(x,y,z) TH_CONCAT_STRING_3_EXPAND(x,y,z)
    #define TH_CONCAT_STRING_3_EXPAND(x,y,z) #x #y #z

    #define TH_CONCAT_STRING_4(x,y,z,w) TH_CONCAT_STRING_4_EXPAND(x,y,z,w)
    #define TH_CONCAT_STRING_4_EXPAND(x,y,z,w) #x #y #z #w

    #define TH_CONCAT_2(x,y) TH_CONCAT_2_EXPAND(x,y)
    #define TH_CONCAT_2_EXPAND(x,y) x ## y

    #define TH_CONCAT_3(x,y,z) TH_CONCAT_3_EXPAND(x,y,z)
    #define TH_CONCAT_3_EXPAND(x,y,z) x ## y ## z

    #define TH_CONCAT_4_EXPAND(x,y,z,w) x ## y ## z ## w
    #define TH_CONCAT_4(x,y,z,w) TH_CONCAT_4_EXPAND(x,y,z,w)

    #define THMin(X, Y)  ((X) < (Y) ? (X) : (Y))
    #define THMax(X, Y)  ((X) > (Y) ? (X) : (Y))

    #if (defined(_MSC_VER) || defined(__MINGW32__))
    #define snprintf _snprintf
    #define popen _popen
    #define pclose _pclose

    #if !defined(HAVE_SSIZE_T)
    typedef SSIZE_T ssize_t;
    #endif

    #endif

    #endif
    */
}

//-------------------------------------------[.cpp/pytorch/aten/src/TH/THGeneral.cpp]

/**
  | Torch Error Handling
  |
  */
pub fn default_error_handler_function(
    msg:  *const u8,
    data: *mut c_void)  {
    
    todo!();
        /*
            throw runtime_error(msg);
        */
}

lazy_static!{
    /*
    static THErrorHandlerFunction defaultErrorHandler = defaultErrorHandlerFunction;
    static void *defaultErrorHandlerData;
    static __thread THErrorHandlerFunction threadErrorHandler = NULL;
    static __thread void *threadErrorHandlerData;
    */
}

pub fn th_error(
    file: *const u8,
    line: i32,
    fmt:  *const u8,
    args: &[&str])  {

    todo!();
        /*
      char msg[2048];
      va_list args;

      /* vasprintf not standard */
      /* vsnprintf: how to handle if does not exists? */
      va_start(args, fmt);
      int n = vsnprintf(msg, 2048, fmt, args);
      va_end(args);

      if(n < 2048) {
        snprintf(msg + n, 2048 - n, " at %s:%d", file, line);
      }

      if (threadErrorHandler)
        (*threadErrorHandler)(msg, threadErrorHandlerData);
      else
        (*defaultErrorHandler)(msg, defaultErrorHandlerData);
      TH_UNREACHABLE;
        */
}

pub fn th_assertion_failed(
    file: *const u8,
    line: i32,
    exp:  *const u8,
    fmt:  *const u8,
    args: &[&str])  {

    todo!();
    /*
      char msg[1024];
      va_list args;
      va_start(args, fmt);
      vsnprintf(msg, 1024, fmt, args);
      va_end(args);
      _THError(file, line, "Assertion `%s' failed. %s", exp, msg);
        */
}

pub fn th_set_error_handler(
    new_handler: THErrorHandlerFunction,
    data:        *mut c_void)  {

    todo!();
        /*
            threadErrorHandler = new_handler;
      threadErrorHandlerData = data;
        */
}

pub fn th_set_default_error_handler(
    new_handler: THErrorHandlerFunction,
    data:        *mut c_void)  {
    
    todo!();
        /*
            if (new_handler)
        defaultErrorHandler = new_handler;
      else
        defaultErrorHandler = defaultErrorHandlerFunction;
      defaultErrorHandlerData = data;
        */
}

/**
  | Torch Arg Checking Handling
  |
  */
pub fn default_arg_error_handler_function(
        arg_number: i32,
        msg:        *const u8,
        data:       *mut c_void)  {
    
    todo!();
        /*
            stringstream new_error;
      new_error << "invalid argument " << argNumber << ": " << msg;
      throw runtime_error(new_error.str());
        */
}

lazy_static!{
    /*
    static THArgErrorHandlerFunction defaultArgErrorHandler = defaultArgErrorHandlerFunction;
    static void *defaultArgErrorHandlerData;
    static __thread THArgErrorHandlerFunction threadArgErrorHandler = NULL;
    static __thread void *threadArgErrorHandlerData;
    */
}

pub fn th_arg_check(
    file:       *const u8,
    line:       i32,
    condition:  i32,
    arg_number: i32,
    fmt:        *const u8,
    args:       &[&str])  {

    todo!();
    /*
       if(!condition) {
        char msg[2048];
        va_list args;

        /* vasprintf not standard */
        /* vsnprintf: how to handle if does not exists? */
        va_start(args, fmt);
        int n = vsnprintf(msg, 2048, fmt, args);
        va_end(args);

        if(n < 2048) {
          snprintf(msg + n, 2048 - n, " at %s:%d", file, line);
        }

        if (threadArgErrorHandler)
          (*threadArgErrorHandler)(argNumber, msg, threadArgErrorHandlerData);
        else
          (*defaultArgErrorHandler)(argNumber, msg, defaultArgErrorHandlerData);
        TH_UNREACHABLE;
      }
        */
}

pub fn th_set_arg_error_handler(
    new_handler: THArgErrorHandlerFunction,
    data:        *mut c_void)  {

    todo!();
        /*
            threadArgErrorHandler = new_handler;
      threadArgErrorHandlerData = data;
        */
}

pub fn th_set_default_arg_error_handler(
    new_handler: THArgErrorHandlerFunction,
    data:        *mut c_void)  {

    todo!();
        /*
            if (new_handler)
        defaultArgErrorHandler = new_handler;
      else
        defaultArgErrorHandler = defaultArgErrorHandlerFunction;
      defaultArgErrorHandlerData = data;
        */
}

lazy_static!{
    /*
    static __thread void (*torchGCFunction)(void *data) = NULL;
    static __thread void *torchGCData;
    */
}

/* 
 | Optional hook for integrating with
 | a garbage-collected frontend.
 |
 | If torch is running with a garbage-collected
 | frontend (e.g. Lua), the GC isn't aware of
 | TH-allocated memory so may not know when it
 | needs to run. These hooks trigger the GC to run
 | in two cases:
 |
 | (1) When a memory allocation (malloc, realloc, ...) 
 | fails
 |
 | (2) When the total TH-allocated memory hits
 |     a dynamically-adjusted soft maximum.
 */
pub fn th_set_gch_andler(
    torch_gcf_unction: fn(data: *mut c_void) -> c_void,
    data:              *mut c_void)  {
    
    todo!();
        /*
            torchGCFunction = torchGCFunction_;
      torchGCData = data;
        */
}

pub fn th_alloc(size: libc::ptrdiff_t)  {
    
    todo!();
        /*
            if(size < 0)
        THError("$ Torch: invalid memory size -- maybe an overflow?");

      return alloc_cpu(size);
        */
}

pub fn th_realloc(
    ptr:  *mut c_void,
    size: libc::ptrdiff_t)  {

    todo!();
        /*
            if(!ptr)
        return(THAlloc(size));

      if(size == 0)
      {
        THFree(ptr);
        return NULL;
      }

      if(size < 0)
        THError("$ Torch: invalid memory size -- maybe an overflow?");

      void *newptr = realloc(ptr, size);

      if(!newptr && torchGCFunction) {
        torchGCFunction(torchGCData);
        newptr = realloc(ptr, size);
      }

      if(!newptr)
        THError("$ Torch: not enough memory: you tried to reallocate %dGB. Buy new RAM!", size/1073741824);

      return newptr;
        */
}

pub fn th_free(ptr: *mut c_void)  {
    
    todo!();
        /*
            free_cpu(ptr);
        */
}

pub fn th_size_desc(
    size: *const i64,
    ndim: i64) -> THDescBuff {

    todo!();
        /*
            const int L = TH_DESC_BUFF_LEN;
      THDescBuff buf;
      char *str = buf.str;
      i64 i;
      i64 n = 0;
      n += snprintf(str, L-n, "[");

      for (i = 0; i < ndim; i++) {
        if (n >= L) break;
        n += snprintf(str+n, L-n, "%" PRId64, size[i]);
        if (i < ndim-1) {
          n += snprintf(str+n, L-n, " x ");
        }
      }

      if (n < L - 2) {
        snprintf(str+n, L-n, "]");
      } else {
        snprintf(str+L-5, 5, "...]");
      }

      return buf;
        */
}
