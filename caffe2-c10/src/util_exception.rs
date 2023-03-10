crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/Exception.h]

/**
  | The primary ATen error class.
  |
  | Provides a complete error message with source
  | location information via `what()`, and a more
  | concise message via
  | `what_without_backtrace()`.
  |
  | Don't throw this directly; use
  | TORCH_CHECK/TORCH_INTERNAL_ASSERT instead.
  |
  | NB: C10ErrorData is handled specially by the default
  | torch to suppress the backtrace, see
  | torch/csrc/Exceptions.h
  |
  */
#[derive(Debug)]
pub struct C10ErrorData {

    /**
     | The actual error message.
     |
     */
    msg:                    String,

    /**
     | Context for the message (in order of
     | decreasing specificity). Context
     | will be automatically formatted appropriately,
     | so it is not necessary to add extra leading/trailing
     | newlines to strings inside this vector
     |
     */
    context:                Vec<String>,

    /**
     | The C++ backtrace at the point when this
     | exception was raised. This may be empty
     | if there is no valid backtrace. (We don't
     | use optional here to reduce the dependencies
     | this file has.)
     |
     */
    backtrace:              String,

    /**
     | These two are derived fields from msg_stack_
     | and backtrace_, but we need fields for
     | the strings so that we can return a const
     | char* (as the signature of exception
     | requires). Currently, the invariant
     | is that these fields are ALWAYS populated
     | consistently with respect to msg_stack_
     | and backtrace_.
     |
     */
    what:                   String,

    what_without_backtrace: String,

    /**
     | This is a little debugging trick: you
     | can stash a relevant pointer in caller,
     | and then when you catch the exception,
     | you can compare against pointers you
     | have on hand to get more information
     | about where the exception came from.
     | In Caffe2, this is used to figure out
     | which operator raised an exception.
     |
     */
    caller:                 *const c_void,
}

impl C10ErrorData {

    /**
      | PyTorch-style C10ErrorData constructor.  NB: the
      | implementation of this is actually in
      | Logging.cpp
      |
      */
    pub fn new(
        source_location: SourceLocation,
        msg:             String) -> Self {
    
        todo!();
        /*
        
        */
    }

    pub fn msg(&self) -> &String {
        
        todo!();
        /*
            return msg_;
        */
    }
    
    pub fn context(&self) -> &Vec<String> {
        
        todo!();
        /*
            return context_;
        */
    }
    
    pub fn backtrace(&self) -> &String {
        
        todo!();
        /*
            return backtrace_;
        */
    }

    /**
      | Returns the complete error message,
      | including the source location.
      |
      | The returned pointer is invalidated if you
      | call add_context() on this object.
      */
    pub fn what(&self) -> *const u8 {
        
        todo!();
        /*
            return what_.c_str();
        */
    }
    
    pub fn caller(&self)  {
        
        todo!();
        /*
            return caller_;
        */
    }

    /**
      | Returns only the error message string,
      | without source location.
      |
      | The returned pointer is invalidated if you
      | call add_context() on this object.
      */
    pub fn what_without_backtrace(&self) -> *const u8 {
        
        todo!();
        /*
            return what_without_backtrace_.c_str();
        */
    }
    
    pub fn new_b(
        msg:       String,
        backtrace: String,
        caller:    *const c_void) -> Self {
    
        todo!();
        /*


            : msg_(move(msg)), backtrace_(move(backtrace)), caller_(caller) 
      refresh_what();
        */
    }

    /**
      | PyTorch-style error message
      |
      | C10ErrorData::C10ErrorData(SourceLocation source_location,
      | const string& msg)
      |
      | NB: This is defined in Logging.cpp for access
      | to GetFetchStackTrace
      |
      | Caffe2-style error message
      |
      */
    pub fn new_c(
        file:      *const u8,
        line:      u32,
        condition: *const u8,
        msg:       &String,
        backtrace: &String,
        caller:    *const c_void) -> Self {
    
        todo!();
        /*
            : C10ErrorData(
              str("[enforce fail at ",
                  StripBasename(file),
                  ":",
                  line,
                  "] ",
                  condition,
                  ". ",
                  msg),
              backtrace,
              caller)
        */
    }
    
    pub fn compute_what(&self, include_backtrace: bool) -> String {
        
        todo!();
        /*
            ostringstream oss;

      oss << msg_;

      if (context_.size() == 1) {
        // Fold error and context in one line
        oss << " (" << context_[0] << ")";
      } else {
        for (const auto& c : context_) {
          oss << "\n  " << c;
        }
      }

      if (include_backtrace) {
        oss << "\n" << backtrace_;
      }

      return oss.str();
        */
    }
    
    pub fn refresh_what(&mut self)  {
        
        todo!();
        /*
            what_ = compute_what(/*include_backtrace*/ true);
      what_without_backtrace_ = compute_what(/*include_backtrace*/ false);
        */
    }
    
    /**
      | Add some new context to the message stack.
      | The last added context will be formatted at the
      | end of the context list upon printing.
      |
      | WARNING: This method is O(n) in the size of the
      | stack, so don't go wild adding a ridiculous
      | amount of context to error messages.
      |
      */
    pub fn add_context(&mut self, new_msg: String)  {
        
        todo!();
        /*
            context_.push_back(move(new_msg));
      // TODO: Calling add_context O(n) times has O(n^2) cost.  We can fix
      // this perf problem by populating the fields lazily... if this ever
      // actually is a problem.
      // NB: If you do fix this, make sure you do it in a thread safe way!
      // what() is almost certainly expected to be thread safe even when
      // accessed across multiple threads
      refresh_what();
        */
    }
    
    /**
      | PyTorch-style error message
      |
      | (This must be defined here for access to
      | GetFetchStackTrace)
      */
    pub fn new_a(
        source_location: SourceLocation,
        msg:             String) -> Self {
    
        todo!();
        /*


            : Error(
              move(msg),
              str("Exception raised from ",
                  source_location,
                  " (most recent call first):\n",
                  (*GetFetchStackTrace())()))
        */
    }
}

/**
  | Note: [Verbatim Warnings]
  |
  | Warnings originating in C++ code can appear
  | out-of-place to Python users: a user runs
  | a line in Python, but the warning references
  | a line in C++.
  |
  | Some parts of PyTorch, like the JIT, are
  | cognizant of this mismatch and take care to map
  | warnings back to the user's program, but most
  | of PyTorch simply throws a context-free
  | warning. To allow warning handlers to add
  | context where appropriate, warn takes the
  | "verbatim" flag.
  |
  | When this is false a warning handler might
  | append the C++ warning to a Python warning
  | message that relates the warning back to the
  | user's program. Callers who have already
  | accounted for context in their warnings should
  | set verbatim to true so their warnings appear
  | without modification.
  */
pub trait WarningHandlerInterface {

    /// The default warning handler. Prints the
    /// message to stderr.
    ///
    fn process(&mut self, 
        source_location: &SourceLocation,
        msg:             &String,
        verbatim:        bool)  {
        
        todo!();
        /*
            LOG_AT_FILE_LINE(WARNING, source_location.file, source_location.line)
          << "Warning: " << msg << " (function " << source_location.function << ")";
        */
    }
}

/**
  | A RAII guard that sets warn_always (not
  | thread-local) on construction, and sets it back
  | to the original value upon destruction.
  |
  */
pub struct WarnAlways {
    prev_setting: bool,
}

impl WarnAlways {
    
    pub fn new_from_setting(setting: Option<bool>) -> Self {

        let setting: bool = setting.unwrap_or(true);

        todo!();
        /*
        
        */
    }
}

/**
  | Used in ATen for out-of-bound indices that can
  | reasonably only be detected lazily inside
  | a kernel (See: advanced indexing).  These turn
  | into IndexError when they cross to Python.
  |
  */
pub struct IndexError {
    base: C10ErrorData,
}

/**
  | Used in ATen for invalid values. These
  | turn into ValueError when they cross
  | to Python.
  |
  */
pub struct ValueError {
    base: C10ErrorData,
}

/**
  | Used in ATen for invalid types. These
  | turn into TypeError when they cross
  | to Python.
  |
  */
pub struct TypeError {
    base: C10ErrorData,
}

/**
  | Used in ATen for functionality that is not
  | implemented.  These turn into
  | NotImplementedError when they cross to Python.
  |
  */
pub struct NotImplementedError {
    base: C10ErrorData,
}

/**
  | Used in ATen for non finite indices.  These
  | turn into ExitException when they cross to
  | Python.
  |
  */
pub struct EnforceFiniteError {
    base: C10ErrorData,
}

/**
  | Used in Onnxifi backend lowering. These
  | turn into ExitException when they cross to Python.
  |
  */
pub struct OnnxfiBackendSystemError {
    base: C10ErrorData,
}

/**
  | Private helper macro for implementing
  | TORCH_INTERNAL_ASSERT and TORCH_CHECK
  |
  | Note: In the debug build With MSVC, __LINE__
  | might be of long type (a.k.a int32_t), which is
  | different from the definition of
  | `SourceLocation` that requires unsigned int
  | (a.k.a uint32_t) and may cause a compile error
  | with the message: error C2397: conversion from
  | 'long' to 'uint32_t' requires a narrowing
  | conversion Here the static cast is used to pass
  | the build. if this is used inside a lambda the
  | __func__ macro expands to operator(), which
  | isn't very useful, but hard to fix in a macro
  | so suppressing the warning.
  |
  */
macro_rules! C10_THROW_ERROR {
    ($err_type:ident, $msg:ident) => {
        /*
        
          throw ::err_type(               
              {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, msg)
        */
    }
}

/**
  | Private helper macro for workaround MSVC
  | misexpansion of nested macro invocations
  | involving __VA_ARGS__.  See
  | https://stackoverflow.com/questions/5134523/msvc-doesnt-expand-va-args-correctly
  */
macro_rules! C10_EXPAND_MSVC_WORKAROUND {
    ($x:ident) => {
        /*
                x
        */
    }
}

/**
  | On nvcc, C10_UNLIKELY thwarts missing return
  | statement analysis.  In cases where the
  | unlikely expression may be a constant, use this
  | macro to ensure return statement analysis keeps
  | working (at the cost of not getting the
  | likely/unlikely annotation on
  | nvcc). https://github.com/pytorch/pytorch/issues/21418
  |
  | Currently, this is only used in the error
  | reporting macros below.  If you want to use it
  | more generally, move me to Macros.h
  |
  | TODO: Brian Vaughan observed that we might be
  | able to get this to work on nvcc by writing
  | some sort of C++ overload that distinguishes
  | constexpr inputs from non-constexpr.  Since
  | there isn't any evidence that losing
  | C10_UNLIKELY in nvcc is causing us perf
  | problems, this is not yet implemented, but this
  | might be an interesting piece of C++ code for
  | an intrepid bootcamper to write.
  */
#[cfg(__CUDACC__)]
macro_rules! C10_UNLIKELY_OR_CONST {
    ($e:ident) => {
        /*
                e
        */
    }
}

#[cfg(not(__CUDACC__))]
macro_rules! C10_UNLIKELY_OR_CONST {
    ($e:ident) => {
        /*
                C10_UNLIKELY(e)
        */
    }
}

#[cfg(STRIP_ERROR_MESSAGES)]
macro_rules! TORCH_RETHROW {
    ($e:ident, $($arg:ident),*) => {
        /*
                throw
        */
    }
}

#[cfg(not(STRIP_ERROR_MESSAGES))]
macro_rules! TORCH_RETHROW {
    ($e:ident, $($arg:ident),*) => {
        /*
        
          do {                                      
            e.add_context(::str(__VA_ARGS__)); 
            throw;                                  
          } while (false)
        */
    }
}

/**
  | A utility macro to provide assert()-like
  | functionality; that is, enforcement of internal
  | invariants in code.  It supports an arbitrary
  | number of extra arguments (evaluated only on
  | failure), which will be printed in the assert
  | failure message using operator<< (this is
  | useful to print some variables which may be
  | useful for debugging.)
  |
  | Usage:
  |    TORCH_INTERNAL_ASSERT(should_be_true);
  |    TORCH_INTERNAL_ASSERT(x == 0, "x = ", x);
  |
  | Assuming no bugs in PyTorch, the conditions
  | tested by this macro should always be true;
  | e.g., it should be possible to disable all of
  | these conditions without changing observable
  | user behavior.  If you would like to do error
  | reporting for user input, please use
  | TORCH_CHECK instead.
  |
  | NOTE: It is SAFE to use this macro in
  | production code; on failure, this simply raises
  | an exception, it does NOT unceremoniously quit
  | the process (unlike assert()).
  |
  */
#[cfg(STRIP_ERROR_MESSAGES)]
macro_rules! TORCH_INTERNAL_ASSERT {
    ($cond:ident, $($arg:ident),*) => {
        /*
        
          if (C10_UNLIKELY_OR_CONST(!(cond))) {                             
            ::torchCheckFail(                                  
                __func__,                                                   
                __FILE__,                                                   
                static_cast<uint32_t>(__LINE__),                            
                #cond "INTERNAL ASSERT FAILED at" C10_STRINGIZE(__FILE__)); 
          }
        */
    }
}

/**
  | It would be nice if we could build a combined
  | string literal out of the TORCH_INTERNAL_ASSERT
  | prefix and a user-provided string literal as
  | the first argument, but there doesn't seem to
  | be any good way to do that while still
  | supporting having a first argument that isn't
  | a string literal.
  |
  */
#[cfg(not(STRIP_ERROR_MESSAGES))]
macro_rules! TORCH_INTERNAL_ASSERT {
    ($cond:ident, $($arg:ident),*) => {
        /*
        
          if (C10_UNLIKELY_OR_CONST(!(cond))) {                                         
            ::torchInternalAssertFail(                                     
                __func__,                                                               
                __FILE__,                                                               
                static_cast<uint32_t>(__LINE__),                                        
                #cond                                                                   
                "INTERNAL ASSERT FAILED at " C10_STRINGIZE(__FILE__) ":" C10_STRINGIZE( 
                    __LINE__) ", please report a bug to PyTorch. ",                     
                str(__VA_ARGS__));                                                 
          }
        */
    }
}

/**
  | A utility macro to make it easier to test for
  | error conditions from user input.
  |
  | Like TORCH_INTERNAL_ASSERT, it supports an
  | arbitrary number of extra arguments (evaluated
  | only on failure), which will be printed in the
  | error message using operator<< (e.g., you can
  | pass any object which has operator<< defined.
  |
  | Most objects in PyTorch have these
  | definitions!)
  |
  | Usage:
  |    TORCH_CHECK(should_be_true); // A default error message will be provided
  |                                 // in this case; but we recommend writing an
  |                                 // explicit error message, as it is more
  |                                 // user friendly.
  |    TORCH_CHECK(x == 0, "Expected x to be 0, but got ", x);
  |
  | On failure, this macro will raise an exception.
  | If this exception propagates to Python, it will
  | convert into a Python RuntimeError.
  |
  | NOTE: It is SAFE to use this macro in
  | production code; on failure, this simply raises
  | an exception, it does NOT unceremoniously quit
  | the process (unlike CHECK() from glog.)
  |
  */
macro_rules! TORCH_CHECK_WITH {
    ($error_t:ident, $cond:ident, $($arg:ident),*) => {
        /*
        
          TORCH_CHECK_WITH_MSG(error_t, cond, "", __VA_ARGS__)
        */
    }
}


#[cfg(STRIP_ERROR_MESSAGES)]
macro_rules! TORCH_CHECK_MSG {
    ($cond:ident, $type:ident, $($arg:ident),*) => {
        /*
        
          (#cond #type " CHECK FAILED at " C10_STRINGIZE(__FILE__))
        */
    }
}

#[cfg(STRIP_ERROR_MESSAGES)]
macro_rules! TORCH_CHECK_WITH_MSG {
    ($error_t:ident, $cond:ident, $type:ident, $($arg:ident),*) => {
        /*
        
          if (C10_UNLIKELY_OR_CONST(!(cond))) {                               
            C10_THROW_ERROR(C10ErrorData, TORCH_CHECK_MSG(cond, type, __VA_ARGS__)); 
          }
        */
    }
}

#[cfg(not(STRIP_ERROR_MESSAGES))]
pub fn torch_check_msg_impl<'a,Args>(
    msg:  &'a str,
    args: &Args) -> &'a str {

    #[cfg(not(STRIP_ERROR_MESSAGES))]
    #[inline] pub fn torch_check_msg_impl_a(msg: *const u8) -> *const u8 {
        
        todo!();
            /*
                return msg;
            */
    }

    /// If there is just 1 user-provided C-string
    /// argument, use it.
    ///
    #[cfg(not(STRIP_ERROR_MESSAGES))]
    #[inline] pub fn torch_check_msg_impl_b(
            msg:  *const u8,
            args: *const u8) -> *const u8 {
        
        todo!();
            /*
                return args;
            */
    }

    todo!();
        /*
            return ::str(args...);
        */

}

#[cfg(not(STRIP_ERROR_MESSAGES))]
macro_rules! TORCH_CHECK_MSG {
    ($cond:ident, $type:ident, $($arg:ident),*) => {
        /*
        
          (::torchCheckMsgImpl(                       
              "Expected " #cond                                    
              " to be true, but got false.  "                      
              "(Could this error message be improved?  If so, "    
              "please report an enhancement request to PyTorch.)", 
              ##__VA_ARGS__))
        */
    }
}

#[cfg(not(STRIP_ERROR_MESSAGES))]
macro_rules! TORCH_CHECK_WITH_MSG {
    ($error_t:ident, $cond:ident, $type:ident, $($arg:ident),*) => {
        /*
        
          if (C10_UNLIKELY_OR_CONST(!(cond))) {                                 
            C10_THROW_ERROR(error_t, TORCH_CHECK_MSG(cond, type, __VA_ARGS__)); 
          }
        */
    }
}

#[cfg(STRIP_ERROR_MESSAGES)]
macro_rules! TORCH_CHECK {
    ($cond:ident, $($arg:ident),*) => {
        /*
        
          if (C10_UNLIKELY_OR_CONST(!(cond))) {          
            ::torchCheckFail(               
                __func__,                                
                __FILE__,                                
                static_cast<uint32_t>(__LINE__),         
                TORCH_CHECK_MSG(cond, "", __VA_ARGS__)); 
          }
        */
    }
}

#[cfg(not(STRIP_ERROR_MESSAGES))]
macro_rules! TORCH_CHECK {
    ($cond:ident, $($arg:ident),*) => {
        /*
        
          if (C10_UNLIKELY_OR_CONST(!(cond))) {            
            ::torchCheckFail(                 
                __func__,                                  
                __FILE__,                                  
                static_cast<uint32_t>(__LINE__),           
                TORCH_CHECK_MSG(cond, "", ##__VA_ARGS__)); 
          }
        */
    }
}

/**
  | An utility macro that does what `TORCH_CHECK`
  | does if compiled in the host code, otherwise
  | does nothing. Supposed to be used in the code
  | shared between host and device code as an
  | alternative for `TORCH_CHECK`.
  |
  */
#[cfg(any(__CUDACC__,__HIPCC__))]
macro_rules! TORCH_CHECK_IF_NOT_ON_CUDA { ($cond:ident, $($arg:ident),*) => { } }

#[cfg(not(any(__CUDACC__,__HIPCC__)))]
macro_rules! TORCH_CHECK_IF_NOT_ON_CUDA {
    ($cond:ident, $($arg:ident),*) => {
        /*
                TORCH_CHECK(cond, ##__VA_ARGS__)
        */
    }
}

/**
  | Debug only version of
  | TORCH_INTERNAL_ASSERT. This macro only checks
  | in debug build, and does nothing in release
  | build.  It is appropriate to use in situations
  | where you want to add an assert to a hotpath,
  | but it is too expensive to run this assert on
  | production builds.
  |
  */

/// Optimized version - generates no code.
#[cfg(NDEBUG)]
macro_rules! TORCH_INTERNAL_ASSERT_DEBUG_ONLY {
    ($($arg:ident),*) => {
        /*
        
          while (false)                               
          C10_EXPAND_MSVC_WORKAROUND(TORCH_INTERNAL_ASSERT(__VA_ARGS__))
        */
    }
}

#[cfg(not(NDEBUG))]
macro_rules! TORCH_INTERNAL_ASSERT_DEBUG_ONLY {
    ($($arg:ident),*) => {
        /*
        
          C10_EXPAND_MSVC_WORKAROUND(TORCH_INTERNAL_ASSERT(__VA_ARGS__))
        */
    }
}


// TODO: We're going to get a lot of similar
// looking string literals this way; check if this
// actually affects binary size.

/**
  | Like TORCH_CHECK, but raises IndexErrors
  | instead of Errors.
  |
  */
macro_rules! TORCH_CHECK_INDEX {
    ($cond:ident, $($arg:ident),*) => {
        /*
        
          TORCH_CHECK_WITH_MSG(IndexError, cond, "INDEX", __VA_ARGS__)
        */
    }
}

/**
  | Like TORCH_CHECK, but raises ValueErrors
  | instead of Errors.
  |
  */
macro_rules! TORCH_CHECK_VALUE {
    ($cond:ident, $($arg:ident),*) => {
        /*
        
          TORCH_CHECK_WITH_MSG(ValueError, cond, "VALUE", __VA_ARGS__)
        */
    }
}

/**
  | Like TORCH_CHECK, but raises TypeErrors
  | instead of Errors.
  |
  */
macro_rules! TORCH_CHECK_TYPE {
    ($cond:ident, $($arg:ident),*) => {
        /*
        
          TORCH_CHECK_WITH_MSG(TypeError, cond, "TYPE", __VA_ARGS__)
        */
    }
}

/**
  | Like TORCH_CHECK, but raises
  | NotImplementedErrors instead of Errors.
  |
  */
macro_rules! TORCH_CHECK_NOT_IMPLEMENTED {
    ($cond:ident, $($arg:ident),*) => {
        /*
        
          TORCH_CHECK_WITH_MSG(NotImplementedError, cond, "TYPE", __VA_ARGS__)
        */
    }
}

/**
  | Report a warning to the user.  Accepts an
  | arbitrary number of extra arguments which are
  | concatenated into the warning message using
  | operator<<
  |
  */
#[cfg(STRIP_ERROR_MESSAGES)]
macro_rules! TORCH_WARN {
    ($($arg:ident),*) => {
        /*
        
          ::Warning::warn(                                      
              {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, 
              ::CompileTimeEmptyString{},               
              false)
        */
    }
}

#[cfg(not(STRIP_ERROR_MESSAGES))]
macro_rules! TORCH_WARN {
    (, $($arg:ident),*) => {
        /*
        
          ::Warning::warn(                                      
              {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, 
              ::str(__VA_ARGS__),                               
              false)
        */
    }
}

/**
  | Report a warning to the user only once.
  | Accepts an arbitrary number of extra arguments
  | which are concatenated into the warning message
  | using operator<<
  |
  */
#[cfg(STRIP_ERROR_MESSAGES)]
macro_rules! _TORCH_WARN_ONCE {
    ($($arg:ident),*) => {
        /*
        
           static const auto C10_ANONYMOUS_VARIABLE(torch_warn_once_) = 
              [&] {                                                               
                ::Warning::warn(                                             
                    {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},        
                    ::CompileTimeEmptyString{},                      
                    false);                                                       
                return true;                                                      
              }()
        */
    }
}

#[cfg(not(STRIP_ERROR_MESSAGES))]
macro_rules! _TORCH_WARN_ONCE {
    ($($arg:ident),*) => {
        /*
        
           static const auto C10_ANONYMOUS_VARIABLE(torch_warn_once_) = 
              [&] {                                                               
                ::Warning::warn(                                             
                    {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},        
                    ::str(__VA_ARGS__),                                      
                    false);                                                       
                return true;                                                      
              }()
        */
    }
}

macro_rules! TORCH_WARN_ONCE {
    ($($arg:ident),*) => {
        /*
        
          if (::Warning::get_warnAlways()) { 
            TORCH_WARN(__VA_ARGS__);              
          } else {                                
            _TORCH_WARN_ONCE(__VA_ARGS__);        
          }
        */
    }
}


// ----------------------------------------------------------------------------
// Deprecated macros
// ----------------------------------------------------------------------------

/**
  | // Deprecation disabled until we fix
  | sites in our codebase
  | 
  | C10_DEPRECATED_MESSAGE("AT_ERROR(msg)
  | is deprecated, use TORCH_CHECK(false,
  | msg) instead.")
  |
  */
#[inline] pub fn deprecated_at_error()  {
    
    todo!();
        /*
        
        */
}

/*
// Deprecation disabled until we fix sites in our codebase
C10_DEPRECATED_MESSAGE("AT_ASSERT is deprecated, if you mean to indicate an
internal invariant failure, use " \
                       "TORCH_INTERNAL_ASSERT instead; if you mean to do user
error checking, use " \ "TORCH_CHECK.  See
https://github.com/pytorch/pytorch/issues/20287 for more details.")
*/
#[inline] pub fn deprecated_at_assert()  {
    
    todo!();
        /*
        
        */
}

/*
// Deprecation disabled until we fix sites in our codebase
C10_DEPRECATED_MESSAGE("AT_ASSERTM is deprecated, if you mean to indicate an
internal invariant failure, use " \
                       "TORCH_INTERNAL_ASSERT instead; if you mean to do user
error checking, use " \ "TORCH_CHECK.  See
https://github.com/pytorch/pytorch/issues/20287 for more details.")
*/
#[inline] pub fn deprecated_at_assertm()  {
    
    todo!();
        /*
        
        */
}

/**
  | Deprecated alias; this alias was deprecated
  | because people kept mistakenly using it for
  | user error checking.  Use TORCH_INTERNAL_ASSERT
  | or TORCH_CHECK instead. See
  | https://github.com/pytorch/pytorch/issues/20287
  | for more details.
  |
  */
macro_rules! AT_ASSERT {
    ($($arg:ident),*) => {
        /*
        
          do {                                                              
            ::deprecated_AT_ASSERT();                          
            C10_EXPAND_MSVC_WORKAROUND(TORCH_INTERNAL_ASSERT(__VA_ARGS__)); 
          } while (false)
        */
    }
}

/**
  | Deprecated alias, like AT_ASSERT.  The new
  | TORCH_INTERNAL_ASSERT macro supports both 0-ary
  | and variadic calls, so having a separate
  | message-accepting macro is not necessary.
  |
  | NB: we MUST include cond explicitly here, as
  | MSVC will miscompile the macro expansion,
  | shunting all of __VA_ARGS__ to cond.  An
  | alternate workaround can be seen at
  | https://stackoverflow.com/questions/5134523/msvc-doesnt-expand-va-args-correctly
  |
  */
macro_rules! AT_ASSERTM {
    ($cond:ident, $($arg:ident),*) => {
        /*
        
          do {                                                                    
            ::deprecated_AT_ASSERTM();                               
            C10_EXPAND_MSVC_WORKAROUND(TORCH_INTERNAL_ASSERT(cond, __VA_ARGS__)); 
          } while (false)
        */
    }
}

/**
  | Deprecated alias; this alias was deprecated
  | because it represents extra API surface that
  | makes it hard for people to understand what
  | macro to use.
  |
  | Use TORCH_CHECK(false, ...) or
  | TORCH_INTERNAL_ASSERT(false, ...) to
  | unconditionally fail at a line of code.
  |
  */
macro_rules! AT_ERROR {
    ($($arg:ident),*) => {
        /*
        
          do {                                                                       
            ::deprecated_AT_ERROR();                                    
            C10_EXPAND_MSVC_WORKAROUND(TORCH_CHECK(false, ::str(__VA_ARGS__))); 
          } while (false)
        */
    }
}


//-------------------------------------------[.cpp/pytorch/c10/util/Exception.cpp]


pub fn torch_check_fail_a(
    func: *const u8,
    file: *const u8,
    line: u32,
    msg:  &String)  {

    todo!();
        /*
            throw ::C10ErrorData({func, file, line}, msg);
        */
}

pub fn torch_check_fail_b(
    func: *const u8,
    file: *const u8,
    line: u32,
    msg:  *const u8)  {
    
    todo!();
        /*
            throw ::C10ErrorData({func, file, line}, msg);
        */
}

/**
  | The str() call that creates userMsg can have
  | 1 of 3 return types depending on the number and
  | types of arguments passed to
  | TORCH_INTERNAL_ASSERT.
  |
  | 0 arguments will get a CompileTimeEmptyString,
  | 1 const char * will be passed straight through,
  | and anything else will get converted to string.
  |
  */
pub fn torch_internal_assert_fail_a(
        func:     *const u8,
        file:     *const u8,
        line:     u32,
        cond_msg: *const u8,
        user_msg: *const u8)  {
    
    todo!();
        /*
            torchCheckFail(func, file, line, str(condMsg, userMsg));
        */
}

/**
  | This should never be called. It is provided in
  | case of compilers that don't do any dead code
  | stripping in debug builds.
  |
  */
pub fn torch_internal_assert_fail_b(
    func:     *const u8,
    file:     *const u8,
    line:     u32,
    cond_msg: *const u8,
    user_msg: &String)  {

    todo!();
    /*
            torchCheckFail(func, file, line, str(condMsg, userMsg));
        */
}

pub fn get_base_handler() -> *mut dyn WarningHandlerInterface {
    
    todo!();
        /*
            static WarningHandler base_warning_handler_ = WarningHandler();
      return &base_warning_handler_;
        */
}

pub struct ThreadWarningHandler {

}

pub mod thread_warning_handler {
    use super::*;

    lazy_static!{
        /*
        static thread_local WarningHandler* warning_handler_;
        thread_local WarningHandler* ThreadWarningHandler::warning_handler_ = nullptr;
        */
    }
}

impl ThreadWarningHandler {
    
    pub fn get_handler() -> *mut dyn WarningHandlerInterface {
        
        todo!();
        /*
            if (!warning_handler_) {
          warning_handler_ = getBaseHandler();
        }
        return warning_handler_;
        */
    }
    
    pub fn set_handler(handler: *mut dyn WarningHandlerInterface)  {
        
        todo!();
        /*
            warning_handler_ = handler;
        */
    }
}

/**
  | Issue a warning with a given message.
  | Dispatched to the current warning handler.
  |
  */
pub fn warn_a(
        source_location: &SourceLocation,
        msg:             &String,
        verbatim:        bool)  {
    
    todo!();
        /*
            ThreadWarningHandler::get_handler()->process(source_location, msg, verbatim);
        */
}

pub fn warn_b(
    source_location: SourceLocation,
    msg:             CompileTimeEmptyString,
    verbatim:        bool)  {
    
    todo!();
        /*
            warn(source_location, "", verbatim);
        */
}

pub fn warn_c(
        source_location: SourceLocation,
        msg:             *const u8,
        verbatim:        bool)  {
    
    todo!();
        /*
            ThreadWarningHandler::get_handler()->process(source_location, msg, verbatim);
        */
}

/**
  | Sets the global warning handler. This is not
  | thread-safe, so it should generally be called
  | once during initialization or while holding
  | the GIL for programs that use python.
  |
  | User is responsible for keeping the
  | WarningHandler alive until it is not needed.
  |
  */
pub fn set_warning_handler(handler: *mut dyn WarningHandlerInterface)  {
    
    todo!();
        /*
            ThreadWarningHandler::set_handler(handler);
        */
}

/// Gets the global warning handler.
pub fn get_warning_handler() -> *mut dyn WarningHandlerInterface {
    
    todo!();
        /*
            return ThreadWarningHandler::get_handler();
        */
}

lazy_static!{
    /*
    bool warn_always = false;
    */
}

/**
  | The TORCH_WARN_ONCE macro is difficult to test
  | for. Use setWarnAlways(true) to turn it into
  | TORCH_WARN, which can be tested for more
  | easily.
  |
  */
pub fn set_warn_always(setting: bool)  {
    
    todo!();
        /*
            warn_always = setting;
        */
}

pub fn get_warn_always() -> bool {
    
    todo!();
        /*
            return warn_always;
        */
}

impl WarnAlways {
    
    pub fn new(setting: Option<bool>) -> Self {

        let setting: bool = setting.unwrap_or(true);

        todo!();
        /*


            : prev_setting(get_warnAlways()) 

      set_warnAlways(setting);
        */
    }
}

impl Drop for WarnAlways {

    fn drop(&mut self) {

        todo!();
        /*
            set_warnAlways(prev_setting);
        */
    }
}

/**
  | A utility function to return an exception
  | string by prepending its exception type before
  | its what() content
  |
  */
pub fn get_exception_string(e: &dyn Error) -> String {
    
    todo!();
        /*
            #ifdef __GXX_RTTI
      return demangle(typeid(e).name()) + ": " + e.what();
    #else
      return string("Exception (no RTTI available): ") + e.what();
    #endif // __GXX_RTTI
        */
}
