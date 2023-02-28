crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/Logging.h]

/**
  | CAFFE2_LOG_THRESHOLD is a compile time flag
  | that would allow us to turn off logging at
  | compile time so no logging message below that
  | level is produced at all. The value should be
  | between INT_MIN and CAFFE_FATAL.
  |
  | If we have not defined the compile time
  | log threshold, we keep all the log cases.
  |
  */
#[cfg(not(CAFFE2_LOG_THRESHOLD))]
pub const CAFFE2_LOG_THRESHOLD: i32 = i32::MIN;

/**
  | Some versions of GLOG support less-spammy
  | version of LOG_EVERY_MS. If it's not available
  | - just short-circuit to the always working one
  | one.
  |
  | We define the C10_ name to avoid confusing
  | other files
  |
  */
#[cfg(LOG_EVERY_MS)]
macro_rules! C10_LOG_EVERY_MS {
    ($severity:ident, $ms:ident) => {
        /*
                LOG_EVERY_MS(severity, ms)
        */
    }
}


#[cfg(not(LOG_EVERY_MS))]
macro_rules! C10_LOG_EVERY_MS {
    ($severity:ident, $ms:ident) => {
        /*
                LOG(severity)
        */
    }
}

// Same for LOG_FIRST_N
#[cfg(LOG_FIRST_N)]
macro_rules! C10_LOG_FIRST_N {
    ($severity:ident, $n:ident) => {
        /*
                LOG_FIRST_N(severity, n)
        */
    }
}

#[cfg(not(LOG_FIRST_N))]
macro_rules! C10_LOG_FIRST_N {
    ($severity:ident, $n:ident) => {
        /*
                LOG(severity)
        */
    }
}

// Same for LOG_EVERY_N
#[cfg(LOG_EVERY_N)]
macro_rules! C10_LOG_EVERY_N {
    ($severity:ident, $n:ident) => {
        /*
                LOG_EVERY_N(severity, n)
        */
    }
}

#[cfg(not(LOG_EVERY_N))]
macro_rules! C10_LOG_EVERY_N {
    ($severity:ident, $n:ident) => {
        /*
                LOG(severity)
        */
    }
}

#[inline] pub fn throw_enforce_not_met_a(
        file:      *const u8,
        line:      i32,
        condition: *const u8,
        msg:       CompileTimeEmptyString,
        caller:    Option<*const c_void>)  {

    todo!();
        /*
            ThrowEnforceNotMet(file, line, condition, "", caller);
        */
}

pub fn throw_enforce_finite_not_met_a(
        file:      *const u8,
        line:      i32,
        condition: *const u8,
        msg:       &String,
        caller:    *const c_void)  {
    
    todo!();
        /*
            throw EnforceFiniteError(
          file, line, condition, msg, (*GetFetchStackTrace())(), caller);
        */
}

pub fn throw_enforce_finite_not_met_b(
        file:      *const u8,
        line:      i32,
        condition: *const u8,
        msg:       *const u8,
        caller:    *const c_void)  {
    
    todo!();
        /*
            ThrowEnforceFiniteNotMet(file, line, condition, string(msg), caller);
        */
}

#[inline] pub fn throw_enforce_finite_not_met_c(
        file:      *const u8,
        line:      i32,
        condition: *const u8,
        msg:       CompileTimeEmptyString,
        caller:    Option<*const c_void>)  {

    todo!();
        /*
            ThrowEnforceFiniteNotMet(file, line, condition, "", caller);
        */
}

pub fn is_using_google_logging() -> bool {
    
    todo!();
        /*
            #ifdef C10_USE_GLOG
      return true;
    #else
      return false;
    #endif
        */
}

#[derive(Debug,Error)]
pub enum EnforceNotMet { 
    Default
}

macro_rules! CAFFE_ENFORCE {
    ($condition:ident, $($arg:ident),*) => {
        /*
        
          do {                                                              
            if (C10_UNLIKELY(!(condition))) {                               
              ::ThrowEnforceNotMet(                                    
                  __FILE__, __LINE__, #condition, ::str(__VA_ARGS__)); 
            }                                                               
          } while (false)
        */
    }
}

macro_rules! CAFFE_ENFORCE_FINITE {
    ($condition:ident, $($arg:ident),*) => {
        /*
        
          do {                                                              
            if (C10_UNLIKELY(!(condition))) {                               
              ::ThrowEnforceFiniteNotMet(                              
                  __FILE__, __LINE__, #condition, ::str(__VA_ARGS__)); 
            }                                                               
          } while (false)
        */
    }
}

macro_rules! CAFFE_ENFORCE_WITH_CALLER {
    ($condition:ident, $($arg:ident),*) => {
        /*
        
          do {                                                                    
            if (C10_UNLIKELY(!(condition))) {                                     
              ::ThrowEnforceNotMet(                                          
                  __FILE__, __LINE__, #condition, ::str(__VA_ARGS__), this); 
            }                                                                     
          } while (false)
        */
    }
}

macro_rules! CAFFE_THROW {
    ($($arg:ident),*) => {
        /*
        
          ::ThrowEnforceNotMet(__FILE__, __LINE__, "", ::str(__VA_ARGS__))
        */
    }
}

/**
 | Rich logging messages
 |
 | CAFFE_ENFORCE_THAT can be used with one of the
 |  "checker functions" that capture input argument
 |  values and add it to the exception
 |  message. E.g. `CAFFE_ENFORCE_THAT(Equals(foo(x),
 |  bar(y)), "Optional additional message")` would
 |  evaluate both foo and bar only once and if the
 |  results are not equal - include them in the
 |  exception message.
 |
 | Some of the basic checker functions like Equals
 |  or Greater are already defined below. Other
 |  header might define customized checkers by
 |  adding functions to enforce_detail
 |  namespace. For example:
 |
 |   namespace caffe2 { namespace enforce_detail {
 |   inline EnforceFailMessage IsVector(const vector<int64_t>& shape) {
 |     if (shape.size() == 1) { return EnforceOK(); }
 |     return str("Shape ", shape, " is not a vector");
 |   }
 |   }}
 |
 | With further usages like
 | `CAFFE_ENFORCE_THAT(IsVector(Input(0).dims()))`
 |
 | Convenient wrappers for binary operations like
 |  CAFFE_ENFORCE_EQ are provided too. Please use
 |  them instead of CHECK_EQ and friends for
 |  failures in user-provided input.
 */
pub fn enforce_fail_msg_impl_a<T1, T2>(x: &T1, y: &T2) -> String {

    todo!();
        /*
            return str(x, " vs ", y);
        */
}

pub fn enforce_fail_msg_impl_b<T1, T2, Args>(
        x:    &T1,
        y:    &T2,
        args: &Args) -> String {

    todo!();
        /*
            return str(x, " vs ", y, ". ", args...);
        */
}

pub fn enforce_that_impl<Pred, T1, T2, Args>(
        p:      Pred,
        lhs:    &T1,
        rhs:    &T2,
        file:   *const u8,
        line:   i32,
        expr:   *const u8,
        caller: *const c_void,
        args:   &Args)  {

    todo!();
        /*
            if (C10_UNLIKELY(!(p(lhs, rhs)))) {
        ::ThrowEnforceNotMet(
            file,
            line,
            expr,
            ::enforce_detail::enforceFailMsgImpl(lhs, rhs, args...),
            caller);
      }
        */
}

macro_rules! CAFFE_ENFORCE_THAT_IMPL {
    ($op:ident, $lhs:ident, $rhs:ident, $expr:ident, $($arg:ident),*) => {
        /*
        
          ::enforce_detail::enforceThatImpl(                
              op, lhs, rhs, __FILE__, __LINE__, expr, nullptr, ##__VA_ARGS__)
        */
    }
}

macro_rules! CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER {
    ($op:ident, $lhs:ident, $rhs:ident, $expr:ident, $($arg:ident),*) => {
        /*
        
          ::enforce_detail::enforceThatImpl(                            
              op, (lhs), (rhs), __FILE__, __LINE__, expr, this, ##__VA_ARGS__)
        */
    }
}

macro_rules! CAFFE_ENFORCE_THAT {
    ($cmp:ident, $op:ident, $lhs:ident, $rhs:ident, $($arg:ident),*) => {
        /*
        
          CAFFE_ENFORCE_THAT_IMPL(cmp, lhs, rhs, #lhs " " #op " " #rhs, ##__VA_ARGS__)
        */
    }
}

macro_rules! CAFFE_ENFORCE_BINARY_OP {
    ($cmp:ident, $op:ident, $x:ident, $y:ident, $($arg:ident),*) => {
        /*
        
          CAFFE_ENFORCE_THAT_IMPL(cmp, x, y, #x " " #op " " #y, ##__VA_ARGS__)
        */
    }
}


macro_rules! CAFFE_ENFORCE_EQ {
    ($x:ident, $y:ident, $($arg:ident),*) => {
        /*
        
          CAFFE_ENFORCE_BINARY_OP(equal_to<void>(), ==, x, y, ##__VA_ARGS__)
        */
    }
}


macro_rules! CAFFE_ENFORCE_NE {
    ($x:ident, $y:ident, $($arg:ident),*) => {
        /*
        
          CAFFE_ENFORCE_BINARY_OP(not_equal_to<void>(), !=, x, y, ##__VA_ARGS__)
        */
    }
}


macro_rules! CAFFE_ENFORCE_LE {
    ($x:ident, $y:ident, $($arg:ident),*) => {
        /*
        
          CAFFE_ENFORCE_BINARY_OP(less_equal<void>(), <=, x, y, ##__VA_ARGS__)
        */
    }
}


macro_rules! CAFFE_ENFORCE_LT {
    ($x:ident, $y:ident, $($arg:ident),*) => {
        /*
        
          CAFFE_ENFORCE_BINARY_OP(less<void>(), <, x, y, ##__VA_ARGS__)
        */
    }
}

macro_rules! CAFFE_ENFORCE_GE {
    ($x:ident, $y:ident, $($arg:ident),*) => {
        /*
        
          CAFFE_ENFORCE_BINARY_OP(greater_equal<void>(), >=, x, y, ##__VA_ARGS__)
        */
    }
}

macro_rules! CAFFE_ENFORCE_GT {
    ($x:ident, $y:ident, $($arg:ident),*) => {
        /*
        
          CAFFE_ENFORCE_BINARY_OP(greater<void>(), >, x, y, ##__VA_ARGS__)
        */
    }
}

macro_rules! CAFFE_ENFORCE_BINARY_OP_WITH_CALLER {
    ($cmp:ident, $op:ident, $x:ident, $y:ident, $($arg:ident),*) => {
        /*
        
          CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(                          
              cmp, x, y, #x " " #op " " #y, ##__VA_ARGS__)
        */
    }
}

macro_rules! CAFFE_ENFORCE_EQ_WITH_CALLER {
    ($x:ident, $y:ident, $($arg:ident),*) => {
        /*
        
          CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(          
              equal_to<void>(), ==, x, y, ##__VA_ARGS__)
        */
    }
}

macro_rules! CAFFE_ENFORCE_NE_WITH_CALLER {
    ($x:ident, $y:ident, $($arg:ident),*) => {
        /*
        
          CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(          
              not_equal_to<void>(), !=, x, y, ##__VA_ARGS__)
        */
    }
}

macro_rules! CAFFE_ENFORCE_LE_WITH_CALLER {
    ($x:ident, $y:ident, $($arg:ident),*) => {
        /*
        
          CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(          
              less_equal<void>(), <=, x, y, ##__VA_ARGS__)
        */
    }
}

macro_rules! CAFFE_ENFORCE_LT_WITH_CALLER {
    ($x:ident, $y:ident, $($arg:ident),*) => {
        /*
        
          CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(less<void>(), <, x, y, ##__VA_ARGS__)
        */
    }
}

macro_rules! CAFFE_ENFORCE_GE_WITH_CALLER {
    ($x:ident, $y:ident, $($arg:ident),*) => {
        /*
        
          CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(          
              greater_equal<void>(), >=, x, y, ##__VA_ARGS__)
        */
    }
}

macro_rules! CAFFE_ENFORCE_GT_WITH_CALLER {
    ($x:ident, $y:ident, $($arg:ident),*) => {
        /*
        
          CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(          
              greater<void>(), >, x, y, ##__VA_ARGS__)
        */
    }
}

/**
 | Very lightweight logging for the first time API
 | usage. It's beneficial for tracking of
 | individual functionality usage in larger
 | applications.
 |
 | In order to ensure light-weightedness of
 |  logging, we utilize static variable trick
 |  - LogAPIUsage will be invoked only once and
 |  further invocations will just do an atomic
 |  check.
 |
 | Example:
 |   // Logs caller info with an arbitrary text event, if there is a usage.
 |   C10_LOG_API_USAGE_ONCE("my_api");
 */
macro_rules! C10_LOG_API_USAGE_ONCE {
    ($($arg:ident),*) => {
        /*
        
           static bool C10_ANONYMOUS_VARIABLE(logFlag) = 
              ::LogAPIUsageFakeReturn(__VA_ARGS__);
        */
    }
}

/**
  | PyTorch ddp usage logging capabilities
  | DDPLoggingData holds data that can be logged in
  | applications for analysis and debugging. Data
  | structure is defined in c10 directory so that
  | it can be easily imported by both c10 and torch
  | files.
  |
  */
pub struct DDPLoggingData {

    /**
     | logging fields that are string types.
     |
     */
    strs_map: HashMap<String,String>,

    /**
     | logging fields that are int64_t types.
     |
     */
    ints_map: HashMap<String,i64>,
}

//-------------------------------------------[.cpp/pytorch/c10/util/Logging.cpp]

pub fn get_fetch_stack_trace() -> *mut fn() -> String {
    
    todo!();
        /*
            static function<string(void)> func = []() {
        return get_backtrace(/*frames_to_skip=*/1);
      };
      return &func;
        */
}

pub fn set_stack_trace_fetcher(fetcher: fn() -> String)  {
    
    todo!();
        /*
            *GetFetchStackTrace() = fetcher;
        */
}

pub fn throw_enforce_not_met_b(
        file:      *const u8,
        line:      i32,
        condition: *const u8,
        msg:       &String,
        caller:    *const c_void)  {
    
    todo!();
        /*
            Error e(file, line, condition, msg, (*GetFetchStackTrace())(), caller);
      if (FLAGS_caffe2_use_fatal_for_enforce) {
        LOG(FATAL) << e.msg();
      }
      throw e;
        */
}

pub fn throw_enforce_not_met_c(
        file:      *const u8,
        line:      i32,
        condition: *const u8,
        msg:       *const u8,
        caller:    *const c_void)  {
    
    todo!();
        /*
            ThrowEnforceNotMet(file, line, condition, string(msg), caller);
        */
}


pub type APIUsageLoggerType = fn(_0: &String) -> ();
pub type DDPUsageLoggerType = fn(_0: &DDPLoggingData) -> ();

pub fn is_api_usage_debug_mode() -> bool {
    
    todo!();
        /*
            const char* val = getenv("PY_USAGE_STDERR");
      return val && *val; // any non-empty value
        */
}

pub fn api_usage_debug(event: &String)  {
    
    todo!();
        /*
            // use stderr to avoid messing with glog
      cerr << "PY_USAGE " << event << endl;
        */
}

pub fn get_api_usage_logger() -> *mut APIUsageLoggerType {
    
    todo!();
        /*
            static APIUsageLoggerType func =
          IsAPIUsageDebugMode() ? &APIUsageDebug : [](const string&) {};
      return &func;
        */
}

pub fn get_ddp_usage_logger() -> *mut DDPUsageLoggerType {
    
    todo!();
        /*
            static DDPUsageLoggerType func = [](const DDPLoggingData&) {};
      return &func;
        */
}

/// API usage logging capabilities
pub fn set_api_usage_logger(logger: fn(_0: &String) -> ())  {
    
    todo!();
        /*
            TORCH_CHECK(logger);
      *GetAPIUsageLogger() = logger;
        */
}

pub fn set_py_torch_ddp_usage_logger(logger: fn(_0: &DDPLoggingData) -> ())  {
    
    todo!();
        /*
            TORCH_CHECK(logger);
      *GetDDPUsageLogger() = logger;
        */
}

pub fn log_api_usage(event: &String)  {
    
    todo!();
        /*
            try {
      if (auto logger = GetAPIUsageLogger())
        (*logger)(event);
    } catch (bad_function_call&) 

      // static destructor race
        */
}

pub fn log_py_torch_ddp_usage(ddp_data: &DDPLoggingData)  {
    
    todo!();
        /*
            try {
      if (auto logger = GetDDPUsageLogger())
        (*logger)(ddpData);
    } catch (bad_function_call&) 

      // static destructor race
        */
}

/**
  | Return value is needed to do the static
  | variable initialization trick
  |
  */
pub fn log_api_usage_fake_return(event: &String) -> bool {
    
    todo!();
        /*
            try {
      if (auto logger = GetAPIUsageLogger())
        (*logger)(event);
      return true;
    } catch (bad_function_call&)

      // static destructor race
      return true;
        */
}


/**
  | Google glog's api does not have an external
  | function that allows one to check if glog is
  | initialized or not.
  |
  | It does have an internal function - so we are
  | declaring it here. This is a hack but has been
  | used by a bunch of others too (e.g. Torch).
  |
  */
#[cfg(C10_USE_GLOG)]
pub fn is_google_logging_initialized() -> bool {
    
    todo!();
        /*
        
        */
}



#[cfg(C10_USE_GLOG)]
pub fn init_caffe_logging(
        argc: *mut i32,
        argv: *mut *mut u8) -> bool {
    
    todo!();
        /*
            if (*argc == 0)
        return true;
    #if !defined(_MSC_VER)
      // This trick can only be used on UNIX platforms
      if (!::google::glog_internal_namespace_::IsGoogleLoggingInitialized())
    #endif
      {
        ::google::InitGoogleLogging(argv[0]);
    #if !defined(_MSC_VER)
        // This is never defined on Windows
        ::google::InstallFailureSignalHandler();
    #endif
      }
      UpdateLoggingLevelsFromFlags();
      return true;
        */
}

#[cfg(C10_USE_GLOG)]
pub fn update_logging_levels_from_flags()  {
    
    todo!();
        /*
            #ifdef FBCODE_CAFFE2
      // TODO(T82645998): Fix data race exposed by TSAN.
      folly::annotate_ignore_thread_sanitizer_guard g(__FILE__, __LINE__);
    #endif
      // If caffe2_log_level is set and is lower than the min log level by glog,
      // we will transfer the caffe2_log_level setting to glog to override that.
      FLAGS_minloglevel = min(FLAGS_caffe2_log_level, FLAGS_minloglevel);
      // If caffe2_log_level is explicitly set, let's also turn on logtostderr.
      if (FLAGS_caffe2_log_level < google::GLOG_WARNING) {
        FLAGS_logtostderr = 1;
      }
      // Also, transfer the caffe2_log_level verbose setting to glog.
      if (FLAGS_caffe2_log_level < 0) {
        FLAGS_v = min(FLAGS_v, -FLAGS_caffe2_log_level);
      }
        */
}

/**
  | A utility to allow one to show log info
  | to stderr after the program starts.
  | 
  | This is similar to calling GLOG's --logtostderr,
  | or setting caffe2_log_level to smaller
  | than INFO. You are recommended to only
  | use this in a few sparse cases, such as
  | when you want to write a tutorial or something.
  | Normally, use the commandline flags
  | to set the log level.
  |
  */
#[cfg(C10_USE_GLOG)]
pub fn show_log_info_to_stderr()  {
    
    todo!();
        /*
            FLAGS_logtostderr = 1;
      FLAGS_minloglevel = min(FLAGS_minloglevel, google::GLOG_INFO);
        */
}

#[cfg(not(C10_USE_GLOG))]
pub fn show_log_info_to_stderr()  {
    
    todo!();
        /*
            FLAGS_caffe2_log_level = GLOG_INFO;
        */
}

#[cfg(not(C10_USE_GLOG))]
pub fn init_caffe_logging(
        argc: *mut i32,
        argv: *mut *mut u8) -> bool {
    
    todo!();
        /*
            // When doing InitCaffeLogging, we will assume that caffe's flag parser has
      // already finished.
      if (*argc == 0)
        return true;
      if (!CommandLineFlagsHasBeenParsed()) {
        cerr << "InitCaffeLogging() has to be called after "
                     "ParseCommandLineFlags. Modify your program to make sure "
                     "of this."
                  << endl;
        return false;
      }
      if (FLAGS_caffe2_log_level > GLOG_FATAL) {
        cerr << "The log level of Caffe2 has to be no larger than GLOG_FATAL("
                  << GLOG_FATAL << "). Capping it to GLOG_FATAL." << endl;
        FLAGS_caffe2_log_level = GLOG_FATAL;
      }
      return true;
        */
}

#[cfg(not(C10_USE_GLOG))]
pub struct MessageLoader { }

#[cfg(not(C10_USE_GLOG))]
impl MessageLoader {
    
    pub fn new(
        file:     *const u8,
        line:     i32,
        severity: i32) -> Self {
    
        todo!();
        /*


            : severity_(severity) 

      if (severity_ < FLAGS_caffe2_log_level) {
        // Nothing needs to be logged.
        return;
      }
    #ifdef ANDROID
      tag_ = "native";
    #else // !ANDROID
      tag_ = "";
    #endif // ANDROID
      /*
      time_t rawtime;
      struct tm * timeinfo;
      time(&rawtime);
      timeinfo = localtime(&rawtime);
      chrono::nanoseconds ns =
          chrono::duration_cast<chrono::nanoseconds>(
              chrono::high_resolution_clock::now().time_since_epoch());
      */
      stream_ << "["
              << CAFFE2_SEVERITY_PREFIX[min(4, GLOG_FATAL - severity_)]
              //<< (timeinfo->tm_mon + 1) * 100 + timeinfo->tm_mday
              //<< setfill('0')
              //<< " " << setw(2) << timeinfo->tm_hour
              //<< ":" << setw(2) << timeinfo->tm_min
              //<< ":" << setw(2) << timeinfo->tm_sec
              //<< "." << setw(9) << ns.count() % 1000000000
              << " " << StripBasename(string(file)) << ":" << line
              << "] ";
        */
    }
}

#[cfg(not(C10_USE_GLOG))]
impl<W: Write> Drop for MessageLogger<W> {

    // Output the contents of the stream to the
    // proper channel on destruction.
    //
    fn drop(&mut self) {
        todo!();
        /*
            if (severity_ < FLAGS_caffe2_log_level) {
        // Nothing needs to be logged.
        return;
      }
      stream_ << "\n";
    #ifdef ANDROID
      static const int android_log_levels[] = {
          ANDROID_LOG_FATAL, // LOG_FATAL
          ANDROID_LOG_ERROR, // LOG_ERROR
          ANDROID_LOG_WARN, // LOG_WARNING
          ANDROID_LOG_INFO, // LOG_INFO
          ANDROID_LOG_DEBUG, // VLOG(1)
          ANDROID_LOG_VERBOSE, // VLOG(2) .. VLOG(N)
      };
      int android_level_index = GLOG_FATAL - min(GLOG_FATAL, severity_);
      int level = android_log_levels[min(android_level_index, 5)];
      // Output the log string the Android log at the appropriate level.
      __android_log_print(level, tag_, "%s", stream_.str().c_str());
      // Indicate termination if needed.
      if (severity_ == GLOG_FATAL) {
        __android_log_print(ANDROID_LOG_FATAL, tag_, "terminating.\n");
      }
    #else // !ANDROID
      if (severity_ >= FLAGS_caffe2_log_level) {
        // If not building on Android, log all output to cerr.
        cerr << stream_.str();
        // Simulating the glog default behavior: if the severity is above INFO,
        // we flush the stream so that the output appears immediately on cerr.
        // This is expected in some of our tests.
        if (severity_ > GLOG_INFO) {
          cerr << flush;
        }
      }
    #endif // ANDROID
      if (severity_ == GLOG_FATAL) {
        DealWithFatal();
      }
        */
    }
}
