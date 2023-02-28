crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/logging_is_not_google_glog.h]

pub const CAFFE2_SEVERITY_PREFIX: &'static str = "FEWIV";

/// Log severity level constants.
pub const GLOG_FATAL:   i32 = 3;
pub const GLOG_ERROR:   i32 = 2;
pub const GLOG_WARNING: i32 = 1;
pub const GLOG_INFO:    i32 = 0;

pub struct MessageLogger<W: Write> {
    tag:      *const u8,
    stream:   W,
    severity: i32,
}

impl<W: Write> MessageLogger<W> {

    pub fn new(
        file:     *const u8,
        line:     i32,
        severity: i32) -> Self {
    
        todo!();
        /*


        
        */
    }

    /// Return the stream associated with the
    /// logger object.
    ///
    pub fn stream(&mut self) -> &mut W {
        
        todo!();
        /*
            return stream_;
        */
    }

    /// When there is a fatal log, we simply
    /// abort.
    ///
    pub fn deal_with_fatal(&mut self)  {
        
        todo!();
        /*
            abort();
        */
    }
}

/**
  | This class is used to explicitly ignore values
  | in the conditional logging macros.
  |
  | This avoids compiler warnings like "value
  | computed is not used" and "statement has no
  | effect".
  |
  */
pub struct LoggerVoidify {

}

impl LoggerVoidify {

}

/// Log a message and terminate.
pub fn log_message_fatal<T>(
        file:    *const u8,
        line:    i32,
        message: &T)  {

    todo!();
        /*
            MessageLogger(file, line, GLOG_FATAL).stream() << message;
        */
}

/**
  | Helpers for CHECK_NOTNULL(). Two are necessary
  | to support both raw pointers and smart
  | pointers.
  |
  */
pub fn check_not_null_common<T>(
        file:  *const u8,
        line:  i32,
        names: *const u8,
        t:     &mut T) -> &mut T {

    todo!();
        /*
            if (t == nullptr) {
        LogMessageFatal(file, line, string(names));
      }
      return t;
        */
}

pub fn check_not_null_a<T>(
        file:  *const u8,
        line:  i32,
        names: *const u8,
        t:     *mut T) -> *mut T {

    todo!();
        /*
            return CheckNotNullCommon(file, line, names, t);
        */
}

pub fn check_not_null_b<T>(
        file:  *const u8,
        line:  i32,
        names: *const u8,
        t:     &mut T) -> &mut T {

    todo!();
        /*
            return CheckNotNullCommon(file, line, names, t);
        */
}

// ---------------------- Logging Macro definitions --------------------------

/// CAFFE2_LOG_THRESHOLD should at most be GLOG_FATAL.
const_assert!{
    CAFFE2_LOG_THRESHOLD <= GLOG_FATAL
}

/**
  | If n is under the compile time caffe log
  | threshold, The _CAFFE_LOG(n) should not
  | generate anything in optimized code.
  |
  */
macro_rules! LOG {
    ($n:ident) => {
        /*
        
          if (::GLOG_##n >= CAFFE2_LOG_THRESHOLD) 
          ::MessageLogger((char*)__FILE__, __LINE__, ::GLOG_##n).stream()
        */
    }
}

macro_rules! VLOG {
    ($n:ident) => {
        /*
        
          if (-n >= CAFFE2_LOG_THRESHOLD) 
          ::MessageLogger((char*)__FILE__, __LINE__, -n).stream()
        */
    }
}

macro_rules! LOG_IF {
    ($n:ident, $condition:ident) => {
        /*
        
          if (::GLOG_##n >= CAFFE2_LOG_THRESHOLD && (condition)) 
          ::MessageLogger((char*)__FILE__, __LINE__, ::GLOG_##n).stream()
        */
    }
}

macro_rules! VLOG_IF {
    ($n:ident, $condition:ident) => {
        /*
        
          if (-n >= CAFFE2_LOG_THRESHOLD && (condition)) 
          ::MessageLogger((char*)__FILE__, __LINE__, -n).stream()
        */
    }
}

macro_rules! VLOG_IS_ON {
    ($verboselevel:ident) => {
        /*
                (CAFFE2_LOG_THRESHOLD <= -(verboselevel))
        */
    }
}

/**
  | Log with source location information override
  | (to be used in generic warning/error handlers
  | implemented as functions, not macros)
  |
  */
macro_rules! LOG_AT_FILE_LINE {
    ($n:ident, $file:ident, $line:ident) => {
        /*
        
          if (::GLOG_##n >= CAFFE2_LOG_THRESHOLD) 
          ::MessageLogger(file, line, ::GLOG_##n).stream()
        */
    }
}

/**
  | Log only if condition is met. Otherwise
  | evaluates to void.
  |
  */
macro_rules! FATAL_IF {
    ($condition:ident) => {
        /*
        
          condition ? (void)0                                                        
                    : ::LoggerVoidify() &                                       
                  ::MessageLogger((char*)__FILE__, __LINE__, ::GLOG_FATAL) 
                      .stream()
        */
    }
}

/// Check for a given boolean condition.
macro_rules! CHECK {
    ($condition:ident) => {
        /*
                FATAL_IF(condition) << "Check failed: " #condition " "
        */
    }
}

/// Debug only version of CHECK
#[cfg(not(NDEBUG))]
macro_rules! DCHECK {
    ($condition:ident) => {
        /*
                FATAL_IF(condition) << "Check failed: " #condition " "
        */
    }
}

/// Optimized version - generates no code.
#[cfg(NDEBUG)]
macro_rules! DCHECK {
    ($condition:ident) => {
        /*
        
          while (false)           
          CHECK(condition)
        */
    }
}

macro_rules! CHECK_OP {
    ($val1:ident, $val2:ident, $op:ident) => {
        /*
        
          FATAL_IF(((val1)op(val2))) << "Check failed: " #val1 " " #op " " #val2 " (" 
                                     << (val1) << " vs. " << (val2) << ") "
        */
    }
}

// Check_op macro definitions
macro_rules! CHECK_EQ {
    ($val1:ident, $val2:ident) => {
        /*
                CHECK_OP(val1, val2, ==)
        */
    }
}

macro_rules! CHECK_NE {
    ($val1:ident, $val2:ident) => {
        /*
                CHECK_OP(val1, val2, !=)
        */
    }
}

macro_rules! CHECK_LE {
    ($val1:ident, $val2:ident) => {
        /*
                CHECK_OP(val1, val2, <=)
        */
    }
}

macro_rules! CHECK_LT {
    ($val1:ident, $val2:ident) => {
        /*
                CHECK_OP(val1, val2, <)
        */
    }
}

macro_rules! CHECK_GE {
    ($val1:ident, $val2:ident) => {
        /*
                CHECK_OP(val1, val2, >=)
        */
    }
}

macro_rules! CHECK_GT {
    ($val1:ident, $val2:ident) => {
        /*
                CHECK_OP(val1, val2, >)
        */
    }
}



// Debug only versions of CHECK_OP macros.
#[cfg(not(NDEBUG))]
macro_rules! DCHECK_EQ {
    ($val1:ident, $val2:ident) => {
        /*
                CHECK_OP(val1, val2, ==)
        */
    }
}

#[cfg(not(NDEBUG))]
macro_rules! DCHECK_NE {
    ($val1:ident, $val2:ident) => {
        /*
                CHECK_OP(val1, val2, !=)
        */
    }
}

#[cfg(not(NDEBUG))]
macro_rules! DCHECK_LE {
    ($val1:ident, $val2:ident) => {
        /*
                CHECK_OP(val1, val2, <=)
        */
    }
}

#[cfg(not(NDEBUG))]
macro_rules! DCHECK_LT {
    ($val1:ident, $val2:ident) => {
        /*
                CHECK_OP(val1, val2, <)
        */
    }
}

#[cfg(not(NDEBUG))]
macro_rules! DCHECK_GE {
    ($val1:ident, $val2:ident) => {
        /*
                CHECK_OP(val1, val2, >=)
        */
    }
}

#[cfg(not(NDEBUG))]
macro_rules! DCHECK_GT {
    ($val1:ident, $val2:ident) => {
        /*
                CHECK_OP(val1, val2, >)
        */
    }
}

// These versions generate no code in optimized mode.
#[cfg(NDEBUG)]
macro_rules! DCHECK_EQ {
    ($val1:ident, $val2:ident) => {
        /*
        
          while (false)               
          CHECK_OP(val1, val2, ==)
        */
    }
}

#[cfg(NDEBUG)]
macro_rules! DCHECK_NE {
    ($val1:ident, $val2:ident) => {
        /*
        
          while (false)               
          CHECK_OP(val1, val2, !=)
        */
    }
}

#[cfg(NDEBUG)]
macro_rules! DCHECK_LE {
    ($val1:ident, $val2:ident) => {
        /*
        
          while (false)               
          CHECK_OP(val1, val2, <=)
        */
    }
}

#[cfg(NDEBUG)]
macro_rules! DCHECK_LT {
    ($val1:ident, $val2:ident) => {
        /*
        
          while (false)               
          CHECK_OP(val1, val2, <)
        */
    }
}

#[cfg(NDEBUG)]
macro_rules! DCHECK_GE {
    ($val1:ident, $val2:ident) => {
        /*
        
          while (false)               
          CHECK_OP(val1, val2, >=)
        */
    }
}

#[cfg(NDEBUG)]
macro_rules! DCHECK_GT {
    ($val1:ident, $val2:ident) => {
        /*
        
          while (false)               
          CHECK_OP(val1, val2, >)
        */
    }
}

/// Check that a pointer is not null.
macro_rules! CHECK_NOTNULL {
    ($val:ident) => {
        /*
        
          ::CheckNotNull(     
              __FILE__, __LINE__, "Check failed: '" #val "' Must be non NULL", (val))
        */
    }
}


// Debug only version of CHECK_NOTNULL
#[cfg(not(NDEBUG))]
macro_rules! DCHECK_NOTNULL {
    ($val:ident) => {
        /*
        
          ::CheckNotNull(      
              __FILE__, __LINE__, "Check failed: '" #val "' Must be non NULL", (val))
        */
    }
}

// Optimized version - generates no code.
#[cfg(NDEBUG)]
macro_rules! DCHECK_NOTNULL {
    ($val:ident) => {
        /*
        
          while (false)             
          CHECK_NOTNULL(val)
        */
    }
}


// ---------------------- Support for std objects --------------------------
macro_rules! INSTANTIATE_FOR_CONTAINER {
    ($container:ident) => {
        /*
        
          template <class... Types>                                
          ostream& operator<<(                                
              ostream& out, const container<Types...>& seq) { 
            PrintSequence(out, seq.begin(), seq.end());       
            return out;                                            
          }
        */
    }
}

INSTANTIATE_FOR_CONTAINER!{vector}
INSTANTIATE_FOR_CONTAINER!{map}
INSTANTIATE_FOR_CONTAINER!{set}

#[inline] pub fn print_sequence<W: Write, Iter>(
    out:   W,
    begin: Iter,
    end:   Iter)  {

    todo!();
        /*
            // Output at most 100 elements -- appropriate if used for logging.
      for (int i = 0; begin != end && i < 100; ++i, ++begin) {
        if (i > 0)
          out << ' ';
        out << *begin;
      }
      if (begin != end) {
        out << " ...";
      }
        */
}
