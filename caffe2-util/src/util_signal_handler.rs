crate::ix!();

use crate::{
    SigAction,
};

pub enum SignalHandlerAction {
    NONE,
    STOP
}

pub struct SignalHandler {
    SIGINT_action:   SignalHandlerAction,
    SIGHUP_action:   SignalHandlerAction,
    my_sigint_count: u64,
    my_sighup_count: u64,
}

#[cfg(c10_android)]
type SYS_gettid = __NR_gettid;

#[cfg(c10_android)]
type SYS_tgkill = __NR_tgkill;

lazy_static!{
    /*
    struct sigaction previousSighup;
    struct sigaction previousSigint;
    std::atomic<int> sigintCount(0);
    std::atomic<int> sighupCount(0);
    std::atomic<int> hookedUpCount(0);
    */
}

#[inline] pub fn handle_signal(signal: i32)  {
    
    todo!();
    /*
        switch (signal) {
        // TODO: what if the previous handler uses sa_sigaction?
        case SIGHUP:
          sighupCount += 1;
          if (previousSighup.sa_handler) {
            previousSighup.sa_handler(signal);
          }
          break;
        case SIGINT:
          sigintCount += 1;
          if (previousSigint.sa_handler) {
            previousSigint.sa_handler(signal);
          }
          break;
      }
    */
}

#[inline] pub fn hookup_handler()  {
    
    todo!();
    /*
        if (hookedUpCount++) {
        return;
      }
      struct sigaction sa;
      // Setup the handler
      sa.sa_handler = &handleSignal;
      // Restart the system call, if at all possible
      sa.sa_flags = SA_RESTART;
      // Block every signal during the handler
      sigfillset(&sa.sa_mask);
      // Intercept SIGHUP and SIGINT
      if (sigaction(SIGHUP, &sa, &previousSighup) == -1) {
        LOG(FATAL) << "Cannot install SIGHUP handler.";
      }
      if (sigaction(SIGINT, &sa, &previousSigint) == -1) {
        LOG(FATAL) << "Cannot install SIGINT handler.";
      }
    */
}

/// Set the signal handlers to the default.
#[inline] pub fn unhook_handler()  {
    
    todo!();
    /*
        if (--hookedUpCount > 0) {
        return;
      }
      struct sigaction sa;
      // Setup the sighub handler
      sa.sa_handler = SIG_DFL;
      // Restart the system call, if at all possible
      sa.sa_flags = SA_RESTART;
      // Block every signal during the handler
      sigfillset(&sa.sa_mask);
      // Intercept SIGHUP and SIGINT
      if (sigaction(SIGHUP, &previousSighup, nullptr) == -1) {
        LOG(FATAL) << "Cannot uninstall SIGHUP handler.";
      }
      if (sigaction(SIGINT, &previousSigint, nullptr) == -1) {
        LOG(FATAL) << "Cannot uninstall SIGINT handler.";
      }
    */
}

pub struct SignalHandlerInfo {
    name:     &'static str,
    signum:   i32,
    previous: Option<SigAction>,
}

#[cfg(caffe2_supports_fatal_signal_handlers)]
lazy_static!{

    /*
    // The mutex protects the bool.
    std::mutex fatalSignalHandlersInstallationMutex;
    bool fatalSignalHandlersInstalled;

    // We need to hold a reference to call the previous SIGUSR2 handler in case
    // we didn't signal it
    struct sigaction previousSigusr2;

    // Flag dictating whether the SIGUSR2 handler falls back to previous handlers
    // or is intercepted in order to print a stack trace.
    std::atomic<bool> fatalSignalReceived(false);

    // Global state set when a fatal signal is received so that backtracing threads
    // know why they're printing a stacktrace.
    const char* fatalSignalName("<UNKNOWN>");

    int fatalSignum(-1);

    // This wait condition is used to wait for other threads to finish writing
    // their stack trace when in fatal sig handler (we can't use pthread_join
    // because there's no way to convert from a tid to a pthread_t).
    pthread_cond_t writingCond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t writingMutex = PTHREAD_MUTEX_INITIALIZER;
    */
}

pub const kSignalHandlers: [SignalHandlerInfo; 7] = [
    SignalHandlerInfo { name: "SIGABRT", signum: libc::SIGABRT, previous: None, },
    SignalHandlerInfo { name: "SIGINT",  signum: libc::SIGINT,  previous: None, },
    SignalHandlerInfo { name: "SIGILL",  signum: libc::SIGILL,  previous: None, },
    SignalHandlerInfo { name: "SIGFPE",  signum: libc::SIGFPE,  previous: None, },
    SignalHandlerInfo { name: "SIGBUS",  signum: libc::SIGBUS,  previous: None, },
    SignalHandlerInfo { name: "SIGSEGV", signum: libc::SIGSEGV, previous: None, },
    SignalHandlerInfo { name: "",        signum: 0,             previous: None, },
];


#[cfg(caffe2_supports_fatal_signal_handlers)]
#[inline] pub fn get_previous_sigaction(signum: i32) -> *mut nix::sys::signal::SigAction {
    
    todo!();
    /*
        for (auto handler = kSignalHandlers; handler->name != nullptr; handler++) {
        if (handler->signum == signum) {
          return &handler->previous;
        }
      }
      return nullptr;
    */
}

#[cfg(caffe2_supports_fatal_signal_handlers)]
#[inline] pub fn get_signal_name(signum: i32) -> *const u8 {
    
    todo!();
    /*
        for (auto handler = kSignalHandlers; handler->name != nullptr; handler++) {
        if (handler->signum == signum) {
          return handler->name;
        }
      }
      return nullptr;
    */
}

#[cfg(caffe2_supports_fatal_signal_handlers)]
#[inline] pub fn unwinder(
    context:   *mut unwind::_Unwind_Context,
    user_info: *mut c_void) -> unwind::_Unwind_Reason_Code 
{
    
    todo!();
    /*
        auto& pcs = *reinterpret_cast<std::vector<uintptr_t>*>(userInfo);
      pcs.push_back(_Unwind_GetIP(context));
      return _URC_NO_REASON;
    */
}

#[cfg(caffe2_supports_fatal_signal_handlers)]
#[inline] pub fn get_backtrace() -> Vec<uintptr_t> {
    
    todo!();
    /*
        std::vector<uintptr_t> pcs;
      _Unwind_Backtrace(unwinder, &pcs);
      return pcs;
    */
}

#[cfg(caffe2_supports_fatal_signal_handlers)]
#[inline] pub fn print_blob_sizes()  {
    
    todo!();
    /*
        ::caffe2::Workspace::ForEach(
          [&](::caffe2::Workspace* ws) { ws->PrintBlobSizes(); });
    */
}

#[cfg(caffe2_supports_fatal_signal_handlers)]
#[inline] pub fn print_stacktrace()  {
    
    todo!();
    /*
        std::vector<uintptr_t> pcs = getBacktrace();
      Dl_info info;
      size_t i = 0;
      for (uintptr_t pcAddr : pcs) {
        const void* pc = reinterpret_cast<const void*>(pcAddr);
        const char* path = nullptr;
        const char* name = "???";
        char* demangled = nullptr;
        int offset = -1;

        std::cerr << "[" << i << "] ";
        if (dladdr(pc, &info)) {
          path = info.dli_fname;
          name = info.dli_sname ?: "???";
          offset = reinterpret_cast<uintptr_t>(pc) -
              reinterpret_cast<uintptr_t>(info.dli_saddr);

          int status;
          demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
          if (status == 0) {
            name = demangled;
          }
        }
        std::cerr << name;
        if (offset >= 0) {
          std::cerr << "+" << reinterpret_cast<void*>(offset);
        }
        std::cerr << "(" << pc << ")";
        if (path) {
          std::cerr << " in " << path;
        }
        std::cerr << std::endl;
        if (demangled) {
          free(demangled);
        }
        i += 1;
      }
    */
}

#[cfg(caffe2_supports_fatal_signal_handlers)]
#[inline] pub fn call_previous_signal_handler(
    action: *mut nix::sys::signal::SigAction,
    signum: i32,
    info:   *mut siginfo_t,
    ctx:    *mut c_void)  
{
    
    todo!();
    /*
        if (!action->sa_handler) {
        return;
      }
      if ((action->sa_flags & SA_SIGINFO) == SA_SIGINFO) {
        action->sa_sigaction(signum, info, ctx);
      } else {
        action->sa_handler(signum);
      }
    */
}

/**
  | needsLock signals whether we need to
  | lock our writing mutex.
  |
  */
#[cfg(caffe2_supports_fatal_signal_handlers)]
#[inline] pub fn stacktrace_signal_handler(needs_lock: bool)  {
    
    todo!();
    /*
        if (needsLock) {
        pthread_mutex_lock(&writingMutex);
      }
      pid_t tid = syscall(SYS_gettid);
      std::cerr << fatalSignalName << "(" << fatalSignum << "), Thread " << tid
                << ": " << std::endl;
      printStacktrace();
      std::cerr << std::endl;
      if (needsLock) {
        pthread_mutex_unlock(&writingMutex);
        pthread_cond_signal(&writingCond);
      }
    */
}

/// Our fatal signal entry point
#[cfg(caffe2_supports_fatal_signal_handlers)]
#[inline] pub fn fatal_signal_handler(signum: i32)  {
    
    todo!();
    /*
        // Check if this is a proper signal that we declared above.
      const char* name = getSignalName(signum);
      if (!name) {
        return;
      }
      if (fatalSignalReceived) {
        return;
      }
      // Set the flag so that our SIGUSR2 handler knows that we're aborting and
      // that it should intercept any SIGUSR2 signal.
      fatalSignalReceived = true;
      // Set state for other threads.
      fatalSignum = signum;
      fatalSignalName = name;
      // Linux doesn't have a nice userland API for enumerating threads so we
      // need to use the proc pseudo-filesystem.
      DIR* procDir = opendir("/proc/self/task");
      if (procDir) {
        pid_t pid = getpid();
        pid_t currentTid = syscall(SYS_gettid);
        struct dirent* entry;
        pthread_mutex_lock(&writingMutex);
        while ((entry = readdir(procDir)) != nullptr) {
          if (entry->d_name[0] == '.') {
            continue;
          }
          pid_t tid = atoi(entry->d_name);
          // If we've found the current thread then we'll jump into the SIGUSR2
          // handler before calling pthread_cond_wait thus deadlocking, so branch
          // our directly to the backtrace handler instead of signaling it.
          if (tid != currentTid) {
            syscall(SYS_tgkill, pid, tid, SIGUSR2);
            pthread_cond_wait(&writingCond, &writingMutex);
          } else {
            stacktraceSignalHandler(false);
          }
        }
        pthread_mutex_unlock(&writingMutex);
      } else {
        perror("Failed to open /proc/self/task");
      }
      printBlobSizes();
      sigaction(signum, getPreviousSigaction(signum), nullptr);
      raise(signum);
    */
}

/// Our SIGUSR2 entry point
#[cfg(caffe2_supports_fatal_signal_handlers)]
#[inline] pub fn stacktrace_signal_handler(
    signum: i32,
    info:   *mut siginfo_t,
    ctx:    *mut c_void)  
{
    todo!();
    /*
        if (fatalSignalReceived) {
        stacktraceSignalHandler(true);
      } else {
        // We don't want to actually change the signal handler as we want to
        // remain the signal handler so that we may get the usr2 signal later.
        callPreviousSignalHandler(&previousSigusr2, signum, info, ctx);
      }
    */
}

/**
  | Installs SIGABRT signal handler so
  | that we get stack traces from every thread
  | on SIGABRT caused exit.
  | 
  | Also installs SIGUSR2 handler so that
  | threads can communicate with each other
  | (be sure if you use SIGUSR2) to install
  | your handler before initing caffe2
  | (we properly fall back to the previous
  | handler if we didn't initiate the SIGUSR2).
  |
  */
#[cfg(caffe2_supports_fatal_signal_handlers)]
#[inline] pub fn install_fatal_signal_handlers()  {
    
    todo!();
    /*
        std::lock_guard<std::mutex> locker(fatalSignalHandlersInstallationMutex);
      if (fatalSignalHandlersInstalled) {
        return;
      }
      fatalSignalHandlersInstalled = true;
      struct sigaction sa;
      sigemptyset(&sa.sa_mask);
      // Since we'll be in an exiting situation it's possible there's memory
      // corruption, so make our own stack just in case.
      sa.sa_flags = SA_ONSTACK | SA_SIGINFO;
      sa.sa_handler = ::fatalSignalHandler;
      for (auto* handler = kSignalHandlers; handler->name != nullptr; handler++) {
        if (sigaction(handler->signum, &sa, &handler->previous)) {
          std::string str("Failed to add ");
          str += handler->name;
          str += " handler!";
          perror(str.c_str());
        }
      }
      sa.sa_sigaction = ::stacktraceSignalHandler;
      if (sigaction(SIGUSR2, &sa, &::previousSigusr2)) {
        perror("Failed to add SIGUSR2 handler!");
      }
    */
}

#[cfg(caffe2_supports_fatal_signal_handlers)]
#[inline] pub fn uninstall_fatal_signal_handlers()  {
    
    todo!();
    /*
        std::lock_guard<std::mutex> locker(fatalSignalHandlersInstallationMutex);
      if (!fatalSignalHandlersInstalled) {
        return;
      }
      fatalSignalHandlersInstalled = false;
      for (auto* handler = kSignalHandlers; handler->name != nullptr; handler++) {
        if (sigaction(handler->signum, &handler->previous, nullptr)) {
          std::string str("Failed to remove ");
          str += handler->name;
          str += " handler!";
          perror(str.c_str());
        } else {
          handler->previous = {};
        }
      }
      if (sigaction(SIGUSR2, &::previousSigusr2, nullptr)) {
        perror("Failed to add SIGUSR2 handler!");
      } else {
        ::previousSigusr2 = {};
      }
    */
}

#[cfg(caffe2_supports_fatal_signal_handlers)]
define_bool!{
    caffe2_print_stacktraces,
    false,
    "If set, prints stacktraces when a fatal signal is raised."
}

impl Drop for SignalHandler {
    fn drop(&mut self) {
        todo!();
        /* 
      unhookHandler();
         */
    }
}

impl SignalHandler {

    /**
      | Constructor. Specify what action to
      | take when a signal is received.
      |
      */
    pub fn new(
        SIGINT_action: SignalHandlerAction,
        SIGHUP_action: SignalHandlerAction) -> Self 
    {
        todo!();
        /*
            : SIGINT_action_(SIGINT_action),
          SIGHUP_action_(SIGHUP_action),
          my_sigint_count_(sigintCount),
          my_sighup_count_(sighupCount) 
      hookupHandler();
        */
    }

    /**
      | Return true iff a SIGINT has been received
      | since the last time this function was
      | called.
      |
      */
    #[inline] pub fn gotSIGINT(&mut self) -> bool {
        
        todo!();
        /*
            uint64_t count = sigintCount;
      bool result = (count != my_sigint_count_);
      my_sigint_count_ = count;
      return result;
        */
    }

    /**
      | Return true iff a SIGHUP has been received
      | since the last time this function was
      | called.
      |
      */
    #[inline] pub fn gotSIGHUP(&mut self) -> bool {
        
        todo!();
        /*
            uint64_t count = sighupCount;
      bool result = (count != my_sighup_count_);
      my_sighup_count_ = count;
      return result;
        */
    }
    
    #[inline] pub fn check_for_signals(&mut self) -> SignalHandlerAction {
        
        todo!();
        /*
            if (GotSIGHUP()) {
        return SIGHUP_action_;
      }
      if (GotSIGINT()) {
        return SIGINT_action_;
      }
      return SignalHandlerAction::NONE;
        */
    }
}

/**
  | This works by setting up certain fatal
  | signal handlers.
  | 
  | Previous fatal signal handlers will
  | still be called when the signal is raised.
  | Defaults to being off.
  |
  */
#[cfg(caffe2_supports_fatal_signal_handlers)]
#[inline] pub fn set_print_stack_traces_on_fatal_signal(print: bool)  {
    
    todo!();
    /*
        if (print) {
        installFatalSignalHandlers();
      } else {
        uninstallFatalSignalHandlers();
      }
    */
}

#[cfg(caffe2_supports_fatal_signal_handlers)]
#[inline] pub fn print_stack_traces_on_fatal_signal() -> bool {
    
    todo!();
    /*
        std::lock_guard<std::mutex> locker(fatalSignalHandlersInstallationMutex);
      return fatalSignalHandlersInstalled;
    */
}

#[cfg(caffe2_supports_fatal_signal_handlers)]
pub fn caffe2_init_fatal_signal_handler(x: *mut i32, x: *mut *mut *mut u8) -> bool {
    todo!();
    /*
      if (FLAGS_caffe2_print_stacktraces) {
        setPrintStackTracesOnFatalSignal(true);
      }
      return true;
    */
}

#[cfg(caffe2_supports_fatal_signal_handlers)]
register_caffe2_init_function!{
    Caffe2InitFatalSignalHandler,
    &Caffe2InitFatalSignalHandler,
    "Inits signal handlers for fatal signals so we can see what if caffe2_print_stacktraces is set."
}

#[cfg(caffe2_supports_signal_handler)]
impl SignalHandler {
    
    /**
      | TODO: Currently we do not support signal
      | handling in non-Linux yet - below is
      | a minimal implementation that makes things
      | compile.
      */
    pub fn new(
        SIGINT_action: SignalHandlerAction,
        SIGHUP_action: SignalHandlerAction) -> Self 
    {
        todo!();
        /*
            SIGINT_action_ = SIGINT_action;
      SIGHUP_action_ = SIGHUP_action;
      my_sigint_count_ = 0;
      my_sighup_count_ = 0;
        */
    }
    
    #[inline] pub fn gotSIGINT(&mut self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    #[inline] pub fn gotSIGHUP(&mut self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    #[inline] pub fn check_for_signals(&mut self) -> SignalHandlerAction {
        
        todo!();
        /*
            return SignalHandlerAction::NONE;
        */
    }
}
