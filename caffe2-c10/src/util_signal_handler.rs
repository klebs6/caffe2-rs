crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/signal_handler.h]

#[cfg(__APPLE__)]
pub const C10_SUPPORTS_SIGNAL_HANDLER: bool = true;

#[cfg(all(__linux__,not(C10_DISABLE_SIGNAL_HANDLERS)))]
pub const C10_SUPPORTS_FATAL_SIGNAL_HANDLERS: bool = true;

#[cfg(all(__linux__,not(C10_DISABLE_SIGNAL_HANDLERS)))]
pub const C10_SUPPORTS_SIGNAL_HANDLER:        bool = true;

pub enum SignalHandlerAction { NONE, STOP }

pub struct C10SignalHandler {
    sigint_action:   SignalHandlerAction,
    sighup_action:   SignalHandlerAction,
    my_sigint_count: u64,
    my_sighup_count: u64,
}

/**
  | This works by setting up certain fatal signal
  | handlers. Previous fatal signal handlers will
  | still be called when the signal is
  | raised. Defaults to being off.
  |
  */
#[cfg(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)]
pub struct FatalSignalHandler {

    /**
     | The mutex protects the bool.
     |
     */
    fatal_signal_handlers_installation_mutex: Mutex,
    fatal_signal_handlers_installed:          bool,

    /**
      | We need to hold a reference to call the
      | previous SIGUSR2 handler in case we
      | didn't signal it
      |
      */
    previous_sigusr2:                         nix::sys::signal::SigAction,


    /**
      | Flag dictating whether the SIGUSR2
      | handler falls back to previous handlers
      | or is intercepted in order to print a
      | stack trace.
      |
      */
    fatal_signal_received:                    AtomicBool,


    /**
      | Global state set when a fatal signal
      | is received so that backtracing threads
      | know why they're printing a stacktrace.
      |
      */
    fatal_signal_name:                        *const u8,

    fatal_signum:                             i32, // default = -1

    /**
      | This wait condition is used to wait for
      | other threads to finish writing their
      | stack trace when in fatal sig handler
      | (we can't use pthread_join because
      | there's no way to convert from a tid to
      | a pthread_t).
      |
      */
    writing_cond:                             PThreadCond,

    writing_mutex:                            PThreadMutex,
}

#[cfg(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)]
lazy_static!{
    /*
    FatalSignalHandler::signal_handler FatalSignalHandler::kSignalHandlers[] = {
        {"SIGABRT", SIGABRT, {}},
        {"SIGINT", SIGINT, {}},
        {"SIGILL", SIGILL, {}},
        {"SIGFPE", SIGFPE, {}},
        {"SIGBUS", SIGBUS, {}},
        {"SIGSEGV", SIGSEGV, {}},
        {nullptr, 0, {}}};
    */
}

#[cfg(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)]
impl FatalSignalHandler {
    
    pub fn set_print_stack_traces_on_fatal_signal(&mut self, print: bool)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn print_stack_traces_on_fatal_signal(&mut self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_instance() -> &mut FatalSignalHandler {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new() -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn install_fatal_signal_handlers(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn uninstall_fatal_signal_handlers(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    
    pub fn fatal_signal_handler_static(signum: i32)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn fatal_signal_handler(&mut self, signum: i32)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn fatal_signal_handler_post_process(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_previous_sigaction(&mut self, signum: i32) -> *mut nix::sys::signal::SigAction {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_signal_name(&mut self, signum: i32) -> *const u8 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn call_previous_signal_handler(&mut self, 
        action: *mut nix::sys::signal::SigAction,
        signum: i32,
        info:   *mut libc::siginfo_t,
        ctx:    *mut c_void)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn stacktrace_signal_handler(&mut self, needs_lock: bool)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn stacktrace_signal_handler_static(
        signum: i32,
        info:   *mut libc::siginfo_t,
        ctx:    *mut c_void)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn stacktrace_signal_handler(&mut self, 
        signum: i32,
        info:   *mut libc::siginfo_t,
        ctx:    *mut c_void)  {
        
        todo!();
        /*
        
        */
    }
}

#[cfg(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)]
pub struct C10SignalHandler {
    name:     *const u8,
    signum:   i32,
    previous: nix::sys::signal::SigAction,
}

//-------------------------------------------[.cpp/pytorch/c10/util/signal_handler.cpp]


/// Normal signal handler implementation.
///
#[cfg(C10_SUPPORTS_SIGNAL_HANDLER)]
#[cfg(C10_ANDROID)]
#[cfg(not(SYS_gettid))]
macro_rules! sys_gettid {
    () => {
        /*
                __NR_gettid
        */
    }
}

#[cfg(C10_SUPPORTS_SIGNAL_HANDLER)]
#[cfg(C10_ANDROID)]
#[cfg(not(SYS_tgkill))]
macro_rules! sys_tgkill {
    () => {
        /*
                __NR_tgkill
        */
    }
}

#[cfg(C10_SUPPORTS_SIGNAL_HANDLER)]
lazy_static!{
    /*
    struct sigaction previousSighup;
    struct sigaction previousSigint;
    atomic<int> sigintCount(0);
    atomic<int> sighupCount(0);
    atomic<int> hookedUpCount(0);
    */
}

#[cfg(C10_SUPPORTS_SIGNAL_HANDLER)]
pub fn handle_signal(signal: i32)  {
    
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

#[cfg(C10_SUPPORTS_SIGNAL_HANDLER)]
pub fn hookup_handler()  {
    
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
///
#[cfg(C10_SUPPORTS_SIGNAL_HANDLER)]
pub fn unhook_handler()  {
    
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

#[cfg(C10_SUPPORTS_SIGNAL_HANDLER)]
#[cfg(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)]
impl FatalSignalHandler {
    
    pub fn get_instance(&mut self) -> &mut FatalSignalHandler {
        
        todo!();
        /*
            // Leaky singleton to avoid module destructor race.
      static FatalSignalHandler* handler = new FatalSignalHandler();
      return *handler;
        */
    }
    
    pub fn new() -> Self {
    
        todo!();
        /*


            : fatalSignalHandlersInstalled(false),
          fatalSignalReceived(false),
          fatalSignalName("<UNKNOWN>"),
          writingCond(PTHREAD_COND_INITIALIZER),
          writingMutex(PTHREAD_MUTEX_INITIALIZER)
        */
    }
    
    pub fn get_previous_sigaction(&mut self, signum: i32) -> *mut SigAction {
        
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
    
    pub fn get_signal_name(&mut self, signum: i32) -> *const u8 {
        
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
    
    pub fn call_previous_signal_handler(&mut self, 
        action: *mut SigAction,
        signum: i32,
        info:   *mut libc::siginfo_t,
        ctx:    *mut c_void)  {
        
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

    /// needsLock signals whether we need to lock
    /// our writing mutex.
    ///
    pub fn stacktrace_signal_handler(&mut self, needs_lock: bool)  {
        
        todo!();
        /*
            if (needsLock) {
        pthread_mutex_lock(&writingMutex);
      }
      pid_t tid = syscall(SYS_gettid);
      string backtrace = fmt::format(
          "{}({}), PID: {}, Thread {}: \n {}",
          fatalSignalName,
          fatalSignum,
          ::getpid(),
          tid,
          get_backtrace());
      cerr << backtrace << endl;
      if (needsLock) {
        pthread_mutex_unlock(&writingMutex);
        pthread_cond_signal(&writingCond);
      }
        */
    }
    
    pub fn fatal_signal_handler_post_process(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn fatal_signal_handler_static(&mut self, signum: i32)  {
        
        todo!();
        /*
            getInstance().fatalSignalHandler(signum);
        */
    }

    /// Our fatal signal entry point
    ///
    pub fn fatal_signal_handler(&mut self, signum: i32)  {
        
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
      fatalSignalHandlerPostProcess();
      sigaction(signum, getPreviousSigaction(signum), nullptr);
      raise(signum);
        */
    }

    /// Our SIGUSR2 entry point
    pub fn stacktrace_signal_handler_static(&mut self, 
        signum: i32,
        info:   *mut libc::siginfo_t,
        ctx:    *mut c_void)  {
        
        todo!();
        /*
            getInstance().stacktraceSignalHandler(signum, info, ctx);
        */
    }
    
    pub fn stacktrace_signal_handler(&mut self, 
        signum: i32,
        info:   *mut libc::siginfo_t,
        ctx:    *mut c_void)  {
        
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
      | Installs SIGABRT signal handler so that we
      | get stack traces from every thread on
      | SIGABRT caused exit.
      |
      | Also installs SIGUSR2 handler so that
      | threads can communicate with each other (be
      | sure if you use SIGUSR2) to install your
      | handler before initing caffe2 (we properly
      | fall back to the previous handler if we
      | didn't initiate the SIGUSR2).
      |
      */
    pub fn install_fatal_signal_handlers(&mut self)  {
        
        todo!();
        /*
            lock_guard<mutex> locker(fatalSignalHandlersInstallationMutex);
      if (fatalSignalHandlersInstalled) {
        return;
      }
      fatalSignalHandlersInstalled = true;
      struct sigaction sa;
      sigemptyset(&sa.sa_mask);
      // Since we'll be in an exiting situation it's possible there's memory
      // corruption, so make our own stack just in case.
      sa.sa_flags = SA_ONSTACK | SA_SIGINFO;
      sa.sa_handler = FatalSignalHandler::fatalSignalHandlerStatic;
      for (auto* handler = kSignalHandlers; handler->name != nullptr; handler++) {
        if (sigaction(handler->signum, &sa, &handler->previous)) {
          string str("Failed to add ");
          str += handler->name;
          str += " handler!";
          perror(str.c_str());
        }
      }
      sa.sa_sigaction = FatalSignalHandler::stacktraceSignalHandlerStatic;
      if (sigaction(SIGUSR2, &sa, &previousSigusr2)) {
        perror("Failed to add SIGUSR2 handler!");
      }
        */
    }
    
    pub fn uninstall_fatal_signal_handlers(&mut self)  {
        
        todo!();
        /*
            lock_guard<mutex> locker(fatalSignalHandlersInstallationMutex);
      if (!fatalSignalHandlersInstalled) {
        return;
      }
      fatalSignalHandlersInstalled = false;
      for (auto* handler = kSignalHandlers; handler->name != nullptr; handler++) {
        if (sigaction(handler->signum, &handler->previous, nullptr)) {
          string str("Failed to remove ");
          str += handler->name;
          str += " handler!";
          perror(str.c_str());
        } else {
          handler->previous = {};
        }
      }
      if (sigaction(SIGUSR2, &previousSigusr2, nullptr)) {
        perror("Failed to add SIGUSR2 handler!");
      } else {
        previousSigusr2 = {};
      }
        */
    }
}

#[cfg(C10_SUPPORTS_SIGNAL_HANDLER)]
impl Drop for C10SignalHandler {

    fn drop(&mut self) {
        todo!();
        /*
            unhookHandler();
        */
    }
}

#[cfg(feature = "signalhandler")]
impl C10SignalHandler {

    /// Constructor. Specify what action to take
    ///  when a signal is received.
    ///
    pub fn new(
        sigint_action: SignalHandlerAction,
        sighup_action: SignalHandlerAction) -> Self {
    
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
    pub fn gotsigint(&mut self) -> bool {
        
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
    pub fn gotsighup(&mut self) -> bool {
        
        todo!();
        /*
            uint64_t count = sighupCount;
      bool result = (count != my_sighup_count_);
      my_sighup_count_ = count;
      return result;
        */
    }
    
    pub fn check_for_signals(&mut self) -> SignalHandlerAction {
        
        todo!();
        /*
            if (GotSIGHUP()) {
        return SIGHUP_action_;
      }
      if (GotSIGINT()) {
        return SIGINT_action_;
      }
      return C10SignalHandler::Action::NONE;
        */
    }
}

#[cfg(C10_SUPPORTS_SIGNAL_HANDLER)]
#[cfg(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)]
impl FatalSignalHandler {
    
    pub fn set_print_stack_traces_on_fatal_signal(&mut self, print: bool)  {
        
        todo!();
        /*
            if (print) {
        installFatalSignalHandlers();
      } else {
        uninstallFatalSignalHandlers();
      }
        */
    }
    
    pub fn print_stack_traces_on_fatal_signal(&mut self) -> bool {
        
        todo!();
        /*
            lock_guard<mutex> locker(fatalSignalHandlersInstallationMutex);
      return fatalSignalHandlersInstalled;
        */
    }
}

#[cfg(not(feature = "signalhandler"))]
impl C10SignalHandler {
    
    // TODO: Currently we do not support signal
    // handling in non-Linux yet - below is
    // a minimal implementation that makes things
    // compile.
    //
    pub fn new(
        sigint_action: SignalHandlerAction,
        sighup_action: SignalHandlerAction) -> Self {
    
        todo!();
        /*


            SIGINT_action_ = SIGINT_action;
      SIGHUP_action_ = SIGHUP_action;
      my_sigint_count_ = 0;
      my_sighup_count_ = 0;
        */
    }
    
    pub fn gotsigint(&mut self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    pub fn gotsighup(&mut self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    pub fn check_for_signals(&mut self) -> SignalHandlerAction {
        
        todo!();
        /*
            return C10SignalHandler::Action::NONE;
        */
    }
}
