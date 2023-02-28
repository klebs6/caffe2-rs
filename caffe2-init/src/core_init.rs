crate::ix!();

type InitFunction = fn(i: *mut i32, c: *mut *mut *mut u8) -> bool;

pub struct Caffe2InitializeRegistry {
    early_init_functions:         Vec<(InitFunction, *const u8)>,
    init_functions:               Vec<(InitFunction, *const u8)>,
    named_functions:              HashMap<String, InitFunction>,
    early_init_functions_run_yet: bool,//false
    init_functions_run_yet:       bool,//false
}

impl Caffe2InitializeRegistry {

    /**
      | Run all registered initialization
      | functions.
      | 
      | This has to be called AFTER all static
      | initialization are finished and main()
      | has started, since we are using logging.
      |
      */
    fn run_registered_init_functions_internal(&mut self,
        functions: &Vec<(InitFunction, *const u8)>,
        pargc: *mut i32,
        pargv: *mut *mut *mut u8) -> bool 
    {
        todo!();
        /*
        for (const auto& init_pair : functions) {
            VLOG(1) << "Running init function: " << init_pair.second;
            if (!(*init_pair.first)(pargc, pargv)) {
                LOG(ERROR) << "Initialization function failed.";
                return false;
            }
        }
        return true;
        */
    }

    #[inline] pub fn run_registered_init_functions(
        &mut self, 
        pargc: *mut i32,
        pargv: *mut *mut *mut u8) -> bool 
    {
        todo!();
        /*
            CAFFE_ENFORCE(!init_functions_run_yet_);
                init_functions_run_yet_ = true;
                return RunRegisteredInitFunctionsInternal(init_functions_, pargc, pargv);
        */
    }
    
    #[inline] pub fn run_named_function(
        &mut self, 
        name: *const u8,
        pargc: *mut i32,
        pargv: *mut *mut *mut u8) -> bool {
        
        todo!();
        /*
            if (named_functions_.count(name)) {
                    return named_functions_[name](pargc, pargv);
                }
                return false;
        */
    }
    
    #[inline] pub fn run_registered_early_init_functions(
        &mut self, 
        pargc: *mut i32,
        pargv: *mut *mut *mut u8) -> bool 
    {
        
        todo!();
        /*
            CAFFE_ENFORCE(!early_init_functions_run_yet_);
                early_init_functions_run_yet_ = true;
                return RunRegisteredInitFunctionsInternal(
                    early_init_functions_, pargc, pargv);
        */
    }
    
    #[inline] pub fn register(
        &mut self, 
        function:    InitFunction,
        run_early:   bool,
        description: *const u8,
        name:        Option<&str>)  
    {

        todo!();
        /*
            if (name) {
                    named_functions_[name] = function;
                }
                if (run_early) {
                    // Disallow registration after GlobalInit of early init functions
                    CAFFE_ENFORCE(!early_init_functions_run_yet_);
                    early_init_functions_.emplace_back(function, description);
                } else {
                    if (init_functions_run_yet_) {
                        // Run immediately, since GlobalInit already ran. This should be
                        // rare but we want to allow it in some cases.
                        LOG(WARNING) << "Running init function after GlobalInit: "
                            << description;
                        // TODO(orionr): Consider removing argc and argv for non-early
                        // registration. Unfortunately that would require a new InitFunction
                        // typedef, so not making the change right now.
                        //
                        // Note that init doesn't receive argc and argv, so the function
                        // might fail and we want to raise an error in that case.
                        int argc = 0;
                        char** argv = nullptr;
                        bool success = (function)(&argc, &argv);
                        CAFFE_ENFORCE(success);
                    } else {
                        // Wait until GlobalInit to run
                        init_functions_.emplace_back(function, description);
                    }
                }
        */
    }
    
    /**
      | Registry() is defined in .cpp file to
      | make registration work across multiple
      | shared libraries loaded with RTLD_LOCAL
      |
      */
    #[inline] pub fn registry(&mut self) -> *mut Caffe2InitializeRegistry {
        
        todo!();
        /*
            static Caffe2InitializeRegistry gRegistry;
                return &gRegistry;
        */
    }
}

///-----------------------------------------
pub struct InitRegisterer;

impl InitRegisterer {

    pub fn new() -> Self {
        todo!();
        /*
        internal::Caffe2InitializeRegistry::InitFunction function,
        bool run_early,
        const char* description,
        const char* name = nullptr) {
            internal::Caffe2InitializeRegistry::Registry()->Register(
                function, run_early, description, name);
        }
        */
    }
}

pub struct GlobalInitIsCalledGuard;

impl GlobalInitIsCalledGuard {
    pub fn new() -> Self {
        todo!();
        /*
        if (!GlobalInitAlreadyRun()) {
            LOG(WARNING)
                << "Caffe2 GlobalInit should be run before any other API calls.";
        }
        */
    }
}

/**
  | Keep track of stages of initialization
  | to differentiate between
  | 
  | (a) Re-entrant calls to GlobalInit
  | (e.g. caller registers a Caffe2 init
  | function which might in turn indirectly
  | invoke GlobalInit).
  | 
  | (b) Successive calls to GlobalInit,
  | which are handled as documented in init.h.
  | 
  | -----------
  | @note
  | 
  | this is NOT attempting to address thread-safety,
  | see comments in init.h.
  |
  */
pub enum State {
    Uninitialized,
    Initializing,
    Initialized,
}

#[inline] pub fn global_init_state<'a>() -> &'a mut State {
    
    todo!();
    /*
        static State state = State::Uninitialized;
                return state;
    */
}

/**
  | -----------
  | @brief
  | 
  | Determine whether GlobalInit has already
  | been run
  |
  */
#[inline] pub fn global_init_already_run() -> bool {

    todo!();
    /*
       return internal::GlobalInitState() == internal::State::Initialized;
       */
}

/**
  | -----------
  | @brief
  | 
  | Initialize the global environment
  | of caffe2.
  | 
  | Caffe2 uses a registration pattern
  | for initialization functions. Custom
  | initialization functions should take
  | the signature bool (*func)(int*, char***)
  | where the pointers to argc and argv are
  | passed in. Caffe2 then runs the initialization
  | in three phases:
  | 
  | (1) Functions registered with 
  | REGISTER_CAFFE2_EARLY_INIT_FUNCTION.
  |
  | Note that since it is possible the logger
  | is not initialized yet, any logging
  | in such early init functions may not
  | be printed correctly.
  | 
  | (2) Parses Caffe-specific commandline
  | flags, and initializes caffe logging.
  | 
  | (3) Functions registered with 
  | REGISTER_CAFFE2_INIT_FUNCTION.
  | 
  | If there is something wrong at each stage,
  | the function returns false. If the global
  | initialization has already been run,
  | the function returns false as well.
  | 
  | GlobalInit is re-entrant safe; a re-entrant
  | call will no-op and exit.
  | 
  | GlobalInit is safe to call multiple
  | times but not idempotent; successive
  | calls will parse flags and re-set caffe2
  | logging levels from flags as needed,
  | but NOT re-run early init and init functions.
  | 
  | GlobalInit is also thread-safe and
  | can be called concurrently.
  |
  */
#[inline] pub fn global_init(
    pargc: *mut i32, 
    pargv: *mut *mut *mut u8) -> bool 
{
    todo!();
    /*
        C10_LOG_API_USAGE_ONCE("caffe2.global_init");
        static std::recursive_mutex init_mutex;
        std::lock_guard<std::recursive_mutex> guard(init_mutex);
        internal::State& init_state = internal::GlobalInitState();
        static StaticLinkingProtector g_protector;
        bool success = true;

        // NOTE: if init_state == internal::State::Initializing at this point, do
        // nothing because that indicates a re-entrant call
        if (init_state == internal::State::Initialized) {
            VLOG(1) << "GlobalInit has already been called: re-parsing gflags only.";
            // Reparse command line flags
            success &= c10::ParseCommandLineFlags(pargc, pargv);
            UpdateLoggingLevelsFromFlags();
        } else if (init_state == internal::State::Uninitialized) {
            init_state = internal::State::Initializing;
            auto init_state_guard = MakeGuard([&] {
                // If an exception is thrown, go back to Uninitialized state
                if (init_state == internal::State::Initializing) {
                    init_state = internal::State::Uninitialized;
                }
            });

            success &= internal::Caffe2InitializeRegistry::Registry()
                ->RunRegisteredEarlyInitFunctions(pargc, pargv);
            CAFFE_ENFORCE(
                success, "Failed to run some early init functions for caffe2.");
            success &= c10::ParseCommandLineFlags(pargc, pargv);
            success &= InitCaffeLogging(pargc, *pargv);
            // Print out the current build version. Using cerr as LOG(INFO) might be off
            if (FLAGS_caffe2_version) {
                std::cerr << "Caffe2 build configuration: " << std::endl;
                for (const auto& it : GetBuildOptions()) {
                    std::cerr << "  " << std::setw(25) << std::left << it.first << " : "
                        << it.second << std::endl;
                }
            }
            // All other initialization functions.
            success &= internal::Caffe2InitializeRegistry::Registry()
                ->RunRegisteredInitFunctions(pargc, pargv);

            init_state =
                success ? internal::State::Initialized : internal::State::Uninitialized;
        }
        CAFFE_ENFORCE(success, "Failed to run some init functions for caffe2.");
        // TODO: if we fail GlobalInit(), should we continue?
        return success;
    */
}

/**
  | -----------
  | @brief
  | 
  | Initialize the global environment
  | without command line arguments
  | 
  | This is a version of the GlobalInit where
  | no argument is passed in.
  | 
  | On mobile devices, use this global init,
  | since we cannot pass the command line
  | options to caffe2, no arguments are
  | passed.
  |
  */
#[inline] pub fn global_init_nocmdline() -> bool {

    todo!();
    /*
    // This is a version of the GlobalInit where no argument is passed in.
    // On mobile devices, use this global init, since we cannot pass the
    // command line options to caffe2, no arguments are passed.
    int mobile_argc = 1;
    static char caffe2_name[] = "caffe2";
    char* mobile_name = &caffe2_name[0];
    char** mobile_argv = &mobile_name;
    return ::caffe2::GlobalInit(&mobile_argc, &mobile_argv);
    */
}

#[inline] pub fn unsafe_run_caffe_2init_function(
    name:  *const u8,
    pargc: *mut i32,
    pargv: *mut *mut *mut u8) -> bool 
{
    todo!();
    /*
        return internal::Caffe2InitializeRegistry::Registry()->RunNamedFunction(
            name, pargc, pargv);
    */
}
