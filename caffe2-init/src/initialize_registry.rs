crate::ix!();

pub type InitFunction = fn(i: *mut i32, c: *mut *mut *mut u8) -> bool;

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
