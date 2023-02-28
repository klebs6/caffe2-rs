/*!
  | Include this file if you want to register
  | operators. It includes all functionality
  | needed to do so for you.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/op_registration/op_registration.h]

/**
  | The first argument of the schema might be of
  | type DispatchKeySet, in which case we remove
  | it.
  |
  | We do this because every argument in a function
  | schema is expected to be convertable to an
  | ivalue, but DispatchKeySet is not a type we
  | want the jit to be aware of.
  |
  | See Note [Plumbing Keys Through The Dispatcher]
  */
pub fn infer_function_schema_from_functor<KernelFunctor>() -> Box<FunctionSchema> {

    todo!();
        /*
            using func_type = typename remove_DispatchKeySet_arg_from_func<KernelFunctor>::func_type;
      return make_unique<FunctionSchema>(inferFunctionSchemaFlattenedReturns<func_type>());
        */
}

/**
  | KernelRegistrationConfig accumulates all
  | information from the config parameters passed
  | to a RegisterOperators::op() call into one
  | object.
  |
  */
pub struct KernelRegistrationConfig {
    dispatch_key:             Option<DispatchKey>,
    func:                     KernelFunction,
    cpp_signature:            Option<CppSignature>,
    inferred_function_schema: Box<FunctionSchema>,
}

impl Default for KernelRegistrationConfig {
    
    fn default() -> Self {
        todo!();
        /*
        : dispatch_key(nullopt),
        : func(),
        : cpp_signature(nullopt),
        : inferred_function_schema(nullptr),

        
        */
    }
}

pub struct RegisterOperatorsOptions {
    schema_or_name:      Option<Either<OperatorName,FunctionSchema>>,
    kernels:             Vec<KernelRegistrationConfig>,
    alias_analysis_kind: Option<AliasAnalysisKind>,
}

impl Default for RegisterOperatorsOptions {
    
    fn default() -> Self {
        todo!();
        /*
        : schema_or_name(nullopt),
        : kernels(),
        : alias_analysis_kind(nullopt),

        
        */
    }
}

impl RegisterOperatorsOptions {

    // internal-only for registering stack based kernels
    // template<KernelFunction::BoxedKernelFunction* kernel_func>
    pub fn kernel(&mut self, dispatch_key: DispatchKey) -> &mut Options {
        
        todo!();
        /*
            return move(*this).kernel(dispatch_key, KernelFunction::makeFromBoxedFunction<kernel_func>(), nullopt, nullptr);
        */
    }

    // internal-only for registering stack based catch-all kernels
    // template<KernelFunction::BoxedKernelFunction* kernel_func>
    pub fn catch_all_kernel(&mut self) -> &mut Options {
        
        todo!();
        /*
            return move(*this).kernel(nullopt, KernelFunction::makeFromBoxedFunction<kernel_func>(), nullopt, nullptr);
        */
    }

    /// internal only for registering caffe2 ops
    pub fn schema(&mut self, schema: FunctionSchema) -> &mut Options {
        
        todo!();
        /*
            TORCH_CHECK(!schemaOrName_.has_value(), "You can only specify the schema once per operator registration.");
            schemaOrName_ = make_right<OperatorName, FunctionSchema>(move(schema));
            return move(*this);
        */
    }

    /**
     | Use this to specify the schema for an
     | operator. You can also specify the operator
     | name only to have the function signature
     | part of the schema be inferred from the
     | kernel function.
     |
     | Example:
     |
     | > // Infer function signature from my_kernel_cpu
     | > static auto registry = RegisterOperators()
     | >     .op(RegisterOperators::options()
     | >         .schema("my_op")
     | >         .kernel<my_kernel_cpu>(DispatchKey::CPU));
     | >
     | >
     | > // Explicitly specify full schema
     | > static auto registry = RegisterOperators()
     | >     .op(RegisterOperators::options()
     | >         .schema("my_op(Tensor a) -> Tensor")
     | >         .kernel<my_kernel_cpu>(DispatchKey::CPU));
     */
    pub fn schema(&mut self, schema_or_name: &String) -> &mut Options {
        
        todo!();
        /*
            TORCH_CHECK(!schemaOrName_.has_value(), "Tried to register operator ", schemaOrName," but specified schema multiple times. You can only specify the schema once per operator registration.");

            #if !defined(EXPOSE_C2_OPS) && defined(CAFFE2_IS_XPLAT_BUILD)
            throw logic_error("Tried to register operator " + schemaOrName + ". We don't support registering c10 ops on mobile yet because the function schema parser isn't present in the mobile build.");
            #else
            schemaOrName_ = TorchJitparseSchemaOrName(schemaOrName);
          #endif

          return move(*this);
        */
    }

    /**
     | Use this to register an operator whose
     | kernel is implemented as a functor.
     |
     | The kernel is only called for inputs
     | matching the given dispatch key.
     |
     | You can register multiple kernels for
     | different dispatch keys.
     |
     | Example:
     |
     | > namespace {
     | >   class my_kernel_cpu final : public OperatorKernel {
     | >   
     | >     Tensor operator()(Tensor a, Tensor b) {...}
     | >   };
     | > }
     | >
     | > static auto registry = RegisterOperators()
     | >     .op(RegisterOperators::options()
     | >         .schema("my_op")
     | >         .kernel<my_kernel_cpu>(DispatchKey::CPU));
     |
     | The functor constructor can take arguments
     | to configure the kernel.
     |
     | The arguments are defined in the kernel
     | registration.
     |
     | Example:
     |
     | > namespace {
     | >   class my_kernel_cpu final : public OperatorKernel {
     | >   
     | >     explicit my_kernel_cpu(string some_configuration, int a, bool b)
     | >         : ... {...}
     | >
     | >     Tensor operator()(Tensor a, Tensor b) {...}
     | >   };
     | > }
     | >
     | > static auto registry = RegisterOperators()
     | >     .op(RegisterOperators::options()
     | >         .schema("my_op")
     | >         .kernel<my_kernel_cpu>(DispatchKey::CPU, "some_configuration", 3, true));
     */
    // template<class KernelFunctor, class... ConstructorParameters>
    // enable_if: only enable it if KernelFunctor is actually a functor
    // enable_if_t<is_functor<KernelFunctor>::value, Options&&> 
    pub fn kernel(&mut self, 
        dispatch_key:           DispatchKey,
        constructor_parameters: ConstructorParameters) -> Options {
        
        todo!();
        /*
            static_assert(is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from OperatorKernel. Please have the functor inherit from it.");
          static_assert(is_constructible<KernelFunctor, ConstructorParameters...>::value, "Wrong argument list for constructor of kernel functor. The arguments to kernel<Functor>(arguments...) must match one of the constructors of Functor.");

          return move(*this).kernel(
            move(dispatch_key),
            KernelFunction::makeFromUnboxedFunctor<false, KernelFunctor>(make_unique<KernelFunctor>(forward<ConstructorParameters>(constructorParameters)...)),
            CppSignature::make<KernelFunctor>(),
            inferFunctionSchemaFromFunctor<KernelFunctor>()
          );
        */
    }

    /**
     | Use this to register an operator whose
     | kernel is implemented as a functor.
     |
     | The kernel is a catch-all kernel, meaning
     | it's called independent from the
     | input. Dispatch is disabled for this
     | operator.
     |
     | Example:
     |
     | > namespace {
     | >   class my_kernel_cpu final : public OperatorKernel {
     | >   
     | >     Tensor operator()(Tensor a, Tensor b) {...}
     | >   };
     | > }
     | >
     | > static auto registry = RegisterOperators()
     | >     .op(RegisterOperators::options()
     | >         .schema("my_op")
     | >         .catchAllKernel<my_kernel_cpu>());
     |
     | The functor constructor can take arguments
     | to configure the kernel.
     |
     | The arguments are defined in the kernel
     | registration.
     |
     | Example:
     |
     | > namespace {
     | >   class my_kernel_cpu final : public OperatorKernel {
     | >   
     | >     explicit my_kernel_cpu(string some_configuration, int a, bool b)
     | >         : ... {...}
     | >
     | >     Tensor operator()(Tensor a, Tensor b) {...}
     | >   };
     | > }
     | >
     | > static auto registry = RegisterOperators()
     | >     .op(RegisterOperators::options()
     | >         .schema("my_op")
     | >         .catchAllKernel<my_kernel_cpu>("some_configuration", 3, true));
     */
    // template<class KernelFunctor, class... ConstructorParameters>
    // enable_if: only enable it if KernelFunctor is actually a functor
    // enable_if_t<is_functor<KernelFunctor>::value, Options&&> 
    pub fn catch_all_kernel(&mut self, constructor_parameters: ConstructorParameters) -> Options {
        
        todo!();
        /*
            static_assert(is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from OperatorKernel. Please have the functor inherit from it.");
          static_assert(is_constructible<KernelFunctor, ConstructorParameters...>::value, "Wrong argument list for constructor of kernel functor. The arguments to kernel<Functor>(arguments...) must match one of the constructors of Functor.");

          return move(*this).kernel(
            nullopt,
            KernelFunction::makeFromUnboxedFunctor<false, KernelFunctor>(make_unique<KernelFunctor>(forward<ConstructorParameters>(constructorParameters)...)),
            CppSignature::make<KernelFunctor>(),
            inferFunctionSchemaFromFunctor<KernelFunctor>()
          );
        */
    }

    /**
     | Use this to register an operator whose
     | kernel is implemented by a function.
     |
     | The kernel is only called for inputs
     | matching the given dispatch key.
     |
     | You can register multiple kernels for
     | different dispatch keys.
     |
     | Example:
     |
     | > namespace { Tensor my_kernel_cpu(Tensor a, Tensor b) {...} }
     | >
     | > static auto registry = RegisterOperators()
     | >     .op(RegisterOperators::options()
     | >         .schema("my_op")
     | >         .kernel<decltype(my_kernel_cpu), &my_kernel_cpu>(DispatchKey::CPU));
     */
    // template<class FuncType, FuncType* kernel_func>
    // enable_if: only enable it if FuncType is actually a function
    // enable_if_t<is_function_type<FuncType>::value, Options&&> 
    pub fn kernel(&mut self, dispatch_key: DispatchKey) -> Options {
        
        todo!();
        /*
            static_assert(!is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, "Tried to register a stackbased (i.e. internal) kernel function using the public kernel<...>() API. Please either use the internal kernel(...) API or also implement the kernel function as defined by the public API.");
          static_assert(kernel_func != nullptr, "Kernel function cannot be nullptr");

          return move(*this).kernel(
            move(dispatch_key),
            KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernel_func)),
            CppSignature::make<FuncType>(),
            // TODO Do schema inference without relying on WrapFunctionIntoFunctor
            inferFunctionSchemaFromFunctor<typename WrapFunctionIntoFunctor<CompileTimeFunctionPointer<FuncType, kernel_func>>::type>()
          );
        */
    }

    /**
     | Use this to register an operator whose
     | kernel is implemented by a function.
     |
     | The kernel is a catch-all kernel, meaning
     | it's called independent from the
     | input. Dispatch is disabled for this
     | operator.
     |
     | Example:
     |
     | > namespace { Tensor my_kernel_cpu(Tensor a, Tensor b) {...} }
     | >
     | > static auto registry = RegisterOperators()
     | >     .op(RegisterOperators::options()
     | >         .schema("my_op")
     | >         .catchAllKernel<decltype(my_kernel_cpu), &my_kernel_cpu>());
     */
    // template<class FuncType, FuncType* kernel_func>
    // enable_if: only enable it if FuncType is actually a function
    // enable_if_t<is_function_type<FuncType>::value, Options&&> 
    pub fn catch_all_kernel(&mut self) -> Options {
        
        todo!();
        /*
            static_assert(!is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, "Tried to register a stackbased (i.e. internal) kernel function using the public kernel<...>() API. Please either use the internal kernel(...) API or also implement the kernel function as defined by the public API.");
          static_assert(kernel_func != nullptr, "Kernel function cannot be nullptr");

          return move(*this).kernel(
            nullopt,
            KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernel_func)),
            CppSignature::make<FuncType>(),
            // TODO Do schema inference without relying on WrapFunctionIntoFunctor
            inferFunctionSchemaFromFunctor<typename WrapFunctionIntoFunctor<CompileTimeFunctionPointer<FuncType, kernel_func>>::type>()
          );
        */
    }

    // template<class FuncType>
    // enable_if: only enable it if FuncType is actually a function
    // enable_if_t<is_function_type<FuncType>::value, Options&&> 
    pub fn kernel(&mut self, 
        dispatch_key: DispatchKey,
        kernel_func:  *mut FuncType) -> Options {
        
        todo!();
        /*
            static_assert(!is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, "Tried to register a stackbased (i.e. internal) kernel function using the public kernel<...>() API. Please either use the internal kernel(...) API or also implement the kernel function as defined by the public API.");
          TORCH_INTERNAL_ASSERT(kernel_func != nullptr, "Kernel function cannot be nullptr");

          return move(*this).kernel(
            move(dispatch_key),
            KernelFunction::makeFromUnboxedRuntimeFunction(kernel_func),
            CppSignature::make<FuncType>(),
            // TODO Do schema inference without relying on WrapFunctionIntoFunctor
            inferFunctionSchemaFromFunctor<WrapFunctionIntoRuntimeFunctor<decay_t<FuncType>>>()
          );
        */
    }

    // template<class FuncType>
    // enable_if: only enable it if FuncType is actually a function
    // enable_if_t<is_function_type<FuncType>::value, Options&&> 
    pub fn catch_all_kernel(&mut self, kernel_func: *mut FuncType) -> Options {
        
        todo!();
        /*
            static_assert(!is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, "Tried to register a stackbased (i.e. internal) kernel function using the public kernel<...>() API. Please either use the internal kernel(...) API or also implement the kernel function as defined by the public API.");
          TORCH_INTERNAL_ASSERT(kernel_func != nullptr, "Kernel function cannot be nullptr");

          return move(*this).kernel(
            nullopt,
            KernelFunction::makeFromUnboxedRuntimeFunction(kernel_func),
            CppSignature::make<FuncType>(),
            // TODO Do schema inference without relying on WrapFunctionIntoFunctor
            inferFunctionSchemaFromFunctor<WrapFunctionIntoRuntimeFunctor<decay_t<FuncType>>>()
          );
        */
    }

    /**
     | Use this to register an operator whose
     | kernel is implemented as a lambda.
     |
     | The kernel is only called for inputs
     | matching the given dispatch key.
     |
     | You can register multiple kernels for
     | different dispatch keys.
     |
     | The lambda must be stateless, i.e. not have
     | a capture. If your kernel needs to store
     | some configuration parameters, write the
     | kernel as a functor instead.
     |
     | Example:
     |
     | > static auto registry = RegisterOperators()
     | >     .op(RegisterOperators::options()
     | >         .schema("my_op")
     | >         .kernel(DispatchKey::CPU, [] (Tensor a) -> Tensor {...}));
     */
    // template<class Lambda>
    // enable_if: only enable it if Lambda is a functor (note: lambdas are functors)
    // enable_if_t< is_functor<decay_t<Lambda>>::value && !is_same<typename infer_function_traits_t<decay_t<Lambda>>::func_type, KernelFunction::BoxedKernelFunction>::value, Options&&> 
    pub fn kernel(&mut self, 
        dispatch_key: DispatchKey,
        functor:      Lambda) -> Options {
        
        todo!();
        /*
            static_assert(!is_base_of<OperatorKernel, decay_t<Lambda>>::value, "The kernel(x) API for registering a kernel is only meant to be used with lambdas. Your kernel is a functor. Please use the kernel<Functor>() API instead.");

          // We don't support stateful lambdas (i.e. lambdas with a capture), because their
          // behavior would be nonobvious. A functor kernel with cache gets a new instance of
          // its cache each time the kernel is looked up from the dispatch table.
          // A lambda with a capture would be global and share its capture between all kernel lookups.
          // So, instead of making users having to think about it (including the thread-safety
          // issues this causes), let's just forbid stateful lambdas altogether.
          static_assert(is_stateless_lambda<decay_t<Lambda>>::value, "The kernel(x) API for registering a kernel only works for stateless lambdas (i.e. lambdas without captures). If you need a cache, please use the functor based API kernel<Functor>() instead.");

          return move(*this).kernel(
            move(dispatch_key),
            KernelFunction::makeFromUnboxedLambda(forward<Lambda>(functor)),
            CppSignature::make<Lambda>(),
            // TODO Do schema inference without relying on WrapFunctionIntoRuntimeFunctor
            inferFunctionSchemaFromFunctor<WrapFunctionIntoRuntimeFunctor<decay_t<Lambda>>>()
          );
        */
    }

    /**
     | Use this to register an operator whose
     | kernel is implemented as a lambda.
     |
     | The kernel is a catch-all kernel, meaning
     | it's called independent from the
     | input. Dispatch is disabled for this
     | operator.
     |
     | The lambda must be stateless, i.e. not have
     | a capture. If your kernel needs to store
     | some configuration parameters, write the
     | kernel as a functor instead.
     |
     | Example:
     |
     | > static auto registry = RegisterOperators()
     | >     .op(RegisterOperators::options()
     | >         .schema("my_op")
     | >         .catchAllKernel([] (Tensor a) -> Tensor {...}));
     */
    // template<class Lambda>
    // enable_if: only enable it if Lambda is a functor (note: lambdas are functors)
    // enable_if_t< is_functor<decay_t<Lambda>>::value && !is_same<typename infer_function_traits_t<decay_t<Lambda>>::func_type, KernelFunction::BoxedKernelFunction>::value, Options&&> 
    pub fn catch_all_kernel(&mut self, lambda: Lambda) -> Options {
        
        todo!();
        /*
            static_assert(!is_base_of<OperatorKernel, decay_t<Lambda>>::value, "The kernel(x) API for registering a kernel is only meant to be used with lambdas. Your kernel is a functor. Please use the kernel<Functor>() API instead.");

          // We don't support stateful lambdas (i.e. lambdas with a capture), because their
          // behavior would be nonobvious.
          // A lambda with a capture would be global and share its capture between all kernel lookups.
          // This would be a likely source for unexpected race conditions, so we forbid it.
          // If a kernel really needs global state, they can just have regular global state
          // in their .cpp file next to the kernel lambda.
          static_assert(is_stateless_lambda<decay_t<Lambda>>::value, "The kernel(x) API for registering a kernel only works for stateless lambdas (i.e. lambdas without captures). If you need a cache, please use the functor based API kernel<Functor>() instead.");

          return move(*this).kernel(
            nullopt,
            KernelFunction::makeFromUnboxedLambda(forward<Lambda>(lambda)),
            CppSignature::make<Lambda>(),
            // TODO Do schema inference without relying on WrapFunctionIntoRuntimeFunctor
            inferFunctionSchemaFromFunctor<WrapFunctionIntoRuntimeFunctor<decay_t<Lambda>>>()
          );
        */
    }
    
    pub fn alias_analysis(&mut self, alias_analysis_kind: AliasAnalysisKind) -> &mut Options {
        
        todo!();
        /*
            TORCH_CHECK(!aliasAnalysisKind_.has_value(), "You can only call aliasAnalysis() once per operator registration.");
          aliasAnalysisKind_ = aliasAnalysisKind;
          return move(*this);
        */
    }
    
    pub fn kernel(&mut self, 
        dispatch_key:             Option<DispatchKey>,
        func:                     KernelFunction,
        cpp_signature:            Option<CppSignature>,
        inferred_function_schema: Box<FunctionSchema>) -> &mut Options {
        
        todo!();
        /*
            KernelRegistrationConfig config;
          config.dispatch_key = dispatch_key;
          config.func = move(func);
          config.cpp_signature = move(cpp_signature);
          config.inferred_function_schema = move(inferred_function_schema);
          kernels.push_back(move(config));
          return move(*this);
        */
    }
}

/**
 | An instance of this class handles the
 | registration for one or more operators.
 |
 | Make sure you keep the RegisterOperators
 | instance around since it will deregister the
 | operator it's responsible for in its
 | destructor.
 |
 | Example:
 |
 | > namespace {
 | >   class my_kernel_cpu final : public OperatorKernel {
 | >   
 | >     Tensor operator()(Tensor a, Tensor b) {...}
 | >   };
 | > }
 | >
 | > static auto registry = RegisterOperators()
 | >     .op(RegisterOperators::options()
 | >         .schema("my_op")
 | >         .kernel<my_kernel_cpu>(DispatchKey::CPU));
 */
pub struct RegisterOperators {
    registrars: Vec<RegistrationHandleRAII>,
}

impl RegisterOperators {

    /**
      | Call this to get an instance of registration
      | options, which can be passed to a call
      | to RegisterOperators::op() to specify
      | these options for the operator registration.
      | 
      | See class doc comment for examples.
      |
      */
    pub fn options() -> Options {
        
        todo!();
        /*
            return {};
        */
    }

    /**
      | Call this to register an operator. See
      | class doc comment for examples.
      |
      */
    pub fn op(&mut self, options: Options) -> &mut RegisterOperators {
        
        todo!();
        /*
            checkSchemaAndRegisterOp_(move(options));
        return move(*this);
        */
    }

    /**
      | Regular mutator version of the && version
      | above
      |
      */
    pub fn op(&mut self, options: Options) -> &mut RegisterOperators {
        
        todo!();
        /*
            checkSchemaAndRegisterOp_(move(options));
        return *this;
        */
    }

    /**
      | This is a shorthand for RegisterOperators::op(Options)
      | where you can specify the operator schema
      | outside of the options parameter.
      | 
      | See class doc comment for examples.
      |
      */
    pub fn op(&mut self, 
        schema_or_name: &String,
        options:        Options) -> &mut RegisterOperators {
        let options: Options = options.unwrap_or(RegisterOperatorsOptions);

        todo!();
        /*
            return move(*this).op(move(options).schema(schemaOrName));
        */
    }

    /**
      | internal only for registering caffe2
      | ops
      |
      */
    pub fn op(&mut self, 
        schema:  FunctionSchema,
        options: Options) -> &mut RegisterOperators {
        
        todo!();
        /*
            return move(*this).op(move(options).schema(move(schema)));
        */
    }
    
    pub fn new<FuncType>(
        schema_or_name: &String,
        func:           FuncType,
        options:        Options) -> Self {

        let options: Options = options.unwrap_or(RegisterOperatorsOptions);

        todo!();
        /*


            : RegisterOperators() 
        move(*this).op(schemaOrName, forward<FuncType>(func), move(options));
        */
    }

  /**
   | This API registers an operator based on
   | a kernel function pointer.
   |
   | Given a kernel
   |
   | > namespace { Tensor my_kernel_cpu(Tensor a, Tensor b) {...} }
   |
   | This API looks like:
   |
   | > static auto registry = RegisterOperators()
   | >     .op("my_op", &my_kernel_cpu);
   |
   | If your kernel is small and the overhead of calling it matters,
   | then this API might be the wrong choice since the following API
   | has a slightly lower overhead for calling into the kernel:
   |
   | > static auto registry = RegisterOperators()
   | >     .op("my_op", RegisterOperators::options()
   | >         .kernel<decltype(my_kernel_cpu), &my_kernel_cpu>());
   |
   | Or, alternatively, write your kernel as a functor:
   |
   | > namespace {
   | >   class my_kernel_cpu final : public OperatorKernel {
   | >   
   | >     Tensor operator()(Tensor a, Tensor b) {...}
   | >   };
   | > }
   | >
   | > static auto registry = RegisterOperators()
   | >     .op("my_op", RegisterOperators::options()
   | >         .kernel<my_kernel_cpu>());
   */
   // template<class FuncType>
   // enable_if: only enable it if FuncType is actually a function, but not a stack based BoxedKernelFunction.
   // enable_if_t<is_function_type<FuncType>::value && !is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, RegisterOperators&&>
    pub fn op(&mut self, 
        schema_or_name: &String,
        func:           *mut FuncType,
        options:        Options) -> RegisterOperators {

        let options: Options = options.unwrap_or(RegisterOperatorsOptions);

        todo!();
        /*
            constexpr bool AllowLegacyTypes = true;
         return move(*this).op(move(options).schema(schemaOrName).kernel(
           nullopt,
           KernelFunction::makeFromUnboxedRuntimeFunction<AllowLegacyTypes>(func),
           CppSignature::make<FuncType>(),
           // TODO Do schema inference without relying on WrapFunctionIntoRuntimeFunctor
           inferFunctionSchemaFromFunctor<WrapFunctionIntoRuntimeFunctor<decay_t<FuncType>>>()
         ));
        */
    }

   /**
    | This API registers an operator based on
    | a kernel lambda.
    |
    | This API looks like:
    |
    | > static auto registry = RegisterOperators()
    | >     .op("my_op", [] (Tensor a, Tensor b) {...});
    |
    | This is equivalent to:
    |
    | > static auto registry = RegisterOperators()
    | >     .op("my_op", RegisterOperators::options()
    | >         .catchAllKernel([] (Tensor a, Tensor b) {...}));
    |
    */
    // template<class Lambda>
    // enable_if: only enable it if Lambda is actually a stateless lambda
    // enable_if_t<is_functor<Lambda>::value && is_stateless_lambda<decay_t<Lambda>>::value, RegisterOperators&&>
    pub fn op(&mut self, 
        schema_or_name: &String,
        lambda:         Lambda,
        options:        Options) -> RegisterOperators {

        let options: Options = options.unwrap_or(RegisterOperatorsOptions);

        todo!();
        /*
            static_assert(!is_base_of<OperatorKernel, Lambda>::value, "OperatorKernel is part of the new kernel registration API and shouldn't be used together with the deprecated registration API. Please use the new RegisterOperators::options().kernel() based API instead.");

          constexpr bool AllowLegacyTypes = true;
          return move(*this).op(move(options).schema(schemaOrName).kernel(
            nullopt,
            KernelFunction::makeFromUnboxedLambda<AllowLegacyTypes>(forward<Lambda>(lambda)),
            CppSignature::make<Lambda>(),
            // TODO Do schema inference without relying on WrapFunctionIntoRuntimeFunctor
            inferFunctionSchemaFromFunctor<WrapFunctionIntoRuntimeFunctor<decay_t<Lambda>>>()
          ));
        */
    }

    // template<class Lambda>
    // enable_if: only enable it if Lambda is actually a functor but not a stateless lambda
    // enable_if_t<is_functor<Lambda>::value && !is_stateless_lambda<decay_t<Lambda>>::value, RegisterOperators&&>
    #[deprecated = "Registering operator kernels with stateful lambdas (i.e. lambdas with a capture) has non-obvious behavior. This is deprecated. Please use a lambda without a capture or a functor class instead."]
    pub fn op(&mut self, 
        schema_or_name: &String,
        lambda:         Lambda,
        options:        Options) -> RegisterOperators {

        let options: Options = options.unwrap_or(RegisterOperatorsOptions);

        todo!();
        /*
            static_assert(!is_base_of<OperatorKernel, Lambda>::value, "OperatorKernel is part of the new kernel registration API and shouldn't be used together with the deprecated registration API. Please use the new RegisterOperators::options().kernel() based API instead.");

          constexpr bool AllowLegacyTypes = true;
          return move(*this).op(move(options).schema(schemaOrName).kernel(
            nullopt,
            KernelFunction::makeFromUnboxedLambda<AllowLegacyTypes>(forward<Lambda>(lambda)),
            CppSignature::make<Lambda>(),
            // TODO Do schema inference without relying on WrapFunctionIntoRuntimeFunctor
            inferFunctionSchemaFromFunctor<WrapFunctionIntoRuntimeFunctor<decay_t<Lambda>>>()
          ));
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/op_registration/op_registration.cpp]

const_assert!(is_nothrow_move_constructible::<Option<RegistrationHandleRAII>>());
const_assert!(is_nothrow_move_assignable::<Option<RegistrationHandleRAII>>());

impl RegisterOperators {
    
    pub fn check_schema_and_register_op(&mut self, options: Options)  {
        
        todo!();
        /*
            TORCH_CHECK(options.schemaOrName_.has_value(), "In operator registration: Tried to register an operator without specifying a schema or operator name.");
      if (options.schemaOrName_->is_right()) {
        // schema was explicitly specified.

        checkNoDuplicateKernels_(options);

        registerOp_(move(options));
      } else {
        // schema wasn't explicitly specified. Take the inferred schema for registering the op.

        OperatorName name = move(*options.schemaOrName_).left();
        FunctionSchema inferred_schema = inferSchemaFromKernels_(name, options);

        options.schemaOrName_ = make_right<OperatorName, FunctionSchema>(
          move(name.name),
          move(name.overload_name),
          inferred_schema.arguments(),
          inferred_schema.returns(),
          inferred_schema.is_vararg(),
          inferred_schema.is_varret()
        );

        checkNoDuplicateKernels_(options);

        // This would have unexpected behavior since an inferred schema will not
        // have aliasing annotations.
        TORCH_CHECK(
            options.aliasAnalysisKind_ != AliasAnalysisKind::FROM_SCHEMA,
            "In operator registration: Tried to register operator ",
            options.schemaOrName_->right(),
            " with AliasAnalysisKind::FROM_SCHEMA, but the schema is inferred.");

        // Register all kernels with the schema we inferred
        registerOp_(move(options));
      }
        */
    }
    
    pub fn infer_schema_from_kernels(&mut self, 
        op_name: &OperatorName,
        options: &RegisterOperatorsOptions) -> FunctionSchema {
        
        todo!();
        /*
            TORCH_CHECK(options.kernels.size() > 0, "Cannot infer operator schema in registration of operator ", opName, " because there is no kernel specified.");

      optional<FunctionSchema> inferred_schema = nullopt;
      for (const auto& kernel : options.kernels) {
        if (nullptr != kernel.inferred_function_schema.get()) {
          if (!inferred_schema.has_value()) {
            inferred_schema = *kernel.inferred_function_schema;
            break;
          }
        }
      }
      TORCH_CHECK(inferred_schema.has_value(), "Cannot infer operator schema for this kind of kernel in registration of operator ", opName, ". Please explicitly specify the operator schema or specify at least one kernel for which we can infer the schema.");

      return *inferred_schema;
        */
    }
    
    pub fn check_no_duplicate_kernels(&mut self, options: &Options)  {
        
        todo!();
        /*
            unordered_set<DispatchKey> dispatch_keys;
      bool has_catchall_kernel = false;

      for (const auto& kernel : options.kernels) {
        if (kernel.dispatch_key.has_value()) {
          TORCH_CHECK(0 == dispatch_keys.count(*kernel.dispatch_key), "In operator registration: Tried to register multiple kernels with same dispatch key ", *kernel.dispatch_key, " for operator schema ", toString(options.schemaOrName_->right()));
          dispatch_keys.insert(*kernel.dispatch_key);
        } else {
          TORCH_CHECK(!has_catchall_kernel, "In operator registration: Tried to register multiple catch-all kernels for operator schema ", toString(options.schemaOrName_->right()));
          has_catchall_kernel = true;
        }
      }
        */
    }
    
    pub fn register_op(&mut self, options: Options)  {
        
        todo!();
        /*
            FunctionSchema schema = move(*options.schemaOrName_).right();

      // HACK: bong in the alias analysis kind from the legacy API directly
      // into schema
      if (options.aliasAnalysisKind_.has_value()) {
        schema.setAliasAnalysis(*options.aliasAnalysisKind_);
      }

      OperatorName op_name = schema.operator_name();

      registrars_.emplace_back(
        Dispatcher::singleton().registerDef(move(schema), "registered by RegisterOperators")
      );

      for (auto& kernel : options.kernels) {
        registrars_.emplace_back(
          Dispatcher::singleton().registerImpl(op_name, kernel.dispatch_key, move(kernel.func), move(kernel.cpp_signature), move(kernel.inferred_function_schema), "registered by RegisterOperators")
        );
      }
        */
    }
}
