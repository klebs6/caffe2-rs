crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h]

impl KernelFunction {
    
    pub fn new() -> Self {
    
        todo!();
        /*


            : functor_(nullptr)
    , boxed_kernel_func_(nullptr)
    , unboxed_kernel_func_(nullptr)
        */
    }
    
    pub fn new(
        functor:             Box<OperatorKernel>,
        boxed_kernel_func:   *mut InternalBoxedKernelFunction,
        unboxed_kernel_func: *mut void) -> Self {
    
        todo!();
        /*


            : functor_(std::move(functor))
    , boxed_kernel_func_(boxed_kernel_func)
    , unboxed_kernel_func_(unboxed_kernel_func)
        */
    }
    
    #[inline] pub fn is_valid_unboxed(&self) -> bool {
        
        todo!();
        /*
            return unboxed_kernel_func_ != nullptr;
        */
    }
    
    #[inline] pub fn make_boxed_function(&mut self, 
        _0:        *mut OperatorKernel,
        op_handle: &OperatorHandle,
        _2:        DispatchKeySet,
        stack:     *mut Stack)  {
        
        todo!();
        /*
            // Note that we're dropping the DispatchKeySet argument.
        // See Note [Plumbing Keys Through The Dispatcher 2] for details.
        func(opHandle, stack);
        */
    }

    #[inline] pub fn make_boxed_function(&mut self, 
        _0:        *mut OperatorKernel,
        op_handle: &OperatorHandle,
        ks:        DispatchKeySet,
        stack:     *mut Stack)  {
        
        todo!();
        /*
            // See Note [Plumbing Keys Through The Dispatcher 2] for details.
        func(opHandle, ks, stack);
        */
    }
    
    #[inline] pub fn is_valid(&self) -> bool {
        
        todo!();
        /*
            return boxed_kernel_func_ != nullptr;
        */
    }
    
    #[inline] pub fn is_fallthrough(&self) -> bool {
        
        todo!();
        /*
            return boxed_kernel_func_ == &fallthrough_kernel;
        */
    }
    
    #[inline] pub fn call_boxed(&self, 
        op_handle:        &OperatorHandle,
        dispatch_key_set: DispatchKeySet,
        stack:            *mut Stack)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            boxed_kernel_func_ != nullptr,
            "Tried to call KernelFunction::callBoxed() on an uninitialized KernelFunction."
        );
        (*boxed_kernel_func_)(functor_.get(), opHandle, dispatchKeySet, stack);
        */
    }

    #[inline] pub fn call_unboxed_kernel_function(&mut self, 
        unboxed_kernel_func: *mut void,
        functor:             *mut OperatorKernel,
        dispatch_key_set:    DispatchKeySet,
        args:                Args) -> Return {
        
        todo!();
        /*
            using ActualSignature = Return (OperatorKernel*, DispatchKeySet, Args...);
        ActualSignature* func = reinterpret_cast<ActualSignature*>(unboxed_kernel_func);
        return (*func)(functor, dispatchKeySet, std::forward<Args>(args)...);
        */
    }
    
    #[inline(always)] pub fn call(&self, 
        op_handle:        &OperatorHandle,
        dispatch_key_set: DispatchKeySet,
        args:             Args) -> Return {
        
        todo!();
        /*
            // note: Args above is intentionally not Args&&. We don't want perfect
        // forwarding, which would require Args to be deduced, but instead we
        // want callers to explicitly specify the Args.

        if (C10_LIKELY(unboxed_kernel_func_ != nullptr)) {
            return callUnboxedKernelFunction<Return, Args...>(unboxed_kernel_func_, functor_.get(), dispatchKeySet, std::forward<Args>(args)...);
        }

        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            boxed_kernel_func_ != nullptr,
            "Tried to call KernelFunction::call() on an uninitialized KernelFunction."
        );

        return impl::BoxedKernelWrapper<Return(Args...)>::call(
            boxed_kernel_func_,
            functor_.get(),
            opHandle,
            dispatchKeySet,
            std::forward<Args>(args)...
        );
        */
    }

    #[inline] pub fn make_from_boxed_function(&mut self) -> KernelFunction {
        
        todo!();
        /*
            return KernelFunction(
            nullptr,  // no functor_ object
            &make_boxed_function<func>,
            nullptr  // no unboxed function pointer
        );
        */
    }
    
    #[inline] pub fn make_from_boxed_function(&mut self) -> KernelFunction {
        
        todo!();
        /*
            return KernelFunction(
            nullptr,  // no functor_ object
            &make_boxed_function<func>,
            nullptr  // no unboxed function pointer
        );
        */
    }
    
    #[inline] pub fn make_fallthrough(&mut self) -> KernelFunction {
        
        todo!();
        /*
            return KernelFunction(
            nullptr,  // no functor_ object
            &fallthrough_kernel,
            nullptr  // no unboxed function pointer
        );
        */
    }
    
    #[inline] pub fn make_ambiguous_autograd_other(&mut self) -> KernelFunction {
        
        todo!();
        /*
            return KernelFunction(
            nullptr,  // no functor_ object
            &ambiguous_autogradother_kernel,
            nullptr  // no unboxed function pointer
        );
        */
    }
    
    #[inline] pub fn make_named_not_supported(&mut self) -> KernelFunction {
        
        todo!();
        /*
            return KernelFunction(
            nullptr,  // no functor_ object
            &named_not_supported_kernel,
            nullptr  // no unboxed function pointer
        );
        */
    }

    #[inline] pub fn make_from_unboxed_functor(&mut self, kernel_functor: Box<OperatorKernel>) -> KernelFunction {
        
        todo!();
        /*
            #ifndef NDEBUG
      // This assertion is costly for build time so it's debug-gated.
        static_assert(guts::is_functor<KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor> but the argument is not a functor.");
    #endif
        static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

        return KernelFunction(
            std::move(kernelFunctor),
            &impl::make_boxed_from_unboxed_functor<KernelFunctor, AllowLegacyTypes>::call,
            reinterpret_cast<void*>(&impl::wrap_kernel_functor_unboxed<KernelFunctor>::call)
        );
        */
    }

    #[inline] pub fn make_from_unboxed_function(&mut self, func_ptr: FuncPtr) -> KernelFunction {
        
        todo!();
        /*
            static_assert(is_compile_time_function_pointer<FuncPtr>::value, "Tried to call KernelFunction::makeFromUnboxedFunction with an invalid parameter. It must be a function pointer created with TORCH_FN.");
        static_assert(!std::is_same<typename FuncPtr::FuncType, BoxedKernelFunction>::value, "Tried to call KernelFunction::makeFromUnboxedFunction with a boxed function pointer. Please use KernelFunction::makeFromBoxedFunction instead.");
        static_assert(FuncPtr::func_ptr() != nullptr, "Kernel function cannot be nullptr");

    #if !defined(C10_MOBILE)
        return makeFromUnboxedFunctor<AllowLegacyTypes, typename impl::WrapFunctionIntoFunctor<FuncPtr>::type>(
            guts::make_unique_base<OperatorKernel, typename impl::WrapFunctionIntoFunctor<FuncPtr>::type>()
        );
    #else
        // On mobile, we rather want to optimize for binary size than for performance,
        // so let's not inline the kernel into the wrapper but use makeFromUnboxedRuntimeFunction
        // instead.
        return makeFromUnboxedRuntimeFunction(func_ptr.func_ptr());
    #endif
        */
    }

    #[inline] pub fn make_from_unboxed_runtime_function(&mut self, func: *mut FuncType) -> KernelFunction {
        
        todo!();
        /*
            static_assert(guts::is_function_type<FuncType>::value, "Tried to call KernelFunction::makeFromUnboxedRuntimeFunction with a non-function type.");
        static_assert(!std::is_same<FuncType, BoxedKernelFunction>::value, "Tried to call KernelFunction::makeFromUnboxedRuntimeFunction with a boxed function pointer. Please use KernelFunction::makeFromBoxedFunction instead.");
        TORCH_INTERNAL_ASSERT(func != nullptr, "Kernel function cannot be nullptr");

        return makeFromUnboxedFunctor<AllowLegacyTypes, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>>(
            guts::make_unique_base<OperatorKernel, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>>(func)
        );
        */
    }

    //template<bool AllowLegacyTypes, class Lambda>
    //enable_if_t<is_stateless_lambda<decay_t<Lambda>>::value, KernelFunction> 
    #[inline] pub fn new(lambda: Lambda) -> Self {
    
        todo!();
        /*


            static_assert(is_functor<decay_t<Lambda>>::value, "Tried to call KernelFunction::makeFromUnboxedLambda with a non-lambda type.");

    #if !defined(C10_MOBILE)
        return makeFromUnboxedFunctor<AllowLegacyTypes, impl::WrapFunctionIntoRuntimeFunctor<decay_t<Lambda>>>(
            make_unique_base<OperatorKernel, impl::WrapFunctionIntoRuntimeFunctor<decay_t<Lambda>>>(forward<Lambda>(lambda))
        );
    #else
        // On mobile, we rather want to optimize for binary size than for performance,
        // so let's not inline the kernel into the wrapper but use makeFromUnboxedRuntimeFunction
        // instead.
        using FuncType = typename infer_function_traits_t<decay_t<Lambda>>::func_type;
        return makeFromUnboxedRuntimeFunction<AllowLegacyTypes, FuncType>(lambda);
    #endif
        */
    }

    //template<bool AllowLegacyTypes, class Lambda>
    //enable_if_t<!is_stateless_lambda<decay_t<Lambda>>::value, KernelFunction> 
    pub fn new(lambda: Lambda) -> Self {
    
        todo!();
        /*


            static_assert(is_functor<decay_t<Lambda>>::value, "Tried to call KernelFunction::makeFromUnboxedLambda with a non-lambda type.");

        return makeFromUnboxedFunctor<AllowLegacyTypes, impl::WrapFunctionIntoRuntimeFunctor<decay_t<Lambda>>>(
            make_unique_base<OperatorKernel, impl::WrapFunctionIntoRuntimeFunctor<decay_t<Lambda>>>(forward<Lambda>(lambda))
        );
        */
    }
}
