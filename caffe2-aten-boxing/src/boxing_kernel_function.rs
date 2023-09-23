crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/boxing/KernelFunction.h]

/**
  | TODO Instead of this, move torch::jit::Stack
  | to the c10 namespace.
  |
  */
pub type Stack = JitStack; 

/**
  | KernelFunction is similar to function
  | but stores a kernel function.
  | 
  | You can create a KernelFunction from
  | a boxed or unboxed function/functor/lambda
  | and call it in a boxed or unboxed way.
  | If the way it was created doesn't match
  | the way it was called, it will do boxing
  | or unboxing as necessary.
  |
  */
pub struct KernelFunction {
    functor:             Arc<OperatorKernel>,
    boxed_kernel_func:   *mut InternalBoxedKernelFunction,
    unboxed_kernel_func: *mut void,
}

pub mod kernel_function {

    use super::*;

    /**
      | This is how boxed kernels are actually stored
      |
      | Note [Plumbing Keys Through The Dispatcher]
      |
      | Benchmarks have shown that it is expensive
      | for the dispatcher to read from thread-local
      | storage (TLS) upon every dispatch call into
      | order to compute which kernel to dispatch to.
      |
      | To mitigate this, we've updated the calling
      | convention inside the dispatcher to expect
      | every kernel that it stores to have a first
      | argument of type DispatchKeySet.
      |
      | What are the invariants of the DispatchKeySet
      | when it gets passed to a kernel?
      |
      | - All keys to the left of the current
      |   dispatch key have been masked
      |   out. (e.g. a Tracing kernel that takes in
      |   the DispatchKeySet will expect the highest
      |   bit to be DispatchKey::Tracer)
      |
      | - All other keys that dispatcher normally
      |   would have computed through TLS + global
      |   state + op arguments are still in the set.
      |
      | Kernels can then opt into using this keyset
      | to save the dispatcher from doing repeated
      | work during redispatches: recalculating the
      | highest-priority dispatch key, which involves
      | reading from TLS.
      |
      | Instead, the kernels that opt in will
      | calculate an updated DispatchKeySet directly
      | from the old one, and pass the updated set
      | directly into the dispatcher upon
      | redispatching.
      |
      | This is an opt-in mechanism: Kernels can
      | automatically opt in by setting the first
      | argument in their signature to be of type
      | DispatchKeySet. See the kernels in
      | VariableTypeEverything.cpp and
      | TraceTypeEverything.cpp for examples.
      |
      | The mechanism for optionally passing that
      | DispatchKeySet into the kernel lives in
      | make_boxed_from_unboxed_functor.h.
      |
      | See Note [Plumbing Keys Through The
      | Dispatcher 2] for details.
      |
      */
    lazy_static!{
        /*
        using InternalBoxedKernelFunction = void(OperatorKernel*, const OperatorHandle&, DispatchKeySet, Stack*);
        */
    }

    /**
      | This is the public API for how boxed kernels
      | are defined
      |
      */
    lazy_static!{
        /*
        using BoxedKernelFunction = void(const OperatorHandle&, Stack*);
          using BoxedKernelFunction_withDispatchKeys = void(const OperatorHandle&, DispatchKeySet, Stack*);
        */
    }
}

impl KernelFunction {

    /**
      | Fast path for dispatch to allow not touching
      | the boxed kernel in the common case where
      | unboxed is available.
      |
      */
    pub fn is_valid_unboxed(&self) -> bool {
        
        todo!();
        /*
        
        */
    }

    /**
      | Call the function in a boxed way.
      | 
      | If the kernel function was created with
      | an unboxed function, this will call
      | an unboxing wrapper which then calls
      | into that unboxed function.
      | 
      | Example: > void boxed_func(OperatorKernel*,
      | Stack* stack) {...} > KernelFunction
      | func = KernelFunction::makeFromBoxedFunction(&boxed_func);
      | > Tensor result = func.callBoxed(stack);
      | 
      | Or, with an unboxed implementation:
      | > KernelFunction func = KernelFunction::makeFromUnboxedLambda(
      | > [] (Tensor a, bool b) -> Tensor {...});
      | > Tensor result = func.callBoxed(stack);
      |
      */
    pub fn call_boxed(&self, 
        op_handle:        &OperatorHandle,
        dispatch_key_set: DispatchKeySet,
        stack:            *mut Stack)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Call the function in an unboxed way.
      | If the kernel function was created with a boxed function,
      | this will box all inputs and then call into that boxed function.
      |
      | Note that this doesn't work for all types yet.
      |
      | Example:
      |
      | > KernelFunction func = KernelFunction::makeFromUnboxedLambda(
      | >      [] (Tensor a, bool b) -> Tensor {...});
      | > Tensor result = func.call<Tensor, Tensor, bool>(tensor1, true);
      |
      | Or, with a boxed implementation:
      |
      | > void boxed_func(OperatorKernel*, Stack* stack) {...}
      | > KernelFunction func = KernelFunction::makeFromBoxedFunction(&boxed_func);
      | > Tensor result = func.call<Tensor, Tensor, bool>(tensor1, true);
      */
    pub fn call(&self, 
        op_handle:        &OperatorHandle,
        dispatch_key_set: DispatchKeySet,
        args:             Args) -> Return {
        
        todo!();
        /*
        
        */
    }

    /**
      | Create a KernelFunction from a boxed
      | function.
      | 
      | Example: > void boxed_func(OperatorKernel*,
      | Stack* stack) {...} > KernelFunction
      | func = KernelFunction::makeFromBoxedFunction<&boxed_func>();
      |
      */
    pub fn make_from_boxed_function() -> KernelFunction {
        
        todo!();
        /*
        
        */
    }

    /**
      | TODO: This will only be useful if we write
      | a backend fallback that plumbs dispatch
      | keys (currently there are none)
      | 
      | See Note [Plumbing Keys Through The
      | Dispatcher] for details.
      |
      */
    pub fn make_from_boxed_function() -> KernelFunction {
        
        todo!();
        /*
        
        */
    }

    /**
      | Create a KernelFunction from an unboxed functor.
      |
      | Example:
      |
      | > class MyFunctor final {
      | >   
      | >     Tensor operator()(Tensor a, Tensor b) {...}
      | > };
      | > KernelFunction func = KernelFunction::makeFromUnboxedFunctor(make_unique<MyFunctor>());
      */
    //template<bool AllowLegacyTypes = false, class KernelFunctor>
    pub fn make_from_unboxed_functor(kernel_functor: Box<OperatorKernel>) -> KernelFunction {
        
        todo!();
        /*
        
        */
    }

    /**
      | Create a KernelFunction from an unboxed function.
      |
      | This is usually better than
      | KernelFunction::makeFromUnboxedRuntimeFunction
      | because knowing the function pointer as
      | a template argument (i.e. at compile time)
      | allows the compiler to inline the function
      | into its unboxing wrapper and yields better
      | performance when calling the function.
      |
      | Example:
      |
      | > Tensor unboxed_func(Tensor a, Tensor b) {...}
      | > KernelFunction func = KernelFunction::makeFromUnboxedFunction<decltype(unboxed_func), &unboxed_func>();
      |
      | template<class FuncPtr, bool AllowLegacyTypes = false>
      */
    pub fn make_from_unboxed_function(_0: FuncPtr) -> KernelFunction {
        
        todo!();
        /*
        
        */
    }

    /**
      | Create a KernelFunction from an unboxed
      | function.
      |
      | KernelFunction::makeFromUnboxedFunction is
      | usually a better choice than this if you know
      | the function pointer at compile time, see doc
      | comment there for an explanation.
      |
      | Example:
      |
      | > Tensor unboxed_func(Tensor a, Tensor b) {...}
      | > KernelFunction func = KernelFunction::makeFromUnboxedRuntimeFunction(&unboxed_func);
      |
      | template<bool AllowLegacyTypes = false, class FuncType>
      */
    pub fn make_from_unboxed_runtime_function(func: *mut FuncType) -> KernelFunction {
        
        todo!();
        /*
        
        */
    }
    
    pub fn make_fallthrough() -> KernelFunction {
        
        todo!();
        /*
        
        */
    }
    
    pub fn make_ambiguous_autograd_other() -> KernelFunction {
        
        todo!();
        /*
        
        */
    }
    
    pub fn make_named_not_supported() -> KernelFunction {
        
        todo!();
        /*
        
        */
    }

    /**
      | Create a KernelFunction from an unboxed
      | lambda.
      | 
      | Example: > KernelFunction func = KernelFunction::makeFromUnboxedLambda(
      | > [] (Tensor a, bool b) -> Tensor {...});
      |
      */
    lazy_static!{
        /*
        template<bool AllowLegacyTypes = false, class Lambda>
          static enable_if_t<guts::is_stateless_lambda<decay_t<Lambda>>::value, KernelFunction> makeFromUnboxedLambda(Lambda&& lambda);
          template<bool AllowLegacyTypes = false, class Lambda>
          static enable_if_t<!guts::is_stateless_lambda<decay_t<Lambda>>::value, KernelFunction> makeFromUnboxedLambda(Lambda&& lambda);
        */
    }

    pub fn dump_state(&self) -> String {
        
        todo!();
        /*
        
        */
    }

    /// For testing internal invariants only
    pub fn equals_boxed_and_unboxed(&self, _0: &KernelFunction) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new(
        functor:             Box<OperatorKernel>,
        boxed_kernel_func:   *mut InternalBoxedKernelFunction,
        unboxed_kernel_func: *mut void) -> Self {
    
        todo!();
        /*


        
        */
    }

    // template<BoxedKernelFunction* func>
    pub fn make_boxed_function(
        _0:        *mut OperatorKernel,
        op_handle: &OperatorHandle,
        _2:        DispatchKeySet,
        stack:     *mut Stack)  {
        
        todo!();
        /*
        
        */
    }

    // template<BoxedKernelFunction_withDispatchKeys* func>
    pub fn make_boxed_function(
        _0:        *mut OperatorKernel,
        op_handle: &OperatorHandle,
        _2:        DispatchKeySet,
        stack:     *mut Stack)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_functor(&self) -> *mut OperatorKernel {
        
        todo!();
        /*
        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/boxing/KernelFunction.cpp]

/**
  | This kernel implements the behavior of falling
  | through to the next available registered
  | dispatch key.  The implementation of this
  | function is FAST; it is no overhead to
  | fallthrough to the next key.  See cpp file for
  | some more implementation notes; notably, this
  | does NOT actually go through the
  | boxing/unboxing codepath.
  |
  | This a "fake" kernel which doesn't actually
  | do anything.
  | 
  | Instead, it is a distinguished kernel
  | which is special cased by the dispatch
  | table to be handled specially.
  | 
  | Its semantics is that it redispatches
  | to the *next* dispatch key that would
  | have been processed, skipping the current
  | one.
  |
  */
pub fn fallthrough_kernel(
        _0: *mut OperatorKernel,
        _1: &OperatorHandle,
        _2: DispatchKeySet,
        _3: *mut Stack)  {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(0,
        "fallthrough_kernel was executed but it should have been short-circuited by the dispatcher. "
        "This could occur if you registered a fallthrough kernel as a override for a specific operator "
        "(as opposed to a backend fallback); this is NOT currently supported, and we do not intend to "
        "add support for it in the near future.  If you do find yourself in need of this, "
        "let us know in the bug tracker.");
        */
}

/**
  | Note [Ambiguity in AutogradOther kernel]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  |
  | This error-reporting kernel is registered to
  | the AutogradOther entry in the dispatch table
  | when there is both a CompositeImplicitAutograd
  | kernel and a backend kernel for ANY backend
  | that maps to AutogradOther.  To see why this is
  | necessary in the AutogradOther case, it's
  | helpful to first see why everything works out
  | fine for a backend that has a reserved Autograd
  | entry (see rule 2.2 in [Note] DispatchTable
  | computation):
  |
  |    CPU   AutogradCPU
  |    reg?  registers with...
  |    -------------------------------------------------
  |    y     Autograd registration takes precedence
  |          over CompositeImplicitAutograd.
  |
  |          This is good, because the CPU specific
  |          backend implementation is more
  |          specialized and typically better; if
  |          we used the composite, we would bypass
  |          it.
  |
  |          (NB: the Autograd key is guaranteed to
  |          exist because the autograd codegen
  |          requires it!)
  |
  |    n     CompositeImplicitAutograd takes
  |          precedence.
  |
  |          This is also good, because the
  |          Autograd registration (if it exists)
  |          would try to redispatch to the
  |          (non-existent) CPU implementation; by
  |          using the composite, we ensure the
  |          operator actually works.
  |
  | As you can see, when we have a specific
  | Autograd key (AutogradCPU), we can decide
  | whether or not to use the
  | CompositeImplicitAutograd kernel or the
  | Autograd kernel based on whether or not the
  | backend kernel exists.
  |
  | However, for AutogradOther (which is the
  | catchall autograd kernel for everything that
  | doesn't have a specific Autograd key), we can't
  | do this trick because there isn't any unique
  | backend to peek at to disambiguate; if there
  | are some backends that have implementations
  | they prefer Autograd, but unimplemented
  | backends would prefer
  | CompositeImplicitAutograd.  Rather than
  | arbitrarily pick one or the other, we just
  | register a kernel that raises an error and let
  | the user decide how to proceed.
  */
pub fn ambiguous_autogradother_kernel(
    _0: *mut OperatorKernel,
    op: &OperatorHandle,
    _2: DispatchKeySet,
    _3: *mut Stack)  {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(0,
        op.operator_name(), " has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. "
        "This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering "
        "(see Note [Ambiguity in AutogradOther kernel]). "
        "If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated "
        "Autograd dispatch key for the backend.\n",
        "If you only want to run inference instead of training, add `InferenceMode mode;` "
        "before model.forward(). Note this guard is only available in C++ but not Python at present.",
        "\nCanonical state\n~~~~~~~~~~~\n", op.dumpState(), "\n\n");
        */
}

/**
  | Note [named_not_supported_kernel]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  |
  | This kernel implements reporting an error
  | message saying that named tensor is not
  | supported.  This kernel doesn't rely on the
  | Stack, and so it is special cased in the
  | dispatcher to be triggered before we attempt
  | boxing (so we can give a good error message in
  | cases when boxing is not supported).  When
  | boxing is universally supported this can be
  | removed.
  */
pub fn named_not_supported_kernel(
        _0: *mut OperatorKernel,
        op: &OperatorHandle,
        _2: DispatchKeySet,
        _3: *mut Stack)  {
    
    todo!();
        /*
            // DO NOT LOOK AT STACK, YOU HAVE SHORT CIRCUITED BOXING
      // See Note [named_not_supported_kernel]
      TORCH_CHECK(0,
        op.operator_name(), " is not yet supported with named tensors. Please drop names via "
        "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
        "and set names on the result of the operation."
        );
        */
}

impl KernelFunction {
    
    // single line summary of state
    pub fn dump_state(&self) -> String {
        
        todo!();
        /*
            ostringstream oss;
      if (boxed_kernel_func_ == fallthrough_kernel) {
        oss << "fallthrough ";
      }
      if (boxed_kernel_func_) {
        oss << "boxed ";
      }
      if (unboxed_kernel_func_) {
        oss << "unboxed ";
      }
      return oss.str();
        */
    }
    
    pub fn equals_boxed_and_unboxed(&self, other: &KernelFunction) -> bool {
        
        todo!();
        /*
            return boxed_kernel_func_ == other.boxed_kernel_func_ &&
             unboxed_kernel_func_ == other.unboxed_kernel_func_;
        */
    }
}
