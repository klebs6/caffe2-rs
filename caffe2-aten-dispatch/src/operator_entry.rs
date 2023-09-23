crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/dispatch/OperatorEntry.h]

/**
  | This data structure represents a kernel that
  | was registered to us from a user.
  |
  | Unlike KernelFunction, AnnotatedKernel contains
  | some extra metadata about the kernel that isn't
  | necessary for actual dispatching (this is why
  | we don't put AnnotatedKernel in the actual
  | DispatchTable), but is useful for giving good
  | error messages.
  |
  */
pub struct AnnotatedKernel {

    kernel:                   KernelFunction,
    inferred_function_schema: Box<FunctionSchema>,

    /**
      | A little debug string to help us identify
      | the kernel in question.
      | 
      | Most importantly it records the TORCH_LIBRARY
      | block that did the registration.
      |
      */
    debug:                    String,
}

impl AnnotatedKernel {

    pub fn new(
        k: KernelFunction,
        s: Box<FunctionSchema>,
        d: String) -> Self {
    
        todo!();
        /*


            : kernel(move(k))
        , inferred_function_schema(move(s))
        , debug(move(d))
        */
    }
}

/**
  | This data structure represents operator schema,
  | with metadata specifying where the registration
  | of this schema occurred
  |
  */
pub struct AnnotatedSchema {
    schema: FunctionSchema,
    debug:  String,
}

impl AnnotatedSchema {
    
    pub fn new(
        s: FunctionSchema,
        d: String) -> Self {
    
        todo!();
        /*


            : schema(move(s))
        , debug(move(d))
        */
    }
}

/**
  | cpp_signature_ stores function signature if
  | any of the kernels was created in a way that
  | allowed us to know the function signature
  | (i.e. by supplying an unboxed C++ kernel
  | function).
  |
  | If this is set, it will be used to check that
  | future kernel registrations match and it will
  | be used in unboxed function calls to verify
  | their arguments against the known function
  | signature.
  |
  */
pub struct CppSignatureWithDebug {
    signature:    CppSignature,
    debug:        String,
    dispatch_key: Option<DispatchKey>,
}

/**
  | Internal data structure that records
  | information about a specific operator.
  |
  | It's not part of the public API; typically,
  | users will interact with OperatorHandle
  | instead.
  |
  | Concurrent writes to OperatorEntry are
  | protected by the GLOBAL Dispatcher lock (this
  | is important because some methods in
  | OperatorEntry access dispatcher state)
  |
  */
pub struct OperatorEntry {

    name:                   OperatorName,
    schema:                 Option<AnnotatedSchema>,
    dispatch_table:         Array<KernelFunction,DispatchKey_NumDispatchKeys>,
    dispatch_key_extractor: DispatchKeyExtractor,


    // kernels_ stores all registered kernels for the corresponding dispatch key
    // and catchAllKernels_ stores the catch-all kernels.
    //
    // If an operator library gets loaded that overwrites an already existing kernel,
    // both kernels will be in that list but only the newer one will be in
    // dispatchTable. If any of the kernels go away (say the library gets
    // unloaded), we remove the kernel from this list and update the
    // dispatchTable if necessary.
    //
    // Kernels in the list are ordered by registration time descendingly,
    // newer registrations are before older registrations.
    // We do not combine dispatchTable and kernels into one hash map because
    // kernels is a larger data structure and accessed quite infrequently
    // while dispatchTable is accessed often and should be kept small to fit
    // into CPU caches.
    //
    // Invariants:
    //  - dispatchTable[dispatch_key] == kernels_[dispatch_key].front()
    //  - dispatchTable[dispatch_key] does not exist if and only if
    //    kernels_[dispatch_key] does not exist
    //  - If kernels_[dispatch_key] exists, then it has elements.
    //    It is never an empty list.
    //
    // Why do we do that?
    // -----
    // We mostly do this to enable Jupyter notebooks where a cell registering
    // a kernel could be executed multiple times and the later execution
    // should overwrite the earlier one. Note that this still fails when the
    // function schema changed between the executions, but it works as long
    // as the function schema didn't change. 
    //
    // A better solution would be to
    // unload the old extension library from the Jupyter cell when the cell is
    // re-executed and then only allow one kernel here, i.e. error if a kernel
    // is already registered, but that's a lot of effort to implement and
    // currently not high-pri.
    //
    kernels:        FlatHashMap<DispatchKey,LinkedList<AnnotatedKernel>>,
    missing_kernel: AnnotatedKernel,
    cpp_signature:  Option<CppSignatureWithDebug>,

    /**
      | Whether this operator needs to be observed
      | with RecordFunction
      |
      */
    is_observed:    bool,
}

pub mod operator_entry {

    use super::*;

    lazy_static!{
        /*
        static const AnnotatedKernel ambiguousAutogradOtherKernel_;
        */
    }
}

impl OperatorEntry {
    
    pub fn new(operator_name: OperatorName) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn schema(&self) -> &FunctionSchema {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(schema_.has_value(), "Tried to access the schema for ", name_, " which doesn't have a schema registered yet");
        return schema_->schema;
        */
    }
    
    pub fn debug(&self) -> &String {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(schema_.has_value());
        return schema_->debug;
        */
    }
    
    pub fn has_schema(&self) -> bool {
        
        todo!();
        /*
            return schema_.has_value();
        */
    }
    
    pub fn is_observed(&self) -> bool {
        
        todo!();
        /*
            return is_observed_;
        */
    }

    /**
      | We may allocate an OperatorEntry for an
      | operator even when we don't have a schema.
      | When we receive the schema registration, we
      | post facto register a schema.
      |
      | NB: registerSchema/deregisterSchema are not
      | idempotent; if you attempt to register
      | a schema when one is already present or vice
      | versa that is an error.  (Refcounting for the
      | registrations is handled in the
      | OperatorHandle in Dispatcher)
      |
      */
    pub fn register_schema(&mut self, 
        _0:    FunctionSchema,
        debug: String)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn deregister_schema(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn operator_name(&self) -> &OperatorName {
        
        todo!();
        /*
            return name_;
        */
    }

    /*
      | Why are kernels and fallback asymmetric?  It
      | has to do with ownership.
      |
      | Kernels and the computed dispatch tables for
      | them are canonically owned by OperatorEntry,
      | but backend fallbacks are specified once and
      | apply for all operators, so they should be
      | owned by Dispatcher.
      |
      | However, the registration of a backend
      | fallback affects the state of the computed
      | dispatch table, so when a backend fallback is
      | updated, we need to update the operator
      | tables too.
      |
      | Thus, registerKernel is the mechanism by
      | which we give kernels to operator entry to
      | own (and update dispatch table), but we only
      | need a non-owning mechanism to update
      | fallback.
      */

    /**
      | Precondition: Dispatcher::mutex_
      | is held
      | 
      | Postcondition: caller is responsible
      | for disposing of the kernel
      |
      */
    pub fn register_kernel(&mut self, 
        dispatcher:               &Dispatcher,
        dispatch_key:             Option<DispatchKey>,
        kernel:                   KernelFunction,
        cpp_signature:            Option<CppSignature>,
        inferred_function_schema: Box<FunctionSchema>,
        debug:                    String) -> AnnotatedKernelIterator {
        
        todo!();
        /*
        
        */
    }

    /// Precondition: Dispatcher::mutex_ is held
    pub fn deregister_kernel(&mut self, 
        dispatcher:   &Dispatcher,
        dispatch_key: Option<DispatchKey>,
        kernel:       AnnotatedKernelIterator)  {
        
        todo!();
        /*
        
        */
    }

    /// Precondition: Dispatcher::mutex_ is held
    pub fn update_fallback(&mut self, 
        dispatcher:   &Dispatcher,
        dispatch_key: DispatchKey)  {
        
        todo!();
        /*
        
        */
    }

    /// Precondition: Dispatcher::mutex_ is held
    pub fn update_schema_alias_analysis(&mut self, a: AliasAnalysisKind)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(schema_.has_value());
        schema_->schema.setAliasAnalysis(a);
        */
    }
    
    pub fn dump_computed_table(&self) -> String {
        
        todo!();
        /*
        
        */
    }
    
    pub fn dump_state(&self) -> String {
        
        todo!();
        /*
        
        */
    }
    
    pub fn check_invariants(&self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn dispatch_key_extractor(&self) -> &DispatchKeyExtractor {
        
        todo!();
        /*
            return dispatchKeyExtractor_;
        */
    }
    
    /**
      | Asserts that the given FuncType is correct
      | for calling this operator in an unboxed
      | way.
      |
      */
    pub fn assert_signature_is_correct<FuncType>(&mut self)  {
    
        todo!();
        /*
            if (C10_UNLIKELY(cpp_signature_.has_value() && (CppSignature::make<FuncType>() != cpp_signature_->signature))) {
          reportSignatureError(CppSignature::make<FuncType>().name());
        }
        */
    }
    
    pub fn report_error(&self, dispatch_key: DispatchKey)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn lookup(&self, k: DispatchKey) -> &KernelFunction {
        
        todo!();
        /*
            const auto& kernel = dispatchTable_[static_cast<u8>(k)];
        // A valid kernel *always* has a boxed kernel and *may* have an
        // unboxed kernel. However, we typically do unboxed calls in 
        // APIs, where the kernel 1) will very likely be valid and 2)
        // should have an unboxed kernel. Checking the unboxed kernel
        // first will allow us to avoid touching the boxed kernel at all
        // in the common case.
        if (C10_UNLIKELY(!kernel.isValidUnboxed())) {
          if (!kernel.isValid()) {
            reportError(k);
          }
        }
        return kernel;
        */
    }
    
    pub fn list_all_dispatch_keys(&self) -> String {
        
        todo!();
        /*
        
        */
    }

    pub fn report_signature_error(&self, name: String)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn compute_dispatch_table_entry(&self, 
        dispatcher:   &Dispatcher,
        dispatch_key: DispatchKey) -> &KernelFunction {
        
        todo!();
        /*
        
        */
    }
    
    pub fn compute_dispatch_table_entry_with_debug(&self, 
        dispatcher:   &Dispatcher,
        dispatch_key: DispatchKey) -> Pair<&AnnotatedKernel,*const u8> {
        
        todo!();
        /*
        
        */
    }

    /**
      | This function re-establishes the invariant
      | that dispatchTable contains the front element
      | from the kernels list for a given runtime
      | dispatch key.
      */
    pub fn update_dispatch_table_entry(&mut self, 
        dispatcher:   &Dispatcher,
        dispatch_key: DispatchKey)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Like above, but also handles alias dispatch
      | keys.
      |
      */
    pub fn update_dispatch_table(&mut self, 
        dispatcher:   &Dispatcher,
        dispatch_key: DispatchKey)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Like above, but for ALL entries in the
      | dispatch table.
      |
      */
    pub fn update_dispatch_table_full(&mut self, dispatcher: &Dispatcher)  {
        
        todo!();
        /*
        
        */
    }

    /// Returns true if kernel_ has entry for any
    /// key in ks.
    ///
    pub fn has_kernel_for_any_dispatch_key(&self, ks: DispatchKeySet) -> bool {
        
        todo!();
        /*
        
        */
    }

    /**
      | Retrieves a pointer to AnnotatedKernel
      | at kernels_.at(dispatch_key).front().
      |
      */
    pub fn get_kernel_for_dispatch_key(&self, dispatch_key: DispatchKey) -> Option<*const AnnotatedKernel> {
        
        todo!();
        /*
        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/dispatch/OperatorEntry.cpp]

pub fn to_string(k: Option<DispatchKey>) -> String {
    
    todo!();
        /*
            if (k.has_value()) {
          return toString(*k);
        } else {
          return "(catch all)";
        }
        */
}

impl OperatorEntry {
    
    pub fn new(operator_name: OperatorName) -> Self {
    
        todo!();
        /*


            : name_(move(operator_name))
    , schema_()
    , dispatchTable_()
    , dispatchKeyExtractor_(DispatchKeyExtractor::makeUninitialized())
    , kernels_()
    , cpp_signature_()
    , is_observed_(ObservedOperators::isObserved(name_))
      // Pick up any backend fallbacks that were registered prior to this
      // OperatorEntry being created
      updateDispatchTableFull_(Dispatcher::singleton());
        */
    }
}

pub fn check_schema(
    name:           &OperatorName,
    from_def:       &FunctionSchema,
    from_def_debug: &String,
    inferred:       &FunctionSchema,
    inferred_debug: &String)  {
    
    todo!();
        /*
            optional<string> schema_difference = findSchemaDifferences(from_def, inferred);
        if (schema_difference.has_value()) {
          TORCH_CHECK(false,
            "Inferred operator schema for a C++ kernel function doesn't match the expected function schema.\n"
            "  operator: ", toString(name), "\n",
            "  expected schema: ", toString(from_def), "\n",
            "    ", from_def_debug, "\n",
            "  inferred schema: ", toString(inferred), "\n",
            "    ", inferred_debug, "\n",
            "  reason: ", *schema_difference);
        }
        */
}

lazy_static!{
    /*
    const AnnotatedKernel OperatorEntry::ambiguousAutogradOtherKernel_ = AnnotatedKernel( KernelFunction::makeAmbiguousAutogradOther(), nullptr, "ambiguous_autogradother");
    */
}

impl OperatorEntry {
    
    pub fn register_schema(&mut self, 
        schema: FunctionSchema,
        debug:  String)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(!schema_.has_value());
      for (auto i = kernels_.begin(); i != kernels_.end(); ++i) {
        for (auto j = i->second.begin(); j != i->second.end(); ++j) {
          if (j->inferred_function_schema != nullptr) {
            checkSchema(name_, schema, debug, *j->inferred_function_schema, j->debug);
          }
        }
      }
      // NB: don't register schema until after we've checked everything!
      dispatchKeyExtractor_.registerSchema(schema);
      schema_ = AnnotatedSchema(move(schema), move(debug));
        */
    }
    
    pub fn deregister_schema(&mut self)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(schema_.has_value());
      schema_ = nullopt;
      dispatchKeyExtractor_.deregisterSchema();
        */
    }
    
    pub fn register_kernel(&mut self, 
        dispatcher:               &Dispatcher,
        dispatch_key:             Option<DispatchKey>,
        kernel:                   KernelFunction,
        cpp_signature:            Option<CppSignature>,
        inferred_function_schema: Box<FunctionSchema>,
        debug:                    String) -> AnnotatedKernelIterator {
        
        todo!();
        /*
            // NB: cpp_signature doesn't get cleared even after the kernel that populated
      // it is deleted.  This means you could poison the value of cpp_signature_
      // with a bad signature value, and then it would permanently stay there until
      // you deregister the schema.  This can't really be fixed, because we
      // only do a typed() test once in the lifetime of a TypedOperatorHandle,
      // which means if you could validly change the type of a cpp_signature, then
      // that would also invalidate the old TypedOperatorHandles.
      if (cpp_signature.has_value()) {
        if (cpp_signature_.has_value()) {
          TORCH_CHECK(*cpp_signature == cpp_signature_->signature,
            "\nMismatch in kernel C++ signatures\n",
            "  operator: ", (this->schema_.has_value() ? toString(this->schema_->schema) : toString(name_)), "\n",
            "    ", (this->schema_.has_value() ? this->schema_->debug : "no debug info"), "\n",
            "  kernel 1: ", cpp_signature_->signature.name(), "\n",
            "    dispatch key: ", toString(cpp_signature_->dispatch_key), "\n",
            "    ", cpp_signature_->debug, "\n",
            "  kernel 2: ", cpp_signature->name(), "\n",
            "    dispatch key: ", toString(dispatch_key), "\n",
            "    ", debug, "\n"
          );
        } else {
          cpp_signature_ = CppSignatureWithDebug { *cpp_signature, debug, dispatch_key };
        }
      }

      if (schema_ && inferred_function_schema) {
        checkSchema(name_, schema_->schema, schema_->debug, *inferred_function_schema, debug);
      }

      // Add the kernel to the kernels list,
      // possibly creating the list if this is the first kernel.
      // Redirect catchAll registrations to CompositeImplicitAutograd.
      auto& k = dispatch_key.has_value() ? kernels_[*dispatch_key] : kernels_[DispatchKey::CompositeImplicitAutograd];

      if (k.size() > 0) {
        TORCH_WARN("Overriding a previously registered kernel for the same operator and the same dispatch key\n",
                   "  operator: ", (schema_.has_value() ? toString(schema_->schema) : toString(name_)), "\n",
                   "    ", (this->schema_.has_value() ? this->schema_->debug : "no debug info"), "\n",
                   "  dispatch key: ", toString(dispatch_key), "\n",
                   "  previous kernel: ", (cpp_signature_.has_value() ? cpp_signature_->debug : "no debug info"), "\n",
                   "       new kernel: ", debug
        );
      }

      k.emplace_front(move(kernel), move(inferred_function_schema), move(debug));
      list<AnnotatedKernel>::iterator inserted = k.begin();
      // update the dispatch table, i.e. re-establish the invariant
      // that the dispatch table points to the newest kernel
      if (dispatch_key.has_value()) {
        updateDispatchTable_(dispatcher, *dispatch_key);
      } else {
        updateDispatchTableFull_(dispatcher);
      }
      return inserted;
        */
    }
    
    pub fn deregister_kernel(&mut self, 
        dispatcher:   &Dispatcher,
        dispatch_key: Option<DispatchKey>,
        kernel:       AnnotatedKernelIterator)  {
        
        todo!();
        /*
            // Redirect catchAll deregistrations to CompositeImplicitAutograd.
      DispatchKey dk = dispatch_key.has_value() ? *dispatch_key : DispatchKey::CompositeImplicitAutograd;
      auto found = kernels_.find(dk);
      TORCH_INTERNAL_ASSERT(found != kernels_.end(), "Tried to deregister a kernel for dispatch key ", toString(dispatch_key), " but there are no kernels registered for this dispatch key. The operator is ", toString(name_));
      auto& k = found->second;
      k.erase(kernel);
      if (k.empty()) {
        // the invariant says we don't want empty lists but instead remove the list from the map
        kernels_.erase(found);
      }
      updateDispatchTable_(dispatcher, dk);
        */
    }
    
    pub fn update_fallback(&mut self, 
        dispatcher:   &Dispatcher,
        dispatch_key: DispatchKey)  {
        
        todo!();
        /*
            updateDispatchTable_(dispatcher, dispatch_key);
        */
    }
    
    pub fn compute_dispatch_table_entry(&self, 
        dispatcher:   &Dispatcher,
        dispatch_key: DispatchKey) -> &KernelFunction {
        
        todo!();
        /*
            return computeDispatchTableEntryWithDebug(dispatcher, dispatch_key).first.kernel;
        */
    }
    
    pub fn has_kernel_for_any_dispatch_key(&self, ks: DispatchKeySet) -> bool {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(kernels_.find(DispatchKey::Undefined) == kernels_.end());
      for (auto& kv : kernels_) {
        if (ks.has(kv.first)) return true;
      }
      return false;
        */
    }
    
    pub fn get_kernel_for_dispatch_key(&self, dispatch_key: DispatchKey) -> Option<*const AnnotatedKernel> {
        
        todo!();
        /*
            auto kern_it = kernels_.find(dispatch_key);
      if (kern_it != kernels_.end()) {
        TORCH_INTERNAL_ASSERT(!kernels_.at(dispatch_key).empty());
        TORCH_INTERNAL_ASSERT(kernels_.at(dispatch_key).front().kernel.isValid());
        return make_optional(&kernels_.at(dispatch_key).front());
      }
      return nullopt;
        */
    }
    
    pub fn compute_dispatch_table_entry_with_debug(&self, 
        dispatcher:   &Dispatcher,
        dispatch_key: DispatchKey) -> Pair<&AnnotatedKernel,*const u8> {
        
        todo!();
        /*
            // [Note] DispatchTable computation
      // dispatchTable contains entries for runtime dispatch keys.
      // For any dispatch key, it'll pick a kernel using the following order:
      //  (1) Use kernel if it's directly registered to this key
      //  (2) Handle runtime keys that have kernels available from alias keys
      //    (2.1) Use kernel from DispatchKey::CompositeExplicitAutograd if available.
      //          This is used to register a kernel that works for all backend in inference. But it requires
      //          separate registration for Autograd keys to support training.
      //    (2.2) Use kernel from DispatchKey::CompositeImplicitAutograd if available.
      //          For autograd keys, we only use kernel from CompositeImplicitAutograd when there's no direct registration
      //          to its corresponding backend key or CompositeExplicitAutograd. See Note [CompositeExplicitAutograd and CompositeImplicitAutograd].
      //          For AutogradOther, we eagerly return ambiguousAutogradOtherKernel_ if there's registration to any of
      //          its backends and ask backend extender to request a decicated Autograd key for the backend.
      //          See Note [Ambiguity in AutogradOther kernel] for more details.
      //          A CompositeExplicitAutograd kernel prevents CompositeImplicitAutograd kernel being used for Autograd keys, but it doesn't
      //          cause confusion for AutogradOther. It's pretty straightforward to use Autograd (if available)
      //          in this case.
      //    (2.3) Use kernel from DispatchKey::Autograd if available
      //    The implementation of (2.2) relies on the invariant that for a given backend,
      //    `computeDispatchTableEntryWithDebug()` will be called for that backend's autograd key after the
      //    backend key. See Note [Refresh Runtime Autograd entries in dispatchTable_]
      //  (3) Use fallthrough kernel that are registered as fallback.
      // Alias Key Precedence:
      //   CompositeExplicitAutograd > CompositeImplicitAutograd > Autograd
      // Note [CompositeExplicitAutograd and CompositeImplicitAutograd]
      //   When there're registrations to both CompositeExplicitAutograd & CompositeImplicitAutograd & Autograd, from (2.2) we know CompositeExplicitAutograd
      //   and Autograd kernels will be picked up and CompositeImplicitAutograd is overriden.
      //   This is fine and in practice CompositeExplicitAutograd and CompositeImplicitAutograd shouldn't co-exist for an op.
      // TODO: Update alias key precedence after we add new alias keys AutogradDispatchCPUOrCUDA .

      // 1. Operator registration
      if (auto direct_registration = getKernelForDispatchKey(dispatch_key)) {
        return {*direct_registration.value(), "kernel"};
      }

      // 2.1 Use CompositeExplicitAutograd kernel if available.
      //     See Note [Undefined in dispatchTable_] for the special handling for Undefined.
      if (dispatch_key == DispatchKey::Undefined || isIncludedInAlias(dispatch_key, DispatchKey::CompositeExplicitAutograd)) {
        if (auto default_backend_registration = getKernelForDispatchKey(DispatchKey::CompositeExplicitAutograd)) {
          return {*default_backend_registration.value(), "default backend kernel"};
        }
      }

      // Note when there's direct registration to CompositeExplicitAutograd, this code path will only be hit by
      // non backend keys (e.g AutogradXXX, Batched etc) due to (2.1).
      bool has_backend_kernel =
        hasKernelForAnyDispatchKey(getBackendKeySetFromAutograd(dispatch_key).add(DispatchKey::CompositeExplicitAutograd));

      // 2.2. Use CompositeImplicitAutograd kernel if available. For autograd keys, we only use kernel from CompositeImplicitAutograd
      //      when there's no direct registration to its corresponding backend key or CompositeExplicitAutograd.
      //      For AutogradOther, we return ambiguousAutogradOtherKernel_ if there's registration
      //      to any of its backends.
      //      See Note [Undefined in dispatchTable_] for the special handling for Undefined.
      if (dispatch_key == DispatchKey::Undefined || isIncludedInAlias(dispatch_key, DispatchKey::CompositeImplicitAutograd)) {
        if (auto math_registration = getKernelForDispatchKey(DispatchKey::CompositeImplicitAutograd)) {
          if (dispatch_key == DispatchKey::AutogradOther
              && hasKernelForAnyDispatchKey(autogradother_backends)) {
            return {ambiguousAutogradOtherKernel_, "ambiguous autogradother"};
          } else if (!has_backend_kernel) {
            return {*math_registration.value(), "math kernel"};
          }
        }
      }

      // 2.3. For autograd backend keys, use kernel from DispatchKey::Autograd if available
      if (isIncludedInAlias(dispatch_key, DispatchKey::Autograd)) {
        if (auto autograd_registration = getKernelForDispatchKey(DispatchKey::Autograd)) {
          return {*autograd_registration.value(), "autograd kernel"};
        }
      }

      // 3. Backend fallback
      auto dispatch_ix = static_cast<u8>(dispatch_key);
      if (dispatcher.backendFallbackKernels_[dispatch_ix].kernel.isValid()) {
        return {dispatcher.backendFallbackKernels_[dispatch_ix], "backend fallback"};
      }

      // 4. Default to error
      return {missingKernel_, "missing"};
        */
    }

    /**
      | synchronizes the dispatch table entry for
      | a given dispatch key with the current state of
      | kernel registrations in the dispatcher.
      |
      | note that this is not a complete update, due to
      | relationships between dispatch keys
      | (e.g. runtime keys and their associated
      | autograd keys, or alias keys and their
      | associated keysets).
      |
      | This function should be considered a private
      | helper for updateDispatchTable_()
      */
    pub fn update_dispatch_table_entry(&mut self, 
        dispatcher:   &Dispatcher,
        dispatch_key: DispatchKey)  {
        
        todo!();
        /*
            auto dispatch_ix = static_cast<u8>(dispatch_key);
      dispatchTable_[dispatch_ix] = computeDispatchTableEntry(dispatcher, dispatch_key);
      dispatchKeyExtractor_.setOperatorHasFallthroughForKey(dispatch_key, dispatchTable_[dispatch_ix].isFallthrough());
        */
    }

    /**
      | synchronizes the dispatch table entries for
      | a given dispatch key *and its associated keys*
      | with the current state of kernel registrations
      | in the dispatcher.
      |
      | After a kernel has been registered to
      | a dispatch key, a call to this function will
      | synchronize the dispatcher state. See
      | e.g. registerKernel()
      */
    pub fn update_dispatch_table(&mut self, 
        dispatcher:   &Dispatcher,
        dispatch_key: DispatchKey)  {
        
        todo!();
        /*
            // Handle Undefined separately since it isn't a runtime key but we have an entry in dispatchTable_.
      // See Note [Undefined in dispatchTable_]
      if (dispatch_key == DispatchKey::Undefined) {
        updateDispatchTableEntry_(dispatcher, dispatch_key);
        return;
      }
      for (auto k : getRuntimeDispatchKeySet(dispatch_key)) {
        updateDispatchTableEntry_(dispatcher, k);
      }
      // Registration to CompositeExplicitAutograd and CompositeImplicitAutograd should be populated to Undefined.
      // We cannot do this above since Undefined cannot be represented in DispatchKeySet.
      if (dispatch_key == DispatchKey::CompositeImplicitAutograd || dispatch_key == DispatchKey::CompositeExplicitAutograd) {
        updateDispatchTableEntry_(dispatcher, DispatchKey::Undefined);
      }
      // Note [Refresh Runtime Autograd entries in dispatchTable_]
      // Registering to backend key might affect computed entry at its Autograd backend key due to (2.1) & (2.3).
      if (isBackendDispatchKey(dispatch_key)) {
        DispatchKey autograd_key = getAutogradKeyFromBackend(dispatch_key);
        updateDispatchTableEntry_(dispatcher, autograd_key);
      }
        */
    }

    /**
      | does a complete update of the dispatch table,
      | synchronizing all runtime dispatch keys with
      | the current state of kernel registrations in
      | the dispatcher.
      |
      | Note that we use updateDispatchTable_() to
      | perform our per-key updating, even though that
      | function is equipped to handle out-of-order
      | updates and alias key updates, neither of which
      | we send it.
      |
      | This is deliberate - the current design is more
      | tractable with all updates funneled through
      | a single per-key update mechanism, than with
      | multiple variations that assume different
      | invariants.
      |
      */
    pub fn update_dispatch_table_full(&mut self, dispatcher: &Dispatcher)  {
        
        todo!();
        /*
            // Note [Undefined in dispatchTable_]
      // DispatchKey Undefined is used in runtime:
      // (1) it gives people place to specify functionality that should run when there are no dispatch keys,
      //     e.g., an op without Tensor inputs or empty &[Tensor] arguments
      // (2) it would let us remove the explicit error checking code in the dispatch hotpath, and so when
      //     no dispatch keys are available we just slide into the undefined handler which would then raise
      //     the error message.
      // In the old world of catchAll, the only way to "register" a kernel to Undefined is by registering it to
      // catchAll. After catchAllKernel_ is removed, Undefined now can get a kernel from either CompositeExplicitAutograd
      // or CompositeImplicitAutograd alias key so that we don't break the support. Ideally isIncludedInAlias(Undefined, CompositeImplicitAutograd)
      // should return true, it returns false because Undefined cannot be represented in a DispatchKeySet.
      for (u8 iter = 0; iter != static_cast<u8>(DispatchKey::NumDispatchKeys); ++iter) {
        updateDispatchTable_(dispatcher, static_cast<DispatchKey>(iter));
      }
        */
    }
    
    pub fn check_invariants(&self)  {
        
        todo!();
        /*
            if (schema_) {
        TORCH_INTERNAL_ASSERT(schema_->schema.operator_name() == name_, dumpState());
        dispatchKeyExtractor().checkInvariants(schema_->schema);
      }
      TORCH_INTERNAL_ASSERT(kernels_.find(DispatchKey::Undefined) == kernels_.end(), dumpState());
      for (const auto& kv : kernels_) {
        TORCH_INTERNAL_ASSERT(kv.second.size() > 0, dumpState());
      }
      for (u8 iter = 0; iter != static_cast<u8>(DispatchKey::NumDispatchKeys); ++iter) {
        auto expected_k = computeDispatchTableEntry(Dispatcher::singleton(), static_cast<DispatchKey>(iter));
        TORCH_INTERNAL_ASSERT(expected_k._equalsBoxedAndUnboxed(dispatchTable_[iter]),
          "Canonical state\n~~~~~~~~~~~\n", dumpState(), "\n\n"
          "Computed table:\n~~~~~~~~~~~\n", dumpComputedTable());
      }
        */
    }
    
    pub fn list_all_dispatch_keys(&self) -> String {
        
        todo!();
        /*
            ostringstream str;
      str << "[";

      bool has_kernels = false;
      for (u8 iter = 0; iter != static_cast<u8>(DispatchKey::NumDispatchKeys); ++iter) {
        if (!dispatchTable_[iter].isValid()) {
          continue;
        }
        if (has_kernels) {
          str << ", ";
        }
        str << static_cast<DispatchKey>(iter);
        has_kernels = true;
      }
      str << "]";
      return str.str();
        */
    }
    
    pub fn report_signature_error(&self, name: String)  {
        
        todo!();
        /*
            TORCH_CHECK(false,
            "\nTried to access or call an operator with a wrong signature.\n",
            "  operator: ", (schema_.has_value() ? toString(schema_->schema) : toString(name_)), "\n",
            "    ", (schema_.has_value() ? schema_->debug : "unknown debug info"), "\n",
            "  correct signature:  ", cpp_signature_->signature.name(), "\n",
            "    ", cpp_signature_->debug, "\n",
            "  accessed/called as: ", name, "\n",
            "This likely happened in a call to OperatorHandle::typed<Return (Args...)>(). ",
            "Please make sure that the function signature matches the signature in the operator registration call."
      );
    }{
        */
    }
    
    pub fn report_error(&self, dispatch_key: DispatchKey)  {
        
        todo!();
        /*
            // If there is an invariant problem, report it now.
      checkInvariants();

      if (dispatchKey == DispatchKey::Undefined) {
        TORCH_CHECK_NOT_IMPLEMENTED(false,
              "There were no tensor arguments to this function (e.g., you passed an "
              "empty list of Tensors), but no fallback function is registered for schema ", name_,
              ".  This usually means that this function requires a non-empty list of Tensors, "
              "or that you (the operator writer) forgot to register a fallback function.  "
              "Available functions are ", listAllDispatchKeys(), ".\n\n", dumpComputedTable())
      }

      TORCH_CHECK_NOT_IMPLEMENTED(false, "Could not run '", name_, "' with arguments",
              " from the '", toString(dispatchKey), "' backend. This could be because "
              "the operator doesn't exist for this backend, or was omitted during ",
              "the selective/custom build process (if using custom build). If you are a ",
              "Facebook employee using PyTorch on mobile, please visit ",
              "https://fburl.com/ptmfixes for possible resolutions. '",
              name_, "' is only available for these backends: ",
              listAllDispatchKeys(), ".\n\n", dumpComputedTable());
        */
    }

    /*
      | INSPECTING DISPATCHER STATE
      | ~~~~~~~~~~~~~~~~~~~~~~~~~~~
      | The dumper functions purposely do not check
      | invariants, as you might be using them to debug
      | situations where the invariants are violated.
      */

    /**
      | Inspect what the computed dispatch table would
      | be (e.g., what updateDispatchTableFull_ would
      | update the dispatch table to be)
      |
      */
    pub fn dump_computed_table(&self) -> String {
        
        todo!();
        /*
            ostringstream oss;
      for (u8 i = 0; i < static_cast<u8>(DispatchKey::NumDispatchKeys); i++) {
        auto k = static_cast<DispatchKey>(i);
        auto kernel_prov = computeDispatchTableEntryWithDebug(Dispatcher::singleton(), k);
        if (kernel_prov.first.kernel.isValid()) {
          oss << toString(k) << ": "
              << (kernel_prov.first.kernel.isFallthrough() ? "fallthrough " : "")
              << kernel_prov.first.debug << " [" << kernel_prov.second << "]\n";
        }
      }
      return oss.str();
        */
    }

    /**
      | Inspect the "canonical" information in
      | OperatorEntry.
      |
      | This only prints out *non-derived* information
      | including kernels registered to alias dispatch
      | keys; i.e., what the source of truth says about
      | the operator.
      |
      | This dumping function is appropriate for expect
      | tests.
      |
      | This WON'T report backend fallbacks.
      */
    pub fn dump_state(&self) -> String {
        
        todo!();
        /*
            ostringstream oss;
      oss << "name: " << name_ << "\n";
      if (schema_) {
        oss << "schema: " << schema_->schema << "\n";
        oss << "debug: " << schema_->debug << "\n";
        oss << "alias analysis kind: " << toString(schema_->schema.aliasAnalysis())
            << (schema_->schema.isDefaultAliasAnalysisKind() ? " (default)" : "") << "\n";
      } else {
        oss << "schema: (none)\n";
      }

      auto print_kernel = [&](const char* k_desc, const list<AnnotatedKernel>& jts, bool is_alias_key=false) {
        i64 i = 0;
        for (const auto& jt : jts) {
          oss << k_desc
              << (is_alias_key ? "[alias]" :  "")
              << (i > 0 ? " (inactive)" : "")
              << ": "
              << jt.debug << " :: "
              << (jt.inferred_function_schema ? toString(*jt.inferred_function_schema) : "(none)")
              << " [ " << jt.kernel.dumpState() << "]\n";
          i++;
        }
      };

      // Iterate over DispatchKey, not the flat hash map, so we have a stable order
      for (u8 i = 0; i <= static_cast<u8>(DispatchKey::EndOfAliasKeys); i++) {
        auto k = static_cast<DispatchKey>(i);
        auto it = kernels_.find(k);
        if (it != kernels_.end()) {
          print_kernel(toString(k), it->second, isAliasDispatchKey(k));
        }
      }
      return oss.str();
        */
    }
}
