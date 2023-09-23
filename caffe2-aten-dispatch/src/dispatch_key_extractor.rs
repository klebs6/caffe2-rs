crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/dispatch/DispatchKeyExtractor.h]

/**
  | Take a DispatchKeySet for a Tensor and
  | determine what the actual dispatch DispatchKey
  | should be, taking into account TLS, and
  | skipping backends which fall through.
  |
  | Unlike Tensor::key_set(), the value of this on
  | a tensor can change depending on TLS.
  |
  | NB: If there is no valid dispatch key, this
  | will return Undefined
  */
#[inline] pub fn compute_dispatch_key_set(

    ks:       DispatchKeySet,

    // The key mask lets us eliminate (by zero
    // entries) keys which should not be
    // considered for dispatch.  There are two
    // cases when we use this:
    //
    // - If an operator's dispatch table contains
    //   a fallthrough entry, we should bypass it
    //   entirely when finding the key
    //
    // - If a user invokes with redispatch, the
    //   mask lets us zero out the key the user
    //   asked us to stop.
    //
    // These excluded backends are NOT tracked in
    // the TLS, but must be applied AFTER TLS
    // (since the backend may have been
    // introduced for consideration by the
    // included TLS), which is why you have to
    // pass them in to this function (as opposed
    // to just applying it to the input 'ks').
    key_mask: DispatchKeySet) 
    -> DispatchKeySet 
{
    todo!();

    /*
        LocalDispatchKeySet local = tls_local_dispatch_key_set();
      // TODO: It's a bit irritating that we have to do logical ORs here, it would
      // be nice to only do one.  Can always_included be folded into the TLS?  Well,
      // it's a bit troublesome, because fastpath TLS access requires the type of
      // the TLS in question to be zero-initialized, so you don't actually win
      // anyting in that case.
      return (((ks | local.included_) - local.excluded_) & key_mask);
        */
}

// A small gadget to extract the DispatchKeySet
// from types which are known to have it.  Used to
// extract dispatch keys from unboxed calls.
//
pub struct MultiDispatchKeySet {
    base: IterArgs<MultiDispatchKeySet>,
    ts:   DispatchKeySet,
}

impl MultiDispatchKeySet {
    
    pub fn invoke(&mut self, x: &Tensor)  {
        
        todo!();
        /*
            ts = ts | x.key_set();
        */
    }
    
    pub fn invoke(&mut self, x: Option<Tensor>)  {
        
        todo!();
        /*
            if (x.has_value()) {
                ts = ts | x->key_set();
            }
        */
    }
    
    pub fn invoke(&mut self, xs: &[Tensor])  {
        
        todo!();
        /*
            for (const auto& x : xs) {
                ts = ts | x.key_set();
            }
        */
    }
    
    pub fn invoke(&mut self, gen: dyn GeneratorInterface)  {
        
        todo!();
        /*
            if (gen.defined()) {
                ts = ts | gen.key_set();
            }
        */
    }
    
    pub fn invoke(&mut self, gen: Option<dyn GeneratorInterface>)  {
        
        todo!();
        /*
            if (gen.has_value() && gen->defined()) {
                ts = ts | gen->key_set();
            }
        */
    }
    
    pub fn invoke<T>(&mut self, x: &T)  {
    
        todo!();
        /*
            // do nothing
        */
    }
}

/**
  | NB: take by const reference (Don't do
  | universal forwarding here! You don't
  | want to move into this function!)
  |
  */
pub fn multi_dispatch_key_set<Args>(args: &Args) -> DispatchKeySet {

    todo!();
        /*
            return MultiDispatchKeySet().apply(args...).ts;
        */
}

/**
 | An instance of DispatchKeyExtractor knows how
 | to get a dispatch key given a list of arguments
 | for an operator call.
 |
 | The instance is specific for a certain operator
 | as:
 |
 |  - In boxed dispatch, different operators have
 |    different ways to extract the dispatch key
 |    (e.g. different numbers of arguments), and
 |    we precompute the stack locations we should
 |    look at; and
 |
 |  - In all dispatch, some backends should be
 |    excluded from dispatch because they have
 |    been registered as fallthrough.  The set of
 |    excluded backends varies from operator, as
 |    some operators may have overridden the
 |    fallthrough with custom behavior.
 */
pub struct DispatchKeyExtractor {

    /**
      | this is a bitset that has ones for each
      | argument index which has to be considered
      | for dispatch. 
      |
      | This avoids having to iterate over the
      | stack to find all the tensors. The bits
      | are stored in reverse order,
      | i.e. dispatch_arg_indices_reverse_[i] ==
      | true, then the i-th argument from the top
      | of the stack (i.e. the i-th last argument
      | of the function) is relevant for dispatch. 
      |
      | dispatch_arg_indices_reverse_ is allowed
      | to have zero bits set; that just means you
      | must do the fallthrough
      |
      */
    dispatch_arg_indices_reverse: BitSet,

    /**
      | Set of keys for which the operator does
      | NOT have fallthrough kernel.
      |
      */
    non_fallthrough_keys: DispatchKeySet,
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/dispatch/DispatchKeyExtractor.cpp]
impl DispatchKeyExtractor {
    
    pub fn make(schema: &FunctionSchema) -> DispatchKeyExtractor {
        
        todo!();
        /*
            return DispatchKeyExtractor(makeBitsetForDispatchArgs(schema));
        */
    }
    
    pub fn make_uninitialized() -> DispatchKeyExtractor {
        
        todo!();
        /*
            return DispatchKeyExtractor(utils::bitset());
        */
    }
    
    pub fn register_schema(&mut self, schema: &FunctionSchema)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(dispatch_arg_indices_reverse_.is_entirely_unset());
        dispatch_arg_indices_reverse_ = makeBitsetForDispatchArgs(schema);
        */
    }
    
    pub fn deregister_schema(&mut self)  {
        
        todo!();
        /*
            dispatch_arg_indices_reverse_ = utils::bitset();
        */
    }
    
    pub fn get_dispatch_key_set_boxed(&self, stack: *const TorchJitStack) -> DispatchKeySet {
        
        todo!();
        /*
            DispatchKeySet ks;
        dispatch_arg_indices_reverse_.for_each_set_bit([&] (usize reverse_arg_index) {
          const auto& ivalue = TorchJitpeek(*stack, 0, reverse_arg_index + 1);
          if (C10_LIKELY(ivalue.isTensor())) {
            // NB: Take care not to introduce a refcount bump (there's
            // no safe toTensorRef method, alas)
            ks = ks | ivalue.unsafeToTensorImpl()->key_set();
          } else if (C10_UNLIKELY(ivalue.isTensorList())) {
            for (const Tensor tensor : ivalue.toTensorList()) {
              ks = ks | tensor.key_set();
            }
          }
        });
        // Keys that are fallthrough should be skipped
        return computeDispatchKeySet(ks, nonFallthroughKeys_);
        */
    }
    
    
    pub fn get_dispatch_key_set_unboxed<Args>(&self, args: &Args) -> DispatchKeySet {
    
        todo!();
        /*
            auto ks = multi_dispatch_key_set(args...);
        // Keys that are fallthrough should be skipped
        return computeDispatchKeySet(ks, nonFallthroughKeys_);
        */
    }
    
    pub fn set_operator_has_fallthrough_for_key(&mut self, 
        k:               DispatchKey,
        has_fallthrough: bool)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn dump_state(&self) -> String {
        
        todo!();
        /*
        
        */
    }
    
    pub fn check_invariants(&self, schema: &FunctionSchema)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn make_bitset_for_dispatch_args(schema: &FunctionSchema) -> BitSet {
        
        todo!();
        /*
            TORCH_CHECK(schema.arguments().size() <= utils::bitset::NUM_BITS(),
            "The function schema has ", schema.arguments().size(),
            " arguments but this PyTorch build only supports ", utils::bitset::NUM_BITS());
        utils::bitset dispatch_arg_indices_reverse;
        for (usize index = 0; index < schema.arguments().size(); ++index) {
          if (schema.arguments()[index].type()->isSubtypeOf(TensorType::get()) ||
              schema.arguments()[index].type()->isSubtypeOf(
                  ListType::ofTensors()) ||
              schema.arguments()[index].type()->isSubtypeOf(
                  OptionalType::ofTensor())) {
            dispatch_arg_indices_reverse.set(schema.arguments().size() - 1 - index);
          }
        }
        return dispatch_arg_indices_reverse;
        */
    }
    
    pub fn new(dispatch_arg_indices_reverse: BitSet) -> Self {
    
        todo!();
        /*
        : dispatch_arg_indices_reverse(dispatch_arg_indices_reverse),
        : non_fallthrough_keys(DispatchKeySet::FULL),

        
        */
    }

    pub fn set_operator_has_fallthrough_for_key(&mut self, 
        k:               DispatchKey,
        has_fallthrough: bool)  {
        
        todo!();
        /*
            if (has_fallthrough) {
        nonFallthroughKeys_ = nonFallthroughKeys_.remove(k);
      } else {
        nonFallthroughKeys_ = nonFallthroughKeys_.add(k);
      }
        */
    }
    
    pub fn dump_state(&self) -> String {
        
        todo!();
        /*
            ostringstream oss;
      for (usize i=0; i < utils::bitset::NUM_BITS(); ++i) {
        if (dispatch_arg_indices_reverse_.get(i)) {
          oss << "1";
        } else {
          oss << "0";
        }
      }
      oss << " " << nonFallthroughKeys_ << "\n";
      return oss.str();
        */
    }

    pub fn check_invariants(&self, schema: &FunctionSchema)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(makeBitsetForDispatchArgs(schema) == dispatch_arg_indices_reverse_);
        */
    }
}
