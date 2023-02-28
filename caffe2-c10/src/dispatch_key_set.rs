crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/DispatchKeySet.h]

pub enum DispatchKeySetFull      { FULL }
pub enum DispatchKeySetFullAfter { FULL_AFTER }
pub enum DispatchKeySetRaw       { RAW }

/**
  | A representation of a set of DispatchKeys.
  | A tensor may have multiple tensor type ids,
  | e.g., a Variable tensor can also be a CPU
  | tensor;
  |
  | the DispatchKeySet specifies what type ids
  | apply.  The internal representation is as
  | a 64-bit bit set (this means only 64 tensor
  | type ids are supported).
  |
  | Note that DispatchKeys are ordered; thus, we
  | can ask questions like "what is the highest
  | priority DispatchKey in the set"?  (The set
  | itself is not ordered; two sets with the same
  | ids will always have the ids ordered in the
  | same way.)
  |
  | At the moment, there are no nontrivial uses of
  | this set; tensors are always singletons.  In
  | the near future, this set will represent
  | variable? + tensor type id.  In the far future,
  | it will be requires grad? + profiling?
  | + tracing? + lazy? + tensor type id.
  |
  | (The difference between variable and requires
  | grad, is that there are currently three states
  | a tensor can be:
  |
  |  1. Not a variable
  |  2. Variable with requires_grad=False
  |  3. Variable with requires_grad=True
  |
  | Eventually, we want to kill state (1), and only
  | dispatch to autograd handling code if one of
  | the inputs requires grad.)
  |
  | An undefined tensor is one with an empty tensor
  | type set.
  |
  */
#[derive(Default,Copy,Clone,PartialEq)]
pub struct DispatchKeySet {
    repr: u64, // default = 0
}

impl DispatchKeySet {
    
    pub fn new_from_full(_0: DispatchKeySetFull) -> Self {
    
        todo!();
        /*


            : repr_(numeric_limits<decltype(repr_)>::max())
        */
    }
    
    pub fn new_from_full_after(
        _0: DispatchKeySetFullAfter,
        t:  DispatchKey) -> Self {
    
        todo!();
        /*


            // LSB after t are OK, but not t itself.
          : repr_((1ULL << (static_cast<uint8_t>(t) - 1)) - 1)
        */
    }

    /**
      | Public version of DispatchKeySet(uint64_t)
      | API; external users must be explicit when
      | they do this!
      |
      */
    pub fn new_from_raw(_0: DispatchKeySetRaw, x: u64) -> Self {
    
        Self {
            repr: x
        }
    }
    
    pub fn new_from_key(t: DispatchKey) -> Self {
    
        let mut repr = match t {
            DispatchKey::Undefined => 0,
            _                      => 1 << (t as u8 - 1),
        };

        Self { repr }
    }
    
    pub fn new_from_keys(ks: &[DispatchKey]) -> Self {

        let mut repr = 0;

        for k in ks.iter() {
            repr |= DispatchKeySet::new_from_key(*k).repr;
        }

        Self { repr }
    }

    /// Test if a DispatchKey is in the set
    ///
    #[inline] pub fn has(&self, t: DispatchKey) -> bool {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(t != DispatchKey::Undefined);
        return static_cast<bool>(repr_ & DispatchKeySet(t).repr_);
        */
    }

    /// Test if DispatchKeySet is a superset of
    /// ks.
    ///
    pub fn is_superset_of(&self, ks: DispatchKeySet) -> bool {
        
        todo!();
        /*
            return (repr_ & ks.repr_) == ks.repr_;
        */
    }

    // Define a new method `iter_bits()` that
    // returns an iterator over the bit positions
    // in the set.
    //
    fn iter_bits(&self) -> BitIterator {
        BitIterator {
            set: self.repr,
            idx: 0,
        }
    }
}

impl BitOr<DispatchKeySet> for DispatchKeySet {

    type Output = DispatchKeySet;

    /// Perform set union
    #[inline] fn bitor(self, other: DispatchKeySet) -> Self::Output {
        todo!();
        /*
            return DispatchKeySet(repr_ | other.repr_);
        */
    }
}

impl BitAnd<DispatchKeySet> for DispatchKeySet {

    type Output = DispatchKeySet;
    
    /// Perform set intersection
    #[inline] fn bitand(self, other: DispatchKeySet) -> Self::Output {
        todo!();
        /*
            return DispatchKeySet(repr_ & other.repr_);
        */
    }
}

impl BitXor<DispatchKeySet> for DispatchKeySet {

    type Output = DispatchKeySet;
    
    /// Compute self ^ other
    #[inline] fn bitxor(self, other: DispatchKeySet) -> Self::Output {
        todo!();
        /*
            return DispatchKeySet(repr_ ^ other.repr_);
        */
    }
}

impl Sub<&DispatchKeySet> for DispatchKeySet {

    type Output = DispatchKeySet;
    
    /// Compute the set difference self - other
    ///
    #[inline]fn sub(self, other: &DispatchKeySet) -> Self::Output {
        todo!();
        /*
            return DispatchKeySet(repr_ & ~other.repr_);
        */
    }
}

// Define a struct `BitIterator` that implements `Iterator` and generates bit positions.
//
pub struct BitIterator {
    set: u64,
    idx: u32,
}

impl Iterator for BitIterator {

    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {

        if self.set == 0 {
            return None;
        }

        // Find the next non-zero bit position in the set.
        let bit = self.set.trailing_zeros() as usize;

        // Clear the bit from the set.
        self.set &= !(1 << bit);

        Some(bit)
    }
}

// Implement `IntoIterator` for `DispatchKeySet` by returning a `BitIterator`.
//
impl IntoIterator for DispatchKeySet {

    type Item = usize;
    type IntoIter = BitIterator;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_bits()
    }
}

impl DispatchKeySet {
    
    /**
      | Add a DispatchKey to the DispatchKey
      | set.
      | 
      | Does NOT mutate, returns the extended
      | DispatchKeySet!
      |
      */
    pub fn add(&self, t: DispatchKey) -> DispatchKeySet {
        
        todo!();
        /*
            return *this | DispatchKeySet(t);
        */
    }

    /**
      | Remove a DispatchKey from the DispatchKey
      | set.  This is generally not an operation you
      | should be doing (it's used to implement
      | operator<<)
      |
      */
    pub fn remove(&self, t: DispatchKey) -> DispatchKeySet {
        
        todo!();
        /*
            return DispatchKeySet(repr_ & ~DispatchKeySet(t).repr_);
        */
    }

    /// Is the set empty?  (AKA undefined tensor)
    ///
    pub fn empty(&self) -> bool {
        
        todo!();
        /*
            return repr_ == 0;
        */
    }
    
    pub fn raw_repr(&mut self) -> u64 {
        
        todo!();
        /*
            return repr_;
        */
    }

    /**
      | Return the type id in this set with the
      | highest priority (i.e., is the largest in
      | the DispatchKey enum).  Intuitively, this
      | type id is the one that should handle
      | dispatch (assuming there aren't any further
      | exclusions or inclusions).
      |
      */
    pub fn highest_priority_type_id(&self) -> DispatchKey {
        
        todo!();
        /*
            // TODO: If I put Undefined as entry 64 and then adjust the
        // singleton constructor to shift from the right, we can get rid of the
        // subtraction here.  It's modestly more complicated to get right so I
        // didn't do it for now.
        return static_cast<DispatchKey>(64 - llvm::countLeadingZeros(repr_));
        */
    }
    
    pub fn highest_priority_backend_type_id(&self) -> DispatchKey {
        
        todo!();
        /*
            return (*this &
                ((1ULL << static_cast<uint8_t>(DispatchKey::EndOfBackendKeys)) - 1))
            .highestPriorityTypeId();
        */
    }
}

impl From<u64> for DispatchKeySet {

    fn from(x: u64) -> Self {

        todo!();
        /*
        : repr(repr),
        */
    }
}

macro_rules! dispatch_key_set{

    ($($arg:expr),*) => {
        DispatchKeySet::new_from_keys(&[ $($arg),* ]);
    }
}

lazy_static!{

    /**
      | autograd_dispatch_keyset should include all
      | runtime autograd keys.
      |
      | Alias key DispatchKey::Autograd maps to
      | autograd_dispatch_keyset.
      |
      | NB: keys in this set also get associated with
      | CompositeImplicitAutograd
      |
      */
    pub static ref AUTOGRAD_DISPATCH_KEYSET: DispatchKeySet = dispatch_key_set!{
        DispatchKey::AutogradCPU,
        DispatchKey::AutogradCUDA,
        DispatchKey::AutogradXLA,
        DispatchKey::AutogradNestedTensor,
        DispatchKey::AutogradMLC,
        DispatchKey::AutogradHPU,
        DispatchKey::AutogradXPU,
        DispatchKey::AutogradPrivateUse1,
        DispatchKey::AutogradPrivateUse2,
        DispatchKey::AutogradPrivateUse3,
        DispatchKey::AutogradOther
    };

    pub static ref AUTOCAST_DISPATCH_KEYSET: DispatchKeySet = dispatch_key_set!{
        DispatchKey::AutocastCPU,
        DispatchKey::AutocastCUDA
    };

    // See Note [TLS Initialization]
    pub static ref DEFAULT_INCLUDED_SET: DispatchKeySet = dispatch_key_set!{
        DispatchKey::BackendSelect,
        DispatchKey::ADInplaceOrView
    };

    pub static ref DEFAULT_EXCLUDED_SET: DispatchKeySet = dispatch_key_set!{
        DispatchKey::AutocastCPU,
        DispatchKey::AutocastCUDA
    };

    pub static ref AUTOGRAD_DISPATCH_KEYSET_WITH_AD_INPLACE_OR_VIEW: DispatchKeySet = *AUTOGRAD_DISPATCH_KEYSET | dispatch_key_set!(DispatchKey::ADInplaceOrView);

    // backend dispatch keys that map to
    // DispatchKey::AutogradOther
    //
    // NB: keys in this set also get associated with
    // CompositeImplicitAutograd
    //
    pub static ref AUTOGRADOTHER_BACKENDS: DispatchKeySet = dispatch_key_set!{
        DispatchKey::HIP,
        DispatchKey::FPGA,
        DispatchKey::MSNPU,
        DispatchKey::Vulkan,
        DispatchKey::Metal,
        DispatchKey::QuantizedCPU,
        DispatchKey::QuantizedCUDA,
        DispatchKey::CustomRNGKeyId,
        DispatchKey::MkldnnCPU,
        DispatchKey::SparseCPU,
        DispatchKey::SparseCUDA,
        DispatchKey::SparseHIP,
        DispatchKey::SparseCsrCPU,
        DispatchKey::SparseCsrCUDA,
        DispatchKey::Meta
    };
}


/// The set of dispatch keys that come after
/// autograd n.b. this relies on the fact that
/// AutogradOther is currently the lowest Autograd
/// key
///
lazy_static!{
    /*
    pub const AFTER_AUTOGRAD_KEYSET: DispatchKeySet = dispatch_key_set!(DispatchKeySet::FullAfter, DispatchKey::AutogradOther);
    */
}

/// The set of dispatch keys that come after
/// ADInplaceOrView
///
lazy_static!{
    /*
    pub const AFTER_AD_INPLACE_OR_VIEW_KEYSET: DispatchKeySet = dispatch_key_set!{
        DispatchKeySet::FullAfter,
        DispatchKey::ADInplaceOrView
    };
    */
}


/**
  | Historically, every tensor only had a single
  | DispatchKey, and it was always something like
  | CPU, and there wasn't any of this business
  | where TLS could cause the DispatchKey of
  | a tensor to change.
  |
  | But we still have some legacy code that is
  | still using DispatchKey for things like
  | instanceof checks; if at all possible, refactor
  | the code to stop using DispatchKey in those
  | cases.
  |
  */
#[inline] pub fn legacy_extract_dispatch_key(s: DispatchKeySet) -> DispatchKey {
    
    todo!();
        /*
            // NB: If you add any extra keys that can be stored in TensorImpl on
      // top of existing "backend" keys like CPU/Cuda, you need to add it
      // here.  At the moment, autograd keys and ADInplaceOrView key need this
      // treatment;
      return (s - autograd_dispatch_keyset_with_ADInplaceOrView -
              autocast_dispatch_keyset)
          .highestPriorityTypeId();
        */
}

/**
  | Given a function type, constructs
  | a function_traits type that drops the first
  | parameter type if the first parameter is of
  | type DispatchKeySet.
  |
  | NB: DispatchKeySet is currently explicitly
  | hidden from JIT (mainly to avoid pushing
  | unnecessary arguments on the stack - see Note
  | [ Plumbing Keys Through the Dispatcher] for
  | details).
  |
  | If at any point in the future we need to expose
  | this type to JIT, revisit the usage of this
  | type alias.
  |
  */
lazy_static!{
    /*
    pub type IsNotDispatchKeySet<T> = Negation<IsSame<DispatchKeySet,T>>;

    template <class FuncType>
    using remove_DispatchKeySet_arg_from_func = make_function_traits_t<
        typename infer_function_traits_t<FuncType>::return_type,
        typename conditional_t<
            is_same<
                DispatchKeySet,
                typename typelist::head_with_default_t<
                    void,
                    typename infer_function_traits_t<
                        FuncType>::parameter_types>>::value,
            typelist::drop_if_nonempty_t<
                typename infer_function_traits_t<FuncType>::parameter_types,
                1>,
            typename infer_function_traits_t<FuncType>::parameter_types>>;
    */
}

//-------------------------------------------[.cpp/pytorch/c10/core/DispatchKeySet.cpp]

/**
  | backend_dispatch_keyset should include all
  | runtime backend keys.
  |
  | Alias key
  | DispatchKey::CompositeExplicitAutograd maps to
  | backend_dispatch_keyset NestedTensor has been
  | explicitly removed due to incompatibility with
  | some kernels, such as structured kernels, that
  | use the DefaultBackend key.
  |
  */
lazy_static!{
    pub static ref BACKEND_DISPATCH_KEYSET: DispatchKeySet = *AUTOGRADOTHER_BACKENDS | dispatch_key_set!{
        DispatchKey::CPU,
        DispatchKey::Cuda,
        DispatchKey::XLA,
        DispatchKey::XPU,
        DispatchKey::PrivateUse1,
        DispatchKey::PrivateUse2,
        DispatchKey::PrivateUse3,
        DispatchKey::MLC,
        DispatchKey::HPU,
        DispatchKey::Meta
    };
}

/// true if t is a backend dispatch key
///
pub fn is_backend_dispatch_key(t: DispatchKey) -> bool {
    
    todo!();
        /*
            return t != DispatchKey::Undefined && backend_dispatch_keyset.has(t);
        */
}

/**
  | math_dispatch_keyset contains all keys in
  | backend_dispatch_keyset and
  | autograd_dispatch_keyset Alias key
  | DispatchKey::CompositeImplicitAutograd maps to
  | math_dispatch_keyset.
  */
lazy_static!{
    pub static ref MATH_DISPATCH_KEYSET: DispatchKeySet = *BACKEND_DISPATCH_KEYSET | *AUTOGRAD_DISPATCH_KEYSET;
}

/**
  | Resolve alias dispatch key to DispatchKeySet
  | if applicable
  |
  */
pub fn get_runtime_dispatch_key_set(t: DispatchKey) -> DispatchKeySet {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(t != DispatchKey::Undefined);
      switch (t) {
        case DispatchKey::Autograd:
          return autograd_dispatch_keyset;
        case DispatchKey::CompositeImplicitAutograd:
          return math_dispatch_keyset;
        case DispatchKey::CompositeExplicitAutograd:
          return backend_dispatch_keyset;
        default:
          return DispatchKeySet(t);
      }
        */
}

/**
  | Returns a DispatchKeySet of all backend keys
  | mapped to Autograd dispatch key t,
  | DispatchKeySet is empty if t is not alias of
  | DispatchKey::Autograd.
  |
  | for a given autograd key, return the
  | (guaranteed nonempty) set of associated backend
  | keys. for a non-autograd key, return the empty
  | keyset.
  |
  */
pub fn get_backend_key_set_from_autograd(t: DispatchKey) -> DispatchKeySet {
    
    todo!();
        /*
            switch (t) {
        case DispatchKey::AutogradCPU:
          return DispatchKeySet(DispatchKey::CPU);
        case DispatchKey::AutogradCUDA:
          return DispatchKeySet(DispatchKey::Cuda);
        case DispatchKey::AutogradXLA:
          return DispatchKeySet(DispatchKey::XLA);
        case DispatchKey::AutogradMLC:
          return DispatchKeySet(DispatchKey::MLC);
        case DispatchKey::AutogradHPU:
          return DispatchKeySet(DispatchKey::HPU);
        case DispatchKey::AutogradNestedTensor:
          return DispatchKeySet(DispatchKey::NestedTensor);
        case DispatchKey::AutogradXPU:
          return DispatchKeySet(DispatchKey::XPU);
        case DispatchKey::AutogradPrivateUse1:
          return DispatchKeySet(DispatchKey::PrivateUse1);
        case DispatchKey::AutogradPrivateUse2:
          return DispatchKeySet(DispatchKey::PrivateUse2);
        case DispatchKey::AutogradPrivateUse3:
          return DispatchKeySet(DispatchKey::PrivateUse3);
        case DispatchKey::AutogradOther:
          return autogradother_backends;
        default:
          return DispatchKeySet();
      }
        */
}

/**
  | Returns a DispatchKeySet of autocast
  | related keys mapped to backend.
  |
  */
pub fn get_autocast_related_key_set_from_backend(t: DispatchKey) -> DispatchKeySet {
    
    todo!();
        /*
            switch (t) {
        case DispatchKey::CPU:
          return DispatchKeySet(DispatchKey::AutocastCPU);
        case DispatchKey::Cuda:
          return DispatchKeySet(DispatchKey::AutocastCUDA);
        default:
          return DispatchKeySet();
      }
        */
}

/**
  | Returns a DispatchKeySet of autograd
  | related keys mapped to backend.
  |
  */
pub fn get_autograd_related_key_set_from_backend(t: DispatchKey) -> DispatchKeySet {
    
    todo!();
        /*
            return DispatchKeySet(
          {DispatchKey::ADInplaceOrView, getAutogradKeyFromBackend(t)});
        */
}

/**
  | This API exists because we have a use case for
  | checking
  | getRuntimeDispatchKeySet(alias).has(DispatchKey::Undefined)
  | in OperatorEntry.cpp but we disallow it in
  | has() API.
  */
pub fn is_included_in_alias(
        k:     DispatchKey,
        alias: DispatchKey) -> bool {
    
    todo!();
        /*
            return k != DispatchKey::Undefined && getRuntimeDispatchKeySet(alias).has(k);
        */
}

pub fn to_string(ts: DispatchKeySet) -> String {
    
    todo!();
        /*
            stringstream ss;
      ss << ts;
      return ss.str();
        */
}

impl fmt::Display for DispatchKeySet {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            if (ts.empty()) {
        os << "DispatchKeySet()";
        return os;
      }
      os << "DispatchKeySet(";
      DispatchKey tid;
      bool first = true;
      while ((tid = ts.highestPriorityTypeId()) != DispatchKey::Undefined) {
        if (!first) {
          os << ", ";
        }
        os << tid;
        ts = ts.remove(tid);
        first = false;
      }
      os << ")";
      return os;
        */
    }
}
