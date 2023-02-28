/**
  | TLS management for DispatchKeySet (the "local"
  | DispatchKeySet(s))
  |
  | This manages two thread-local DispatchKeySets:
  |
  |  - The included type set, which adds a tensor
  |    type for consideration in dispatch.  (For
  |    example, you might add Profiling to the
  |    included type set to turn on profiling on
  |    all tensor operations.)
  |
  |  - The excluded type set, which disqualifies
  |    a tensor type from dispatch. (For example,
  |    after redispatching on variable, we
  |    disqualify Autograd so we don't attempt to
  |    handle variable again.) (Exclusion wins over
  |    inclusion.)
  |
  | NB: Originally, I implemented the excluded type
  | set as storing the inverted set, but TLS is
  | defined to be zero-initialized, so this doesn't
  | actually work (if it's inverted, you want the
  | set to be -1 initialized).
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/impl/LocalDispatchKeySet.h]

/**
  | POD version of LocalDispatchKeySet.  Declared
  | here just so that we can put it in the guards.
  |
  | This struct encapsulates special handling for
  | TLS initialization in set_included()/included()
  | API so that they reflect the truth.
  |
  | If you want to create PODLocalDispatchKeySet
  | with non-zero state, use set_included() instead
  | of default constructor.
  |
  */
pub struct PODLocalDispatchKeySet {
    included: u64,
    excluded: u64,
}

impl PODLocalDispatchKeySet {

    /// See Note [TLS Initialization]
    pub fn included(&self) -> DispatchKeySet {
        
        todo!();
        /*
            return DispatchKeySet(DispatchKeySet::RAW, included_) ^
            default_included_set;
        */
    }
    
    pub fn excluded(&self) -> DispatchKeySet {
        
        todo!();
        /*
            return DispatchKeySet(DispatchKeySet::RAW, excluded_) ^
            default_excluded_set;
        */
    }
    
    pub fn set_included(&mut self, x: DispatchKeySet)  {
        
        todo!();
        /*
            included_ = (x ^ default_included_set).raw_repr();
        */
    }
    
    pub fn set_excluded(&mut self, x: DispatchKeySet)  {
        
        todo!();
        /*
            excluded_ = (x ^ default_excluded_set).raw_repr();
        */
    }
}

/// PODLocalDispatchKeySet must be a POD type
unsafe impl Pod for PODLocalDispatchKeySet {}

pub struct LocalDispatchKeySet {
    included: DispatchKeySet,
    excluded: DispatchKeySet,
}

impl LocalDispatchKeySet {
    
    pub fn new(x: PODLocalDispatchKeySet) -> Self {
    
        todo!();
        /*
        : included(x.included()),
        : excluded(x.excluded()),

        
        */
    }
}

/**
  | thread_local variables cannot be  on Windows.
  |
  | Inlining this seems to break
  | AutoDispatchBelowAutograd on Android.
  |
  */
#[cfg(any(_MSC_VER,C10_ANDROID))]
pub fn tls_local_dispatch_key_set() -> LocalDispatchKeySet {
    
    todo!();
        /*
        
        */
}

#[cfg(not(any(_MSC_VER,C10_ANDROID)))]
lazy_static!{
    /*
    extern  thread_local PODLocalDispatchKeySet raw_local_dispatch_key_set;
    */
}

#[cfg(not(any(_MSC_VER,C10_ANDROID)))]
#[inline] pub fn tls_local_dispatch_key_set() -> LocalDispatchKeySet {
    
    todo!();
        /*
            // Don't let people fiddle with the thread_local directly just
      // because they include this header.
      return raw_local_dispatch_key_set;
        */
}

/**
  | RAII API for manipulating the thread-local
  | dispatch state.
  |
  */
pub struct IncludeDispatchKeyGuard {

    /**
      | A little micro-optimization to save
      | us from tls_get_addr call on destruction
      |
      */
    tls:     *mut PODLocalDispatchKeySet,
    include: DispatchKeySet,
}

impl IncludeDispatchKeyGuard {
    
    pub fn new_from_key(k: DispatchKey) -> Self {
    
        todo!();
        /*
        : include_dispatch_key_guard(DispatchKeySet(k)),

        
        */
    }
}

pub struct ExcludeDispatchKeyGuard {

    /**
      | A little micro-optimization to save
      | us from tls_get_addr call on destruction
      |
      */
    tls:     *mut PODLocalDispatchKeySet,
    exclude: DispatchKeySet,
}

impl ExcludeDispatchKeyGuard {
    
    pub fn new_from_key(k: DispatchKey) -> Self {
    
        todo!();
        /*
        : exclude_dispatch_key_guard(DispatchKeySet(k)),
        */
    }
}

//-------------------------------------------[.cpp/pytorch/c10/core/impl/LocalDispatchKeySet.cpp]

/**
  | NB: POD, must be zero initialized!
  |
  | Note [TLS Initialization]
  |
  | We wanted raw_local_dispatch_key_set to be
  | initialized with non-zero state
  | e.g. BackendSelect and ADInplaceOrView in
  | included set.  But certain Windows compiler
  | (e.g the one used in ARVR tests) only allow TLS
  | to be zero-initialized. To preserve the
  | invariant that raw TLS storage of the default
  | state is zero, we obtain the actual include
  | keyset by XORing
  | raw_local_dispatch_key_set.included_ with
  | default_included_set.
  |
  | This logic is encapsulated in struct
  | PODLocalDispatchKeySet.
  |
  */
lazy_static!{
    /*
    thread_local PODLocalDispatchKeySet raw_local_dispatch_key_set;
    */
}

#[cfg(any(_MSC_VER,C10_ANDROID))]
pub fn tls_local_dispatch_key_set() -> LocalDispatchKeySet {
    
    todo!();
        /*
            return raw_local_dispatch_key_set;
        */
}

/// Internal, use ThreadLocalStateGuard
pub fn force_tls_local_dispatch_key_set(key_set: LocalDispatchKeySet)  {
    
    todo!();
        /*
            raw_local_dispatch_key_set.set_included(key_set.included_);
      raw_local_dispatch_key_set.set_excluded(key_set.excluded_);
        */
}

impl IncludeDispatchKeyGuard {

    /**
      | An RAII guard could snapshot and restore the
      | entire state (entire DispatchKeySet) as opposed
      | to only snapshotting and restoring the state of
      | its assigned DispatchKeySet.
      |
      | I'm not sure which is better.
      |
      | If only the RAII API is used, the two choices
      | are not distinguishable.
      |
      | However, if the guard chooses to snapshot and
      | restore the entire DispatchKeySet, the
      | interaction with the non-RAII API changes.
      | Consider this sequence of events:
      |
      | - An RAII guard is declared for a particular
      | DispatchKeySet, but snapshots the entire
      | current DispatchKeySet.
      |
      | - A call to the non-RAII API changes the state
      | for DispatchKeys outside the assigned set.
      |
      | - The RAII guard goes out of scope, restoring
      | the entire DispatchKeySet it snapshotted
      |
      |   (which restores the state for its own
      |   assigned DispatchKey and wipes out the state
      |   for the other DispatchKeys set by the
      |   non-RAII API).
      |
      | RAII API
      */
    pub fn new(include: DispatchKeySet) -> Self {
    
        todo!();
        /*


            : tls_(&raw_local_dispatch_key_set), include_(include - tls_->included()) 

      if (!include_.empty()) {
        tls_->set_included(tls_->included() | include_);
      }
        */
    }
}

impl Drop for IncludeDispatchKeyGuard {

    fn drop(&mut self) {
        todo!();
        /*
            if (!include_.empty()) {
        tls_->set_included(tls_->included() - include_);
      }
        */
    }
}

impl ExcludeDispatchKeyGuard {
    
    pub fn new(exclude: DispatchKeySet) -> Self {
    
        todo!();
        /*


            : tls_(&raw_local_dispatch_key_set), exclude_(exclude - tls_->excluded()) 

      if (!exclude_.empty()) {
        tls_->set_excluded(tls_->excluded() | exclude_);
      }
        */
    }
}

impl Drop for ExcludeDispatchKeyGuard {

    fn drop(&mut self) {
        todo!();
        /*
            if (!exclude_.empty()) {
        tls_->set_excluded(tls_->excluded() - exclude_);
      }
        */
    }
}

/**
  | Non-RAII API
  |
  | Please prefer using the RAII API. See
  | declarations in LocalDispatchKeySet.h for
  | details.
  |
  | Non-RAII API for manipulating the thread-local
  | dispatch state.
  |
  | Please prefer the RAII API.  The non-RAII API
  | may be useful when the included/excluded state
  | of a given DispatchKey must span many calls
  | from the Python to the C++, so you cannot
  | conveniently use an RAII guard.
  |
  | Example use case:  a Python context manager
  | that includes a certain DispatchKey, to ensure
  | ops running under the context manager dispatch
  | through that DispatchKey's registered
  | overrides.
  |
  | The non-RAII API is less efficient than the
  | RAII guards because both the getter and setter
  | will do a tls_getaddr lookup (the RAII struct
  | only needs one!)
  */
pub fn tls_is_dispatch_key_excluded(x: DispatchKey) -> bool {
    
    todo!();
        /*
            return raw_local_dispatch_key_set.excluded().has(x);
        */
}

pub fn tls_set_dispatch_key_excluded(
    x:             DispatchKey,
    desired_state: bool)  {
    
    todo!();
        /*
            auto* tls = &raw_local_dispatch_key_set;
      bool current_state = tls->excluded().has(x);
      if (desired_state != current_state) {
        if (desired_state) {
          tls->set_excluded(tls->excluded().add(x));
        } else {
          tls->set_excluded(tls->excluded().remove(x));
        }
      }
        */
}

pub fn tls_is_dispatch_key_included(x: DispatchKey) -> bool {
    
    todo!();
        /*
            return raw_local_dispatch_key_set.included().has(x);
        */
}

pub fn tls_set_dispatch_key_included(
    x:             DispatchKey,
    desired_state: bool)  {
    
    todo!();
        /*
            auto* tls = &raw_local_dispatch_key_set;
      bool current_state = tls->included().has(x);
      if (desired_state != current_state) {
        if (desired_state) {
          tls->set_included(tls->included().add(x));
        } else {
          tls->set_included(tls->included().remove(x));
        }
      }
        */
}

pub fn tls_is_dispatch_keyset_excluded(ks: DispatchKeySet) -> bool {
    
    todo!();
        /*
            return raw_local_dispatch_key_set.excluded().isSupersetOf(ks);
        */
}

pub fn tls_is_dispatch_keyset_included(ks: DispatchKeySet) -> bool {
    
    todo!();
        /*
            return raw_local_dispatch_key_set.included().isSupersetOf(ks);
        */
}
