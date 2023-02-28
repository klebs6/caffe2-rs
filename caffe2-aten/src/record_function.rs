crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/record_function.h]

/// Kind of record function scope;
///
#[repr(u8)]
pub enum RecordScope {

    /// c10/ATen ops, autograd nodes
    FUNCTION = 0,

    /// Functions/nodes called from the autograd
    BACKWARD_FUNCTION,

    /// TorchScript functions, methods
    TORCHSCRIPT_FUNCTION,

    /// Kernel Function dtype Tag
    KERNEL_FUNCTION_DTYPE,

    /// User defined scope (e.g. with
    /// record_function())
    USER_SCOPE,

    /// must be the last in the list
    NUM_SCOPES, 
}

lazy_static!{
    /*
    struct hash<RecordScope> {
      usize operator()(
          const RecordScope& sc) const {
        return static_cast<usize>(sc);
      }
    };
    */
}

pub struct StringView {
    owned_str_ptr: Arc<String>,
    str_ptr:       *const u8,
}

impl Default for StringView {
    
    fn default() -> Self {
        todo!();
        /*
        : string_view(nullptr),

        
        */
    }
}

impl StringView {
    
    pub fn new(str_ptr: *const u8) -> Self {
    
        todo!();
        /*
        : owned_str_ptr(nullptr),
        : str_ptr(str_ptr),

        
        */
    }
    
    pub fn new(str_: String) -> Self {
    
        todo!();
        /*


            : owned_str_ptr_(make_shared<string>(move(str))),
          str_ptr_(owned_str_ptr_->c_str())
        */
    }
    
    pub fn str_(&self) -> *const u8 {
        
        todo!();
        /*
            return str_ptr_;
        */
    }
}

impl fmt::Display for StringView {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            os << dt.str();
        return os;
        */
    }
}

impl PartialEq<StringView> for StringView {
    
    #[inline] fn eq(&self, other: &StringView) -> bool {
        todo!();
        /*
            return strcmp(lhs.str(), rhs.str()) == 0;
        */
    }
}

/// Soft limit on the number of callbacks to use;
///
pub const K_SOFT_LIMIT_CALLBACKS: usize = 4;

/**
  | An abstract base class for various observer
  | contexts that can be attached to the
  | RecordFunction.
  |
  */
pub struct ObserverContext {

}

pub type CallbackHandles      = SmallVector<u64,kSoftLimitCallbacks>;
pub type ObserverContextList  = Vec<Box<ObserverContext>>;
pub type RecordFunctionHandle = u64;

pub struct RecordFunctionState {

    /**
      | Whether any of the picked callbacks
      | require inputs
      |
      */
    needs_inputs:                 bool, // default = false

    /**
      | Whether any of the picked callbacks
      | require outputs
      |
      */
    needs_outputs:                bool, // default = false

    /**
      | In cases when RecordFunction might
      | be active but we chose not to use the observers
      | (e.g. operator is not observed), this
      | boolean flag is used to check whether
      | the start callbacks were called
      |
      */
    called_start_callbacks:       bool, // default = false

    /**
      | Whether the RecordFunction is pre-sampled
      |
      */
    pre_sampled:                  bool, // default = false

    /**
      | Used internally to keep track of thread
      | local and global callbacks that were
      | picked to run; must be sorted;
      |
      */
    sorted_active_tls_handles:    CallbackHandles,

    sorted_active_global_handles: CallbackHandles,

    /**
      | Stores various ObserverContext objects
      | with event metadata for thread local
      | callbacks.
      |
      */
    tls_ctx:                      ObserverContextList,

    /**
      | Stores various ObserverContext objects
      | with event metadata for global callbacks.
      |
      */
    global_ctx:                   ObserverContextList,

    name:                         StringView,
    sequence_nr:                  i64, // default = -1
    inputs:                       Vec<IValue>,
    outputs:                      Vec<IValue>,
    operator_name:                Option<OperatorName>,
    op_input_size:                usize, // default = { 0 }
    op_output_size:               usize, // default = { 0 }

    /**
      | Kind of scope this RecordFunction is
      | observing
      |
      */
    scope:                        RecordScope,

    /**
      | The logical thread_id that this RecordFunction
      | was created with
      |
      */
    thread_id:                    u64, // default = 0

    /**
      | For backward functions - thread id of
      | the the forward function
      |
      */
    fwd_thread_id:                u64, // default = 0

    /**
      | Unique id for this RecordFunction,
      | used in callbacks to track start and
      | end of ranges
      |
      */
    handle:                       RecordFunctionHandle, // default = { 0 }

    /**
      | Whether this record_function corresponds
      | to an async event or not. Async events
      | can complete in different threads or
      | follow a future-like pattern of use.
      |
      */
    is_async: bool, // default = { false }
}

impl RecordFunctionState {

    pub fn new(scope: RecordScope) -> Self {
    
        todo!();
        /*
        : scope(scope),
        */
    }
}

pub struct RecordFunction {
    state: Box<RecordFunctionState>,
}

impl RecordFunction {

    /**
      | Default constructor is used with before
      | function called afterwards:
      |
      |  scope - record scope that this function
      |  tracks
      |
      |  pre_sampled - whether this RecordFunction
      |    was already pre-sampled with kLowProb
      |    probability
      */
    pub fn new(
        scope:       RecordScope,
        pre_sampled: bool) -> Self {

        let scope: RecordScope = scope.unwrap_or(RecordScope_FUNCTION);

        let pre_sampled: bool = pre_sampled.unwrap_or(false);

        todo!();
        /*


        
        */
    }
    
    pub fn before<F>(&mut self, 
        fn_:                 F,
        args:                *const Vec<IValue>,
        current_sequence_nr: i64)  {

        let current_sequence_nr: i64 = current_sequence_nr.unwrap_or(-1);

        todo!();
        /*
            if (!isActive()) {
          return;
        }
        state_->inputs_ = *args;
        before(fn, current_sequence_nr);
        */
    }
    
    pub fn name(&self) -> &StringView {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called name() on inactive RecordFunction");
        return state_->name_;
        */
    }
    
    pub fn seq_nr(&self) -> i64 {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called seqNr() on inactive RecordFunction");
        return state_->sequence_nr_;
        */
    }
    
    pub fn inputs(&self) -> &Vec<IValue> {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called inputs() on inactive RecordFunction");
        return state_->inputs_;
        */
    }
    
    pub fn outputs(&self) -> &Vec<IValue> {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called outputs() on inactive RecordFunction");
        return state_->outputs_;
        */
    }
    
    pub fn set_outputs(&self, outputs: Vec<IValue>)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called setOutputs() on inactive RecordFunction");
        state_->outputs_ = move(outputs);
        */
    }
    
    pub fn set_outputs(&self, outputs: &[IValue])  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called setOutputs() on inactive RecordFunction");
        state_->outputs_ = outputs.vec();
        */
    }
    
    pub fn num_inputs(&self) -> usize {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called num_inputs() on inactive RecordFunction");
        return state_->op_input_size;
        */
    }
    
    pub fn num_outputs(&self) -> usize {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called num_outputs() on inactive RecordFunction");
        return state_->op_output_size;
        */
    }

    /**
      | Retrieves the thread_id that this
      | RecordFunction ran start callbacks with.
      |
      | Useful for writing thread safe end callbacks
      | that may be potentially executed in
      | a different thread (async ops)
      */
    pub fn thread_id(&self) -> u64 {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called threadId() on inactive RecordFunction");
        return state_->thread_id_;
        */
    }

    /**
      | For backward functions - thread id of the
      | corresponding forward function, or zero
      | otherwise;
      |
      | used alongside with sequence number to
      | correlate backward functions with the forward
      | ones
      */
    pub fn forward_thread_id(&self) -> u64 {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called forwardThreadId() on inactive RecordFunction");
        return state_->fwd_thread_id_;
        */
    }
    
    pub fn set_forward_thread_id(&mut self, thread_id: u64)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called setForwardThreadId() on inactive RecordFunction");
        state_->fwd_thread_id_ = thread_id;
        */
    }
    
    pub fn scope(&self) -> RecordScope {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called scope() on inactive RecordFunction");
        return state_->scope_;
        */
    }

    /**
      | Returns logical thread_id for the current
      | thread
      |
      */
    pub fn current_thread_id() -> u64 {
        
        todo!();
        /*
        
        */
    }

    // Internal functions, do not use directly;
    // used in python's context manager

    /**
      | before functions initialize RecordFunction
      | members and call start callbacks
      |
      */
    pub fn before(&mut self, 
        name:        *const u8,
        sequence_nr: i64)  {

        let sequence_nr: i64 = sequence_nr.unwrap_or(-1);

        todo!();
        /*
        
        */
    }
    
    pub fn before(&mut self, 
        name:        String,
        sequence_nr: i64)  {

        let sequence_nr: i64 = sequence_nr.unwrap_or(-1);

        todo!();
        /*
        
        */
    }
    
    pub fn before(&mut self, 
        op:          &OperatorHandle,
        sequence_nr: i64)  {

        let sequence_nr: i64 = sequence_nr.unwrap_or(-1);

        todo!();
        /*
        
        */
    }

    /// Sets node ID for distributed profiling
    pub fn set_default_node_id(default_node_id: i64)  {
        
        todo!();
        /*
        
        */
    }

    /// Gets node ID for distributed profiling
    pub fn get_default_node_id() -> i64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn before<F>(&mut self, 
        fn_:                 F,
        args:                &[IValue],
        current_sequence_nr: i64)  {

        let current_sequence_nr: i64 = current_sequence_nr.unwrap_or(-1);

        todo!();
        /*
            if (!isActive()) {
          return;
        }
        state_->inputs_ = args.vec();
        before(fn, current_sequence_nr);
        */
    }
    
    pub fn before<F>(&mut self, 
        fn_:                 F,
        args:                Vec<IValue>,
        current_sequence_nr: i64)  {

        let current_sequence_nr: i64 = current_sequence_nr.unwrap_or(-1);

        todo!();
        /*
            if (!isActive()) {
          return;
        }
        state_->inputs_ = move(args);
        before(fn, current_sequence_nr);
        */
    }

    /**
      | Calls end callbacks. After end(),
      | accessors will no longer provide useful
      | results.
      |
      */
    pub fn end(&mut self)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Internal-only, used only force async
      | event for distributed events profiling.
      |
      */
    pub fn set_async(&mut self)  {
        
        todo!();
        /*
        
        */
    }

    /// Returns whether this RecordFunction
    /// corresponds to an async event orn ot.
    ///
    pub fn is_async(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn handle(&self) -> RecordFunctionHandle {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called handle() on inactive RecordFunction");
        return state_->handle_;
        */
    }
    
    pub fn operator_name(&self) -> Option<OperatorName> {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called operator_name() on inactive RecordFunction");
        return state_->operator_name_;
        */
    }
    
    pub fn set_handle(&mut self, handle: RecordFunctionHandle)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called setHandle() on inactive RecordFunction");
        state_->handle_ = handle;
        */
    }

    /// Whether this RecordFunction runs any
    /// callbacks.
    ///
    pub fn is_active(&self) -> bool {
        
        todo!();
        /*
            return state_ != nullptr;
        */
    }
    
    pub fn needs_inputs(&self) -> bool {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called needsInputs() on inactive RecordFunction");
        return state_->needs_inputs;
        */
    }
    
    pub fn needs_outputs(&self) -> bool {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called needsOutputs() on inactive RecordFunction");
        return state_->needs_outputs;
        */
    }
}

/* ------- PyTorch callbacks/observers API:  ------- */

/** 
 | RecordFunctionCallback represents a pair of
 | callbacks to be used with
 |
 | RecordFunction, members:
 |
 |   start, end - the callbacks to run when
 |     entering and exiting the scope; optionally,
 |     the start callback may return an
 |     ObserverContext which will be passed to the
 |     end callback, use appropriate constructor
 |     accordingly.
 |
 |   needs_inputs - whether the callbacks need the
 |     inputs passed from the observed
 |     function/range; NOTE: passing the inputs
 |     incurs an additional overhead;
 |
 |   sampling_probability - if not 1.0, then the
 |     callback is probabilistically sampled to
 |     run; NOTE: start and end callbacks always
 |     run as a pair and are sampled together;
 |
 |   scopes - types of scopes to execute the
 |     callbacks on (see RecordScope); passing
 |     empty set means the callbacks will be
 |     executed for all possible scope types
 |
 |   should_run - optional function that returns
 |     whether this callback should run;
 |     overwrites the effect of setting
 |     sampling_probability
 |
 */
pub struct RecordFunctionCallback {
    start:         StartCallback,
    end:           EndCallback,
    sampling_prob: f64, // default = 1.0
    scopes:        Array<bool,RecordScope_NUM_SCOPES>, // default = {}
    needs_inputs:  bool, // default = false
    needs_outputs: bool, // default = false
    needs_ids:     bool, // default = false
}

pub mod record_function_callback {

    use super::*;

    pub type StartCallback = fn(_0: &RecordFunction) -> Box<ObserverContext>;
    pub type EndCallback   = fn(_0: &RecordFunction, _1: *mut ObserverContext) -> c_void;
}

impl RecordFunctionCallback {

    /**
      | This interface supports observers that
      | require passing an ObserverContext between
      | start and end callbacks.
      |
      */
    pub fn new(
        start: StartCallback,
        end:   EndCallback) -> Self {
        let end: EndCallback = end.unwrap_or(nullptr);
        todo!();
        /*
        : start(start),
        : end(end),

            scopes_.fill(true);
        */
    }
    
    pub fn needs_inputs(&mut self, needs_inputs: bool) -> &mut RecordFunctionCallback {
        
        todo!();
        /*
            needs_inputs_ = needs_inputs;
        return *this;
        */
    }
    
    pub fn needs_outputs(&mut self, needs_outputs: bool) -> &mut RecordFunctionCallback {
        
        todo!();
        /*
            needs_outputs_ = needs_outputs;
        return *this;
        */
    }
    
    pub fn needs_ids(&mut self, needs_ids: bool) -> &mut RecordFunctionCallback {
        
        todo!();
        /*
            needs_ids_ = needs_ids;
        return *this;
        */
    }
    
    pub fn sampling_prob(&mut self, sampling_prob: f64) -> &mut RecordFunctionCallback {
        
        todo!();
        /*
            TORCH_CHECK(sampling_prob >= 0.0 && sampling_prob <= 1.0,
            "Invalid sampling probability");
        sampling_prob_ = sampling_prob;
        return *this;
        */
    }
    
    pub fn scopes(&mut self, scopes: &HashSet<RecordScope,HashMap<RecordScope>>) -> &mut RecordFunctionCallback {
        
        todo!();
        /*
            if (!scopes.empty()) {
          scopes_.fill(false);
          for (auto sc : scopes) {
            scopes_[static_cast<usize>(sc)] = true;
          }
        } else {
          scopes_.fill(true);
        }
        return *this;
        */
    }
    
    pub fn needs_inputs(&self) -> bool {
        
        todo!();
        /*
            return needs_inputs_;
        */
    }
    
    pub fn needs_outputs(&self) -> bool {
        
        todo!();
        /*
            return needs_outputs_;
        */
    }
    
    pub fn needs_ids(&self) -> bool {
        
        todo!();
        /*
            return needs_ids_;
        */
    }
    
    pub fn sampling_prob(&self) -> f64 {
        
        todo!();
        /*
            return sampling_prob_;
        */
    }
    
    pub fn check_scope(&self, sc: RecordScope) -> bool {
        
        todo!();
        /*
            return scopes_[(usize)sc];
        */
    }
    
    pub fn start(&self) -> StartCallback {
        
        todo!();
        /*
            return start_;
        */
    }
    
    pub fn end(&self) -> EndCallback {
        
        todo!();
        /*
            return end_;
        */
    }
}

/**
  | Using macro to minimize inputs copies,
  | optional argument - function's seq_no
  |
  */
#[macro_export] macro_rules! record_function_with_scope {
    ($scope:ident, $fn:ident, $inputs:ident, $($arg:ident),*) => {
        /*
        
          RecordFunction guard(scope); 
          if (guard.isActive()) {          
            if (guard.needsInputs()) {                 
              guard.before(fn, inputs, ##__VA_ARGS__); 
            } else { 
              guard.before(fn, ##__VA_ARGS__); 
            } 
          }
        */
    }
}

#[macro_export] macro_rules! record_function {
    ($fn:ident, $inputs:ident, $($arg:ident),*) => {
        /*
        
          RECORD_FUNCTION_WITH_SCOPE( 
            RecordScope::FUNCTION, 
            fn, inputs, ##__VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! record_torchscript_function {
    ($mn:ident, $inputs:ident) => {
        /*
        
          RECORD_FUNCTION_WITH_SCOPE( 
            RecordScope::TORCHSCRIPT_FUNCTION, mn, inputs)
        */
    }
}

/**
  | Custom user scopes in C++; similar to
  | Python's 'with record_function("..."):'
  |
  */
#[macro_export] macro_rules! record_user_scope {
    ($fn:ident) => {
        /*
        
          RECORD_FUNCTION_WITH_SCOPE( 
            RecordScope::USER_SCOPE, fn, {})
        */
    }
}

/// RECORD_USER_SCOPE with inputs
#[macro_export] macro_rules! record_user_scope_with_inputs {
    ($fn:ident, $inputs:ident) => {
        /*
        
          RECORD_FUNCTION_WITH_SCOPE( 
            RecordScope::USER_SCOPE, fn, inputs)
        */
    }
}

/**
  | Notes:
  |
  |  - two types of callbacks are provided: thread
  |  local and global
  |
  |     - thread local callbacks are added/removed
  |       only for the given thread and are stored
  |       locally for each thread and separately
  |       from the list of the global callbacks
  |
  |     - global callbacks are stored in a single
  |       per process list and are invoked by every
  |       RecordFunction, in addition to the thread
  |       local callbacks specific to the given
  |       thread
  |
  |  - we allow the added callbacks to be sampled,
  |    by specifying a sampling probability for
  |    each callback pair, if the start callback is
  |    not picked to run, the corresponding end
  |    callback won't be called
  |
  |  - a typical use case for the global callbacks
  |    is passive monitoring in the background
  |    (e.g. fleet-wide monitoring), without
  |    focusing on the specific peice of code
  |
  |  - in contrast, thread local callbacks are
  |    enabled locally, on demand, for the specific
  |    piece of code (range) and are not sampled
  |
  |  - a typical use case for thread local
  |    callbacks is profiler and code execution
  |    tracer
  |
  |  - note, thread local callbacks are
  |    automatically propagated with
  |    ThreadLocalState across JIT continuations
  |    and async tasks (launch)
  |
  |  - adding/removing global callbacks is not
  |    thread safe and should be done only when no
  |    other code is running, e.g. during the
  |    initialization
  */
pub type CallbackHandle = u64;

pub struct GlobalRecordFunctionCallbacksEntry {
    callback: RecordFunctionCallback,
    enabled:  AtomicBool,
    handle:   CallbackHandle,
}

impl GlobalRecordFunctionCallbacksEntry {

    pub fn new(
        cb: RecordFunctionCallback,
        h:  CallbackHandle) -> Self {
    
        todo!();
        /*
        : callback(move(cb)),
        : enabled(true),
        : handle(h),

        
        */
    }

    /**
      | Copying is fine despite atomic<bool> not
      | being supposed to have a copy/move
      | constructor: adding & removing callbacks is
      | already not thread-safe.
      */
    pub fn new(rhs: &GlobalRecordFunctionCallbacksEntry) -> Self {
    
        todo!();
        /*
        : callback(rhs.callback),
        : enabled(rhs.enabled.load()),
        : handle(rhs.handle),

        
        */
    }
    
    pub fn assign_from(&mut self, rhs: &GlobalRecordFunctionCallbacksEntry) -> &mut GlobalRecordFunctionCallbacksEntry {
        
        todo!();
        /*
            callback = rhs.callback;
        enabled = rhs.enabled.load();
        handle = rhs.handle;
        return *this;
        */
    }
    
    pub fn new(rhs: GlobalRecordFunctionCallbacksEntry) -> Self {
    
        todo!();
        /*


            : callback(move(rhs.callback)), enabled(rhs.enabled.load()), handle(rhs.handle)
        */
    }
    
    pub fn assign_from(&mut self, rhs: GlobalRecordFunctionCallbacksEntry) -> &mut GlobalRecordFunctionCallbacksEntry {
        
        todo!();
        /*
            callback = move(rhs.callback);
        enabled = rhs.enabled.load();
        handle = rhs.handle;
        return *this;
        */
    }

    /**
      | Returns true if the status changed,
      | false otherwise.
      |
      */
    pub fn disable(&mut self) -> bool {
        
        todo!();
        /*
            bool expected = true;
        // NOTE: we use sequentially consistent access here and in
        // enable() because updating further atomic flags depends on this
        // operation.
        return enabled.compare_exchange_strong(expected, false);
        */
    }

    /**
      | Returns true if the status changed,
      | false otherwise.
      |
      */
    pub fn enable(&mut self) -> bool {
        
        todo!();
        /*
            bool expected = false;
        return enabled.compare_exchange_strong(expected, true);
        */
    }

    /**
      | Read the flag. Note that it is neither
      | necessary nor correct to check this before
      | calling enable() or disable().
      |
      */
    pub fn is_enabled(&self) -> bool {
        
        todo!();
        /*
            return enabled.load(memory_order_relaxed);
        */
    }
}

/**
  | It is unnecessary to use atomic operations for
  | enabling thread-local function
  | callbacks. Moreover, it prevents saving to
  | ThreadLocalState because atomic is
  | non-copyable.
  |
  */
pub struct ThreadLocalRecordFunctionCallbacksEntry {
    callback: RecordFunctionCallback,
    enabled:  bool, // default = true
    handle:   CallbackHandle,
}

impl ThreadLocalRecordFunctionCallbacksEntry {
    
    pub fn new(
        cb: RecordFunctionCallback,
        h:  CallbackHandle) -> Self {
    
        todo!();
        /*
        : callback(move(cb)),
        : handle(h),

        
        */
    }
    
    pub fn disable(&mut self) -> bool {
        
        todo!();
        /*
            auto old = enabled;
        enabled = false;
        return old != enabled;
        */
    }
    
    pub fn enable(&mut self) -> bool {
        
        todo!();
        /*
            auto old = enabled;
        enabled = true;
        return old != enabled;
        */
    }
    
    pub fn is_enabled(&self) -> bool {
        
        todo!();
        /*
            return enabled;
        */
    }
}

/**
  | Holds pairs (callbacks, unique_id)
  |
  */
pub type GlobalRecordFunctionCallbacks      = Vec<GlobalRecordFunctionCallbacksEntry>;
pub type ThreadLocalRecordFunctionCallbacks = Vec<ThreadLocalRecordFunctionCallbacksEntry>;

pub struct RecordFunctionGuard {
    prev_value: bool, // default = false
}

impl Drop for RecordFunctionGuard {
    fn drop(&mut self) {
        todo!();
        /*
            enableRecordFunction(prev_value_);
        */
    }
}

impl RecordFunctionGuard {
    
    pub fn new(is_enabled: bool) -> Self {
        let is_enabled: bool = is_enabled.unwrap_or(true);
        todo!();
        /*
        : prev_value(isRecordFunctionEnabled()),

            enableRecordFunction(is_enabled);
        */
    }
}

pub struct DisableRecordFunctionGuard {
    base: RecordFunctionGuard,
}

impl Default for DisableRecordFunctionGuard {
    
    fn default() -> Self {
        todo!();
        /*
        : record_function_guard(false),
        */
    }
}

pub struct RecordFunctionTLS {

    /**
      | Thread local vector of callbacks, holds
      | pairs (callbacks, unique_id); must
      | be sorted in increasing handles order
      |
      */
    sorted_tls_callbacks:        ThreadLocalRecordFunctionCallbacks,
    tls_record_function_enabled: bool, // default = true

    /**
      | Stores the number of coin flips before
      | the next successful coin flip
      |
      */
    tries_left:                  i32, // default = 0
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/record_function.cpp]

/// Used to generate unique callback handles
///
pub fn next_unique_callback_handle() -> CallbackHandle {
    
    todo!();
        /*
            static atomic<u64> unique_cb_id {1};
      return CallbackHandle(unique_cb_id++);
        */
}

pub fn next_unique_record_function_handle() -> RecordFunctionHandle {
    
    todo!();
        /*
            static atomic<u64> unique_rf_id {1};
      return RecordFunctionHandle(unique_rf_id++);
        */
}

pub fn rf_tls() -> &mut RecordFunctionTLS {
    
    todo!();
        /*
            #if defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
      static ThreadLocal<RecordFunctionTLS> rf_tls_;
      return rf_tls_.get();
    #else // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
      static thread_local RecordFunctionTLS rf_tls_;
      return rf_tls_;
    #endif // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
        */
}

lazy_static!{
    /*
    atomic<i64> defaultNodeId(-1);

    // Enumerates thread ids logically;
    // note: this_thread::get_id may return potentially
    // reused thread id
    atomic<u64> next_thread_id_ {0};

    thread_local u64 current_thread_id_ = 0;
    */
}


/// Low probability constant
pub const LOW_PROB: f64 = 0.001;

#[derive(Default)]
pub struct CoinflipTLS {
    tries_left:    i32,
    gen_geo:       Mt19937,
    gen_zero_one:  Mt19937,
    dist_geo:      GeometricDistribution<i32>,
    dist_zero_one: UniformRealDistribution<f64>,
}

impl Default for CoinflipTLS {
    
    fn default() -> Self {
    
        todo!();
        /*
            : tries_left_(0), genGeo_(random_device()()), genZeroOne_(random_device()()), distGeo_(kLowProb), distZeroOne_(0.0, 1.0)
        */
    }
}

pub fn coinflip_tls() -> &mut CoinflipTLS {
    
    todo!();
        /*
            #if defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
      static ThreadLocal<CoinflipTLS> coinflip_tls_;
      return coinflip_tls_.get();
    #else // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
      static thread_local CoinflipTLS coinflip_tls_;
      return coinflip_tls_;
    #endif // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
        */
}

pub fn sample_geometric() -> i32 {
    
    todo!();
        /*
            return coinflip_tls().distGeo_(coinflip_tls().genGeo_);
        */
}

pub fn sample_zero_one() -> f64 {
    
    todo!();
        /*
            return coinflip_tls().distZeroOne_(coinflip_tls().genZeroOne_);
        */
}

pub fn get_record_function_tls() -> &RecordFunctionTLS {
    
    todo!();
        /*
            return rf_tls();
        */
}

pub fn set_record_function_tls(tls: &RecordFunctionTLS)  {
    
    todo!();
        /*
            rf_tls() = tls;
        */
}

pub enum ToggledCallbackResult {
    NotFound,
    FoundButNotToggled,
    FoundAndToggled,
}

pub fn find_and_toggle_callback<RecordFunctionCallbacks>(
    cbs:     &mut RecordFunctionCallbacks,
    handle:  CallbackHandle,
    enabled: bool) -> ToggledCallbackResult {

    todo!();
        /*
            auto it = find_if(
          cbs.begin(), cbs.end(),
          [handle](
              const auto& el) {
            return el.handle == handle;
          });
      if (it != cbs.end()) {
        bool changed = enabled ? it->enable() : it->disable();
        if (!changed) {
          return ToggledCallbackResult::FoundButNotToggled;
        }
        if (it->callback.samplingProb() > kLowProb) {
          // try to disable/restore pre-sampling of RecordFunction
          if (enabled) {
            bumpRecordAllFunctions();
          } else {
            releaseRecordAllFunctions();
          }
        }
        return ToggledCallbackResult::FoundAndToggled;
      }
      return ToggledCallbackResult::NotFound;
        */
}

pub fn find_and_remove_callback<RecordFunctionCallbacks>(
    cbs:    &mut RecordFunctionCallbacks,
    handle: CallbackHandle) -> bool {

    todo!();
        /*
            auto it = find_if(
          cbs.begin(), cbs.end(),
          [handle](
              const auto& el) {
            return el.handle == handle;
          });
      if (it != cbs.end()) {
        // We do not need to try to call releaseRecordAllFunctions here
        // because findAndRemoveCallback is used only as a helper in
        // removeCallback. removeCallback calls disableCallback, which
        // calls findAndToggleCallback, which already will do a
        // releaseRecordAllFunctions for us.
        cbs.erase(it);
        return true;
      }
      return false;
        */
}

pub struct CallbackManager {

    /**
      | Global callbacks; must be sorted in
      | increasing handle order
      |
      */
    sorted_global_callbacks:      GlobalRecordFunctionCallbacks,
    num_enabled_global_callbacks: Atomic<UintFast32>,
}

impl Default for CallbackManager {
    
    fn default() -> Self {
        todo!();
        /*
        : num_enabled_global_callbacks(0),
        */
    }
}

impl CallbackManager {
    
    pub fn add_thread_local_callback(&mut self, cb: RecordFunctionCallback) -> CallbackHandle {
        
        todo!();
        /*
            if (cb.samplingProb() > kLowProb) {
          // pre-sampling of RecordFunction with prob. kLowProb cannot be used
          bumpRecordAllFunctions();
        }
        // note: monotonically increasing callbacks_unique_id keeps
        // sorted_tls_callbacks_ sorted
        auto handle = next_unique_callback_handle();
        rf_tls().sorted_tls_callbacks_.emplace_back(move(cb), handle);
        return handle;
        */
    }
    
    pub fn add_global_callback(&mut self, cb: RecordFunctionCallback) -> CallbackHandle {
        
        todo!();
        /*
            if (cb.samplingProb() > kLowProb) {
          // pre-sampling of RecordFunction with prob. kLowProb cannot be used
          bumpRecordAllFunctions();
        }
        auto handle = next_unique_callback_handle();
        sorted_global_callbacks_.emplace_back(move(cb), handle);
        num_enabled_global_callbacks_.fetch_add(1, memory_order_relaxed);
        return handle;
        */
    }
    
    pub fn remove_callback(&mut self, handle: CallbackHandle)  {
        
        todo!();
        /*
            // This could be implemented more efficiently, but callback
        // addition/removal is not intended to run in performance-critical
        // paths (it's not thread-safe and should be done during
        // initialization).
        disableCallback(handle);
        auto found = findAndRemoveCallback(rf_tls().sorted_tls_callbacks_, handle);
        if (!found) {
          found = findAndRemoveCallback(sorted_global_callbacks_, handle);
        }
        if (!found) {
          LOG(WARNING) << "Requested callback is not found";
        }
        */
    }
    
    pub fn disable_callback(&mut self, handle: CallbackHandle)  {
        
        todo!();
        /*
            auto found = findAndToggleCallback(
            rf_tls().sorted_tls_callbacks_, handle, false);
        if (found == ToggledCallbackResult::NotFound) {
          found = findAndToggleCallback(
              sorted_global_callbacks_, handle, false);
          if (found == ToggledCallbackResult::FoundAndToggled) {
            const auto previousCount = num_enabled_global_callbacks_.fetch_sub(1, memory_order_relaxed);
            TORCH_CHECK(previousCount > 0, previousCount);
          }
        }
        if (found == ToggledCallbackResult::NotFound) {
          LOG(WARNING) << "Requested callback is not found";
        }
        */
    }
    
    pub fn reenable_callback(&mut self, handle: CallbackHandle)  {
        
        todo!();
        /*
            auto found = findAndToggleCallback(
            rf_tls().sorted_tls_callbacks_, handle, true);
        if (found == ToggledCallbackResult::NotFound) {
          found = findAndToggleCallback(
              sorted_global_callbacks_, handle, true);
          if (found == ToggledCallbackResult::FoundAndToggled) {
            num_enabled_global_callbacks_.fetch_add(1, memory_order_relaxed);
          }
        }
        if (found == ToggledCallbackResult::NotFound) {
          LOG(WARNING) << "Requested callback is not found";
        }
        */
    }
    
    pub fn clear_global_callbacks(&mut self)  {
        
        todo!();
        /*
            sorted_global_callbacks_.clear();
        num_enabled_global_callbacks_ = 0;
        */
    }
    
    pub fn clear_thread_local_callbacks(&mut self)  {
        
        todo!();
        /*
            rf_tls().sorted_tls_callbacks_.clear();
        */
    }
    
    #[inline] pub fn has_global_callbacks(&self) -> bool {
        
        todo!();
        /*
            return num_enabled_global_callbacks_.load(memory_order_relaxed) > 0;
        */
    }
    
    #[inline] pub fn has_thread_local_callbacks(&self) -> bool {
        
        todo!();
        /*
            return !rf_tls().sorted_tls_callbacks_.empty();
        */
    }

    /**
      | We need this function to be inlined: init()
      | is a hot path and callbackShouldRun is even
      | hotter because it's called multiple times per
      | init().
      |
      | Profiling shows that the function prologue is
      | taking up a significant fraction of the time.
      */
    #[inline(always)] pub fn callback_should_run(
        cb:          &RecordFunctionCallback,
        scope:       RecordScope,
        pre_sampled: bool) -> bool {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
            !pre_sampled || (cb.sampling_prob_ <= kLowProb),
            "Incorrect usage of a pre-sampled RecordFunction with a high-frequency "
            " or non-sampled callback");

        // first check whether this callback is interested in
        // the given scope type
        if (!cb.checkScope(scope)) {
          return false;
        }

        // otherwise potentially do the sampling
        double sampling_prob = cb.sampling_prob_;
        if (pre_sampled) {
          // adjust the sampling rate to account for kLowProb pre-sampling of
          // the RecordFunction
          sampling_prob /= kLowProb;
        }

        if (sampling_prob < 1.0) {
          // model the low probability events as events happening
          // with probability kLowProb followed by another sampling with
          // probability (sampling_prob / kLowProb), then replace the coin
          // flip for kLowProb with a thread local number of tries tries_left_
          // sampled from the geometric distribution.
          if (sampling_prob < kLowProb) {
            if (coinflip_tls().tries_left_ == 0) {
              coinflip_tls().tries_left_ = sample_geometric();
              return (sample_zero_one() < sampling_prob / kLowProb);
            } else {
              --coinflip_tls().tries_left_;
              return false;
            }
          } else {
            return (sample_zero_one() < sampling_prob);
          }
        }

        return true;
        */
    }

    /**
      | init is called by RecordFunction in
      | constructor to determine which thread local
      | and global callbacks are going to be executed
      | and whether any of them need inputs
      */
    #[inline] pub fn init(&mut self, 
        rec_fn:      &mut RecordFunction,
        scope:       RecordScope,
        pre_sampled: bool)  {
        
        todo!();
        /*
            bool found_needs_inputs = false;
        bool found_needs_outputs = false;
        bool found_needs_ids = false;

        for (const auto& cb: rf_tls().sorted_tls_callbacks_) {
          if (cb.isEnabled() && callbackShouldRun(cb.callback, scope, pre_sampled)) {
            if (cb.callback.needsInputs()) {
              found_needs_inputs = true;
            }
            if (cb.callback.needsOutputs()) {
              found_needs_outputs = true;
            }
            if (cb.callback.needsIds()) {
              found_needs_ids = true;
            }
            if (!rec_fn.state_) {
              rec_fn.state_ = make_unique<RecordFunction::State>(scope);
            }
            rec_fn.state_->sorted_active_tls_handles_.push_back(cb.handle);
          }
        }

        for (const auto& cb: sorted_global_callbacks_) {
          if (cb.isEnabled() && callbackShouldRun(cb.callback, scope, pre_sampled)) {
            if (cb.callback.needsInputs()) {
              found_needs_inputs = true;
            }
            if (cb.callback.needsOutputs()) {
              found_needs_outputs = true;
            }
            if (cb.callback.needsIds()) {
              found_needs_ids = true;
            }
            if (!rec_fn.state_) {
              rec_fn.state_ = make_unique<RecordFunction::State>(scope);
            }
            rec_fn.state_->sorted_active_global_handles_.push_back(cb.handle);
          }
        }

        if (!rec_fn.state_) {
          return;
        }

        // Pre-allocate observer context list with nullptr.
        rec_fn.state_->tls_ctx_.resize(rec_fn.state_->sorted_active_tls_handles_.size());
        rec_fn.state_->global_ctx_.resize(rec_fn.state_->sorted_active_global_handles_.size());

        rec_fn.state_->needs_inputs = found_needs_inputs;
        rec_fn.state_->needs_outputs = found_needs_outputs;
        if (found_needs_ids) {
          rec_fn.setHandle(next_unique_record_function_handle());
        }
        */
    }
    
    pub fn run_start_callbacks(&mut self, rf: &mut RecordFunction)  {
        
        todo!();
        /*
            mergeRunCallbacks(
            sorted_global_callbacks_,
            rf.state_->sorted_active_global_handles_,
            rf.state_->global_ctx_,
            /* is_start */ true,
            rf);
        mergeRunCallbacks(
            rf_tls().sorted_tls_callbacks_,
            rf.state_->sorted_active_tls_handles_,
            rf.state_->tls_ctx_,
            /* is_start */ true,
            rf);
        rf.state_->called_start_callbacks_ = true;
        */
    }
    
    pub fn run_end_callbacks(&mut self, rf: &mut RecordFunction)  {
        
        todo!();
        /*
            mergeRunCallbacks(
            sorted_global_callbacks_,
            rf.state_->sorted_active_global_handles_,
            rf.state_->global_ctx_,
            /* is_start */ false,
            rf);
        mergeRunCallbacks(
            rf_tls().sorted_tls_callbacks_,
            rf.state_->sorted_active_tls_handles_,
            rf.state_->tls_ctx_,
            /* is_start */ false,
            rf);
        */
    }
    
    pub fn try_run_callback(&mut self, 
        rfcb:     &RecordFunctionCallback,
        rf:       &mut RecordFunction,
        ctx:      &mut Box<ObserverContext>,
        is_start: bool) -> bool {
        
        todo!();
        /*
            try {
          if (is_start) {
            ctx = rfcb.start() ? rfcb.start()(rf) : nullptr;
          }
          else {
            if (rfcb.end()) {
              rfcb.end()(rf, ctx.get());
            }
          }
          return true;
        } catch (const exception &e) {
          LOG(WARNING) << "Exception in RecordFunction callback: "
              << e.what() << " , for the range " << rf.name();
          return false;
        } catch (...) {
          LOG(WARNING) << "Exception in RecordFunction callback: unknown"
              << " , for the range " << rf.name();
          return false;
        }
        */
    }
    
    
    pub fn merge_run_callbacks<RecordFunctionCallbacks>(&mut self, 
        sorted_callbacks: &RecordFunctionCallbacks,
        sorted_handles:   &CallbackHandles,
        ctx_list:         &mut ObserverContextList,
        is_start:         bool,
        rf:               &mut RecordFunction)  {
    
        todo!();
        /*
            usize num_executed = 0;
        usize idx_c = 0;
        for (usize idx_h = 0; idx_h < sorted_handles.size() && idx_h < ctx_list.size(); ++idx_h) {
          while (idx_c < sorted_callbacks.size() &&
                sorted_callbacks[idx_c].handle < sorted_handles[idx_h]) {
            ++idx_c;
          }
          if (idx_c >= sorted_callbacks.size()) {
            break;
          }
          if (sorted_callbacks[idx_c].handle == sorted_handles[idx_h]) {
            tryRunCallback(sorted_callbacks[idx_c].callback, rf, ctx_list[idx_h], is_start);
            ++num_executed;
          }
        }

        if (num_executed != sorted_handles.size()) {
          C10_LOG_EVERY_MS(WARNING, 1000)
              << "Could not match some of the start callbacks with the corresponding end callbacks, "
              << "callbacks changed during RecordFunction lifetime; you might be trying to profile "
              << "the code after profiler is finished";
        }
        */
    }
}

/**
  | Keeping this static manager local.
  |
  */
pub fn manager() -> &mut CallbackManager {
    
    todo!();
        /*
            static CallbackManager _manager;
        return _manager;
        */
}

/**
  | for both thread local and global callbacks
  |
  */
pub fn has_callbacks() -> bool {
    
    todo!();
        /*
            auto& m = manager();
      return m.hasGlobalCallbacks() || m.hasThreadLocalCallbacks();
        */
}

/**
  | hasGlobalCallbacks returns whether
  | there're global callbacks registered
  | with pushGlobalCallback
  |
  */
pub fn has_global_callbacks() -> bool {
    
    todo!();
        /*
            return manager().hasGlobalCallbacks();
        */
}

/**
  | hasThreadLocalCallbacks returns
  | whether there're callbacks registered
  | with addThreadLocalCallback
  |
  */
pub fn has_thread_local_callbacks() -> bool {
    
    todo!();
        /*
            return manager().hasThreadLocalCallbacks();
        */
}

/**
  | addThreadLocalCallback adds a thread
  | local callback to run with RecordFunction,
  | returns handle to use with removeThreadLocalCallback
  |
  */
pub fn add_thread_local_callback(cb: RecordFunctionCallback) -> CallbackHandle {
    
    todo!();
        /*
      return manager().addThreadLocalCallback(move(cb));
        */
}

/**
  | addGlobalCallback adds a global callback
  | to run with RecordFunction:
  | 
  | WARNING: not thread safe, typically
  | addGlobalCallback can be called only
  | during the program initialization
  |
  */
pub fn add_global_callback(cb: RecordFunctionCallback) -> CallbackHandle {
    
    todo!();
        /*
      return manager().addGlobalCallback(move(cb));
        */
}

/**
  | removeCallback removes a callback
  | given the handle returned by addThreadLocalCallback
  | or addGlobalCallback;
  | 
  | WARNING: removing a global callback
  | is not thread safe, no other code can
  | run simultaneously
  |
  */
pub fn remove_callback(handle: CallbackHandle)  {
    
    todo!();
        /*
            manager().removeCallback(handle);
        */
}

/**
  | Prevent the given callback from executing.
  | If handle is invalid, does nothing.
  |
  */
pub fn disable_callback(handle: CallbackHandle)  {
    
    todo!();
        /*
            manager().disableCallback(handle);
        */
}

/**
  | Allow the given callback, previously
  | disabled with disableCallback, to
  | execute again. If handle is invalid,
  | does nothing.
  |
  */
pub fn reenable_callback(handle: CallbackHandle)  {
    
    todo!();
        /*
            manager().reenableCallback(handle);
        */
}

/**
  | clearGlobalCallbacks removes all
  | global callbacks
  | 
  | WARNING: not thread safe
  |
  */
pub fn clear_global_callbacks()  {
    
    todo!();
        /*
            manager().clearGlobalCallbacks();
        */
}

/**
  | clearThreadLocalCallbacks removes
  | all thread local callbacks
  |
  */
pub fn clear_thread_local_callbacks()  {
    
    todo!();
        /*
            manager().clearThreadLocalCallbacks();
        */
}

/**
  | not thread safe
  |
  */
pub fn clear_callbacks()  {
    
    todo!();
        /*
            auto& m = manager();
      m.clearGlobalCallbacks();
      m.clearThreadLocalCallbacks();
        */
}

/**
  | isRecordFunctionEnabled returns
  | whether RecordFunction is enabled
  | thread locally
  |
  */
pub fn is_record_function_enabled() -> bool {
    
    todo!();
        /*
            return rf_tls().tls_record_function_enabled_;
        */
}

/**
  | enableRecordFunction enables RecordFunction
  | thread locally
  |
  */
pub fn enable_record_function(enable: bool)  {

    let enable: bool = enable.unwrap_or(true);
    
    todo!();
        /*
            rf_tls().tls_record_function_enabled_ = enable;
        */
}

impl Drop for RecordFunction {
    fn drop(&mut self) {
        todo!();
        /*
            end();
        */
    }
}

impl RecordFunction {
    
    pub fn new(
        scope:       RecordScope,
        pre_sampled: bool) -> Self {
    
        todo!();
        /*


            auto* rf_tls_ptr = &rf_tls();
      if (rf_tls_ptr->tls_record_function_enabled_) {
        auto& m = manager();
        if (!m.sorted_global_callbacks_.empty() || !rf_tls_ptr->sorted_tls_callbacks_.empty()) {
          m.init(*this, scope, pre_sampled);
        }
      }
        */
    }
    
    pub fn current_thread_id(&mut self) -> u64 {
        
        todo!();
        /*
            if (!current_thread_id_) {
        // happens only once per thread
        current_thread_id_ = ++next_thread_id_;
      }
      return current_thread_id_;
        */
    }
    
    pub fn before(&mut self, 
        name:        *const u8,
        sequence_nr: i64)  {
        
        todo!();
        /*
            if (!isActive()) {
        return;
      }
      state_->name_ = StringView(name);
      state_->sequence_nr_ = sequence_nr;
      state_->thread_id_ = currentThreadId();
      state_->operator_name_.reset();

      manager().runStartCallbacks(*this);
        */
    }
    
    pub fn before(&mut self, 
        name:        String,
        sequence_nr: i64)  {
        
        todo!();
        /*
            if (!isActive()) {
        return;
      }
      state_->name_ = StringView(move(name));
      state_->sequence_nr_ = sequence_nr;
      state_->thread_id_ = currentThreadId();
      state_->operator_name_.reset();

      manager().runStartCallbacks(*this);
        */
    }
    
    pub fn before(&mut self, 
        op:          &OperatorHandle,
        sequence_nr: i64)  {
        
        todo!();
        /*
            if (!isActive()) {
        return;
      }
      state_->sequence_nr_ = sequence_nr;
      state_->thread_id_ = currentThreadId();
      state_->operator_name_ = op.operator_name();
      state_->op_input_size = op.schema().arguments().size();
      state_->op_output_size = op.schema().returns().size();
      state_->name_ = StringView(op.schema().name());

      manager().runStartCallbacks(*this);
        */
    }
    
    pub fn set_default_node_id(&mut self, new_default_node_id: i64)  {
        
        todo!();
        /*
            TORCH_CHECK(newDefaultNodeId >= 0, "setDefaultNodeId expects an id >= 0.");
      defaultNodeId = newDefaultNodeId;
        */
    }
    
    pub fn get_default_node_id(&mut self) -> i64 {
        
        todo!();
        /*
            return defaultNodeId;
        */
    }
    
    pub fn end(&mut self)  {
        
        todo!();
        /*
            if (isActive() && state_->called_start_callbacks_) {
        manager().runEndCallbacks(*this);
        state_.reset();
      }
        */
    }
    
    pub fn set_async(&mut self)  {
        
        todo!();
        /*
            if (isActive()) {
        state_->is_async_ = true;
      }
        */
    }
    
    pub fn is_async(&self) -> bool {
        
        todo!();
        /*
            if (isActive()) {
        return state_->is_async_;
      }
      return false;
        */
    }
}

// RecordFunction pre-sampling
//
// Whether to try to create RecordFunction on each
// call (>0) or use pre-sampling (=0)
//
lazy_static!{
    /*
    atomic<int> global_record_all_functions_ {0};
    */
}

/**
  | The following functions are used to
  | disable/enable pre-sampling of RecordFunction
  | when high-frequency/non-sampled callbacks are
  | added/removed.
  |
  | Note: every call to bumpRecordAllFunctions() is
  | supposed to be matched with the corresponding
  | releaseRecordAllFunctions() call.
  |
  | Note: disabling pre-sampling of RecordFunction
  | incurs an extra overhead, since RecordFunction
  | will be created for each operator call.
  |
  */
pub fn bump_record_all_functions()  {
    
    todo!();
        /*
            global_record_all_functions_.fetch_add(1, memory_order_relaxed);
        */
}


pub fn release_record_all_functions()  {
    
    todo!();
        /*
            TORCH_CHECK(global_record_all_functions_.fetch_sub(1, memory_order_relaxed) > 0);
        */
}


pub fn check_record_all_functions() -> bool {
    
    todo!();
        /*
            return (global_record_all_functions_.load(memory_order_relaxed) > 0);
        */
}

/**
  | Checks whether RecordFunction should be called,
  | sets boolean pointed by the argument to whether
  | pre-sampling was used
  |
  */
pub fn should_run_record_function(pre_sampled: *mut bool) -> bool {
    
    todo!();
        /*
            auto* rf_tls_ptr = &rf_tls();
      if (rf_tls_ptr->sorted_tls_callbacks_.empty() && !manager().hasGlobalCallbacks()) {
        *pre_sampled = false;
        return false;
      }
      if (global_record_all_functions_.load(memory_order_relaxed) > 0) {
        *pre_sampled = false;
        return true;
      }
      if (!rf_tls_ptr->tls_record_function_enabled_) {
        *pre_sampled = false;
        return false;
      }

      *pre_sampled = true;
      auto* coinflip_tls_ptr = &coinflip_tls();
      if (coinflip_tls_ptr->tries_left_ == 0) {
        coinflip_tls_ptr->tries_left_ = sample_geometric();
        return true;
      } else {
        --coinflip_tls_ptr->tries_left_;
        return false;
      }
        */
}
