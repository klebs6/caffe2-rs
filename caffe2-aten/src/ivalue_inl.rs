/*!
  | For custom class __init__ registration, we need
  | to pass in a function that looks like this:
  | [](IValue x, args...)
  |
  | However, make_boxed_from_unboxed_functor.h
  | automatically sets the input types of the
  | function by introspecting the types of the
  | functor (which is IValue in this
  | case). However, we need the type it binds to be
  | Foo.
  |
  | Instead, we pass in a lambda
  | [](ivalue_holder<CurClass> x, args...) from
  | which getTypePtr can recover the original class
  | pointer.
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/ivalue_inl.h]

pub fn static_intrusive_pointer_cast<T, U>(r: IntrusivePtr<U>) -> IntrusivePtr<T> {

    todo!();
        /*
            return intrusive_ptr<T>::reclaim(static_cast<T*>(r.release()));
        */
}

pub fn dynamic_intrusive_pointer_cast<T, U>(r: IntrusivePtr<U>) -> IntrusivePtr<T> {

    todo!();
        /*
            return intrusive_ptr<T>::reclaim(dynamic_cast<T*>(r.release()));
        */
}

pub struct TaggedCapsule<TaggedCapsuleType> {
    ivalue: IValue,
}

impl TaggedCapsule<TaggedCapsuleType> {

}

impl IValue {
    
    pub fn move_to_intrusive_ptr<T, NullType>(&mut self) -> IntrusivePtr<T,NullType> {
    
        todo!();
        /*
            auto t = intrusive_ptr<T, NullType>::reclaim(
          payload.u.as_intrusive_ptr == UndefinedTensorImpl::singleton()
          ? NullType::singleton()
          : static_cast<T*>(payload.u.as_intrusive_ptr));
      clearToNone();
      return t;
        */
    }
    
    pub fn to_intrusive_ptr<T, NullType>(&self) -> IntrusivePtr<T,NullType> {
    
        todo!();
        /*
            if (payload.u.as_intrusive_ptr == UndefinedTensorImpl::singleton()) {
        return intrusive_ptr<T, NullType>();
      }
      raw::intrusive_ptr::incref(payload.u.as_intrusive_ptr);
      return intrusive_ptr<T, NullType>::reclaim(
          static_cast<T*>(payload.u.as_intrusive_ptr));
        */
    }
    
    #[inline] pub fn to_future(&mut self) -> IntrusivePtr<Future> {
        
        todo!();
        /*
            AT_ASSERT(isFuture(), "Expected Future but got ", tagKind());
      return moveToIntrusivePtr<Future>();
        */
    }
    
    #[inline] pub fn to_future(&mut self) -> IntrusivePtr<Future> {
        
        todo!();
        /*
            AT_ASSERT(isFuture(), "Expected Future but got ", tagKind());
      return toIntrusivePtr<Future>();
        */
    }
    
    #[inline] pub fn to_rref(&mut self) -> IntrusivePtr<RRefInterface> {
        
        todo!();
        /*
            AT_ASSERT(isRRef(), "Expected RRef but got ", tagKind());
      return moveToIntrusivePtr<RRefInterface>();
        */
    }
    
    #[inline] pub fn to_rref(&mut self) -> IntrusivePtr<RRefInterface> {
        
        todo!();
        /*
            AT_ASSERT(isRRef(), "Expected RRef but got ", tagKind());
      return toIntrusivePtr<RRefInterface>();
        */
    }
    
    #[inline] pub fn to_quantizer(&mut self) -> IntrusivePtr<Quantizer> {
        
        todo!();
        /*
            AT_ASSERT(isQuantizer(), "Expected Quantizer but got ", tagKind());
      return moveToIntrusivePtr<Quantizer>();
        */
    }
    
    #[inline] pub fn to_quantizer(&mut self) -> IntrusivePtr<Quantizer> {
        
        todo!();
        /*
            AT_ASSERT(isQuantizer(), "Expected Quantizer but got ", tagKind());
      return toIntrusivePtr<Quantizer>();
        */
    }
    
    #[inline] pub fn to_string(&mut self) -> IntrusivePtr<ConstantString> {
        
        todo!();
        /*
            AT_ASSERT(isString(), "Expected String but got ", tagKind());
      return moveToIntrusivePtr<ConstantString>();
        */
    }
    
    #[inline] pub fn to_string(&mut self) -> IntrusivePtr<ConstantString> {
        
        todo!();
        /*
            AT_ASSERT(isString(), "Expected String but got ", tagKind());
      return toIntrusivePtr<ConstantString>();
        */
    }
    
    #[inline] pub fn to_object(&mut self) -> IntrusivePtr<Object> {
        
        todo!();
        /*
            AT_ASSERT(isObject(), "Expected Object but got ", tagKind());
      return moveToIntrusivePtr<Object>();
        */
    }
    
    #[inline] pub fn to_object(&mut self) -> IntrusivePtr<Object> {
        
        todo!();
        /*
            AT_ASSERT(isObject(), "Expected Object but got ", tagKind());
      return toIntrusivePtr<Object>();
        */
    }
    
    #[inline] pub fn to_py_object_holder(&mut self) -> IntrusivePtr<PyObjectHolder> {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(isPyObject(), "Expected PyObject but got ", tagKind());
      return moveToIntrusivePtr<PyObjectHolder>();
        */
    }
    
    #[inline] pub fn to_py_object_holder(&mut self) -> IntrusivePtr<PyObjectHolder> {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(isPyObject(), "Expected PyObject but got ", tagKind());
      return toIntrusivePtr<PyObjectHolder>();
        */
    }
    
    #[inline] pub fn to_enum_holder(&mut self) -> IntrusivePtr<EnumHolder> {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(isEnum(), "Expected Enum but got ", tagKind());
      return moveToIntrusivePtr<EnumHolder>();
        */
    }
    
    #[inline] pub fn to_enum_holder(&mut self) -> IntrusivePtr<EnumHolder> {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(isEnum(), "Expected Enum but got ", tagKind());
      return toIntrusivePtr<EnumHolder>();
        */
    }
    
    #[inline] pub fn to_complex_double(&self) -> Complex<f64> {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(isComplexDouble(), "Expected ComplexDouble but got ", tagKind());
      auto ptr = toIntrusivePtr<ComplexHolder>();
      return (*ptr).val;
        */
    }
    
    #[inline] pub fn to_tensor(&mut self) -> Tensor {
        
        todo!();
        /*
            if (C10_UNLIKELY(!isTensor())) {
        reportToTensorTypeError();
      }
      auto result = move(payload.as_tensor);
      // As far as I can tell, omitting the usual explicit destructor call
      // is not UB in and of itself, and it's a slight perf win. The
      // destructor is a no-op, because the moved-from Tensor is
      // effectively an intrusive_ptr in the null state, so we don't need
      // the behavior for correctness reasons either. Leaving this
      // explanatory comment, including commented-out destructor call, to
      // make this abundantly clear.
      //
      // payload.as_tensor.~Tensor();
      clearToNone();
      return result;
        */
    }
    
    #[inline] pub fn to_tensor(&mut self) -> &mut Tensor {
        
        todo!();
        /*
            if (C10_UNLIKELY(!isTensor())) {
        reportToTensorTypeError();
      }
      return payload.as_tensor;
        */
    }
    
    #[inline] pub fn to_tensor(&mut self) -> &Tensor {
        
        todo!();
        /*
            if (C10_UNLIKELY(!isTensor())) {
        reportToTensorTypeError();
      }
      return payload.as_tensor;
        */
    }
    
    #[inline] pub fn to_storage(&mut self) -> Storage {
        
        todo!();
        /*
            AT_ASSERT(isStorage(), "Expected Storage but got ", tagKind());
      return Storage(
          moveToIntrusivePtr<StorageImpl>());
        */
    }
    
    #[inline] pub fn to_storage(&mut self) -> Storage {
        
        todo!();
        /*
            AT_ASSERT(isStorage(), "Expected Storage but got ", tagKind());
      return Storage(toIntrusivePtr<StorageImpl>());
        */
    }
    
    #[inline] pub fn to_stream(&mut self) -> Stream {
        
        todo!();
        /*
            return Stream::unpack(payload.u.as_int);
        */
    }
    
    #[inline] pub fn to_stream(&mut self) -> Stream {
        
        todo!();
        /*
            return Stream::unpack(payload.u.as_int);
        */
    }
    
    #[inline] pub fn to_blob(&mut self) -> IntrusivePtr<Blob> {
        
        todo!();
        /*
            AT_ASSERT(isBlob(), "Expected Blob but got ", tagKind());
      return moveToIntrusivePtr<Blob>();
        */
    }
    
    #[inline] pub fn to_blob(&mut self) -> IntrusivePtr<Blob> {
        
        todo!();
        /*
            AT_ASSERT(isBlob(), "Expected Blob but got ", tagKind());
      return toIntrusivePtr<Blob>();
      ;
        */
    }
    
    #[inline] pub fn to_capsule(&mut self) -> IntrusivePtr<TorchCustomClassHolder> {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(isCapsule());
      return moveToIntrusivePtr<TorchCustomClassHolder>();
        */
    }
    
    #[inline] pub fn to_capsule(&mut self) -> IntrusivePtr<TorchCustomClassHolder> {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(isCapsule());
      return toIntrusivePtr<TorchCustomClassHolder>();
        */
    }
    
    #[inline] pub fn to_generator(&mut self) -> Generator {
        
        todo!();
        /*
            AT_ASSERT(isGenerator(), "Expected Generator but got ", tagKind());
      return Generator(moveToIntrusivePtr<GeneratorImpl>());
        */
    }
    
    #[inline] pub fn to_generator(&mut self) -> Generator {
        
        todo!();
        /*
            AT_ASSERT(isGenerator(), "Expected Generator but got ", tagKind());
      return Generator(toIntrusivePtr<GeneratorImpl>());
        */
    }
}

pub fn check_custom_class_type(
        expected_type: *const Type,
        actual_type:   *const Type)  {
    
    todo!();
        /*
        
        */
}

pub type Shared<T> = IntrusivePtr<T>;

pub struct ConstantString {
    base: IntrusivePtrTarget,
    str_: String,
}

impl ConstantString {
    
    pub fn new(str_: String) -> Self {
    
        todo!();
        /*
        : str_(move(str)),

        
        */
    }
    
    pub fn new(str_: StringView) -> Self {
    
        todo!();
        /*
        : str_(string(str)),

        
        */
    }
    
    pub fn create(str_: String) -> IntrusivePtr<ConstantString> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn create(str_: StringView) -> IntrusivePtr<ConstantString> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn create(str_: *const u8) -> IntrusivePtr<ConstantString> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn string(&self) -> &String {
        
        todo!();
        /*
            return str_;
        */
    }
    
    pub fn string_view(&self) -> StringView {
        
        todo!();
        /*
            return str_;
        */
    }
    
    pub fn operator_const_string_ref(&self) -> &str {
        
        todo!();
        /*
            return string();
        */
    }
}


///--------------------------
pub struct Tuple {

    base:     IntrusivePtrTarget,

    elements: Vec<IValue>,

    /**
      | lazily computed for unnamed tuples
      |
      */
    ty:       RefCell<Arc<TupleType>>,
}

impl Tuple {
 
    /**
      | named tuples have additional type information,
      | so we directly create them tagged
      |
      */
    pub fn create_named(
        elements: Vec<IValue>,
        ty:       Arc<TupleType>) -> IntrusivePtr<Tuple> {
        
        todo!();
        /*
            return make_intrusive<Tuple>(move(elements_), type_);
        */
    }
    
    pub fn create(elements: Vec<IValue>) -> IntrusivePtr<Tuple> {
        
        todo!();
        /*
            return make_intrusive<Tuple>(move(elements_));
        */
    }
    
    pub fn create<Args>(elements: Args) -> IntrusivePtr<Tuple> {
    
        todo!();
        /*
            return make_intrusive<Tuple>(
            vector<IValue>{IValue(forward<Args>(elements_))...});
        */
    }
    
    pub fn elements(&mut self) -> &Vec<IValue> {
        
        todo!();
        /*
            return elements_;
        */
    }
    
    pub fn operator_const_vector_i_value_ref(&mut self) -> &Vec<IValue>  {
        
        todo!();
        /*
            return elements();
        */
    }
    
    pub fn elements(&mut self) -> &mut Vec<IValue> {
        
        todo!();
        /*
            return elements_;
        */
    }
    
    pub fn elements(&mut self) -> &mut Vec<IValue> {
        
        todo!();
        /*
            return move(elements_);
        */
    }
    
    pub fn ty(&self) -> Arc<TupleType> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn hash(t: &Tuple) -> usize {
        
        todo!();
        /*
            return get_hash(t.elements());
        */
    }
    
    pub fn new(
        elements: Vec<IValue>,
        ty:       Arc<TupleType>) -> Self {
        let ty: Arc<TupleType> = ty.unwrap_or(nullptr);
        todo!();
        /*
        : elements(move(elements)),
        : ty(move(type)),

        
        */
    }
}

#[derive(Default)]
pub struct FutureError {
    base:      Exception,
    error_msg: String,
}

impl FutureError {
    
    pub fn new(error_msg: String) -> Self {
    
        todo!();
        /*
        : error_msg(move(error_msg_)),

        
        */
    }
    
    pub fn what(&self) -> *const u8 {
        
        todo!();
        /*
            return error_msg.c_str();
        */
    }
}

pub struct Future {

    base:           IntrusivePtrTarget,

    mutex:          RefCell<Mutex>,

    /**
      | is this future complete
      |
      */
    completed:      AtomicBool, // default = {false}

    finished_cv:    ConditionVariable,

    /**
      | when finished the value
      |
      */
    value:          IValue,

    ty:             TypePtr,
    callbacks:      Vec<fn(_0: &mut Future) -> ()>,
    eptr:           ExceptionPtr,

    /**
      | An upcast pointer to a virtual class
      | which allows us to manipulate events,
      | streams, ... in a generic way, without
      | an explicit dependency on CUDA.
      |
      */
    impl_:          VirtualGuardImpl,


    /**
      | The device that was current when markCompleted
      | was called, which we'll restore when
      | invoking callbacks. It's optional
      | because we'll only store it if the future
      | completes successfully.
      |
      */
    current_device: Option<Device>,


    /**
      | The events that correspond to the completion
      | of the async I/O kernels. They are recorded
      | on the appropriate streams when the
      | future is marked completed and can then
      | be queried/waited/blocked on. There
      | is one event for each distinct device
      | on which the value's tensors reside.
      |
      */
    events:         Vec<Event>,


    /**
      | A cached version of the data ptrs extracted
      | from the value when the future is first
      | marked completed.
      |
      */
    data_ptrs:      Vec<ReferenceWrapper<DataPtr>>,


    /**
      | The bounding set of devices that this
      | future, and any of its children, is allowed
      | to use. This is a superset of the set of
      | devices used by the events above. We
      | need this to know what streams (for which
      | devices) to set as current when invoking
      | a callback, thus allowing the callback
      | to use devices that the parent future
      | didn't use. This field is set to the value
      | provided in the constructor and will
      | be "inherited" by all child futures.
      |
      */
    devices:        Vec<Device>,
}

impl Future {

    /**
      | Keep this private in order to force users to
      | go through make_intrusive and thus prevent
      | creating a Future that's not held by an
      | intrusive_ptr.
      |
      */
    pub fn new(
        ty:      TypePtr,
        devices: Vec<Device>) -> Self {
    
        todo!();
        /*


            : type_(move(type)),
            impl_(getTypeOfDevices(devices)),
            devices_(sortAndDeduplicateDevices(impl_, move(devices)))
        */
    }

    /**
      | Wait on the future until it completes.
      |
      */
    pub fn wait(&mut self)  {
        
        todo!();
        /*
            unique_lock<mutex> lock(mutex_);
        finished_cv_.wait(lock, [&]() -> bool { return completed_; });
        synchronizeWithCurrentStreams();
        */
    }

    /**
      | Wait on the future until it completes
      | and throw an exception if an error exists.
      |
      */
    pub fn wait_and_throw(&mut self)  {
        
        todo!();
        /*
            wait();

        if (eptr_) {
          rethrow_exception(eptr_);
        }
        */
    }

    /**
      | Explicitly mark the future as completed
      | with the output value. Optionally,
      | the storage pointers for all tensors
      | in IValue can be passed as well.
      | 
      | These
      | 
      | DataPtrs are used to synchronize CUDA
      | streams.
      | 
      | If data_ptrs isn't given we will attempt
      | to extract it from the value, if we need
      | to (this happens if a non-empty set of
      | devices was given to the constructor).
      | 
      | Thus one only needs to provide data_ptrs
      | when 1) DataPtrs cannot be extracted
      | through IValue's getSubValues() or
      | through pickling in case of Python object;
      | or when 2) customized DataPtrs extraction
      | is more efficient.
      |
      */
    pub fn mark_completed(&mut self, 
        value:     IValue,
        data_ptrs: Option<Vec<ReferenceWrapper<DataPtr>>>)  {
        let data_ptrs: Option<Vec<ReferenceWrapper<DataPtr>>> = data_ptrs.unwrap_or(nullopt);

        todo!();
        /*
            // Start by performing all steps that can throw, before setting any field.
        // Do this before even acquiring the mutex, because extractDataPtrs might
        // acquire the GIL, which could lead to a lock inversion with our mutex.
        // See https://github.com/pytorch/pytorch/issues/58239.
        vector<reference_wrapper<const DataPtr>> actualDataPtrs;
        vector<Device> usedDevices;
        try {
          // FIXME We should always extract DataPtrs, in order to catch the case of
          // users using CUDA values but forgetting to set devices, which currently
          // leads to a silent synchronization/correctness issue. However, as this
          // might worsen perf in CPU-only cases, we should only do so after careful
          // benchmarks.
          if (impl_.type() != kCPU) {
            actualDataPtrs =
                data_ptrs.has_value() ? move(*data_ptrs) : extractDataPtrs(value);
            usedDevices = getDevicesOfDataPtrs(impl_, actualDataPtrs);
            ensureIsSubsetOfDevices(usedDevices, devices_);
          }
        } catch (const exception&) {
          setError(current_exception());
          return;
        }

        unique_lock<mutex> lock(mutex_);
        TORCH_CHECK(
            !completed(),
            "Attempting to mark a completed Future as complete again. Note that "
            "a Future can only be marked completed once.");

        // Only set value_ and completed_ flag once all checks and preparation steps
        // have returned successfully to allow for proper error propagation.
        value_ = move(value);
        completed_ = true;

        currentDevice_ = impl_.getDevice();
        dataPtrs_ = move(actualDataPtrs);
        for (const Device& device : usedDevices) {
          Event event(impl_.type());
          event.record(impl_.getStream(device));
          events_.push_back(move(event));
        }

        vector<function<void(Future&)>> cbs;
        cbs.swap(callbacks_);
        lock.unlock();

        finished_cv_.notify_all();
        for (auto& callback : cbs) {
          invokeCallback(move(callback));
        }
        */
    }
    
    pub fn mark_completed(&mut self)  {
        
        todo!();
        /*
            markCompleted(IValue{});
        */
    }
    
    pub fn set_error(&mut self, eptr: ExceptionPtr)  {
        
        todo!();
        /*
            unique_lock<mutex> lock(mutex_);
        setErrorInternal(move(eptr), lock);
        */
    }
    
    pub fn set_error_if_needed(&mut self, eptr: ExceptionPtr)  {
        
        todo!();
        /*
            unique_lock<mutex> lock(mutex_);
        if (completed_) {
          // This should be rare and shouldn't cause log spew. Its important to
          // log errors and thats why we have this log here.
          string msg = str(
              "Skipping setting following error on the Future since "
              "it is already marked completed (this is not necessarily "
              "an error):\n",
              tryRetrieveErrorMessageInternal(eptr));
          if (eptr_) {
            msg += str(
                ", \nOriginal exception:\n",
                tryRetrieveErrorMessageInternal(eptr_));
          }
          LOG(INFO) << msg;
          return;
        } else {
          setErrorInternal(move(eptr), lock);
        }
        */
    }

    /**
      | Get the result of the current future.
      |
      */
    pub fn value(&mut self) -> IValue {
        
        todo!();
        /*
            unique_lock<mutex> lock(mutex_);
        AT_ASSERT(completed());
        if (eptr_) {
          rethrow_exception(eptr_);
        }
        return value_;
        */
    }

    /**
      | This accessor should only be used if
      | we know that the future is completed()
      | with no error.
      |
      */
    pub fn const_value(&self) -> &IValue {
        
        todo!();
        /*
            unique_lock<mutex> lock(mutex_);
        AT_ASSERT(completed());
        AT_ASSERT(!eptr_);
        return value_;
        */
    }

    /**
      | This accessor should only be used if
      | we know that the future is completed()
      | with no error.
      |
      */
    pub fn data_ptrs(&self) -> &Vec<ReferenceWrapper<DataPtr>> {
        
        todo!();
        /*
            unique_lock<mutex> lock(mutex_);
        AT_ASSERT(completed());
        AT_ASSERT(!eptr_);
        return dataPtrs_;
        */
    }

    /**
      | Add a callback to the future.
      | 
      | The callbacks will be executed once
      | the future completes.
      | 
      | If the future has already completed,
      | this function will execute the callback
      | immediately.
      |
      */
    pub fn add_callback<T>(&mut self, callback: T)  {
    
        todo!();
        /*
            #if __cpp_lib_is_invocable >= 201703
        static_assert(
            is_invocable_r<void, T, Future&>::value,
            "The callback must have signature void(Future&)");
    #endif
        unique_lock<mutex> lock(mutex_);
        if (completed()) {
          lock.unlock();
          invokeCallback(move(callback));
          return;
        }
        callbacks_.emplace_back(move(callback));
        */
    }

    /**
      | Add a callback to the future, and return
      | another Future to hold the return value
      | of the callback. This is necessary when
      | the callback provider needs to know
      | for sure when the callback has finished.
      |
      */
    pub fn then<T>(&mut self, 
        callback: T,
        ty:       TypePtr) -> IntrusivePtr<Future> {
    
        todo!();
        /*
            using IValueWithDataPtrs = 
            tuple<IValue, vector<reference_wrapper<const DataPtr>>>;
    #if __cpp_lib_is_invocable >= 201703
        static_assert(
            disjunction<
                is_invocable_r<IValue, T, Future&>,
                is_invocable_r<IValueWithDataPtrs, T, Future&>>::value,
            "The callback must have signature IValue(Future&) or "
            "tuple<IValue, vector<reference_wrapper<const DataPtr>>>(Future&)");
    #endif
        auto childFut = createInstance(move(type));
        addCallback([childFut,
                     cb = move(callback)](Future& parentFut) mutable {
          try {
            if_constexpr<is_convertible<
                typename result_of<T && (Future&)>::type,
                IValueWithDataPtrs>::value>(
                [&](auto identity) {
                  IValue value;
                  vector<reference_wrapper<const DataPtr>> dataPtrs;
                  tie(value, dataPtrs) = identity(cb)(parentFut);
                  childFut->markCompleted(move(value), move(dataPtrs));
                },
                [&](auto identity) {
                  childFut->markCompleted(identity(cb)(parentFut));
                });
          } catch (exception&) {
            childFut->setError(current_exception());
          }
        });
        return childFut;
        */
    }
    
    pub fn then_async<T>(&mut self, 
        callback: T,
        ty:       TypePtr) -> IntrusivePtr<Future> {
    
        todo!();
        /*
            #if __cpp_lib_is_invocable >= 201703
        static_assert(
            is_invocable_r<intrusive_ptr<Future>, T, Future&>::value,
            "The callback must have signature intrusive_ptr<Future>(Future&)");
    #endif
        auto childFut = createInstance(move(type));
        addCallback(
            [childFut, cb = move(callback)](Future& parentFut) mutable {
              intrusive_ptr<Future> intermediateFut;
              try {
                intermediateFut = cb(parentFut);
              } catch (exception&) {
                childFut->setError(current_exception());
                return;
              }
              intermediateFut->addCallback(
                  [childFut = move(childFut)](Future& intermediateFut) {
                    if (intermediateFut.hasError()) {
                      childFut->setError(intermediateFut.exception_ptr());
                    } else {
                      childFut->markCompleted(
                          intermediateFut.value(), intermediateFut.dataPtrs());
                    }
                  });
            });
        return childFut;
        */
    }

    /// Tries to retrieve the error message from
    /// exception_ptr.
    ///
    pub fn try_retrieve_error_message(&self) -> String {
        
        todo!();
        /*
            TORCH_CHECK(hasError(), "No error present on the future.");
        unique_lock<mutex> lock(mutex_);
        return tryRetrieveErrorMessageInternal(eptr_);
        */
    }

    /// Check if the current future has completed
    ///
    pub fn completed(&self) -> bool {
        
        todo!();
        /*
            return completed_;
        */
    }
    
    pub fn has_value(&self) -> bool {
        
        todo!();
        /*
            unique_lock<mutex> lock(mutex_);
        return completed_ && !eptr_;
        */
    }
    
    pub fn has_error(&self) -> bool {
        
        todo!();
        /*
            unique_lock<mutex> lock(mutex_);
        return eptr_ ? true : false;
        */
    }
    
    pub fn exception_ptr(&self) -> ExceptionPtr {
        
        todo!();
        /*
            unique_lock<mutex> lock(mutex_);
        return eptr_;
        */
    }
    
    pub fn element_type(&self) -> TypePtr {
        
        todo!();
        /*
            return type_;
        */
    }
    
    pub fn devices(&self) -> &Vec<Device> {
        
        todo!();
        /*
            return devices_;
        */
    }

    /**
      | This method should be used when one intends
      | to manually create a child future, for
      | example when implementing a customized
      | version of then().
      |
      */
    pub fn create_instance(&mut self, ty: TypePtr) -> IntrusivePtr<Future> {
        
        todo!();
        /*
            return make_intrusive<Future>(move(type), devices_);
        */
    }

    /**
      | This method should always be used when
      | invoking a callback (regardless of how/when
      | that happens) as it will ensure that the
      | proper "environment" is set up before running
      | the callback, as in, it will set up the CUDA
      | streams, synchronize them with the value, and
      | so on (if needed).
      */
    pub fn invoke_callback<T>(&mut self, callback: T)  {
    
        todo!();
        /*
            #if __cpp_lib_is_invocable >= 201703
        static_assert(
            is_invocable_r<void, T, Future&>::value,
            "The callback must have signature void(Future&)");
    #endif

        OptionalDeviceGuard deviceGuard(currentDevice_);

        vector<Stream> streams;
        for (const Device& device : devices_) {
          streams.push_back(impl_.getStreamFromGlobalPool(device));
        }
        MultiStreamGuard streamGuard(streams);
        synchronizeWithCurrentStreams();

        callback(*this);
        */
    }

    /**
      | This method should be called before this
      | future's value is used, as it ensures that
      | the CUDA streams that are "current" at the
      | callsite properly synchronize with the value.
      */
    pub fn synchronize_with_current_streams(&mut self)  {
        
        todo!();
        /*
            for (Event& event : events_) {
          event.block(impl_.getStream(event.device()));
        }

        for (const DataPtr& data_ptr : dataPtrs_) {
          if (!data_ptr.device().is_cpu()) {
            impl_.recordDataPtrOnStream(
                data_ptr, impl_.getStream(data_ptr.device()));
          }
        }
        */
    }
    
    pub fn set_error_internal(&mut self, 
        eptr: ExceptionPtr,
        lock: &mut UniqueLock<Mutex>)  {
        
        todo!();
        /*
            TORCH_CHECK(
            !eptr_,
            "Error already set on this Future: ",
            tryRetrieveErrorMessageInternal(eptr_),
            ", trying to set error: ",
            tryRetrieveErrorMessageInternal(eptr));
        TORCH_INTERNAL_ASSERT(!completed(), "Future is already marked completed");
        completed_ = true;
        eptr_ = move(eptr);

        vector<function<void(Future&)>> cbs;
        cbs.swap(callbacks_);
        lock.unlock();

        finished_cv_.notify_all();
        for (auto& callback : cbs) {
          invokeCallback(move(callback));
        }
        */
    }

    /**
      | Tries to retrieve the error message
      | from exception_ptr.
      |
      */
    pub fn try_retrieve_error_message_internal(&self, eptr: ExceptionPtr) -> String {
        
        todo!();
        /*
            try {
          rethrow_exception(eptr);
        } catch (const exception& e) {
          return e.what();
        } catch (...) {
          return "Unknown Exception Type";
        }
        */
    }

    /// Defined in ivalue.cpp.
    ///
    pub fn extract_data_ptrs(value: &IValue) -> Vec<ReferenceWrapper<DataPtr>> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_devices_of_data_ptrs(
        impl_:     &VirtualGuardImpl,
        data_ptrs: &Vec<ReferenceWrapper<DataPtr>>) -> Vec<Device> {
        
        todo!();
        /*
            DeviceIndex deviceCount = impl.deviceCount();
        vector<bool> isDeviceUsed(deviceCount, false);
        for (const DataPtr& data_ptr : data_ptrs) {
          if (!data_ptr.device().is_cpu()) {
            TORCH_CHECK_VALUE(
                data_ptr.device().type() == impl.type(),
                "Expected all data ptrs to be on a device of type ",
                impl.type(),
                ", got one on device ",
                data_ptr.device());
            isDeviceUsed[data_ptr.device().index()] = true;
          }
        }
        vector<Device> devices;
        for (DeviceIndex idx = 0; idx < deviceCount; idx++) {
          if (isDeviceUsed[idx]) {
            devices.emplace_back(impl.type(), idx);
          }
        }
        return devices;
        */
    }
    
    pub fn format_set_of_devices(devices: &Vec<Device>) -> String {
        
        todo!();
        /*
            if (devices.empty()) {
          return "(none)";
        }
        ostringstream oss;
        oss << devices[0];
        for (usize idx = 1; idx < devices.size(); idx++) {
          if (idx == devices.size() - 1) {
            oss << " and ";
          } else {
            oss << ", ";
          }
          oss << devices[idx];
        }
        return oss.str();
        */
    }
    
    pub fn get_type_of_devices(devices: &Vec<Device>) -> DeviceType {
        
        todo!();
        /*
            if (devices.empty()) {
          return kCPU;
        }
        DeviceType deviceType = devices[0].type();
        for (usize idx = 1; idx < devices.size(); idx++) {
          TORCH_CHECK_VALUE(
              devices[idx].type() == deviceType,
              "Expected all devices to be of the same type, but got a mismatch between ",
              devices[0],
              " and ",
              devices[idx]);
        }
        return deviceType;
        */
    }

    /**
      | We need devices to be sorted in order
      | to use ensureIsSubsetOfDevices.
      |
      */
    pub fn sort_and_deduplicate_devices(
        impl_:   &VirtualGuardImpl,
        devices: Vec<Device>) -> Vec<Device> {
        
        todo!();
        /*
            sort(
          devices.begin(), devices.end(),
          [](const Device& a, const Device& b) { return a.index() < b.index(); });
        // Deduplicate by compacting.
        usize targetIdx = 0;
        for (usize sourceIdx = 0; sourceIdx < devices.size(); sourceIdx++) {
          TORCH_CHECK_VALUE(
              devices[sourceIdx].has_index(),
              "Expected devices to have indices, got ", devices[sourceIdx]);
          if (targetIdx > 0 && devices[targetIdx - 1].index() == devices[sourceIdx].index()) {
            // It's a duplicate, skip it.
            continue;
          }
          if (sourceIdx != targetIdx) {
            devices[targetIdx] = devices[sourceIdx];
          }
          targetIdx++;
        }
        // If there were duplicates there's now a gap at the end: trim it. Resizing
        // requires the item type to be default-constructible (which Device is
        // not) because in principle it could be required to create new items. Since
        // we know we'll shrink the vector, we provide a custom dummy value instead.
        devices.resize(targetIdx, Device(kCPU));
        return devices;
        */
    }
    
    pub fn ensure_is_subset_of_devices(
        subset:   &Vec<Device>,
        superset: &Vec<Device>)  {
        
        todo!();
        /*
            // We assume the devices in both vectors have the same consistent type, and
        // their indices are unique and sorted.
        vector<Device> excessDevices;
        set_difference(
            subset.begin(),
            subset.end(),
            superset.begin(),
            superset.end(),
            back_inserter(excessDevices),
            [](const Device& a, const Device& b) { return a.index() < b.index(); });
        TORCH_CHECK_VALUE(
            excessDevices.empty(),
            "The result contained tensors residing on device(s) ",
            formatSetOfDevices(excessDevices),
            " which are not among the expected device(s) ",
            formatSetOfDevices(superset));
        */
    }
}

/**
  | Input is a list of Futures with the same target
  | type.
  |
  | Output is a Future to the List of completed
  | Futures.
  |
  */
pub fn collect_all(srcs: List<IntrusivePtr<Future>>) -> IntrusivePtr<Future> {
    
    todo!();
        /*
        
        */
}

/**
  | Input is a List of Futures with the same target
  | type.
  |
  | Output is a Future that will be updated with
  | a seen value.
  |
  */
pub fn collect_any(srcs: List<IntrusivePtr<Future>>) -> IntrusivePtr<Future> {
    
    todo!();
        /*
        
        */
}

/// User-defined object.
pub struct Object {
    base: IntrusivePtrTarget,
    ty:    StrongTypePtr,
    slots: Vec<IValue>,
}

impl Object {
    
    pub fn new(
        ty:        StrongTypePtr,
        num_slots: usize) -> Self {
    
        todo!();
        /*
        : ty(move(type)),

            slots_.resize(numSlots);
        */
    }
    
    pub fn create(
        ty:        StrongTypePtr,
        num_slots: usize) -> IntrusivePtr<Object> {
        
        todo!();
        /*
            return make_intrusive<Object>(move(type), numSlots);
        */
    }

    /**
      | Slot API.
      | 
      | Attributes are stored as a simple vector
      | so that lookups are fast at runtime.
      | A "slot" is just an index into that vector,
      | which can be computed statically if
      | you have access to the class type. Use
      | this API if you are writing compiler
      | stuff.
      |
      */
    pub fn set_slot(&mut self, 
        slot: usize,
        v:    IValue)  {
        
        todo!();
        /*
            if (slot >= slots_.size()) {
          // for module types, it is possible that the members of the class have
          // expanded after the object was created. In this case, we expand
          // the slots to the right size
          resizeObject(slot);
        }
        slots_[slot] = move(v);
        */
    }
    
    pub fn get_slot(&self, slot: usize) -> &IValue {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(slot < slots_.size());
        // NOTE: This lookup is fairly hot, so we use unchecked access to the
        // vector.  Errors should still be detectable with ASan.
        return slots_[slot];
        */
    }
    
    pub fn unsafe_remove_slot(&mut self, slot: usize)  {
        
        todo!();
        /*
            TORCH_CHECK(slot < slots_.size());
        slots_.erase(slots_.begin() + slot);
        */
    }

    /**
      | Attribute API.
      | 
      | Wrappers around the slot stuff so that
      | users can access attributes directly.
      | Use this API if you are a user.
      | 
      | -----------
      | @note
      | 
      | Unlike in Python, TorchScript must
      | make a distinction between attributes
      | (which are IValues) and methods (which
      | are Methods). If you want a method, use
      | `obj.type()->getMethod()`
      |
      */
    pub fn get_attr(&self, name: &String) -> IValue {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_attr(&mut self, 
        name: &String,
        v:    IValue)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Remove attribute by name, caller is
      | responsible for the safety of this operation
      |
      | We didn't remove the attribute in the type
      | because the type might be shared by multiple
      | objects.
      |
      | Therefore after removing attribute, the
      | object is in an inconsistent state where it
      | has more attribute types in its Type than the
      | attribute slots it has, user needs to make
      | sure the object has consistent by removing
      | the attribute in type as well
      */
    pub fn unsafe_remove_attr(&mut self, name: &String)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn name(&self) -> String {
        
        todo!();
        /*
        
        */
    }
    
    pub fn slots(&self) -> &Vec<IValue> {
        
        todo!();
        /*
            return slots_;
        */
    }
    
    pub fn ty(&self) -> Arc<ClassType> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn compilation_unit(&mut self) -> Arc<TorchJitCompilationUnit> {
        
        todo!();
        /*
            return type_.cu_;
        */
    }
    
    pub fn copy_(&self) -> IntrusivePtr<Object> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn deepcopy(&self) -> IntrusivePtr<Object> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn deepcopy(&self, memo: &mut HashAliasedIValueMap) -> IntrusivePtr<Object> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn resize_object(&mut self, slot: usize)  {
        
        todo!();
        /*
        
        */
    }
}

/**
  | virtual ivalue PyObjectHolder that hold
  | a py::object, we make this virtual because the
  | py::object and refcounting logic should happen
  | in libtorch_python see concrete implementation
  | in python_ivalue.h
  |
  */
pub trait PyObjectHolderInterface:
 IntrusivePtrTarget
+ GetPyObject
+ TryToInferType
+ ToIvalue
+ ToStr
+ ExtractTensors {}

pub trait GetPyObject {
    
    fn get_py_object(&mut self) -> *mut PyObject;
}

pub trait TryToInferType {

    fn try_to_infer_type(&mut self) -> InferredType;
}

pub trait ToIvalue {
    
    fn to_ivalue(&mut self, 
        ty: &TypePtr,
        N:  Option<i32>) -> IValue;
}

pub trait ToStr {

    fn to_str(&mut self) -> String;
}

pub trait ExtractTensors {
    
    fn extract_tensors(&mut self) -> Vec<Tensor>;
}

pub struct EnumHolder {
    base:  IntrusivePtrTarget,
    ty:    Arc<EnumType>,
    name:  String,
    value: IValue,
}

impl EnumHolder {
    
    pub fn new(
        ty:    Arc<EnumType>,
        name:  String,
        value: IValue) -> Self {
    
        todo!();
        /*
        : ty(move(type)),
        : name(move(name)),
        : value(move(value)),

        
        */
    }
    
    pub fn is(&mut self, rhs: &EnumHolder) -> bool {
        
        todo!();
        /*
            return *this == rhs;
        */
    }
    
    pub fn qualified_class_name(&self) -> String {
        
        todo!();
        /*
        
        */
    }
    
    pub fn unqualified_class_name(&self) -> String {
        
        todo!();
        /*
        
        */
    }
    
    pub fn name(&self) -> &String {
        
        todo!();
        /*
            return name_;
        */
    }
    
    pub fn value(&self) -> &IValue {
        
        todo!();
        /*
            return value_;
        */
    }
    
    pub fn ty(&self) -> Arc<EnumType> {
        
        todo!();
        /*
            return type_;
        */
    }
}

pub struct GuardedUnsignedLongUniqueDummy {

}

impl GuardedUnsignedLongUniqueDummy {
    
    pub fn new(_0: i64) -> Self {
    
        todo!();
        /*


            }{
        */
    }
}

lazy_static!{
    /*
    using _guarded_unsigned_long = conditional_t<
        is_same<unsigned long, u32>::value ||
            is_same<unsigned long, u64>::value,
        _guarded_unsigned_long_unique_dummy,
        unsigned long>;
    */
}

impl IValue {
    
    #[inline] pub fn to_object_ref(&self) -> &Object {
        
        todo!();
        /*
            AT_ASSERT(isObject(), "Expected Object but got ", tagKind());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(payload.u.as_intrusive_ptr != UndefinedTensorImpl::singleton(), "Attempted to create null reference");
      return *static_cast<const Object*>(payload.u.as_intrusive_ptr);
        */
    }
}

/**
  | note: when adding a DEFINE_TO case here you
  | should also add a toX method to IValue. These
  | named methods are much more discoverable than
  | the to templated function.
  */
#[macro_export] macro_rules! define_to {
    ($T:ty, $method_name:ty) => {
        /*
        
          template <>                                              
          inline T IValue::to<T>()&& {                             
            return static_cast<T>(move(*this).method_name()); 
          }                                                        
          template <>                                              
          inline &T IValue::to<T>() const& { 
            return this->method_name();            
          }
        */
    }
}

define_to!{Tensor                        , toTensor}
define_to!{Storage                       , toStorage}
define_to!{Stream                        , toStream}
define_to!{f32                           , toDouble}
define_to!{f64                           , toDouble}
define_to!{Complex<f64>                  , toComplexDouble}
define_to!{u8                            , toInt}
define_to!{i8                            , toInt}
define_to!{u16                           , toInt}
define_to!{i16                           , toInt}
define_to!{i32                           , toInt}
define_to!{u32                           , toInt}
define_to!{u64                           , toInt}
define_to!{_guarded_unsigned_long        , toInt}
define_to!{i64                           , toInt}
define_to!{bool                          , toBool}
define_to!{intrusive_ptr<Blob>           , toBlob}
define_to!{intrusive_ptr<ConstantString> , toString}
define_to!{intrusive_ptr<Object>         , toObject}
define_to!{Scalar                        , toScalar}
define_to!{List<i64>                     , toIntList}
define_to!{List<f64>                     , toDoubleList}
define_to!{List<complex<f64>>            , toComplexDoubleList}
define_to!{List<bool>                    , toBoolList}
define_to!{List<Tensor>                  , toTensorList}
define_to!{GenericList                   , toList}
define_to!{GenericDict                   , toGenericDict}
define_to!{intrusive_ptr<Tuple>          , toTuple}
define_to!{string                        , toStringRef}
define_to!{string_view                   , toStringView}
define_to!{intrusive_ptr<Future>         , toFuture}
define_to!{intrusive_ptr<RRefInterface>  , toRRef}
define_to!{intrusive_ptr<Quantizer>      , toQuantizer}
define_to!{IValue                        , toIValue}
define_to!{Device                        , toDevice}
define_to!{ScalarType                    , toScalarType}
define_to!{Layout                        , toLayout}
define_to!{MemoryFormat                  , toMemoryFormat}
define_to!{QScheme                       , toQScheme}
define_to!{Dimname                       , toDimname}
define_to!{Generator                     , toGenerator}

struct FakeType<T> {}

/**
  | generic_to<T> converts an IValue from a generic list or generic dict
  | to a concrete list/dict type likelike List<T>, Dict<...> or optional<T>.
  | Note that in the case of lists, this only works for IValue-based lists,
  | i.e. not for i64, double, ...
  | generic_to<T> is an implementation detail of IValue::to<T> and not
  | supposed to be called directly.
  | The _fake_type<T> parameter allows us to overload
  | based on the return type.
  |
  | TODO this is deprecated but we don't throw a warning because a lot of ops in
  | native_functions.yaml still return vector.
  | C10_DEPRECATED_MESSAGE("IValues based on vector<T> are potentially slow
  | and deprecated. Please use TorchList<T> instead.")
  |
  */
pub fn generic_to_with_fake_type_vec_elem<Elem>(
    ivalue: IValue,
    _1:     FakeType<Vec<Elem>>) -> Vec<Elem> {

    todo!();
        /*
            // We need to do a deep copy of the vector because there might be other
      // references to this same IValue that also use the list. We can't just
      // move the elements out.
      auto list = move(ivalue).to<List<Elem>>();
      vector<Elem> result;
      result.reserve(list.size());
      for (Elem v : list) {
        result.push_back(move(v));
      }
      return result;
        */
}

impl IValue {
    
    pub fn to_custom_class<T>(&mut self) -> IntrusivePtr<T> {
    
        todo!();
        /*
            static_assert(
          is_base_of<TorchCustomClassHolder, T>::value == true,
          "toCustomClass requires that template parameter T must inherit "
          "from TorchCustomClassHolder");
      auto obj = toObject();
      TORCH_CHECK(
          obj->slots().size() == 1,
          "Tried to cast IValue to custom class but it did "
          "not contain a custom class!");
      const Type* expected_type = getCustomClassType<intrusive_ptr<T>>().get();
      checkCustomClassType(expected_type, type().get());
      auto userObj =
          static_intrusive_pointer_cast<T>(obj->getSlot(0).toCapsule());
      return userObj;
        */
    }
    
    pub fn to_custom_class<T>(&mut self) -> IntrusivePtr<T> {
    
        todo!();
        /*
            static_assert(
          is_base_of<TorchCustomClassHolder, T>::value == true,
          "toCustomClass requires that template parameter T must inherit "
          "from TorchCustomClassHolder");
      auto obj = toObject();
      TORCH_CHECK(
          obj->slots().size() == 1,
          "Tried to cast IValue to custom class but it did "
          "not contain a custom class!");
      const Type* expected_type = getCustomClassType<intrusive_ptr<T>>().get();
      checkCustomClassType(expected_type, type().get());
      auto userObj =
          static_intrusive_pointer_cast<T>(obj->getSlot(0).toCapsule());
      return userObj;
        */
    }
}

pub fn generic_to_with_fake_type_t<T>(
        ivalue: IValue,
        _1:     FakeType<T>) -> T {

    todo!();
        /*
            using ElemType = typename remove_pointer<T>::type::element_type;
      return move(ivalue).toCustomClass<ElemType>();
        */
}

pub fn generic_to_with_fake_type_tagged_capsule_t<T>(
        ivalue: IValue,
        _1:     FakeType<TaggedCapsule<T>>) -> TaggedCapsule<T> {

    todo!();
        /*
            return tagged_capsule<T>{move(ivalue)};
        */
}

pub fn generic_to_with_fake_type_list_elem<Elem>(
        ivalue: IValue,
        _1:     FakeType<List<Elem>>) -> List<Elem> {

    todo!();
        /*
            return toTypedList<Elem>(move(ivalue).toList());
        */
}

pub fn create_vector_from_list<T>(impl_: &List<T>) -> Vec<T> {

    todo!();
        /*
            vector<T> result;
      result.reserve(impl.size());
      for (usize i = 0, N = impl.size(); i < N; ++i) {
        result.push_back(impl[i]);
      }
      return result;
        */
}

pub fn generic_to_with_fake_type_optional_array_t<T>(
    ivalue: IValue,
    _1:     FakeType<OptionalArray<T>>) -> OptionalArray<T> {

    todo!();
        /*
            if (ivalue.isNone()) {
        return {};
      }
      return createVectorFromList<T>(
        move(ivalue).to<List<T>>()
      );
        */
}

lazy_static!{
    /*
    template <typename Elem, usize... I>
    array<Elem, sizeof...(I)> generic_to_array(
        IValue ivalue,
        _fake_type<array<Elem, sizeof...(I)>>,
        index_sequence<I...>) {
      // We need to do a deep copy of the array because there might be other
      // references to this same IValue that also use the list. We can't just
      // move the elements out.
      auto list = move(ivalue).to<List<Elem>>();
      TORCH_CHECK(
          list.size() == sizeof...(I),
          "Tried to convert a List with ",
          list.size(),
          " elements to a fixed-size array of size ",
          sizeof...(I));
      return {list[I]...};
    }
    */
}

pub fn generic_to_with_fake_type_arrayn_elem<Elem, const N: usize>(
        ivalue: IValue,
        ft:     FakeType<Array<Elem,N>>) -> Array<Elem,N> {

    todo!();
        /*
            return generic_to_array(ivalue, ft, make_index_sequence<N>());
        */
}

pub fn generic_to_with_fake_type_dict<Key, Value>(
        ivalue: IValue,
        _1:     FakeType<Dict<Key,Value>>) -> Dict<Key,Value> {

    todo!();
        /*
            return toTypedDict<Key, Value>(move(ivalue).toGenericDict());
        */
}

#[deprecated = "IValues based on unordered_map are slow and deprecated. Please use Dict<K, V> instead."]
pub fn generic_to_with_fake_type_hashmap<K, V>(
        ivalue: IValue,
        _1:     FakeType<HashMap<K,V>>) -> HashMap<K,V> {

    todo!();
        /*
            unordered_map<K, V> specialized_dict;

      for (const auto& item : move(ivalue).toGenericDict()) {
        specialized_dict[item.key().to<K>()] = item.value().to<V>();
      }

      return specialized_dict;
        */
}

pub fn generic_to_with_fake_type_option_t<T>(
        ivalue: IValue,
        _1:     FakeType<Option<T>>) -> Option<T> {

    todo!();
        /*
            if (ivalue.isNone()) {
        return nullopt;
      }
      return move(ivalue).to<T>();
        */
}

pub fn generic_to_tuple_impl<Tuple, const INDEX: usize>(
        t:  &Vec<IValue>,
        _1: IndexSequence<INDEX>) -> Tuple {

    todo!();
        /*
            return make_tuple(
          t[INDEX].to<typename tuple_element<INDEX, Tuple>::type>()...);
        */
}

lazy_static!{
    /*
    template <
        typename... Args,
        typename Indices = make_index_sequence<sizeof...(Args)>,
        enable_if_t<
            !disjunction<
                is_lvalue_reference<Args>...,
                negation<is_constructible<IValue, Args>>...>::value,
            nullptr_t> = nullptr>
    tuple<Args...> generic_to(IValue ivalue, _fake_type<tuple<Args...>>) {
      auto vals = ivalue.toTuple()->elements();
      TORCH_CHECK(vals.size() == sizeof...(Args));
      return generic_to_tuple_impl<tuple<Args...>>(vals, Indices{});
    }
    */
}

impl IValue {
    
    #[inline] pub fn to<T>(&mut self) -> T {
    
        todo!();
        /*
            return generic_to(move(*this), _fake_type<T>{});
        */
    }
    
    #[inline] pub fn to(&mut self) -> Option<StringView> {
        
        todo!();
        /*
            // In the default implementation, the IValue is destroyed with move.
      // But if the unboxed type is optional<string_view> we cannot destroy
      // the IValue.
      return generic_to(*this, _fake_type<optional<string_view>>{});
        */
    }
    
    #[inline] pub fn to<T>(&mut self) -> &T {
    
        todo!();
        /*
            return generic_to(*this, _fake_type<T>{});
        */
    }
    
    #[inline] pub fn to_int_list(&mut self) -> List<i64> {
        
        todo!();
        /*
            AT_ASSERT(isIntList(), "Expected IntList but got ", tagKind());
      return List<i64>(moveToIntrusivePtr<ListImpl>());
        */
    }
    
    #[inline] pub fn to_int_list(&mut self) -> List<i64> {
        
        todo!();
        /*
            AT_ASSERT(isIntList(), "Expected IntList but got ", tagKind());
      return List<i64>(toIntrusivePtr<ListImpl>());
        */
    }
    
    #[inline] pub fn to_int_vector(&mut self) -> Vec<i64> {
        
        todo!();
        /*
            AT_ASSERT(isIntList(), "Expected IntList but got ", tagKind());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          payload.u.as_intrusive_ptr != UndefinedTensorImpl::singleton(),
          "called toIntVector on null intrusive_ptr IValue");
      return createVectorFromList<i64>(
          static_cast<const ListImpl*>(payload.u.as_intrusive_ptr));
        */
    }
    
    #[inline] pub fn to_double_list(&mut self) -> List<f64> {
        
        todo!();
        /*
            AT_ASSERT(isDoubleList(), "Expected DoubleList but got ", tagKind());
      return List<double>(moveToIntrusivePtr<ListImpl>());
        */
    }
    
    #[inline] pub fn to_double_list(&mut self) -> List<f64> {
        
        todo!();
        /*
            AT_ASSERT(isDoubleList(), "Expected DoubleList but got ", tagKind());
      return List<double>(toIntrusivePtr<ListImpl>());
        */
    }
    
    #[inline] pub fn to_double_vector(&self) -> Vec<f64> {
        
        todo!();
        /*
            AT_ASSERT(isDoubleList(), "Expected DoubleList but got ", tagKind());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          payload.u.as_intrusive_ptr != UndefinedTensorImpl::singleton(),
          "called toDoubleVector on null intrusive_ptr IValue");
      return createVectorFromList<double>(
          static_cast<const ListImpl*>(payload.u.as_intrusive_ptr));
        */
    }
    
    #[inline] pub fn to_complex_double_list(&mut self) -> List<Complex<f64>> {
        
        todo!();
        /*
            AT_ASSERT(isComplexDoubleList(), "Expected ComplexDoubleList but got ", tagKind());
      return List<complex<double>>(moveToIntrusivePtr<ListImpl>());
        */
    }
    
    #[inline] pub fn to_complex_double_list(&mut self) -> List<Complex<f64>> {
        
        todo!();
        /*
            AT_ASSERT(isComplexDoubleList(), "Expected ComplexDoubleList but got ", tagKind());
      return List<complex<double>>(toIntrusivePtr<ListImpl>());
        */
    }
    
    #[inline] pub fn to_complex_double_vector(&self) -> Vec<Complex<f64>> {
        
        todo!();
        /*
            AT_ASSERT(isComplexDoubleList(), "Expected ComplexDoubleList but got ", tagKind());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          payload.u.as_intrusive_ptr != UndefinedTensorImpl::singleton(),
          "called toComplexDoubleVector on null intrusive_ptr IValue");
      return createVectorFromList<complex<double>>(
          static_cast<const ListImpl*>(payload.u.as_intrusive_ptr));
        */
    }
    
    #[inline] pub fn to_bool_list(&mut self) -> List<bool> {
        
        todo!();
        /*
            AT_ASSERT(isBoolList(), "Expected BoolList but got ", tagKind());
      return List<bool>(moveToIntrusivePtr<ListImpl>());
        */
    }
    
    #[inline] pub fn to_bool_list(&mut self) -> List<bool> {
        
        todo!();
        /*
            AT_ASSERT(isBoolList(), "Expected BoolList but got ", tagKind());
      return List<bool>(toIntrusivePtr<ListImpl>());
        */
    }
    
    #[inline] pub fn to_tensor_list(&mut self) -> List<Tensor> {
        
        todo!();
        /*
            AT_ASSERT(isTensorList(), "Expected TensorList but got ", tagKind());
      return List<Tensor>(moveToIntrusivePtr<ListImpl>());
        */
    }
    
    #[inline] pub fn to_tensor_list(&mut self) -> List<Tensor> {
        
        todo!();
        /*
            AT_ASSERT(isTensorList(), "Expected TensorList but got ", tagKind());
      return List<Tensor>(toIntrusivePtr<ListImpl>());
        */
    }
    
    #[inline] pub fn to_tensor_vector(&self) -> Vec<Tensor> {
        
        todo!();
        /*
            AT_ASSERT(isTensorList(), "Expected TensorList but got ", tagKind());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          payload.u.as_intrusive_ptr != UndefinedTensorImpl::singleton(),
          "called toTensorVector on null intrusive_ptr IValue");
      return createVectorFromList<Tensor>(
          static_cast<const ListImpl*>(payload.u.as_intrusive_ptr));
        */
    }
    
    #[inline] pub fn to_list(&mut self) -> List<IValue> {
        
        todo!();
        /*
            AT_ASSERT(isList(), "Expected GenericList but got ", tagKind());
      return List<IValue>(moveToIntrusivePtr<ListImpl>());
        */
    }
    
    #[inline] pub fn to_list(&mut self) -> List<IValue> {
        
        todo!();
        /*
            AT_ASSERT(isList(), "Expected GenericList but got ", tagKind());
      return List<IValue>(toIntrusivePtr<ListImpl>());
        */
    }
    
    #[inline] pub fn to_list_ref(&self) -> ArrayRef<IValue> {
        
        todo!();
        /*
            AT_ASSERT(isList(), "Expected GenericList but got ", tagKind());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          payload.u.as_intrusive_ptr != UndefinedTensorImpl::singleton(),
          "called toListRef on null intrusive_ptr IValue");
      return static_cast<const ListImpl*>(payload.u.as_intrusive_ptr)
          ->list;
        */
    }
    
    #[inline] pub fn to_generic_dict(&mut self) -> Dict<IValue,IValue> {
        
        todo!();
        /*
            AT_ASSERT(isGenericDict(), "Expected GenericDict but got ", tagKind());
      return Dict<IValue, IValue>(moveToIntrusivePtr<DictImpl>());
        */
    }
    
    #[inline] pub fn to_generic_dict(&mut self) -> Dict<IValue,IValue> {
        
        todo!();
        /*
            AT_ASSERT(isGenericDict(), "Expected GenericDict but got ", tagKind());
      return Dict<IValue, IValue>(toIntrusivePtr<DictImpl>());
        */
    }
    
    #[inline] pub fn to_tuple(&mut self) -> IntrusivePtr<Tuple> {
        
        todo!();
        /*
            AT_ASSERT(isTuple(), "Expected Tuple but got ", tagKind());
      return moveToIntrusivePtr<Tuple>();
        */
    }
    
    #[inline] pub fn to_tuple(&mut self) -> IntrusivePtr<Tuple> {
        
        todo!();
        /*
            AT_ASSERT(isTuple(), "Expected Tuple but got ", tagKind());
      return toIntrusivePtr<Tuple>();
        */
    }
    
    pub fn new(v: IntrusivePtr<Tuple>) -> Self {
    
        todo!();
        /*
        : tag(Tag::Tuple),
        : is_intrusive_ptr(true),

            payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
        */
    }

    pub fn new(v: IntrusivePtr<ConstantString>) -> Self {

        lazy_static!{
            /*
            template <
                typename... Args,
                enable_if_t<
                    !disjunction<
                        is_lvalue_reference<Args>...,
                        negation<is_constructible<IValue, Args>>...>::value,
                    nullptr_t>>
            inline IValue::IValue(const tuple<Args...>& t)
                : IValue(
                      move(apply(Tuple::create<const Args&...>, t))) {
            }

            template <
                typename... Args,
                enable_if_t<
                    !disjunction<
                        is_lvalue_reference<Args>...,
                        negation<is_constructible<IValue, Args>>...>::value,
                    nullptr_t>>
            inline IValue::IValue(tuple<Args...>&& t)
                : IValue(
                      move(apply(Tuple::create<Args&&...>, move(t)))) {
            }
            */
        }
    
        todo!();
        /*
        : tag(Tag::String),
        : is_intrusive_ptr(true),

            payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
        */
    }
    
    pub fn new(v: String) -> Self {
    
        todo!();
        /*


            : IValue(ConstantString::create(move(v)))
        */
    }
    
    pub fn new(v: GenericList) -> Self {
    
        todo!();

        /*
        : tag(Tag::GenericList), is_intrusive_ptr(true) 
          payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.impl_.release());
        */

        lazy_static!{
            /*
            template <class T, IValue::enable_if_ivalue_constructible<T>>
            inline IValue::IValue(List<T>&& v) : IValue(toList<T>(move(v))) {}
            template <class T, IValue::enable_if_ivalue_constructible<T>>
            inline IValue::IValue(const List<T>& v) : IValue(toList<T>(v)) {}
            */
        }

        lazy_static!{
            /*
            template <class T, IValue::enable_if_ivalue_constructible<T>>
            inline IValue::IValue(ArrayRef<T> v) : IValue(List<T>()) {
              auto list = to<List<T>>();
              list.reserve(v.size());
              for (const auto& e : v) {
                list.push_back(e);
              }
            }
            template <class T, IValue::enable_if_ivalue_constructible<T>>
            inline IValue::IValue(const vector<T>& v) : IValue(List<T>()) {
              auto list = to<List<T>>();
              list.reserve(v.size());
              for (const auto& e : v) {
                list.push_back(e);
              }
            }
            */
        }
    }
    
    pub fn new<const N: usize>(v: [T;N]) -> Self {
    
        todo!();
        /*


            : IValue(List<T>()) 

      auto list = to<List<T>>();
      list.reserve(v.size());
      for (auto& e : v) {
        list.push_back(move(e));
      }
        */
    }
    
    pub fn new(v: GenericDict) -> Self {
    
        todo!();
        /*


            : tag(Tag::GenericDict), is_intrusive_ptr(true) 

      payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.impl_.release());
        */
    }
    
    pub fn new<Key, Value>(v: Dict<Key,Value>) -> Self {
    
        todo!();
        /*


            : IValue(toGenericDict(move(v)))
        */
    }
    
    pub fn new<T, Key, Value>(v: T) -> Self 
        where T: Into<HashMap<Key,Value>> 
    {
        todo!();

        /*
            : IValue(Dict<Key, Value>()) 

              auto dict = to<Dict<Key, Value>>();
              dict.reserve(v.size());
              for (auto& e : v) {
                dict.insert(move(e.first), move(e.second));
              }
        */

        lazy_static!{
            /*
            template <class T, IValue::enable_if_ivalue_constructible<T>>
            inline IValue::IValue(optional<T> v) : IValue() {
              if (v.has_value()) {
                *this = IValue(move(*v));
              }
            }
            */
        }
    }
    
    pub fn new(v: IntrusivePtr<Object>) -> Self {
    
        todo!();
        /*


            : tag(Tag::Object), is_intrusive_ptr(true) 

      payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
        */
    }
    
    pub fn new(v: IntrusivePtr<PyObjectHolder>) -> Self {
    
        todo!();
        /*


            : tag(Tag::PyObject), is_intrusive_ptr(true) 

      payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
        */
    }
    
    pub fn new(v: IntrusivePtr<EnumHolder>) -> Self {
    
        todo!();
        /*


            : tag(Tag::Enum), is_intrusive_ptr(true) 

      payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
        */
    }
    
    #[inline] pub fn make_capsule(&mut self, blob: IntrusivePtr<TorchCustomClassHolder>) -> IValue {
        
        todo!();
        /*
              IValue iv;
              iv.tag = Tag::Capsule;
              iv.is_intrusive_ptr = true;
              iv.payload.u.as_intrusive_ptr = null_to_undefined_tensor(blob.release());
              return iv;
        */

        lazy_static!{
            /*
            template <
                typename T,
                enable_if_t<is_base_of<TorchCustomClassHolder, T>::value, int>>
            IValue::IValue(intrusive_ptr<T> custom_class) {
              TypePtr classType = []() {
                try {
                  return getCustomClassType<intrusive_ptr<T>>();
                } catch (const Error&) {
                  throw Error(
                      "Trying to instantiate a class that isn't a registered custom class: " +
                      string(util::get_fully_qualified_type_name<T>()),
                      "");
                }
              }();
              auto ivalue_obj = Object::create(
                  StrongTypePtr(nullptr, classType), /*num_slots=*/1);
              ivalue_obj->setSlot(0, IValue::make_capsule(move(custom_class)));
              payload.u.as_intrusive_ptr = null_to_undefined_tensor(ivalue_obj.release());
              tag = Tag::Object;
              is_intrusive_ptr = true;
            }
            */
        }
    }
    
    pub fn new(v: IntrusivePtr<Future>) -> Self {
    
        todo!();
        /*
        : tag(Tag::Future),
        : is_intrusive_ptr(true),

            payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
        */
    }
    
    pub fn new(v: IntrusivePtr<RRefInterface>) -> Self {
    
        todo!();
        /*
        : tag(Tag::RRef),
        : is_intrusive_ptr(true),

            payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
        */
    }
    
    pub fn new(v: IntrusivePtr<Quantizer>) -> Self {
    
        todo!();
        /*
        : tag(Tag::Quantizer),
        : is_intrusive_ptr(true),

            payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
        */
    }
    
    pub fn new<T>(c: Complex<T>) -> Self {
    
        todo!();
        /*
        : tag(Tag::ComplexDouble),
        : is_intrusive_ptr(true),

            auto v = make_intrusive<ComplexHolder>(c);
      payload.u.as_intrusive_ptr = v.release();
        */
    }
    
    #[inline] pub fn to_string_ref(&self) -> &String {
        
        todo!();
        /*
            AT_ASSERT(isString(), "Expected String but got ", tagKind());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          payload.u.as_intrusive_ptr != UndefinedTensorImpl::singleton(),
          "called toStringRef on null intrusive_ptr IValue");
      return static_cast<const ConstantString*>(
                 payload.u.as_intrusive_ptr)
          ->string();
        */
    }
    
    #[inline] pub fn to_optional_string_ref(&self) -> Option<ReferenceWrapper<String>> {
        
        todo!();
        /*
            if (isNone()) {
        return nullopt;
      }
      AT_ASSERT(isString(), "Expected optional<string> but got ", tagKind());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          payload.u.as_intrusive_ptr != UndefinedTensorImpl::singleton(),
          "called toOptionalStringRef on null intrusive_ptr IValue");
      return reference_wrapper<const string>(
          static_cast<const ConstantString*>(payload.u.as_intrusive_ptr)
              ->string());
        */
    }
    
    #[inline] pub fn to_string_view(&self) -> StringView {
        
        todo!();
        /*
            AT_ASSERT(isString(), "Expected String but got ", tagKind());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          payload.u.as_intrusive_ptr != UndefinedTensorImpl::singleton(),
          "called toStringView on null intrusive_ptr IValue");
      return static_cast<const ConstantString*>(
            payload.u.as_intrusive_ptr)
        ->string_view();
        */
    }
    
    #[inline] pub fn to_py_object(&self) -> *mut PyObject {
        
        todo!();
        /*
            return toPyObjectHolder()->getPyObject();
        */
    }
    
    
    #[inline] pub fn to_optional<T>(&mut self) -> Option<T> {
    
        todo!();
        /*
            if (this->isNone()) {
        return nullopt;
      }
      return this->to<T>();
        */
    }
    
    
    #[inline] pub fn to_optional<T>(&self) -> Option<T> {
    
        todo!();
        /*
            if (this->isNone()) {
        return nullopt;
      }
      return this->to<T>();
        */
    }
    
    #[inline] pub fn is_custom_class(&self) -> bool {
        
        todo!();
        /*
            return TorchisCustomClass(*this);
        */
    }
    
    #[inline] pub fn is_same_identity(&self, rhs: &IValue) -> bool {
        
        todo!();
        /*
            // We choose to not use memcmp for payload check due to potential random
      // padding characters on union type

      // Semantics:
      // 1. Immutable primitive values of the same type (Int, Double, None, Bool,
      // Str) return value equality
      // 2. If it is a tensor type, we need to take undefined tensor into account
      // 3. Undefined_tensor is None and vice versa should be true
      // 4. If it is a reference type (i.e. is_intrusive_ptr), then is is True when
      // the pointed-to object is the same.
      // 5. False for all other comparisons.
      if (this->isNone() && rhs.isNone()) {
        return true;
      } else if (this->isBool() && rhs.isBool()) {
        // for bool type, do equality check
        return this->toBool() == rhs.toBool();
      } else if (this->isTensor() && rhs.isTensor()) {
        return this->payload.as_tensor.is_same(rhs.payload.as_tensor);
      } else if (this->isTensor() && rhs.isNone()) {
        // special case: undefined tensor and None are the same identity
        return !this->payload.as_tensor.defined();
      } else if (this->isNone() && rhs.isTensor()) {
        // special case: undefined tensor and None are the same identity
        return !rhs.payload.as_tensor.defined();
      } else if (this->isInt() && rhs.isInt()) {
        return this->toInt() == rhs.toInt();
      } else if (this->isDouble() && rhs.isDouble()) {
        return this->toDouble() == rhs.toDouble();
      } else if (this->isString() && rhs.isString()) {
        return this->toStringRef() == rhs.toStringRef();
      } else {
        // for objects holding in IValue, do shallow compare on pointer address to
        // testify the identity
        return this->is_intrusive_ptr && rhs.is_intrusive_ptr &&
            this->payload.u.as_intrusive_ptr == rhs.payload.u.as_intrusive_ptr;
      }
        */
    }
}

impl<T> From<T> for IValue {

    fn from(x: T) -> IValue {

        /*
        pub fn from<T>(
                x:  T,
                _1: TrueType) -> IValue {

            todo!();
                /*
                    return IValue(forward<T>(x));
                */
        }

        pub fn from<T>(
                x:  IntrusivePtr<T>,
                _1: FalseType) -> IValue {

            todo!();
                /*
                    return IValue(move(x));
                */
        }

        pub fn from<T>(
                x:  T,
                _1: FalseType) -> IValue {

            todo!();
                /*
                    static_assert(
                  false_t<T>::value,
                  "You are calling from with a type that it doesn't support, and isn't a potential custom class (ie: is an intrusive_ptr)");
              return IValue();
                */
        }
        */

        todo!();
            /*
            return from_(
              forward<T>(x), typename is_constructible<IValue, T>::type{});
            */
    }
}
