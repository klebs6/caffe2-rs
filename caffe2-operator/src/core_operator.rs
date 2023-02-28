crate::ix!();

pub type OperatorObserver = ObserverBase<OperatorStorage>;

const kNoNetPositionSet: i32 = -1;

pub struct OperatorInfo {
    tensor_infos:  Vec<TensorInfo>,
    type_:          String,
}

///--------------------------------
pub struct OperatorStorage {

    //TODO: what to do with this?
    base: Observable<OperatorStorage>,

    operator_ws:             *mut Workspace,
    operator_def:            Arc<OperatorDef>,
    device_option:           DeviceOption,
    engine:                  String,
    type_:                   String,
    inputs:                  Vec<*const Blob>,
    outputs:                 Vec<*mut Blob>,

    /**
      | Preferably use c10::optional, but
      | nvcc doesn't work
      |
      */
    #[cfg(c2_available)]
    fn_schema:               Box<FunctionSchema>,

    #[cfg(c2_available)]
    newstyle_inputs:         Vec<IValue>,

    #[cfg(c2_available)]
    newstyle_outputs:        List<Tensor>,

    /**
      | HACK
      |
      | We preserve the fact that Output() returns
      | Tensor* by storing Tensor in a vector owned
      | by the operator.
      */
    input_tensors:           Vec<Tensor>,
    output_tensors:          Vec<Tensor>,
    input_size:              i32,
    net_position:            i32, // default = kNoNetPositionSet
    helper:                  *mut ExecutorHelper, // default = nullptr

    /// An event used by asynchronous execution.
    event:                   Box<Event>,
}

///-----------------------------
pub trait RunOnDevice {

    fn run_on_device(&mut self) -> bool;
}

pub trait CheckIsInputOutputAlias {

    /**
      | Check whether output j is an alias of
      | input i by comparing Blob pointers, note
      | this does not check if the two Blobs
      | points to the same Tensor, or if the
      | Tensor pointers point to the same
      | TensorImpl, or if the Storages alias
      */
    #[inline] fn is_input_output_alias(
        &mut self, 
        i: i32,
        j: i32) -> bool 
    {
        todo!();
        /*
            CAFFE_ENFORCE(
            isLegacyOperator(),
            "IsInputOutputAlias(i, j) not (yet) supported for operators exported to c10.");
        return inputs_.at(i) == outputs_.at(j);
        */
    }
}

pub trait CheckInputIsType {

    #[inline] fn input_is_type<T>(idx: i32) -> bool {
        todo!();
        /*
            CAFFE_ENFORCE(
                isLegacyOperator(),
                "InputIsType(idx) not (yet) supported for operators exported to c10.");
            static_assert(
                !std::is_same<T, Tensor>::value,
                "You should use InputIsTensorType(int, DeviceType) for "
                "Tensor.");
            return inputs_.at(idx)->template IsType<T>();
        */
    }

    #[inline] fn input_is_tensor_type(
        &mut self, 
        idx: i32,
        device_type: DeviceType) -> bool 
    {
        todo!();
        /*
            CAFFE_ENFORCE(
            isLegacyOperator(),
            "InputIsTensorType(idx, device_type) not (yet) supported for operators exported to c10.");
        return BlobIsTensorType(*inputs_.at(idx), device_type);
        */
    }
}

pub trait CheckOutputIsType {

    #[inline] fn output_is_type<T>(idx: i32) -> bool {
        todo!();
        /*
            CAFFE_ENFORCE(
                isLegacyOperator(),
                "OutputIsType(idx) not (yet) supported for operators exported to c10.");
            static_assert(
                !std::is_same<T, Tensor>::value,
                "You should use OutputIsTensorType(int, DeviceType) for "
                "Tensor.");
            return outputs_.at(idx)->template IsType<T>();
        */
    }

    #[inline] fn output_is_tensor_type(
        &mut self, 
        idx: i32,
        ty: DeviceType) -> bool 
    {
        todo!();
        /*
            CAFFE_ENFORCE(
            isLegacyOperator(),
            "OutputIsTensorType(idx, type) not (yet) supported for operators exported to c10.");
        return BlobIsTensorType(*outputs_.at(idx), type);
        */
    }
}

pub trait GetType {

    #[inline] fn type_(&self) -> &String {
        
        todo!();
        /*
            return type_;
        */
    }
}

pub trait GetInputs {

    #[inline] fn inputs(&self) -> &Vec<*const Blob> {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            isLegacyOperator(),
            "Inputs() not supported for operators exported to c10.");
        return inputs_;
        */
    }
}

pub trait GetOutputs {

    #[inline] fn outputs(&mut self) -> &Vec<*mut Blob> {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            isLegacyOperator(),
            "Outputs() not supported for operators exported to c10.");
        return outputs_;
        */
    }
}

pub trait GetInputTensorShapes {

    #[inline] fn input_tensor_shapes(&self) -> Vec<TensorShape> {
        
        todo!();
        /*
            CAFFE_ENFORCE(
          isLegacyOperator(),
          "InputTensorShapes() not supported for operators exported to c10.");
      vector<TensorShape> tps;
      for (const auto& blob : inputs_) {
        tps.push_back(GetTensorShapeOfBlob(blob));
      }
      return tps;
        */
    }
}

//TODO: clean up
pub trait GetInputAtIndex {

    /**
      | Retrieve a non-owning reference to the
      | input at position 'idx' for this operator.
      | The returned reference is valid for the
      | duration of the RunOnDevice call.  The
      | optional 'type' parameter can be used to
      | assert a required device type for the
      | input (by default, we assert that the
      | tensor is consistent with the device type
      | implied by the Context parameter of an
      | Operator.)
      */
    #[inline] fn input(
        &mut self, 
        idx: i32,
        ty: Option<DeviceType>) -> &Tensor 
    {
        let ty = todo!(); // ty.unwrap_or(Context::GetDeviceType());
        
        todo!();
        /*
            return OperatorStorage::template Input<Tensor>(idx, type);
        */
    }

    /**
      | Get the inputs and outputs as specific
      | types.
      |
      */
    #[inline] fn input_from_index<'a, T>(idx: i32) -> &'a T {
        todo!();
        /*
            static_assert(
                !std::is_same<T, Tensor>::value,
                "You should use Input<Tensor>(int, DeviceType) for "
                "Tensor.");
            DCHECK_LT((size_t)idx, inputs_.size());
            try {
              return inputs_.at(idx)->template Get<T>();
            } catch (::caffe2::EnforceNotMet& enf) {
              if (has_debug_def()) {
                TORCH_RETHROW(enf, "Offending Blob name: ", debug_def().input(idx), ".");
              }
              throw enf;
            }
        */
    }

    /**
      | TODO(jerryzh): Remove template and the type
      | argument?
      |
      | This is to keep the API changes minimal and
      | make refactoring a bit easier
      */
    #[inline] fn input_from_index_and_device_type<'a, T>(
        idx: i32, 
        ty:  DeviceType) -> &'a T 
    {
        todo!();
        /*
            if (isLegacyOperator()) {
              static_assert(
                  std::is_same<T, Tensor>::value,
                  "Input(int, DeviceType) is only available for Tensor");
              DCHECK_LT((size_t)idx, inputs_.size());
              try {
                // TODO(jerryzh): We'll need to check device type in Get<T>() later
                // Get<T>() -> Get<T>(type)
                const auto& tensor = inputs_.at(idx)->template Get<T>();
                return tensor;
              } catch (::caffe2::EnforceNotMet& enf) {
                if (has_debug_def()) {
                  TORCH_RETHROW(enf, "Offending Blob name: ", debug_def().input(idx), ".");
                }
                throw enf;
              }
            }
        #if defined(EXPOSE_C2_OPS) || \
            !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
            DCHECK_LT(0U, newstyle_inputs_.size());
            IValue ival;
            if (newstyle_inputs_[0].isTensorList()) {
              // if the first input is a tensor list, we get input tensors by indexing
              // into that list. currently, this means that only tensors from that list
              // are accessible as inputs. any hypothetical input tensors that come
              // after the list are not accessible.
              auto tensorList = newstyle_inputs_[0].toTensorVector();
              DCHECK_LT((size_t)idx, tensorList.size());
              ival = tensorList[idx];
            } else {
              // if the first input is not a tensor list, we get input tensors by
              // indexing into the inputs.
              DCHECK_LT((size_t)idx, newstyle_inputs_.size());
              ival = newstyle_inputs_[idx];
            }
            CAFFE_ENFORCE(
                ival.isTensor(),
                "Input(int, DeviceType) is only available for IValues that store Tensors");
            auto t = ival.toTensor();
            if (!t.is_contiguous()) {
              t = t.contiguous();
            }
            Tensor tensor = caffe2::Tensor(std::move(t));
            CAFFE_ENFORCE_EQ(tensor.GetDeviceType(), type);
            input_tensors_[idx] = std::move(tensor);
            return input_tensors_[idx];
        #else
            CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
        #endif
        */
    }
}

pub trait GetXOutput {

    /**
      | XOutput is a modernized version of Output
      | which returns a Tensor rather than
      | a Tensor* (the raw pointer in the latter
      | case is useless, as Tensor is a pointer
      | type.)
      */
    #[inline] fn x_output(
        &mut self, 
        idx: i32,
        dims: &[i32],
        options: TensorOptions) -> Tensor 
    {
        todo!();
        /*
            // We'll default device to the device of the current Operator Context
        if (options.device_opt() == c10::nullopt) {
          return OperatorStorage::XOutputTensor(
              idx, dims, options.device(context_.device()));
        }
        return OperatorStorage::XOutputTensor(idx, dims, options);
        */
    }

    #[inline] fn xOutput_tensor(
        &mut self, 
        idx: i32,
        dims: &[i32],
        options: TensorOptions) -> Tensor 
    {
        todo!();
        /*
            CAFFE_ENFORCE_WITH_CALLER(
            options.device_opt() != c10::nullopt,
            "device must be provided in option.");
        if (isLegacyOperator()) {
          return XBlobGetMutableTensor(outputs_.at(idx), dims, options);
        }

        return OutputTensor(idx, dims, options)->UnsafeSharedInstance();
        */
    }
}

pub trait SetOutputTensor {

    #[inline] fn set_output_tensor(
        &mut self, 
        idx: i32,
        tensor: Tensor)  
    {
        todo!();
        /*
            if (!isLegacyOperator()) {
    #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
          newstyle_outputs_[idx] = at::Tensor(tensor);

          // also update the tensor in the hack
          output_tensors_[idx] = std::move(tensor);
    #else
          CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
    #endif
        } else {
          // update the tensor in the workspace
          BlobSetTensor(outputs_.at(idx), std::move(tensor));
        }
        */
    }
}

pub trait GetOutputAtIndex_Legacy {

    /**
      | Legacy: please consider using the version
      | of Output() which also takes dtype and
      | size as arguments.
      |
      */
    #[inline] fn output_with_idx_and_device_type(
        &mut self, 
        idx: i32,
        ty: Option<DeviceType>) -> *mut Tensor 
    {
        let ty = todo!(); //ty.unwrap_or(Context::GetDeviceType());
        
        todo!();
        /*
            return OperatorStorage::template Output<Tensor>(idx, type);
        */
    }
}

pub trait GetOutputAtIndexTensorCopy {

    /**
      | Get the output Tensor of an operator
      | (allocating it if it is not already
      | initialized), and copy the contents of src
      | into it.
      |
      | You probably don't actually want to use
      | this function (the fact that you have
      | a Tensor to copy from is probably
      | a mistake: you should have written the
      | output into the output tensor, from
      | Output, directly in the first place), but
      | this method is situationally useful.
      */
    #[inline] fn output_tensor_copy_from(
        &mut self, 
        idx:     i32,
        options: TensorOptions,
        src:     &Tensor,
        async_:  Option<bool>) -> *mut Tensor 
    {
        let async_: bool = async_.unwrap_or(false);

        todo!();
        /*
            if (options.device_opt() == c10::nullopt) {
          return OperatorStorage::OutputTensorCopyFrom(
              idx, options.device(context_.device()), src, async);
        }
        return OperatorStorage::OutputTensorCopyFrom(idx, options, src, async);
        */
    }
}

pub trait Finish {

    #[inline] fn finish(&mut self)  {
        
        todo!();
        /*
            if (event_) {
          event_->Finish();
        }
        */
    }
}

pub trait WaitEvent {

    //TODO is this the fallback?
    #[inline] fn wait_event_fallback(&mut self, ev: &Event, stream_id: Option<i32>)  {
        let stream_id = stream_id.unwrap_or(-1);
        
        todo!();
        /*
            ev.Finish();
        */
    }

    #[inline] fn wait_event(
        &mut self, 
        ev:        &Event,
        stream_id: i32)  
    {
        todo!();
        /*
            if (stream_id >= 0) {
          context_.SwitchToDevice(stream_id);
        }
        context_.WaitEvent(ev);
        */
    }
}

pub trait WaitEvents: WaitEvent {

    //TODO is this the fallback?
    #[inline] fn wait_events_fallback(
        &mut self, 
        events:    &Vec<*const Event>,
        stream_id: Option<i32>)  
    {
        let stream_id = stream_id.unwrap_or(-1);
        
        todo!();
        /*
            for (const auto& ev : events) {
          ev->Finish();
        }
        */
    }
    
    #[inline] fn wait_events(
        &mut self, 
        events:    &Vec<*const Event>,
        stream_id: Option<i32>)  
    {
        let stream_id = stream_id.unwrap_or(-1);
        
        todo!();
        /*
            if (stream_id >= 0) {
            context_.SwitchToDevice(stream_id);
        }
        for (const auto& ev : events) {
            context_.WaitEvent(*ev);
        }
        */
    }
}

pub trait Wait {

    #[inline] fn wait<Context>(
        &mut self, 
        other: &OperatorStorage,
        stream_id: Option<i32>)  
    {
        let stream_id = stream_id.unwrap_or(-1);
        
        todo!();
        /*
            if (!other.IsEventDisabled()) {
          WaitEvent(other.event(), stream_id);
        }
        */
    }
}

pub trait Run {

    #[inline] fn run(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            CAFFE_NOT_IMPLEMENTED;
        */
    }
}

pub trait RunStream {
 
    /**
      | The run function of Operator switches to
      | the device, and then carries out the
      | actual computation with RunOnDevice(). You
      | should implement RunOnDevice instead of
      | Run().
      |
      | Note: Run does not update operator's event
      | and can be used only with non-async
      | executors that do not rely on events
      */
    #[inline] fn run(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            try {
          StartAllObservers();

          context_.SwitchToDevice(stream_id);

          // Clear floating point exception flags before RunOnDevice. We will test
          // exception flags afterwards, and raise an error if an exception has
          // happened.
          if (FLAGS_caffe2_operator_throw_if_fp_exceptions ||
              FLAGS_caffe2_operator_throw_if_fp_overflow_exceptions) {
            std::feclearexcept(FE_ALL_EXCEPT);
          }

    #ifdef __GNU_LIBRARY__
          // If glibc is available, use feenableexcept that will raise exception
          // right away.
          int old_enabled_exceptions = 0;
          if (FLAGS_caffe2_operator_throw_on_first_occurrence_if_fp_exceptions) {
            if (FLAGS_caffe2_operator_throw_if_fp_exceptions ||
                FLAGS_caffe2_operator_throw_if_fp_overflow_exceptions) {
              int flag = 0;
              if (FLAGS_caffe2_operator_throw_if_fp_exceptions) {
                flag |= FE_DIVBYZERO | FE_INVALID;
              }
              if (FLAGS_caffe2_operator_throw_if_fp_overflow_exceptions) {
                flag |= FE_OVERFLOW;
              }
              old_enabled_exceptions = feenableexcept(flag);
            }
          }
    #endif
          bool result = RunOnDevice();
    #ifdef __GNU_LIBRARY__
          if (FLAGS_caffe2_operator_throw_on_first_occurrence_if_fp_exceptions) {
            if (FLAGS_caffe2_operator_throw_if_fp_exceptions ||
                FLAGS_caffe2_operator_throw_if_fp_overflow_exceptions) {
              fedisableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
              std::feclearexcept(FE_ALL_EXCEPT);
              feenableexcept(old_enabled_exceptions);
            }
          }
    #endif
          if (FLAGS_caffe2_operator_throw_if_fp_exceptions) {
            CAFFE_ENFORCE(
                !std::fetestexcept(FE_DIVBYZERO),
                "Division by zero floating point exception (FE_DIVBYZERO) reported.");
            CAFFE_ENFORCE(
                !std::fetestexcept(FE_INVALID),
                "Invalid floating point exception (FE_INVALID) reported.");
          }
          if (FLAGS_caffe2_operator_throw_if_fp_overflow_exceptions) {
            CAFFE_ENFORCE(
                !std::fetestexcept(FE_OVERFLOW),
                "Overflow floating point exception (FE_OVERFLOW) reported.");
          }
          if (!result) {
            this->RecordLastFailedOpNetPosition();
          }
          context_.FinishDeviceComputation(); // throws on error

          StopAllObservers();

          return result;
        } catch (EnforceNotMet& err) {
          if (has_debug_def()) {
            err.add_context(
                "Error from operator: \n" + ProtoDebugString(debug_def()));
            AddRelatedBlobInfo(&err);
          }
          this->RecordLastFailedOpNetPosition();
          StopAllObservers();
          throw;
        } catch (...) {
          this->RecordLastFailedOpNetPosition();
          StopAllObservers();
          throw;
        }
        */
    }
}

pub trait CheckStream {

    #[inline] fn is_stream_free(&self, stream_id: i32) -> bool {
        
        todo!();
        /*
            return context_.IsStreamFree(device_option(), stream_id);
        */
    }
}

pub trait CheckAsync {

    #[inline] fn has_async_part_fallback(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    #[inline] fn supports_async_scheduling_fallback(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }

    /**
      | Returns whether operator has async on
      | device part.
      |
      | CUDA operators by default have async
      | parts, CPU operators by default don't have
      | async parts and are finished after
      | RunOnDevice call.
      |
      | Events of operators that don't have async
      | parts are automatically set to finished
      | state by RunAsync.
      |
      | Defaulting to the value from context (true
      | for CUDA, false for CPU).
      |
      | Override in case of async CPU operators
      |
      | Async CPU operators are expected to catch
      | all exceptions in async parts and set
      | Event to finished/failed state with
      | Event::SetFinished or
      | SetFinishedWithException call.
      */
    #[inline] fn has_async_part(&self) -> bool {

        todo!();
        /*
            return context_.HasAsyncPartDefault();
        */
    }

    /**
      | Returns whether operator's RunOnDevice
      | schedules async on device part and can be
      | run without waiting for parent operator's
      | async part to be finished on the same
      | device.
      |
      | Note: when true, RunOnDevice must not
      | access the content of the input blobs as
      | they might not be computed yet
      |
      | Note: when true, operator's device needs
      | to support async scheduling:
      |
      |  - supports concept of streams: async ops
      |    scheduled on the same stream are
      |    guaranteed to be executed in the same
      |    order they were scheduled
      |
      |  - provides non-blocking cross
      |    device/cross stream synchronization
      |    primitives
      |
      | By default, assuming an op with an async
      | part can be scheduled asynchronously if
      | device supports async scheduling
      */
    #[inline] fn supports_async_scheduling(&self) -> bool {
        
        todo!();
        /*
            return HasAsyncPart() && context_.SupportsAsyncScheduling();
        */
    }
}

pub trait Cancel {
    #[inline] fn cancel(&mut self)  {
        
        todo!();
        /*
        
        */
    }
}

pub trait CancelAsyncCallback {

    #[inline] fn cancel_async_callback(&mut self)  {
        
        todo!();
        /*
        
        */
    }
}

pub trait DebugInfoString {

    #[inline] fn debug_info_string(&self) -> String {
        
        todo!();
        /*
            return "";
        */
    }
    
}

pub trait GetDebugDef {

    #[inline] fn debug_def(&self) -> &OperatorDef {
        
        todo!();
        /*
            CAFFE_ENFORCE(has_debug_def(), "operator_def was null!");
        return *operator_def_;
        */
    }
}

pub trait SetDebugDef {

    #[inline] fn set_debug_def(&mut self, operator_def: &Arc<OperatorDef>)  {
        
        todo!();
        /*
            operator_def_ = operator_def;
        */
    }
}

pub trait CheckHasDebugDef {
    #[inline] fn has_debug_def(&self) -> bool {
        
        todo!();
        /*
            return operator_def_ != nullptr;
        */
    }
}

pub trait CheckNetPosition {

    #[inline] fn net_position(&self) -> i32 {
        
        todo!();
        /*
            return net_position_;
        */
    }
}

pub trait SetNetPosition {

    #[inline] fn set_net_position(&mut self, idx: i32)  {
        
        todo!();
        /*
            net_position_ = idx;
        */
    }
}

pub trait GetDeviceOption {

    #[inline] fn device_option(&self) -> &DeviceOption {
        
        todo!();
        /*
            return device_option_;
        */
    }
}

pub trait GetEvent {

    #[inline] fn event(&self) -> &Event {
        
        todo!();
        /*
            CAFFE_ENFORCE(event_, "Event is disabled");
        return *event_;
        */
    }
}

pub trait GetEventMut {

    #[inline] fn event_mut<'a>(&'a mut self) -> &'a mut Event {
        
        todo!();
        /*
            CAFFE_ENFORCE(event_, "Event is disabled");
        return *event_;
        */
    }
}

pub trait ResetEvent {
    #[inline] fn reset_event(&mut self)  {
        
        todo!();
        /*
            if (event_) {
          event_->Reset();
        }
        */
    }
}

pub trait DisableEvent {

    #[inline] fn disable_event(&mut self)  {
        
        todo!();
        /*
            event_ = nullptr;
        */
    }
}

pub trait CheckEventDisabled {

    #[inline] fn is_event_disabled(&self) -> bool {
        
        todo!();
        /*
            return !event_;
        */
    }
}

pub trait SyncDeviceBarrierForObservers {

    #[inline] fn sync_device_barrier_for_observers_fallback(&mut self)  {
        
        todo!();
        /*
            CAFFE_NOT_IMPLEMENTED;
        */
    }

    /// Internal API invoked by observers. Normal callers shouldn't invoke it.
    #[inline] fn sync_device_barrier_for_observers(&mut self)  {
        
        todo!();
        /*
            context_.FinishDeviceComputation();
        */
    }
}

pub trait CheckStreamFree {

    /**
      | Checks whether stream is ready to execute
      | new computation, used in stream allocation
      | optimization to skip stream that is
      | currently busy. Depends on context and
      | operator's device, returns true by default
      */
    #[inline] fn is_stream_free(&self, unused: i32) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
}

pub trait GetContext<Context> {

    #[inline] fn get_context(&self) -> *const Context {
        
        todo!();
        /*
            return &context_;
        */
    }
}

pub trait GetContextMut<Context> {

    #[inline] fn get_context_mut(&mut self) -> *mut Context {
        
        todo!();
        /*
            return &context_;
        */
    }
}

pub trait RecordEvent {

    #[inline] fn record_event_fallback(&mut self, err_msg: *const u8)  {
        
        todo!();
        /*
            CAFFE_NOT_IMPLEMENTED;
        */
    }

    #[inline] fn record_event(&mut self, err_msg: *const u8)  {
        
        todo!();
        /*
            if (event_) {
          context_.Record(event_.get(), err_msg);
        }
        */
    }
}

pub trait RecordLastFailedOpNetPosition {

    #[inline] fn record_last_failed_op_net_position(&mut self)  {
        
        todo!();
        /*
            if (net_position_ != kNoNetPositionSet) {
          VLOG(1) << "Operator with id " << net_position_ << " failed";
          operator_ws_->last_failed_op_net_position = net_position_;
        } else {
          VLOG(1) << "Failed operator doesn't have id set";
        }
        */
    }
}

pub trait AnnotateEngine {

    #[inline] fn annotate_engine(&mut self, engine: &String)  {
        
        todo!();
        /*
            engine_ = engine;
        */
    }
}

pub trait GetEngine {

    #[inline] fn engine(&self) -> &String {
        
        todo!();
        /*
            return engine_;
        */
    }
}

pub trait GetVectorFromIValueList {

    #[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
    #[inline] fn get_vector_from_ivalue_list<T>(value: &IValue) -> Vec<T> {
        todo!();
        /*
            return value.template to<List<T>>().vec();
        */
    }

    #[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
    #[inline] fn get_vector_fromi_value_listi32(&self, value: &IValue) -> Vec<i32> {
        
        todo!();
        /*
            auto vs = value.toIntVector();
      vector<int> out;
      out.reserve(vs.size());
      for (int64_t v : vs) {
        out.emplace_back(v);
      }
      return out;
        */
    }
    
    #[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
    #[inline] fn get_vector_fromi_value_listf32(&self, value: &IValue) -> Vec<f32> {
        
        todo!();
        /*
            const auto& vs = value.toDoubleVector();
      vector<float> out;
      out.reserve(vs.size());
      for (double v : vs) {
        out.emplace_back(v);
      }
      return out;
        */
    }
    
    #[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
    #[inline] fn get_vector_fromi_value_list_string(&self, value: &IValue) -> Vec<String> {
        
        todo!();
        /*
            auto vs = value.template to<c10::List<string>>();
      vector<string> out;
      out.reserve(vs.size());
      for (string v : vs) {
        out.emplace_back(v);
      }
      return out;
        */
    }
    
    /**
      | We need this specialisation because IValue
      | based lists don't support int16_t. We need
      | to load it as List<int64_t> and transform
      | to int16_t.
      */
    #[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
    #[inline] fn get_vector_fromi_value_listi16(&self, value: &IValue) -> Vec<i16> {
        
        todo!();
        /*
            auto list = value.template to<c10::List<int64_t>>();
      std::vector<int16_t> result;
      result.reserve(list.size());
      for (int64_t elem : list) {
        result.push_back(static_cast<int16_t>(elem));
      }
      return result;
        */
    }
}

pub trait CheckLegacyOperator {

    /**
      | -----------
      | @brief
      | 
      | Return true if the operator was instantiated
      | with OperatorDef
      | 
      | New operators should be instantiated
      | with
      | 
      | FunctionSchema
      |
      */
    #[inline] fn is_legacy_operator(&self) -> bool {
        
        todo!();
        /*
            #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
        return !fn_schema_;
    #else
        return true;
    #endif
        */
    }
}

pub trait GetFunctionSchema {

    #[inline] fn get_function_schema(&self) -> &FunctionSchema {
        
        todo!();
        /*
            CAFFE_ENFORCE(!isLegacyOperator());
    #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
        return *fn_schema_.get();
    #else
        CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
    #endif
        */
    }
}

pub trait GetSingleArgument {

    /**
      | Functions that deal with
      | arguments. Basically, this allows us to
      | map an argument name to a specific type of
      | argument that we are trying to access.
      */
    #[inline] fn get_single_argument<T>(name: &String, default_value: &T) -> T {
        todo!();
        /*
            if (isLegacyOperator()) {
              CAFFE_ENFORCE(operator_def_, "operator_def was null!");
              return ArgumentHelper::GetSingleArgument<OperatorDef, T>(
                  *operator_def_, name, default_value);
            }
        #if defined(EXPOSE_C2_OPS) || \
            !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
            auto index = argumentIndexWithName(name);
            CAFFE_ENFORCE(index.has_value(), "Couldn't get index for argument!", name);
            const auto& value = newstyle_inputs_[index.value()];
            return value.template to<T>();
        #else
            CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
        #endif
        */
    }

    #[inline] fn get_single_argument_net_def(
        &self, 
        name:          &String, 
        default_value: &NetDef) -> NetDef 
    {
        todo!();
        /*
            if (isLegacyOperator()) {
        CAFFE_ENFORCE(operator_def_, "operator_def was null!");
        return ArgumentHelper::GetSingleArgument<OperatorDef, NetDef>(
            *operator_def_, name, default_value);
      }
      CAFFE_THROW("Cannot get NetDefs from IValue");
      return NetDef();
        */
    }
}

pub trait MoveNewstyleOutputs {

    #[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
    #[inline] fn move_newstyle_outputs(&mut self) -> List<Tensor> {
        
        todo!();
        /*
            return std::move(newstyle_outputs_);
        */
    }
}

pub trait GetRepeatedArgument {

    #[inline] fn get_repeated_argument<T>(
        &self,
        name: &String,
        default_value: &Vec<T>) -> Vec<T> 
    {
        todo!();
        /*
            if (isLegacyOperator()) {
            CAFFE_ENFORCE(operator_def_, "operator_def was null!");
            return ArgumentHelper::GetRepeatedArgument<OperatorDef, T>(
                *operator_def_, name, default_value);
          }
        #if defined(EXPOSE_C2_OPS) || \
            !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
          auto index = argumentIndexWithName(name);
          CAFFE_ENFORCE(index.has_value(), "Couldn't get index for argument!", name);
          const auto& value = newstyle_inputs_[index.value()];
          return GetVectorFromIValueList<T>(value);
        #else
          CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
        #endif
        */
    }

    /**
      | We need this specialisation because IValue
      | based lists don't support int16_t. We need
      | to load it as List<int64_t> and transform
      | to int16_t.
      */
    #[inline] fn get_repeated_argumenti16(&self, name: &String, default_value: &Vec<i16>) -> Vec<i16> {
        
        todo!();
        /*
            if (isLegacyOperator()) {
        CAFFE_ENFORCE(operator_def_, "operator_def was null!");
        return ArgumentHelper::GetRepeatedArgument<OperatorDef, int16_t>(
            *operator_def_, name, default_value);
      }
    #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
      auto index = argumentIndexWithName(name);
      CAFFE_ENFORCE(index.has_value(), "Couldn't get index for argument!", name);
      const auto& value = newstyle_inputs_[index.value()];
      auto vec = GetVectorFromIValueList<int64_t>(value);
      std::vector<int16_t> result;
      result.reserve(vec.size());
      for (int64_t elem : vec) {
        result.push_back(static_cast<int16_t>(elem));
      }
      return result;
    #else
      CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
    #endif
        */
    }
}

pub trait AddRelatedBlobInfo {

    #[inline] fn add_related_blob_info(&mut self, err: *mut EnforceNotMet)  {
        
        todo!();
        /*
            CAFFE_ENFORCE(
          isLegacyOperator(),
          "AddRelatedBlobInfo(err) not supported for operators exported to c10.");

      if (!has_debug_def()) {
        return;
      }

      bool found_input = false;
      bool found_output = false;
      if (err->caller() != nullptr) {
        std::ostringstream oss;
        for (size_t i = 0; i < inputs_.size(); i++) {
          if (inputs_[i]->GetRaw() == err->caller()) {
            found_input = true;
            oss << "while accessing input: " << debug_def().input(i);
            break;
          }
        }
        for (size_t i = 0; i < outputs_.size(); i++) {
          if (outputs_[i]->GetRaw() == err->caller()) {
            found_output = true;
            if (found_input) {
              oss << " OR ";
            }
            oss << "while accessing output: " << debug_def().output(i);
            break;
          }
        }
        if (found_input || found_output) {
          err->add_context(oss.str());
        }
      }
        */
    }
}

pub trait CheckArgumentWithName {

    /**
      | -----------
      | @brief
      | 
      | Checks if the operator has an argument
      | of the given name.
      |
      */
    #[inline] fn has_argument(&self, name: &String) -> bool {
        
        todo!();
        /*
            if (isLegacyOperator()) {
          CAFFE_ENFORCE(operator_def_, "operator_def was null!");
          return ArgumentHelper::HasArgument(*operator_def_, name);
        }
        return argumentIndexWithName(name).has_value();
        */
    }

    #[inline] fn argument_index_with_name(&self, name: &String) -> Option<i32> {
        
        todo!();
        /*
            #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
      return getFunctionSchema().argumentIndexWithName(name);
    #else
      CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
    #endif
        */
    }
}

pub trait CheckHasSingleArgumentOfType {

    #[inline] fn has_single_argument_of_type<T>(name: &String) -> bool {
        todo!();
        /*
            CAFFE_ENFORCE(operator_def_, "operator_def was null!");
            return ArgumentHelper::HasSingleArgumentOfType<OperatorDef, T>(
                *operator_def_, name);
        */
    }
}

pub trait SetEventFinished {

    #[inline] fn set_event_finished(&mut self, err_msg: *const u8)  {
        
        todo!();
        /*
            if (event_) {
          event_->SetFinished(err_msg);
        }
        */
    }

    #[inline] fn set_event_finished_with_exception(&mut self, err_msg: *const u8)  {
        
        todo!();
        /*
            if (event_) {
          event_->SetFinishedWithException(err_msg);
        }
        */
    }
}

pub trait GetErrorMessage {

    #[inline] fn get_error_msg(&mut self) -> String {
        
        todo!();
        /*
            if (has_debug_def()) {
          return "Error from operator: " + ProtoDebugString(debug_def());
        } else {
          return "Error from operator: no op def";
        }
        */
    }
}

pub trait RunAsyncStream {
    
    /**
      | RunAsync, if implemented by the specific
      | operators, will schedule the computation
      | on the corresponding context and record
      | the event in its event_ member object.
      | 
      | If the specific operator does not support
      | 
      | RunAsync, it will simply be synchronous
      | as a fallback.
      |
      */
    #[inline] fn run_async_fallback(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            try {
        auto result = Run(stream_id);
        if (result) {
          if (HasAsyncPart()) {
            RecordEvent();
          } else {
            SetEventFinished();
          }
        } else {
          SetEventFinished(getErrorMsg().c_str());
        }
        return result;
      } catch (EnforceNotMet& err) {
        SetEventFinishedWithException(err.what());
        throw;
      } catch (const std::exception& err) {
        SetEventFinishedWithException(err.what());
        throw;
      } catch (...) {
        SetEventFinishedWithException(getErrorMsg().c_str());
        throw;
      }
        */
    }

    #[inline] fn run_async(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            try {
          StartAllObservers();

          context_.SwitchToDevice(stream_id);
          auto result = RunOnDevice();
          if (result) {
            if (HasAsyncPart()) {
              RecordEvent();
            } else {
              // Manually set CPU operator's event status to finished,
              // unless this is an async CPU operator
              SetEventFinished();
            }
          } else {
            SetEventFinished(getErrorMsg().c_str());
            this->RecordLastFailedOpNetPosition();
          }

          StopAllObservers();

          return result;
        } catch (EnforceNotMet& err) {
          if (has_debug_def()) {
            err.add_context(
                "Error from operator: \n" + ProtoDebugString(debug_def()));
            AddRelatedBlobInfo(&err);
          }
          SetEventFinishedWithException(err.what());
          this->RecordLastFailedOpNetPosition();
          StopAllObservers();
          throw;
        } catch (const std::exception& err) {
          SetEventFinishedWithException(err.what());
          this->RecordLastFailedOpNetPosition();
          StopAllObservers();
          throw;
        } catch (...) {
          SetEventFinishedWithException(getErrorMsg().c_str());
          this->RecordLastFailedOpNetPosition();
          StopAllObservers();
          throw;
        }
        */
    }
}


pub trait GetOutputAtIndex {
    
    /**
      | Retrieve a non-owning pointer to the
      | output at position 'idx', initializing it
      | to have size 'dims' and properties
      | 'options' if there is no pre-existing
      | output or the pre-existing output does not
      | have the correct options.  The returned
      | pointer is valid for the duration of the
      | RunOnDevice call.  If device is not
      | explicitly specified in options, we
      | default to allocating output on the
      | current device of the device type implied
      | by the Context parameter of this Operator.
      |
      | Note [Operator::Output what?]
      | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |
      | The contract of Operator::Output is
      | somewhat complex; it is perhaps better
      | understood in terms of what was
      | historically an idiomatic Caffe2 operator
      | implementation:
      |
      |     void RunOnDevice() override {
      |         auto* output = Output(0, output_size, dtype<float>());
      |         float* output_ptr = output->data<float>();
      |         // write into output_ptr
      |     }
      |
      | In the simple case, this code does the
      | following things:
      |
      |   1. Allocates a new tensor with size
      |      'output_size' and dtype 'float' (and
      |      device type whatever the Operator's
      |      device type is)
      |
      |   2. "Registers" this tensor as the 0th
      |      output tensor of this operator
      |      (Caffe2 operators don't "return"
      |      outputs; instead, outputs are shoved
      |      into an output vector which the
      |      executor reads out.)
      |
      |   3. Returns the tensor, so the operator
      |      implementation can write the actual
      |      output data into the tensor.
      |
      | So what's this business with
      | "pre-existing" outputs?  Caffe2 commonly
      | applies an optimization whereby it reuses
      | tensors on subsequent runs of operators in
      | a graph.  It doesn't know ahead of time
      | what intermediate tensors it will need, so
      | the first time it runs a graph it has all
      | of the operators create the outputs
      | necessary (as described above).  However,
      | the second time around, it will reuse all
      | of the tensors created from the first
      | time. If they are lucky, this time the
      | Output() call is a no-op and just returns
      | the old tensor.
      |
      | However, we cannot /guarantee/ that the
      | output size will be the same the next time
      | the Operator is called; for example,
      | output size may be data dependent and vary
      | between runs.  In this case, we have to
      | resize it to the correct size.  Resizing
      | is still helpful, as we may be able to fit
      | the output in the same space that was
      | previously used.
      |
      */
    #[inline] fn output(
        &mut self, 
        idx:     i32,
        dims:    &[i32],
        options: TensorOptions) -> *mut Tensor 
    {
        todo!();
        /*
            // We'll default device to the device of the current Operator Context
        if (options.device_opt() == c10::nullopt) {
          return OperatorStorage::OutputTensor(
              idx, dims, options.device(context_.device()));
        }
        return OperatorStorage::OutputTensor(idx, dims, options);
        */
    }

    #[inline] fn output_from_idx<T>(idx: i32) -> *mut T {
        todo!();
        /*
            CAFFE_ENFORCE(
                isLegacyOperator(),
                "Output(idx) not supported for operators exported to c10. Please use XOutput instead.");

            static_assert(
                !std::is_same<T, Tensor>::value,
                "You should use Output<Tensor>(int, DeviceType) for "
                "Tensor.");
            return outputs_.at(idx)->template GetMutable<T>();
        */
    }

    // TODO(jerryzh): Remove this template
    #[inline] fn output_from_idx_and_device_type<T>(idx: i32, ty: DeviceType) -> *mut T {
        todo!();
        /*
            if (isLegacyOperator()) {
              static_assert(
                  std::is_same<T, Tensor>::value,
                  "Output(int, DeviceType) is only available for Tensor");
              // When you get a Tensor here it is not fully initialized
              return BlobGetMutableTensor(outputs_.at(idx), type);
            }
        #if defined(EXPOSE_C2_OPS) || \
            !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
            at::Tensor output = newstyle_outputs_[idx];
            if (!output.defined() || caffe2::Tensor(output).GetDeviceType() != type) {
              // Fix tensor type
              Tensor tensor = Tensor(type);
              output = at::Tensor(std::move(tensor.getIntrusivePtr()));
            }
            output_tensors_[idx] = caffe2::Tensor(output);
            newstyle_outputs_[idx] = std::move(output);
            return &output_tensors_[idx];
        #else
            CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
        #endif
        */
    }

    #[inline] fn output_tensor_or_undefined(&mut self, idx: i32) -> Tensor {
        
        todo!();
        /*
            if (isLegacyOperator()) {
          return BlobGetTensorOrUndefined(*outputs_.at(idx));
        }
        return output_tensors_[idx].UnsafeSharedInstance();
        */
    }

    #[inline] fn output_tensor(
        &mut self, 
        idx: i32,
        dims: &[i32],
        options: TensorOptions) -> *mut Tensor 
    {
        todo!();
        /*
            if (isLegacyOperator()) {
          CAFFE_ENFORCE_WITH_CALLER(
              options.device_opt() != c10::nullopt,
              "device must be provided in options.");
          return BlobGetMutableTensor(outputs_.at(idx), dims, options);
        }
    #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
        at::Tensor output = newstyle_outputs_[idx];
        Tensor tensor = output.defined()
            ? GetSizedTensorWithOptions(caffe2::Tensor(output), dims, options)
            : caffe2::empty(dims, options);
        // assign it back in case it changed
        output = at::Tensor(std::move(tensor.getIntrusivePtr()));

        output_tensors_[idx] = caffe2::Tensor(output);
        newstyle_outputs_[idx] = std::move(output);
        return &output_tensors_[idx];
    #else
        CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
    #endif
        */
    }

    /**
      | Get output Tensor of the operator and
      | CopyFrom the given Tensor
      |
      */
    #[inline] fn output_tensor_copy_from_base(
        &mut self, 
        idx:     i32,
        options: TensorOptions,
        src:     &Tensor,
        async_:  Option<bool>) -> *mut Tensor 
    {
        let async_: bool = async_.unwrap_or(false);

        todo!();
        /*
            CAFFE_ENFORCE_WITH_CALLER(
            options.device_opt() != c10::nullopt,
            "device must be provided in options.");
        // Ouptut Tensor will always have the same data type as `src`
        if (!options.has_dtype()) {
          options = options.dtype(src.dtype());
        }
        CAFFE_ENFORCE_WITH_CALLER(
            options.dtype() == src.dtype(),
            "We don't allow change of src data type in OutputTensorCopyFrom");
        Tensor* t = OutputTensor(idx, src.sizes(), options);
        t->CopyFrom(src, async);
        return t;
        */
    }

    #[inline] fn output_tensor_alias(
        &mut self, 
        idx: i32,
        src: &Tensor) -> *mut Tensor 
    {
        todo!();
        /*
            CAFFE_ENFORCE(
            isLegacyOperator(),
            "OutputTensorAlias(idx, src) not (yet) supported for operators exported to c10.");
        return BlobSetTensor(OutputBlob(idx), src.Alias());
        */
    }

    #[inline] fn output_base<T>(idx: i32, allocated: *mut T) -> *mut T {
        todo!();
        /*
            CAFFE_ENFORCE(
                isLegacyOperator(),
                "Output(idx, allocated) not supported for operators exported to c10. Please use XOutput.");
            outputs_.at(idx)->Reset(allocated);
            return allocated;
        */
    }
}

pub trait GetInputBlob {

    #[inline] fn input_blob(&mut self, idx: i32) -> &Blob {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            isLegacyOperator(),
            "InputBlob(idx) not (yet) supported for operators exported to c10.");
        return *inputs_.at(idx);
        */
    }
}

pub trait GetOutputBlob {

    #[inline] fn output_blob(&mut self, idx: i32) -> *mut Blob {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            isLegacyOperator(),
            "OutputBlob(idx) not (yet) supported for operators exported to c10.");
        return outputs_.at(idx);
        */
    }
}

pub trait CheckInputSize {

    #[inline] fn input_size(&self) -> i32 {
        
        todo!();
        /*
            return input_size_;
        */
    }
}

pub trait CheckOutputSize {

    #[inline] fn output_size(&self) -> i32 {
        
        todo!();
        /*
            if (isLegacyOperator()) {
          return outputs_.size();
        }
    #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
        return newstyle_outputs_.size();
    #else
        CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
    #endif
        */
    }
}

/**
  | Operator is the class that you usually
  | want to derive, if your operator will
  | run on different devices. You should
  | then implement the RunOnDevice() function.
  |
  */
pub trait Operator {

    fn new_with_operator_def_and_workspace(
        operator_def: &OperatorDef, 
        ws: *mut Workspace) -> Self where Self: Sized
    {
        todo!();
        /*
            : OperatorStorage(operator_def, ws), context_(operator_def.device_option()) 

        // In the constructor, we switch to the device so that the child class
        // constructors will run on that device.
        context_.SwitchToDevice();
        */
    }
    
    #[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
    fn new_from_fn_schema_inputs_and_outputs(
        fn_schema: &FunctionSchema,
        inputs: Vec<IValue>,
        outputs: List<Tensor>) -> Self where Self: Sized
    {
        todo!();
        /*
            : OperatorStorage(fn_schema, std::move(inputs), std::move(outputs)) 
                  // In the constructor, we switch to the device so that the child class
                  // constructors will run on that device.
                  context_.SwitchToDevice();
        */
    }
    
    fn new_with_operator_def_and_workspace_base(
        operator_def: &OperatorDef, 
        ws: *mut Workspace) -> Self where Self: Sized
    {
        todo!();
        /*
            : operator_ws_(ws),
          operator_def_(std::make_shared<OperatorDef>(operator_def)),
          device_option_(
              operator_def.has_device_option() ? operator_def.device_option()
                                               : DeviceOption()),
    #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
          newstyle_outputs_(),
    #endif
          input_size_(operator_def.input_size()),
          event_(std::make_unique<Event>(device_option_)) 



      static GlobalInitIsCalledGuard guard;
      inputs_.reserve(operator_def.input_size());
      for (const string& input_str : operator_def.input()) {
        auto* blob = ws->GetBlob(input_str);
        CAFFE_ENFORCE(
            blob != nullptr,
            "op ",
            operator_def.type(),
            ": Encountered a non-existing input blob: ",
            input_str);
        inputs_.push_back(blob);
      }

      GetOperatorLogger()(operator_def);

      outputs_.reserve(operator_def.output_size());
      for (const string& output_str : operator_def.output()) {
        outputs_.push_back(CHECK_NOTNULL(ws->CreateBlob(output_str)));
      }

      type_ = operator_def.type();
        */
    }
    
    /**
      | Notes: All outputs ivalues must be tensors.
      | Input ivalue list must start with all
      | tensors ("inputs" in caffe2 terminology),
      | followed by non-tensors ("arguments"
      | in caffe2 terminology).
      | 
      | Alternatively, inputs can be one tensor
      | list ivalue followed by non-tensors
      | to represent operators with a variable
      | number of inputs.
      |
      */
    #[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
    fn new_from_fn_schema_inputs_and_outputs_base(
        fn_schema: &FunctionSchema,
        inputs: Vec<IValue>,
        outputs: List<Tensor>) -> Self where Self: Sized
    {
        todo!();
        /*
            : fn_schema_(make_unique<c10::FunctionSchema>(std::move(fn_schema))),
            newstyle_inputs_(std::move(inputs)),
            newstyle_outputs_(std::move(outputs)),
            input_size_(compute_input_size_(newstyle_inputs_)) 

                input_tensors_.resize(input_size_);
            output_tensors_.resize(newstyle_outputs_.size());
        */
    }
}

pub trait GetExecutorHelper {

    #[inline] fn get_executor_helper(&self) -> *mut ExecutorHelper {
        
        todo!();
        /*
            return helper_;
        */
    }
}

pub trait SetExecutorHelper {

    #[inline] fn set_executor_helper(&mut self, helper: *mut ExecutorHelper)  {
        
        todo!();
        /*
            helper_ = helper;
        */
    }
}

/**
  | Helpers to implement runtime op
  | polymorphism. Often it's convenient to make an
  | op work on different input types (e.g. i32 vs
  | i64 indices) or special-case it for particular
  | input size (e.g. ScatterWeightedSum for block
  | size of 1 doesn't need to call Eigen).
  |
  | DispatchHelper provides compile-time generation
  | of nested "if" statements,
  | e.g. `DispatchHelper<FixedValues<1,
  | 4>>::call(this, block_size);` unrolls into:
  |
  | @code
  |   if (block_size == 1) {
  |     return DoRunWithValue<1>();
  |   } else if (block_size = 4) {
  |     return DoRunWithValue<4>();
  |   } else {
  |     return DoRunWithValue<-1>();
  |   }`
  | @endcode
  |
  | DoRunWithValue implementation can use template
  | arguments to do "if" statements or proxy to
  | functions in math.h which often provide fixed
  | size implementation.
  |
  | Similarly `TensorTypes<int32_t, int64_t>(this,
  | Input(0))` provides branching based on type of
  | the first input and calls DoRunWithType.
  |
  | Note, that the same instance of Op class is
  | used as the method, not class is templated. We
  | might consider adding static class-level
  | polymorphism later.
  |
  | Convenient macro USE_DISPATCH_HELPER is
  | provided for declaring friendship in case
  | DoRunWithValue or DoRunWithType are declared
  | non-public.
  */
#[macro_export] macro_rules! use_dispatch_helper {
    () => {
        todo!();
        /*
        
          template <typename FirstArg, typename... ExtraArgs> 
          friend struct DispatchHelper
        */
    }
}

pub struct FixedValues<Values> {
    values: PhantomData<Values>,
}

pub struct TensorTypes<Types>  {
    types: PhantomData<Types>,
}

/**
  | Special tag that can be listed in TensorTypes
  | to denote that a special implementation
  | in 'RunWithOtherType' needs to be called
  | instead of failing
  | 
  | Obviously this needs to be the last item
  | in lists, e.g.
  | 
  | TensorTypes<float, double, GenericTensorImplementation>
  |
  */
pub struct GenericTensorImplementation {}

/// Same as TensorTypes but call DoRunWithType2
pub struct TensorTypes2<Types> {
    types: PhantomData<Types>,
}

///----------------------------------

impl<FixedValues,ExtraArgs> 
DispatchHelper<FixedValues, ExtraArgs> {
    
    #[inline] pub fn call_with_value<Op>(op: *mut Op, value: i32) -> bool {
    
        todo!();
        /*
            if (FirstVal == value) {
          return op->template DoRunWithValue<ExtraArgs..., FirstVal>();
        }
        return DispatchHelper<FixedValues<Values...>, ExtraArgs...>::template call<
            Op>(op, value);
        */
    }
}

///-----------------------------------
pub struct DispatchHelper<FixedValues, ExtraArgs> {
    phantomA: PhantomData<FixedValues>,
    phantomB: PhantomData<ExtraArgs>,
}

impl<FixedValues, ExtraArgs> 
DispatchHelper<FixedValues, ExtraArgs> {
    
    #[inline] pub fn call<Op>(op: *mut Op, size: i64) -> bool {
    
        todo!();
        /*
            return op->template DoRunWithValue<ExtraArgs..., -1>();
        */
    }
}

#[macro_export] macro_rules! define_tensor_types_dispatcher {
    () => {
        /*
                (                                    
            TensorTypes, DoRunWithType, DoRunWithOtherType)                            
          template <typename FirstType, typename... Types, typename... ExtraArgs>      
          struct DispatchHelper<TensorTypes<FirstType, Types...>, ExtraArgs...> {      
            template <typename Op>                                                     
            static bool call(Op* op, const TypeMeta meta) {                           
              static_assert(                                                           
                  !std::is_same<GenericTensorImplementation, FirstType>::value,        
                  "GenericTensorImplementation must be the last in TensorTypes list"); 
              if (meta.Match<FirstType>()) {                                           
                return op->template DoRunWithType<ExtraArgs..., FirstType>();          
              }                                                                        
              return DispatchHelper<TensorTypes<Types...>, ExtraArgs...>::             
                  template call<Op>(op, meta);                                         
            }                                                                          
            template <typename Op>                                                     
            static bool call(Op* op, const Tensor& tensor) {                           
              return call<Op>(op, tensor.dtype());                                     
            }                                                                          
            template <typename Op>                                                     
            static bool call(Op* op, const Blob& blob) {                               
              return call<Op>(op, blob.meta());                                        
            }                                                                          
          };                                                                           
                                                                                       
          template <typename... ExtraArgs>                                             
          struct DispatchHelper<TensorTypes<>, ExtraArgs...> {                         
            template <typename Op>                                                     
            static bool call(Op* /* unused */, const TypeMeta meta) {                 
              CAFFE_THROW("Unsupported type of tensor: ", meta.name());                
            }                                                                          
            template <typename Op>                                                     
            static bool call(Op* op, const Tensor& tensor) {                           
              return call<Op>(op, tensor.dtype());                                     
            }                                                                          
            template <typename Op>                                                     
            static bool call(Op* op, const Blob& blob) {                               
              return call<Op>(op, blob.meta());                                        
            }                                                                          
          };                                                                           
                                                                                       
          template <typename... ExtraArgs>                                             
          struct DispatchHelper<                                                       
              TensorTypes<GenericTensorImplementation>,                                
              ExtraArgs...> {                                                          
            template <typename Op>                                                     
            static bool call(Op* op, const TypeMeta) {                                
              return op->template DoRunWithOtherType<ExtraArgs...>();                  
            }                                                                          
            template <typename Op>                                                     
            static bool call(Op* op, const Tensor& tensor) {                           
              return call<Op>(op, tensor.dtype());                                     
            }                                                                          
            template <typename Op>                                                     
            static bool call(Op* op, const Blob& blob) {                               
              return call<Op>(op, blob.meta());                                        
            }                                                                          
          };
        */
    }
}

define_tensor_types_dispatcher!{
    /*
    TensorTypes,
    DoRunWithType,
    DoRunWithOtherType
    */
}

define_tensor_types_dispatcher!{
    /*
    TensorTypes2,
    DoRunWithType2,
    DoRunWithOtherType2
    */
}

/**
  | The device type registry. This works
  | in two phases:
  | 
  | (1) gDeviceTypeRegistry() maps the
  | device types values to the actual operator
  | registry function.
  | 
  | (2) Then, one can call the operator
  | registry function to further create
  | the operators.
  |
  */
pub type OperatorRegistry<'a> = Registry<String, 
    Box<OperatorStorage>, 
    (&'a OperatorDef, *mut Workspace)>;

pub type RegistryFunction<'a> = fn() -> *mut OperatorRegistry<'a>;

///-----------------------------------
pub struct DeviceTypeRegisterer {
    
}

impl DeviceTypeRegisterer {
    
    pub fn new_with_device_type_and_registry_function<'a>(
        ty:   DeviceType, 
        func: RegistryFunction<'a>) -> Self {
    
        todo!();
        /*
            if (gDeviceTypeRegistry()->count(type)) {
          std::cerr << "Device type " << DeviceTypeName(type)
                    << "registered twice. This should not happen. Did you have "
                       "duplicated numbers assigned to different devices?";
          std::exit(1);
        }
        // Calling the registry function to get the actual registry pointer.
        gDeviceTypeRegistry()->emplace(type, func());
        */
    }
}

/**
  | The operator registry. Since we are
  | not expecting a great number of devices,
  | we will simply have an if-then type command
  | and allocate the actual generation
  | to device-specific registerers.
  | 
  | -----------
  | @note
  | 
  | although we have CUDA and CUDNN here,
  | the registerers themselves do not depend
  | on specific cuda or cudnn libraries.
  | This means that we will be able to compile
  | it even when there is no cuda available
  | - we simply do not link any cuda or cudnn
  | operators.
  |
  */
declare_registry!{
    CPUOperatorRegistry,
    OperatorStorage,
    OperatorDef,
    Workspace
}

declare_registry!{
    CUDAOperatorRegistry,
    OperatorStorage,
    OperatorDef,
    Workspace
}

// Macros for HIP operators
declare_registry!{
    HIPOperatorRegistry,
    OperatorStorage,
    OperatorDef,
    Workspace
}

#[macro_export] macro_rules! register_hip_operator_creator {
    ($key:ident, $($arg:ident),*) => {
        /*
        
          C10_REGISTER_CREATOR(HIPOperatorRegistry, key, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_hip_operator {
    ($name:ident, $($arg:ident),*) => {
        /*
        
          C10_IMPORT void CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();  
          static void CAFFE2_UNUSED CAFFE_ANONYMOUS_VARIABLE_HIP##name() { 
            CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();                
          }                                                                
          C10_REGISTER_CLASS(HIPOperatorRegistry, name, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_hip_operator_str {
    ($str_name:ident, $($arg:ident),*) => {
        /*
        
          C10_REGISTER_TYPED_CLASS(HIPOperatorRegistry, str_name, __VA_ARGS__)
        */
    }
}


#[macro_export] macro_rules! register_hip_operator_with_engine {
    ($name:ident, $engine:ident, $($arg:ident),*) => {
        /*
        
          C10_REGISTER_CLASS(HIPOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_miopen_operator {
    ($name:ident, $($arg:ident),*) => {
        /*
        
          REGISTER_HIP_OPERATOR_WITH_ENGINE(name, MIOPEN, __VA_ARGS__) 
          REGISTER_HIP_OPERATOR_WITH_ENGINE(                           
              name, CUDNN, __VA_ARGS__) // Make CUDNN an alias of MIOPEN for HIP ops
        */
    }
}

/**
  | StaticLinkingProtector is a helper
  | class that ensures that the Caffe2 library
  | is linked correctly with whole archives
  | (in the case of static linking). What
  | happens is that when
  | 
  | CreateOperator is called for the first
  | time, it instantiates an OperatorLinkingProtector
  | object to check if the operator registry
  | is empty. If it is empty, this means that
  | we are not properly linking the library.
  | 
  | You should not need to use this class.
  |
  */
pub struct StaticLinkingProtector {
    
}

impl Default for StaticLinkingProtector {
    
    fn default() -> Self {
        todo!();
        /*
            const auto registered_ops = CPUOperatorRegistry()->Keys().size();
        // Note: this is a check failure instead of an exception, because if
        // the linking is wrong, Caffe2 won't be able to run properly anyway,
        // so it's better to fail loud.
        // If Caffe2 is properly linked with whole archive, there should be more
        // than zero registered ops.
        if (registered_ops == 0) {
          LOG(FATAL)
              << "You might have made a build error: the Caffe2 library does not seem "
                 "to be linked with whole-static library option. To do so, use "
                 "-Wl,-force_load (clang) or -Wl,--whole-archive (gcc) to link the "
                 "Caffe2 library.";
        
        */
    }
}

/**
  | An exception that can be thrown by an
  | operator constructor that notifies
  | that it does not support the given setting.
  | This can be usually used for specific
  | engines that only implement a subset
  | of the features required by the original
  | operator schema.
  | 
  | TODO(jiayq): make more feature-complete
  | exception message.
  |
  */
pub struct UnsupportedOperatorFeature {
    msg: String,
}

/**
  | A helper macro that should ONLY be used
  | in the operator constructor to check
  | if needed features are met. If not, throws
  | the UnsupportedOperatorFeature exception
  | with the given message.
  |
  */
#[macro_export] macro_rules! operator_needs_feature {
    ($condition:ident, $($arg:ident),*) => {
        /*
        
          if (!(condition)) {                                          
            throw UnsupportedOperatorFeature(::c10::str(__VA_ARGS__)); 
          }
        */
    }
}

/**
  | User can set the preferred engines as
  | a list of engine names, in descending
  | order of preference.
  |
  */
pub type EnginePrefType = Vec<String>;

/// {device_type -> {operator_name -> EnginePrefType}}
pub type PerOpEnginePrefType = HashMap<DeviceType,HashMap<String,EnginePrefType>>;

/// {device_type -> EnginePrefType}
pub type GlobalEnginePrefType = HashMap<DeviceType,EnginePrefType>;

/**
  | This is for transferring tensor data
  | between C2 and backends.
  |
  */
#[cfg(not(c10_mobile))]
pub struct ExternalTensorDescriptor {
    data_type:            u64,
    dimensions:           u32,
    shape:                *const u64,
    is_offline:           u8, // default = 0
    quantization_axis:    u32,
    quantization_params:  u64,
    scales:               *const f32,
    biases:               *const i32,
    buffer:               u64,
}

///----------------------------
pub trait ExternalTensorFunctionsBase {

    fn is_quantized(&self) -> bool;

    fn is_same_meta_type(&self, id: TypeIdentifier) -> bool;

    fn setup_external_tensor_descriptor(
        &self,
        blob:        *const Blob,
        shapes:      *mut Vec<Vec<u64>>,
        all_scales:  *mut Vec<Vec<f32>>,
        all_offsets: *mut Vec<Vec<i32>>,
        desc:        *mut ExternalTensorDescriptor) {
        todo!();
    }

    fn load_info_of_blob(
        &self,
        blob:   *const Blob,
        scale:  *mut Vec<f32>,
        offset: *mut Vec<f32>,
        axis:   *mut u32);

    fn get_type_meta_id(&self) -> TypeIdentifier;

    fn get_external_tensor_type(
        &self,
        c: *const c_void) -> TypeMeta;

    fn get_external_tensor_info(
        &mut self,
        c:        *const c_void,
        capacity: *mut usize,
        device:   *mut DeviceOption) -> Vec<i64> {
        todo!();
    }
}

#[macro_export] macro_rules! register_external_tensor_functions {
    ($id:expr, $($arg:expr),*) => {
        /*
        
          C10_REGISTER_TYPED_CLASS(ExternalTensorFunctionsBaseRegistry, id, __VA_ARGS__)
        */
    }
}

#[inline] pub fn create_external_tensor_functions(id: TypeIdentifier) -> Box<dyn ExternalTensorFunctionsBase> {
    
    todo!();
    /*
        return ExternalTensorFunctionsBaseRegistry()->Create(id);
    */
}

#[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
#[inline] pub fn compute_input_size_(inputs: &Vec<IValue>) -> i32 {
    
    todo!();
    /*
        if (inputs.empty()) {
        return 0;
      }
      if (inputs[0].isTensorList()) {
        // if the first input is a tensor list, we get input tensors by indexing
        // into that list. currently, this means that only tensors from that list
        // are accessible as inputs. any hypothetical input tensors that come after
        // the list are not accessible.
        return inputs[0].toTensorVector().size();
      }
      // it's not a tensor list. Count the number of tensor inputs and return them.
      size_t num_tensor_inputs = 0;
      bool found_nontensor = false;
      for (const auto& input : inputs) {
        if (input.isTensor()) {
          AT_ASSERTM(
              !found_nontensor,
              "All tensor arguments must come before non-tensor arguments");
          ++num_tensor_inputs;
        } else {
          found_nontensor = true;
        }
      }
      return num_tensor_inputs;
    */
}

#[inline] pub fn g_per_op_engine_pref<'a>() -> &'a mut PerOpEnginePrefType {
    
    todo!();
    /*
        static auto* g_per_op_engine_pref_ = new PerOpEnginePrefType();
      return *g_per_op_engine_pref_;
    */
}

#[inline] pub fn g_global_engine_pref<'a>() -> &'a mut GlobalEnginePrefType {
    
    todo!();
    /*
        static auto* g_global_engine_pref_ =
          new GlobalEnginePrefType{{CUDA, {"CUDNN"}}, {HIP, {"MIOPEN"}}};
      return *g_global_engine_pref_;
    */
}

#[inline] pub fn try_create_operator<Context>(
    key:          &String,
    operator_def: &OperatorDef,
    ws:           *mut Workspace) -> Box<OperatorStorage> 
{
    
    todo!();
    /*
        const auto& type_proto = operator_def.device_option().device_type();
      const auto& type = ProtoToType(static_cast<DeviceTypeProto>(type_proto));
      CAFFE_ENFORCE(
          gDeviceTypeRegistry()->count(type),
          "Device type ",
          type,
          " not registered.");
      OperatorRegistry* registry = gDeviceTypeRegistry()->at(type);
      VLOG(1) << "Creating operator with device type " << type;
      try {
        return registry->Create(key, operator_def, ws);
      } catch (const UnsupportedOperatorFeature& err) {
        LOG(WARNING) << "Operator " << operator_def.type()
                     << " does not support the requested feature. Msg: "
                     << err.what()
                     << ". Proto is: " << ProtoDebugString(operator_def);
        return nullptr;
      }
    */
}

#[inline] pub fn create_operator<Context>(
    operator_def: &OperatorDef,
    ws: *mut Workspace) -> Box<OperatorStorage> 
{
    todo!();
    /*
        static StaticLinkingProtector g_protector;
      const auto& op_type = operator_def.type();
      const auto& device_type_proto = operator_def.device_option().device_type();
      const auto& device_type =
          ProtoToType(static_cast<DeviceTypeProto>(device_type_proto));

    #ifndef CAFFE2_NO_OPERATOR_SCHEMA
      // first, check with OpSchema if the operator is legal.
      auto* schema = OpSchemaRegistry::Schema(op_type);
      if (schema) {
        CAFFE_ENFORCE(
            schema->Verify(operator_def),
            "Operator def did not pass schema checking: ",
            ProtoDebugString(operator_def));
      } else {
        // We would like to recommend every op to register its schema, so if there
        // is not one, we print a LOG_ERROR. But we will still allow the operator
        // to be constructed.
        LOG(ERROR) << "Cannot find operator schema for " << op_type
                   << ". Will skip schema checking.";
      }
    #endif

      // second try engines specified in the operator_def and preferred engines
      std::vector<std::string> engines{};
      if (operator_def.engine().size()) {
        const auto op_def_engines = split(',', operator_def.engine());
        engines.insert(engines.end(), op_def_engines.begin(), op_def_engines.end());
      }
      if (!FLAGS_caffe2_disable_implicit_engine_preference &&
          g_per_op_engine_pref().count(device_type) &&
          g_per_op_engine_pref()[device_type].count(op_type)) {
        const auto& preferred_engines =
            g_per_op_engine_pref()[device_type][op_type];
        VLOG(2) << "Inserting per-op engine preference: " << preferred_engines;
        engines.insert(
            engines.end(), preferred_engines.begin(), preferred_engines.end());
      }
      if (!FLAGS_caffe2_disable_implicit_engine_preference &&
          g_global_engine_pref().count(device_type)) {
        const auto& preferred_engines = g_global_engine_pref()[device_type];
        VLOG(2) << "Inserting global engine preference: " << preferred_engines;
        engines.insert(
            engines.end(), preferred_engines.begin(), preferred_engines.end());
      }
      for (const auto& engine : engines) {
        const std::string key = OpRegistryKey(op_type, engine);
        VLOG(1) << "Trying to create operator " << op_type << " with engine "
                << engine;
        auto op = TryCreateOperator(key, operator_def, ws);
        if (op) {
          if (engine.size() <=
              (unsigned)FLAGS_caffe2_operator_max_engine_name_length) {
            op->annotate_engine(engine);
          } else {
            op->annotate_engine(
                engine.substr(0, FLAGS_caffe2_operator_max_engine_name_length));
          }
          return op;
        } else {
          // If the above fails, we will just return the normal case with the
          // default implementation.
          VLOG(1) << "Engine " << engine << " is not available for operator "
                  << op_type << ".";
        }
      }
      if (operator_def.engine().size() && !VLOG_IS_ON(1)) {
        static int log_occurrences = 0;
        if (log_occurrences <= 64) {
          ++log_occurrences;
          LOG(INFO) << "Engine " << operator_def.engine()
                    << " is not available for operator " << op_type << ".";
        }
      }
      VLOG(1) << "Using default implementation.";

      // Lastly, if the engine does not work here, try using the default engine.
      auto op = TryCreateOperator(op_type, operator_def, ws);
      CAFFE_ENFORCE(
          op,
          "Cannot create operator of type '",
          op_type,
          "' on the device '",
          DeviceTypeName(device_type),
          "'. Verify that implementation for the corresponding device exist. It "
          "might also happen if the binary is not linked with the operator "
          "implementation code. If Python frontend is used it might happen if "
          "dyndep.InitOpsLibrary call is missing. Operator def: ",
          ProtoDebugString(operator_def));
      return op;
    */
}

#[inline] pub fn op_registry_key(
    op_type: &String,
    engine: &String) -> String 
{
    todo!();
    /*
        if (engine == "" || engine == "DEFAULT") {
        return op_type;
      } else {
        return op_type + "_ENGINE_" + engine;
      }
    */
}


#[inline] pub fn set_per_op_engine_pref(
    per_op_engine_pref: &PerOpEnginePrefType)
{
    todo!();
    /*
        for (const auto& device_pref_pair : per_op_engine_pref) {
        const auto& device_type = device_pref_pair.first;
        CAFFE_ENFORCE(
            gDeviceTypeRegistry()->count(device_type),
            "Device type ",
            device_type,
            " not registered.");
        auto* registry = gDeviceTypeRegistry()->at(device_type);

        for (const auto& op_pref_pair : device_pref_pair.second) {
          const auto& op_type = op_pref_pair.first;
          CAFFE_ENFORCE(
              registry->Has(op_type),
              "Operator type ",
              op_type,
              " not registered in ",
              device_type,
              " registry.");
        }
      }
      g_per_op_engine_pref() = per_op_engine_pref;
    */
}


#[inline] pub fn set_global_engine_pref(
    global_engine_pref: &GlobalEnginePrefType)
{
    todo!();
    /*
        for (const auto& device_pref_pair : global_engine_pref) {
        const auto& device_type = device_pref_pair.first;
        CAFFE_ENFORCE(
            gDeviceTypeRegistry()->count(device_type),
            "Device type ",
            device_type,
            " not registered.");
      }
      g_global_engine_pref() = global_engine_pref;
    */
}

#[inline] pub fn set_engine_pref(
    per_op_engine_pref: &PerOpEnginePrefType,
    global_engine_pref: &GlobalEnginePrefType)
{
    
    todo!();
    /*
        SetPerOpEnginePref(per_op_engine_pref);
      SetGlobalEnginePref(global_engine_pref);
    */
}

#[inline] pub fn set_op_engine_pref(
    op_type: &String,
    op_pref: &HashMap<DeviceType, EnginePrefType>)  
{
    todo!();
    /*
        for (const auto& device_pref_pair : op_pref) {
        const auto& device_type_proto = device_pref_pair.first;
        const auto& device_type =
            ProtoToType(static_cast<DeviceTypeProto>(device_type_proto));
        CAFFE_ENFORCE(
            gDeviceTypeRegistry()->count(device_type),
            "Device type ",
            device_type,
            " not registered.");
        CAFFE_ENFORCE(
            gDeviceTypeRegistry()->at(device_type)->Has(op_type),
            "Operator type ",
            op_type,
            " not registered in ",
            device_type,
            " registry.");
        g_per_op_engine_pref()[device_type][op_type] = device_pref_pair.second;
      }
    */
}

/**
  | Creates an operator with the given operator
  | definition.
  |
  | Throws on error and never returns nullptr
  */
#[inline] pub fn create_operator_with_net_position<Context>(
    operator_def: &OperatorDef,
    ws:           *mut Workspace,
    net_position: Option<i32>) -> Box<OperatorStorage> 
{
    let net_position: i32 = net_position.unwrap_or(kNoNetPositionSet);

    todo!();
    /*
        try {
        auto op = _CreateOperator(operator_def, ws);
        op->set_net_position(net_position);
        return op;
      } catch (...) {
        if (net_position != 0) {
          VLOG(1) << "Operator constructor with net position " << net_position
                  << " failed";
          ws->last_failed_op_net_position = net_position;
        } else {
          VLOG(1) << "Failed operator constructor doesn't have an id set";
        }
        throw;
      }
    */
}

#[inline] pub fn g_device_type_registry<'a>() 
-> *mut HashMap<DeviceType, *mut OperatorRegistry<'a>> 
{
    todo!();
    /*
        static std::map<DeviceType, OperatorRegistry*> g_device_type_registry;
      return &g_device_type_registry;
    */
}

#[inline] pub fn get_gradient_for_op(
    def: &OperatorDef,
    g_output: &Vec<GradientWrapper>) -> GradientOpsMeta 
{
    todo!();
    /*
        C10_LOG_API_USAGE_ONCE("caffe2.gradient_maker");
      std::unique_ptr<GradientMakerBase> maker(
          GradientRegistry()->Create(def.type(), def, g_output));
      CAFFE_ENFORCE(
          maker, "Gradient maker for operator ", def.type(), " not implemented.");
      GradientOpsMeta meta = maker->Get();
      // Copy device option, engine, and arguments if needed.
      if (maker->CopyDeviceOption() && def.has_device_option()) {
        for (OperatorDef& grad_def : meta.ops_) {
          grad_def.mutable_device_option()->CopyFrom(def.device_option());
        }
      }
      // Copy engine if needed.
      if (maker->CopyEngine() && def.has_engine()) {
        for (OperatorDef& grad_def : meta.ops_) {
          grad_def.set_engine(def.engine());
        }
      }
      // Copy arguments if needed.
      if (maker->CopyArguments() && def.arg_size()) {
        for (OperatorDef& grad_def : meta.ops_) {
          for (auto& arg : def.arg()) {
            grad_def.add_arg()->CopyFrom(arg);
          }
        }
      }
      // VLOG for debugging purposes.
      for (const OperatorDef& grad_def : meta.ops_) {
        VLOG(1) << "Gradient ops: " << ProtoDebugString(grad_def);
      }
      // Check if the gradient computation has returned the right size for the
      // gradient vector.
      CAFFE_ENFORCE_EQ(meta.g_input_.size(), def.input_size());
      VLOG(1) << "Gradients:";
      for (const GradientWrapper& grad : meta.g_input_) {
        // The gradient should either be (1) not set, or (2) dense, or (3) sparse,
        // but cannot be both dense and sparse.
        if (!grad.IsDense() && !grad.IsSparse()) {
          VLOG(1) << "\t [no gradient]";
        } else if (grad.IsDense()) {
          VLOG(1) << "\t [dense]" << grad.dense_;
        } else {
          CAFFE_ENFORCE(
              grad.indices_.size() && grad.values_.size(),
              "For sparse gradient, one should set both indices and values. "
              "Currently we have: (" +
                  grad.indices_ + ", " + grad.values_ + ").");
          VLOG(1) << "\t [sparse] " << grad.indices_ << ", " << grad.values_;
        }
      }
      return meta;
    */
}

#[inline] pub fn infer_blob_shapes_and_types(
    blob_desc: &mut HashMap<String, TensorShape>,
    nets:      &Vec<*mut NetDef>)  
{
    todo!();
    /*
        for (auto& defptr : nets) {
        // Hack to work with auto split gradients
        CaffeMap<string, string> unmatched_sum_blobs;
        CaffeMap<string, TensorShape> reshape_cache;
        CaffeMap<string, vector<TensorShape>> split_cache;

        for (const OperatorDef& op : defptr->op()) {
          // Hack to ignore queues
          if (op.type().find("Dequeue") != std::string::npos ||
              op.type().find("Enqueue") != std::string::npos) {
            continue;
          }

          vector<TensorShape> input_desc;
          bool found_all = true;
          for (const string& in : op.input()) {
            auto inp_desc = blob_desc.find(in);
            if (inp_desc == blob_desc.end()) {
              LOG(WARNING) << "Shape and type inference failed for input: " << in
                           << " for op " << op.type() << ", skipping.";
              found_all = false;
              break;
            }
            input_desc.push_back(inp_desc->second);
          }
          if (!found_all) {
            continue;
          }
          auto op_schema = OpSchemaRegistry::Schema(op.type());
          if (op_schema == nullptr) {
            LOG(WARNING) << "Shape inference failed, no schema for: " << op.type();
            continue;
          }

          // Special handling for Sum as it used with the autosplits, which have
          // different naming convention. Assuming that all sum inputs must be of
          // same size, we can infer their shapes.
          if (op.type() == "Sum") {
            TensorShape sum_shape;
            for (auto inp : op.input()) {
              auto it = blob_desc.find(inp);
              if (it != blob_desc.end() && !it->second.unknown_shape()) {
                if (it->second.dims_size() > 0) {
                  sum_shape = blob_desc[inp];
                  break;
                }
              }
            }
            for (auto inp : op.input()) {
              auto it = blob_desc.find(inp);
              if (it == blob_desc.end() || it->second.unknown_shape()) {
                blob_desc[inp] = sum_shape;
                if (sum_shape.dims_size() == 0) {
                  // Match later with the output
                  unmatched_sum_blobs[inp] = op.output(0);
                }
              }
            }
          }

          if (op.type() == "Reshape" && op.is_gradient_op()) {
            CAFFE_ENFORCE(reshape_cache.find(op.input(1)) != reshape_cache.end());
            TensorShape cached = reshape_cache[op.input(1)];
            blob_desc[op.output(0)] = cached;
            TensorShape dims;
            dims.add_dims(cached.dims_size());
            dims.set_data_type(TensorProto_DataType_INT64);
            blob_desc[op.output(1)] = dims;
            continue;
          } else if (
              op.type() == "Split" && op.input_size() == 2 && op.is_gradient_op()) {
            CAFFE_ENFORCE(split_cache.find(op.input(1)) != split_cache.end());
            vector<TensorShape> cached = split_cache[op.input(1)];
            CAFFE_ENFORCE_EQ(op.output_size(), cached.size());
            for (size_t i = 0; i < cached.size(); i++) {
              blob_desc[op.output(i)] = cached[i];
            }
            continue;
          }

          std::vector<TensorShape> out;
          try {
            out = op_schema->InferTensor(op, input_desc);
            if (op.is_gradient_op() && out.size()) {
              // Special handling for gradient ops. We can assume gradients
              // are of same size as the corresponding variables. This is bit
              // ugly to base on string matching, but we don't have the connection
              // between variable and its gradient specified

              CaffeMap<string, string> grads_to_params =
                  GradientMakerBase::MatchGradsToParams(op);

              for (size_t i = 0; i < out.size(); i++) {
                if (out[i].unknown_shape()) {
                  std::string gradout = op.output(i);

                  if (grads_to_params.find(gradout) != grads_to_params.end()) {
                    std::string var = grads_to_params[gradout];
                    if (blob_desc.find(var) != blob_desc.end()) {
                      out[i] = blob_desc[var];
                    }
                  }
                }
              }
            }

            if (op.type() == "Reshape") {
              // Reshape stores the original input shape to its second output
              // blob. We need this for gradient reshape.
              reshape_cache[op.output(1)] = input_desc[0];
            } else if (op.type() == "Concat") {
              // Split needs the input sizes from Concat.
              split_cache[op.output(1)] = input_desc;
            }

          } catch (::caffe2::EnforceNotMet& enf) {
            LOG(ERROR) << "Shape inference error: " << enf.what();
            LOG(ERROR) << "Operator: " << ProtoDebugString(op) << std::endl;
            LOG(ERROR) << "Returning empty results.";

            TensorShapes tps;
            return tps;
          }

          if (out.size() != (unsigned)op.output_size()) {
            if (op.type() == "Slice") {
              CAFFE_ENFORCE(
                  out.size() == 0,
                  "For Slice operator, either shape of all output blobs are "
                  "inferred or shape of none can be inferred.");
            } else {
              CAFFE_THROW(
                  "Invalid shape inference for operator ",
                  op.type(),
                  " Expected ",
                  op.output_size(),
                  " outputs, but got ",
                  out.size());
            }
          } else {
            for (size_t i = 0; i < out.size(); i++) {
              blob_desc[op.output(i)] = out[i];
            }
          }
        } // net.ops

        for (auto& unmatched : unmatched_sum_blobs) {
          if (blob_desc.find(unmatched.second) != blob_desc.end()) {
            blob_desc[unmatched.first] = blob_desc[unmatched.second];
          }
        }

      } // nets
      TensorShapes tps;
      for (auto kv : blob_desc) {
        TensorShape& tp = kv.second;
        TensorShape* tpnew = tps.add_shapes();
        tpnew->CopyFrom(tp);
        tpnew->set_name(kv.first);
      }
      return tps;
    */
}

#[inline] pub fn load_int_8tensor_info_of_blob(
    scale:  *mut Vec<f32>,
    offset: *mut Vec<f32>,
    axis:   *mut u32,
    b:      *const Blob)  
{
    todo!();
    /*
        const int8::Int8TensorCPU* int8_tensor =
          static_cast<const int8::Int8TensorCPU*>(b->GetRaw());
      scale->clear();
      offset->clear();
      scale->push_back(int8_tensor->scale);
      offset->push_back(int8_tensor->zero_point);
      *axis = 1;
    */
}

#[inline] pub fn get_tensor_shape_of_blob(b: *const Blob) -> TensorShape {
    
    todo!();
    /*
        TensorShape tp;
    #ifndef C10_MOBILE
      auto function_ptr =
          ExternalTensorFunctionsBaseRegistry()->Create(b->meta().id());
      if (function_ptr != nullptr) {
        // This is dnnlowp tensor and we cant deal with it using regular path
        auto dtype = function_ptr->GetExternalTensorType(b->GetRaw());
        tp.set_data_type(TypeMetaToDataType(dtype));

        size_t _capacity;
        DeviceOption _device;
        auto dshape =
            function_ptr->GetExternalTensorInfo(b->GetRaw(), &_capacity, &_device);
        for (auto d : dshape) {
          tp.add_dims(d);
        }
        return tp;
      }
    #endif

      TypeCall type_fun = GetTypeCallFunction(b->meta().id());
      TensorInfoCall tensor_info_fun = GetTensorInfoFunction(b->meta().id());
      if (type_fun) {
        tp.set_data_type(TypeMetaToDataType(type_fun(b->GetRaw())));
      }
      if (tensor_info_fun) {
        size_t _capacity;
        DeviceOption _device;
        auto shape = tensor_info_fun(b->GetRaw(), &_capacity, &_device);
        for (auto d : shape) {
          tp.add_dims(d);
        }
      } else {
        tp.set_unknown_shape(true);
      }
      return tp;
    */
}

#[inline] pub fn infer_blob_shapes_and_types_from_workspace(
    ws:   *mut Workspace,
    nets: &Vec<*mut NetDef>)  
{
    
    todo!();
    /*
        CaffeMap<string, TensorShape> blob_desc;
      // Populate shapes from workplace
      const std::vector<string>& ws_blobs = ws->Blobs();
      for (const auto& s : ws_blobs) {
        Blob* b = ws->GetBlob(s);
        TensorShape tp = GetTensorShapeOfBlob(b);
        blob_desc[s] = tp;
      }
      return InferBlobShapesAndTypes(blob_desc, nets);
    */
}

#[inline] pub fn infer_blob_shapes_and_types_from_map(
    blob_dimensions: &HashMap<String,Vec<i64>>,
    nets: &Vec<*mut NetDef>)  
{
    todo!();
    /*
        CaffeMap<string, TensorShape> blob_desc;
      // Populate shapes from known blobs
      for (const auto& blob : blob_dimensions) {
        TensorShape tp;
        for (auto d : blob.second) {
          CAFFE_ENFORCE_GE(d, 0, blob.first);
          tp.add_dims(d);
        }
        blob_desc[blob.first] = tp;
      }
      return InferBlobShapesAndTypes(blob_desc, nets);
    */
}


#[inline] pub fn infer_blob_shapes_and_types_from_map_with_blob_types(
    blob_dimensions: &HashMap<String,Vec<i64>>,
    blob_types: &HashMap<String,TensorProto_DataType>,
    nets: &Vec<*mut NetDef>)  
{
    todo!();
    /*
        CaffeMap<string, TensorShape> blob_desc;
      // Populate shapes from known blobs
      for (const auto& blob : blob_dimensions) {
        TensorShape tp;
        for (auto d : blob.second) {
          CAFFE_ENFORCE_GE(d, 0, blob.first);
          tp.add_dims(d);
        }
        auto blob_type = blob_types.find(blob.first);
        if (blob_type == blob_types.end()) {
          LOG(WARNING) << "Missing type of " << blob.first
                       << "; assuming to be UNDEFINED";
          tp.set_data_type(TensorProto_DataType_UNDEFINED);
        } else {
          tp.set_data_type(blob_type->second);
        }
        blob_desc[blob.first] = tp;
      }
      return InferBlobShapesAndTypes(blob_desc, nets);
    */
}


#[inline] pub fn validate_tensor_devices<Context>(
    op: &mut OperatorStorage,
    op_def: &OperatorDef) -> HashMap<String,(DeviceOption,DeviceOption)> 
{
    todo!();
    /*
        std::map<string, std::pair<DeviceOption, DeviceOption>> mismatches;
      DeviceOption op_device = op_def.device_option();

    #ifndef CAFFE2_NO_OPERATOR_SCHEMA
      // Check from op schema if this op is used for crossing devices
      auto op_schema = OpSchemaRegistry::Schema(op_def.type());
      if (op_schema != nullptr) {
        if (op_schema->inputs_can_cross_devices()) {
          return mismatches;
        }
      }
    #endif // CAFFE2_NO_OPERATOR_SCHEMA

      auto Check = [&](const Blob& blob, std::string blob_name) {
        TensorInfoCall tensor_info_fun = GetTensorInfoFunction(blob.meta().id());
        if (tensor_info_fun) {
          size_t _capacity;
          DeviceOption blob_device;
          tensor_info_fun(
              const_cast<Blob&>(blob).GetRaw(), &_capacity, &blob_device);

          if ((blob_device.device_type() == PROTO_CUDA ||
               blob_device.device_type() == PROTO_HIP) &&
              blob_device.device_id() != op_device.device_id()) {
            mismatches[blob_name] = std::make_pair(op_device, blob_device);
          }
        }
      };

      // Check that inputs have same device type as the op
      for (int i = 0; i < op.InputSize(); i++) {
        Check(op.InputBlob(i), op_def.input(i));
      }
      for (int i = 0; i < op.OutputSize(); i++) {
        Check(*op.OutputBlob(i), op_def.output(i));
      }
      return mismatches;
    */
}

/// Get a set of registered operator names
#[inline] pub fn get_registered_operators() -> HashSet<String> {
    
    todo!();
    /*
        std::set<std::string> all_keys;

      // CPU operators
      for (const auto& name : CPUOperatorRegistry()->Keys()) {
        all_keys.emplace(name);
      }
      // CUDA operators
      for (const auto& name : CUDAOperatorRegistry()->Keys()) {
        all_keys.emplace(name);
      }

      // HIP operators
      for (const auto& name : HIPOperatorRegistry()->Keys()) {
        all_keys.emplace(name);
      }

      return all_keys;
    */
}

pub fn operator_logger_default(def: &OperatorDef) { }

lazy_static!{
    static ref operator_logger: fn(def: &OperatorDef) -> () = operator_logger_default;
}

/// Operator logging capabilities
pub fn set_operator_logger(tracer: fn(def: &OperatorDef) -> ()) {
    todo!();
    /*
       OperatorLogger = tracer;
       */
}

pub fn get_operator_logger() -> fn(def: &OperatorDef) -> () {
    todo!();
    /*
       return OperatorLogger;
       */
}

#[cfg(c10_mobile)]
define_typed_registry!{
    ExternalTensorFunctionsBaseRegistry,
    TypeIdentifier,
    ExternalTensorFunctionsBase,
    Box
}
