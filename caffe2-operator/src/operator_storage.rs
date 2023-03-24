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

