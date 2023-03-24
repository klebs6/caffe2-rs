crate::ix!();

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
