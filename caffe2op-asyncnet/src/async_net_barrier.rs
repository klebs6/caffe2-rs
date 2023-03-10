crate::ix!();

/**
 | This is a pretty much no-op operator, since it's
 | only purposes is make sure that async_scheduling
 | will schedule certian operations earlier than
 | others.
 |
 | Examjple where this operator can work well - mixture
 | of data-parallel and model- parallel training,
 | where one wants to force that all copies are
 | started before data-parallel part starts.
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AsyncNetBarrierOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{AsyncNetBarrier, (1,INT_MAX)}

num_outputs!{AsyncNetBarrier, (1,INT_MAX)}

args!{AsyncNetBarrier, 
    0 => ("cross_device", "Specifies either inputs should be across different devices in dev inference options")
}

identical_type_and_shape!{AsyncNetBarrier}

inputs_can_cross_devices!{AsyncNetBarrier}

allow_one_to_one_inplace!{AsyncNetBarrier}

device_inference_function!{AsyncNetBarrier, asyncBarrierOpDevInfer }

should_not_do_gradient!{AsyncNetBarrier}

register_cpu_operator!{
    AsyncNetBarrier, 
    AsyncNetBarrierOp<CPUContext>
}
