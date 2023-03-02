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
pub struct AsyncNetBarrierOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
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

pub struct AsyncBarrierOp {}

impl AsyncBarrierOp {
    
    /*
      | This is a pretty much no-op operator,
      | since it's only purposes is make sure that
      | async_scheduling will schedule certian
      | operations earlier than others.
      |
      | Example where this operator can work
      | well - mixture of data-parallel and model
      | parallel training, where one wants to force
      | that all copies are started before
      | data-parallel part starts.
      */
    #[inline] pub fn run_on_device(&mut self) -> bool {
        true
    }

    #[inline] pub fn async_barrier_op_dev_infer(def: &OperatorDef) -> (Vec<DeviceOption>,Vec<DeviceOption>) {
        
        todo!();

        /*
        let op_device = match def.has_device_option() {
            true   => def.device_option,
            false  => panic!("looking for `device_option`"), //device_option
        };

        let helper: ArgumentHelper = ArgumentHelper::new(def);

        let cross_device = ArgumentHelper::get_single_argument::<i32>(def, "cross_device", 0);

        let mut opt = Vec::<DeviceOption>::default();

        for i in 0..def.input.len() {

            if cross_device == 1 {
                let mut dev = DeviceOption::default();
                dev.set_device_type(op_device.device_type());
                dev.set_device_id(i.try_into().unwrap());
                opt.push(dev);
            } else {
                opt.push(op_device);
            }
        }

        (opt, opt)
        */
    }
}
