crate::ix!();

/** Crashes the program. Use for testing */
#[cfg(target_os = "linux")]
pub struct CrashOp {
    context: Operator<CPUContext>,

}

#[cfg(target_os = "linux")]
impl CrashOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
           : Operator<CPUContext>(operator_def, ws)
           */
    }

    #[inline] pub fn run_on_device(&mut self) -> bool {

        todo!();
        /*
           raise(SIGABRT);
           return true;
           */
    }
}

num_inputs!{Crash, 0}

num_outputs!{Crash, 0}

register_cpu_operator!{Crash, CrashOp}
