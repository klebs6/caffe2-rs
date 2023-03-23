crate::ix!();

pub struct MIOPENActivationOp<MIOPENActivationMode> {
    base:                        MIOPENActivationOpBase,
    phantomMIOPENActivationMode: PhantomData<MIOPENActivationMode>,
}
