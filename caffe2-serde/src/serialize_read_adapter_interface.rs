crate::ix!();

/// this is the interface for the (file/stream/memory) reader in
/// PyTorchStreamReader. with this interface, we can extend the support
/// besides standard istream
pub trait ReadAdapterInterface {

    fn size(&self) -> usize;

    fn read(&self,
        pos:  u64,
        buf:  *mut c_void,
        n:    usize,
        what: *const u8) -> usize;
}
