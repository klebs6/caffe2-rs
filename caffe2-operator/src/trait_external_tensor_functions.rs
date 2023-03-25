crate::ix!();

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
