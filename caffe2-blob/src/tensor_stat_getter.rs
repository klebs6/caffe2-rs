crate::ix!();

///----------------------------------------
pub struct TensorStatGetter {
    base: dyn BlobStatGetter,
}

impl TensorStatGetter {
    
    #[inline] pub fn size_bytes(&self, blob: &Blob) -> usize {
        
        todo!();
        /*
            const auto& tensor = blob.Get<Tensor>();
        auto nbytes = tensor.nbytes();
        if (nbytes > 0 && tensor.IsType<std::string>()) {
          const auto* data = tensor.data<std::string>();
          for (int i = 0; i < tensor.numel(); ++i) {
            nbytes += data[i].size();
          }
        }
        return nbytes;
        */
    }
}

register_blob_stat_getter!{Tensor, TensorStatGetter}
