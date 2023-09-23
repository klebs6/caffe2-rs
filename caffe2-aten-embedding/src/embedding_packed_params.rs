crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/embedding_packed_params.h]

pub trait EmbeddingPackedParamsBaseInterface:
TorchJitCustomClassHolder
+ EmbeddingbagByte
+ Embeddingbag4bit
+ Unpack
+ BitRate
+ Version {}

pub trait EmbeddingbagByte {
    
    fn embeddingbag_byte(&mut self, 
        indices:                    &Tensor,
        offsets:                    &Option<Tensor>,
        pruned_weights:             bool,
        per_sample_weights:         &Option<Tensor>,
        compressed_indices_mapping: &Option<Tensor>,
        include_last_offset:        bool,
        is_embedding_op:            bool) -> Tensor;
}

pub trait Embeddingbag4bit {
    
    fn embeddingbag_4bit(&mut self, 
        indices:                    &Tensor,
        offsets:                    &Option<Tensor>,
        pruned_weights:             bool,
        per_sample_weights:         &Option<Tensor>,
        compressed_indices_mapping: &Option<Tensor>,
        include_last_offset:        bool) -> Tensor;
}

pub trait Unpack {
    
    fn unpack(&mut self) -> Tensor;
}

pub trait BitRate {
    
    fn bit_rate(&self) -> i64;
}

pub trait Version {

    fn version(&self) -> i64;
}
