crate::ix!();

/**
 | Embedding lookup with reduction.
 |
 | `input` of size data_size * (block_size + 8B)
 | `indices` of size index_size
 | `offsets` of size output_size
 | `weights` nullptr or array of size index_size
 | `out` of size output_size * block_size
 |
 | Note that block_size should be the number of
 | quantized values per row in the data,
 |  i.e. excluding the scale and bias. The total
 |  (fused) block size is assumed to be this
 |  block_size, plus 4 bytes for scale and 4 bytes
 |  for bias.
 |
 | Behavior is roughly equivalent to pseudocode:
 |
 | pos = 0
 | fused_block_size = block_size + 8B // quantized values and scale and bias
 | for (i = 0..output_size-1)
 |   for (k = 0..block_size-1)
 |     out[i*block_size + k] = 0
 |   start_offset = offsets[i]
 |   end_offset = i == output_size-1 ? index_size : offsets[i+1] - 1
 |   length = end_offset - start_offset
 |   for (j = start_offset..end_offset)
 |     for (k = 0..block_size-1)
 |       out[i*block_size + k] += input[indices[pos]*(fused_block_size) + k] *
 |           (weights ? weights[IS_WEIGHT_POSITIONAL ? j : pos] : 1.0)
 |     pos += 1
 |   if (normalize_weights && length > 0)
 |     for (k = 0..block_size-1)
 |       out[i*block_size + k] /= length
 |
 |
 |  bool IS_WEIGHT_POSITIONAL = false>
 |  const float* weights, // optional, can be null for non-weighted sum
 */
#[inline] pub fn fused_8bit_rowwise_embedding_lookup_idx<IndexType, InType, OutType, const IS_WEIGHT_POSITIONAL: bool>(
    block_size:           i64,
    output_size:          i64,
    index_size:           i64,
    data_size:            i64,
    input:                *const InType,
    indices:              *const IndexType,
    offsets:              *const IndexType,
    weights:              *const f32,
    normalize_by_lengths: bool,
    out:                  *mut OutType)  {

    todo!();
    /*
    
    */
}

/**
 | Base implementation does runtime dispatch for
 | each segment of reduction
 |
 | @return false if there is an out-of-bound error
 |
 |  const float* weights, // optional, can be null for sum reducer
 */
#[inline] pub fn fused_8bit_rowwise_embedding_lookup_generic_slow_idx<IndexType, InType, OutType, const IS_WEIGHT_POSITIONAL: bool>(
    block_size:           i64,
    output_size:          i64,
    index_size:           i64,
    data_size:            i64,
    input:                *const InType,
    indices:              *const IndexType,
    offsets:              *const IndexType,
    weights:              *const f32,
    normalize_by_lengths: bool,
    out:                  *mut OutType) -> bool {

    todo!();
    /*
        // block_size is the number of elements and fused_block_size is the size of
      // an entire row, including scale and bias.
      const auto scale_bias_offset = 8 / sizeof(InType);
      const int64_t fused_block_size = block_size + scale_bias_offset;
      int64_t current = 0;
      for (const auto m : c10::irange(output_size)) {
        memset(out, 0, sizeof(OutType) * block_size);
        if (current != offsets[m] - offsets[0]) {
          return false;
        }
        int64_t start_offset = offsets[m];
        int64_t end_offset = offsets[m + 1];
        int64_t length = end_offset - start_offset;
        for (const auto i : c10::irange(start_offset, end_offset)) {
          int64_t idx = indices[current];
          if (idx < 0 || idx >= data_size) {
            return false;
          }
    #ifdef __GNUC__
          if (current + 1 < index_size) {
            __builtin_prefetch(
                input + fused_block_size * indices[current + 1], 0, 1);
          }
    #endif // __GNUC__

          const float* scale_bias = reinterpret_cast<const float*>(
              input + fused_block_size * indices[current] + block_size);

          float weight = 1.0f;
          if (weights) {
            weight = weights[IS_WEIGHT_POSITIONAL ? i : current];
          }
          const float scale = weight * scale_bias[0];
          const float bias = weight * scale_bias[1];

          for (const auto j : c10::irange(block_size)) {
            out[j] += scale * input[fused_block_size * indices[current] + j] + bias;
          }

          ++current;
        }
        if (normalize_by_lengths && length) {
          float scale = 1.f / length;
          for (const auto j : c10::irange(block_size)) {
            out[j] *= scale;
          }
        }
        out += block_size;
      }
      return current == index_size;
    */
}

/// Proxy back to generic implementation
#[macro_export] macro_rules! fused_8bit_rowwise_embedding_idx_specialization {
    ($IndexType:ty, $OutType:ty) => {
        /*
        
          bool                                                                                      
              Fused8BitRowwiseEmbeddingLookupIdx_##IndexType##_uint8_t_##OutType##_false__base(     
                  const int64_t block_size,                                                         
                  const int64_t output_size,                                                        
                  const int64_t index_size,                                                         
                  const int64_t data_size,                                                          
                  const uint8_t* input,                                                             
                  const IndexType* indices,                                                         
                  const IndexType* offsets,                                                           
                  const float* weights,                                                             
                  bool normalize_by_lengths,                                                        
                  OutType* out) {                                                                   
            return Fused8BitRowwiseEmbeddingLookupGenericSlowIdx<                                   
                IndexType,                                                                          
                uint8_t,                                                                            
                OutType,                                                                            
                false>(                                                                             
                block_size,                                                                         
                output_size,                                                                        
                index_size,                                                                         
                data_size,                                                                          
                input,                                                                              
                indices,                                                                            
                offsets,                                                                            
                weights,                                                                            
                normalize_by_lengths,                                                               
                out);                                                                               
          }                                                                                         
          decltype(                                                                                 
              Fused8BitRowwiseEmbeddingLookupIdx_##IndexType##_uint8_t_##OutType##_false__base)     
              Fused8BitRowwiseEmbeddingLookupIdx_##IndexType##_uint8_t_##OutType##_false__avx2_fma; 
          bool Fused8BitRowwiseEmbeddingLookupIdx_##IndexType##_uint8_t_##OutType(                  
              const int64_t block_size,                                                             
              const int64_t output_size,                                                            
              const int64_t index_size,                                                             
              const int64_t data_size,                                                              
              const uint8_t* input,                                                                 
              const IndexType* indices,                                                             
              const IndexType* offsets,                                                               
              const float* weights,                                                                 
              bool normalize_by_lengths,                                                            
              OutType* out) {                                                                       
            const int32_t one = 1;                                                                  
            CAFFE_ENFORCE_EQ(                                                                       
                reinterpret_cast<const uint8_t*>(&one)[0],                                          
                1,                                                                                  
                "Fused8BitRowwiseEmbeddingLookup is not supported on this platform");               
            AVX2_FMA_DO(                                                                            
                Fused8BitRowwiseEmbeddingLookupIdx_##IndexType##_uint8_t_##OutType##_false,         
                block_size,                                                                         
                output_size,                                                                        
                index_size,                                                                         
                data_size,                                                                          
                input,                                                                              
                indices,                                                                            
                offsets,                                                                            
                weights,                                                                            
                normalize_by_lengths,                                                               
                out);                                                                               
            BASE_DO(                                                                                
                Fused8BitRowwiseEmbeddingLookupIdx_##IndexType##_uint8_t_##OutType##_false,         
                block_size,                                                                         
                output_size,                                                                        
                index_size,                                                                         
                data_size,                                                                          
                input,                                                                              
                indices,                                                                            
                offsets,                                                                            
                weights,                                                                            
                normalize_by_lengths,                                                               
                out);                                                                               
          }                                                                                         
          template <>                                                                               
          void Fused8BitRowwiseEmbeddingLookupIdx<IndexType, uint8_t, OutType, false>(              
              const int64_t block_size,                                                             
              const int64_t output_size,                                                            
              const int64_t index_size,                                                             
              const int64_t data_size,                                                              
              const uint8_t* input,                                                                 
              const IndexType* indices,                                                             
              const IndexType* offsets,                                                               
              const float* weights,                                                                 
              bool normalize_by_lengths,                                                            
              OutType* out) {                                                                       
            bool success =                                                                          
                Fused8BitRowwiseEmbeddingLookupIdx_##IndexType##_uint8_t_##OutType(                 
                    block_size,                                                                     
                    output_size,                                                                    
                    index_size,                                                                     
                    data_size,                                                                      
                    input,                                                                          
                    indices,                                                                        
                    offsets,                                                                        
                    weights,                                                                        
                    normalize_by_lengths,                                                           
                    out);                                                                           
            if (success) {                                                                          
              return;                                                                               
            }                                                                                       
            int64_t current = 0;                                                                    
            for (int m = 0; m < output_size; ++m) {                                                 
              for (int64_t i = offsets[m]; i < offsets[m + 1]; ++i) {                               
                CAFFE_ENFORCE_LT(current, index_size);                                              
                IndexType idx = indices[current];                                                   
                CAFFE_ENFORCE(                                                                      
                    0 <= idx && idx < data_size,                                                    
                    "Index ",                                                                       
                    current,                                                                        
                    " is out of bounds: ",                                                          
                    idx,                                                                            
                    ", range 0 to ",                                                                
                    data_size);                                                                     
                ++current;                                                                          
              }                                                                                     
            }                                                                                       
            CAFFE_ENFORCE_EQ(                                                                       
                current,                                                                            
                index_size,                                                                         
                "Your input seems to be incorrect: the sum of lengths values should be "            
                "the size of the indices tensor, but it appears not.");                             
          }
        */
    }
}

fused_8bit_rowwise_embedding_idx_specialization!{i32, f32}
fused_8bit_rowwise_embedding_idx_specialization!{i64, f32}
