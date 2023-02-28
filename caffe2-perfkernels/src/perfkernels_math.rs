crate::ix!();

/**
 | Returns the quantized and compressed values of
 | floating inputs
 |
 | The "fused" representation stores the
 | [bitwidth][tail][min][max] with the quantized
 | data in one array. Since we store 8/bitwidth
 | quantized data in one byte, the last buckets of
 | some bytes may have unused bits. There are totally
 | tail buckets are unused.
 |
 | We encode *bitwidth* and *tail* at the
 | beginning, following by 32-bit floating data
 | respresenting min and max.
 |
 | | bitwidth | tail | min | max | ... int8 data ... |
 | |    1B    |  1B  |  4B |  4B | ...output_data....|
 |
 | In output_data: the b-th bucket of the i-th byte
 | stores the i-th data of the b-th segment of
 | input row
 */
#[inline] pub fn quantize_and_compress(
    input_data:    *const f32,
    output_data:   *mut u8,
    input_size:    u64,
    bitwidth:      u64,
    random:        bool,
    random_buffer: *const f32)  {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn decompress_and_dequantize(
    input_data:  *const u8,
    output_data: *mut f32,
    input_size:  u64)  {
    
    todo!();
    /*
    
    */
}
