crate::ix!();

/**
  | Read a batch of rows from the given text
  | file reader instance.
  | 
  | Expects the number of fields to be equal
  | to the number of outputs.
  | 
  | Each output is a 1D tensor containing
  | the values for the given field for each
  | row.
  | 
  | When end of file is reached, returns
  | empty tensors.
  |
  */
pub struct TextFileReaderInstance<'a> {
    file_reader:       FileReader<'a>,
    tokenizer:         BufferedTokenizer,
    field_types:       Vec<i32>,
    field_metas:       Vec<TypeMeta>,
    field_byte_sizes:  Vec<usize>,
    rows_read:         usize, // {0};

    /**
      | hack to guarantee thread-safeness of the
      | read op
      |
      | TODO(azzolini): support multi-threaded
      | reading.
      */
    global_mutex:      parking_lot::RawMutex,
}

register_cpu_operator!{TextFileReaderRead, TextFileReaderReadOp}

num_inputs!{TextFileReaderRead, 1}

num_outputs!{TextFileReaderRead, (1,INT_MAX)}

inputs!{TextFileReaderRead, 
    0 => ("handler", "Pointer to an existing TextFileReaderInstance.")
}

args!{TextFileReaderRead, 
    0 => ("batch_size", "Maximum number of rows to read.")
}

no_gradient!{TextFileReaderRead}

impl<'a> TextFileReaderInstance<'a> {
    
    pub fn new(
        delims:     &Vec<u8>,
        escape:     u8,
        filename:   &String,
        num_passes: i32,
        types:      &Vec<i32>) -> Self {

        todo!();
        /*
            : fileReader(filename),
            tokenizer(Tokenizer(delims, escape), &fileReader, numPasses),
            fieldTypes(types) 

            for (const auto dt : fieldTypes) {
                fieldMetas.push_back(
                    DataTypeToTypeMeta(static_cast<TensorProto_DataType>(dt)));
                fieldByteSizes.push_back(fieldMetas.back().itemsize());
            }
        */
    }
}
