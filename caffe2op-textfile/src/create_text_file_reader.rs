crate::ix!();

/**
  | Create a text file reader. Fields are
  | delimited by <TAB>.
  |
  */
pub struct CreateTextFileReaderOp {
    storage:      OperatorStorage,
    context:      CPUContext,

    filename:     String,
    num_passes:   i32,
    field_types:  Vec<i32>,
}

register_cpu_operator!{CreateTextFileReader, CreateTextFileReaderOp}

num_inputs!{CreateTextFileReader, 0}

num_outputs!{CreateTextFileReader, 1}

outputs!{CreateTextFileReader, 
    0 => ("handler", "Pointer to the created TextFileReaderInstance.")
}

args!{CreateTextFileReader, 
    0 => ("filename",    "Path to the file."),
    1 => ("num_passes",  "Number of passes over the file."),
    2 => ("field_types", "List with type of each field. Type enum is found at core.DataType.")
}

scalar_type!{CreateTextFileReader, TensorProto::UNDEFINED}

no_gradient!{CreateTextFileReader}

impl CreateTextFileReaderOp {
    
    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            filename_(GetSingleArgument<string>("filename", "")),
            numPasses_(GetSingleArgument<int>("num_passes", 1)),
            fieldTypes_(GetRepeatedArgument<int>("field_types")) 

        CAFFE_ENFORCE(fieldTypes_.size() > 0, "field_types arg must be non-empty");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            *OperatorStorage::Output<std::unique_ptr<TextFileReaderInstance>>(0) =
            std::unique_ptr<TextFileReaderInstance>(new TextFileReaderInstance(
                {'\n', '\t'}, '\0', filename_, numPasses_, fieldTypes_));
        return true;
        */
    }
}
