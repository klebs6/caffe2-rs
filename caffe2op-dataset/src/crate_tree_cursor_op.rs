crate::ix!();

/**
 | Creates a cursor to iterate through a list of
 | tensors, where some of those tensors contain the
 | lengths in a nested schema. The schema is
 | determined by the `fields` arguments.
 |
 | For example, to represent the following schema:
 |
 |   Struct(
 |       a=Int(),
 |       b=List(List(Int)),
 |       c=List(
 |           Struct(
 |              c1=String,
 |              c2=List(Int),
 |           ),
 |       ),
 |   )
 |
 | the field list will be:
 |   [
 |       "a",
 |       "b:lengths",
 |       "b:values:lengths",
 |       "b:values:values",
 |       "c:lengths",
 |       "c:c1",
 |       "c:c2:lengths",
 |       "c:c2:values",
 |   ]
 |
 | And for the following instance of the struct:
 |
 |   Struct(
 |       a=3,
 |       b=[[4, 5], [6, 7, 8], [], [9]],
 |       c=[
 |           Struct(c1='alex', c2=[10, 11]),
 |           Struct(c1='bob', c2=[12]),
 |       ],
 |   )
 |
 | The values of the fields will be:
 |   {
 |       "a": [3],
 |       "b:lengths": [4],
 |       "b:values:lengths": [2, 3, 0, 1],
 |       "b:values:values": [4, 5, 6, 7, 8, 9],
 |       "c:lengths": [2],
 |       "c:c1": ["alex", "bob"],
 |       "c:c2:lengths": [2, 1],
 |       "c:c2:values", [10, 11, 12],
 |   }
 |
 | In general, every field name in the format
 | "{prefix}:lengths" defines a domain "{prefix}",
 | and every subsequent field in the format
 | "{prefix}:{field}" will be in that domain, and the
 | length of the domain is provided for each entry of
 | the parent domain. In the example, "b:lengths"
 | defines a domain of length 4, so every field under
 | domain "b" will have 4 entries. The "lengths"
 | field for a given domain must appear before any
 | reference to that domain.
 |
 | Returns a pointer to an instance of the Cursor,
 | which keeps the current offset on each of the
 | domains defined by `fields`. Cursor also ensures
 | thread-safety such that ReadNextBatch and
 | ResetCursor can be used safely in parallel.
 |
 | A cursor does not contain data per se, so calls to
 | ReadNextBatch actually need to pass a list of
 | blobs containing the data to read for each one of
 | the fields.
 |
 */
pub struct CreateTreeCursorOp {
    storage: OperatorStorage,
    context: CPUContext,
    fields:  Vec<String>,
}

num_inputs!{CreateTreeCursor, 0}

num_outputs!{CreateTreeCursor, 1}

outputs!{CreateTreeCursor, 
    0 => ("cursor", "A blob pointing to an instance of a new TreeCursor.")
}

args!{CreateTreeCursor, 
    0 => ("fields", "A list of strings each one representing a field of the dataset.")
}

impl CreateTreeCursorOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            fields_(OperatorStorage::GetRepeatedArgument<std::string>("fields"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            *OperatorStorage::Output<std::unique_ptr<TreeCursor>>(0) =
            std::unique_ptr<TreeCursor>(new TreeCursor(TreeIterator(fields_)));
        return true;
        */
    }
}

