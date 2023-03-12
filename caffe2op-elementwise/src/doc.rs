crate::ix!();

pub const kBroadcastDoc: &'static str = "
| If necessary the right-hand-side argument will be
| broadcasted to match the shape of
| left-hand-side argument. When broadcasting is
| specified, the second tensor can either be of
| size 1 (a scalar value), or having its shape
| as a contiguous subset of the first tensor's
| shape. The starting of the mutually equal
| shape is specified by the argument \"axis\",
| and if it is not set, suffix matching is
| assumed. 1-dim expansion doesn't work yet.
|
| For example, the following tensor shapes are
| supported (with broadcast=1):
|
| ```
|   shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
|   shape(A) = (2, 3, 4, 5), shape(B) = (5,)
|   shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
|   shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
|   shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
| ```
| Argument `broadcast=1` needs to be passed to
| enable broadcasting.
|
| Github Links:
|
| - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc
";

    
pub fn math_doc_generator(
    name: &str, 
    extra: &str) -> fn(schema: &OpSchema) -> () 
{
    todo!();
    /*
      return [=](OpSchema& schema) {
        string doc = R"DOC(
    Performs element-wise binary {name} (with limited broadcast support).
    {broadcast_doc}

    {extra}
    )DOC";
        c10::ReplaceAll(doc, "{name}", name);
        c10::ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
        c10::ReplaceAll(doc, "{extra}", extra);
        schema.SetDoc(doc);
        schema.Arg(
            "broadcast", "*(type: int; default: 0)* Pass 1 to enable broadcasting");
        schema.Arg("axis", "*(type: int; default: -1)* Axis to concatenate on.");
        schema.Input(
            0,
            "A",
            "*(type: Tensor`<float>`)* First operand, should share the type with the second operand.");
        schema.Input(
            1,
            "B",
            "*(type: Tensor`<float>`)* Second operand. With broadcasting can be of smaller size than A. "
            "If broadcasting is disabled it should be of the same size as A.");
        schema.Output(
            0,
            "C",
            "*(type: Tensor`<float>`)* Output tensor with same dimensions and type as A.");
      };

    */
}

#[inline] pub fn comparison_doc_generator(
    name:  *const u8,
    desc:  *const u8,
    extra: *const u8) -> fn(_u0: &mut OpSchema) -> c_void {
    
    todo!();
    /*
        return [=](OpSchema& schema) {
        string doc = R"DOC(
    Performs element-wise {desc} comparison **{name}** (with limited broadcast support).

    {broadcast_doc}

    {extra}
    )DOC";
        c10::ReplaceAll(doc, "{name}", name);
        c10::ReplaceAll(doc, "{desc}", desc);
        c10::ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
        c10::ReplaceAll(doc, "{extra}", extra);
        schema.SetDoc(doc);
        schema.Arg( "broadcast", "*(type: int; default: 0)* Pass 1 to enable broadcasting.");
        schema.Arg( "axis", "*(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.");
        schema.Input( 0, "A", "*(type: Tensor`<bool>`)* First operand, should share the type with the second operand.");
        schema.Input( 1, "B", "*(type: Tensor`<bool>`)* Second operand. With broadcasting can be of smaller size than `A`. If broadcasting is disabled it should be of the same size.");
        schema.Output( 0, "C", "*(type: Tensor`<bool>`)* Output tensor with same dimensions as `A`.");
      };
    */
}

#[inline] pub fn logical_doc_generator(
    name: *const u8, 
    extra: *const u8) -> fn(_u0: &mut OpSchema) -> c_void 
{
    todo!();
    /*
        return [=](OpSchema& schema) {
        string doc = R"DOC(
    Performs element-wise logical operation **{name}** (with limited broadcast support).
    Both input operands should be of type `bool`.

    {broadcast_doc}

    {extra}
        )DOC";
        c10::ReplaceAll(doc, "{name}", name);
        c10::ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
        c10::ReplaceAll(doc, "{extra}", extra);
        schema.SetDoc(doc);
        schema.Arg(
            "broadcast",
            "*(type: int; default: 0)* Pass 1 to enable broadcasting.");
        schema.Arg(
            "axis",
            "*(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.");
        schema.Input(0, "A", "*(type: Tensor`<bool>`)* First operand.");
        schema.Input(
            1,
            "B",
            "*(type: Tensor`<bool>`)* Second operand. With broadcasting can be of smaller size than `A`. "
            "If broadcasting is disabled it should be of the same size.");
        schema.Output(
            0,
            "C",
            "*(type: Tensor`<bool>`)* Output tensor of booleans. Has same dimensions as input `A`.");
      };
    */
}

#[inline] pub fn bitwise_doc_generator(
    name: *const u8) -> fn(_u0: &mut OpSchema) -> c_void 
{
    todo!();
    /*
        return [=](OpSchema& schema) {
        string doc = R"DOC(
    Performs element-wise bitwise operation `{name}` (with limited broadcast support).
    Both input operands should be of type `bool`.
    {broadcast_doc})DOC";
        c10::ReplaceAll(doc, "{name}", name);
        c10::ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
        schema.SetDoc(doc);
        schema.Arg(
            "broadcast",
            "*(type: int; default: 0)* Pass 1 to enable broadcasting.");
        schema.Arg(
            "axis",
            "*(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.");
        schema.Input(0, "A", "*(type: Tensor)* First operand.");
        schema.Input(
            1,
            "B",
            "*(type: Tensor)* Second operand. With broadcasting can be of smaller size than `A`. "
            "If broadcasting is disabled it should be of the same size.");
        schema.Output(
            0,
            "C",
            "*(type: Tensor)* Output tensor. Has same dimensions as input `A`.");
      };
    */
}

