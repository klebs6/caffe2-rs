crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/templates/RegisterSchema.cpp]

lazy_static!{
    /*
    TORCH_LIBRARY(aten, m) {
      ${schema_registrations};

      // String Ops
      // Implementations located in torch/csrc/jit/runtime/register_prim_ops.cpp
      m.def(TORCH_SELECTIVE_SCHEMA("splitlines(str self, bool keepends=False) -> str[]"));
      m.def(TORCH_SELECTIVE_SCHEMA(
          "slice.str(str string, int? start=0, int? end=9223372036854775807, int step=1) -> str"));
      m.def(TORCH_SELECTIVE_SCHEMA("isupper(str self) -> bool"));
      m.def(TORCH_SELECTIVE_SCHEMA("islower(str self) -> bool"));
      m.def(TORCH_SELECTIVE_SCHEMA("capitalize(str self) -> str"));
      m.def(TORCH_SELECTIVE_SCHEMA("title(str self) -> str"));
      m.def(TORCH_SELECTIVE_SCHEMA("center(str self, int width, str fillchar=' ') -> str"));
      m.def(TORCH_SELECTIVE_SCHEMA("count(str self, str substr, int start=0, int end=-1) -> int"));
      m.def(TORCH_SELECTIVE_SCHEMA("endswith(str self, str substr, int start=0, int end=-1) -> bool"));
      m.def(TORCH_SELECTIVE_SCHEMA("startswith(str self, str substr, int start=0, int end=-1) -> bool"));
      m.def(TORCH_SELECTIVE_SCHEMA("expandtabs(str self, int tabsize=8) -> str"));
      m.def(TORCH_SELECTIVE_SCHEMA("find(str self, str substr, int start=0, int end=-1) -> int"));
      m.def(TORCH_SELECTIVE_SCHEMA("rfind(str self, str substr, int start=0, int end=-1) -> int"));
      m.def(TORCH_SELECTIVE_SCHEMA("index.str(str self, str substr, int start=0, int end=-1) -> int"));
      m.def(TORCH_SELECTIVE_SCHEMA("rindex(str self, str substr, int start=0, int end=-1) -> int"));
      m.def(TORCH_SELECTIVE_SCHEMA("isidentifier(str self) -> bool"));
      m.def(TORCH_SELECTIVE_SCHEMA("istitle(str self) -> bool"));
      m.def(TORCH_SELECTIVE_SCHEMA("isprintable(str self) -> bool"));
      m.def(TORCH_SELECTIVE_SCHEMA("ljust(str self, int width, str fillchar=' ') -> str"));
      m.def(TORCH_SELECTIVE_SCHEMA("rjust(str self, int width, str fillchar=' ') -> str"));
      m.def(TORCH_SELECTIVE_SCHEMA("zfill(str self, int width) -> str"));
      m.def(TORCH_SELECTIVE_SCHEMA("lstrip(str self, str chars=' \\n\\t\\f\\v') -> str"));
      m.def(TORCH_SELECTIVE_SCHEMA("rstrip(str self, str chars=' \\n\\t\\f\\v') -> str"));
      m.def(TORCH_SELECTIVE_SCHEMA("strip(str self, str chars=' \\n\\t\\f\\v') -> str"));
      m.def(TORCH_SELECTIVE_SCHEMA("replace(str self, str old, str new, int max=-1) -> str"));
      m.def(TORCH_SELECTIVE_SCHEMA("partition(str self, str separator) -> (str, str, str)"));
      m.def(TORCH_SELECTIVE_SCHEMA("rpartition(str self, str separator) -> (str, str, str)"));
      m.def(TORCH_SELECTIVE_SCHEMA("split.str(str self, str? separator=None, int max=-1) -> str[]"));
      m.def(TORCH_SELECTIVE_SCHEMA("rsplit(str self, str separator=' ', int max=-1) -> str[]"));
      m.def(TORCH_SELECTIVE_SCHEMA("join(str self, str[] values) -> str"));

      // Distributed Ops
      // Implementations located in torch/csrc/jit/runtime/register_distributed_ops.cpp
      m.def("get_gradients(int context_id) -> Dict(Tensor, Tensor)");
    }
    */
}
