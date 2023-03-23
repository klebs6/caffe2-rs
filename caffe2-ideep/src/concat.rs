crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPConcatOp {
    base:     IDEEPOperator,
    axis:     i32,
    add_axis: i32,
    fallback: FALLBACK_OP,
}

register_ideep_operator!{Concat, IDEEPConcatOp}

input_tags!{
    IDEEPConcatOp {
        Input0
    }
}

output_tags!{
    IDEEPConcatOp {
        Output,
        AxisInfo
    }
}

impl IDEEPConcatOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            fallback_(operator_def, ws) 

        CAFFE_ENFORCE(
          !(OperatorStorage::HasArgument("axis") && OperatorStorage::HasArgument("order")),
            "You shouldn't specify both the dim to concat, and the order "
            "in the case of 4-D images.");
        if (OperatorStorage::HasArgument("axis")) {
          axis_ = OperatorStorage::GetSingleArgument<int>("axis", -1);
          add_axis_ = OperatorStorage::GetSingleArgument<int>("add_axis", 0);
        } else {
          axis_ = 1;
          add_axis_ = 0;
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            bool fallback_to_cpu = false;
        vector<itensor> inputs_itensor;

        for (int i = 0; i < InputSize(); ++i) {
          if (OperatorStorage::InputBlob(i).template IsType<itensor>()) {
            auto& tensor_ideep = Input(i);
            if (tensor_ideep.ndims() == 0 || tensor_ideep.get_nelems() == 0)
              continue;
            inputs_itensor.emplace_back(tensor_ideep);
          } else {
            CAFFE_ENFORCE(
                BlobIsTensorType(OperatorStorage::InputBlob(i), CPU),
                "Expect cpu tensor if not itensor");
            auto& tensor_cpu = OperatorStorage::Input<Tensor>(i, CPU);
            if (tensor_cpu.sizes().size() == 0 || tensor_cpu.numel() == 0)
              continue;
            fallback_to_cpu = true;
            break;
          }
        }

        if (!fallback_to_cpu) {
          int adj_size = inputs_itensor[0].ndims() + (add_axis_ ? 1 : 0);
          int canonical_axis = canonical_axis_index_(axis_, adj_size);
          auto* output = Output(OUTPUT);
          Tensor* axis_info = OutputTensor(AXIS_INFO,
            vector<int64_t>(1, InputSize()), at::dtype<int>().device(CPU));
          auto* axis_data = axis_info->template mutable_data<int>();
          auto axis_vdata =
            ideep::concat::compute(inputs_itensor, canonical_axis, add_axis_, *output);
          for (int i = 0; i < axis_vdata.size(); i++) {
            axis_data[i] = axis_vdata[i];
          }
          return true;
        }

        return fallback_.Run(0);
        */
    }
}

