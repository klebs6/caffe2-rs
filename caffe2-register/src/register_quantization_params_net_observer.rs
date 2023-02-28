crate::ix!();

use crate::{
    NetObserver,
    NetBase
};

/**
  | Set quantization parameters of operators
  | based on min/max collected from OutputMinMaxObserver
  |
  */
pub struct RegisterQuantizationParamsNetObserver {
    base: NetObserver,
}

impl RegisterQuantizationParamsNetObserver {

    pub fn new(
        subject:                  *mut NetBase,
        min_max_file_name:        &String,
        is_weight:                Option<bool>,
        qparams_output_file_name: &String) -> Self {

        let is_weight: bool = is_weight.unwrap_or(false);

        todo!();
        /*
            : NetObserver(subject) 

      ifstream f(min_max_file_name);

      // check the format by looking at the first line
      string first_line, word;
      getline(f, first_line);
      f.seekg(0, f.beg);
      istringstream ist(first_line);
      int nwords_first_line = 0;
      while (ist >> word) {
        ++nwords_first_line;
      }

      bool new_format = nwords_first_line == 6;
      if (!new_format && nwords_first_line != 5) {
        LOG(WARNING) << "min_max file " << min_max_file_name
                     << " has an invalid format";
      }

      // Optionally dump quantization params to file
      ofstream fout;
      if (!qparams_output_file_name.empty()) {
        fout.open(qparams_output_file_name);
        if (!fout) {
          LOG(WARNING) << this << ": can't open " << qparams_output_file_name;
        }
      }

      // parse the input file
      int op_index = 0;
      for (auto* op : subject->GetOperators()) {
        for (int i = 0; i < op->OutputSize(); ++i) {
          int op_index2, i2;
          string op_type, tensor_name;
          float min, max;

          if (new_format) {
            f >> op_index2 >> op_type >> i2 >> tensor_name >> min >> max;
          } else {
            f >> op_index2 >> i2 >> tensor_name >> min >> max;
          }
          assert(op_index2 == op_index);
          assert(i2 == i);
          assert(tensor_name == op->debug_def().output(i));

          TensorQuantizationParams qparams;
          if (max > min) {
            unique_ptr<QuantizationFactory> qfactory(GetQuantizationFactoryOf(op));
            qparams = qfactory->ChooseQuantizationParams(min, max, is_weight);
          } else {
            qparams.scale = 0.1f;
            qparams.zero_point = -min / qparams.scale;
            qparams.precision = 8;
          }

          if (HasDNNLowPEngine_(*op)) {
            SetStaticQuantizationParams(op, i, qparams);
          }

          if (fout.is_open()) {
            fout << op_index << " " << op_type << " " << i << " " << tensor_name
                 << " " << qparams.Min() << " " << qparams.Max() << " "
                 << qparams.scale << " " << qparams.zero_point << " "
                 << qparams.precision << endl;
          }
        }
        ++op_index;
      }

      if (fout.is_open()) {
        fout.close();
      }
        */
    }
}


