crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/QEngine.h]

/**
  | QEngine is an enum that is used to select
  | the engine to run quantized ops.
  | 
  | Keep this enum in sync with get_qengine_id()
  | in torch/backends/quantized/__init__.py
  |
  */
#[repr(u8)]
pub enum QEngine {
    NoQEngine = 0,
    FBGEMM    = 1,
    QNNPACK   = 2,
}

pub const K_NO_QENGINE: QEngine = QEngine::NoQEngine;
pub const KFBGEMM:      QEngine = QEngine::FBGEMM;
pub const KQNNPACK:     QEngine = QEngine::QNNPACK;

#[inline] pub fn to_string(qengine: QEngine) -> String {
    
    todo!();
        /*
            switch (qengine) {
        case kNoQEngine:
          return "NoQEngine";
        case kFBGEMM:
          return "FBGEMM";
        case kQNNPACK:
          return "QNNPACK";
        default:
          TORCH_CHECK(
              false, "Unrecognized Quantized Engine: ", static_cast<int>(qengine));
      }
        */
}
