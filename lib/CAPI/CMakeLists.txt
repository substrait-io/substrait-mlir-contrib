add_mlir_public_c_api_library(SubstraitMLIRCAPI
    Dialects.cpp

  LINK_LIBS PUBLIC
    MLIRCAPIIR
    MLIRIR
    MLIRPass
    MLIRSubstraitDialect
    MLIRTargetSubstraitPB
)
