set(OP_NAME ml_congestion)

set(TARGET_NAME ${OP_NAME})

set(MODELS models/routenet.py)
set(WEIGHTS models/work_dir/routenet/model_iters_20000.pth)
install(FILES ${MODELS} DESTINATION dreamplace/ops/${TARGET_NAME})
install(FILES ${WEIGHTS} DESTINATION . RENAME congestion.pth)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${TARGET_NAME})
