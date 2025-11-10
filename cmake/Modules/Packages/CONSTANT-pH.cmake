set(CPH_SOURCE_DIR ${LAMMPS_SOURCE_DIR}/CONSTANT_PH)
set(CONSTANT_pH_SOURCES
  ${CPH_SOURCE_DIR}/fix_constant_pH.cpp
  ${CPH_SOURCE_DIR}/fix_nh_constant_pH.cpp
  ${CPH_SOURCE_DIR}/fix_nvt_constant_pH.cpp
  ${CPH_SOURCE_DIR}/fix_npt_constant_pH.cpp
  ${CPH_SOURCE_DIR}/fix_adaptive_protonation.cpp
  ${CPH_SOURCE_DIR}/constant_pH_structures.cpp
  ${CPH_SOURCE_DIR}/compute_temp_constant_pH.cpp
  ${CPH_SOURCE_DIR}/compute_GFF_constant_pH.cpp
)


lammps_add_sources(${CONSTANT_pH_SOURCES})

