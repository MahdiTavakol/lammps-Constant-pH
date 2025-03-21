/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(temp/constant_pH,ComputeTempConstantPH);
// clang-format on
#else

#ifndef LMP_COMPUTE_TEMP_CONSTANT_PH_H
#define LMP_COMPUTE_TEMP_CONSTANT_PH_H

#include "compute.h"
#include "compute_temp.h"
#include "fix_constant_pH.h"


namespace LAMMPS_NS {

class ComputeTempConstantPH : public ComputeTemp {
 public:
  ComputeTempConstantPH(class LAMMPS *, int, char **);
  ~ComputeTempConstantPH() override;
  void setup() override;
  double compute_scalar() override;
  // May be I need to implement compute_vector() to be used in the barostat section of the fix_nh.cpp

 protected:
  double tfactor;

  void dof_compute() override;


  // lambda variables from the fix constant pH  
  FixConstantPH *fix_constant_pH;
  char *fix_constant_pH_id;
  double **x_lambdas, **v_lambdas, **a_lambdas, **m_lambdas;
  int n_lambdas;

};

}    // namespace LAMMPS_NS

#endif
#endif
