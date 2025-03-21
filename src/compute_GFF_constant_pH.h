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
ComputeStyle(constant_pH/GFF,ComputeGFFConstantPH);
// clang-format on
#else

#ifndef COMPUTE_GFF_CONSTANT_PH_H
#define COMPUTE_GFF_CONSTANT_PH_H

#include "fix_constant_pH.h"

#include "compute.h"
#include "pair.h"

namespace LAMMPS_NS {

class ComputeGFFConstantPH : public Compute {
 public:
  ComputeGFFConstantPH(class LAMMPS *, int, char **);
  ~ComputeGFFConstantPH() override;
  void setup() override;
  void init() override;
  void compute_array() override;
  void compute_peratom() override {}; // I just wanted LAMMPS to consider this as peratom compute so the peratom energies be tallied in this timestep.
  
  
  
 private:
  // flags
  int flags; 
  
  // dlambda for the calculation of dU/dlambda
  double lambda, dlambda, lambda_buff;

  // lambda variables from the fix constant pH  
  FixConstantPH *fix_constant_pH;
  char *fix_constant_pH_id;
  double** x_lambdas, **v_lambdas, **a_lambdas, **m_lambdas, *H_lambdas;
  double x_lambda_buff, v_lambda_buff, a_lambda_buff, m_lambda_buff;
  double T_lambda;
  int n_lambdas;
  int N_buff;
  int lambda_every;

  class Fix *fixgpu;

  /* HA and HB are for the thermodynamic integration and they should not be confused with the HA and HB
   * in the fix constant pH command.
   * HAs[i] ==> H[lambda_i](lambda-dlambda), HBs[i] ==> H[lambda_i](lambda+dlambda), 
   * HCs[i] ==> H[lambda_i](lambda), dH_dLambda ==> (HB-HA)/(2*dlambda)
   */
   
  double *HAs, *HBs, *HCs, *dH_dLambda;
  

  // Memory allocation and deallocation
  void allocate_storage();
  void deallocate_storage();

};

}    // namespace LAMMPS_NS

#endif
#endif

