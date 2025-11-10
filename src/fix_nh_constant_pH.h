
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

/* ----------------------------------------------------------------------
   Contributing author: Mahdi Tavakol (Oxford)
   mahditavakol90@gmail.com
------------------------------------------------------------------------- */

#ifndef LMP_FIX_NH_CONSTANT_PH_H
#define LMP_FIX_NH_CONSTANT_PH_H

#include <ctime>

#include "fix.h"    // IWYU pragma: export
#include "fix_constant_pH.h"
#include "fix_nh.h"
#include "random_mars.h"

namespace LAMMPS_NS {

class FixNHConstantPH : public FixNH {
 public:
  FixNHConstantPH(class LAMMPS *, int, char **);
  void init() override;
  double memory_usage() override;

 protected:
  // integration functions (x and lambdas)
  void nve_x() override;
  void nve_v() override;
  void nh_v_temp() override;

  // functions related to lambdas
  void deallocate_lambda_storage();
  void allocate_lambda_storage();

  // constraining total charge through change lambdas and lambda_buff
  template <int mode> void constrain_lambdas();
  // computing the total charge
  double compute_q_total();

  // lambda variables from the fix constant pH
  FixConstantPH *fix_constant_pH;
  std::string fix_constant_pH_id;

  // pH_state
  std::unique_ptr<constant_pH_state> pH_state;
  double T_lambda;


  // Integration flags for lambda, should I constrain total charge and also is there any buffer
  int lambda_integration_flags;
  // The style of the thermostat for the lambdas
  int lambda_thermostat_type;

  // Changes in the charge
  double mols_charge_change, buff_charge_change, total_charge;

  // Parameter for Andersen thermostat
  double t_andersen;

  // Parameter for Bussi thermostat
  double tau_t_bussi;
  double zeta_bussi;

  // Parameters for Nose-Hoover thermostat
  double Q_lambda_nose_hoover;
  double zeta_nose_hoover;

 private:
  std::unique_ptr<RanMars> ranMars;
  int ranMarsSeed;
};

}    // namespace LAMMPS_NS

#endif
