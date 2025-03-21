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
ComputeStyle(solute_coordination,ComputeSoluteCoordination);
// clang-format on
#else

#ifndef LMP_COMPUTE_SOLUTE_COORDINATION_H
#define LMP_COMPUTE_SOLUTE_COORDINATION_H

#include "compute.h"
#include <map>

namespace LAMMPS_NS {

class ComputeSoluteCoordination : public Compute {
 public:
  int nchunk, ncoord, compress, idsflag, lockcount;
  int computeflag;    // 1 if this compute invokes other computes
  double chunk_volume_scalar;
  double *chunk_volume_vec;
  double **coord;
  int *ichunk, *chunkID;

  ComputeSoluteCoordination(class LAMMPS *, int, char **);
  ~ComputeSoluteCoordination() override;
  void init() override;
  void compute_peratom() override;
  void init_list(int, class NeighList*) override;


 private:
  int typeOW;
  int nmax;


   // Neighborlist is required for accessing neighbors
   class NeighList* list;



};

}    // namespace LAMMPS_NS

#endif
#endif
