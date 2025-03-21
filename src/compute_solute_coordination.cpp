/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
/* ---------- v0.10.02----------------- */
// Please remove unnecessary includes 
#include "compute_solute_coordination.h"

#include "atom.h"
#include "atom_masks.h"
#include "domain.h"
#include "error.h"
#include "input.h"
#include "math_const.h"
#include "memory.h"
#include "comm.h"
#include "modify.h"
#include "region.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "output.h"
#include "respa.h"
#include "update.h"
#include "variable.h"


#include "force.h"
#include "pair.h"
#include "improper.h"
#include "dihedral.h"
#include "kspace.h"
#include "angle.h"
#include "bond.h"
#include "atom.h"
#include "group.h"

#include "thermo.h"
#include <cmath>
#include <cstring>
#include <stdio.h>

using namespace LAMMPS_NS;
using namespace MathConst;


/* --------------------------------------------------------------------------------------- */

ComputeSoluteCoordination::ComputeSoluteCoordination(LAMMPS* lmp, int narg, char** arg) : Compute(lmp, narg, arg) 
{
   if (narg < 3) utils::missing_cmd_args(FLERR, "fix adaptive_protonation", error);

   peratom_flag = 1;
   scalar_flag = 0;
   extscalar = 0;
   size_peratom_cols = 0;

   
   typeOW = utils::numeric(FLERR, arg[3], false, lmp);

   if (typeOW <=0 || typeOW > atom->ntypes) error->all(FLERR,"Wrong atom type");

   nmax = atom->nmax;
   memory->create(vector_atom, nmax, "solute_coordination:vector_atom");
}

/* --------------------------------------------------------------------------------------- */

ComputeSoluteCoordination::~ComputeSoluteCoordination()
{
   memory->destroy(vector_atom);
}

/* --------------------------------------------------------------------------------------- */

void ComputeSoluteCoordination::init()
{
   // Request a fulintl neighbor list
   int list_flags = NeighConst::REQ_OCCASIONAL; // | NeighConst::REQ_FULL;


   // request for a neighbor list
   neighbor->add_request(this, list_flags);


}

/* ---------------------------------------------------------------------------------------
    It is need to access the neighbor list
   --------------------------------------------------------------------------------------- */

void ComputeSoluteCoordination::init_list(int /*id*/, NeighList* ptr) 
{
   list = ptr;
}

/* --------------------------------------------------------------------------------------- */

void ComputeSoluteCoordination::compute_peratom()
{
   // Building the neighbor
  neighbor->build_one(list);
 
  int* type = atom->type;
  invoked_peratom = update->ntimestep;


  if (atom->nmax > nmax) {
    memory->destroy(vector_atom);
    nmax = atom->nmax;
    memory->create(vector_atom, nmax, "solute_coordination:vector_atom");
  }


  int *ilist, *jlist, *numneigh, **firstneigh;
  int inum, jnum;
  int wnum; // number of surrounding water molecules

  inum = list->inum; // I do not ghost atoms for inum. however, I need them in jnum
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;



  for (int ii = 0; ii < inum; ii++) {
     wnum = 0.0;
     int i = ilist[ii];

     jlist = firstneigh[i];
     jnum = numneigh[i];
     for (int jj = 0; jj < jnum; jj++) {
       int j = jlist[jj];
       j &= NEIGHMASK;

       if (type[j] == typeOW)
         wnum++;  // Just considering the Oxygens. It is possible that both O and H from the same water molecule are close to this atom.
     }

     vector_atom[i] = wnum;
   }
}
