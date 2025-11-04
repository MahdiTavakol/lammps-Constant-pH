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

#include "compute_temp_constant_pH.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "group.h"
#include "update.h"
#include "modify.h"
#include "memory.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeTempConstantPH::ComputeTempConstantPH(LAMMPS *lmp, int narg, char **arg) :
 ComputeTemp(lmp, narg-1, arg)
{
  if (narg != 4) error->all(FLERR, "Illegal compute temp constant pH command");
  fix_constant_pH_id = std::string(arg[3]);

  n_lambdas = 1;
  int iarg = 4;
  while (iarg < narg) {
     if (!strcmp(arg[iarg], "n_lambdas")) {
        n_lambdas = utils::numeric(FLERR,arg[iarg+1],false,lmp);
        iarg += 2;
     } else error->all(FLERR,"Illegal compute temp constant pH command");
  }

  scalar_flag = vector_flag = 1;
  extscalar = 0;
  extvector = 1;
  tempflag = 1;

}

/* ---------------------------------------------------------------------- */

void ComputeTempConstantPH::setup()
{
  fix_constant_pH = dynamic_cast<FixConstantPH*>(modify->get_fix_by_id(fix_constant_pH_id.c_str()));
  if (!fix_constant_pH)
    error->one(FLERR,"Wrong fix type in the compute_temp_constant_pH");
  fix_constant_pH->return_nparams(n_lambdas);

  dynamic = 0;
  if (dynamic_user || group->dynamic[igroup]) dynamic = 1;
  dof_compute();
}


/* ---------------------------------------------------------------------- */

void ComputeTempConstantPH::dof_compute()
{
  adjust_dof_fix();
  natoms_temp = group->count(igroup);
  dof = domain->dimension * natoms_temp + 3*n_lambdas; // 
  dof -= extra_dof + fix_dof;
  if (dof > 0.0)
    tfactor = force->mvv2e / (dof * force->boltz);
  else
    tfactor = 0.0;
}

/* ---------------------------------------------------------------------- */

double ComputeTempConstantPH::compute_scalar()
{
  invoked_scalar = update->ntimestep;

  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double t = 0.0;

  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
        t += (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]) * rmass[i];
  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
        t += (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]) * mass[type[i]];
  }

  MPI_Allreduce(&t, &scalar, 1, MPI_DOUBLE, MPI_SUM, world);
  
  fix_constant_pH->return_params(pH_state);
  
  double scaling_factor = 100.0;
  double** v_lambdas = pH_state->v_lambdas;
  double** m_lambdas = pH_state->m_lambdas;

  for (int i = 0; i < n_lambdas; i++)
     for (int j = 0; j < 3; j++)
        scalar += v_lambdas[i][j]*v_lambdas[i][j] * m_lambdas[i][j] *scaling_factor;

  
  if (dynamic) dof_compute();
  if (dof < 0.0 && natoms_temp > 0.0)
    error->all(FLERR, "Temperature compute degrees of freedom < 0");
  scalar *= tfactor;
  return scalar;
}
