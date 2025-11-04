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


#include "compute_GFF_constant_pH.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "input.h"
#include "kspace.h"
#include "memory.h"
#include "modify.h"
#include "pair.h"
#include "pair_hybrid.h"
#include "timer.h"
#include "update.h"
#include "variable.h"

using namespace LAMMPS_NS;


enum { 
       NONE=0,
       BUFFER=1<<0, 
     };

/* ---------------------------------------------------------------------------- */

ComputeGFFConstantPH::ComputeGFFConstantPH(LAMMPS* lmp, int narg, char** arg) : Compute(lmp, narg, arg), 
       fix_constant_pH_id(nullptr)
{
   if (narg < 6) error->all(FLERR, "Illegal number of arguments in compute constant_pH/GFF");
   fix_constant_pH_id = utils::strdup(arg[3]);
   n_lambdas = utils::numeric(FLERR,arg[4],false,lmp);
   lambda = utils::numeric(FLERR,arg[5],false,lmp);
   dlambda = utils::numeric(FLERR,arg[6],false,lmp);
   if (dlambda < 0) error->all(FLERR,"Illegal compute constant_pH/GFF dlambda value {}", dlambda);
   
   int iarg = 7;
   
   while (iarg < narg) {
      if (strcmp(arg[iarg],"buffer") == 0) {
         if (iarg + 2 >= narg)
            error->all(FLERR, "Illegal compute constant_pH/GFF");
         lambda_buff = utils::numeric(FLERR,arg[iarg+1],false,lmp);
         N_buff = utils::numeric(FLERR,arg[iarg+2],false,lmp);
         flags |= BUFFER;
         iarg+=2;
      }
      else if (strcmp(arg[iarg],"m_lambda") == 0) {
         mass_lambda[0] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
         mass_lambda[1] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
         iarg+=3;
      } 
      else
         error->all(FLERR, "Illegal compute constant_pH/GFF");
   }

    
    
   array_flag = 1;
   size_array_rows = 4;
   size_array_cols = n_lambdas + 1;
    
   extvector = 0;

}

/* ---------------------------------------------------------------------------- */
ComputeGFFConstantPH::~ComputeGFFConstantPH()
{
   if (fix_constant_pH_id) delete [] fix_constant_pH_id;
   deallocate_storage();
}

/* ---------------------------------------------------------------------------- */
void ComputeGFFConstantPH::init()
{
  fix_constant_pH = dynamic_cast<FixConstantPH*>(modify->get_fix_by_id(fix_constant_pH_id));
  if (!fix_constant_pH)
   error->all(FLERR, "Wrong fix type in the compute_GFF_constant_pH");
  fix_constant_pH->return_nparams(n_lambdas);
  pH_state = std::make_unique<constant_pH_state>(lmp,mass_lambda,N_buff);
  allocate_storage();
}

/* ---------------------------------------------------------------------------- */
void ComputeGFFConstantPH::setup()
{
   
}

/* --------------------------------------------------------------------- */
void ComputeGFFConstantPH::compute_array()
{
   // Checking if there is enough space in our arrays for all the lambas
   int n_params;
   fix_constant_pH->return_nparams(n_params);
   if (n_lambdas < n_params) {
      int n_lambdas = n_params;
      pH_state->reset_lambdas(n_lambdas,nullptr);
      deallocate_storage();
      allocate_storage();
   }
	
	
   double** x_lambdas = pH_state->lambdas;
   pH_state->lambda_buff = lambda_buff;
   if (0) {
      // Calculating the HA, HB, HC and dH_dLambda
      for (int i = 0; i < n_lambdas; i++) {
         for (int j = 0; j < n_lambdas; j++)
            x_lambdas[j][0] = lambda;
      

         fix_constant_pH->reset_params(pH_state,1);
         fix_constant_pH->calculate_H_once();
         fix_constant_pH->return_H_lambdas(HCs);
	   
         x_lambdas[i][0] = lambda - dlambda;
         fix_constant_pH->reset_params(pH_state,1);
         fix_constant_pH->calculate_H_once();
         fix_constant_pH->return_H_lambdas(HAs);

         x_lambdas[i][0] = lambda + dlambda;
         fix_constant_pH->reset_params(pH_state,1);
         fix_constant_pH->calculate_H_once();
         fix_constant_pH->return_H_lambdas(HBs);
         
         x_lambdas[i][0] = lambda;
         fix_constant_pH->reset_params(pH_state,1);
         
         array[0][i] = HAs[i];
         array[1][i] = HBs[i];
         array[2][i] = HCs[i];
         array[3][i] = (HBs[i]-HAs[i])/(2*dlambda);
      }
   }
   else {
      for (int i = 0; i < n_lambdas; i++)
         x_lambdas[i][0] = lambda;
      
      fix_constant_pH->reset_params(pH_state,1);
      fix_constant_pH->calculate_H_once();
      fix_constant_pH->return_H_lambdas(HCs);
      
      for (int i = 0; i < n_lambdas; i++)
         x_lambdas[i][0] = lambda - dlambda;
      fix_constant_pH->reset_params(pH_state,1);
      fix_constant_pH->calculate_H_once();
      fix_constant_pH->return_H_lambdas(HAs);
      
      for (int i = 0; i < n_lambdas; i++)
         x_lambdas[i][0] = lambda + dlambda;
      fix_constant_pH->reset_params(pH_state,1);
      fix_constant_pH->calculate_H_once();
      fix_constant_pH->return_H_lambdas(HBs);
      
      for (int i = 0; i < n_lambdas; i++)
         x_lambdas[i][0] = lambda;
      fix_constant_pH->reset_params(pH_state,1);
      
      array[0] = HAs;
      array[1] = HBs;
      array[2] = HCs;
      
      for (int j = 0; j < n_lambdas; j++) {
         array[3][j] = (HBs[j]-HAs[j])/(2*dlambda);
      }      
   }
}

/* ----------------------------------------------------------------------
   manage storage for charge, force, energy, virial arrays
   taken from src/FEP/compute_fep.cpp
------------------------------------------------------------------------- */

void ComputeGFFConstantPH::allocate_storage()
{
  memory->create(array,n_lambdas,4,"compute_GFF_constant_pH:array");
  memory->create(HAs, n_lambdas, "compute_GFF_constant_pH:HAs");
  memory->create(HBs, n_lambdas, "compute_GFF_constant_pH:HBs");
  memory->create(HCs, n_lambdas, "compute_GFF_constant_pH:HCs");
}

/* ---------------------------------------------------------------------- */

void ComputeGFFConstantPH::deallocate_storage()
{
  if (array) memory->destroy(array);
  if (HAs) memory->destroy(HAs);
  if (HBs) memory->destroy(HBs);
  if (HCs) memory->destroy(HCs);
  array = nullptr;
  HAs = nullptr;
  HBs = nullptr;
  HCs = nullptr; 
}

