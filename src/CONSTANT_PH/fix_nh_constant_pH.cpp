// clang-format off
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

/* ----------------------------------------------------------------------
   Contributing author: Mahdi Tavakol (Oxford)
   mahditavakol90@gmail.com
------------------------------------------------------------------------- */

#include "fix_constant_pH.h"
#include "fix_nh_constant_pH.h"


#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "irregular.h"
#include "modify.h"
#include "random_mars.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstring>
#include <array>

using namespace LAMMPS_NS;
using namespace FixConst;


enum{NOBIAS,BIAS};

// enums for the lambda integration
enum {LAMBDA_NONE,LAMBDA_ANDERSEN,LAMBDA_BUSSI,LAMBDA_NOSEHOOVER};
enum { 
       NONE_LAMBDA=0,
       BUFFER=1<<0, 
       CONSTRAIN=1<<1,
     };


constexpr double eps = 1e-20;

/* ----------------------------------------------------------------------
   NVT,NPH,NPT integrators for lambdas
 ---------------------------------------------------------------------- */

FixNHConstantPH::FixNHConstantPH(LAMMPS *lmp, int narg, char **arg) :
    FixNH{lmp, narg, arg}, 
    fix_constant_pH{nullptr},  
    lambda_integration_flags{0},lambda_thermostat_type{NONE_LAMBDA},
    ranMarsSeed{1111}
{
  if (narg < 5) utils::missing_cmd_args(FLERR, std::string("fix ") + style, error);  

  int iarg = 3;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"fix_constant_pH_id") == 0) {
       fix_constant_pH_id = std::string(arg[iarg+1]);
       if (fix_constant_pH_id.empty())
         error->all(FLERR,"fix_nh_constant_pH requires 'fix_constant_pH_id <id>'");
       iarg += 2;
    } else if (strcmp(arg[iarg],"lambda_andersen") == 0) {
       lambda_thermostat_type = LAMBDA_ANDERSEN;
       t_andersen = utils::numeric(FLERR,arg[iarg+1],false,lmp);
       if (t_andersen <= 0.0) error->all(FLERR,"lambda_andersen requires t_andersen > 0");
       iarg+=2;
    } else if (strcmp(arg[iarg],"lambda_bussi") == 0) {
       lambda_thermostat_type = LAMBDA_BUSSI;
       tau_t_bussi = utils::numeric(FLERR,arg[iarg+1],false,lmp);
       if (tau_t_bussi <= 0.0) error->all(FLERR,"lambda_bussi requires tau_t_bussi > 0");
       iarg+=2;
    } else if (strcmp(arg[iarg],"lambda_nose-hoover") == 0) {
       lambda_thermostat_type = LAMBDA_NOSEHOOVER;
       Q_lambda_nose_hoover = utils::numeric(FLERR,arg[iarg+1],false,lmp);
       if (Q_lambda_nose_hoover <= 0.0) error->all(FLERR,"lambda_nose-hoover requires Q > 0");
       iarg+=2;
    } else if (strcmp(arg[iarg],"buffer") == 0) {
       lambda_integration_flags |= BUFFER;
       iarg++;
    } else if (strcmp(arg[iarg],"constrain_total_charge") == 0) {
       lambda_integration_flags |= CONSTRAIN;
       mols_charge_change = utils::numeric(FLERR,arg[iarg+1],false,lmp);
       buff_charge_change = utils::numeric(FLERR,arg[iarg+2],false,lmp);
       total_charge = utils::numeric(FLERR,arg[iarg+3],false,lmp);
       iarg += 4;
    } else if (strcmp(arg[iarg],"lambda_seed") == 0) {
      ranMarsSeed = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      if (ranMarsSeed <= 0)
         error->one(FLERR,"The constant_pH seed must be positive");
      iarg+=2;
    } else {
       // skip to next argument; argument check for unknown keywords is done in FixNH
       ++iarg;
    }
  }

  if ((lambda_integration_flags & (BUFFER | CONSTRAIN)) == CONSTRAIN)
   error->one(FLERR,"Constrain total charge in absence of a buffer is not supported yet!");

   

}

/* ---------------------------------------------------------------------- */

void FixNHConstantPH::init()
{
  FixNH::init();
  // dynamic_cast so that if it is not of FixConstantPH* type, no coversion happens!
  fix_constant_pH = dynamic_cast<FixConstantPH*>(modify->get_fix_by_id(fix_constant_pH_id.c_str()));
  if (!fix_constant_pH)
   error->all(FLERR,"fix {} is not a FixConstantPH", fix_constant_pH_id); 

  fix_constant_pH->return_params(pH_state);
  zeta_nose_hoover = 0.0;
  ranMars = std::make_unique<RanMars>(lmp,ranMarsSeed);
}

/* ----------------------------------------------------------------------
   perform half-step update of velocities
   --------------------------------------------------------------------- */

void FixNHConstantPH::nve_v()
{
  FixNH::nve_v();
  
  // Getting the lambda_parameters from the fix_constant_pH.
  // Here while fixNH is modifiying pH_state no other classes
  // wants to use it.. so may be I can use the std::move(pH_state)
  // to call the rvalue return_params function.
  fix_constant_pH->return_params(pH_state);
  double** v_lambdas   = pH_state->v_lambdas;
  const int& n_lambdas = pH_state->n_lambdas;
  double** a_lambdas   = pH_state->a_lambdas;

  for (int i = 0; i < n_lambdas; i++) 
   for (int j = 0; j < 3; j++)
    v_lambdas[i][j] += dtf * a_lambdas[i][j];


 if (lambda_integration_flags & BUFFER) {
  double& v_lambda_buff = pH_state->v_lambda_buff;
  auto&   a_lambda_buff = pH_state->a_lambda_buff;
  v_lambda_buff += dtf * a_lambda_buff;
 }

 // Returning the modified parameters to the fix_constant_pH.
 fix_constant_pH->reset_params(pH_state);

 if (lambda_integration_flags & CONSTRAIN)
   constrain_v_lambdas();
}

/* ----------------------------------------------------------------------
   perform full-step update of positions
-----------------------------------------------------------------------*/

void FixNHConstantPH::nve_x()
{
  FixNH::nve_x();

  // Getting the lambda_parameters from the fix_constant_pH.
  fix_constant_pH->return_params(pH_state);
  double** x_lambdas = pH_state->lambdas;
  double** const v_lambdas = pH_state->v_lambdas;
  const int& n_lambdas = pH_state->n_lambdas;


  for (int i = 0; i < n_lambdas; i++)
   for (int j = 0; j < 3; j++)
    x_lambdas[i][j] += dtv * v_lambdas[i][j];
  
     
  if (lambda_integration_flags & BUFFER) {
   auto& x_lambda_buff = pH_state->lambda_buff;
   auto& v_lambda_buff = pH_state->v_lambda_buff;
   x_lambda_buff += dtv * v_lambda_buff;
  }

  // Returning the modified parameters to the fix_constant_pH.
  fix_constant_pH->reset_params(pH_state);
  // This function sets the charges (qs) in the system based on the current value of x_lambdas and x_lambda_buffs
  fix_constant_pH->reset_qs();
  
  if (lambda_integration_flags & CONSTRAIN)
   constrain_lambdas();
}

/* ----------------------------------------------------------------------
   perform half-step thermostat scaling of velocities
   ---------------------------------------------------------------------- */

void FixNHConstantPH::nh_v_temp()
{
  FixNH::nh_v_temp();
  // The timestep, the current step and the kT of course! 
  double dt = update->dt;
  double kT = force->boltz * t_target;
  // unit conversion
  double mvv2e = force->mvv2e;

  // Getting the lambda_parameters from the fix_constant_pH.
  fix_constant_pH->return_params(pH_state);
  const int& n_lambdas = pH_state->n_lambdas;
  double** x_lambdas = pH_state->lambdas;
  double** v_lambdas = pH_state->v_lambdas;
  auto& x_lambda_buff = pH_state->lambda_buff;
  auto& v_lambda_buff = pH_state->v_lambda_buff;
  double** m_lambdas = pH_state->m_lambdas;
  const double& m_lambda_buff = pH_state->m_lambda_buff;
  const int& N_buff = pH_state->N_buff; 
     
  // The number of degrees of freedom
  double n_dof_1 = static_cast<double>(n_lambdas);
  double n_dof_2 = static_cast<double>(2*n_lambdas);
  n_dof_1 = (lambda_integration_flags & BUFFER) ? n_dof_1 + 1.0 : n_dof_1;
  n_dof_1 = (lambda_integration_flags & CONSTRAIN) ? n_dof_1 - 1.0 : n_dof_1;
  double Nf_lambdas = static_cast<double>(3*n_lambdas);

     

  // Temperature
  std::array<double,3> t_lambda_current;
  double t_lambda_target = t_target;
  fix_constant_pH->return_T_lambda(t_lambda_current[1],0);
  fix_constant_pH->return_T_lambda(t_lambda_current[2],1);
  fix_constant_pH->return_T_lambda(t_lambda_current[0],2);

  auto checkOutBounds = [&](void)
  {
    for (int i = 0; i < n_lambdas; i++) {
      if (x_lambdas[i][0] < -0.1 || x_lambdas[i][0] > 1.1)
       v_lambdas[i][0] = -(x_lambdas[i][0]/std::abs(x_lambdas[i][0]))*std::abs(v_lambdas[i][0]);

      for (int j = 1; j < 3; j++) {
       if (x_lambdas[i][j] < 0.0 && v_lambdas[i][j] < 0.0)
          x_lambdas[i][j] += 1.0;
       if (x_lambdas[i][j] > 1.0 && v_lambdas[i][j] > 0.0)
          x_lambdas[i][j] -= 1.0;
     }
    }

    if (lambda_integration_flags & BUFFER) {
       if (x_lambda_buff < -0.1 || x_lambda_buff > 1.1)
          v_lambda_buff = -(x_lambda_buff/std::abs(x_lambda_buff))*std::abs(v_lambda_buff);
    }
  };

  
  if (lambda_thermostat_type == LAMBDA_ANDERSEN && comm->me == 0) {
    double P = dt/t_andersen;
   

    if (which == NOBIAS) {
      // Dealing with lambdas
      for (int i = 0; i < n_lambdas; i++) 
        for (int j = 0; j < 3; j++) {
           double r = ranMars->uniform();
           if (r < P) {
              double mean = 0.0;
              double sigma = std::sqrt(kT/(m_lambdas[i][j]*mvv2e));
              v_lambdas[i][j] = ranMars->gaussian(mean,sigma);
           }
         }
      // Dealing with the buffer
      if (lambda_integration_flags & BUFFER) {
        double r = ranMars->uniform();
        if (r < P) {
           double mean = 0.0;
           double sigma = std::sqrt(kT/(N_buff*m_lambda_buff*mvv2e));
           v_lambda_buff = ranMars->gaussian(mean,sigma);
        }
      }
      checkOutBounds();
    } else if (which == BIAS) {
      // This needs to be implemented
      error->one(FLERR,"The bias keyword for the fix_nh_constant_pH has not been implemented yet!");
    }


  } else if (lambda_thermostat_type == LAMBDA_BUSSI  && comm->me == 0) {
    //tau_t_bussi should be 1000
     

    // Calculate the Bussi scaling factor
    zeta_bussi = std::exp(-dt/tau_t_bussi);
    
    double r11 = ranMars->gaussian(0.0,1.0);
    double r12 = ranMars->gaussian(0.0,1.0);
    double sum_r21 = 0.0;
    double sum_r22 = 0.0;

    for (int j = 1; j < n_dof_1; j++) {
       double r = ranMars->gaussian(0.0,1.0);
       sum_r21 += r*r;
    }

    for (int j = 1; j < n_dof_2; j++) {
       double r = ranMars->gaussian(0.0,1.0);
       sum_r22 += r*r;
    }

    
    double t_lambda_new_1 = t_lambda_current[1];
    double t_lambda_new_2 = t_lambda_current[2];
    double param1 = (t_lambda_target*t_lambda_current[1]/n_dof_1)*(1-zeta_bussi)*zeta_bussi;
    param1 = std::max(eps,param1);
    t_lambda_new_1 +=  (1-zeta_bussi)*(t_lambda_target*(r11*r11+sum_r21)/n_dof_1-t_lambda_current[1]);
    t_lambda_new_1 += 2*r11*std::sqrt(param1);
    double param2 = (t_lambda_target*t_lambda_current[2]/n_dof_2)*(1-zeta_bussi)*zeta_bussi;
    param2 = std::max(eps,param2);
    t_lambda_new_2 +=  (1-zeta_bussi)*(t_lambda_target*(r12*r12+sum_r22)/n_dof_2-t_lambda_current[2]);
    t_lambda_new_2 += 2*r12*std::sqrt(param2);
    double ratio1 = std::max(eps, t_lambda_new_1/t_lambda_current[1]);
    double ratio2 = std::max(eps, t_lambda_new_2/t_lambda_current[2]);
    double alpha_bussi1 = std::sqrt(ratio1);
    double alpha_bussi2 = std::sqrt(ratio2);


    if (which == NOBIAS) {

       // first, the lambdas
       for (int i = 0; i < n_lambdas; i++) {
          v_lambdas[i][0] *= alpha_bussi1;
          for (int j = 1; j < 3; j++) {
            v_lambdas[i][j] *= alpha_bussi2;
          }
       }
       // and then the buffer 
       if (lambda_integration_flags & BUFFER) {
          v_lambda_buff *= alpha_bussi1;
       }
       checkOutBounds();
    } else if (which == BIAS) {
       // This needs to be implemented
       error->one(FLERR,"The bias keyword for the fix_nh_constant_pH has not been implemented yet!");
    }
  } else if (lambda_thermostat_type == LAMBDA_NOSEHOOVER && comm->me == 0) {  
     zeta_nose_hoover += dt * (t_lambda_current[0] - t_lambda_target);

     if (which == NOBIAS) {
        // first the lambdas
        for (int i = 0; i < n_lambdas; i++) 
           for (int j = 0; j < 3; j++) {
              v_lambdas[i][j] *= std::exp(-zeta_nose_hoover * dt);
        }
        // and then the buffer
        if (lambda_integration_flags & BUFFER)
           v_lambda_buff *= std::exp(-zeta_nose_hoover * dt);

        checkOutBounds();
           
     } else if (which == BIAS) {
        // This needs to be implemented
        error->one(FLERR,"The bias keyword for the fix_nh_constant_pH has not been implemented yet!");
     }
  }

  if (n_lambdas == 0)
     return;
 

  // v_lambdas[0] is the location of the contigous memory allocated 
  // for the double ** v_lambdas
  MPI_Bcast(v_lambdas[0],n_lambdas*3,MPI_DOUBLE,0,world);
  if (lambda_integration_flags & BUFFER)
     MPI_Bcast(&v_lambda_buff,1,MPI_DOUBLE,0,world);
   
  // constraining the v_lambdas
  constrain_v_lambdas();
   
  fix_constant_pH->reset_params(pH_state);
}

/* ---------------------------------------------------------------------
   Applies the shake algorithm to the sum of the lambdas 

   It adds a  constraint according to the Donnine et al JCTC 2016 
   equation (13).

   The constraint equations were taken from the Tuckerman statistical mechanics
   book 2nd edition pages 106.

   
   --------------------------------------------------------------------- */
   

void FixNHConstantPH::constrain_lambdas()
{
   double omega = 0.0;
   double domega = omegaPrev;
   double q_total;
   double sigma_lambda;
   double sigma_mass_inverse;
   

   constexpr int maxCycles = 10000;
   constexpr double alpha = 0.5; 
   constexpr double etol = 1e-6; 
   int cycle = 0;

   /* Some sanity checks */
   if (comm->me == 0) {
      const int n_lambdas = pH_state->n_lambdas;
      int N_buff = pH_state->N_buff;
      double ** m_lambdas = pH_state->m_lambdas;
      double m_lambda_buff = pH_state->m_lambda_buff;

      /* Checking if the charge content of the N_buff is large enough for n_lambdas
       * Since there is a possibility that the n_lambdas change during the simulation by 
       * the fix_adaptive_protonation.cpp command, the check should be done here. 
       */
      if (buff_charge_change*N_buff < mols_charge_change*n_lambdas)
         error->one(FLERR,"The charge content of N_buff={} is not large enough for n_lambdas={}: Please increase the N_buff\n",N_buff,n_lambdas);
      for (int i =0; i < n_lambdas; i++)
         if (m_lambdas[i][0] == 0) error->all(FLERR,"m_lambdas({},0) is zero in fix_nh_constant_pH",i);
      if (m_lambda_buff == 0) error->all(FLERR,"Buffer mass is zero in fix_nh_constant_pH");
   }
   
   
   
   /* The do while loop was used on purpose so that even when the loop termination condition
      is satisfied the q_total is calculated for the last time with final values of lambdas */
   do {
      // Just doing this on the root and then broadcasting the results
      sigma_lambda = 0.0;
      sigma_mass_inverse = 0.0;

      fix_constant_pH->return_params(pH_state);
      const int n_lambdas = pH_state->n_lambdas;
      const int N_buff = pH_state->N_buff;
      const double N_buff_double = static_cast<double>(N_buff);
      double** x_lambdas = pH_state->lambdas;
      auto& x_lambda_buff = pH_state->lambda_buff;
      double** m_lambdas = pH_state->m_lambdas;
      double m_lambda_buff = pH_state->m_lambda_buff;

      // At the first iteration there is an update with the initial value
      // of domega which is the omegaPrev

      omega += domega;
      for (int i = 0; i < n_lambdas; i++)
         x_lambdas[i][0] += (domega * mols_charge_change / m_lambdas[i][0]);
      x_lambda_buff += buff_charge_change * domega / m_lambda_buff;
     

      fix_constant_pH->reset_params(pH_state,1);
      fix_constant_pH->reset_qs();
      
      
      for (int i = 0; i < n_lambdas; i++) {
         sigma_lambda += x_lambdas[i][0];
         sigma_mass_inverse += (1.0/m_lambdas[i][0]);
      }

   
      q_total = compute_q_total();

      
      double denom = (mols_charge_change*mols_charge_change*sigma_mass_inverse + (N_buff_double*buff_charge_change*buff_charge_change/m_lambda_buff));

      if (!std::isfinite(denom) || std::abs(denom) < eps)
         error->one(FLERR,"Denominator is invalid (non-finite or too small) in constrain_lambdas");

      domega = -alpha*q_total / denom;

   } while (std::abs(q_total) > etol && ++cycle < maxCycles);

   if (comm->me == 0 && cycle >= maxCycles)
      error->warning(FLERR,"Charge constrain did not reach convergence after {} iterations: {}",maxCycles,q_total);

   const int n_lambdas = pH_state->n_lambdas;
   double** v_lambdas = pH_state->v_lambdas;
   double** m_lambdas = pH_state->m_lambdas;
   double v_lambda_buff = pH_state->v_lambda_buff;
   double m_lambda_buff = pH_state->m_lambda_buff;
   double dt = update->dt;

   // v_lambdas constraining in the first half step of velocity verlet
   //for (int i = 0; i < n_lambdas; i++)
      //v_lambdas[i][0] += omega*mols_charge_change / (dt*m_lambdas[i][0]);
   //v_lambda_buff += omega*buff_charge_change / (dt*m_lambda_buff);
   
   fix_constant_pH->reset_params(pH_state,1);
   fix_constant_pH->reset_qs();

   // keeping the omega for the next step
   omegaPrev = omega;
}

/* ---------------------------------------------------------------------
   sigma ( delta_q_i * (v_i + alpha * delta_q_i / M_i )) = 0
   alpha = nom/denom
   v_i += alpha*delta_q_i / M_i

   --------------------------------------------------------------------- */

void FixNHConstantPH::constrain_v_lambdas()
{
   const int n_lambdas = pH_state->n_lambdas;
   const double N_buff_double = static_cast<double>(pH_state->N_buff);
   double** v_lambdas = pH_state->v_lambdas;
   double** m_lambdas = pH_state->m_lambdas;
   double v_lambda_buff = pH_state->v_lambda_buff;
   double m_lambda_buff = pH_state->m_lambda_buff;

   double mu = 0.0;

   double nom = 0.0;
   double denom = 0.0;

   for (int i = 0; i < n_lambdas; i++) {
      nom += -(mols_charge_change*v_lambdas[i][0]);
      denom += mols_charge_change*mols_charge_change/ m_lambdas[i][0];
   }

   nom += -N_buff_double*buff_charge_change*v_lambda_buff;
   denom += N_buff_double*buff_charge_change*buff_charge_change/m_lambda_buff;

   mu = nom/denom;

   for (int i = 0; i < n_lambdas; i++)
      v_lambdas[i][0] += mu*mols_charge_change / m_lambdas[i][0];
   v_lambda_buff += mu*buff_charge_change / m_lambda_buff;

   fix_constant_pH->reset_params(pH_state,1);
   fix_constant_pH->reset_qs();
}

/* ----------------------------------------------------------------------
   computes the q_total to be used in the constrain_lambdas() function
   ---------------------------------------------------------------------- */
double FixNHConstantPH::compute_q_total()
{
   double * q = atom->q;
   int nlocal = atom->nlocal;
   double q_local = 0.0;
   double q_total = 0.0;

   for (int i = 0; i <nlocal; i++)
      q_local += q[i];

   MPI_Allreduce(&q_local,&q_total,1,MPI_DOUBLE,MPI_SUM,world);
   
   return q_total;
}


/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double FixNHConstantPH::memory_usage()
{
  int& n_lambdas = pH_state->n_lambdas;
  double bytes = 0.0;
  bytes += 4.0*3.0*n_lambdas*sizeof(double); // x_lambdas, v_lambdas, a_lambdas and m_lambdas
  if (irregular) bytes += irregular->memory_usage();
  return bytes;
}
