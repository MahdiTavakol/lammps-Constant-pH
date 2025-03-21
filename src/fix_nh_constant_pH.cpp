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
   Contributing authors: Mark Stevens (SNL), Aidan Thompson (SNL)
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Constant pH support added by: Mahdi Tavakol (Oxford)
   v0.08.52
------------------------------------------------------------------------- */

#include "fix_constant_pH.h"
#include "fix_nh_constant_pH.h"


#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "fix_deform.h"
#include "force.h"
#include "group.h"
#include "irregular.h"
#include "kspace.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "respa.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <random>

using namespace LAMMPS_NS;
using namespace FixConst;

static constexpr double DELTAFLIP = 0.1;
static constexpr double TILTMAX = 1.5;
static constexpr double EPSILON = 1.0e-6;

enum{NOBIAS,BIAS};
enum{NONE,XYZ,XY,YZ,XZ};
enum{ISO,ANISO,TRICLINIC};

// enums for the lambda integration
enum {LAMBDA_NONE,LAMBDA_ANDERSEN,LAMBDA_BUSSI,LAMBDA_NOSEHOOVER};
enum { 
       NONE_LAMBDA=0,
       BUFFER=1<<0, 
       CONSTRAIN=1<<1,
     };

/* ----------------------------------------------------------------------
   NVT,NPH,NPT integrators for improved Nose-Hoover equations of motion
 ---------------------------------------------------------------------- */

FixNHConstantPH::FixNHConstantPH(LAMMPS *lmp, int narg, char **arg) :
    FixNH(lmp, narg, arg), 
    fix_constant_pH(nullptr), fix_constant_pH_id(nullptr), 
    x_lambdas(nullptr), v_lambdas(nullptr), a_lambdas(nullptr), m_lambdas(nullptr)
{
  if (narg < 5) utils::missing_cmd_args(FLERR, std::string("fix ") + style, error);
  
  lambda_thermostat_type = NONE_LAMBDA;
  lambda_integration_flags = 0;
  

  int iarg = 3;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"fix_constant_pH_id") == 0) {
       lambda_thermostat_type = NONE_LAMBDA;
       lambda_integration_flags = 0;
       fix_constant_pH_id = utils::strdup(arg[iarg+1]);
       iarg += 2;
    } else if (strcmp(arg[iarg],"lambda_andersen") == 0) {
       lambda_thermostat_type = LAMBDA_ANDERSEN;
       t_andersen = utils::numeric(FLERR,arg[iarg+1],false,lmp);
       iarg+=2;
    } else if (strcmp(arg[iarg],"lambda_bussi") == 0) {
       lambda_thermostat_type = LAMBDA_BUSSI;
       tau_t_bussi = utils::numeric(FLERR,arg[iarg+1],false,lmp);
       iarg+=2;
    } else if (strcmp(arg[iarg],"lambda_nose-hoover") == 0) {
       lambda_thermostat_type = LAMBDA_NOSEHOOVER;
       Q_lambda_nose_hoover = utils::numeric(FLERR,arg[iarg+1],false,lmp);
       iarg+=2;
    } else if (strcmp(arg[iarg],"buffer") == 0) {
       lambda_integration_flags |= BUFFER;
       iarg++;
    } else if (strcmp(arg[iarg],"constrain_total_charge") == 0) {
       if (!(lambda_integration_flags & BUFFER))
          error->one(FLERR,"Constrain total charge in absence of a buffer is not supported yet!");
       lambda_integration_flags |= CONSTRAIN;
       mols_charge_change = utils::numeric(FLERR,arg[iarg+1],false,lmp);
       buff_charge_change = utils::numeric(FLERR,arg[iarg+2],false,lmp);
       total_charge = utils::numeric(FLERR,arg[iarg+3],false,lmp);
       iarg += 4;
    } else if (strcmp(arg[iarg],"lambda_every") == 0) {
       lambda_every = utils::numeric(FLERR,arg[iarg+1],false,lmp);
       if (lambda_every <= 0)
          error->one(FLERR,"The lambda_every parameter must be positive");
       iarg+=2; 
    } else {
       // skip to next argument; argument check for unknown keywords is done in FixNH
       ++iarg;
    }
  }

  if (fix_constant_pH_id == nullptr) error->all(FLERR, "Invalid fix_nh constant_pH");

}

/* ---------------------------------------------------------------------- */

FixNHConstantPH::~FixNHConstantPH()
{
  if (fix_constant_pH_id) delete [] fix_constant_pH_id;
  deallocate_lambda_storage();
}

/* ---------------------------------------------------------------------- */

void FixNHConstantPH::init()
{
  FixNH::init();

  
  fix_constant_pH = static_cast<FixConstantPH*>(modify->get_fix_by_id(fix_constant_pH_id));
  fix_constant_pH->return_nparams(n_lambdas);

  allocate_lambda_storage();

  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  zeta_nose_hoover = 0.0;
}

/* ----------------------------------------------------------------------
    deallocating the storage of the lambda parameters
   ---------------------------------------------------------------------- */
   
void FixNHConstantPH::deallocate_lambda_storage()
{
  if (x_lambdas) memory->destroy(x_lambdas);
  if (v_lambdas) memory->destroy(v_lambdas);
  if (a_lambdas) memory->destroy(a_lambdas);
  if (m_lambdas) memory->destroy(m_lambdas);
  
  x_lambdas = nullptr;
  v_lambdas = nullptr;
  a_lambdas = nullptr;
  m_lambdas = nullptr; 
}

/* ----------------------------------------------------------------------
    allocating the storage of the lambda parameters
   ---------------------------------------------------------------------- */
   
void FixNHConstantPH::allocate_lambda_storage()
{
  memory->create(x_lambdas,n_lambdas,3,"nh_constant_pH:x_lambdas");
  memory->create(v_lambdas,n_lambdas,3,"nh_constant_pH:v_lambdas");
  memory->create(a_lambdas,n_lambdas,3,"nh_constant_pH:a_lambdas");
  memory->create(m_lambdas,n_lambdas,3,"nh_constant_pH:m_lambdas");
}

/* ----------------------------------------------------------------------
   updating the lambda parameters from the fix constant_pH
   ---------------------------------------------------------------------- */
   
void FixNHConstantPH::update_lambda_params()
{
  int n_lambdas_current;
  fix_constant_pH->return_nparams(n_lambdas_current);
  // We need to check if the number of lambdas have changed
  if (n_lambdas != n_lambdas_current) {
    n_lambdas = n_lambdas_current;
    deallocate_lambda_storage();
    allocate_lambda_storage();
  }
  fix_constant_pH->return_params(x_lambdas,v_lambdas,a_lambdas,m_lambdas);
}

/* ----------------------------------------------------------------------
   perform half-step update of velocities
-----------------------------------------------------------------------*/

void FixNHConstantPH::nve_v()
{
  FixNH::nve_v();
  
  bigint ntimestep = update->ntimestep;
  
  update_lambda_params();

  for (int i = 0; i < n_lambdas; i++) 
     for (int j = 0; j < 3; j++)
        v_lambdas[i][j] += dtf * a_lambdas[i][j];
  
   
  fix_constant_pH->reset_params(x_lambdas,v_lambdas,a_lambdas,m_lambdas);

  if (lambda_integration_flags & BUFFER) {
     fix_constant_pH->return_buff_params(x_lambda_buff,v_lambda_buff,a_lambda_buff,m_lambda_buff,N_buff);
     v_lambda_buff += dtf * a_lambda_buff;
     fix_constant_pH->reset_buff_params(x_lambda_buff,v_lambda_buff,a_lambda_buff, m_lambda_buff);
  }
}

/* ----------------------------------------------------------------------
   perform full-step update of positions
-----------------------------------------------------------------------*/

void FixNHConstantPH::nve_x()
{
  FixNH::nve_x();
  bigint ntimestep = update->ntimestep;
  update_lambda_params();
  for (int i = 0; i < n_lambdas; i++)
     for (int j = 0; j < 3; j++)
        x_lambdas[i][j] += dtv * v_lambdas[i][j];
  
     
  // Resets the parameters for x_lambdas to be used in the constrain
  fix_constant_pH->reset_params(x_lambdas,v_lambdas,a_lambdas,m_lambdas);

  // This function sets the charges (qs) in the system based on the current value of x_lambdas and x_lambda_buffs
  fix_constant_pH->reset_qs();     
     
  if (lambda_integration_flags & BUFFER) {
     fix_constant_pH->return_buff_params(x_lambda_buff,v_lambda_buff,a_lambda_buff,m_lambda_buff,N_buff);
     x_lambda_buff += dtv * v_lambda_buff;
     if (lambda_integration_flags & CONSTRAIN) constrain_lambdas<2>();
  }
}

/* ----------------------------------------------------------------------
   perform half-step thermostat scaling of velocities
   ---------------------------------------------------------------------- */

void FixNHConstantPH::nh_v_temp()
{
  FixNH::nh_v_temp();
  // The timestep, the current step and the kT of course! 
  double dt = update->dt;
  bigint ntimestep = update->ntimestep;
  double kT = force->boltz * t_target;
  // remove the center of mass velocity
  double v_cm = 0.0;
  // unit conversion
  double mvv2e = force->mvv2e;

  // Lets extract the parameters from the fix_constant_pH again
  update_lambda_params();
  
  // and the buffer if present
  if (lambda_integration_flags & BUFFER)
     fix_constant_pH->return_buff_params(x_lambda_buff,v_lambda_buff,a_lambda_buff,m_lambda_buff,N_buff);
     
  // The number of degrees of freedom
  double Nf_lambdas = static_cast<double>(3*n_lambdas);

  if (lambda_integration_flags & BUFFER)
     Nf_lambdas += 1.0;
  
  if (lambda_integration_flags & CONSTRAIN)
     Nf_lambdas -= 1.0;
     

  // Temperature
  double t_lambda_current, t_lambda_current_1, t_lambda_current_2;
  double t_lambda_target = t_target;
  fix_constant_pH->return_T_lambda(t_lambda_current_1,0);
  fix_constant_pH->return_T_lambda(t_lambda_current_2,1);
  fix_constant_pH->return_T_lambda(t_lambda_current,2);
  
  if (lambda_thermostat_type == LAMBDA_ANDERSEN && comm->me == 0) {
    double P = dt/t_andersen;
   

    if (which == NOBIAS) {
      // Dealing with lambdas
      for (int i = 0; i < n_lambdas; i++) 
        for (int j = 0; j < 3; j++) {
           double r = static_cast<double>(rand())/ RAND_MAX;
           if (r < P) {
              double mean = 0.0;
              double sigma = std::sqrt(kT/(m_lambdas[i][j]*mvv2e));
              v_lambdas[i][j] = random_normal(mean, sigma);
           }
           if (j == 0) {
              if (x_lambdas[i][j] < -0.1 || x_lambdas[i][j] > 1.1)
                 v_lambdas[i][j] = -(x_lambdas[i][j]/std::abs(x_lambdas[i][j]))*std::abs(v_lambdas[i][j]);
           }
           if (j > 0) {
              if (x_lambdas[i][j] < 0.0 && v_lambdas[i][j] < 0.0)
                 x_lambdas[i][j] += 1.0;
              if (x_lambdas[i][j] > 1.0 && v_lambdas[i][j] > 0.0)
                 x_lambdas[i][j] -= 1.0;            
           }
      }
      // Dealing with the buffer
      if (lambda_integration_flags & BUFFER) {
        double r = static_cast<double>(rand())/ RAND_MAX;
        if (r < P) {
           double mean = 0.0;
           double sigma = std::sqrt(kT/(N_buff*m_lambda_buff*mvv2e));
           v_lambda_buff = random_normal(mean,sigma);
        }
        if (x_lambda_buff < -0.1 || x_lambda_buff > 1.1)
           v_lambda_buff = -(x_lambda_buff/std::abs(x_lambda_buff))*std::abs(v_lambda_buff);
      }
    } else if (which == BIAS) {
      // This needs to be implemented
      error->one(FLERR,"The bias keyword for the fix_nh_constant_pH has not been implemented yet!");
    }
  } else if (lambda_thermostat_type == LAMBDA_BUSSI  && comm->me == 0) {
    //tau_t_bussi should be 1000
     

    // Calculate the Bussi scaling factor
    zeta_bussi = std::exp(-dt/tau_t_bussi);
    
    double r11 = random_normal(0,1);
    double r12 = random_normal(0,1);
    double sum_r21 = 0.0;
    double sum_r22 = 0.0;

    for (int j = 1; j < n_lambdas; j++) {
       double r = random_normal(0,1);
       sum_r21 += r*r;
    }
    for (int j = 1; j < 2*n_lambdas; j++) {
       double r = random_normal(0,1);
       sum_r22 += r*r;
    }

    
    double t_lambda_new_1 = t_lambda_current_1;
    double t_lambda_new_2 = t_lambda_current_2;
    t_lambda_new_1 +=  (1-zeta_bussi)*(t_lambda_target*(r11*r11+sum_r21)/n_lambdas-t_lambda_current_1);
    t_lambda_new_1 += 2*r11*std::sqrt((t_lambda_target*t_lambda_current_1/n_lambdas)*(1-zeta_bussi)*zeta_bussi);
    t_lambda_new_2 +=  (1-zeta_bussi)*(t_lambda_target*(r12*r12+sum_r22)/(2*n_lambdas)-t_lambda_current_2);
    t_lambda_new_2 += 2*r12*std::sqrt((t_lambda_target*t_lambda_current_2/(2*n_lambdas))*(1-zeta_bussi)*zeta_bussi);
    double alpha_bussi1 = std::sqrt(t_lambda_new_1 / t_lambda_current_1);
    double alpha_bussi2 = std::sqrt(t_lambda_new_2 / t_lambda_current_2);

    if (which == NOBIAS) {

       // first, the lambdas
       for (int i = 0; i < n_lambdas; i++) {
          v_lambdas[i][0] *= alpha_bussi1;
          for (int j = 1; j < 3; j++) {
             v_lambdas[i][j] *= alpha_bussi2;

             if (j == 0) {
               if (x_lambdas[i][j] < -0.1 || x_lambdas[i][j] > 1.1)
                 v_lambdas[i][j] = -(x_lambdas[i][j]/std::abs(x_lambdas[i][j]))*std::abs(v_lambdas[i][j]);
             } else {
               if (x_lambdas[i][j] < 0.0 && v_lambdas[i][j] < 0.0)
                 x_lambdas[i][j] += 1.0;
               if (x_lambdas[i][j] > 1.0 && v_lambdas[i][j] > 0.0)
                 x_lambdas[i][j] -= 1.0;
             }
          }
       }
       // and then the buffer 
       if (lambda_integration_flags & BUFFER) {
          v_lambda_buff *= alpha_bussi1;
          if (x_lambda_buff < -0.1 || x_lambda_buff > 1.1)
            v_lambda_buff = -(x_lambda_buff/std::abs(x_lambda_buff))*std::abs(v_lambda_buff);
       }
    } else if (which == BIAS) {
       // This needs to be implemented
       error->one(FLERR,"The bias keyword for the fix_nh_constant_pH has not been implemented yet!");
    }
  } else if (lambda_thermostat_type == LAMBDA_NOSEHOOVER && comm->me == 0) {  
     zeta_nose_hoover += dt * (t_lambda_current - t_lambda_target);

     if (which == NOBIAS) {
        // first the lambdas
        for (int i = 0; i < n_lambdas; i++) 
           for (int j = 0; j < 3; j++) {
              v_lambdas[i][j] *= std::exp(-zeta_nose_hoover * dt);
           
              if (j == 0) {
                if (x_lambdas[i][j] < -0.1 || x_lambdas[i][j] > 1.1)
                  v_lambdas[i][j] = -(x_lambdas[i][j]/std::abs(x_lambdas[i][j]))*std::abs(v_lambdas[i][j]);
              }
              else {
                if (x_lambdas[i][j] < 0.0 && v_lambdas[i][j] < 0.0)
                  x_lambdas[i][j] += 1.0;
                if (x_lambdas[i][j] > 1.0 && v_lambdas[i][j] > 0.0)
                  x_lambdas[i][j] -= 1.0;
           }
        }
        // and then the buffer
        if (lambda_integration_flags & BUFFER)
           v_lambda_buff *= std::exp(-zeta_nose_hoover * dt);
           
     } else if (which == BIAS) {
        // This needs to be implemented
        error->one(FLERR,"The bias keyword for the fix_nh_constant_pH has not been implemented yet!");
     }
  }

  if (n_lambdas == 0)
     return;
 
  MPI_Bcast(v_lambdas[0],n_lambdas*3,MPI_DOUBLE,0,world);
  if (lambda_integration_flags & BUFFER)
     MPI_Bcast(&v_lambda_buff,1,MPI_DOUBLE,0,world);  
   
  for (int i = 0; i < n_lambdas; i++) {
     v_cm += v_lambdas[i][0]*mols_charge_change;
  }
  
  if (lambda_integration_flags & BUFFER) {
     v_cm += N_buff * v_lambda_buff * buff_charge_change;
     v_cm /= (static_cast<double>(n_lambdas)*mols_charge_change + static_cast<double>(N_buff)*buff_charge_change);
  }
  else
     v_cm /= (static_cast<double>(n_lambdas)*mols_charge_change);
     
  for (int i = 0; i < n_lambdas; i++)
     v_lambdas[i][0] -= v_cm;
  if (lambda_integration_flags & BUFFER)
     v_lambda_buff -= v_cm; 
  
  fix_constant_pH->reset_params(x_lambdas,v_lambdas,a_lambdas,m_lambdas);

  if (lambda_integration_flags & BUFFER) fix_constant_pH->reset_buff_params(x_lambda_buff,v_lambda_buff,a_lambda_buff, m_lambda_buff);
}

/* ---------------------------------------------------------------------
   applies the shake algorithm to the sum of the lambdas 

   It adds a  constraint according to the Donnine et al JCTC 2016 
   equation (13).

   The constraint equations were taken from the Tuckerman statistical mechanics
   book 2nd edition pages 106.

   Just one step of the shake iteration is enough as the constraint is very simple.
   
   --------------------------------------------------------------------- */
   
template <int mode>
void FixNHConstantPH::constrain_lambdas()
{
   double omega = 0.0;
   double domega;
   double etol = 1e-6;
   double q_total;
   double sigma_lambda;
   double sigma_mass_inverse;
   double N_buff_double = static_cast<double>(N_buff);
   

   int maxCycles = 10000;
   int cycle = 0;
   
   
   
   /* The while(true) loop was used on purpose so that even when the loop termination condition
      is satisfied the q_total is calculated for the last time with final values of lambdas */
   while(true) {
      // Just doing this on the root and then broadcasting the results
      sigma_lambda = 0.0;
      sigma_mass_inverse = 0.0;
      
      fix_constant_pH->return_nparams(n_lambdas);
      fix_constant_pH->return_params(x_lambdas,v_lambdas,a_lambdas,m_lambdas);
      fix_constant_pH->return_buff_params(x_lambda_buff,v_lambda_buff,a_lambda_buff,m_lambda_buff,N_buff);
      
      /* Checking if the charge content of the N_buff is large enough for n_lambdas
       * Since there is a possibility that the n_lambdas change during the simulation by 
       * the fix_adaptive_protonation.cpp command, the check should be done here. 
       */
      
      if (cycle == 0 && comm->me == 0) 
         if (N_buff < mols_charge_change*n_lambdas)
            error->one(FLERR,"The charge content of N_buff is not large enough for n_lambdas: Please increase the N_buff");
      
      for (int i = 0; i < n_lambdas; i++) {
         sigma_lambda += x_lambdas[i][0];
         if (m_lambdas[i][0] == 0) error->all(FLERR,"m_lambdas[{},0] is zero in fix_nh_constant_pH",i);
         sigma_mass_inverse += (1.0/m_lambdas[i][0]);
      }

      if (m_lambda_buff == 0) error->all(FLERR,"Buffer mass is zero in fix_nh_constant_pH");
      
      if (mode == 1)
         q_total = mols_charge_change*sigma_lambda+buff_charge_change*N_buff_double*x_lambda_buff-total_charge;
      else if (mode == 2) 
         q_total = compute_q_total();
      else error->one(FLERR,"You should never have reached here!!!");


      if (std::abs(q_total) < etol || cycle++ > maxCycles) {
         if (comm->me == 0 && cycle > maxCycles)
             error->warning(FLERR,"The charge constrain did not reach convergence after {} iterations",maxCycles);
         break;
      }
      
      domega = -q_total \ 
         / (mols_charge_change*mols_charge_change*sigma_mass_inverse + (N_buff_double*buff_charge_change*buff_charge_change/m_lambda_buff));

      omega += domega;
      
      for (int i = 0; i < n_lambdas; i++)
         x_lambdas[i][0] += (omega * mols_charge_change / m_lambdas[i][0]);

      x_lambda_buff += buff_charge_change * omega / m_lambda_buff;
     

      fix_constant_pH->reset_params(x_lambdas,v_lambdas,a_lambdas,m_lambdas,0);
      fix_constant_pH->reset_buff_params(x_lambda_buff,v_lambda_buff,a_lambda_buff, m_lambda_buff,0);
      fix_constant_pH->reset_qs();
   }
   
   fix_constant_pH->reset_params(x_lambdas,v_lambdas,a_lambdas,m_lambdas);
   fix_constant_pH->reset_buff_params(x_lambda_buff,v_lambda_buff,a_lambda_buff, m_lambda_buff);
   fix_constant_pH->reset_qs();
}

/* ----------------------------------------------------------------------
   computes the q_total to be used in the constrain_lambdas() function
   ---------------------------------------------------------------------- */
double FixNHConstantPH::compute_q_total()
{
   double * q = atom->q;
   double nlocal = atom->nlocal;
   double q_local = 0.0;
   double q_total = 0.0;
   double tolerance = 1e-6; //0.001;

   for (int i = 0; i <nlocal; i++)
      q_local += q[i];

   MPI_Allreduce(&q_local,&q_total,1,MPI_DOUBLE,MPI_SUM,world);
   
   return q_total;
}

/* ----------------------------------------------------------------------
   random number generator
   ---------------------------------------------------------------------- */

double FixNHConstantPH::random_normal(double mean, double stddev)
{
  static std::mt19937 generator(std::random_device{}());
  std::normal_distribution<double> distribution(mean, stddev);
  return distribution(generator);
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double FixNHConstantPH::memory_usage()
{
  double bytes = 0.0;
  bytes += 4.0*n_lambdas*sizeof(double); // x_lambdas, v_lambdas, a_lambdas and m_lambdas
  if (irregular) bytes += irregular->memory_usage();
  return bytes;
}
