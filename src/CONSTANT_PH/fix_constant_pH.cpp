// clang-format on
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

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "input.h"
#include "kspace.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "pair.h"
#include "random_park.h"
#include "timer.h"
#include "update.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <array>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum {
  NONE = 0,
  BUFFER = 1 << 0,
  ADAPTIVE = 1 << 1,
  ZEROCHARGE = 1 << 2,
  CONSTRAIN = 1 << 3,
  COMMANDS = 1 << 4,
  INTERMEDIATE = 1 << 5,
  INTERPOLATION = 1 << 6
};

enum {
  NONE_FP = 0,
  LAMBDA_FP = 1 << 0,
  V_LAMBDA_FP = 1 << 1,
  A_LAMBDA_FP = 1 << 2,
  H_LAMBDA_FP = 1 << 3,
  LAMBDA_S_FP = 1 << 4,
  HA_LAMBDA_FP = 1 << 5,
  HB_LAMBDA_FP = 1 << 6
};

static constexpr double tol = 1e-5;
static constexpr double max_lambda_buff_0 = 1.05;
static constexpr double min_lambda = -0.1;
static constexpr double max_lambda = 1.1;
static constexpr double environment_coupling = 1.0;

/* ---------------------------------------------------------------------- */

FixConstantPH::FixConstantPH(LAMMPS *lmp, int narg, char **arg) :
    Fix{lmp, narg, arg},
    flags{0}, 
    ncommands{0}, mu{0.0},
    random_number_seed{1152}, 
    lambda_masses{{20.0,20.0}}, 
    GFF_flag{false}, GFF{nullptr},
    print_Udwp_flag{false},
    qHWs{0.278}, qOWs{-0.834}, 
    fix_adaptive_protonation{nullptr},
    fp_flags{0}, write_lambda_nevery{1}, 
    fixgpu{nullptr}, 
    q_orig{nullptr}, f_orig{nullptr}, peatom_orig{nullptr}, pvatom_orig{nullptr}, keatom_orig{nullptr}, kvatom_orig{nullptr}
{
  if (narg < 9) utils::missing_cmd_args(FLERR, "fix constant_pH", error);

  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nevery <= 0) error->all(FLERR, "Illegal fix constant_pH every value {}", nevery);

  //files that contain the charges before and after protonation/deprotonation
  fileName1 = arg[4];
  fileName2 = arg[5];

  pK = utils::numeric(FLERR, arg[6], false, lmp);
  pH = utils::numeric(FLERR, arg[7], false, lmp);
  T  = utils::numeric(FLERR, arg[8], false, lmp);


  int iarg = 9;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "GFF") == 0) {
      GFF_flag = true;
      if (comm->me == 0) {
        fp.open(arg[iarg + 1], std::ifstream::in);
        if (!fp.is_open())
          error->one(FLERR, "Cannot find fix constant_pH the GFF correction file {}", arg[iarg+1]);
      }
      iarg += 2;
    } else if (strcmp(arg[iarg], "Print_Udwp") == 0) {
      print_Udwp_flag = true;
      if (comm->me == 0) {
        Udwp_fp.open(arg[iarg + 1], std::ofstream::out);
        if (!Udwp_fp.is_open())
          error->one(FLERR, "Cannot open fix constant_pH the Print_Udwp file {} for printing",arg[iarg+1]);
      }
      iarg += 2;
    } else if (strcmp(arg[iarg], "molids") == 0) {
      n_lambdas_input = utils::inumeric(FLERR, arg[iarg+1], false, lmp);
      if (flags & ADAPTIVE)
        error->all(FLERR, "molids and Fix_adapative_protonation cannot be used at the same time");
      iarg += 2;
      molids_input = std::make_unique<int[]>(n_lambdas_input);
      if (narg < iarg + n_lambdas_input) utils::missing_cmd_args(FLERR,"fix constant_pH",error);
      for (int i = 0; i < n_lambdas_input; i++) {
        molids_input[i] = utils::inumeric(FLERR, arg[iarg], false, lmp);
        iarg++;
      }
    } else if (strcmp(arg[iarg], "mu") == 0) {
      if (narg < iarg + 2) utils::missing_cmd_args(FLERR, "fix constant_pH", error);
      mu = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "buffer") == 0) {
      flags |= BUFFER;
      if (narg < iarg + 6) utils::missing_cmd_args(FLERR, "fix constant_pH", error);
      N_buff  = utils::inumeric(FLERR, arg[iarg+1], false, lmp);
      typeOWs = utils::inumeric(FLERR, arg[iarg+2], false, lmp);
      typeHWs = utils::inumeric(FLERR, arg[iarg+3], false, lmp);
      qOWs = utils::numeric(FLERR, arg[iarg+4], false, lmp);
      qHWs = utils::numeric(FLERR, arg[iarg+5], false, lmp);
      if (typeOWs > atom->ntypes)
        error->all(FLERR, "Illegal fix constant_pH atom type {}", typeOWs);
      if (typeHWs > atom->ntypes)
        error->all(FLERR, "Illegal fix constant_pH atom type {}", typeHWs);
      iarg += 6;
    } else if (strcmp(arg[iarg], "Fix_adaptive_protonation") == 0) {
      flags |= ADAPTIVE;
      if (molids_input)
        error->all(FLERR, "molids and Fix_adapative_protonation cannot be used at the same time");
      if (narg < iarg + 3) utils::missing_cmd_args(FLERR, "fix constant_pH", error);
      fix_adaptive_protonation_id = std::string(arg[iarg + 1]);
      nevery_fix_adaptive = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      fix_adaptive_protonation = dynamic_cast<FixAdaptiveProtonation *>(
          modify->get_fix_by_id(fix_adaptive_protonation_id.c_str()));
      if (!fix_adaptive_protonation)
        error->all(FLERR, "Wrong fix type in the adaptive keyword for the constant pH");
      iarg += 3;
    } else if (strcmp(arg[iarg], "constrain") == 0) {
      flags |= CONSTRAIN;
      iarg++;
    } else if (strcmp(arg[iarg], "seed") == 0) {
      random_number_seed = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "zero_total_charge") == 0) {
      flags |= ZEROCHARGE;
      iarg++;
    } else if (strcmp(arg[iarg],"write_lambda_nevery") == 0) {
      write_lambda_nevery = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg+=2;
    } else if (strcmp(arg[iarg], "lambda_file") == 0) {
      fp_flags |= LAMBDA_FP;
      if (comm->me == 0) lambda_fp.open(arg[iarg + 1], std::ofstream::out);
      iarg += 2;
    }  else if (strcmp(arg[iarg], "v_lambda_file") == 0) {
      fp_flags |= V_LAMBDA_FP;
      if (comm->me == 0) v_lambda_fp.open(arg[iarg + 1], std::ofstream::out);
      iarg += 2;
    } else if (strcmp(arg[iarg], "a_lambda_file") == 0) {
      fp_flags |= A_LAMBDA_FP;
      if (comm->me == 0) a_lambda_fp.open(arg[iarg + 1], std::ofstream::out);
      iarg += 2;
    } else if (strcmp(arg[iarg], "H_lambda_file") == 0) {
      fp_flags |= H_LAMBDA_FP;
      if (comm->me == 0) H_lambda_fp.open(arg[iarg + 1], std::ofstream::out);
      iarg += 2;
    } else if (strcmp(arg[iarg],"HA_lambda_file") == 0) {
      fp_flags |= HA_LAMBDA_FP;
      if (comm->me == 0) HA_lambda_fp.open(arg[iarg + 1],std::ofstream::out);
      iarg += 2;
    } else if (strcmp(arg[iarg],"HB_lambda_file") == 0) {
      fp_flags |= HB_LAMBDA_FP;
      if (comm->me == 0) HB_lambda_fp.open(arg[iarg + 1],std::ofstream::out);
      iarg += 2;
    } else if (strcmp(arg[iarg], "lambda_s_file") == 0) {
      if (narg < iarg + 3) utils::missing_cmd_args(FLERR, "fix constant_pH", error);
      fp_flags |= LAMBDA_S_FP;
      if (comm->me == 0) {
        lambda_1_fp.open(arg[iarg + 1], std::ofstream::out);
        lambda_2_fp.open(arg[iarg + 2], std::ofstream::out);
        if (!lambda_1_fp.is_open() || !lambda_2_fp.is_open())
          error->one(FLERR,"Cannot open lambda_s_file outputs: {},{}",arg[iarg+1],arg[iarg+2]);
      }
      iarg += 3;
    } else if (strcmp(arg[iarg], "commands") == 0) {
      flags |= COMMANDS;
      if (comm->me == 0) {
        commandsFile.open(arg[iarg + 1], std::ifstream::in);
        if (!commandsFile.is_open()) error->one(FLERR, "Unable to open the commands file");
      }
      read_commands_file();
      iarg += 2;
    } else if (strcmp(arg[iarg], "intermediate_file") == 0) {
      flags |= INTERMEDIATE;
      if (comm->me == 0) { intermediate_file_name = arg[iarg + 1]; }
      iarg += 2;
    } else if (strcmp(arg[iarg],"m_lambda") == 0)  {
      if (narg < iarg + 3) utils::missing_cmd_args(FLERR,"fix constant_pH",error);
      lambda_masses[0] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      lambda_masses[1] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      iarg += 3;
    } else if (strcmp(arg[iarg],"interpolation") == 0) {
      flags |= INTERPOLATION;
      iarg += 1;
    } else {
      error->all(FLERR, "Unknown fix constant_pH keyword: {}", arg[iarg]);
    }
  }

  if (!(flags & ADAPTIVE) && (flags & COMMANDS))
    error->warning(FLERR,
                   "The keyword \"commands\" has been used without the keyword \"adaptive\"");
  if (write_lambda_nevery == 1 && comm->me == 0)
    error->warning(FLERR,"The default value of write_lambda_nevery leads to large output files in long simulations!");


  array_flag = 1;
  size_array_rows = 13;
  size_array_cols = 3 * n_lambdas_input + ((flags & BUFFER) ? 1 : 0);
  peratom_flag = 1;
  size_peratom_cols = 0;
  peratom_freq = nevery;
  extarray = 0;

  atom->add_callback(Atom::GROW);
  //atom->add_callback(Atom::COPY);
  atom->add_callback(Atom::BORDER);
}

/* ---------------------------------------------------------------------- */

FixConstantPH::~FixConstantPH()
{
  // According to RAII I do not need to close std::ifstreams
 
  // Since it is allocated with lmp->utils->strdup, it must be deallocated with delete []

  if (GFF) memory->destroy(GFF);

  // deallocate the memories with size dependent on the n_lambda
  delete_lambdas();

  // deallocate memories whose size is dependent on natoms
  deallocate_storage();

  atom->delete_callback(id,Atom::GROW);
  //atom->delete_callback(id,Atom::COPY);
   
  atom->delete_callback(id,Atom::BORDER);

}

/* ----------------------------------------------------------------------

   ---------------------------------------------------------------------- */

int FixConstantPH::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;    // Calculates the a_lambda
  mask |= POST_FORCE;    // Updates the a_lambda before the second step of the velocity verlet
  return mask;
}

/* ----------------------------------------------------------------------
   Setup
   ---------------------------------------------------------------------- */

void FixConstantPH::init()
{
  // default values from Donnini, Ullmann, J Chem Theory Comput 2016 - Table S2
  w = 1000; //200;
  s = 0.3;//0.3;
  h = 10; //7.0;
  k = 6.267; //4.417;       //2.553;
  a = 0.05130; //0.04208;     //0.03401;
  b = 0.001411; //0.002957;    //0.005238;
  r = 21.428;//16.458;
  m = 0.1078;//0.1507;
  d = 5.0; //3.50;    //2.0; //The height of the barrier is 2*d

  // default values for the buffer potential with h = 0 from Donnin J Chem Theory Comput 2016 - Table S2
  w_buff = 1000;
  s_buff = 0.3;
  h_buff = 0.0;
  k_buff = 0.0;
  a_buff = 0.04764;
  b_buff = -0.09706;
  r_buff = 16.458;
  m_buff = 0.70; //0.1507; Increased the m so that there is no force on the lambda buff at 1.0
  d_buff = 0.0;

  // Reading the pH structure files
  pH_structure_storage = std::make_unique<constant_pH_structures>(lmp, fileName1, fileName2);
  pH_structure_storage->read_pH_structure_files();

  /*
   * Allocating the storage so that the copy_arrays called
   * in the Verlet::Setup would not lead to out of range.
   * This setup method is called before the fix_constant_pH::setup
   */
  allocate_storage();

  // setting the HCalcNSteps to zero
  HCalcNSteps = 0;

  // dynamic states for lambdas
  pH_state = std::make_unique<constant_pH_state>(lmp,molids_input,n_lambdas_input,lambda_masses,N_buff);

   // forcefield variables for lambdas
  set_lambdas();
}

/* ---------------------------------------------------------------------- */

void FixConstantPH::setup(int /*vflag*/)
{
  // Getting the required parameters from the pH_state variable.
  double** lambdas = pH_state->lambdas;
  double& lambda_buff = pH_state->lambda_buff;
  double& v_lambda_buff = pH_state->v_lambda_buff;
  int& n_lambdas = pH_state->n_lambdas;
  int& N_buff = pH_state->N_buff; 

  // Checking if we have correct number of hydronium ions
  if (flags & BUFFER) check_num_OWs_HWs();

  fixgpu = modify->get_fix_by_id("package_gpu");

  /* 
   * As it is hypothesized that the initial values for 
   * lambdas are zero the initial value for the lambda_buff
   * should be one so there is enough protons to be exchanged
   * between the lambdas and buffer due to the contraint on
   * the lambas[0] + ... + lambdas[n] + lambda_buff
   */

  if (GFF_flag) init_GFF();

  if (print_Udwp_flag) print_Udwp();



  // I have put this part here on purpose so if the fix_adaptive_protonation reads the initial molids, it is set here
  if (flags & ADAPTIVE) { 
    //
    // The get_protonable_molids method of the fix_adaptive_protonation class
    // itself allocates the molids_input with n_lambdas so no need for manual allocation here!
    fix_adaptive_protonation->get_protonable_molids(molids_input);
    // the molids_input does not have any info on its size.
    fix_adaptive_protonation->get_n_protonable(n_lambdas_input);
  }

  if (flags & BUFFER) {
    lambda_buff = lambda_buff_0;
    modify_q_buff(lambda_buff);
    double q_total = compute_q_total(true);
    lambda_buff = lambda_buff_0 - q_total/static_cast<double>(N_buff);

    if (lambda_buff >= max_lambda_buff_0 ) {
      double dlambda_buff = lambda_buff - max_lambda_buff_0;
      lambda_buff = max_lambda_buff_0 ;
      error->warning(FLERR,"Reducing the lambda_buff by {} through increase the lambda values.. The simulation might become unstable!",dlambda_buff);
      double dlambda = static_cast<double>(N_buff)*dlambda_buff /static_cast<double>(n_lambdas);
      for (int i = 0; i < n_lambdas; i++)
        lambdas[i][0] += dlambda;
    }
    v_lambda_buff = 0.0;
    reset_qs();
    compute_q_total();
  }

  if (fp_flags != NONE_FP) write_lambdas_header();
}

/* ----------------------------------------------------------------------
   This part calculates the acceleration of the lambdas parameter
   which is obtained from the force acting on it
   ---------------------------------------------------------------------- */

void FixConstantPH::initial_integrate(int /*vflag*/)
{
  if (flags & ADAPTIVE) {
    bigint endstep_backup = update->endstep;
    if (!(update->ntimestep % nevery_fix_adaptive)) {
      int n_changes;
      fix_adaptive_protonation->get_n_changes(n_changes);
      if (n_changes) {
        /* If there is a minimization command
             * , the update->endstep is set to zero
             * which causes the t_target to be inf.
             * So, I have backed up the update->endstep
             */
        endstep_backup = update->endstep;
        /* Writing the molids in a file to be read by fix_adaptive_protonation afterwards */
        fix_adaptive_protonation->write_molids(intermediate_file_name);
        // add those commands ------>
        if (flags & COMMANDS) {
          modify->clearstep_compute();
          for (int i = 0; i < ncommands; i++) {
            input->one(commands[i]);
            modify->addstep_compute(update->ntimestep);    // I am not sure about this part yet!
          }

          // Since there is a possibly to having the fix_adaptive_protonation deleted in 
          // the commands we need to retrieve it again.
          fix_adaptive_protonation = dynamic_cast<FixAdaptiveProtonation*>(modify->get_fix_by_id(fix_adaptive_protonation_id.c_str()));
          if (!fix_adaptive_protonation)
            error->all(FLERR, "Wrong fix type in the adaptive keyword for the constant pH");
        }
        // <------ add those commands

        // Updating the endstep
        update->endstep = endstep_backup;

        /*
         * Backing up lambdas, v_lambdas, a_lambdas,
         * m_lambdas and H_lambdas
         * If a lambda remains the same during fix_adaptive_protonation,
         * I do not want to reset their lambdas, v_lambdas, a_lambdas and m_lambdas;
         */
        pH_state_prev = std::move(pH_state);

        /* 
         *  we reread the n_lambdas after backing up the lambdas
         */
        // get_protonable_molids itself allocates the required storage.
        fix_adaptive_protonation->get_protonable_molids(molids_input);
        // The molids_input even though allocated by the fix_adaptive_protonation
        // it does not include any information regarding its size
        fix_adaptive_protonation->get_n_protonable(n_lambdas_input);

        pH_state = std::make_unique<constant_pH_state>(lmp,molids_input,
          n_lambdas_input,lambda_masses,N_buff);
        // forcefield variables for lambdas
        // This part used the pH_state_prev to keep the lambdas available in the previous step.
        set_lambdas(); 

        modify->clearstep_compute();
        modify->addstep_compute(update->ntimestep);

        if (fp_flags != NONE_FP) write_lambdas_header();
      }
    }
  }

  calculate_dfs();
  calculate_dUs();
  calculate_Hs();
  update_a_lambda();

  // priting the precent of steps where the HA and HB values have ben calculated
  if (update->ntimestep && (update->ntimestep % 1000 == 0)) {
    double ratio = 100.0*static_cast<double>(HCalcNSteps)/1000.0;

    // priting the info 
    if (comm->me == 0) {
      auto mesg = fmt::format("HA and HB were updated {:.2f}% of simulation time from fix_constant_pH\n",ratio);
      utils::logmesg(lmp,mesg);
    }
    HCalcNSteps = 0;
  }
}

/* ----------------------------------------------------------------------
   The second step of the integration 
   ----------------------------------------------------------------------  */

void FixConstantPH::post_force(int /*vflag*/)
{
  calculate_dfs();
  calculate_dUs();
  //calculate_Hs();
  update_a_lambda();
  if (!(update->ntimestep % write_lambda_nevery)) write_lambdas();
}

/* ----------------------------------------------------------------------
   This function deallocates the storage for memories whose sizes are 
   dependent on the n_lambdas
   ----------------------------------------------------------------------  */

void FixConstantPH::delete_lambdas()
{
  HAs.reset();
  HBs.reset();
  fs.reset();
  dfs.reset();
  Us.reset();
  dUs.reset();
  lambdas_j.reset();
  GFF_lambdas.reset();
  H_lambdas.reset();
}

/* ----------------------------------------------------------------------
   This function allocates the storage for memories whose sizes are 
   dependent on the n_lambdas
   ----------------------------------------------------------------------  */

void FixConstantPH::set_lambdas()
{
  const int& n_lambdas = pH_state->n_lambdas;
  HAs = std::make_unique<double[]>(n_lambdas);
  HBs = std::make_unique<double[]>(n_lambdas);
  fs  = std::make_unique<double[]>(n_lambdas);
  dfs = std::make_unique<double[]>(n_lambdas);
  Us  = std::make_unique<double[]>(n_lambdas);
  dUs = std::make_unique<double[]>(n_lambdas);
  lambdas_j   = std::make_unique<double[]>(n_lambdas);
  GFF_lambdas = std::make_unique<double[]>(n_lambdas);
  H_lambdas   = std::make_unique<double[]>(n_lambdas);

  int to = pH_state->reset_lambdas(pH_state_prev);

  if (n_lambdas) {
    // Initializing lambdas based on the current charge of protonable molecules so there is no jump in the system total charge
    initialize_lambda(to);
    // This would not work in the initialize section as the m_lambda has not been set yet!
    // initialize_v_lambda(this->T,0);
    initialize_v_lambda(this->T,to);
  }

  // Resetting the vector_atom to the default value
  int nmax = atom->nmax;
  std::fill_n(vector_atom,nmax,-1);
}

/* ----------------------------------------------------------------------
   Initializing lambdas based on the current charge of protonable molecules 
   so there is no jump in the system total charge
   ---------------------------------------------------------------------- */

void FixConstantPH::initialize_lambda(const int& to)
{
  const int nlocal     = atom->nlocal;
  double *q            = atom->q;
  int *type            = atom->type;
  int *molecule        = atom->molecule;
  const int& n_lambdas = pH_state->n_lambdas;
  const int length     = n_lambdas - to;

  if (length <= 0) return;

  // These three are not safe for the pH*qs I should
  // use the mdspan with std::unique_ptr and 
  // for the protonable I have to use std::unique_ptr
  // and set up the get functions to return a cons ref
  // to them as std::unique_ptr is not copyable.
  double **pH1qs  = pH_structure_storage->pH1qs;
  double **pH2qs  = pH_structure_storage->pH2qs;
  int *protonable = pH_structure_storage->protonable.get(); 

  double **lambdas = pH_state->lambdas;
  auto& molids     = pH_state->molids;
  auto& lambda_buff = pH_state->lambda_buff;


  auto q_local     = std::make_unique<double []>(length);
  auto q_local_pH1 = std::make_unique<double []>(length);
  auto q_local_pH2 = std::make_unique<double []>(length);
  auto q_total     = std::make_unique<double []>(length);
  auto q_total_pH1 = std::make_unique<double []>(length);
  auto q_total_pH2 = std::make_unique<double []>(length);

  std::fill_n(q_local.get(),length,0.0);
  std::fill_n(q_local_pH1.get(),length,0.0);
  std::fill_n(q_local_pH2.get(),length,0.0);

  for (int i = 0; i < nlocal; i++) {
    const int type_i = type[i];
    if (!protonable[type_i])
      continue;
    for (int j = to; j < n_lambdas; j++) {
      const int indx = j - to;
      if (molecule[i] == molids[j]) {
        q_local[indx]     += q[i];
        q_local_pH1[indx] += pH1qs[type_i][0];
        q_local_pH2[indx] += pH2qs[type_i][0];
      }
    }
  }

  MPI_Allreduce(q_local.get(),q_total.get(),length,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(q_local_pH1.get(),q_total_pH1.get(),length,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(q_local_pH2.get(),q_total_pH2.get(),length,MPI_DOUBLE,MPI_SUM,world);

  for (int j = 0; j < length; j++) {
    double denom = q_total_pH2[j] - q_total_pH1[j];
    if (std::abs(denom) < tol) {
      lambdas[to+j][0] = 0.0;
      continue;
    }
    double lambda_j = (q_total[j]-q_total_pH1[j])/denom;
    if (comm->me == 0 && (lambda_j < min_lambda || lambda_j > max_lambda)) {
      error->warning(FLERR,"out of range value for the initialization of the lambda_{}=={}, The simulation might crash!",j,lambda_j);
      lambdas[to+j][0] = std::clamp(lambda_j,min_lambda,max_lambda);
    } 
    else
      lambdas[to+j][0] = lambda_j;
  }

  // Neutralizing the simulation box just in case. 
  //lambda_buff = neutralize();
}

/* ---------------------------------------------------------------------- */

void FixConstantPH::update_a_lambda()
{
  if (GFF_flag) calculate_GFFs();
  double mvv2e = 1.0; // force->mvv2e;
  double kcal2kj = 4.184;
  double kT = 0.008314*T; // force->boltz * T;
  double aUnit = 1e-6; // convert the acceleration form ps^-2 to fs^-2
  double nStructures1Barrier = 0.5 * kT;
  double nStructures2Barrier = 0.5 * kT; 

  double** lambdas       = pH_state->lambdas;
  double** v_lambdas     = pH_state->v_lambdas;
  double** a_lambdas     = pH_state->a_lambdas;
  double** m_lambdas     = pH_state->m_lambdas;
  double&  lambda_buff   = pH_state->lambda_buff;
  double&  v_lambda_buff = pH_state->v_lambda_buff;
  double&  a_lambda_buff = pH_state->a_lambda_buff;
  double&  m_lambda_buff = pH_state->m_lambda_buff;
  int&     n_lambdas     = pH_state->n_lambdas;

  int pHnStructures1 = pH_structure_storage->pHnStructures1;
  int pHnStructures2 = pH_structure_storage->pHnStructures2;


  for (int i = 0; i < n_lambdas; i++) {
    double f_lambda_0 = -(environment_coupling*kcal2kj*(HAs[i] - HBs[i]) -dfs[i] * kT * log(10) * (pK - pH) + dUs[i] - GFF_lambdas[i]);    
    // The df sign should be positive if the lambda = 0 is for the protonated state
    double f_lambda_1 = 2 * M_PI * nStructures1Barrier * pHnStructures1 *
        sin(2 * M_PI * pHnStructures1 * lambdas[i][1]);
    double f_lambda_2 = 2 * M_PI * nStructures2Barrier * pHnStructures2 *
        sin(2 * M_PI * pHnStructures2 * lambdas[i][2]);

    a_lambdas[i][0] = aUnit * f_lambda_0 / m_lambdas[i][0]; 
    a_lambdas[i][1] = f_lambda_1 / m_lambdas[i][1];
    a_lambdas[i][2] = f_lambda_2 / m_lambdas[i][2];

    // I am not sure about the sign of the f*kT*log(10)*(pK-pH)
    this->H_lambdas[i] = environment_coupling*kcal2kj*(lambdas[i][0]*HAs[i] + (1.0-lambdas[i][0])*HBs[i]) - fs[i] * kT * log(10) * (pK - pH) + Us[i]; 
        // + (m_lambdas[i][0] / 2.0) * (v_lambdas[i][0] * v_lambdas[i][0]) * (force->mvv2e);    
      // This might not be needed. May be I need to tally this into energies.
    // I might need to use the leap-frog integrator and so this function might need to be in other functions than postforce()
  }

  if (flags & BUFFER) {
    double f_lambda_buff = - dU_buff / static_cast<double>(N_buff);
    a_lambda_buff = aUnit * 
        f_lambda_buff / m_lambda_buff;    // the fix_nh_constant_pH itself takes care of units
    this->H_lambda_buff =  
         U_buff + 0.0*N_buff * (m_lambda_buff / 2.0) * (v_lambda_buff * v_lambda_buff) * (force->mvv2e);
  }
}

/* ----------------------------------------------------------------------- 
    This function is called by compute_GFF for thermodynamic integration 
    of the GFF value.
   ----------------------------------------------------------------------- */

void FixConstantPH::calculate_H_once()
{
  calculate_dfs();
  calculate_dUs();
  update_a_lambda();
}

/* ----------------------------------------------------------------------
   returns the number of the lambda parameters
  ----------------------------------------------------------------------- */

void FixConstantPH::return_nparams(int &_n_params) const
{
  _n_params = this->pH_state->n_lambdas;
}

/* ----------------------------------------------------------------------
   returns the x_lambdas, v_lambdas, ....
   The memories for these should be allocated before hands
   ---------------------------------------------------------------------- */

void FixConstantPH::return_params(std::unique_ptr<constant_pH_state>& pH_state_) const
{
  pH_state_ = std::make_unique<constant_pH_state>(*pH_state);
} 


/* ---------------------------------------------------------------------
    This function sets the value of qs based on the value of lambdas()
    and lambda_buffs() calculated by the nve_x() fuction of the 
    fix_nh_constant_pH. Accordingly, this function should be called from
    the last line of the nve_x() fuction.
   --------------------------------------------------------------------- */

void FixConstantPH::reset_qs()
{
  double** lambdas = pH_state->lambdas;
  auto& lambda_buff = pH_state->lambda_buff;
  modify_qs(lambdas);

  if (flags & BUFFER) modify_q_buff(lambda_buff);

  /* This should be here just for debugging
       since it used MPI_Allreduce to calculate
       the total charge it has some overhead not 
       advised in the production run
    */
  /*
       There is no need for this anymore 
       since the q_total = sigma_lambdas * mol_charge_change + N_buff* lambda_buff*buff_charge_change
    */
  if (0) { compute_q_total(); }
}

/* ---------------------------------------------------------------------
    This function returns the H_lambdas
   --------------------------------------------------------------------- */

void FixConstantPH::return_H_lambdas(double *_H_lambdas) const
{
  auto& n_lambdas = pH_state->n_lambdas;
  for (int i = 0; i < n_lambdas; i++) _H_lambdas[i] = H_lambdas[i];
}

/* ---------------------------------------------------------------------
    This one just returns the value of T_lambda
   --------------------------------------------------------------------- */

void FixConstantPH::return_T_lambda(double &_T_lambda, int component)
{
  if (component < 0 || component > 2) error->one(FLERR, "Illegal function input");
  //calculate_T_lambda();
  _T_lambda = this->T_lambdas[component];
}

/* ---------------------------------------------------------------------
    sets the values of the x_lambdas, v_lambdas, ... possibly by the intergrating
    fix styles
    --------------------------------------------------------------------- */

void FixConstantPH::reset_params(const std::unique_ptr<constant_pH_state>& pH_state_, const int mode)
{
  pH_state = std::make_unique<constant_pH_state>(*pH_state_);
  if (mode == 1)
    pH_state->broadcast();
}

void FixConstantPH::reset_params(std::unique_ptr<constant_pH_state>&& pH_state_, const int mode)
{
  pH_state = std::move(pH_state_);
  if (mode == 1)
    pH_state->broadcast();
}


/* ----------------------------------------------------------------------
    Reading the file containing the commands run whenever a lambdas array 
    is modified.
   ---------------------------------------------------------------------- */
void FixConstantPH::read_commands_file()
{
  using std::getline, std::string;
  /*
    * The file format
    * ncommands
    * comment-#1
    * comment-#2
    * command-#1
    * command-#2
    * command-#3
    * command-#4
    * ...
    * command-#n
    */

  string line;
  if (comm->me == 0) {
    if (!getline(commandsFile, line)) error->one(FLERR, "Error reading commands file");
    std::stringstream iss(line);

    iss >> ncommands;
  }

  MPI_Bcast(&ncommands, 1, MPI_INT, 0, world);
  commands = std::make_unique<string []>(ncommands);


  if (comm->me == 0) {
    getline(commandsFile, line);    // comment-#1
    getline(commandsFile, line);    // comment-#2

    for (int i = 0; i < ncommands; i++) {
      if (!getline(commandsFile, line)) error->one(FLERR, "Error reading commands lines");
      commands[i] = line;
    }
  }


  for (int i = 0; i < ncommands; i++)
  {
    int len = (comm->me == 0)?commands[i].size()+1:0;
    MPI_Bcast(&len,1,MPI_INT,0,world);

    std::vector<char> buffer(len);
    if (comm->me == 0)
      std::memcpy(buffer.data(), commands[i].c_str(),len);
    MPI_Bcast(buffer.data(),len,MPI_CHAR,0,world);
    if (comm->me != 0)
      commands[i] = std::string(buffer.data());
  }
}

/* ----------------------------------------------------------------------
   Checking the number of Oxygen and hydrogen atoms of hydronium ions in 
   the simulation box.
   ---------------------------------------------------------------------- */

void FixConstantPH::check_num_OWs_HWs()
{
  int *type = atom->type;
  int nlocal = atom->nlocal;

  std::array<int,2> num_local{0,0};
  std::array<int,2> num_total{0,0};


  for (int i = 0; i < nlocal; i++) {
    if (type[i] == typeHWs) num_local[0]++;
    if (type[i] == typeOWs) num_local[1]++;
  }

  MPI_Allreduce(num_local.data(), num_total.data(), 2, MPI_INT, MPI_SUM, world);
  num_HWs = num_total[0];
  num_OWs = num_total[1];

  if (comm->me == 0) {
    if (num_HWs != 3 * num_OWs)
      error->one(FLERR,
               "Number of HWs in the fix constant pH {} is not three times the number of OWs {}",
               num_HWs, num_OWs);
    if (num_OWs != N_buff)
      error->one(FLERR, "Wrong number of N_buff in the fix constant pH: {}", N_buff);
  }
}

/* ---------------------------------------------------------------------- */

void FixConstantPH::calculate_dfs()
{
  auto&  n_lambdas = pH_state->n_lambdas;
  double** lambdas = pH_state->lambdas;

  // If pH == pK, everything is zero; skip work.
  if (std::abs(pH -pK) < 1e-12) {
      std::fill(fs.get(),fs.get()+n_lambdas,0.0);
      std::fill(dfs.get(),dfs.get()+n_lambdas,0.0);
      return;
  }

  //Taken from https://gitlab.com/gromacs-constantph/constantph/-/blob/main/gromacs-constantph/src/gromacs/applied_forces/constant_ph/constant_ph.cpp
  const double k_step  = 5.0 * r;   
  const double x0_step  = 2.0 * a;

  auto step = [&](double &x) {
      if (pH < pK)      x = 1.0 / (1.0 + std::exp(-k_step  * (x + x0_step  - 1.0)));
      else /* pH > pK */x = 1.0 / (1.0 + std::exp(-k_step  * (x - x0_step )));
  };

  auto dstep = [&](double s) {
      return k_step  * s * (1.0 - s);
  };
  
  for (int i = 0; i < n_lambdas; i++)
    fs[i] = lambdas[i][0];

  // Map x -> sigma(x) in-place into fs
  std::for_each(fs.get(), fs.get()+n_lambdas, step);

  // Derivative from sigma: k * s * (1 - s)
  std::transform(fs.get(), fs.get()+n_lambdas, dfs.get(), dstep);

}

/* ----------------------------------------------------------------------- */

void FixConstantPH::calculate_dUs()
{
  auto&    n_lambdas   = pH_state->n_lambdas;
  double** lambdas     = pH_state->lambdas;
  double   lambda_buff = pH_state->lambda_buff;

  double U1, U2, U3, U4, U5;
  double dU1, dU2, dU3, dU4, dU5;
  for (int j = 0; j < n_lambdas; j++) {
    U1 = -k * std::exp(-(lambdas[j][0] - 1.0 - mu - b) * (lambdas[j][0] - 1.0 - mu - b) / (2.0 * a * a));
    U2 = -k * std::exp(-(lambdas[j][0] + mu + b) * (lambdas[j][0] + mu + b) / (2.0 * a * a));
    U3 =  d * std::exp(-(lambdas[j][0] - 0.5) * (lambdas[j][0] - 0.5) / (2.0 * s * s));
    U4 = 0.5 * w * (1.0 - std::erf(r * (lambdas[j][0] + m)));
    U5 = 0.5 * w * (1.0 + std::erf(r * (lambdas[j][0] - 1.0 - m)));
    dU1 = -((lambdas[j][0] - 1.0 - mu - b) / (a * a)) * U1;
    dU2 = -((lambdas[j][0] + mu + b) / (a * a)) * U2;
    dU3 = -((lambdas[j][0] - 0.5) / (s * s)) * U3;
    dU4 = -0.5 * w * r * 2 * std::exp(-r * r * (lambdas[j][0] + m) * (lambdas[j][0] + m)) / std::sqrt(M_PI);
    dU5 = 0.5 * w * r * 2 * std::exp(-r * r * (lambdas[j][0] - 1.0 - m) * (lambdas[j][0] - 1.0 - m)) /
        std::sqrt(M_PI);

    Us[j] = U1 + U2 + U3 + U4 + U5;
    dUs[j] = dU1 + dU2 + dU3 + dU4 + dU5;
  }

  if (flags & BUFFER) {
    U1 = -k_buff *
        std::exp(-(lambda_buff - 1.0 - b_buff) * (lambda_buff - 1.0 - b_buff) / (2.0 * a_buff * a_buff));
    U2 = -k_buff * std::exp(-(lambda_buff + b_buff) * (lambda_buff + b_buff) / (2.0 * a_buff * a_buff));
    U3 = d_buff * std::exp(-(lambda_buff - 0.5) * (lambda_buff - 0.5) / (2 * s_buff * s_buff));
    U4 = 0.5 * w_buff * (1.0 - std::erf(r_buff * (lambda_buff + m_buff)));
    U5 = 0.5 * w_buff * (1.0 + std::erf(r_buff * (lambda_buff - 1.0 - m_buff)));
    dU1 = -((lambda_buff - 1.0 - b_buff) / (a_buff * a_buff)) * U1;
    dU2 = -((lambda_buff + b_buff) / (a_buff * a_buff)) * U2;
    dU3 = -((lambda_buff - 0.5) / (s_buff * s_buff)) * U3;
    dU4 = -0.5 * w_buff * r_buff * 2 *
    std::exp(-r_buff * r_buff * (lambda_buff + m_buff) * (lambda_buff + m_buff)) / std::sqrt(M_PI);
    dU5 = 0.5 * w_buff * r_buff * 2 *
    std::exp(-r_buff * r_buff * (lambda_buff - 1.0 - m_buff) * (lambda_buff - 1.0 - m_buff)) /
        std::sqrt(M_PI);

    U_buff = U1 + U2 + U3 + U4 + U5;
    dU_buff = dU1 + dU2 + dU3 + dU4 + dU5;
  }
}

/* ----------------------------------------------------------------------- */

void FixConstantPH::calculate_dU(const double &_lambda, double &_U, double &_dU)
{
  double U1, U2, U3, U4, U5;
  double dU1, dU2, dU3, dU4, dU5;
  U1 = -k * std::exp(-(_lambda - 1 - mu - b) * (_lambda - 1 - mu - b) / (2 * a * a));
  U2 = -k * std::exp(-(_lambda + mu + b) * (_lambda + mu + b) / (2 * a * a));
  U3 = d * std::exp(-(_lambda - 0.5) * (_lambda - 0.5) / (2 * s * s));
  U4 = 0.5 * w * (1 - std::erf(r * (_lambda + m)));
  U5 = 0.5 * w * (1 + std::erf(r * (_lambda - 1 - m)));
  dU1 = -((_lambda - 1 - mu - b) / (a * a)) * U1;
  dU2 = -((_lambda + mu + b) / (a * a)) * U2;
  dU3 = -((_lambda - 0.5) / (s * s)) * U3;
  dU4 = -0.5 * w * r * 2 * std::exp(-r * r * (_lambda + m) * (_lambda + m)) / std::sqrt(M_PI);
  dU5 = 0.5 * w * r * 2 * std::exp(-r * r * (_lambda - 1 - m) * (_lambda - 1 - m)) / std::sqrt(M_PI);

  _U = U1 + U2 + U3 + U4 + U5;
  _dU = dU1 + dU2 + dU3 + dU4 + dU5;
}

/* ---------------------------------------------------------------------- */

void FixConstantPH::print_Udwp()
{
  double lambda_Udwp, U_Udwp, dU_Udwp;

  constexpr int n_points = 100;
  constexpr double dlambda_Udwp = 2.0 / static_cast<double>(n_points);

  lambda_Udwp = -0.5;
  if (comm->me == 0) {
    if (!Udwp_fp.is_open()) error->one(FLERR, "Udwp_fp file stream is not open");

    Udwp_fp << "Lambda,U,dU" << std::endl;
    Udwp_fp << std::fixed << std::setprecision(8);
    for (int i = 0; i <= n_points; i++) {
      calculate_dU(lambda_Udwp, U_Udwp, dU_Udwp);
      Udwp_fp << lambda_Udwp << "," << U_Udwp << "," << dU_Udwp << std::endl;
      lambda_Udwp += dlambda_Udwp;
    }
  }
}

/* ----------------------------------------------------------------------
   manage storage for charge, force, energy, virial arrays
   taken from src/FEP/compute_fep.cpp
------------------------------------------------------------------------- */

void FixConstantPH::allocate_storage()
{
  /* It should be nmax since in the case that 
     the newton flag is on the force in the 
     ghost atoms also must be update and the 
     nmax contains the maximum number of nlocal 
     and nghost atoms.
  */
  nmax = atom->nmax;
  memory->create(q_orig, nmax, "constant_pH:q_orig");
  memory->create(f_orig, nmax, 3, "constant_pH:f_orig");
  memory->create(peatom_orig, nmax, "constant_pH:peatom_orig");
  memory->create(pvatom_orig, nmax, 6, "constant_pH:pvatom_orig");
  if (force->kspace) {
    memory->create(keatom_orig, nmax, "constant_pH:keatom_orig");
    memory->create(kvatom_orig, nmax, 6, "constant_pH:kvatom_orig");
  }

  vector_atom = new double[nmax];
}

/* ---------------------------------------------------------------------- */

void FixConstantPH::deallocate_storage()
{
  if (q_orig) memory->destroy(q_orig);
  if (f_orig) memory->destroy(f_orig);
  if (peatom_orig) memory->destroy(peatom_orig);
  if (pvatom_orig) memory->destroy(pvatom_orig);
  /* If kspace->force is true these two have been already allocated (they are true) and 
     there is no need to check it since the lammps destructor first destructs
     the kspace so that "if (force->kspace)" in the destructor for 
     ComputeFEEConstantPH leads to an error!
  */

  if (keatom_orig) memory->destroy(keatom_orig);
  if (kvatom_orig) memory->destroy(kvatom_orig);

  delete[] vector_atom;

  q_orig = nullptr;
  f_orig = nullptr;
  peatom_orig = nullptr;
  pvatom_orig = nullptr;
  keatom_orig = nullptr;
  kvatom_orig = nullptr;

  vector_atom = nullptr;
}

/* ----------------------------------------------------------------------
   Forward-reverse copy function to be used in backup_restore_qfev()
   ---------------------------------------------------------------------- */

template <int direction> void FixConstantPH::forward_reverse_copy(double &a, double &b)
{
  if (direction == 1) a = b;
  if (direction == -1) b = a;
}

template <int direction> void FixConstantPH::forward_reverse_copy(double *a, double *b, int i)
{
  if (direction == 1) a[i] = b[i];
  if (direction == -1) b[i] = a[i];
}

template <int direction>
void FixConstantPH::forward_reverse_copy(double **a, double **b, int i, int j)
{
  if (direction == 1) a[i][j] = b[i][j];
  if (direction == -1) b[i][j] = a[i][j];
}

/* ----------------------------------------------------------------------
   backup and restore arrays with charge, force, energy, virial
   taken from src/FEP/compute_fep.cpp
   backup ==> direction == 1
   restore ==> direction == -1
------------------------------------------------------------------------- */

template <int direction> void FixConstantPH::backup_restore_qfev()
{
  int i;

  int natom = atom->nlocal;
  if (force->newton || (force->kspace && force->kspace->tip4pflag)) natom += atom->nghost;

  double **f = atom->f;
  for (i = 0; i < natom; i++)
    for (int j = 0; j < 3; j++) forward_reverse_copy<direction>(f_orig, f, i, j);

  double *q = atom->q;
  for (int i = 0; i < natom; i++) forward_reverse_copy<direction>(q_orig, q, i);

  forward_reverse_copy<direction>(eng_vdwl_orig, force->pair->eng_vdwl);
  forward_reverse_copy<direction>(eng_coul_orig, force->pair->eng_coul);

  for (int i = 0; i < 6; i++) forward_reverse_copy<direction>(pvirial_orig, force->pair->virial, i);

  if (update->eflag_atom) {
    double *peatom = force->pair->eatom;
    for (i = 0; i < natom; i++) forward_reverse_copy<direction>(peatom_orig, peatom, i);
  }
  if (update->vflag_atom) {
    double **pvatom = force->pair->vatom;
    for (i = 0; i < natom; i++)
      for (int j = 0; j < 6; j++) forward_reverse_copy<direction>(pvatom_orig, pvatom, i, j);
  }

  if (force->kspace) {
    forward_reverse_copy<direction>(energy_orig, force->kspace->energy);
    for (int j = 0; j < 6; j++)
      forward_reverse_copy<direction>(kvirial_orig, force->kspace->virial, j);

    if (update->eflag_atom) {
      double *keatom = force->kspace->eatom;
      for (i = 0; i < natom; i++) forward_reverse_copy<direction>(keatom_orig, keatom, i);
    }
    if (update->vflag_atom) {
      double **kvatom = force->kspace->vatom;
      for (i = 0; i < natom; i++)
        for (int j = 0; j < 6; j++) forward_reverse_copy<direction>(kvatom_orig, kvatom, i, j);
    }
  }
}

/* --------------------------------------------------------------
   modify just q of one lambda
   Warning: It selects the proper configurations for pH1 and pH2
   based on lambdas[:][1] and lambdas[:][2]!
   -------------------------------------------------------------- */

void FixConstantPH::modify_qs(double scale, int j)
{
  int nlocal = atom->nlocal;
  int *type = atom->type;
  double *q = atom->q;

  int *protonable = pH_structure_storage->protonable
                        .get();    // Not safe, I should use std::shared_ptr instead...
  double **pH1qs = pH_structure_storage->pH1qs;
  double **pH2qs = pH_structure_storage->pH2qs;
  int pHnStructures1 = pH_structure_storage->pHnStructures1;
  int pHnStructures2 = pH_structure_storage->pHnStructures2;

  double **lambdas = pH_state->lambdas;
  auto& molids = pH_state->molids;


  std::unique_ptr<double []> q_changes_local = std::make_unique<double []>(4);
  std::unique_ptr<double []> q_changes = std::make_unique<double []>(4);
  std::fill(q_changes_local.get(),q_changes_local.get()+4,0.0);
  std::fill(q_changes.get(),q_changes.get()+4,0.0);


  double scale0 = scale;

  int indx11 = std::floor(lambdas[j][1] * pHnStructures1 - 0.5);
  int indx12 = std::ceil(lambdas[j][1] * pHnStructures1 - 0.5);
  int indx21 = std::floor(lambdas[j][2] * pHnStructures2 - 0.5);
  int indx22 = std::ceil(lambdas[j][2] * pHnStructures2 - 0.5);

  // Wrapping around 
  while (indx11 < 0) indx11 += pHnStructures1;
  while (indx12 < 0) indx12 += pHnStructures1;
  while (indx21 < 0) indx21 += pHnStructures2;
  while (indx22 < 0) indx22 += pHnStructures2;
  while (indx11 > pHnStructures1 - 1) indx11 -= pHnStructures1;
  while (indx12 > pHnStructures1 - 1) indx12 -= pHnStructures1;
  while (indx21 > pHnStructures2 - 1) indx21 -= pHnStructures2;
  while (indx22 > pHnStructures2 - 1) indx22 -= pHnStructures2;

  int denom1 = indx12 - indx11;
  int denom2 = indx22 - indx21;

  double scale1 = (denom1 == 0) ? 0.0: (lambdas[j][1] * pHnStructures1 - 0.5 - static_cast<double>(indx11)) /
    static_cast<double>(denom1);
  double scale2 = (denom2 == 0) ? 0.0: (lambdas[j][2] * pHnStructures2 - 0.5 - static_cast<double>(indx21)) /
    static_cast<double>(denom2);


  for (int i = 0; i < nlocal; i++) {
    int molid_i = atom->molecule[i];

    if ((protonable[type[i]] == 1) && (molid_i == molids[j])) {
      double q_init = q_orig[i];
      double pH1q =
          pH1qs[type[i]][indx11] + scale1 * (pH1qs[type[i]][indx12] - pH1qs[type[i]][indx11]);
      double pH2q =
          pH2qs[type[i]][indx21] + scale2 * (pH2qs[type[i]][indx22] - pH2qs[type[i]][indx21]);
      q[i] = pH1q + scale0 * (pH2q - pH1q);    // scale == 1 should be for the protonated state
      q_changes_local[0]++;
      q_changes_local[1] += (q[i] - q_init);
    }
  }

  /* If the buffer is set the modify_q_buffer modifies the charge of the buffer 
       and the constraint in the fix_nh_constant_pH would constrain the total charge.
       So, nothing lefts to do here! */
  if (!(flags & BUFFER) || (flags & ZEROCHARGE)) {
    MPI_Allreduce(q_changes_local.get(), q_changes.get(), 2, MPI_DOUBLE, MPI_SUM, world);
    double HW_q_change = -q_changes[1] / static_cast<double>(num_HWs);

    for (int i = 0; i < nlocal; i++) {
      if (type[i] == typeHWs) {
        double q_init = q_orig[i];
        q[i] = q_init + HW_q_change;    //The total charge should be neutral
        q_changes_local[2]++;
        q_changes_local[3] += (q[i] - q_init);
      }
    }

    /* The purpose of this part this is just to debug the total charge.
           So, in the final version of the code this part should be 
           commented out!
        */
    /*if (update->ntimestep % nevery == 0) {
    	      MPI_Allreduce(q_changes_local.get(),q_changes.get(),4,MPI_DOUBLE,MPI_SUM,world);
    	     if (comm->me == 0) error->warning(FLERR,"protonable q change = {}, HW q change = {}, protonable charge change = {}, HW charge change = {}",q_changes[0],q_changes[2],q_changes[1],q_changes[3]);
        }
        compute_q_total();*/
  }

}

/* --------------------------------------------------------------
   modify the q of the lambdas
   -------------------------------------------------------------- */

void FixConstantPH::modify_qs(double **scales)
{
  int nlocal = atom->nlocal;
  int *type = atom->type;
  double *q = atom->q;

  // Not safe, I should use std::shared_ptr instead.....
  int *protonable = pH_structure_storage->protonable.get();    
  double **pH1qs = pH_structure_storage->pH1qs;
  double **pH2qs = pH_structure_storage->pH2qs;
  int pHnStructures1 = pH_structure_storage->pHnStructures1;
  int pHnStructures2 = pH_structure_storage->pHnStructures2;
  double** lambdas = pH_state->lambdas;
  auto& molids = pH_state->molids;
  auto& n_lambdas = pH_state->n_lambdas;
  

  std::unique_ptr<double []> q_changes_local = std::make_unique<double []>(5);
  std::unique_ptr<double []> q_changes       = std::make_unique<double []>(5);
  std::fill_n(q_changes_local.get(),5,0.0);


  std::fill_n(vector_atom,nmax,-1);


  // update the charges
  for (int j = 0; j < n_lambdas; j++) {
    double scale0 = scales[j][0];
    int indx11 = std::floor(lambdas[j][1] * pHnStructures1 - 0.5);
    int indx12 = std::ceil(lambdas[j][1]  * pHnStructures1 - 0.5);
    int indx21 = std::floor(lambdas[j][2] * pHnStructures2 - 0.5);
    int indx22 = std::ceil(lambdas[j][2]  * pHnStructures2 - 0.5);


    // Wrapping around 
    while (indx11 < 0) indx11 += pHnStructures1;
    while (indx12 < 0) indx12 += pHnStructures1;
    while (indx21 < 0) indx21 += pHnStructures2;
    while (indx22 < 0) indx22 += pHnStructures2;
    while (indx11 > pHnStructures1 - 1) indx11 -= pHnStructures1;
    while (indx12 > pHnStructures1 - 1) indx12 -= pHnStructures1;
    while (indx21 > pHnStructures2 - 1) indx21 -= pHnStructures2;
    while (indx22 > pHnStructures2 - 1) indx22 -= pHnStructures2;

    int denom1 = indx12 - indx11;
    int denom2 = indx22 - indx21;

    double scale1 = (denom1 == 0) ? 0.0: 
      (lambdas[j][1] * pHnStructures1 - 0.5 - static_cast<double>(indx11)) /static_cast<double>(denom1);
    double scale2 = (denom2 == 0) ? 0.0: 
      (lambdas[j][2] * pHnStructures2 - 0.5 - static_cast<double>(indx21)) /static_cast<double>(denom2);



    for (int i = 0; i < nlocal; i++) {
      int molid_i = atom->molecule[i];

      if ((protonable[type[i]] == 1) && (molid_i == molids[j])) {
        double q_init = q_orig[i];
        double pH1q =
            pH1qs[type[i]][indx11] + scale1 * (pH1qs[type[i]][indx12] - pH1qs[type[i]][indx11]);
        double pH2q =
            pH2qs[type[i]][indx21] + scale2 * (pH2qs[type[i]][indx22] - pH2qs[type[i]][indx21]);
        q[i] = pH1q + scale0 * (pH2q - pH1q);    // scale == 1 should be for the protonated state
        q_changes_local[0]++;
        q_changes_local[1] += (q[i] - q_init);
        //q_changes_local[1] += pH1q;
        q_changes_local[2] += pH2q;
        q_changes_local[3] += (q[i] - pH1q);
        q_changes_local[4] += q_init;

        vector_atom[i] = scale0;
      }
    }
  }

  //MPI_Allreduce(q_changes_local.get(), q_changes.get(), 5, MPI_DOUBLE, MPI_SUM, world);

  if (comm->me == 0 && false) {
    double sigma_scale = 0.0;
    for (int i = 0; i < n_lambdas; i++) sigma_scale += scales[i][0];
  }

  /* If the buffer is set the modify_q_buffer modifies the charge of the buffer 
       and the constraint in the fix_nh_constant_pH would constrain the total charge.
       So, nothing lefts to do here! */
  if (!(flags & BUFFER) || (flags & ZEROCHARGE)) {
    MPI_Allreduce(q_changes_local.get(), q_changes.get(), 4, MPI_DOUBLE, MPI_SUM, world);
    double HW_q_change = -q_changes[1] / static_cast<double>(num_HWs);

    for (int i = 0; i < nlocal; i++) {
      if (type[i] == typeHWs) {
        double q_init = q_orig[i];
        q[i] = q_init + HW_q_change;    //The total charge should be neutral
        q_changes_local[2]++;
        q_changes_local[3] += (q[i] - q_init);
      }
    }
  }

}

/* --------------------------------------------------------------
   modify the q of the buffer
   -------------------------------------------------------------- */

void FixConstantPH::modify_q_buff(const double _scale)
{
  int nlocal = atom->nlocal;
  int *type = atom->type;
  double *q = atom->q;

  // update the charges
  for (int i = 0; i < nlocal; i++) {
    if (type[i] == typeHWs) {
      q[i] = (_scale - qOWs) / 3.0;
    } else if (type[i] == typeOWs) {
      q[i] = qOWs;  
      // Just to assure if the charge of Oxygen atoms of the hydronium ions are correct!
    }
  }
}

/* ----------------------------------------------------------------------
   modify force and kspace in lammps according
   ---------------------------------------------------------------------- */

void FixConstantPH::update_lmp()
{
  int eflag = ENERGY_GLOBAL;
  int vflag = 0;
  timer->stamp();
  if (force->pair && force->pair->compute_flag) {
    force->pair->compute(eflag, vflag);
    timer->stamp(Timer::PAIR);
  }
  if (force->kspace && force->kspace->compute_flag) {
    force->kspace->compute(eflag, vflag);
    timer->stamp(Timer::KSPACE);
  }

  // accumulate force/energy/virial from /gpu pair styles
  if (fixgpu) fixgpu->post_force(vflag);
}

/* ---------------------------------------------------------------------
   Add forcefield correction term deltaGFF in equation 2 of
   https://pubs.acs.org/doi/full/10.1021/acs.jctc.5b01160
   --------------------------------------------------------------------- */

void FixConstantPH::calculate_GFFs()
{
  auto& n_lambdas = pH_state->n_lambdas;
  double** lambdas = pH_state->lambdas;
  for (int j = 0; j < n_lambdas; j++) {
    int i = 0;
    while (i < GFF_size && GFF[i][0] < lambdas[j][0]) i++;

    if (i == 0) {
      error->warning(
          FLERR,
          "Warning lambda of {} in Fix constant_pH out of the range, it usually should not happen",
          lambdas[j][0]);
      GFF_lambdas[j] = GFF[0][1] +
          ((GFF[1][1] - GFF[0][1]) / (GFF[1][0] - GFF[0][0])) * (lambdas[j][0] - GFF[0][0]);
    }
    if (i > 0 && i < GFF_size - 1)
      GFF_lambdas[j] = GFF[i - 1][1] +
          ((GFF[i][1] - GFF[i - 1][1]) / (GFF[i][0] - GFF[i - 1][0])) *
              (lambdas[j][0] - GFF[i - 1][0]);
    if (i == GFF_size - 1) {
      error->warning(
          FLERR,
          "Warning lambda of {} in Fix constant_pH out of the range, it usually should not happen",
          lambdas[j][0]);
      GFF_lambdas[j] = GFF[i][1] +
          ((GFF[i][1] - GFF[i - 1][1]) / (GFF[i][0] - GFF[i - 1][0])) * (lambdas[j][0] - GFF[i][0]);
    }
  }
}

/* ---------------------------------------------------------------------
   Read the data file containing the term deltaGFF in equation 2 of 
   https://pubs.acs.org/doi/full/10.1021/acs.jctc.5b01160
   --------------------------------------------------------------------- */

void FixConstantPH::init_GFF()
{
  /*
   * file format:
   * number of GFF element
   * lambda_1,GFF_1
   * lambda_2,GFF_2
   * lambda_3,GFF_3
   * ....
   * lambda_n,GFF_n
   */

  std::string line;
  std::stringstream iss;

  std::getline(fp, line);
  iss.clear();
  iss.str(line);
  iss >> GFF_size;

  memory->create(GFF, GFF_size, 2, "constant_pH:GFF");
  int i = 0;

  while (std::getline(fp, line) && i < GFF_size) {    
    i++;
    double _lambda, _GFF;
    std::stringstream iss2(line);
    std::string token;

    if (!std::getline(iss2, token, ','))
      error->one(FLERR, "The GFF correction file in the fix constant_pH is in a wrong format!");
    _lambda = std::stod(token);

    if (!std::getline(iss2, token, ','))
      error->one(FLERR, "The GFF correction file in the fix constant_pH is in a wrong format!");
    _GFF = std::stod(token);

    GFF[i][0] = _lambda;
    GFF[i][1] = _GFF;
  }

  if (i != GFF_size)
    error->warning(FLERR, "Not enough data in GFF correction file in the fix constant_pH!");
}


/* ----------------------------------------------------------------------
   Environment coupling 
   --------------------------------------------------------------------- */

void FixConstantPH::calculate_Hs()
{
  if (neighbor->ago && update->ntimestep)
    return ;
  int nlocal = atom->nlocal;
  int* molecule = atom->molecule;
  int* type = atom->type;
  double* q = atom->q;
  bigint natoms = atom->natoms;
   
  if (!q) error->all(FLERR, "Atom style has no charges");

  double **pH1qs = pH_structure_storage->pH1qs; 
  double **pH2qs = pH_structure_storage->pH2qs; 
  int *protonable = pH_structure_storage->protonable.get();
  double** lambdas = pH_state->lambdas;
  auto& molids = pH_state->molids;
  auto& n_lambdas = pH_state->n_lambdas;
   
   
  // you could have also used a map which seems more natural.
  auto distArray = std::make_unique<int[]>(nlocal);
  for (int i = 0; i < nlocal; i++)
  {
    int mol = molecule[i];
    auto itr = std::find(molids.get(), molids.get() + n_lambdas, mol);
    int dist = static_cast<int>(std::distance(molids.get(), itr));
    distArray[i] = dist;
  }
   

  // backing up qs
  backup_restore_qfev<1>();
  if (flags & INTERPOLATION) {
    /*The linear charge interpolation method in Aho et al JCTC 2022*/
    /*
     * According to the equation 17 in the https://pubs.acs.org/doi/full/10.1021/acs.jctc.2c00516,
     * we loop through lambdas.  For each lambda we put the charge as delta q and for
     * the rest we put the charge as lambda*qProt + (1-lambda)*qDeprot
     * My lambda = 1 - their lambda;
    */
    for (int j = 0; j < n_lambdas; j++)
    {
      double H_lambda = 0.0;
      for (int i = 0; i < nlocal; i++)
      {
        if (!protonable[type[i]]) continue;
        int dist = distArray[i];
        if (dist == j)
          q[i] = pH2qs[type[i]][0] - pH1qs[type[i]][0];
        else if (dist != n_lambdas)
          q[i] = lambdas[dist][0] * pH2qs[type[i]][0] + (1 - lambdas[dist][0]) * pH1qs[type[i]][0];
      }
      // Neutralizing the system 
      //neutralize();
      // forward comm so that ghost atoms are consistent
      comm->forward_comm();
      // calculating the energies
      update_lmp();
      // getting the electrostatic energy + kspace energy
      H_lambda = compute_epair();
      // restore qs
      backup_restore_qfev<-1>();
      // saving
      HAs[j] = H_lambda;
      HBs[j] = 0.0;
    }   
  }
  else if (!(flags & INTERPOLATION)) {
    for (int j = 0; j < n_lambdas; j++) {
      // protonated
      double lambda_j = 1.0;
      // modifying the atom charges
      modify_qs(lambda_j,j);
      // Neutralizing the system 
      neutralize();
      // forward comm so that ghost atoms are consistent
      comm->forward_comm();
      // calculating the energies
      update_lmp();
      // getting the electrostatic energy + kspace energy
      HAs[j] = compute_epair();
      // deprotonated
      lambda_j = 0.0;
      // modifying the atom charges
      modify_qs(lambda_j,j);
      // Neutralizing the system 
      neutralize();
      // forward comm so that ghost atoms are consistent
      comm->forward_comm();
      // calculating the energies
      update_lmp();
      // getting the electrostatic energy + kspace energy
      HBs[j] = compute_epair();
      // restore qs
      backup_restore_qfev<-1>();
    }
  }

  // backing up qs
  backup_restore_qfev<1>();
  // For the buffer
  double H_lamda_buff = 0.0;
  // protonated
  double lambda_buff_temp = 1.0;
  // modifying the atom charges
  modify_q_buff(lambda_buff_temp);
  // forward comm so that ghost atoms are consistent
  comm->forward_comm();
  // calculating the energies
  update_lmp();
  // getting the electrostatic energy + kspace energy
  HA_buff = compute_epair();
  // deprotonated
  lambda_buff_temp = 0.0;
  // modifying the atom charges
  modify_q_buff(lambda_buff_temp);
  // forward comm so that ghost atoms are consistent
  comm->forward_comm();
  // calculating the energies
  update_lmp();
  // getting the electrostatic energy + kspace energy
  HB_buff = compute_epair();
  // restore qs
  backup_restore_qfev<-1>();


  HCalcNSteps++;
}

/* --------------------------------------------------------------------- 
    Neutralizing the total charges in the calculate_Hs()
   --------------------------------------------------------------------- */

double FixConstantPH::neutralize(bool buffer)
{
  if (buffer) {
    double q_total = compute_q_total(true);
    double N_buff_double = static_cast<double>(pH_state->N_buff);
    double lambda_buff_temp = -q_total / N_buff_double;
    modify_q_buff(lambda_buff_temp);
    return lambda_buff_temp;
  }
  return 0.0;
}

/* --------------------------------------------------------------------- 
    Write the header for the lambda output files which is called 
    whenever the n_lambdas change
   --------------------------------------------------------------------- */

void FixConstantPH::write_lambdas_header()
{
  auto& molids  = pH_state->molids;
  int n_lambdas = pH_state->n_lambdas;

  if (!(flags & ADAPTIVE) && !molids)
    error->all(FLERR, "fix constant_pH requires either 'molids' or 'Fix_adaptive_protonation'.");
  if (comm->me != 0) return;    // Only rank 0 writes

  const struct {
    int flag;
    std::ofstream *fp;
  } files[] = {{LAMBDA_FP, &lambda_fp},     {V_LAMBDA_FP, &v_lambda_fp},
               {A_LAMBDA_FP, &a_lambda_fp}, {H_LAMBDA_FP, &H_lambda_fp},
               {LAMBDA_S_FP, &lambda_1_fp}, {LAMBDA_S_FP, &lambda_2_fp},
               {HA_LAMBDA_FP,&HA_lambda_fp},{HB_LAMBDA_FP,&HB_lambda_fp}};

  for (auto &file : files) {
    if (fp_flags & file.flag && file.fp) {
      *(file.fp) << "n_lambdas=" << n_lambdas << std::endl;
      for (int i = 0; i < n_lambdas - 1; i++)
        *(file.fp) << "lambda-" << molids[i] << ",";
      if (n_lambdas > 0)
        *(file.fp) << "lambda-" << molids[n_lambdas - 1];
      if (file.flag == LAMBDA_S_FP) {
        *(file.fp) << std::endl;
        continue;
      } else if (flags & BUFFER) {
        if (n_lambdas > 0) *(file.fp) << ",";
        *(file.fp) << "lambda-buffer";
      }
      *(file.fp) << std::endl;
    }
  }
}

/* ---------------------------------------------------------------------
   writes the output in each step. Since with fix adaptive protonation 
   the number of lambdas change during the simulation the vector method of 
   fix will not work.
   --------------------------------------------------------------------- */

void FixConstantPH::write_lambdas()
{
  double** lambdas = pH_state->lambdas;
  double** v_lambdas = pH_state->v_lambdas;
  double** a_lambdas = pH_state->a_lambdas;
  auto& molids = pH_state->molids; 
  int n_lambdas = pH_state->n_lambdas;
  double lambda_buff = pH_state->lambda_buff;
  double v_lambda_buff = pH_state->v_lambda_buff;
  double a_lambda_buff = pH_state->a_lambda_buff;

  if (!(flags & ADAPTIVE) && !molids)
    error->all(FLERR, "fix constant_pH requires either 'molids' or 'Fix_adaptive_protonation'.");

  if (comm->me != 0) return;    // Only rank 0 writes

  const struct {
    int flag;
    std::ofstream *fp;
    std::unique_ptr<double []>& content;
    double buff_value;
  } filesp [] = {{H_LAMBDA_FP, &H_lambda_fp, H_lambdas,H_lambda_buff},
                 {HA_LAMBDA_FP,&HA_lambda_fp,HAs,HA_buff},
                 {HB_LAMBDA_FP,&HB_lambda_fp,HBs,HB_buff}};

  for (auto& file: filesp) {
    if (fp_flags & file.flag && file.fp) {
      for (int i = 0; i < n_lambdas - 1; i++)
        *(file.fp) << file.content[i] << ",";
      if (n_lambdas > 0)
        *(file.fp) << file.content[n_lambdas - 1];
      if (flags & BUFFER) {
        if(n_lambdas > 0) *(file.fp) << ",";
        *(file.fp) << file.buff_value;
      }
      *(file.fp) << std::endl;
    }
  }

  const struct {
    int flag;
    std::ofstream *fp;
    double **content;
    int j;
    double buff_value;
  } files[] = {{V_LAMBDA_FP, &v_lambda_fp, v_lambdas, 0, v_lambda_buff},
               {A_LAMBDA_FP, &a_lambda_fp, a_lambdas, 0, a_lambda_buff},
               {LAMBDA_FP, &lambda_fp, lambdas, 0, lambda_buff},
               {LAMBDA_S_FP, &lambda_1_fp, lambdas, 1, 0.0},
               {LAMBDA_S_FP, &lambda_2_fp, lambdas, 2, 0.0}};

  for (auto &file : files) {
    if (fp_flags & file.flag && file.fp) {
      for (int i = 0; i < n_lambdas - 1; i++)
        *(file.fp) << file.content[i][file.j] << ",";
      if (n_lambdas > 0)
        *(file.fp) << file.content[n_lambdas - 1][file.j];
      if (file.flag == LAMBDA_S_FP) {
        *(file.fp) << std::endl;
        continue;
      } else if (flags & BUFFER) {
        if (n_lambdas > 0) *(file.fp) << ",";
        *(file.fp) << file.buff_value;
      }
      *(file.fp) << std::endl;
    }
  }
}

/* --------------------------------------------------------------------- */

void FixConstantPH::initialize_v_lambda(const double _T_lambda, const int& to)
{
  auto& v_lambdas     = pH_state->v_lambdas;
  auto& v_lambda_buff = pH_state->v_lambda_buff;
  auto& m_lambdas     = pH_state->m_lambdas;
  auto& m_lambda_buff = pH_state->m_lambda_buff;
  auto& n_lambdas     = pH_state->n_lambdas;
  auto& N_buff        = pH_state->N_buff;

  std::unique_ptr<RanPark> random = std::make_unique<RanPark>(lmp, random_number_seed);

  int length = n_lambdas - to;

  double T_current[2] = {0.0,0.0};
  while (std::abs(T_current[0]) < tol || std::abs(T_current[1]) < tol ) {
    for (int i = to; i < n_lambdas; i++)
      for (int j = 0; j < 3; j++)
        v_lambdas[i][j] = 1e-6 * random->gaussian() / std::sqrt(m_lambdas[i][j]);

    if (flags & BUFFER) v_lambda_buff = 1e-6 * random->gaussian() / std::sqrt(m_lambda_buff);

    this->calculate_T_lambda(to,T_current);
  }


  double scaling_factor1 = std::sqrt(_T_lambda / T_current[0]);
  double scaling_factor2 = std::sqrt(_T_lambda / T_current[1]);

  for (int i = to; i < n_lambdas; i++) {
    v_lambdas[i][0] *= scaling_factor1;
    for (int j = 1; j < 3; j++)
      v_lambdas[i][j] *= scaling_factor2;
  }

  if (flags & BUFFER) v_lambda_buff *= scaling_factor1;


  if (n_lambdas > 0)
    MPI_Bcast(v_lambdas[to], length * 3, MPI_DOUBLE, 0, world);
  
  if (flags & BUFFER) MPI_Bcast(&v_lambda_buff,1,MPI_DOUBLE,0,world);
  


  // Updating the T_lambdas
  this->calculate_T_lambda();


}

/* --------------------------------------------------------------------- */

void FixConstantPH::calculate_T_lambda(const int& to, double* T_)
{
  auto& n_lambdas = pH_state->n_lambdas;
  auto& v_lambdas = pH_state->v_lambdas;
  auto& v_lambda_buff = pH_state->v_lambda_buff;
  auto& m_lambdas = pH_state->m_lambdas;
  auto& m_lambda_buff = pH_state->m_lambda_buff;
  auto& N_buff = pH_state->N_buff;

  double KE_lambdas[3] = {0.0, 0.0, 0.0};    // lambdas[0][;], lambdas[1:][;], lambdas[;][;]
  double Nfs[3];
  double kB = 0.008314; //force->boltz;
  double mvv2e = 1e6; //force->mvv2e;

  Nfs[0] = static_cast<double>(n_lambdas - to);
  Nfs[1] = static_cast<double>(2 * (n_lambdas - to));
  Nfs[2] = Nfs[0] + Nfs[1];

  if (comm->me == 0) {
    if (flags & BUFFER) {
      Nfs[0] += 1.0;
      Nfs[2] += 1.0;
    }
    if (flags & CONSTRAIN) {
      Nfs[0] -= 1.0;
      Nfs[2] -= 1.0;
    }

    for (int j = to; j < n_lambdas; j++) {
      KE_lambdas[0] += 0.5 * m_lambdas[j][0] * v_lambdas[j][0] * v_lambdas[j][0] * mvv2e;
      for (int k = 1; k < 3; k++)
        KE_lambdas[1] += 0.5 * m_lambdas[j][k] * v_lambdas[j][k] * v_lambdas[j][k] * mvv2e;
    }
    KE_lambdas[2] = KE_lambdas[0] + KE_lambdas[1];

    if (flags & BUFFER) {
      KE_lambdas[0] += 0.5 * N_buff * m_lambda_buff * v_lambda_buff * v_lambda_buff * mvv2e;
      KE_lambdas[2] += 0.5 * N_buff * m_lambda_buff * v_lambda_buff * v_lambda_buff * mvv2e;
    }


    if (kB == 0) error->one(FLERR, "The k value is zero");
    if (T_ == nullptr) {
      if (Nfs[0] > 0.0)
        T_lambdas[0] = 2.0 * KE_lambdas[0] / (Nfs[0] * kB);
      else
        T_lambdas[0] = 0.0;
      if (Nfs[1] > 0.0)
        T_lambdas[1] = 2.0 * KE_lambdas[1] / (Nfs[1] * kB);
      else
        T_lambdas[1] = 0.0;
      if (Nfs[2] > 0.0)
        T_lambdas[2] = 2.0 * KE_lambdas[2] / (Nfs[2] * kB);
      else
        T_lambdas[2] = 0.0;
    } else {
      if (Nfs[0] > 0.0) {
        T_[0] = 2.0*KE_lambdas[0] / (Nfs[0] * kB);
      }
      else
        T_[0] = 0.0;
      if (Nfs[1] > 0.0) {
        T_[1] = 2.0*KE_lambdas[1] / (Nfs[1] * kB);
      }
      else
        T_[1] = 0.0;
    }
  }

  if (T_ == nullptr)
    MPI_Bcast(T_lambdas, 3, MPI_DOUBLE, 0, world);
  else {
    MPI_Bcast(T_,2,MPI_DOUBLE,0,world);
  }
}

/* --------------------------------------------------------------------- */

double FixConstantPH::compute_q_total(const bool silent)
{
  double *q = atom->q;
  int nlocal = atom->nlocal;
  bigint ntimestep = update->ntimestep;
  double q_local = 0.0;

  for (int i = 0; i < nlocal; i++)
    q_local += q[i];

  MPI_Allreduce(&q_local, &q_total, 1, MPI_DOUBLE, MPI_SUM, world);

  if (!silent) {
    if (std::abs(q_total) > tol && comm->me == 0)
      error->warning(FLERR, "q_total in fix constant-pH is non-zero: {} at step {}", q_total, ntimestep);
  }

  return q_total;
}

/* --------------------------------------------------------------------- */

double FixConstantPH::compute_epair()
{
  // As I am calling update_lmp() before the compute_epair
  // the energies are tallied.
  //if (update->eflag_global != update->ntimestep)
  //   error->all(FLERR,"Energy was not tallied on the needed timestep");


  double one = 0.0;
  double energy;
  if (force->pair) one += force->pair->eng_coul; // + force->pair->eng_vdwl;
  


  /* As the bond, angle, dihedral and improper energies 
      do not change with the lambda, we do not need to 
      include them in the energy. We are interested in 
      their difference afterall */

  // the eng_coul and eng_vdwl are accumulated per rank.
  // Since the vdwl energy does not change too much with the lambda value
  // we do not include it here.
  MPI_Allreduce(&one,&energy,1,MPI_DOUBLE,MPI_SUM,world);


  // Adding the kspace component
  // the kspace energy is the value accumulated for all the ranks.
  // Look at src/compute_pe.cpp
  //if (force->kspace)
  //  energy += force->kspace->energy;

  
  
  /*
   * This value must not be divided by the number of atoms for two reasons:
   * (1) in the lammps doc (https://docs.lammps.org/compute_pe.html)
   *     this value has been mentioned as extensive.
   *     The same procedure in the compute_pe.cpp has been used.
   * (2) in the updatelamdas function of the gromacs implementation 
   *     an extensive value has been used. 
   */ 

  return energy;
}

/* --------------------------------------------------------------------------------------- */

void FixConstantPH::grow_arrays(int nmax_new)
{
  double *new_vector_atom = new double[nmax_new];
  int keep = std::min(nmax,nmax_new);
  if (keep > 0) std::copy(vector_atom,vector_atom+keep,new_vector_atom);
  if (keep < nmax_new) std::fill(new_vector_atom+keep,new_vector_atom+nmax_new,0.0);
  delete [] vector_atom;
  vector_atom = new_vector_atom;
  nmax = nmax_new;
}

/* --------------------------------------------------------------------------------------- */

void FixConstantPH::copy_arrays(int i, int j , int /*deflag*/)
{
  vector_atom[j] = vector_atom[i];
}

/* --------------------------------------------------------------------------------------- */

int FixConstantPH::pack_exchange(int i, double *buf) {
  buf[0] = vector_atom[i];
  return 1;
}

/* --------------------------------------------------------------------------------------- */

int FixConstantPH::unpack_exchange(int nlocal, double *buf) {
  vector_atom[nlocal] = buf[0];
  return 1;
}

/* ----------------------------------------------------------------------
   implementing a method to access the HA, HB, lambda, v_lambda and 
   a_lambda values 
   ---------------------------------------------------------------------- */

double FixConstantPH::compute_array(int i, int j)
{
  double kj2kcal = 0.239006;
  double kT = 0.008314*T; //force->boltz * T;
  auto& N_buff = pH_state->N_buff;
  auto& n_lambdas = pH_state->n_lambdas;
  auto& lambdas = pH_state->lambdas;
  auto& v_lambdas = pH_state->v_lambdas;
  auto& a_lambdas = pH_state->a_lambdas;
  auto& lambda_buff = pH_state->lambda_buff;
  auto& v_lambda_buff = pH_state->v_lambda_buff;
  auto& a_lambda_buff = pH_state->a_lambda_buff;

  switch (i) {
    case 0:
      // 1
      if (j < n_lambdas)
        return HAs[j];
      else if ((j == n_lambdas) && (flags & BUFFER))
        return N_buff * HA_buff;
      else
        return -1.0;
    case 1:
      // 2
      if (j < n_lambdas)
        return HBs[j];
      else if ((j == n_lambdas) && (flags & BUFFER))
        return N_buff * HB_buff;
      else
        return -1.0;
    case 2:
      // 3
      if (j < n_lambdas)
        return dfs[j] * kT * log(10) * (pK - pH);
      else if ((j == n_lambdas) && (flags & BUFFER))
        return 0.0;
      else
        return -1.0;
    case 3:
      // 4
      if (j < n_lambdas)
        return kj2kcal * dUs[j];
      else if ((j == n_lambdas) && (flags & BUFFER))
        return kj2kcal * dU_buff;
      else
        return -1.0;
    case 4:
      // 5
      if (j < n_lambdas)
        return GFF_lambdas[j];
      else if ((j == n_lambdas) && (flags & BUFFER))
        return 0.0;
      else
        return -1.0;
    case 5:
      // 6
      if (j < 3 * n_lambdas)
        return lambdas[j % n_lambdas][j / n_lambdas];
      else if ((j == 3 * n_lambdas) && (flags & BUFFER))
        return lambda_buff;
      else
        return -1.0;
    case 6:
      // 7
      if (j < 3 * n_lambdas)
        return v_lambdas[j % n_lambdas][j / n_lambdas];
      else if ((j == 3 * n_lambdas) && (flags & BUFFER))
        return v_lambda_buff;
      else
        return -1.0;
    case 7:
        // 8
        if (j < 3 * n_lambdas)
          return a_lambdas[j % n_lambdas][j / n_lambdas];
        else if ((j == 3 * n_lambdas) && (flags & BUFFER))
          return a_lambda_buff;
        else
          return -1.0;
    case 8:
      // 9
      calculate_T_lambda();
      if (j >= 0 && j <= 2) return T_lambdas[j];
      return -1.0; 
    case 9:
      // 10
      if (j < n_lambdas)
        return H_lambdas[j];
      else if ((j == n_lambdas) && (flags & BUFFER))
        return H_lambda_buff;
      else
        return -1.0;
    case 10:
      // 11
      compute_q_total();
      return q_total;
    case 11:
      // 12
      if (j < n_lambdas)
        return HAs[j];
      else if ((j == n_lambdas) && (flags & BUFFER))
        return HA_buff;
      else
        return -1.0;
    case 12:
      // 13
      if (j < n_lambdas)
        return HBs[j];
      else if ((j == n_lambdas) && (flags & BUFFER))
        return HB_buff;
      else
        return -1.0;
  }
  return 0.0;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array --> Needs to be updated at the end
   ---------------------------------------------------------------------- */

double FixConstantPH::memory_usage()
{
  int nmax = atom->nmax;
  int nlocal = atom->nlocal;
  int ntypes = atom->ntypes;
  double GFF_bytes = 2.0 * GFF_size * sizeof(double);
  double epsilon_init_bytes = (double) (ntypes + 1) * (double) (ntypes + 1) * sizeof(double);
  double q_orig_bytes = (double) nlocal * sizeof(double);
  double f_orig_bytes = (double) nlocal * 3.0 * sizeof(double);
  double peatom_orig_bytes = (double) nlocal * sizeof(double);
  double pvatom_orig_bytes = (double) nlocal * 6.0 * sizeof(double);
  double keatom_orig_bytes = (double) nlocal * sizeof(double);
  double kvatom_orig_bytes = (double) nlocal * 6.0 * sizeof(double);
  double bytes = GFF_bytes + epsilon_init_bytes + q_orig_bytes + f_orig_bytes + peatom_orig_bytes +
      pvatom_orig_bytes + keatom_orig_bytes + kvatom_orig_bytes;
  return bytes;
}
