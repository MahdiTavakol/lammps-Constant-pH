
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

#ifdef FIX_CLASS
// clang-format off
FixStyle(constant_pH,FixConstantPH);
// clang-format on
#else

#ifndef LMP_FIX_CONSTANTPH_H
#define LMP_FIX_CONSTANTPH_H

#include <fstream>

#include "fix.h"
#include "fix_adaptive_protonation.h"
#include "pair.h"

namespace LAMMPS_NS {
  
class FixConstantPH : public Fix {
  friend class FixNHConstantPH;
  friend class ComputeGFFConstantPH;
  friend class ComputeTempConstantPH;

 public:
  FixConstantPH(class LAMMPS *, int, char **);
  ~FixConstantPH() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void post_force(int) override;
  double compute_array(int, int) override;
  double memory_usage() override;

  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int i, double *buf) override;
  int unpack_exchange(int nlocal, double *buf) override;

 protected:
  int flags;

  // Sturcture files
  std::string fileName1, fileName2;

  // The structure information
  std::unique_ptr<constant_pH_structures> pH_structure_storage;

  // Commands that run whenever the lambdas array is modified
  int ncommands;
  std::unique_ptr<string []> commands;
  std::ifstream commandsFile;

  // Input variables for constant values
  double pK, pH, T;

  // Forcefield parameters
  double a, b, s, m, w, r, d, k, h;
  // Buffer potential parameters
  double a_buff, b_buff, s_buff, m_buff, w_buff, r_buff, d_buff, k_buff, h_buff;

  // Environment correlation parameters
  std::unique_ptr<double[]> HAs;
  std::unique_ptr<double[]> HBs;
  double HA_buff, HB_buff;


  // The step function 
  std::unique_ptr<double[]> Us;
  std::unique_ptr<double[]> dUs;
  double U_buff, dU_buff;

  // The smoothing function
  std::unique_ptr<double[]> fs;
  std::unique_ptr<double[]> dfs;

  // parameter for shifting the minima of the potential near lambda = 0 and lambda = 1
  double mu;

  // Random number seed for the creation of initial v_lambdas
  double random_number_seed;


  // input params for the pH_state
  int n_lambdas_input = 0;
  std::unique_ptr<int []> molids_input;
  std::array<double,2> lambda_masses;

  // lambda energies
  std::unique_ptr<double []> H_lambdas;
  double H_lambda_buff;
  double H_lambda_prev;
  

  // lambda temperature
  double T_lambdas[3];

  // pH protonation state
  std::unique_ptr<constant_pH_state> pH_state;
  // pH protonation state from the previous fix_adaptive_protonation step to keep similar lambdas.
  std::unique_ptr<constant_pH_state> pH_state_prev;
  

  // Temporary array to change lambdas in order to get HAs and HBs
  std::unique_ptr<double[]> lambdas_j;

  // Parameters for the forcefield modification term
  bool GFF_flag;
  std::ifstream fp;
  double **GFF;
  int GFF_size;
  std::unique_ptr<double[]> GFF_lambdas;

  // Parameters for printing the Udwp
  bool print_Udwp_flag;
  std::ofstream Udwp_fp;
  void print_Udwp();

  // lambda_buff at step0
  double lambda_buff_0 = 1.0;
  // number of buffer points
  int N_buff;

  // Hydrogen and Oxygens types of the hydronium ions
  int typeHWs, typeOWs;
  double qHWs, qOWs;
  int num_HWs, num_OWs; 
 
  // Functions needed to communicate with fix adaptive protonation command
  std::string fix_adaptive_protonation_id;
  int nevery_fix_adaptive;
  FixAdaptiveProtonation *fix_adaptive_protonation;

  // Output files when we have adaptive protonation
  int fp_flags;
  int write_lambda_nevery;
  std::ofstream lambda_fp, lambda_1_fp, lambda_2_fp, v_lambda_fp, a_lambda_fp, H_lambda_fp;
  std::ofstream HA_lambda_fp, HB_lambda_fp;

  // The name of the intermediate file written by the fix_adaptive_protonation
  std::string intermediate_file_name;

  // output methods for a variable sized lambdas, v_lambdas, ...
  void write_lambdas_header();
  void write_lambdas();

  // Function required to be called by the compute_GFF
  void calculate_H_once();
  // Number of steps when HA was calculated
  int HCalcNSteps; 

  // the total charge parameter and functions.
  double q_total;
  double compute_q_total(const bool silent = false);
  void check_q_total();

  class Fix *fixgpu;

  // maximum local atoms
  int nmax;
  // maximum local + ghost atoms
  int natoms;

  // These pointers are allocated and deallocated through allocate_storage() and deallocate_storage() functions
  // _org is for value of parameters before the update_lmp() with modified parameters act on them
  double *q_orig;
  double **f_orig;
  double eng_vdwl_orig, eng_coul_orig;
  double pvirial_orig[6];
  double *peatom_orig, **pvatom_orig;
  double energy_orig;
  double kvirial_orig[6];
  double *keatom_orig, **kvatom_orig;

  // Functions for accessing or reseting the lambda dynamics parameters
  void return_params(std::unique_ptr<constant_pH_state>& pH_state) const;
  void reset_params(const std::unique_ptr<constant_pH_state>& pH_state_, const int mode = 1);
  void reset_params(std::unique_ptr<constant_pH_state>&& pH_state_, const int mode = 1);
  void return_nparams(int &_n_params) const;
  void return_H_lambdas(double *_H_lambdas) const;
  void return_T_lambda(double &_T_lambda, int component = 2);


  // Function to set the charges based on the lambdas and lambda_buff values
  void reset_qs();

  // The function to calculate Hs
  void calculate_Hs();
  // Resetting the total charge (used in the calculate_Hs())
  double neutralize(bool buffer = true);
  // The function that checks that the ratio of OWs to HWs is 1.0 to 3.0
  void check_num_OWs_HWs();
  // Reading the structures at different pH values
  void read_pH_structure_files();
  // Reading the commands that should run after each time fix_adaptive_protonation is called
  void read_commands_file();
  // 
  void restore_epsilon();
  void delete_lambdas();
  void set_lambdas();
  void delete_lambdas_prev();
  void set_lambdas_prev();
  void initialize_lambda(const int& to=0);
  void calculate_dq();
  void calculate_dfs();
  void calculate_dUs();
  void calculate_dU(const double &_lambda, double &_U, double &_dU);
  void calculate_T_lambda(const int& to=0);
  void initialize_v_lambda(const double _T_lambda, const int& to=0);
  void integrate_lambda();
  void allocate_storage();
  void deallocate_storage();
  template <int direction> void forward_reverse_copy(double &a, double &b);
  template <int direction> void forward_reverse_copy(double *a, double *b, int i);
  template <int direction> void forward_reverse_copy(double **a, double **b, int i, int j);
  template <int direction> void backup_restore_qfev();
  void init_GFF();
  void calculate_GFFs();
  void modify_qs(double scale, int j);
  void modify_qs(double **scales);
  void modify_q_buff(const double scale);
  void update_lmp();
  double compute_epair();
  void update_a_lambda();
};

}    // namespace LAMMPS_NS

#endif
#endif
