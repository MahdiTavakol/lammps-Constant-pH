
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

#ifdef FIX_CLASS
// clang-format off
FixStyle(constant_pH,FixConstantPH);
// clang-format on
#else

#ifndef LMP_FIX_CONSTANTPH_H
#define LMP_FIX_CONSTANTPH_H

#include "fix.h"
#include "fix_adaptive_protonation.h"
#include "pair.h"


namespace LAMMPS_NS {

  class FixConstantPH: public Fix {
     friend class FixNHConstantPH;
     friend class ComputeGFFConstantPH;
     friend class ComputeTempConstantPH;
     public:
	FixConstantPH(class LAMMPS*, int, char**);
	~FixConstantPH() override;
	int setmask() override;
	void init() override;
	void setup(int) override;
	void initial_integrate(int) override;
	void post_force(int) override;
        double compute_array(int, int) override;
	double memory_usage() override;



     protected:
        int flags;
	// Sturcture files
        FILE *pHStructureFile1, *pHStructureFile2;

	// Atom types and charges that change due to protonation
        int pHnStructures1, pHnStructures2;
        int pHnTypes1, pHnTypes2;
        double **pH1qs, **pH2qs;
        int * typePerProtMol;
        int * protonable;

        // Commands that run whenever the lambdas array is modified
        int ncommands;
        char ** commands;
        FILE *commandsFile;

	// Input variables for constant values
	double pK, pH, T;

	double a, b, s, m, w, r, d, k, h;
	double* HAs, * HBs;
	double* Us, * dUs;
	
	// parameter for shifting the minima of the potential near lambda = 0 and lambda = 1
	double mu;

        // Random number seed for the creation of initial v_lambdas
        double random_number_seed;

	// Buffer potential parameters
	double a_buff, b_buff, s_buff, m_buff, w_buff, r_buff, d_buff, k_buff, h_buff;
	

        // Lambda arrays
        double ** lambdas, ** v_lambdas, ** a_lambdas, ** m_lambdas, * H_lambdas;
        double T_lambdas[3];
        int * molids;
        int n_lambdas;
        
        
        // Temp array to change lambdas in order to get HAs and HBs
        double * lambdas_j;

	// The smoothing function 
	double * fs, * dfs;

	// Parameters for the forcefield modifiction term
        bool GFF_flag;
	FILE *fp;
	double **GFF;
	int GFF_size;
	double* GFF_lambdas;

        // Parameters for printing the Udwp
        bool print_Udwp_flag;
        FILE *Udwp_fp;
        void print_Udwp();

        // Parameters for the buffer
	double lambda_buff, v_lambda_buff, a_lambda_buff, m_lambda_buff, H_lambda_buff;
        int N_buff;

	// Hydrogen and Oxygens types of the hydronium ions
	int typeHWs, typeOWs;
	double qHWs, qOWs;
        int num_HWs, num_OWs;

        // Buffer potential
        double U_buff, dU_buff;
        double HA_buff, HB_buff;
        

	// Functions needed to communicate with fix adaptive protonation command
	char * fix_adaptive_protonation_id;
	int nevery_fix_adaptive;
	FixAdaptiveProtonation* fix_adaptive_protonation;

        // Output files when we have adaptive protonation
        int fp_flags;
        FILE* lambda_fp, *lambda_1_fp, *lambda_2_fp, *v_lambda_fp, *a_lambda_fp, *H_lambda_fp;


        // The name of the intermediate file written by the fix_adaptive_protonation 
        char* intermediate_file_name;
        
        // output methods for a variable sized lambdas, v_lambdas, ... 
        void write_lambdas_header();
        void write_lambdas();

        // Function required to be called by the compute_GFF
        void calculate_H_once();

        // The q_total used to calculate the HW charges
        double q_total;
        double compute_q_total();
        void check_q_total();


	class Fix *fixgpu;


        int nmax;

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
        void return_nparams(int& _n_params) const;
        void return_params(double** const _x_lambdas, double** const _v_lambdas, 
                           double** const _a_lambdas, double** const _m_lambdas) const;
        void reset_params(double** const _x_lambdas, double** const _v_lambdas, 
                          double** const _a_lambdas, double** const _m_lambdas, const int mode);
        void reset_params(double** const _x_lambdas, double** const _v_lambdas, 
                          double** const _a_lambdas, double** const _m_lambdas) {
            reset_params(_x_lambdas,_v_lambdas,_a_lambdas,_m_lambdas,1);              
        }
        void return_H_lambdas(double* _H_lambdas) const;
        void return_T_lambda(double& _T_lambda, int component = 2);


	// Functions to return the buffer parameters
	void return_buff_params(double& _x_lambda_buff, double& _v_lambda_buff, 
                                  double& _a_lambda_buff, double& _m_lambda_buff, int& _N_buff) const;
	void reset_buff_params(const double _x_lambda_buff, const double _v_lambda_buff, 
                                 const double _a_lambda_buff, const double _m_lambda_buff, const int mode);
        void reset_buff_params(const double _x_lambda_buff, const double _v_lambda_buff,
                                 const double _a_lambda_buff, const double _m_lambda_buff) {
            reset_buff_params(_x_lambda_buff,_v_lambda_buff,_a_lambda_buff,_m_lambda_buff,1);
        }
           
        // Function to set the charges based on the lambdas and lambda_buff values
        void reset_qs();


	void compute_Hs();
        void check_num_OWs_HWs();
        void read_pH_structure_files();
        void read_commands_file();
        void restore_epsilon();
	void delete_lambdas();
	void set_lambdas();
	void initialize_lambda();
	void calculate_dq();
	void calculate_dfs();
	void calculate_dUs();
	void calculate_dU(const double& _lambda, double& _U, double& _dU);
	void calculate_T_lambda();
	void initialize_v_lambda(const double _T_lambda);
	void integrate_lambda();
	void allocate_storage();
	void deallocate_storage();
        template < int direction>
	void forward_reverse_copy(double& a, double& b);
        template < int direction>
	void forward_reverse_copy(double* a, double* b, int i);
        template < int direction>
	void forward_reverse_copy(double** a, double** b, int i, int j);
	template <int direction>
	void backup_restore_qfev();
	void init_GFF();
	void calculate_GFFs();
	void modify_qs(double scale, int j);
	void modify_qs(double **scales);
        void modify_q_buff(const double scale);
	void update_lmp();
        void compute_f_lambda_charge_interpolation();
	double compute_epair();
	void update_a_lambda();
	};

}    // namespace LAMMPS_NS


#endif
#endif
