#pragma once

#include <fstream>
#include <string>
#include <array>

#include "pointers.h"

using std::ifstream;
using std::string;

namespace LAMMPS_NS {
class constant_pH_structures : protected Pointers {
  friend class FixAdaptiveProtonation;
  friend class FixConstantPH;

 public:
  constant_pH_structures(LAMMPS *lmp, const string &fileName1, const string &fileName2);

  ~constant_pH_structures();
  constant_pH_structures(const constant_pH_structures &rhs) = delete;
  constant_pH_structures &operator=(const constant_pH_structures &rhs) = delete;
  constant_pH_structures(constant_pH_structures &&rhs) = delete;
  constant_pH_structures &operator=(constant_pH_structures &&rhs) = delete;

  void read_pH_structure_files();

 private:
  ifstream pHStructureFile1, pHStructureFile2;

  /* These variables will be accessed by fix_adaptive_protonation and fix_constant_pH*/
  double **pH1qs, **pH2qs;
  std::unique_ptr<int[]> typePerProtMol;
  std::unique_ptr<int[]> protonable;
  int pHnStructures1, pHnStructures2;
  int pHnTypes1, pHnTypes2;
};

class constant_pH_state : protected Pointers {
  friend class FixConstantPH;
  friend class FixNHConstantPH;
  friend class FixAdaptiveProtonation;
  friend class ComputeGFFConstantPH;
  friend class ComputeTempConstantPH;
 public:
  // with fix_adaptive_protonation.h class
  constant_pH_state(LAMMPS *lmp, const std::array<double,2>& lambda_masses, const int& N_buff_);
  // set in the input arguments of the fix_constant_pH.h class
  constant_pH_state(LAMMPS *lmp, std::unique_ptr<int []>& molids_, const int& n_lambdas_, const std::array<double,2>& lambda_masses, const int& N_buff_);
  // based on the values of the prev_pH_state_
  constant_pH_state(LAMMPS *lmp, std::unique_ptr<int []>& molids_, 
    const int& n_lambdas_, const std::array<double,2>& lambda_masses, const int& N_buff_, 
    const std::unique_ptr<constant_pH_state>& prev_pH_state_);

  ~constant_pH_state();
  constant_pH_state(const constant_pH_state& rhs);
  constant_pH_state& operator=(const constant_pH_state& rhs);
  constant_pH_state(constant_pH_state&& rhs) noexcept;
  constant_pH_state& operator=(constant_pH_state&& rhs) noexcept;


  int  reset_lambdas(const int& n_lambdas_, const std::unique_ptr<constant_pH_state>& prev_pH_state_);
  void set_zero();
  void broadcast();

 private:
  
   // Lambda arrays
   double **lambdas, **v_lambdas, **a_lambdas, **m_lambdas;
   std::unique_ptr<int[]> molids;
   int n_lambdas;
   double mass_lambda;

     // Parameters for the buffer
  double lambda_buff, v_lambda_buff, a_lambda_buff, m_lambda_buff;
  // number of buffer points
  int N_buff;

  void allocate_lambdas();
  void deallocate_lambdas();

};
}    // namespace LAMMPS_NS

