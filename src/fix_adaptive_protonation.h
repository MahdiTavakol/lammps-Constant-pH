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
/* ---v0.10.15----- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(adaptive_protonation,FixAdaptiveProtonation);
// clang-format on
#else

#ifndef LMP_FIX_ADAPTIVE_PROTONATION_H
#define LMP_FIX_ADAPTIVE_PROTONATION_H

#include "fix.h"
#include "atom_vec.h"

namespace LAMMPS_NS {

   class FixAdaptiveProtonation : public Fix {
   public:
      FixAdaptiveProtonation(class LAMMPS*, int, char**);
      ~FixAdaptiveProtonation() override;
      int setmask() override;
      void init() override;
      void setup(int) override;
      void initial_integrate(int) override;
      double compute_scalar() override;
      double compute_vector(int) override;
      double memory_usage() override;
      void init_list(int, class NeighList*) override;


      // Getting the number of protonable molids;
      void get_n_protonable(int & n_lambdas) const
      {
         n_lambdas = this->n_protonable;
      }
      // Getting the protonable molids (The molids array must be allocated otherwise an error is produced)
      void get_protonable_molids(int * molids) const;
      // Changes in the environment which is needed in the fix_constant_pH to check if it needs to get the protonable_molids or not.
      void get_n_changes(int & _nchanges) const
      {
         _nchanges = this->nchanges[0];
      }
      
      // Writes molids information to a file which can be used later with fix constant_pH whenever the commands keyword of the fix_constant_pH is invoked
      void write_molids(const char* const file_name ) const;

   protected:

      /// -----> This part is similar to the part in the fix_constant_pH.cpp, so should be a separate class
      // The input files 
      FILE* pHStructureFile1, * pHStructureFile2;

   
      // The information on the protonable species
      int pHnStructures1, pHnStructures2;
      int pHnTypes1, pHnTypes2;
      double **pH1qs, **pH2qs;
      int * typePerProtMol;
      int *protonable;

      void read_pH_structure_files();

      // <------ This part is similar to the part in the fix_constant_pH.cpp, so should be a separate class
      
      int flags;

      /* When a huge number of lambdas is added to the system the simulation becomes unstable.
         So, there might be a need to input the molids of the lambdas
      */
      FILE* init_molid_file;
      


      // I need to detect water molecules
      int typeOW;
      // The threshold for the number of neighboring water molecules
      double threshold;
   

      // Neighborlist is required for accessing neighbors
      class NeighList* list;


      /*
       * -1 ---> NEITHER
       *  0 ---> SOLID
       *  1 ---> SOLVENT
       */

      int * mark;
      int * mark_local;
      int * mark_prev; // For the previous step
      int * mark_per_mol; // If one atom have mark == 1 all the atoms of that molecule should have mark == 1

      int * molecule_size; // used to average the mark for each molecule
      int * molecule_size_local;
      
      // Tracking the changes in the q_total
      double q_change;

      // array to access the molids of the protonable molecules
      int * protonable_molids;
      int n_protonable;

      // Changes in the environment which is needed in the fix_constant_pH to check if it needs to get the protonable_molids or not.
      int nchanges[3]; // changes, solid_to_water, water_to_solid


      // maximum number of atoms and number of molecules
      int nmax;
      int nmolecules;


      // Deallocating storage 
      void deallocate_storage();
      // Allocating storage
      void allocate_storage();
      // Reseting all the molids 
      void set_molecule_id();
      // Reading the molids from a file <--> For the file structure please have a look at the implementation file.
      void read_molids_file();
      // Mark phosphate atoms for protonation/deprotonation
      void mark_protonation_deprotonation();
      // Modifying the protonation state
      void modify_protonation_state();
      // Setting the mark for the previous state
      void set_mark_prev();

   };

}    // namespace LAMMPS_NS

#endif
#endif
