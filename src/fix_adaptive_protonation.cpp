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
#include "fix_adaptive_protonation.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "math_const.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "update.h"

#include <algorithm>       // std::fill, std::fill_n
#include <array>           // std::array
#include <cstring>         // std::strcmp
#include <fstream>         // std::ifstream, std::ofstream
#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum { NEITHER = -1, SOLID = 0, SOLVENT = 1 };
enum { F_NONE, RESET_MID = 1 << 1, INIT_MID = 1 << 2 };

static constexpr double frac_low  = 0.4;
static constexpr double frac_high = 0.6;
static constexpr int max_moleset_iter = 10;

/* --------------------------------------------------------------------------------------- */

FixAdaptiveProtonation::FixAdaptiveProtonation(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), n_protonable{0}
{
  if (narg < 8) utils::missing_cmd_args(FLERR, "fix adaptive_protonation", error);

  nevery = utils::numeric(FLERR, arg[3], false, lmp);

  if (nevery < 0) error->all(FLERR, "Illegal fix adaptive_protonation every value {}", nevery);

  fileName1 = arg[4];
  fileName2 = arg[5];

  typeOW = utils::numeric(FLERR, arg[6], false, lmp);
  threshold = utils::numeric(FLERR, arg[7], false, lmp);

  flags = 0;
  int iarg = 8;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "reset_molecule_ids") == 0) {
      flags |= RESET_MID;
      iarg++;
    } else if (strcmp(arg[iarg], "initial_molids") == 0) {
      flags |= INIT_MID;
      if (comm->me == 0) {
        init_molid_file.open(arg[iarg + 1], std::ifstream::in);
        if (!init_molid_file.is_open())
          error->one(FLERR, "Unable to open the file {}", arg[iarg + 1]);
      }
      iarg += 2;
    } else if (strcmp(arg[iarg], "intermediate_file") == 0) {
      flags |= INIT_MID;
      if (comm->me == 0) {
        init_molid_file.open(arg[iarg + 1], std::ifstream::in);
        if (!init_molid_file.is_open())
          error->one(FLERR, "Unable to open the intermediate file {}", arg[iarg + 1]);
      }
      iarg += 2;
    } else if (strcmp(arg[iarg],"cutoff") == 0) {
      rprobe = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;
    } else
      error->all(FLERR, "Unknown keyword");
  }

  if ((flags & RESET_MID) && (flags & INIT_MID))
    error->one(FLERR,
               "It is not possible to have both the initial_molids and reset_molecule_ids keywords "
               "in the fix adaptive_protonation");

  dynamic_group_allow = 0;
  scalar_flag = 1;
  vector_flag = 1;
  peratom_flag = 1;
  comm_forward = 0;
  maxexchange = 1;
  size_vector = 3;
  size_peratom_cols = 1;
  peratom_freq = nevery;
  extscalar = 0;
  extvector = 0;


  /* This part used to be in the setup() function, 
    * however since this fix adaptive protonation is
    * deleted and added everytime the number of protonation
    * state changes this part has to be inside the constructor
    */

  nmax = atom->nmax;
  vector_atom = new double[nmax];

  nmolecules = 0;

  int nlocal = atom->nlocal;
  int *molecule = atom->molecule;

  int nmolecules_local = 0;
  int nmolecules_total;

  for (int i = 0; i < nlocal; i++) {
    if (atom->molecule[i] > nmolecules_local) nmolecules_local = atom->molecule[i];
  }

  MPI_Allreduce(&nmolecules_local, &nmolecules_total, 1, MPI_INT, MPI_MAX, world);
  nmolecules = nmolecules_total;

  // Adding another molecule id for atoms with molecule[i] == 0
  nmolecules++;
  // molecule_id is in 1-based indexing
  for (int i = 0; i < nlocal; i++) {
    if (molecule[i] == 0) molecule[i] = nmolecules;
  }

  // The allocate_storage() function needs to know the nmolecules to set the arrays
  allocate_storage();

  if (flags & INIT_MID) read_molids_file();

  /* Only if it has not read the molids from a file the n_protonable should set to zero.
    *  Otherwise, it had been set by the read_molids_file()
    */
  if (!(flags & INIT_MID)) n_protonable = 0;
}

/* --------------------------------------------------------------------------------------- */

FixAdaptiveProtonation::~FixAdaptiveProtonation()
{
  if (vector_atom) delete[] vector_atom;

  // this is not needed since I am using std::unique_ptr
  // In destructin it will be deallocated on its own.
  deallocate_storage();

  vector_atom = nullptr;
}

/* --------------------------------------------------------------------------------------- */

int FixAdaptiveProtonation::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  return mask;
}

/* --------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::init()
{
  // Checking if the atom style contains the molecules information
  if (atom->molecular != 1) error->all(FLERR, "Illegal atom style in the fix adpative protonation");

  // The atom_style should contain the charge information.
  if (!atom->q_flag) error->all(FLERR, "Atom style has no charges for adaptive_protonation");

  // Reading the pH structure files
  pH_structure_storage = std::make_unique<constant_pH_structures>(lmp, fileName1, fileName2);
  pH_structure_storage->read_pH_structure_files();


  // Request a full neighbor list
  int list_flags = NeighConst::REQ_OCCASIONAL | NeighConst::REQ_FULL;

  // request for a neighbor list
  neighbor->add_request(this, list_flags);

  std::fill(nchanges.begin(), nchanges.end(), 0);

  if (flags & RESET_MID) set_molecule_id();

  //
  q_orig = std::make_unique<double []>(nmax);

}

/* ---------------------------------------------------------------------------------------
    The neighbor list cannot be made in the init step so we do it here in the setup step
   --------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::setup(int /*vflag*/) 
{
  protonation_deprotonation();
}


/* ---------------------------------------------------------------------------------------
    It is need to access the neighbor list
   --------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* --------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::initial_integrate(int /*vflag*/)
{
  protonation_deprotonation();
}

/* --------------------------------------------------------------------------------------- 
    The function to transfer q_orig between ranks
   --------------------------------------------------------------------------------------- */

int FixAdaptiveProtonation::pack_exchange(int i, double* buf)
{
  buf[0] = q_orig[i];
  buf[1] = vector_atom[i];
  return 2;
}

int FixAdaptiveProtonation::unpack_exchange(int nlocal, double* buf)
{
  q_orig[nlocal] = buf[0];
  vector_atom[nlocal] = buf[1];
  return 2;
}

void FixAdaptiveProtonation::grow_arrays(int nmax_new)
{
  auto q_new = std::make_unique<double []>(nmax_new);
  int keep = std::min(nmax,nmax_new);
  if (keep > 0) std::copy(q_orig.get(),q_orig.get()+keep,q_new.get());
  if (keep < nmax_new) std::fill(q_new.get()+keep,q_new.get()+nmax_new,0.0);
  q_orig.swap(q_new);

  double *new_vector_atom = new double[nmax_new];
  if (keep > 0) std::copy(vector_atom,vector_atom+keep,new_vector_atom);
  if (keep < nmax_new) std::fill(new_vector_atom+keep,new_vector_atom+nmax_new,0.0);
  delete [] vector_atom;
  vector_atom = new_vector_atom;
  nmax = nmax_new;
}

void FixAdaptiveProtonation::copy_arrays(int i, int j , int /*deflag*/)
{
  q_orig[j] = q_orig[i];
  vector_atom[j] = vector_atom[i];
}

/* ----------------------------------------------------------------------------------------
    Checking the protonation deprotonation
   ---------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::protonation_deprotonation()
{
  /* 
    * Building the neighbor list
    * every nevery steps 
    */
    if (!list) error->all(FLERR, "Neighbor list not initialized for adaptive_protonation");
    neighbor->build_one(list);
  
    if (atom->nmax > nmax) {
      nmax = atom->nmax;
      if (vector_atom) delete[] vector_atom;
      vector_atom = nullptr;
      vector_atom = new double[nmax];
      std::fill_n(vector_atom,nmax,0);
    }
  
    // If I do not put this to zero, it will have a very large value making the if statement false.
    int nmolecules_local = 0;
    int nmolecules_total;
  
    for (int i = 0; i < atom->nlocal; i++) {
      if (atom->molecule[i] > nmolecules_local) nmolecules_local = atom->molecule[i];
    }
  
    MPI_Allreduce(&nmolecules_local, &nmolecules_total, 1, MPI_INT, MPI_MAX, world);
    nmolecules_total++;
  
    if (nmolecules_total > nmolecules) {
      nmolecules = nmolecules_total;
      deallocate_storage();
      allocate_storage();
    }
  
    // Counting the number of water molecules surrounding the protonable molecules
    if (update->ntimestep%nevery == 0) {
      rampStep = 1;
      mark_protonation_deprotonation();
      backup_init_qs();
    }
  
    // This is required since the fix_constant_pH.cpp does not deal with those molecules in the solid
    modify_protonation_state();
    rampStep++;
  

    // Resetting the mark_prev parameter to help us keep the track of which molecule moves from solid to solvent and vice versa
    if (update->ntimestep%nevery == 0)
      set_mark_prev();
}

/* ----------------------------------------------------------------------------------------
   Writing molids into a file
   ---------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::write_molids(const std::string &file_name) const
{
  if (comm->me == 0) {
    if (file_name.empty()) error->one(FLERR, "The wrong file name in fix adaptive protonation");
    std::ofstream output_file(file_name, std::ofstream::out);
    if (!output_file.is_open()) error->one(FLERR, "Cannot open the molid files for writing");
    output_file << n_protonable << std::endl;
    output_file << "The molids file" << std::endl;
    output_file << "with the intermediate information of the fix adaptive protonation command"
                << std::endl;
    for (int i = 0; i < n_protonable; i++) { output_file << protonable_molids[i] << std::endl; }
  }
}

/* ----------------------------------------------------------------------------------------
   Deallocating the storage
   ---------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::deallocate_storage()
{
  protonable_molids.reset();
  mark.reset();
  mark_prev.reset();
  mark_local.reset();
  protonable_size.reset();
  protonable_size_local.reset();
}

/* ----------------------------------------------------------------------------------------
   Allocating the storage

   ---------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::allocate_storage()
{
  using std::make_unique, std::fill_n;
  protonable_molids     = make_unique<int[]>(nmolecules);
  mark                  = make_unique<int[]>(nmolecules + 1);
  mark_prev             = make_unique<int[]>(nmolecules + 1);
  mark_local            = make_unique<int[]>(nmolecules + 1);
  protonable_size       = make_unique<int[]>(nmolecules + 1);
  protonable_size_local = make_unique<int[]>(nmolecules + 1);
  fill_n(protonable_molids.get(), nmolecules, -1);
  fill_n(mark.get(), nmolecules + 1, 0);
  fill_n(mark_prev.get(), nmolecules + 1,NEITHER);
  fill_n(mark_local.get(), nmolecules + 1, 0);
  fill_n(protonable_size.get(), nmolecules + 1, 0);
  fill_n(protonable_size_local.get(), nmolecules + 1, 0);
  /* I put it on purpose so in the first step every molecule changes unless 
    * INIT_MIDS is set in which case the read_init_mids() function rewrites this.
    */
}

/* ----------------------------------------------------------------------------------------
   getting the number of water molecules near phosphates and 
   flag them for protonation/deprotonation
   ---------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::mark_protonation_deprotonation()
{
  int *ilist, *jlist, *numneigh, **firstneigh;
  int inum, jnum;
  int nlocal = atom->nlocal;
  double** x = atom->x;

  const int* protonable = pH_structure_storage->protonable.get();

  // resetting the mark_local and molecule_size_local before going through atoms
  std::fill_n(mark_local.get(),nmolecules+1,0);
  std::fill_n(protonable_size_local.get(),nmolecules+1,0);
  std::fill_n(vector_atom,nmax,0.0);

  inum = list->inum;    // I do not need ghost atoms for inum. however, I need them in jnum
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  int *type = atom->type;
  int *molecule = atom->molecule;

  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];

    // Check if this atom is protonable --> if not do not bother with it.
    if (protonable[type[i]] == 0) {
      continue;
    } else {
      protonable_size_local[molecule[i]]++;
    }

    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;

      if (type[j] != typeOW) continue;

      double dx = x[i][0]-x[j][0];
      double dy = x[i][1]-x[j][1];
      double dz = x[i][2]-x[j][2];
      domain->minimum_image(dx,dy,dz);
      double rsq = std::sqrt(dx*dx+dy*dy+dz*dz);
      if (rsq < rprobe)
        vector_atom[i] += 1.0;    // Just considering the Oxygens. It is possible that both O and H from the same water molecule are close to this atom.
    }

    if (vector_atom[i] >= threshold) {
      mark_local[molecule[i]] += SOLVENT;
    } else {
      mark_local[molecule[i]] += SOLID;
    }
  }

  // Reducing the values from various cpus
  MPI_Allreduce(mark_local.get(), mark.get(), nmolecules + 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(protonable_size_local.get(), protonable_size.get(), nmolecules + 1, MPI_INT, MPI_SUM,
                world);

  constexpr double eps = 0.01;

  for (int i = 1; i < nmolecules + 1; i++) {
    if (!protonable_size[i]) {
      mark[i] = NEITHER;
      continue;
    }
    double test_condition = static_cast<double>(mark[i]) / static_cast<double>(protonable_size[i]);
    if (test_condition >= 0 && test_condition <= frac_low)
      mark[i] = SOLID;
    else if (test_condition >= frac_high && test_condition <= 1)
      mark[i] = SOLVENT;
    else if (test_condition > 1 + eps || test_condition < -1 - eps)
      error->one(FLERR, "Error in fix adaptive_protonation: You should never have reached here!");
    else {
      int prev = mark_prev[i];
      if (prev == SOLID || prev == SOLVENT) mark[i] = prev;
      else mark[i] = SOLID;
    }
  }

}

/* ----------------------------------------------------------------------------------------
   Backup the initial charges
   ---------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::backup_init_qs()
{
  int nlocal = atom->nlocal;
  double* q = atom->q;

  for (int i = 0; i < nlocal; i++)
    q_orig[i] = q[i];
}

/* ----------------------------------------------------------------------------------------
   Setting separate molecule ids for different phosphate ions 
   It might need to be a separate command in LAMMPS
   ---------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::set_molecule_id()
{
  int nlocal = atom->nlocal;
  int *molecule = atom->molecule;
  int *num_bond = atom->num_bond;
  int **bond_atom = atom->bond_atom;
   
  bool changed;
  for (int iter = 0; iter < max_moleset_iter; iter++) {
    changed = false;
    for (int i = 0; i < nlocal; i++) {
      int mi = molecule[i];
      for (int k = 0; k < num_bond[i]; k++) {
        const int j = atom->map(bond_atom[i][k]);
        if (j < 0) continue;
        const int mmin = MIN(mi, molecule[j]);
        if (mmin != mi) { 
          mi = mmin;
          changed = true;
        }
      }
      molecule[i] = mi;
    }
    int any = changed ? 1 : 0, any_global = 0;
    MPI_Allreduce(&any, &any_global, 1, MPI_INT, MPI_MAX, world);
    if (!any_global) break;
    comm->exchange();
  }
}

/* ----------------------------------------------------------------------------------------
   Reading the file containing the initial molids
   ---------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::read_molids_file()
{
  using std::getline, std::string, std::stoi, std::fill;
  /*
    *  File format
    *  comment_1
    *  comment_2
    *  n_molids 
    *  molid1
    *  molid2
    *  ...
    *
    *  molidn
    */

  string line;
  if (comm->me == 0) {
    if (!init_molid_file.is_open())
      error->one(FLERR,"init_molid_file is not open!");
    // n_protonable
    getline(init_molid_file, line);
    n_protonable = stoi(line);
    // comment-1
    getline(init_molid_file, line);
    // comment-2
    getline(init_molid_file, line);

    // Checking that if there is enough space in the allocated arrays
    if (n_protonable > nmolecules) error->one(FLERR, "Unknown error");

    for (int i = 0; i < n_protonable; i++) {
      if (!getline(init_molid_file, line))
        error->one(FLERR, "Error in reading the init_molid_file");
      protonable_molids[i] = stoi(line);
    }
  }

  // First broadcasting the size;
  MPI_Bcast(&n_protonable, 1, MPI_INT, 0, world);
  // Then broadcasting the individual molids
  MPI_Bcast(protonable_molids.get(), n_protonable, MPI_INT, 0, world);

  // chaning the mark_prev from NEITHER to 0 does not matter
  // as in the modify_protonation_state we check if the atom 
  // type is protonable or not (NEITHER)
  fill(mark_prev.get(), mark_prev.get() + nmolecules + 1, 0);    // zero is for SOLID
  for (int i = 0; i < n_protonable; i++) mark_prev[protonable_molids[i]] = SOLVENT;
  // protonable molecules are exposed to the SOLVENT.
}

/* ----------------------------------------------------------------------------------------
   Getting the molids for protonable molecules
   The molids must be allocated otherwise an error occurs
   ---------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::get_protonable_molids(int *_molids) const
{
  if (_molids == nullptr && n_protonable != 0)
    error->all(FLERR,
               "The _molids array in the fix adaptive protonation get protonable_molids must be "
               "allocated");

  for (int i = 0; i < n_protonable; i++) { _molids[i] = protonable_molids[i]; }
}

/* ----------------------------------------------------------------------------------------
   Changing from the protonated to deprotonated states --> Moving from the solvent to the solid phase
   ---------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::modify_protonation_state()
{
  int nlocal = atom->nlocal;
  double *q = atom->q;
  int *type = atom->type;
  int *molecule = atom->molecule;
  double q_change_local = 0;
  double q_init;
  double step = static_cast<double>(rampStep);
  double nstepInv = 1.0/static_cast<double>(nevery);

  // I am not sure if this is necessary or not.
  double **pH1qs = pH_structure_storage->pH1qs;
  double **pH2qs = pH_structure_storage->pH2qs;
  const int *protonable = pH_structure_storage->protonable.get();
  std::fill_n(nchanges.data(),3,0);

  if (comm->me == 0) {
    for (int m = 1; m <= nmolecules; ++m) {

      const int cur  = mark[m];
      const int prev = mark_prev[m];
  
      if (cur == SOLVENT && (prev == SOLID || prev == NEITHER)) {
        nchanges[0]++;   // total flips
        nchanges[1]++;   // to solvent
      } else if (cur == SOLID && (prev == SOLVENT || prev == NEITHER)) {
        nchanges[0]++;
        nchanges[2]++;   // to solid
      }
    }
  }

  MPI_Bcast(nchanges.data(),3,MPI_INT,0,world);

  //double frac = std::min(step*nstepInv,1.0);
  // I am not clamping it on purpose so that 
  // I can check if there is any atomic 
  // exchange that make q_orig irrelevant.
  double frac = step*nstepInv;
  double q_new;

  for (int i = 0; i < nlocal; i++) {
    if (!protonable[type[i]]) continue;
    switch (mark[molecule[i]]) {
      case NEITHER:    // Not protonable ----> nothing to do here
        break;

      case SOLVENT:    // The molecule is in the water
        // The molecule was in the solid before or it is the first step
        if (mark_prev[molecule[i]] == SOLID || mark_prev[molecule[i]]== NEITHER)
        {
          q_init = q[i];
          q_new = q_orig[i] + frac*(pH2qs[type[i]][0]-q_orig[i]);
          if (!std::isfinite(q_new)) error->one(FLERR,"The q[{}] is infinite!",i);
          q[i] = q_new;
          q_change_local += q[i] - q_init;
        }
        else if (mark_prev[molecule[i]] == SOLVENT)
          break;
        else
          error->all(FLERR, "Unexpected value in mark_prev[molecule[i]] for SOLVENT case: {}",mark_prev[molecule[i]]);
        break;    //  Prevent fall-through

      case SOLID:    // The molecule is in the solid
        // It came from the water ----> deprotonate it
        // I do not want to mess up the initial charge distribution in the interior of the solid
        if (mark_prev[molecule[i]] == SOLVENT) {
          q_init = q[i];
          q_new = q_orig[i] + frac*(pH1qs[type[i]][0]-q_orig[i]);
          if (!std::isfinite(q_new)) error->one(FLERR,"The q[{}] is infinite!",i);
          q[i] = q_new;
          q_change_local += q[i] - q_init;
          break;
        } else if (mark_prev[molecule[i]] == SOLID || mark_prev[molecule[i]] == NEITHER)
          break;
        else
          error->all(FLERR, "Unexpected value in mark_prev[molecule[i]] for SOLID case: {}",mark_prev[molecule[i]]);
        break;    //  Prevent fall-through

      default:    // Catch unexpected values in `mark[molecule[i]]`
        error->all(FLERR, "Unexpected value in mark[molecule[i]]");
        break;
    }
  }

  MPI_Allreduce(&q_change_local, &q_change, 1, MPI_DOUBLE, MPI_SUM, world);

  // Check if we need to change n_protonable and protonable_molids
  /*
    * It is possible that nchanges_local[1] and nchanges_local[2] cancel each other,
    * however, since different molecules are protonable, I would prefer to deallocate
    * and reallocate the protonable_molids so that fix_constant_pH is informed of the change
    * and it reinitializes the v_lambdas.
    */
  if (nchanges[0]) {
    int j = 0;
    for (int i = 1; i <= nmolecules; i++)
      if (mark[i] == SOLVENT) protonable_molids[j++] = i;

    n_protonable = j;
  }
}

/* --------------------------------------------------------------------------
   Set the mark_prev
   -------------------------------------------------------------------------- */

void FixAdaptiveProtonation::set_mark_prev()
{
  for (int i = 0; i < nmolecules + 1; i++) mark_prev[i] = mark[i];
}

/* --------------------------------------------------------------------------
   Output the changes in the number of hydrogen atoms
   -------------------------------------------------------------------------- */

double FixAdaptiveProtonation::compute_scalar()
{
  return nchanges[0];
}

/* --------------------------------------------------------------------------
   Output the changes in the topology --> nbonds and nangles
   -------------------------------------------------------------------------- */

double FixAdaptiveProtonation::compute_vector(int n)
{
  switch (n) {
    // 1
    case 0:
      return static_cast<double>(nchanges[0]);
    case 1:
      return static_cast<double>(nchanges[1]);
    case 2:
      return static_cast<double>(nchanges[2]);
  }
  return -1;
}

/* --------------------------------------------------------------------------
   This part needs to be updated in the final version ....
   -------------------------------------------------------------------------- */

double FixAdaptiveProtonation::memory_usage()
{
  const size_t ints = (5*(nmolecules+1) + nmolecules) * sizeof(int);
  const size_t dbls = static_cast<size_t>(nmax) * sizeof(double);
  return static_cast<double>(ints + dbls);
}
