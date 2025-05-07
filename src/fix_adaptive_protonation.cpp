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
/* ---------- v0.10.15----------------- */
// Please remove unnecessary includes
#include "fix_adaptive_protonation.h"

#include "atom.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "input.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "output.h"
#include "region.h"
#include "respa.h"
#include "update.h"
#include "variable.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "dihedral.h"
#include "force.h"
#include "group.h"
#include "improper.h"
#include "kspace.h"
#include "pair.h"

#include "thermo.h"
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdio.h>

#include <iostream>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum { NEITHER = -1, SOLID = 0, SOLVENT = 1 };
enum { F_NONE, RESET_MID = 1 << 1, INIT_MID = 1 << 2 };

/* --------------------------------------------------------------------------------------- */

FixAdaptiveProtonation::FixAdaptiveProtonation(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), n_protonable{0}
{
  if (narg < 7) utils::missing_cmd_args(FLERR, "fix adaptive_protonation", error);

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
  size_vector = 3;
  size_peratom_cols = 0;
  peratom_freq = nevery;
  extscalar = 0;
  extvector = 0;

  // Enabling the comm_forward
  comm_forward = 1;

  /* This part used to be in the setup() function, 
    * however since this fix adaptive protonation is
    * deleted and added everytime the number of protonation
    * state changes this part has to be inside the constructor
    */

  nmax = atom->nmax;
  vector_atom = new double[nmax];

  nmolecules = 0;

  if (flags & RESET_MID) set_molecule_id();

  int nlocal = atom->nlocal;
  int *molecule = atom->molecule;

  int nmolecules_local = 0;
  int nmolecules_total;

  for (int i = 0; i < nlocal; i++) {
    if (atom->molecule[i] > nmolecules_local) nmolecules_local = atom->molecule[i];
  }

  MPI_Allreduce(&nmolecules_local, &nmolecules_total, 1, MPI_INT, MPI_MAX, world);
  nmolecules = nmolecules_total;

  nmolecules++;
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
  // Reading the pH structure files
  pH_structure_storage = std::make_unique<constant_pH_structures>(lmp, fileName1, fileName2);
  pH_structure_storage->read_pH_structure_files();

  // Checking if the atom style contains the molecules information
  if (atom->molecular != 1) error->all(FLERR, "Illegal atom style in the fix adpative protonation");

  // Request a full neighbor list
  int list_flags = NeighConst::REQ_OCCASIONAL | NeighConst::REQ_FULL;

  // request for a neighbor list
  neighbor->add_request(this, list_flags);

  std::fill(nchanges.begin(), nchanges.end(), 0);
}

/* ---------------------------------------------------------------------------------------
   Setup
   --------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::setup(int /*vflag*/) {}

/* ---------------------------------------------------------------------------------------
    It is need to access the neighbor list
   --------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* --------------------------------------------------------------------------------------- */

int FixAdaptiveProtonation::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = vector_atom[j];
  }
  return m;
}

/* -------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) vector_atom[i] = buf[m++];
}

/* --------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::initial_integrate(int /*vflag*/)
{
  if (update->ntimestep % nevery) return;

  /* 
    * Building the neighbor list
    * every nevery steps 
    */
  neighbor->build_one(list);

  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    if (vector_atom) delete[] vector_atom;
    vector_atom = nullptr;
    vector_atom = new double[nmax];
  }

  // If I do not put this to zero, it will have a very large value making the if statement false.
  int nmolecules_local = 0;
  int nmolecules_total;

  for (int i = 0; i < atom->nlocal; i++) {
    if (atom->molecule[i] > nmolecules_local) nmolecules_local = atom->molecule[i];
  }

  MPI_Allreduce(&nmolecules_local, &nmolecules_total, 1, MPI_INT, MPI_MAX, world);

  if (nmolecules_total > nmolecules) {
    nmolecules = nmolecules_total;
    deallocate_storage();
    allocate_storage();
  }

  // Counting the number of water molecules surrounding the protonable molecules
  mark_protonation_deprotonation();

  // Communicating the ghost atom information
  comm->forward_comm(this);

  // This is required since the fix_constant_pH.cpp does not deal with those molecules in the solid
  modify_protonation_state();

  // Resetting the mark_prev parameter to help us keep the track of which molecule moves from solid to solvent and vice versa
  set_mark_prev();
}

/* ----------------------------------------------------------------------------------------
   Writing molids into a file
   ---------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::write_molids(const std::string& file_name) const
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
  molecule_size.reset();
  molecule_size_local.reset();
}

/* ----------------------------------------------------------------------------------------
   Allocating the storage

   ---------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::allocate_storage()
{
  using std::make_unique, std::fill;
  protonable_molids = make_unique<int[]>(nmolecules);
  mark = make_unique<int[]>(nmolecules + 1);
  mark_prev = make_unique<int[]>(nmolecules + 1);
  mark_local = make_unique<int[]>(nmolecules + 1);
  molecule_size = make_unique<int[]>(nmolecules + 1);
  molecule_size_local = make_unique<int[]>(nmolecules + 1);
  fill(protonable_molids.get(), protonable_molids.get() + nmolecules, -1);
  fill(mark.get(), mark.get() + nmolecules + 1, 0);
  fill(mark_local.get(), mark_local.get() + nmolecules + 1, 0);
  fill(molecule_size.get(), molecule_size.get() + nmolecules + 1, 0);
  fill(molecule_size_local.get(), molecule_size_local.get() + nmolecules + 1, 0);
  fill(mark_prev.get(), mark_prev.get() + nmolecules + 1,-1);
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
  int wnum;    // number of surrounding water molecules

  int *protonable = pH_structure_storage->protonable.get();
  // Not safe, you should use std::shared_ptr instead..

  inum = list->inum;    // I do not need ghost atoms for inum. however, I need them in jnum
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  int *type = atom->type;
  int *molecule = atom->molecule;

  for (int ii = 0; ii < inum; ii++) {
    wnum = 0.0;
    int i = ilist[ii];
    molecule_size_local[molecule[i]] = molecule_size_local[molecule[i]] + 1;

    // Check if this atom is protonable --> if not do not bother with it.
    if (protonable[type[i]] == 0) {
      mark_local[molecule[i]] = NEITHER;
      vector_atom[i] = 0;
      continue;
    }

    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;

      if (type[j] == typeOW)
        wnum++;    // Just considering the Oxygens. It is possible that both O and H from the same water molecule are close to this atom.
    }
    if (wnum >= threshold) {
      mark_local[molecule[i]] += SOLVENT;
    } else {
      mark_local[molecule[i]] += SOLID;
    }
    vector_atom[i] = static_cast<double>(wnum);
  }

  // Reducing the values from various cpus
  MPI_Allreduce(mark_local.get(), mark.get(), nmolecules + 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(molecule_size_local.get(), molecule_size.get(), nmolecules + 1, MPI_INT, MPI_SUM,
                world);

  for (int i = 1; i < nmolecules + 1; i++) {
    if (molecule_size[i] == 0) {
      mark[i] = NEITHER;
      continue;
    }
    double test_condition = static_cast<double>(mark[i]) / static_cast<double>(molecule_size[i]);
    if (test_condition >= 0 && test_condition <= 0.5)
      mark[i] = SOLID;
    else if (test_condition > 0.5 && test_condition <= 1)
      mark[i] = SOLVENT;
    else if (test_condition > 1 || test_condition < -1)
      error->one(FLERR, "Error in fix adaptive_protonation: You should never have reached here!");
    else
      mark[i] = NEITHER;
  }
}

/* ----------------------------------------------------------------------------------------
   Setting separate molecule ids for different phosphate ions 
   It might need to be a separate command in LAMMPS
   ---------------------------------------------------------------------------------------- */

void FixAdaptiveProtonation::set_molecule_id()
{
  int natom = atom->nlocal + atom->nghost;
  int nlocal = atom->nlocal;
  int nmax = atom->nmax;
  int *molecule = atom->molecule;
  int *num_bond = atom->num_bond;
  int **bond_atom = atom->bond_atom;
  int *tag = atom->tag;    // atom-id

  for (int i = 0; i < nlocal; i++) {
    for (int k = 0; k < num_bond[i]; k++) {
      int jtag = bond_atom[i][k];    // the tag (atom-id) of kth bonds of atom i
      int j = atom->map(jtag);
      if (j == -1) {
        error->warning(FLERR, "Bond atom missing in fix AdaptiveProtonation");
        continue;
      }
      molecule[i] = MIN(molecule[i], molecule[j]);    // I am not sure about header for the MIN
      molecule[j] = molecule[i];
    }
  }

  // You need to think about neighbor exchange;
  /*
   no need for an exchange 
   as LAMMPS itself takes care of exchange.

   if atom i is in the proc n connected to atom j in proc n both of the atoms 
   will be among the ghost atoms of the other atom. And as the minimum of the molecule_id
   is the same in both the procs both the atoms would end up having the same 
   molecule id and so there is no need for an atom exchange.
   Of course if the molecules are long spanning multiple procs there is a need
   for atom exchange here.
   */
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

  fill(mark_prev.get(), mark_prev.get() + nmolecules + 1, 0);    // zero is for SOLID
  for (int i = 0; i < n_protonable; i++)
    mark_prev[protonable_molids[i]] = SOLVENT;
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
  std::array<int, 3> nchanges_local = {0, 0, 0};
  double q_change_local = 0;
  double q_init;

  // I am not sure if this is necessary or not.
  double **pH1qs = pH_structure_storage->pH1qs;
  double **pH2qs = pH_structure_storage->pH2qs;

  for (int i = 0; i < nlocal; i++) {
    switch (mark[molecule[i]]) {
      case NEITHER:    // Not protonable ----> nothing to do here
        break;

      case SOLVENT:    // The molecule is in the water
        switch (mark_prev[molecule[i]]) {
          case SOLID:      // The molecule was in the solid before
          case NEITHER:    // First step (initial value of mark_prev is -1)
            q_init = q[i];
            q[i] = pH2qs[type[i]][0];
            q_change_local += q[i] - q_init;
            nchanges_local[0]++;
            nchanges_local[1]++;
            break;

          case SOLVENT:    // The molecule was already in water → do nothing
            break;

          default:    // Catch unexpected values
            error->all(FLERR, "Unexpected value in mark_prev[molecule[i]] for SOLVENT case");
            break;
        }
        break;    //  Prevent fall-through

      case SOLID:    // The molecule is in the solid
        switch (mark_prev[molecule[i]]) {
          case SOLVENT:    // It came from the water ----> deprotonate it
          case NEITHER:    // First step (initial value of mark_prev is -1)
            q_init = q[i];
            q[i] = pH1qs[type[i]][0];
            q_change_local += q[i] - q_init;
            nchanges_local[0]++;
            nchanges_local[2]++;
            break;

          case SOLID:    // It was already in the solid ----> do nothing
            break;

          default:    // Catch unexpected values
            error->all(FLERR, "Unexpected value in mark_prev[molecule[i]] for SOLID case");
            break;
        }
        break;    //  Prevent fall-through

      default:    // Catch unexpected values in `mark[molecule[i]]`
        error->all(FLERR, "Unexpected value in mark[molecule[i]]");
        break;
    }
  }

  MPI_Allreduce(&q_change_local, &q_change, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(nchanges_local.data(), nchanges.data(), 3, MPI_INT, MPI_SUM, world);

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
  return 0.0;
}
