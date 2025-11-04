#include "constant_pH_structures.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"

#include <algorithm>
#include <sstream>
#include <unordered_map>

using namespace LAMMPS_NS;

constant_pH_structures::constant_pH_structures(LAMMPS *lmp, const string &fileName1,
                                               const string &fileName2) :
    Pointers{lmp},
    pH1qs{nullptr}, pH2qs{nullptr}
{
  if (comm->me == 0) {
    pHStructureFile1.open(fileName1, std::ifstream::in);
    pHStructureFile2.open(fileName2, std::ifstream::in);

    if (!pHStructureFile1.is_open() || !pHStructureFile2.is_open())
      error->one(FLERR, "Unable to open the file");
  }
}

constant_pH_structures::~constant_pH_structures()
{
  if (pH1qs) memory->destroy(pH1qs);
  if (pH2qs) memory->destroy(pH2qs);

  pH1qs = nullptr;
  pH2qs = nullptr;
}

void constant_pH_structures::read_pH_structure_files()
{
  using std::string, std::getline, std::stoi, std::stof, std::stringstream;
  /* File format
    * Comment
    * pHnStructures
    * pHnTypes
    * type1,  number of type1 atoms in the protonable molecule, qState1, qState2, qState3
    * ... 
    * ...  
    */

  /*Allocating the required memory*/
  int ntypes = atom->ntypes;
  protonable = std::make_unique<int[]>(ntypes + 1);    //ntypes+1 so the atom types start from 1.
  typePerProtMol = std::make_unique<int[]>(ntypes + 1);

  auto parse_file = [&](std::ifstream &file, int &nStructures, double **&pHqs, int &pHnTypes,
                        const char *memory_string) {
    string line;
    if (comm->me == 0) {
      //comment
      getline(file, line);

      // pHnStructures
      getline(file, line);
      nStructures = stoi(line);
    }

    MPI_Bcast(&nStructures, 1, MPI_INT, 0, world);
    memory->create(pHqs, ntypes + 1, nStructures, memory_string);

    if (comm->me == 0) {
      getline(file, line);
      pHnTypes = stoi(line);

      for (int i = 1; i < ntypes + 1; i++) {
        protonable[i] = 0;
        typePerProtMol[i] = 0;
        for (int j = 0; j < nStructures; j++) pHqs[i][j] = 0.0;
      }

      for (int i = 0; i < pHnTypes; i++) {
        if (!getline(file, line))
          error->one(FLERR, "Error in reading the pH structure file in fix constant_pH");

        stringstream iss(line);

        string field;
        int type;
        getline(iss, field, ',');
        type = stoi(field);
        protonable[type] = 1;
        int type_per_prot_mol;
        getline(iss, field, ',');
        type_per_prot_mol = stoi(field);
        typePerProtMol[type] = type_per_prot_mol;

        double q;
        for (int j = 0; j < nStructures; j++) {
          getline(iss, field, ',');
          q = stod(field);
          pHqs[type][j] = q;
        }
      }
    }

    MPI_Bcast(&pHnTypes,1,MPI_INT,0,world);
    MPI_Bcast(protonable.get(), ntypes + 1, MPI_INT, 0, world);
    MPI_Bcast(typePerProtMol.get(), ntypes + 1, MPI_INT, 0, world);
    MPI_Bcast(pHqs[0], (ntypes + 1) * (nStructures), MPI_DOUBLE, 0, world);
  };

  parse_file(pHStructureFile1, pHnStructures1, pH1qs, pHnTypes1, "constant_pH:pH1qs");
  parse_file(pHStructureFile2, pHnStructures2, pH2qs, pHnTypes2, "constant_pH:pH2qs");
}

constant_pH_state::constant_pH_state(LAMMPS *lmp, const std::array<double,2>& lambda_masses, const int& N_buff_):
  Pointers{lmp},
  lambdas{nullptr}, v_lambdas{nullptr}, a_lambdas{nullptr}, m_lambdas{nullptr},
  n_lambdas{0}, mass_lambda{lambda_masses[0]},
  lambda_buff{0.0}, v_lambda_buff{0.0}, a_lambda_buff{0.0}, m_lambda_buff{lambda_masses[1]},
  N_buff{N_buff_} {}

constant_pH_state::constant_pH_state(LAMMPS *lmp, std::unique_ptr<int []>& molids_, 
  const int& n_lambdas_, const std::array<double,2>& lambda_masses, const int& N_buff_):
  Pointers{lmp},
  lambdas{nullptr}, v_lambdas{nullptr}, a_lambdas{nullptr}, m_lambdas{nullptr},
  molids{std::move(molids_)},
  n_lambdas{n_lambdas_}, mass_lambda{lambda_masses[0]},
  lambda_buff{0.0}, v_lambda_buff{0.0}, a_lambda_buff{0.0}, m_lambda_buff{lambda_masses[1]},
  N_buff{N_buff_}
{
  allocate_lambdas();
}

constant_pH_state::constant_pH_state(LAMMPS *lmp, std::unique_ptr<int []>& molids_, 
  const int& n_lambdas_, const std::array<double,2>& lambda_masses, const int& N_buff_, 
  const std::unique_ptr<constant_pH_state>& prev_pH_state_):
  constant_pH_state{lmp,molids_,n_lambdas_,lambda_masses,N_buff_}
{
  reset_lambdas(n_lambdas,prev_pH_state_);
}

constant_pH_state::~constant_pH_state()
{
  deallocate_lambdas();
}

constant_pH_state::constant_pH_state(const constant_pH_state& rhs):
  Pointers{rhs.lmp},
  lambdas{nullptr}, v_lambdas{nullptr}, a_lambdas{nullptr}, m_lambdas{nullptr},
  n_lambdas{rhs.n_lambdas}, mass_lambda{rhs.mass_lambda},
  lambda_buff{rhs.lambda_buff}, v_lambda_buff{rhs.v_lambda_buff},
  a_lambda_buff{rhs.a_lambda_buff}, m_lambda_buff{rhs.m_lambda_buff},
  N_buff{rhs.N_buff}
{
  allocate_lambdas();
  if (n_lambdas) {
    std::copy_n(&rhs.lambdas[0][0],3*n_lambdas,&lambdas[0][0]);
    std::copy_n(&rhs.v_lambdas[0][0],3*n_lambdas,&v_lambdas[0][0]);
    std::copy_n(&rhs.a_lambdas[0][0],3*n_lambdas,&a_lambdas[0][0]);
    std::copy_n(&rhs.m_lambdas[0][0],3*n_lambdas,&m_lambdas[0][0]);
    std::copy_n(rhs.molids.get(),n_lambdas,molids.get());
  }
}

constant_pH_state& constant_pH_state::operator=(const constant_pH_state& rhs)
{
  // Checking for self assignment
  if (this != &rhs) {
    // May be the number of lambdas are not the same
    // so we need to reallocate
    if (n_lambdas != rhs.n_lambdas) {
      deallocate_lambdas();
      mass_lambda = rhs.mass_lambda;
      n_lambdas = rhs.n_lambdas;
      allocate_lambdas();
      // or the mass_lambda is different
    } else if (mass_lambda != rhs.mass_lambda || m_lambda_buff != rhs.m_lambda_buff) {
      mass_lambda = rhs.mass_lambda;
      m_lambda_buff = rhs.m_lambda_buff;
    }
    

    if (n_lambdas) {
      std::copy_n(&rhs.lambdas[0][0],3*n_lambdas,&lambdas[0][0]);
      std::copy_n(&rhs.v_lambdas[0][0],3*n_lambdas,&v_lambdas[0][0]);
      std::copy_n(&rhs.a_lambdas[0][0],3*n_lambdas,&a_lambdas[0][0]);
      std::copy_n(&rhs.m_lambdas[0][0],3*n_lambdas,&m_lambdas[0][0]);
      std::copy_n(rhs.molids.get(),n_lambdas, molids.get());
    }
    lambda_buff = rhs.lambda_buff;
    v_lambda_buff = rhs.v_lambda_buff;
    a_lambda_buff = rhs.a_lambda_buff;
    m_lambda_buff = rhs.m_lambda_buff;
    N_buff = rhs.N_buff;
  }

  return *this;
}

constant_pH_state::constant_pH_state(constant_pH_state&& rhs) noexcept:
 Pointers{rhs.lmp},
 lambdas{rhs.lambdas}, v_lambdas{rhs.v_lambdas},
 a_lambdas{rhs.a_lambdas}, m_lambdas{rhs.m_lambdas},
 molids{std::move(rhs.molids)},
 n_lambdas{rhs.n_lambdas}, mass_lambda{rhs.mass_lambda},
 lambda_buff{rhs.lambda_buff}, v_lambda_buff{rhs.v_lambda_buff},
 a_lambda_buff{rhs.a_lambda_buff}, m_lambda_buff{rhs.m_lambda_buff},
 N_buff{rhs.N_buff}
{
  rhs.lambdas  = nullptr;
  rhs.v_lambdas = nullptr;
  rhs.a_lambdas = nullptr;
  rhs.m_lambdas = nullptr;
  rhs.n_lambdas = 0;
  rhs.mass_lambda = 0.0;
  rhs.lambda_buff = 0.0;
  rhs.v_lambda_buff = 0.0;
  rhs.a_lambda_buff = 0.0;
  rhs.m_lambda_buff = 0;
  rhs.N_buff = 0;
}

constant_pH_state& constant_pH_state::operator=(constant_pH_state&& rhs) noexcept
{
  if (this != &rhs) {
    deallocate_lambdas();
    // we have just one movable item here
    molids = std::move(rhs.molids);
    // manual moving 
    lambdas = rhs.lambdas;
    v_lambdas = rhs.v_lambdas;
    a_lambdas = rhs.a_lambdas;
    m_lambdas = rhs.m_lambdas;
    n_lambdas = rhs.n_lambdas;

    mass_lambda = rhs.mass_lambda;
    lambda_buff = rhs.lambda_buff;
    v_lambda_buff = rhs.v_lambda_buff;
    a_lambda_buff = rhs.a_lambda_buff;
    m_lambda_buff = rhs.m_lambda_buff;
    N_buff = rhs.N_buff;

    rhs.lambdas  = nullptr;
    rhs.v_lambdas = nullptr;
    rhs.a_lambdas = nullptr;
    rhs.m_lambdas = nullptr;
    rhs.n_lambdas = 0;

    rhs.mass_lambda = 0.0;
    rhs.lambda_buff = 0.0;
    rhs.v_lambda_buff = 0.0;
    rhs.a_lambda_buff = 0.0;
    rhs.m_lambda_buff = 0;
    rhs.n_lambdas = 0;
    rhs.N_buff = 0;
  }
  return *this;
}


int constant_pH_state::reset_lambdas(const int& n_lambdas_, const std::unique_ptr<constant_pH_state>& prev_pH_state_) {

  if (n_lambdas != n_lambdas_) {
    deallocate_lambdas();
    n_lambdas = n_lambdas_;
    allocate_lambdas();
  }

  int to = 0;
  if (prev_pH_state_ && prev_pH_state_->molids) {
    int n_lambdas_prev = prev_pH_state_->n_lambdas;
    //molid to index map : find has O(1) runtime
    std::unordered_map<int,int> idx;
    idx.reserve(n_lambdas_prev);
    for (int k = 0; k < n_lambdas_prev; k++)
      idx[prev_pH_state_->molids[k]] = k;


    for (int i = 0; i < n_lambdas; i++) {
      auto iter = idx.find(molids[i]);
      if (iter != idx.end()) {
        int from = iter->second;
        for (int j = 0; j < 3; j++) {
          lambdas[to][j] = prev_pH_state_->lambdas[from][j];
          v_lambdas[to][j] = prev_pH_state_->v_lambdas[from][j];
          a_lambdas[to][j] = prev_pH_state_->a_lambdas[from][j];
          m_lambdas[to][j] = prev_pH_state_->m_lambdas[from][j];
        }
        to++;
      }
    }
  }


  for (int i = to; i < n_lambdas; i++) {
    for (int j = 0; j < 3; j++) {
      lambdas[i][j] = 0.0;
      v_lambdas[i][j] = 0.0;
      a_lambdas[i][j] = 0.0;
      m_lambdas[i][j] = mass_lambda; // m_lambda == 20.0u taken from https://www.mpinat.mpg.de/627830/usage
    }
  }

  return to;
}

void constant_pH_state::set_zero()
{
  if (n_lambdas) {
    std::fill_n(&lambdas[0][0],3*n_lambdas,0.0);
    std::fill_n(&v_lambdas[0][0],3*n_lambdas,0.0);
    std::fill_n(&a_lambdas[0][0],3*n_lambdas,0.0);
  } 
}

void constant_pH_state::broadcast()
{
  if (n_lambdas) {
    MPI_Bcast(lambdas[0], n_lambdas * 3, MPI_DOUBLE, 0, world);
    MPI_Bcast(v_lambdas[0], n_lambdas * 3, MPI_DOUBLE, 0, world);
    MPI_Bcast(a_lambdas[0], n_lambdas * 3, MPI_DOUBLE, 0, world);
    MPI_Bcast(m_lambdas[0], n_lambdas * 3, MPI_DOUBLE, 0, world);
  }
  MPI_Bcast(&lambda_buff, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&v_lambda_buff, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&a_lambda_buff, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&m_lambda_buff, 1, MPI_DOUBLE, 0, world);
}

void constant_pH_state::allocate_lambdas()
{
  if (n_lambdas) {
    memory->create(lambdas, n_lambdas, 3, "constant_pH:lambdas");
    memory->create(v_lambdas, n_lambdas, 3, "constant_pH:v_lambdas");
    memory->create(a_lambdas, n_lambdas, 3, "constant_pH:a_lambdas");
    memory->create(m_lambdas, n_lambdas, 3, "constant_pH:m_lambdas");

    molids = std::make_unique<int []>(n_lambdas);

    std::fill_n(&m_lambdas[0][0],3*n_lambdas,mass_lambda);
  }
}

void constant_pH_state::deallocate_lambdas()
{
  if (lambdas) memory->destroy(lambdas);
  if (v_lambdas) memory->destroy(v_lambdas);
  if (a_lambdas) memory->destroy(a_lambdas);
  if (m_lambdas) memory->destroy(m_lambdas);

  molids.reset();

  lambdas = nullptr;
  v_lambdas = nullptr;
  a_lambdas = nullptr;
  m_lambdas = nullptr;
}