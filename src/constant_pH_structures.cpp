#include "constant_pH_structures.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"

#include <sstream>

using namespace LAMMPS_NS;

constant_pH_structures::constant_pH_structures(LAMMPS *lmp, const string &fileName1,
                                               const string &fileName2) :
    Pointers(lmp),
    pH1qs(nullptr), pH2qs(nullptr)
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
        stringstream iss;

        if (!getline(file, line))
          error->one(FLERR, "Error in reading the pH structure file in fix constant_pH");

        iss.str(line);

        string field;
        int type;
        getline(iss, field, ',');
        type = stoi(field);
        protonable[type+1] = 1;
        int type_per_prot_mol;
        getline(iss, field, ',');
        type_per_prot_mol = stoi(field);
        typePerProtMol[type+1] = type_per_prot_mol;

        double q;
        for (int j = 0; j < nStructures; j++) {
          getline(iss, field, ',');
          q = stod(field);
          pHqs[type+1][j] = q;
        }
      }
    }

    MPI_Bcast(protonable.get(), ntypes + 1, MPI_INT, 0, world);
    MPI_Bcast(typePerProtMol.get(), ntypes + 1, MPI_INT, 0, world);
    MPI_Bcast(pHqs[0], (ntypes + 1) * (nStructures), MPI_DOUBLE, 0, world);
  };

  parse_file(pHStructureFile1, pHnStructures1, pH1qs, pHnTypes1, "constant_pH:pH1qs");
  parse_file(pHStructureFile2, pHnStructures2, pH2qs, pHnTypes2, "constant_pH:pH2qs");
}