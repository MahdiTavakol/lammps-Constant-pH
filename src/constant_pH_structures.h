#pragma once

#include <string>
#include <fstream>


using std::string, std::ifstream;

class constant_pH_structures
{
public:
    constant_pH_structures(const string& fileName1,const string& fileName2, 
                          double**& pH1qs_, double**& pH2qs_,
                          int& pHnStructures1_, int& pHnStructures2_,
                          int& pHnTypes1_, int& pHnTypes2_, 
                          int*& typePerProtMol_, int*& protonable_);

    ~constant_pH_structures();
    constant_pH_structures(const constant_pH_structures& rhs) = delete;
    constant_pH_structures& operator=(const constant_pH_structures& rhs) = delete;
    constant_pH_structures(const constant_pH_structures&& rhs) = delete;
    constant_pH_structures& operator=(const constant_pH_structures&& rhs) = delete;

    void read_pH_structure_files();

private:
    double **&pH1qs, **&pH2qs;
    int *&typePerProtMol, *&protonable;
    int &pHnStructures1, &pHnStructures2;
    int &pHnTypes1, &pHnTypes2;
    ifstream pHStructureFile1, pHStructureFile2;
};