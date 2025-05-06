#pragma once

#include <string>
#include <fstream>


using std::string, std::ifstream;

class constant_pH_structures
{
    friend class FixAdaptiveProtonation;
    friend class FixConstantPH;
public:
    constant_pH_structures(const string& fileName1,const string& fileName2);

    ~constant_pH_structures();
    constant_pH_structures(const constant_pH_structures& rhs) = delete;
    constant_pH_structures& operator=(const constant_pH_structures& rhs) = delete;
    constant_pH_structures(const constant_pH_structures&& rhs) = delete;
    constant_pH_structures& operator=(const constant_pH_structures&& rhs) = delete;

    void read_pH_structure_files();



private:
    ifstream pHStructureFile1, pHStructureFile2;


    /* These variables will be accessed by fix_adaptive_protonation and fix_constant_pH*/
    double **pH1qs, **pH2qs;
    std::unique_ptr<int []> typePerProtMol;
    std::unique_ptr<int []> protonable;
    int pHnStructures1, pHnStructures2;
    int pHnTypes1, pHnTypes2;
};