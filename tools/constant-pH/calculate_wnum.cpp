#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "wnum.h"

using namespace std;

void write_c_distribution(const string output, double *zs, double *cs, const int num_data);

int main()
{
    ifstream input_stream("dump.npt.lammpstrj");
    string output("zs-cs.csv");
    int num_frames = 200;
    int num_data;

    double * cs, *zs;

    Trajectory trj(input_stream,num_frames);

    input_stream.close();
    trj.calculate_c_distribution(zs,cs, num_data);
    write_c_distribution(output, zs, cs, num_data);

    free(cs);
    free(zs);

    return 0;
}

void write_c_distribution(const string output, double *zs, double *cs, const int num_data)
{
    ofstream output_stream(output);


    output_stream << "zs,cs" << std::endl;

    for (int i = 0; i < num_data; i++)
        output_stream << zs[i] << "," << cs[i] << std::endl;

    output_stream.close();
}

	
