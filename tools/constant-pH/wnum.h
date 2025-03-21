#ifndef WNUM_H
#define WNUM_H

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_set>
#include <unordered_map>

using namespace std;

class Atom
{
    public:
        Atom(double _x, double _y, double _z, double _mass);
        Atom(double _x, double _y, double _z, double _mass, int _type, int _id, int _mid, double _q, int _ix, int _iy, int _iz, double _c4);
        Atom();
        Atom(const Atom& _atom);
        ~Atom();
        void setAtom(double _x, double _y, double _z, double _mass, int _id, int _mol, double _q, double c_4);
        double getMass();
        double distance(const Atom& _atom) const;
        void getCoord(double& _x, double& _y, double& _z)
        {
            _x = x; _y = y; _z = z;
        }
        void getc(double & _c) {_c = c;}
        void getid(int &_id) const
        {
            _id = id;
        }
        void setid(const int _id) {id = _id;}
    protected:
        double x, y, z, mass, q, type;
        int id, mid, ix, iy, iz;
        double c;
};

typedef struct { double x, y, z; } Coor;
typedef struct { int a, b, type; } Bond;
typedef struct { int a, b, c, type;} Angle;

class Data {
    public:
        Data();
        Data(const Data& _data);
        ~Data();
        
        // Assignment Operator
        Data& operator=(const Data& _data);
        void modifyBondAngle();
        void reID();

        Atom *atoms;
        Bond *bonds;
        Angle *angles;
	    int num_atoms, num_bonds, num_angles;
	    vector<string> headers;
	};

class Trajectory {
    public:
        Trajectory() {num_frames = 0;}
        Trajectory(ifstream& input_stream, const int& num_frames);
        ~Trajectory();
        void allocate_storage(const int num_frames);
        void deallocate_storage();
        void read_trj_file(ifstream& input_stream, const int& num_frames);
        void calculate_c_distribution(double *& zs, double *& cs, int& num_data);

    protected:
        Data* datas;
        int num_frames;
};

#endif#ifndef WNUM_H
#define WNUM_H

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_set>
#include <unordered_map>

using namespace std;

class Atom
{
    public:
        Atom(double _x, double _y, double _z, double _mass);
        Atom(double _x, double _y, double _z, double _mass, int _type, int _id, int _mid, double _q, int _ix, int _iy, int _iz, double _c4);
        Atom();
        Atom(const Atom& _atom);
        ~Atom();
        void setAtom(double _x, double _y, double _z, double _mass, int _id, int _mol, double _q, double c_4);
        double getMass();
        double distance(const Atom& _atom) const;
        void getCoord(double& _x, double& _y, double& _z)
        {
            _x = x; _y = y; _z = z;
        }
        void getc(double & _c) {_c = c;}
        void getid(int &_id) const
        {
            _id = id;
        }
        void setid(const int _id) {id = _id;}
    protected:
        double x, y, z, mass, q, type;
        int id, mid, ix, iy, iz;
        double c;
};

typedef struct { double x, y, z; } Coor;
typedef struct { int a, b, type; } Bond;
typedef struct { int a, b, c, type;} Angle;

class Data {
    public:
        Data();
        Data(const Data& _data);
        ~Data();
        
        // Assignment Operator
        Data& operator=(const Data& _data);
        void modifyBondAngle();
        void reID();

        Atom *atoms;
        Bond *bonds;
        Angle *angles;
	    int num_atoms, num_bonds, num_angles;
	    vector<string> headers;
	};

class Trajectory {
    public:
        Trajectory() {num_frames = 0;}
        Trajectory(ifstream& input_stream, const int& num_frames);
        ~Trajectory();
        void allocate_storage(const int num_frames);
        void deallocate_storage();
        void read_trj_file(ifstream& input_stream, const int& num_frames);
        void calculate_c_distribution(double *& zs, double *& cs, int& num_data);

    protected:
        Data* datas;
        int num_frames;
};

#endif
