
#include "wnum.h"

#include <iostream>

using namespace std;

Atom::Atom(double _x, double _y, double _z, double _mass) : x(_x), y(_y), z(_z), mass(_mass), c(0.0) {}

Atom::Atom(double _x, double _y, double _z, double _mass, int _type, int _id, int _mid, double _q, int _ix, int _iy, int _iz, double _c4) : 
    x(_x), y(_y), z(_z), mass(_mass), id(_id), mid(_mid), q(_q), ix(_ix), iy(_iy), iz(_iz), type(_type), c(_c4) {}


Atom::Atom() {
    x = 0.0; 
    y = 0.0;
    z = 0.0;
    mass = 0.0;
    id=0;
    mid=0;
    q=0;
    ix=0;
    iy=0;
    iz=0;
    type=0;
    c = 0.0;
    }


Atom::Atom(const Atom& _atom) : x(_atom.x), y(_atom.y), z(_atom.z), mass(_atom.mass),
                              id(_atom.id), mid(_atom.mid), q(_atom.mid),
                              ix(_atom.ix), iy(_atom.iy), iz(_atom.iz), type(_atom.type), c(_atom.c) {}
Atom::~Atom() {}

void Atom::setAtom(double _x, double _y, double _z, double _mass, int _id, int _mol, double _q, double c_4)
{
    x = _x;
    y = _y;
    z = _z;
    mass = _mass;
    id = _id;
    mid = _mol;
    q = _q;
    c = c_4;
}

double Atom::getMass() { return mass; }

double Atom::distance(const Atom& _atom) const {
    return sqrt((this->x - _atom.x) * (this->x - _atom.x) +
                    (this->y - _atom.y) * (this->y - _atom.y) +
                    (this->z - _atom.z) * (this->z - _atom.z));
}

Data::Data():
    atoms(nullptr), bonds(nullptr), angles(nullptr), num_atoms(0),num_bonds(0),num_angles(0) {}
       
Data::Data(const Data& _data): atoms(_data.atoms), bonds(_data.bonds), angles(_data.angles), num_atoms(_data.num_atoms), num_bonds(_data.num_bonds), num_angles(_data.num_angles)
{
    atoms  = new Atom[num_atoms]();
    bonds  = new Bond[num_bonds]();
    angles = new Angle[num_angles]();
            
    copy(_data.atoms , _data.atoms + _data.num_atoms ,atoms );
    copy(_data.bonds , _data.bonds + _data.num_bonds ,bonds );
    copy(_data.angles, _data.angles+ _data.num_angles,angles);          
}

Data::~Data()
{
    delete [] atoms;
    delete [] bonds;
    delete [] angles;
    atoms = nullptr;
    bonds = nullptr;
    angles = nullptr;
    num_atoms = 0;
    num_bonds = 0;
    num_angles = 0;
}
        
// Assignment Operator
Data& Data::operator=(const Data& _data) {
    if (this != &_data) {
        delete[] atoms;
        delete[] bonds;
        delete[] angles;

        num_atoms = _data.num_atoms;
        num_bonds = _data.num_bonds;
        num_angles = _data.num_angles;

        atoms = new Atom[num_atoms];
        bonds = new Bond[num_bonds];
        angles = new Angle[num_angles];

        std::copy(_data.atoms, _data.atoms + num_atoms, atoms);
        std::copy(_data.bonds, _data.bonds + num_bonds, bonds);
        std::copy(_data.angles, _data.angles + num_angles, angles);
    }
    return *this;
}

void Data::modifyBondAngle()
{
    //Check for bonds and angles with invalid atom ids
    unordered_set<int> valid_ids;
    for (int i = 0; i < num_atoms; ++i) {
        int id;
        atoms[i].getid(id);
        valid_ids.insert(id);
    }

    //  Remove invalid bonds
    int valid_bond_count = 0;
    for (int i = 0; i < num_bonds; ++i) {
        if (valid_ids.find(bonds[i].a) != valid_ids.end() && valid_ids.find(bonds[i].b) != valid_ids.end()) {
            bonds[valid_bond_count++] = bonds[i];
        }
    }
    
    num_bonds = valid_bond_count;

    // Remove invalid angles
    int valid_angle_count = 0;
    for (int i = 0; i < num_angles; ++i) {
        if (valid_ids.find(angles[i].a) != valid_ids.end() &&
            valid_ids.find(angles[i].b) != valid_ids.end() &&
            valid_ids.find(angles[i].c) != valid_ids.end()) {
            angles[valid_angle_count++] = angles[i];
        }
    }
    
    num_angles = valid_angle_count;
            
    // Allocate new memory for valid bonds and angles
    Bond* new_bonds = new Bond[valid_bond_count];
    Angle* new_angles = new Angle[valid_angle_count];
    
    copy(bonds, bonds + valid_bond_count, new_bonds);
    copy(angles, angles + valid_angle_count, new_angles);
                 
    delete [] bonds;
    delete [] angles;
    bonds = new_bonds;
    angles = new_angles;
    num_bonds = valid_bond_count;
    num_angles = valid_angle_count; 
}


void Data::reID()
{
    sort(atoms, atoms + num_atoms, [](const Atom& a, const Atom& b){
        int aid;
        int bid;
        a.getid(aid);
        b.getid(bid);
        return aid < bid;});
            
    unordered_map<int, int> id_map;
    for (int i = 0; i < num_atoms; ++i) {
        int old_id;
        atoms[i].getid(old_id);
        int new_id = i + 1;
        atoms[i].setid(new_id);
        id_map[old_id] = new_id;
    }

    // Update the bond and angle arrays
    for (int i = 0; i < num_bonds; ++i) {
        bonds[i].a = id_map[bonds[i].a];
        bonds[i].b = id_map[bonds[i].b];
    }

    for (int i = 0; i < num_angles; ++i) {
        angles[i].a = id_map[angles[i].a];
        angles[i].b = id_map[angles[i].b];
        angles[i].c = id_map[angles[i].c];
    }
}

Trajectory::Trajectory(ifstream& input_stream, const int& num_frames)
{
    read_trj_file(input_stream,num_frames);
}

Trajectory::~Trajectory()
{
    deallocate_storage();
}

void Trajectory::allocate_storage(const int num_frames)
{
    this->num_frames = num_frames;
    datas = (Data*) malloc(num_frames * sizeof(Data));
    for (int i = 0; i < num_frames; i++)
    {
        new (&datas[i]) Data();
    }
}

void Trajectory::deallocate_storage()
{

    for (int i = 0; i < num_frames; i++)
        datas[i].~Data();

    free(datas);
    num_frames = 0;
}

void Trajectory::read_trj_file(ifstream& input, const int& num_frames)
{
    string line;

    deallocate_storage();

    this->num_frames = num_frames;
    allocate_storage(num_frames);

    stringstream iss;
    int current_frame = -1;
    while (getline(input,line)) {
        if (line.empty()) continue;

        iss.clear();
        iss.str("");

        iss << line;

        double bounds[3][2];

        if (line == "ITEM: TIMESTEP")
        {
            current_frame++;
            getline(input, line); // frame_number
        }
        else if (line == "ITEM: NUMBER OF ATOMS") 
        {
            int num_atoms;
            getline(input, line); // num_atoms
            iss.clear();
            iss.str("");
            iss << line;
            iss >> num_atoms;


            this->datas[current_frame].num_atoms = num_atoms;
            this->datas[current_frame].atoms = new Atom[num_atoms];

        }
        else if (line == "ITEM: BOX BOUNDS pp pp pp")
        {
            for (int i = 0; i < 3; i++) {
                getline(input,line);
                iss.clear();
                iss.str("");
                iss << line;
                iss >> bounds[i][0];
                iss >> bounds[i][1];
            }
        }
        else if (line == "ITEM: ATOMS mass id mol q type xs ys zs c_4")
        {
            int num_atoms = this->datas[current_frame].num_atoms;
            for (int i = 0; i < num_atoms; i++)
            {
                double mass, q, xs, ys, zs, c_4;
                int id, mol,type;

                getline(input,line);
                iss.clear();
                iss.str("");
                iss << line;

                iss >> mass >> id >> mol >> q >> type >> xs >> ys >> zs >> c_4;

                double x = bounds[0][0] + xs * (bounds[0][1] - bounds[0][0]);
                double y = bounds[1][0] + ys * (bounds[1][1] - bounds[1][0]);
                double z = bounds[2][0] + zs * (bounds[2][1] - bounds[2][0]);

                Atom atom(x, y, z, mass, type, id, mol, q, x, y, z, c_4);
                datas[current_frame].atoms[i] = atom;

            }
        }
    }
}


void Trajectory::calculate_c_distribution(double*& zs, double*& cs, int& num_data)
{
    num_data = 0;

    for (int i = 0; i < num_frames; i++)
        num_data  += datas[i].num_atoms;


    zs = (double *) malloc(num_data * sizeof(double));
    cs = (double* ) malloc(num_data * sizeof(double));

    int loc = 0;
    for (int i = 0; i < num_frames; i++)
    {
        int num_atoms = datas[i].num_atoms;
        for (int j = 0; j < num_atoms; j++)
        {
            double x, y, z, c;
            datas[i].atoms[j].getCoord(x,y,z);
            datas[i].atoms[j].getc(c);
            zs[loc] = z;
            cs[loc] = c;
            loc++;
        }

    }
    
}

