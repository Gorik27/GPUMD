/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
The semi-grand canonical (SGC) and variance-constrained SGC (VCSGC) ensembles
for MCMD.

[1] B. Sadigh, P. Erhart, A. Stukowski, A. Caro, E. Martinez, and L. Zepeda-Ruiz
Scalable parallel Monte Carlo algorithm for atomistic simulations of
precipitation in alloys, Phys. Rev. B 85, 184203 (2012).

[2] B. Sadigh and P. Erhart
Calculations of excess free energies of precipitates via direct thermodynamic
integration across phase boundaries, Phys. Rev. B 86, 134204 (2012).
------------------------------------------------------------------------------*/

#include "mc_ensemble_sgc.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include <map>
#include <cstring>

const std::map<std::string, double> MASS_TABLE{
  {"H", 1.0080000000},
  {"He", 4.0026020000},
  {"Li", 6.9400000000},
  {"Be", 9.0121831000},
  {"B", 10.8100000000},
  {"C", 12.0110000000},
  {"N", 14.0070000000},
  {"O", 15.9990000000},
  {"F", 18.9984031630},
  {"Ne", 20.1797000000},
  {"Na", 22.9897692800},
  {"Mg", 24.3050000000},
  {"Al", 26.9815385000},
  {"Si", 28.0850000000},
  {"P", 30.9737619980},
  {"S", 32.0600000000},
  {"Cl", 35.4500000000},
  {"Ar", 39.9480000000},
  {"K", 39.0983000000},
  {"Ca", 40.0780000000},
  {"Sc", 44.9559080000},
  {"Ti", 47.8670000000},
  {"V", 50.9415000000},
  {"Cr", 51.9961000000},
  {"Mn", 54.9380440000},
  {"Fe", 55.8450000000},
  {"Co", 58.9331940000},
  {"Ni", 58.6934000000},
  {"Cu", 63.5460000000},
  {"Zn", 65.3800000000},
  {"Ga", 69.7230000000},
  {"Ge", 72.6300000000},
  {"As", 74.9215950000},
  {"Se", 78.9710000000},
  {"Br", 79.9040000000},
  {"Kr", 83.7980000000},
  {"Rb", 85.4678000000},
  {"Sr", 87.6200000000},
  {"Y", 88.9058400000},
  {"Zr", 91.2240000000},
  {"Nb", 92.9063700000},
  {"Mo", 95.9500000000},
  {"Tc", 98},
  {"Ru", 101.0700000000},
  {"Rh", 102.9055000000},
  {"Pd", 106.4200000000},
  {"Ag", 107.8682000000},
  {"Cd", 112.4140000000},
  {"In", 114.8180000000},
  {"Sn", 118.7100000000},
  {"Sb", 121.7600000000},
  {"Te", 127.6000000000},
  {"I", 126.9044700000},
  {"Xe", 131.2930000000},
  {"Cs", 132.9054519600},
  {"Ba", 137.3270000000},
  {"La", 138.9054700000},
  {"Ce", 140.1160000000},
  {"Pr", 140.9076600000},
  {"Nd", 144.2420000000},
  {"Pm", 145},
  {"Sm", 150.3600000000},
  {"Eu", 151.9640000000},
  {"Gd", 157.2500000000},
  {"Tb", 158.9253500000},
  {"Dy", 162.5000000000},
  {"Ho", 164.9303300000},
  {"Er", 167.2590000000},
  {"Tm", 168.9342200000},
  {"Yb", 173.0450000000},
  {"Lu", 174.9668000000},
  {"Hf", 178.4900000000},
  {"Ta", 180.9478800000},
  {"W", 183.8400000000},
  {"Re", 186.2070000000},
  {"Os", 190.2300000000},
  {"Ir", 192.2170000000},
  {"Pt", 195.0840000000},
  {"Au", 196.9665690000},
  {"Hg", 200.5920000000},
  {"Tl", 204.3800000000},
  {"Pb", 207.2000000000},
  {"Bi", 208.9804000000},
  {"Po", 210},
  {"At", 210},
  {"Rn", 222},
  {"Fr", 223},
  {"Ra", 226},
  {"Ac", 227},
  {"Th", 232.0377000000},
  {"Pa", 231.0358800000},
  {"U", 238.0289100000},
  {"Np", 237},
  {"Pu", 244},
  {"Am", 243},
  {"Cm", 247},
  {"Bk", 247},
  {"Cf", 251},
  {"Es", 252},
  {"Fm", 257},
  {"Md", 258},
  {"No", 259},
  {"Lr", 262}};

MC_Ensemble_SGC::MC_Ensemble_SGC(
  const int num_atoms,
  const char** param,
  int num_param,
  int num_steps_mc_input,
  bool is_vcsgc_input,
  std::vector<std::string>& species_input,
  std::vector<int>& types_input,
  std::vector<int>& num_atoms_species_input,
  std::vector<double>& mu_or_phi_input,
  double kappa_input)
  : MC_Ensemble(param, num_param, num_atoms)
{
  num_steps_mc = num_steps_mc_input;
  is_vcsgc = is_vcsgc_input;
  species = species_input;
  types = types_input;
  num_atoms_species = num_atoms_species_input;
  mu_or_phi = mu_or_phi_input;
  kappa = kappa_input;
  NN_ij.resize(1);
  NL_ij.resize(300);
  pe_before_local.resize(300);
  delta_pe.resize(300);
  NN_angular_i.resize(1);
}

MC_Ensemble_SGC::~MC_Ensemble_SGC(void) { mc_output.close(); }

static __global__ void get_neighbors_of_i(
  const int N,
  const Box box,
  const int i,
  const float rc_radial_square,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_i,
  int* g_NL_i,
  float* g_pe_before,
  float* g_pe_before_local)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N && n != i) {
    double x0 = g_x[n];
    double y0 = g_y[n];
    double z0 = g_z[n];
    double x0i = g_x[i] - x0;
    double y0i = g_y[i] - y0;
    double z0i = g_z[i] - z0;

    apply_mic(box, x0i, y0i, z0i);
    float distance_square_i = float(x0i * x0i + y0i * y0i + z0i * z0i);

    if (distance_square_i < rc_radial_square) {
      int index = atomicAdd(g_NN_i, 1);
      g_pe_before_local[index] = g_pe_before[n]; 
      g_NL_i[index] = n;
    }
  }
}

static __global__ void create_inputs_for_energy_calculator(
  NEP_Energy::ParaMB paramb,
  const int N_local,
  const int i,
  const int* atom_local,
  const Box box,
  const float rc_radial_square,
  const float rc_angular_square,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* g_type,
  int* g_NN_angular,
  int* g_t2_radial,
  float* g_x12_radial,
  float* g_y12_radial,
  float* g_z12_radial,
  bool* g_is_neigh_angular,
  float* g_q_radial,
  float* g_s_angular,
  float* g_q_radial_local,
  float* g_s_angular_local)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x; // neighbors of the swapped atom i
  if (k<N_local) {
    int n1 = atom_local[k];
    double x2 = g_x[n1];
    double y2 = g_y[n1];
    double z2 = g_z[n1];
    double x12 = x2 - g_x[i];
    double y12 = y2 - g_y[i];
    double z12 = z2 - g_z[i];
    apply_mic(box, x12, y12, z12);
    float distance_square = float(x12 * x12 + y12 * y12 + z12 * z12);
    if (distance_square < rc_radial_square) {
      g_t2_radial[k] = g_type[n1];
      g_x12_radial[k] = float(x12);
      g_y12_radial[k] = float(y12);
      g_z12_radial[k] = float(z12);
      
      for (int n = 0; n <= paramb.n_max_radial; ++n){
        int index, index_local;
        index_local = k*(paramb.n_max_radial+1) + n;
        index = n1*(paramb.n_max_radial+1) + n;
        g_q_radial_local[index_local] = g_q_radial[index];
      }
      
      for (int n = 0; n <= paramb.n_max_angular; ++n){
        for (int l = 0; l<NUM_OF_ABC; ++l){
          int index, index_local;
          index_local = k*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
          index =      n1*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
          g_s_angular_local[index_local] = g_s_angular[index];
        }
      }
      if (distance_square < rc_angular_square) {
        //atomicAdd(g_NN_angular, 1);
        g_is_neigh_angular[k] = true;
      }
      else {
        g_is_neigh_angular[k] = false;
      }
    }
  }
  else if (k = N_local){// central (i) atom
    for (int n = 0; n <= paramb.n_max_radial; ++n){
        int index, index_local;
        index_local = k*(paramb.n_max_radial+1) + n;
        index = i*(paramb.n_max_radial+1) + n;
        g_q_radial_local[index_local] = g_q_radial[index];
      }

    for (int n = 0; n <= paramb.n_max_angular; ++n){
        for (int l = 0; l<NUM_OF_ABC; ++l){
          int index, index_local;
          index_local = k*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
          index =      i*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
          g_s_angular_local[index_local] = g_s_angular[index];
        }
      }
  }
}


// a kernel with a single thread <<<1, 1>>>
static __global__ void gpu_flip(
  const int i,
  const int type_j,
  const double mass_j,
  const double mass_scaler,
  int* g_type,
  double* g_mass,
  double* g_vx,
  double* g_vy,
  double* g_vz)
{
  g_type[i] = type_j;
  g_mass[i] = mass_j;
  g_vx[i] *= mass_scaler; // momentum conservation
  g_vy[i] *= mass_scaler;
  g_vz[i] *= mass_scaler;
}

bool MC_Ensemble_SGC::allowed_species(std::string& species_found)
{
  for (int k = 0; k < species.size(); ++k) {
    if (species[k] == species_found) {
      index_old_species = k;
      return true;
    }
  }
  return false;
}

void MC_Ensemble_SGC::compute(
  int md_step,
  double temperature,
  Atom& atom,
  Box& box,
  std::vector<Group>& groups,
  int grouping_method,
  int group_id)
{
  if (check_if_small_box(nep_energy.paramb.rc_radial, box)) {
    printf("Cannot use small box for MCMD.\n");
    exit(1);
  }

  int group_size =
    grouping_method >= 0 ? groups[grouping_method].cpu_size[group_id] : atom.number_of_atoms;
  std::uniform_int_distribution<int> r1(0, group_size - 1);

  nep_energy.compute_large_box(
    box, 
    atom.type, 
    atom.position_per_atom);

  int num_accepted = 0;
  //mc_output << "MC step" << std::endl; // ***todo*** debug
  for (int step = 0; step < num_steps_mc; ++step) {
    int i = -1;
    int type_i = -1;
    std::string species_found;
    while (!allowed_species(species_found)) {
      i = grouping_method >= 0
            ? groups[grouping_method]
                .cpu_contents[groups[grouping_method].cpu_size_sum[group_id] + r1(rng)]
            : r1(rng);
      species_found = atom.cpu_atom_symbol[i];
      type_i = atom.cpu_type[i];
    }

    int type_j = type_i;
    std::uniform_int_distribution<int> rand_int2(0, types.size() - 1);
    while (type_j == type_i) {
      index_new_species = rand_int2(rng);
      type_j = types[index_new_species];
    }

    NN_ij.fill(0);
    pe_before_local.fill(0.0f);

    get_neighbors_of_i<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      box,
      i,
      nep_energy.paramb.rc_radial * nep_energy.paramb.rc_radial,
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + atom.number_of_atoms,
      atom.position_per_atom.data() + atom.number_of_atoms * 2,
      NN_ij.data(),
      NL_ij.data(),
      nep_energy.nep_data.pe.data(),
      pe_before_local.data());
    GPU_CHECK_KERNEL

    int NN_ij_cpu;
    NN_ij.copy_to_host(&NN_ij_cpu);

    gpuMemcpy(&pe_before_local.data()[NN_ij_cpu], &nep_energy.nep_data.pe.data()[i], sizeof(float), gpuMemcpyDeviceToDevice); 

    NN_angular_i.fill(0);
    nep_energy.nep_data.q_radial_local.fill(0.0f);
    nep_energy.nep_data.s_angular_local.fill(0.0f);
    nep_energy.nep_data.q_radial_i.fill(0.0f);
    nep_energy.nep_data.s_angular_i.fill(0.0f);
    is_neigh_angular.fill(false);

    create_inputs_for_energy_calculator<<<((NN_ij_cpu + 1) - 1) / 64 + 1, 64>>>(// (NN_ij_cpu + 1) due to the fact that array[N_ij_cpu] contain information about central (i) atom
      nep_energy.paramb,
      NN_ij_cpu,
      i,
      NL_ij.data(),
      box,
      nep_energy.paramb.rc_radial * nep_energy.paramb.rc_radial,
      nep_energy.paramb.rc_angular * nep_energy.paramb.rc_angular,
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + atom.number_of_atoms, 
      atom.position_per_atom.data() + atom.number_of_atoms * 2, 
      atom.type.data(),
      NN_angular_i.data(),
      t2_radial.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      is_neigh_angular.data(),
      nep_energy.nep_data.q_radial.data(),
      nep_energy.nep_data.s_angular.data(),
      nep_energy.nep_data.q_radial_local.data(),
      nep_energy.nep_data.s_angular_local.data());
    GPU_CHECK_KERNEL
    
    nep_energy.nep_data.q_radial_trial_local.fill(0.0f);
    nep_energy.nep_data.s_angular_trial_local.fill(0.0f);

    nep_energy.find_energy(
      NN_ij_cpu,
      i,
      NN_angular_i.data(),
      type_i,
      type_j,
      t2_radial.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      is_neigh_angular.data(),
      delta_pe.data(),
      pe_before_local.data());
    
    std::vector<float> delta_pe_cpu(NN_ij_cpu+1);
    delta_pe.copy_to_host(delta_pe_cpu.data(), NN_ij_cpu+1);

    float energy_difference = 0.0f;
    for (int n = 0; n < NN_ij_cpu; ++n) {
      energy_difference += delta_pe_cpu[n];
    }
    energy_difference += delta_pe_cpu[NN_ij_cpu]; // delta energy of the central (i) atom
    //mc_output << i << "; " << type_i << "; " << type_j << "; " << energy_difference << std::endl;

    if (!is_vcsgc) {
      energy_difference += mu_or_phi[index_new_species] - mu_or_phi[index_old_species];
    } else {
      energy_difference +=
        kappa * K_B * temperature / atom.number_of_atoms *
        (atom.number_of_atoms * (mu_or_phi[index_new_species] - mu_or_phi[index_old_species]) +
         2 * (num_atoms_species[index_new_species] - num_atoms_species[index_old_species]) + 1.0);
    }

    std::uniform_real_distribution<float> r2(0, 1);
    float random_number = r2(rng);
    float probability = exp(-energy_difference / (K_B * temperature));
    //mc_output << "prob " << probability << std::endl;
    if (random_number < probability) {
      ++num_accepted;
      //mc_output << "acc" << std::endl;
      ++num_atoms_species[index_new_species];
      --num_atoms_species[index_old_species];

      atom.cpu_type[i] = type_j;
      atom.cpu_atom_symbol[i] = species[index_new_species];
      double mass_old = atom.cpu_mass[i];
      double mass_new = MASS_TABLE.at(species[index_new_species]);
      atom.cpu_mass[i] = mass_new;

      gpu_flip<<<1, 1>>>(
        i,
        type_j,
        mass_new,
        mass_old / mass_new,
        atom.type.data(),
        atom.mass.data(),
        atom.velocity_per_atom.data(),
        atom.velocity_per_atom.data() + atom.number_of_atoms,
        atom.velocity_per_atom.data() + atom.number_of_atoms * 2);

      nep_energy.accept_trial(
        NN_ij_cpu, 
        NL_ij.data(),
        delta_pe.data(),
        i);
    }
  }

  mc_output << md_step << "  " << num_accepted / double(num_steps_mc) << " ";
  for (int t = 0; t < types.size(); ++t) {
    mc_output << num_atoms_species[t] / double(atom.number_of_atoms) << " ";
  }
  mc_output << std::endl;
}
