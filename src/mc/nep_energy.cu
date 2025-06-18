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
The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "nep_energy.cuh"
#include "force/neighbor.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

void NEP_Energy::initialize(const char* file_potential, const int num_atoms)
{

  std::ifstream input(file_potential);
  if (!input.is_open()) {
    std::cout << "Failed to open " << file_potential << std::endl;
    exit(1);
  }

  // nep3 1 C
  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
    exit(1);
  }
  if (tokens[0] == "nep3") {
    paramb.version = 3;
    zbl.enabled = false;
  } else if (tokens[0] == "nep3_zbl") {
    paramb.version = 3;
    zbl.enabled = true;
  } else if (tokens[0] == "nep4") {
    paramb.version = 4;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_zbl") {
    paramb.version = 4;
    zbl.enabled = true;
  } else if (tokens[0] == "nep5") {
    paramb.version = 5;
    zbl.enabled = false;
  } else if (tokens[0] == "nep5_zbl") {
    paramb.version = 5;
    zbl.enabled = true;
  } else {
    std::cout << tokens[0]
              << " is an unsupported NEP model. We only support NEP3 and NEP4 models now."
              << std::endl;
    exit(1);
  }
  paramb.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != 2 + paramb.num_types) {
    std::cout << "The first line of nep.txt should have " << paramb.num_types << " atom symbols."
              << std::endl;
    exit(1);
  }

  if (paramb.num_types == 1) {
    printf("    Use the NEP%d potential with %d atom type.\n", paramb.version, paramb.num_types);
  } else {
    printf("    Use the NEP%d potential with %d atom types.\n", paramb.version, paramb.num_types);
  }

  for (int n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    zbl.atomic_numbers[n] = atomic_number;
    paramb.atomic_numbers[n] = atomic_number - 1;
    printf("        type %d (%s with Z = %d).\n", n, tokens[2 + n].c_str(), zbl.atomic_numbers[n]);
  }

  // zbl 0.7 1.4
  if (zbl.enabled) {
    tokens = get_tokens(input);
    if (tokens.size() != 3) {
      std::cout << "This line should be zbl rc_inner rc_outer." << std::endl;
      exit(1);
    }
    zbl.rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0 && zbl.rc_outer == 0) {
      zbl.flexibled = true;
      printf("        has the flexible ZBL potential\n");
    } else {
      printf(
        "        has the universal ZBL with inner cutoff %g A and outer cutoff %g A.\n",
        zbl.rc_inner,
        zbl.rc_outer);
    }
  }

  // cutoff 4.2 3.7 80 47
  tokens = get_tokens(input);
  if (tokens.size() != 5 && tokens.size() != 8) {
    std::cout << "This line should be cutoff rc_radial rc_angular MN_radial MN_angular "
                 "[radial_factor] [angular_factor] [zbl_factor].\n";
    exit(1);
  }
  paramb.rc_radial = get_double_from_token(tokens[1], __FILE__, __LINE__);
  paramb.rc_angular = get_double_from_token(tokens[2], __FILE__, __LINE__);
  printf("        radial cutoff = %g A.\n", paramb.rc_radial);
  printf("        angular cutoff = %g A.\n", paramb.rc_angular);

  int MN_radial = get_int_from_token(tokens[3], __FILE__, __LINE__);
  int MN_angular = get_int_from_token(tokens[4], __FILE__, __LINE__);
  printf("        MN_radial = %d.\n", MN_radial);
  printf("        MN_angular = %d.\n", MN_angular);
  paramb.MN_radial = int(ceil(MN_radial * 1.25));
  paramb.MN_angular = int(ceil(MN_angular * 1.25));
  printf("        enlarged MN_radial = %d.\n", paramb.MN_radial);
  printf("        enlarged MN_angular = %d.\n", paramb.MN_angular);

  if (tokens.size() == 8) {
    paramb.typewise_cutoff_radial_factor = get_double_from_token(tokens[5], __FILE__, __LINE__);
    paramb.typewise_cutoff_angular_factor = get_double_from_token(tokens[6], __FILE__, __LINE__);
    paramb.typewise_cutoff_zbl_factor = get_double_from_token(tokens[7], __FILE__, __LINE__);
    if (paramb.typewise_cutoff_radial_factor > 0.0f) {
      paramb.use_typewise_cutoff = true;
    }
    if (paramb.typewise_cutoff_zbl_factor > 0.0f) {
      paramb.use_typewise_cutoff_zbl = true;
    }
  }

  // n_max 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be n_max n_max_radial n_max_angular." << std::endl;
    exit(1);
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  printf("        n_max_radial = %d.\n", paramb.n_max_radial);
  printf("        n_max_angular = %d.\n", paramb.n_max_angular);

  // basis_size 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be basis_size basis_size_radial basis_size_angular."
              << std::endl;
    exit(1);
  }
  paramb.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  printf("        basis_size_radial = %d.\n", paramb.basis_size_radial);
  printf("        basis_size_angular = %d.\n", paramb.basis_size_angular);

  // l_max
  tokens = get_tokens(input);
  if (tokens.size() != 4) {
    std::cout << "This line should be l_max l_max_3body l_max_4body l_max_5body." << std::endl;
    exit(1);
  }

  paramb.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  printf("        l_max_3body = %d.\n", paramb.L_max);
  paramb.num_L = paramb.L_max;

  int L_max_4body = get_int_from_token(tokens[2], __FILE__, __LINE__);
  int L_max_5body = get_int_from_token(tokens[3], __FILE__, __LINE__);
  printf("        l_max_4body = %d.\n", L_max_4body);
  printf("        l_max_5body = %d.\n", L_max_5body);
  if (L_max_4body == 2) {
    paramb.num_L += 1;
  }
  if (L_max_5body == 1) {
    paramb.num_L += 1;
  }

  paramb.dim_angular = (paramb.n_max_angular + 1) * paramb.num_L;

  // ANN
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be ANN num_neurons 0." << std::endl;
    exit(1);
  }
  annmb.num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
  annmb.dim = (paramb.n_max_radial + 1) + paramb.dim_angular;
  printf("        ANN = %d-%d-1.\n", annmb.dim, annmb.num_neurons1);

  // calculated parameters:
  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  paramb.num_types_sq = paramb.num_types * paramb.num_types;

  if (paramb.version == 3) {
    annmb.num_para = (annmb.dim + 2) * annmb.num_neurons1 + 1;
  } else if (paramb.version == 4) {
    annmb.num_para = (annmb.dim + 2) * annmb.num_neurons1 * paramb.num_types + 1;
  } else {
    annmb.num_para = ((annmb.dim + 2) * annmb.num_neurons1 + 1) * paramb.num_types + 1;
  }

  printf("        number of neural network parameters = %d.\n", annmb.num_para);
  int num_para_descriptor =
    paramb.num_types_sq * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                           (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  printf("        number of descriptor parameters = %d.\n", num_para_descriptor);
  annmb.num_para += num_para_descriptor;
  printf("        total number of parameters = %d.\n", annmb.num_para);

  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  // NN and descriptor parameters
  std::vector<float> parameters(annmb.num_para);
  for (int n = 0; n < annmb.num_para; ++n) {
    tokens = get_tokens(input);
    parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
  nep_parameters.resize(annmb.num_para);
  nep_parameters.copy_from_host(parameters.data());
  update_potential(nep_parameters.data(), annmb);
  for (int d = 0; d < annmb.dim; ++d) {
    tokens = get_tokens(input);
    paramb.q_scaler[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }

  // flexible zbl potential parameters
  if (zbl.flexibled) {
    int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      tokens = get_tokens(input);
      zbl.para[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    zbl.num_types = paramb.num_types;
  }
  
  nep_data.NN_radial.resize(num_atoms);
  nep_data.NL_radial.resize(num_atoms * paramb.MN_radial);
  nep_data.NN_angular.resize(num_atoms);
  nep_data.NL_angular.resize(num_atoms * paramb.MN_angular);
  nep_data.cell_count.resize(num_atoms);
  nep_data.cell_count_sum.resize(num_atoms);
  nep_data.cell_contents.resize(num_atoms);
  nep_data.cpu_NN_radial.resize(num_atoms);
  nep_data.cpu_NN_angular.resize(num_atoms);
  nep_data.q_radial.resize(num_atoms * (paramb.n_max_radial + 1));
  nep_data.s_angular.resize(num_atoms * (paramb.n_max_angular + 1) * NUM_OF_ABC);
  nep_data.q_radial_local_size = 300 * (paramb.n_max_radial + 1);
  nep_data.s_angular_local_size = 300 * (paramb.n_max_angular + 1) * NUM_OF_ABC;
  nep_data.q_radial_local.resize(nep_data.q_radial_local_size);
  nep_data.s_angular_local.resize(nep_data.s_angular_local_size);
  nep_data.q_radial_i.resize(nep_data.q_radial_local_size);
  nep_data.s_angular_i.resize(nep_data.s_angular_local_size);
  nep_data.q_radial_trial_local.resize(nep_data.q_radial_local_size);
  nep_data.s_angular_trial_local.resize(nep_data.s_angular_local_size);
  nep_data.pe.resize(num_atoms);
}

NEP_Energy::NEP_Energy(void)
{
  // nothing
}

NEP_Energy::~NEP_Energy(void)
{
  // nothing
}

void NEP_Energy::update_potential(float* parameters, ANN& ann)
{
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    if (t > 0 && paramb.version == 3) { // Use the same set of NN parameters for NEP3
      pointer -= (ann.dim + 2) * ann.num_neurons1;
    }
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1;
    if (paramb.version == 5) {
      pointer += 1; // one extra bias for NEP5 stored in ann.w1[t]
    }
  }
  ann.b1 = pointer;
  ann.c = ann.b1 + 1;
}

static __global__ void find_energy_nep(
  NEP_Energy::ParaMB paramb,
  NEP_Energy::ANN annmb,
  const int N,
  const int i,
  const int t1_before,
  const int t1_after,
  const int* __restrict__ g_t2_radial,
  const float* __restrict__ g_x12_radial,
  const float* __restrict__ g_y12_radial,
  const float* __restrict__ g_z12_radial,
  const bool* __restrict__ g_is_neigh_angular,
  float* g_delta_pe,
  float* g_pe,
  float* g_q_radial,
  float* g_s_angular,
  float* g_q_radial_i,
  float* g_s_angular_i,
  float* g_q_radial_trial,
  float* g_s_angular_trial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    float q[MAX_DIM] = {0.0f};

    // get radial descriptors
    float r12[3] = {g_x12_radial[n1], g_y12_radial[n1], g_z12_radial[n1]};
    float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    float fc12;
    int t2 = g_t2_radial[n1];
    double rc = paramb.rc_radial;
    double rcinv = paramb.rcinv_radial;
    if (paramb.use_typewise_cutoff) {
      printf("typewise cutoff unsupported in MC now");
    }
    find_fc(rc, rcinv, d12, fc12);

    float fn12[MAX_NUM_N];
    find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      float dgn12_n1 = 0.0f;
      float dgn12_i = 0.0f;
      for (int k = 0; k <= paramb.basis_size_radial; ++k) {
        int c_index_base = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
        int c_index_before_n1 = c_index_base + t2 * paramb.num_types + t1_before;
        int c_index_after_n1  = c_index_base + t2 * paramb.num_types + t1_after;
        int c_index_before_i = c_index_base + t1_before * paramb.num_types + t2;
        int c_index_after_i  = c_index_base + t1_after * paramb.num_types + t2;
        dgn12_n1 += fn12[k] * (annmb.c[c_index_after_n1]-annmb.c[c_index_before_n1]);
        dgn12_i += fn12[k] * (annmb.c[c_index_after_i]-annmb.c[c_index_before_i]);
      }
      int index_n1 = n1*(paramb.n_max_radial+1) + n;
      q[n] = g_q_radial[index_n1] + dgn12_n1;
      g_q_radial_trial[index_n1] = q[n];//save trial to global memory (on GPU)

      //int index_i = n1*(paramb.n_max_radial+1) + n;
      //g_q_radial_i[index_i] = dgn12_i;//save impact of n1 atom to the central (i) atom's descriptor
      int index_i = N*(paramb.n_max_radial+1) + n;
      atomicAdd(&g_q_radial_trial[index_i], dgn12_i);
    }
    
    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f}; 
      for (int l = 0; l<NUM_OF_ABC; ++l){
          int index_local = n1*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
          s[l] = g_s_angular[index_local];
      }
      if (g_is_neigh_angular[n1]){// if n1 is angular neighbor of i
        float r12[3] = {g_x12_radial[n1], g_y12_radial[n1], g_z12_radial[n1]};
        float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
        float fc12;
        double rc = paramb.rc_angular;
        double rcinv = paramb.rcinv_angular;
        if (paramb.use_typewise_cutoff) {
          printf("typewise cutoff unsupported in MC now");
        }
        find_fc(rc, rcinv, d12, fc12);

        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float dgn12_n1 = 0.0f;
        float dgn12_i = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int base_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          int c_index_before_n1 = base_index + t2 * paramb.num_types + t1_before + paramb.num_c_radial;
          int c_index_after_n1  = base_index + t2 * paramb.num_types + t1_after + paramb.num_c_radial;
          
          int c_index_before_i = base_index + t1_before * paramb.num_types + t2 + paramb.num_c_radial;
          int c_index_after_i  = base_index + t1_after * paramb.num_types + t2 + paramb.num_c_radial;

          dgn12_n1 += fn12[k] * (annmb.c[c_index_after_n1] - annmb.c[c_index_before_n1]);
          dgn12_i += fn12[k] * (annmb.c[c_index_after_i] - annmb.c[c_index_before_i]);
        }
        float delta_s_n1[NUM_OF_ABC] = {0.0f};
        accumulate_s(paramb.L_max, d12, -r12[0], -r12[1], -r12[2], dgn12_n1, delta_s_n1);
        for (int l = 0; l<NUM_OF_ABC; ++l){
          s[l] = s[l] + delta_s_n1[l];
          int index_local = n1*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
          g_s_angular_trial[index_local] = s[l];//save trial to global memory (on GPU)
        }

        float delta_s_i[NUM_OF_ABC] = {0.0f};
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], dgn12_i, delta_s_i);
        for (int l = 0; l<NUM_OF_ABC; ++l){
          //int index_local = n1*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
          //g_s_angular[index_local] = delta_s_i[l];//save trial to global memory (on GPU)
          int index_local = N*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
          atomicAdd(&g_s_angular_trial[index_local], delta_s_i[l]);//save trial to global memory (on GPU)
        }
      }
      else {// if n1 is not angular neighbor of i
        for (int l = 0; l<NUM_OF_ABC; ++l){
          int index_local = n1*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
          g_s_angular_trial[index_local] = s[l];//save trial (unchanged) to global memory (on GPU)
        }
      }
      find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
    }

    // normalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
    }
    
    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    if (paramb.version == 5) {
      apply_ann_one_layer_nep5(
        annmb.dim, annmb.num_neurons1, annmb.w0[t2], annmb.b0[t2], annmb.w1[t2], annmb.b1, q, F, Fp);
    } else {
      apply_ann_one_layer(
        annmb.dim, annmb.num_neurons1, annmb.w0[t2], annmb.b0[t2], annmb.w1[t2], annmb.b1, q, F, Fp);
    }
    g_delta_pe[n1] = F-g_pe[n1];
  }
}

// a kernel with a single thread <<<1, 1>>>
static __global__ void find_i_energy_nep_old(
  NEP_Energy::ParaMB paramb,
  NEP_Energy::ANN annmb,
  const int N,
  const int i,
  const int g_NN_radial,
  const int* g_NN_angular,
  const int t1_before,
  const int t1_after,
  const int* __restrict__ g_t2_radial,
  const float* __restrict__ g_x12_radial,
  const float* __restrict__ g_y12_radial,
  const float* __restrict__ g_z12_radial,
  const bool* __restrict__ g_is_neigh_angular,
  float* g_delta_pe,
  float* g_q_radial_i,
  float* g_s_angular_i,
  float* g_q_radial_trial,
  float* g_s_angular_trial)
{
  float q_after[MAX_DIM] = {0.0f};
  float q_before[MAX_DIM] = {0.0f};

  // get radial descriptors 
  for (int i1 = 0; i1 < g_NN_radial; ++i1) {
    float r12[3] = {g_x12_radial[i1], g_y12_radial[i1], g_z12_radial[i1]};
    float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    float fc12;
    int t2 = g_t2_radial[i1];
    double rc = paramb.rc_radial;
    double rcinv = paramb.rcinv_radial;
    if (paramb.use_typewise_cutoff) {
      printf("typewise cutoff unsupported in MC now");
      //throw std::exception();
      /* rc = min(
        (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
          COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
          paramb.typewise_cutoff_radial_factor,
        rc);
      rcinv = 1.0f / rc; */
    }
    find_fc(rc, rcinv, d12, fc12);

    float fn12[MAX_NUM_N];
    find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      float gn12_before = 0.0f;
      float gn12_after = 0.0f;
      for (int k = 0; k <= paramb.basis_size_radial; ++k) {
        int c_index_before = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
        int c_index_after = c_index_before;
        c_index_before += t1_before * paramb.num_types + t2;
        c_index_after += t1_after * paramb.num_types + t2;
        gn12_before += fn12[k] * annmb.c[c_index_before];
        gn12_after += fn12[k] * annmb.c[c_index_after];
      }
      q_before[n] += gn12_before;
      q_after[n] += gn12_after;
    }
  }
  for (int n = 0; n <= paramb.n_max_radial; ++n) {
    int index = N*(paramb.n_max_radial+1) + n;
    g_q_radial_trial[index] = q_after[n];//save trial to global memory (on GPU)
  }

  // get angular descriptors
  for (int n = 0; n <= paramb.n_max_angular; ++n) {
    float s_before[NUM_OF_ABC] = {0.0f};
    float s_after[NUM_OF_ABC] = {0.0f};
    for (int i1 = 0; i1 < g_NN_radial; ++i1) {/// radial !!!! since g_x12_angular has shape of g_x12_radial
      float r12[3] = {g_x12_radial[i1], g_y12_radial[i1], g_z12_radial[i1]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      if (g_is_neigh_angular){
        float fc12;
        int t2 = g_t2_radial[i1];
        double rc = paramb.rc_angular;
        double rcinv = paramb.rcinv_angular;
        if (paramb.use_typewise_cutoff) {
          printf("typewise cutoff unsupported in MC now");
          //throw std::exception();
          /* rc = min(
            (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
              COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
              paramb.typewise_cutoff_angular_factor,
            rc);
          rcinv = 1.0f / rc; */
        }
        find_fc(rc, rcinv, d12, fc12);

        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12_before = 0.0f;
        float gn12_after = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index_before = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          int c_index_after = c_index_before;
          c_index_before += t1_before * paramb.num_types + t2 + paramb.num_c_radial;
          c_index_after += t1_after * paramb.num_types + t2 + paramb.num_c_radial;
          gn12_before += fn12[k] * annmb.c[c_index_before];
          gn12_after += fn12[k] * annmb.c[c_index_after];
        }
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12_before, s_before);
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12_after, s_after);
      }
    }
    find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s_before, q_before + (paramb.n_max_radial + 1));
    find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s_after, q_after + (paramb.n_max_radial + 1));
    for (int l = 0; l<NUM_OF_ABC; ++l){
      int index_local = N*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
      g_s_angular_trial[index_local] = s_after[l];//save trial to global memory (on GPU)
    }
  }

  // normalize descriptor
  for (int d = 0; d < annmb.dim; ++d) {
    q_before[d] = q_before[d] * paramb.q_scaler[d];
    q_after[d] = q_after[d] * paramb.q_scaler[d];
  }

  // get energy and energy gradient
  float F_before = 0.0f, F_after = 0.0f, Fp_before[MAX_DIM] = {0.0f}, Fp_after[MAX_DIM] = {0.0f};
  if (paramb.version == 5) {
    apply_ann_one_layer_nep5(
      annmb.dim, annmb.num_neurons1, annmb.w0[t1_before], annmb.b0[t1_before], annmb.w1[t1_before], annmb.b1, q_before, F_before, Fp_before);
      apply_ann_one_layer_nep5(
      annmb.dim, annmb.num_neurons1, annmb.w0[t1_after], annmb.b0[t1_after], annmb.w1[t1_after], annmb.b1, q_after, F_after, Fp_after);
  } else {
    apply_ann_one_layer(
      annmb.dim, annmb.num_neurons1, annmb.w0[t1_before], annmb.b0[t1_before], annmb.w1[t1_before], annmb.b1, q_before, F_before, Fp_before);
    apply_ann_one_layer(
      annmb.dim, annmb.num_neurons1, annmb.w0[t1_after], annmb.b0[t1_after], annmb.w1[t1_after], annmb.b1, q_after, F_after, Fp_after);
  }
  g_delta_pe[N] = F_after-F_before;
  //printf("old %.6f\n", g_delta_pe[N]);
}

// a kernel with a single thread <<<1, 1>>>
static __global__ void find_i_energy_nep(
  NEP_Energy::ParaMB paramb,
  NEP_Energy::ANN annmb,
  const int N,
  const int i,
  const int g_NN_radial,
  const int* g_NN_angular,
  const int t1_after,
  const int* __restrict__ g_t2_radial,
  const float* __restrict__ g_x12_radial,
  const float* __restrict__ g_y12_radial,
  const float* __restrict__ g_z12_radial,
  const bool* __restrict__ g_is_neigh_angular,
  float* g_pe,
  float* g_delta_pe,
  float* g_q_radial,
  float* g_s_angular,
  float* g_delta_q_radial_i,
  float* g_delta_s_angular_i,
  float* g_q_radial_trial,
  float* g_s_angular_trial)
{
  float q[MAX_DIM] = {0.0f};

  // get radial descriptors 
  for (int n = 0; n <= paramb.n_max_radial; ++n) {
    int index_i = N*(paramb.n_max_radial+1) + n;
/*     q[n] = g_q_radial[index_i];
    for (int n1 = 0; n1 < g_NN_radial; ++n1) {
      int index = n1*(paramb.n_max_radial+1) + n;
      q[n] += g_delta_q_radial_i[index];
    } */
    q[n] = g_q_radial[index_i] + g_q_radial_trial[index_i];
    g_q_radial_trial[index_i] = q[n];//save trial to global memory (on GPU)
  }

  // get angular descriptors
  for (int n = 0; n <= paramb.n_max_angular; ++n) {
    float s[NUM_OF_ABC] = {0.0f};
    for (int l = 0; l<NUM_OF_ABC; ++l){
      int index = N*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
/*       s[l] = g_s_angular[index];
      for (int n1 = 0; n1 < g_NN_radial; ++n1) {/// radial !!!! since g_x12_angular has shape of g_x12_radial
        int index_n1 = n1*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
        s[l] += g_delta_s_angular_i[index_n1];
      } */
        s[l] = g_s_angular[index] + g_s_angular_trial[index];
        g_s_angular_trial[index] = s[l];//save trial to global memory (on GPU)
    }
    find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
/*     for (int l = 0; l<NUM_OF_ABC; ++l){
      int index_local = N*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
      g_s_angular_trial[index_local] = s[l];//save trial to global memory (on GPU)
    } */
  }
  
  // normalize descriptor
  for (int d = 0; d < annmb.dim; ++d) {
    q[d] = q[d] * paramb.q_scaler[d];
  }

  // get energy and energy gradient
  float F = 0.0f, Fp[MAX_DIM] = {0.0f};
  if (paramb.version == 5) {
      apply_ann_one_layer_nep5(
      annmb.dim, annmb.num_neurons1, annmb.w0[t1_after], annmb.b0[t1_after], annmb.w1[t1_after], annmb.b1, q, F, Fp);
  } else {
    apply_ann_one_layer(
      annmb.dim, annmb.num_neurons1, annmb.w0[t1_after], annmb.b0[t1_after], annmb.w1[t1_after], annmb.b1, q, F, Fp);
  }
  g_delta_pe[N] = F-g_pe[N];
  //printf("i energy new %.6f ", g_delta_pe[N]);
}

static __global__ void find_energy_zbl(
  const int N,
  const NEP_Energy::ParaMB paramb,
  const NEP_Energy::ZBL zbl,
  const int* g_NN,
  const int* __restrict__ g_type,
  const int* g_t2_angular,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    float s_pe = 0.0f;
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1];
    float pow_zi = pow(float(zi), 0.23f);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_t2_angular[index];
      int zj = zbl.atomic_numbers[type2];
      float a_inv = (pow_zi + pow(float(zj), 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      if (zbl.flexibled) {
        int t1, t2;
        if (type1 < type2) {
          t1 = type1;
          t2 = type2;
        } else {
          t1 = type2;
          t2 = type1;
        }
        int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
        float ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        float rc_inner = zbl.rc_inner;
        float rc_outer = zbl.rc_outer;
        if (paramb.use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = min(
            (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor,
            rc_outer);
          rc_inner = rc_outer * 0.5f;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      s_pe += f * 0.5f;
    }
    g_pe[n1] += s_pe;
  }
}

void NEP_Energy::find_energy(
  const int N,
  const int i,
  const int* g_NN_angular,
  const int type_i,
  const int type_j,
  const int* g_t2_radial,
  const float* g_x12_radial,
  const float* g_y12_radial,
  const float* g_z12_radial,
  const bool* g_is_neigh_angular,
  float* g_delta_pe,
  float* g_pe)
{
  find_energy_nep<<<(N - 1) / 64 + 1, 64>>>(
    paramb,
    annmb,
    N,
    i,
    type_i,
    type_j,
    g_t2_radial,
    g_x12_radial,
    g_y12_radial,
    g_z12_radial,
    g_is_neigh_angular,
    g_delta_pe,
    g_pe,
    nep_data.q_radial_local.data(),
    nep_data.s_angular_local.data(),
    nep_data.q_radial_i.data(),
    nep_data.s_angular_i.data(),
    nep_data.q_radial_trial_local.data(),
    nep_data.s_angular_trial_local.data());
  GPU_CHECK_KERNEL

  find_i_energy_nep<<<1,1>>>(
    paramb,
    annmb,
    N,
    i,
    N,
    g_NN_angular,
    type_j,
    g_t2_radial,
    g_x12_radial,
    g_y12_radial,
    g_z12_radial,
    g_is_neigh_angular,
    g_pe,
    g_delta_pe,
    nep_data.q_radial_local.data(),
    nep_data.s_angular_local.data(),
    nep_data.q_radial_i.data(),
    nep_data.s_angular_i.data(),
    nep_data.q_radial_trial_local.data(),
    nep_data.s_angular_trial_local.data());

/*   find_i_energy_nep_old<<<1,1>>>(
    paramb,
    annmb,
    N,
    i,
    N,
    g_NN_angular,
    type_i,
    type_j,
    g_t2_radial,
    g_x12_radial,
    g_y12_radial,
    g_z12_radial,
    g_is_neigh_angular,
    g_delta_pe,
    nep_data.q_radial_i.data(),
    nep_data.s_angular_i.data(),
    nep_data.q_radial_trial_local.data(),
    nep_data.s_angular_trial_local.data()); */

  /*  *** todo *** zbl support
  if (zbl.enabled) {
    find_energy_zbl<<<(N - 1) / 64 + 1, 64>>>(
      N,
      paramb,
      zbl,
      g_NN_angular,
      g_type,
      g_t2_angular,
      g_x12_angular,
      g_y12_angular,
      g_z12_angular,
      g_pe);
    GPU_CHECK_KERNEL
  }
  */
}

static __global__ void accept_trial_nep(
  NEP_Energy::ParaMB paramb,
  const int N_local,
  const int* atom_local,
  float* q_radial,
  float* s_angular,
  float* q_radial_trial,
  float* s_angular_trial,
  float* g_pe_before,
  float* g_delta_pe,
  const int i)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k <= N_local) {
    int n1;
    if (k < N_local){ 
      n1 = atom_local[k];
    }
    else {
      n1 = i;
    }
    for (int n = 0; n<=paramb.n_max_radial; ++n){
      int index_local = k*(paramb.n_max_radial+1) + n;
      int index = n1*(paramb.n_max_radial+1) + n;
      q_radial[index] = q_radial_trial[index_local];
    }
    for (int n = 0; n<=paramb.n_max_angular; ++n){
      for (int l = 0; l<NUM_OF_ABC; ++l){
        int index_local = k*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
        int index = n1*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
        s_angular[index] = s_angular_trial[index_local];
      }
    }
    g_pe_before[n1] += g_delta_pe[k];
  }
}

void NEP_Energy::accept_trial(
  const int N_local,
  const int* atom_local,
  float* g_delta_pe,
  const int i)
{
  accept_trial_nep<<<((N_local+1) - 1) / 64 + 1, 64>>>(
    NEP_Energy::paramb,
    N_local,
    atom_local,
    nep_data.q_radial.data(),
    nep_data.s_angular.data(),
    nep_data.q_radial_trial_local.data(),
    nep_data.s_angular_trial_local.data(),
    nep_data.pe.data(),
    g_delta_pe,
    i);
}

//static __global__ void compute_and_save_q_rad_s_ang()

static __global__ void find_neighbor_list_large_box(
  NEP_Energy::ParaMB paramb,
  const int N,
  const int nx,
  const int ny,
  const int nz,
  const Box box,
  const int* g_type,
  const int* __restrict__ g_cell_count,
  const int* __restrict__ g_cell_count_sum,
  const int* __restrict__ g_cell_contents,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }

  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int t1 = g_type[n1];
  int count_radial = 0;
  int count_angular = 0;

  int cell_id;
  int cell_id_x;
  int cell_id_y;
  int cell_id_z;
  find_cell_id(
    box,
    x1,
    y1,
    z1,
    2.0f * paramb.rcinv_radial,
    nx,
    ny,
    nz,
    cell_id_x,
    cell_id_y,
    cell_id_z,
    cell_id);

  const int z_lim = box.pbc_z ? 2 : 0;
  const int y_lim = box.pbc_y ? 2 : 0;
  const int x_lim = box.pbc_x ? 2 : 0;

  for (int zz = -z_lim; zz <= z_lim; ++zz) {
    for (int yy = -y_lim; yy <= y_lim; ++yy) {
      for (int xx = -x_lim; xx <= x_lim; ++xx) {
        int neighbor_cell = cell_id + zz * nx * ny + yy * nx + xx;
        if (cell_id_x + xx < 0)
          neighbor_cell += nx;
        if (cell_id_x + xx >= nx)
          neighbor_cell -= nx;
        if (cell_id_y + yy < 0)
          neighbor_cell += ny * nx;
        if (cell_id_y + yy >= ny)
          neighbor_cell -= ny * nx;
        if (cell_id_z + zz < 0)
          neighbor_cell += nz * ny * nx;
        if (cell_id_z + zz >= nz)
          neighbor_cell -= nz * ny * nx;

        const int num_atoms_neighbor_cell = g_cell_count[neighbor_cell];
        const int num_atoms_previous_cells = g_cell_count_sum[neighbor_cell];

        for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
          const int n2 = g_cell_contents[num_atoms_previous_cells + m];

          if (n2 < 0 || n2 >= N || n1 == n2) {
            continue;
          }

          double x12double = g_x[n2] - x1;
          double y12double = g_y[n2] - y1;
          double z12double = g_z[n2] - z1;
          apply_mic(box, x12double, y12double, z12double);
          float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
          float d12_square = x12 * x12 + y12 * y12 + z12 * z12;

          int t2 = g_type[n2];
          float rc_radial = paramb.rc_radial;
          float rc_angular = paramb.rc_angular;
          if (paramb.use_typewise_cutoff) {
            int z1 = paramb.atomic_numbers[t1];
            int z2 = paramb.atomic_numbers[t2];
            rc_radial = min(
              (COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * paramb.typewise_cutoff_radial_factor,
              rc_radial);
            rc_angular = min(
              (COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * paramb.typewise_cutoff_angular_factor,
              rc_angular);
          }

          if (d12_square >= rc_radial * rc_radial) {
            continue;
          }

          g_NL_radial[count_radial++ * N + n1] = n2;

          if (d12_square < rc_angular * rc_angular) {
            g_NL_angular[count_angular++ * N + n1] = n2;
          }
        }
      }
    }
  }

  g_NN_radial[n1] = count_radial;
  g_NN_angular[n1] = count_angular;
}

static __global__ void find_descriptor(
  NEP_Energy::ParaMB paramb,
  NEP_Energy::ANN annmb,
  const int N,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const bool is_polarizability,
#ifdef USE_TABLE
  const float* __restrict__ g_gn_radial,
  const float* __restrict__ g_gn_angular,
#endif
  float* g_pe,
  float* g_q,
  float* g_s)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float q[MAX_DIM] = {0.0f};

    // get radial descriptors
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);

#ifdef USE_TABLE
      int index_left, index_right;
      float weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_radial, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * paramb.num_types + g_type[n2];
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        q[n] +=
          g_gn_radial[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_left +
          g_gn_radial[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_right;
      }
#else
      float fc12;
      int t2 = g_type[n2];
      float rc = paramb.rc_radial;
      if (paramb.use_typewise_cutoff) {
        rc = min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
      }
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];

      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        q[n] += gn12;
      }
#endif
    }
    int index;
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
        index = n1*(paramb.n_max_radial+1) + n;
        g_q[index] = q[n]; // save to global memory (on GPU)
      }

    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int n2 = g_NL_angular[n1 + N * i1];
        double x12double = g_x[n2] - x1;
        double y12double = g_y[n2] - y1;
        double z12double = g_z[n2] - z1;
        apply_mic(box, x12double, y12double, z12double);
        float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
#ifdef USE_TABLE
        int index_left, index_right;
        float weight_left, weight_right;
        find_index_and_weight(
          d12 * paramb.rcinv_angular, index_left, index_right, weight_left, weight_right);
        int t12 = t1 * paramb.num_types + g_type[n2];
        float gn12 =
          g_gn_angular[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n] *
            weight_left +
          g_gn_angular[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n] *
            weight_right;
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
#else
        float fc12;
        int t2 = g_type[n2];
        float rc = paramb.rc_angular;
        if (paramb.use_typewise_cutoff) {
          rc = min(
            (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
             COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
              paramb.typewise_cutoff_angular_factor,
            rc);
        }
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
#endif
      }
      int index;
      for (int l = 0; l<NUM_OF_ABC; ++l){
        index = n1*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
        g_s[index] = s[l];
      }// save to global memory (on GPU)
      find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
    }
    
    // normalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

    apply_ann_one_layer(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1],
        annmb.b0[t1],
        annmb.w1[t1],
        annmb.b1,
        q,
        F,
        Fp);
    g_pe[n1] = F;
  }
}

// large box fo MD applications
void NEP_Energy::compute_large_box(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N - 1) / BLOCK_SIZE + 1;

  const double rc_cell_list = 0.5 * paramb.rc_radial;

  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);

  find_cell_list(
    rc_cell_list,
    num_bins,
    box,
    position_per_atom,
    nep_data.cell_count,
    nep_data.cell_count_sum,
    nep_data.cell_contents);

  find_neighbor_list_large_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    N,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    box,
    type.data(),
    nep_data.cell_count.data(),
    nep_data.cell_count_sum.data(),
    nep_data.cell_contents.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data());
  GPU_CHECK_KERNEL


  gpu_sort_neighbor_list<<<N, paramb.MN_radial, paramb.MN_radial * sizeof(int)>>>(
    N, nep_data.NN_radial.data(), nep_data.NL_radial.data());
  GPU_CHECK_KERNEL

  gpu_sort_neighbor_list<<<N, paramb.MN_angular, paramb.MN_angular * sizeof(int)>>>(
    N, nep_data.NN_angular.data(), nep_data.NL_angular.data());
  GPU_CHECK_KERNEL

  bool is_polarizability = false;
  find_descriptor<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    is_polarizability,
#ifdef USE_TABLE
    nep_data.gn_radial.data(),
    nep_data.gn_angular.data(),
#endif
    nep_data.pe.data(), 
    nep_data.q_radial.data(), 
    nep_data.s_angular.data());
  GPU_CHECK_KERNEL
}