/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            K.U. Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

//Inclued UNO files
#include "tools/Options.hpp"
#include "tools/Timer.hpp"

#include "optimization/Iterate.hpp"
#include "optimization/ModelFactory.hpp"
#include "optimization/ScaledModel.hpp"
#include "preprocessing/Preprocessing.hpp"
#include "linear_algebra/CSCSymmetricMatrix.hpp"

#include "ingredients/globalization_strategy/GlobalizationStrategyFactory.hpp"
#include "ingredients/globalization_mechanism/GlobalizationMechanismFactory.hpp"
#include "ingredients/constraint_relaxation_strategy/ConstraintRelaxationStrategyFactory.hpp"

#include "tools/Logger.hpp"

#include "Uno.hpp"
Level Logger::logger_level = INFO;

// Casadi Includes
#include "uno_interface.hpp"
#include "casadi/core/casadi_misc.hpp"
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include <type_traits>


namespace casadi {

  extern "C"
  int CASADI_NLPSOL_UNO_EXPORT
  casadi_register_nlpsol_uno(Nlpsol::Plugin* plugin) {
    plugin->creator = UnoInterface::creator;
    plugin->name = "uno";
    plugin->doc = UnoInterface::meta_doc.c_str();
    plugin->version = CASADI_VERSION;
    plugin->options = &UnoInterface::options_;
    plugin->deserialize = &UnoInterface::deserialize;
    return 0;
  }

  extern "C"
  void CASADI_NLPSOL_UNO_EXPORT casadi_load_nlpsol_uno() {
    Nlpsol::registerPlugin(casadi_register_nlpsol_uno);
  }

  /*---------------------------------------------
  Constructor Destructor UnoInterface
  ----------------------------------------------*/

  UnoInterface::UnoInterface(const std::string& name, const Function& nlp)
    : Nlpsol(name, nlp) {
  }


  UnoInterface::~UnoInterface() {
    clear_mem();
  }

  const Options UnoInterface::options_
  = {{&Nlpsol::options_},
     {{"uno",
       {OT_DICT,
        "Options to be passed to UNO"}}
     }
  };

  UnoMemory::UnoMemory(const UnoInterface& uno_interface) : self(uno_interface), NlpsolMemory() {
    
  }

  UnoMemory::~UnoMemory() {
    
  }

  /*----------------------------------------------------------------
  From here CasadiModel function definition
  ---------------------------------------------------------------*/

  CasadiModel::CasadiModel(const std::string& file_name, const UnoInterface& uno_interface, UnoMemory* mem) :
    Model(file_name, uno_interface.nx_, uno_interface.ng_, NONLINEAR),
    mem_(mem),
    // allocate vectors
    casadi_tmp_gradient(this->number_variables),
    casadi_tmp_multipliers(this->number_constraints),
    casadi_tmp_constraint_jacobian(mem->self.get_function("nlp_jac_g").sparsity_out(0).nnz()),
    casadi_tmp_hessian(mem->self.get_function("nlp_hess_l").sparsity_out(0).nnz()),
    variables_bounds(this->number_variables),
    constraint_bounds(this->number_constraints),
    variable_status(this->number_variables),
    constraint_type(this->number_constraints),
    constraint_status(this->number_constraints) {
    
    // this->asl->i.congrd_mode = 0;

    // dimensions
    this->objective_sign = 1.;//(this->asl->i.objtype_[0] == 1) ? -1. : 1.;

    // variables
    this->generate_variables();

    // constraints
    this->equality_constraints.reserve(this->number_constraints);
    this->inequality_constraints.reserve(this->number_constraints);
    this->linear_constraints.reserve(this->number_constraints);
    this->generate_constraints();
    this->set_function_types(file_name);

    // compute number of nonzeros
    this->objective_gradient_maximum_number_nonzeros = static_cast<size_t>(0);//static_cast<size_t>(this->asl->i.nzo_);
    this->jacobian_maximum_number_nonzeros = static_cast<size_t>(mem_->self.jacg_sp_.nnz());
    this->set_number_hessian_nonzeros();

  }

  void CasadiModel::set_number_hessian_nonzeros() {
   // compute the maximum number of nonzero elements, provided that all multipliers are non-zero
   // int (*Sphset) (ASL*, SputInfo**, int nobj, int ow, int y, int uptri);
   const int objective_number = -1;
   const int upper_triangular = 1;
  //  this->hessian_maximum_number_nonzeros = static_cast<size_t>((*(this->asl)->p.Sphset)(this->asl, nullptr, objective_number, 1, 1,
  //        upper_triangular));
   this->hessian_maximum_number_nonzeros = static_cast<size_t>(this->mem_->self.hesslag_sp_.nnz());
   this->casadi_tmp_hessian.reserve(this->hessian_maximum_number_nonzeros);

   // use Lagrangian scale: in AMPL, the Lagrangian is f + lambda.g, while Uno uses f - lambda.g
   int nerror{};
  //  lagscale_ASL(this->asl, -1., &nerror);
}

  /* ------- Define functions to evaluate solver functions  -----------*/
  double CasadiModel::evaluate_objective(const std::vector<double>& x) const {

    double obj;
    mem_->arg[0] = get_ptr(x);
    mem_->arg[1] = mem_->d_nlp.p;
    mem_->res[0] = &obj;
    casadi_assert(mem_->self.calc_function(mem_, "nlp_f")==0, "Failed to evaluate objective function.");
    return obj;
  }

  void CasadiModel::evaluate_objective_gradient(const std::vector<double>& x, SparseVector<double>& gradient) const {

    // Evaluate the objective gradient
    mem_->arg[0] = get_ptr(x);
    mem_->arg[1] = mem_->d_nlp.p;
    mem_->res[0] = get_ptr(casadi_tmp_gradient);
    casadi_assert(mem_->self.calc_function(mem_, "nlp_grad_f")==0, "Failed to evaluate gradient of objective.");

    // Write everything into the SparseVector
    for (unsigned int index = 0; index < this->number_variables; ++index){
      // Is the index correct? Start counting at zero?
      gradient.insert(index, casadi_tmp_gradient[index]);
    }
  }

  void CasadiModel::evaluate_constraints(const std::vector<double>& x, std::vector<double>& constraints) const {

    mem_->arg[0] = get_ptr(x);
    mem_->arg[1] = mem_->d_nlp.p;
    mem_->res[0] = get_ptr(constraints);
    casadi_assert(mem_->self.calc_function(mem_, "nlp_g")==0, "Failed to evaluate constraints.");
  }

  void CasadiModel::evaluate_constraint_gradient(const std::vector<double>& x, size_t j, SparseVector<double>& gradient) const {
    // Sparsity pattern of jacobian is saved in mem->self ...
    // Evaluate numerically
      mem_->arg[0] = get_ptr(x);
      mem_->arg[1] = mem_->d_nlp.p;
      mem_->res[0] = get_ptr(casadi_tmp_constraint_jacobian);
      casadi_assert(mem_->self.calc_function(mem_, "nlp_jac_g")==0, "Failed to evaluate constraint jacobian.");

      // Write everything into RectangularMatrix...
      gradient.clear();
      for (size_t k= mem_->self.jacg_sp_.colind()[j]; k< mem_->self.jacg_sp_.colind()[j + 1];++k) {
        const size_t i = mem_->self.jacg_sp_.row()[k];
        const double entry = casadi_tmp_hessian[k];
        gradient.insert(i, entry);
      }
  }

  void CasadiModel::evaluate_constraint_jacobian(const std::vector<double>& x, RectangularMatrix<double>& constraint_jacobian) const {
    // Sparsity pattern of jacobian is saved in mem->self ...
    // Evaluate numerically
      mem_->arg[0] = get_ptr(x);
      mem_->arg[1] = mem_->d_nlp.p;
      mem_->res[0] = get_ptr(casadi_tmp_constraint_jacobian);
      casadi_assert(mem_->self.calc_function(mem_, "nlp_jac_g")==0, "Failed to evaluate constraint jacobian.");

      // Write everything into RectangularMatrix...
      size_t n_columns = constraint_jacobian.size();
      for (size_t j=0; j<n_columns; ++j) {

        constraint_jacobian[j].clear();
        for (size_t k= mem_->self.jacg_sp_.colind()[j]; k< mem_->self.jacg_sp_.colind()[j + 1];++k) {
          const size_t i = mem_->self.jacg_sp_.row()[k];
          const double entry = casadi_tmp_hessian[k];
          constraint_jacobian[j].insert(i, entry);
        }
      }
  }

  void CasadiModel::evaluate_lagrangian_hessian(const std::vector<double>& x, double objective_multiplier, const std::vector<double>& multipliers,
         SymmetricMatrix<double>& hessian) const {
    
    // scale by the objective sign
    objective_multiplier *= this->objective_sign;
    // Evaluate numerically
    mem_->arg[0] = get_ptr(x);
    mem_->arg[1] = mem_->d_nlp.p;
    mem_->arg[2] = &objective_multiplier;
    casadi_copy(get_ptr(multipliers), this->number_constraints, get_ptr(casadi_tmp_multipliers));
    casadi_scal(this->number_constraints, -1., get_ptr(casadi_tmp_multipliers));
    mem_->arg[3] = get_ptr(casadi_tmp_multipliers);
    mem_->res[0] = get_ptr(casadi_tmp_hessian);
    casadi_assert(mem_->self.calc_function(mem_, "nlp_hess_l")==0, "Failed to evaluate Lagrangian hessian.");

    hessian.reset();
    // Write the hessian into Symmetric matrix ....
    for (size_t j=0; j<this->number_variables;++j) {
      for (size_t k=mem_->self.hesslag_sp_.colind()[j]; k< mem_->self.hesslag_sp_.colind()[j + 1];++k) {
         const size_t i = mem_->self.hesslag_sp_.row()[k];
         const double entry = casadi_tmp_hessian[k];
         hessian.insert(entry, i, j);
      }
      hessian.finalize_column(j);
   }
  }

  double CasadiModel::get_variable_lower_bound(size_t i) const {
    return this->variables_bounds[i].lb;//mem_->d_nlp.lbz[i];
  }

  double CasadiModel::get_variable_upper_bound(size_t i) const {
    return this->variables_bounds[i].ub;//mem_->d_nlp.ubz[i];
  }

  BoundType CasadiModel::get_variable_bound_type(size_t i) const {
    return this->variable_status[i];
  }

  double CasadiModel::get_constraint_lower_bound(size_t j) const {
    return this->constraint_bounds[j].lb;//mem_->d_nlp.lbz[j+this->number_variables];
  }

  double CasadiModel::get_constraint_upper_bound(size_t j) const {
    return this->constraint_bounds[j].ub;//mem_->d_nlp.ubz[j+this->number_variables];
  }

  FunctionType CasadiModel::get_constraint_type(size_t j) const {
    return this->constraint_type[j];
  }

  BoundType CasadiModel::get_constraint_bound_type(size_t j) const {
    return this->constraint_status[j];
  }

  size_t CasadiModel::get_maximum_number_objective_gradient_nonzeros() const {
    return this->objective_gradient_maximum_number_nonzeros;;
  }
  size_t CasadiModel::get_maximum_number_jacobian_nonzeros() const {
    return this->jacobian_maximum_number_nonzeros;
  }
  size_t CasadiModel::get_maximum_number_hessian_nonzeros() const {
    return this->hessian_maximum_number_nonzeros;
  }

  void CasadiModel::get_initial_primal_point(std::vector<double>& x) const {
    assert(x.size() >= this->number_variables);
    std::copy(mem_->d_nlp.x0, mem_->d_nlp.x0 + this->number_variables, begin(x));
    std::cout << "Vector is " << x << std::endl;
  }
  void CasadiModel::get_initial_dual_point(std::vector<double>& multipliers) const {
    assert(multipliers.size() >= this->number_constraints);
   std::copy(mem_->d_nlp.lam_g0, mem_->d_nlp.lam_g0 + this->number_constraints, begin(multipliers));
  }
  void CasadiModel::postprocess_solution(Iterate& iterate, TerminationStatus termination_status) const {
  // do nothing
  }

  void CasadiModel::generate_variables() {
    // Calculate variable bounds
    for (size_t i=0; i< this->number_variables; ++i) {
      double lb = (mem_->d_nlp.lbx != nullptr) ? *mem_->d_nlp.lbx+i : -INF<double>;
      double ub = (mem_->d_nlp.ubx != nullptr) ? *mem_->d_nlp.ubx+i : INF<double>;
      if (lb == ub) {
          WARNING << "Variable x" << i << " has identical bounds\n";
      }
      this->variables_bounds[i] = {lb, ub};
    }

    Model::determine_bounds_types(this->variables_bounds, this->variable_status);
    // figure out the bounded variables
    for (size_t i=0; i< this->number_variables; ++i) {
      const BoundType status = this->get_variable_bound_type(i);
      if (status == BOUNDED_LOWER || status == BOUNDED_BOTH_SIDES) {
          this->lower_bounded_variables.push_back(i);
          if (status == BOUNDED_LOWER) {
            this->single_lower_bounded_variables.push_back(i);
          }
      }
      if (status == BOUNDED_UPPER || status == BOUNDED_BOTH_SIDES) {
          this->upper_bounded_variables.push_back(i);
          if (status == BOUNDED_UPPER) {
            this->single_upper_bounded_variables.push_back(i);
          }
      }
    }
  }

  void CasadiModel::generate_constraints() {
  //  auto d_nlp = &this->mem_->d_nlp;
   for (size_t i=0; i< this->number_constraints; ++i) {
      double lb = (mem_->d_nlp.lbg != nullptr) ? *this->mem_->d_nlp.lbg+i : -INF<double>;
      double ub = (mem_->d_nlp.ubg != nullptr) ? *this->mem_->d_nlp.ubg+i : INF<double>;
      // std::cout << "Memory location " << d_nlp->x0 << std::endl;
      // double lb = d_nlp->x0[0];//+this->number_variables+i;
      // double ub = *mem_->d_nlp.ubg;//+this->number_variables+i;
      this->constraint_bounds[i] = {lb, ub};
      std::cout << "lb: " << lb  << std::endl;
      std::cout << "ub: " << ub << std::endl;

   }
   Model::determine_bounds_types(this->constraint_bounds, this->constraint_status);
   this->determine_constraints();
  }

  void CasadiModel::set_function_types(std::string file_name) {
    // // allocate a temporary ASL to read Hessian sparsity pattern
    // ASL* asl_fg = ASL_alloc(ASL_read_fg);
    // // char* stub = getstops(file_name, option_info);
    // //if (file_name == nullptr) {
    // //	usage_ASL(option_info, 1);
    // //}

    // FILE* nl = jac0dim_ASL(asl_fg, file_name.data(), static_cast<int>(file_name.size()));
    // // specific read function
    // // qp_read_ASL(asl_fg, nl, ASL_findgroups);

    // // // constraints
    // // if (asl_fg->i.n_con_ != static_cast<int>(this->number_constraints)) {
    // //     throw std::length_error("AMPLModel.set_function_types: inconsistent number of constraints");
    // // }

    this->constraint_type.reserve(this->number_constraints);
    this->problem_type = NONLINEAR;

    // // determine the type of each constraint and objective function
    // // determine if the problem is nonlinear (non-quadratic objective or nonlinear constraints)
    // this->problem_type = LINEAR;
    // int* rowq;
    // int* colqp;
    // double* delsqp;
    // for (size_t j: Range(this->number_constraints)) {
    //     int qp = nqpcheck_ASL(asl_fg, static_cast<int>(-(j + 1)), &rowq, &colqp, &delsqp);

    //     if (0 < qp) {
    //       this->constraint_type[j] = QUADRATIC;
    //       this->problem_type = NONLINEAR;
    //     }
    //     else if (qp == 0) {
    //       this->constraint_type[j] = LINEAR;
    //       this->linear_constraints.push_back(j);
    //     }
    //     else {
    //       this->constraint_type[j] = NONLINEAR;
    //       this->problem_type = NONLINEAR;
    //     }
    // }
    // // objective function
    // int qp = nqpcheck_ASL(asl_fg, 0, &rowq, &colqp, &delsqp);
    // if (0 < qp) {
    //     if (this->problem_type == LINEAR) {
    //       this->problem_type = QUADRATIC;
    //     }
    // }
    // else if (qp != 0) {
    //     this->problem_type = NONLINEAR;
    // }
    // qp_opify_ASL(asl_fg);

    // // deallocate memory
    // ASL_free(&asl_fg);
  }

  /*-------------------------------------------
  UnoInterface function definitions
  -------------------------------------------*/

  void UnoInterface::init(const Dict& opts) {
    // Call the init method of the base class
    Nlpsol::init(opts);

    // Read user options
    for (auto&& op : opts) {
      if (op.first=="uno") {
        opts_ = op.second;
      }

    }

    // Setup NLP functions
    // create_function("nlp_fg", {"x", "p"}, {"f", "g"});
    // Function gf_jg_fcn = create_function("nlp_gf_jg", {"x", "p"}, {"grad:f:x", "jac:g:x"});
    // jacg_sp_ = gf_jg_fcn.sparsity_out(1);

    // Function hess_l_fcn = create_function("nlp_hess_l", {"x", "p", "lam:f", "lam:g"},
    //                               {"hess:gamma:x:x"},
    //                               {{"gamma", {"f", "g"}}});
    // hesslag_sp_ = hess_l_fcn.sparsity_out(0);

    create_function("nlp_f", {"x", "p"}, {"f"});
    create_function("nlp_g", {"x", "p"}, {"g"});
    create_function("nlp_grad_f", {"x", "p"}, {"grad:f:x"});
    Function gf_jg_fcn = create_function("nlp_jac_g", {"x", "p"}, {"jac:g:x"});
    jacg_sp_ = gf_jg_fcn.sparsity_out(0);

    Function hess_l_fcn = create_function("nlp_hess_l", {"x", "p", "lam:f", "lam:g"},
                                  {"hess:gamma:x:x"},
                                  {{"gamma", {"f", "g"}}});
    hesslag_sp_ = hess_l_fcn.sparsity_out(0);

    // Allocate persistent memory
    alloc_w(nx_, true); // wlbx_
    alloc_w(nx_, true); // wubx_
    alloc_w(ng_, true); // wlbg_
    alloc_w(ng_, true); // wubg_
  }

  int UnoInterface::init_mem(void* mem) const {
    int return_nlpsol = Nlpsol::init_mem(mem);
    if (return_nlpsol > 0){
      printf("Return code: %d!", return_nlpsol);
    }
    auto m = static_cast<UnoMemory*>(mem);

// -------------------------------------------------------------------

   // Casadi model
   // memory should be freed somewhere else
  //  std::cout << "Init Memort acces" << std::endl;
  //  m->model = new CasadiModel("casadi_model", *this, m);
   return 0;

  }
//----------------------------------------------------------

  // }

  void UnoInterface::set_work(void* mem, const double**& arg, double**& res,
                                 casadi_int*& iw, double*& w) const {
    // auto m = static_cast<UnoMemory*>(mem);

    // Set work in base classes
    Nlpsol::set_work(mem, arg, res, iw, w);

    auto m = static_cast<UnoMemory*>(mem);

// -------------------------------------------------------------------

   // Casadi model
   // memory should be freed somewhere else
    m->model = new CasadiModel("casadi_model", *this, m);

    std::cout << "Set Work acces" << std::endl;

  }

  int casadi_KN_puts(const char * const str, void * const userParams) {
    std::string s(str);
    uout() << s << std::flush;
    return s.size();
  }

  ::Options rewrite_options(const Dict& options){
    	// This function writes the casadi options into UNO options
      ::Options uno_options;
      Dict::const_iterator  it;
      for (it = options.begin(); it != options.end(); it++)
      {
          // static_assert(std::is_same<decltype(it->second), std::string>::value, "type must be 'std::string'"); 
          uno_options[it->first] = uno_options[it->second];
      }
      return uno_options;
  }

  int UnoInterface::solve(void* mem) const {
    auto m = static_cast<UnoMemory*>(mem);
    auto d_nlp = &m->d_nlp;
    std::cout << "Memory location " << d_nlp->x0 << std::endl;


    // CasadiModel casadi_model = &m->model;

    //Define options file..so far a bit cheated

    // ::Options uno_options = rewrite_options(opts_);
    ::Options uno_options = get_default_options("/home/david/casadi_fork/casadi/casadi/interfaces/uno/uno.options");
    set_logger(uno_options.get_string("logger"));

   // initialize initial primal and dual points
  //  Iterate first_iterate(ampl_model.number_variables, ampl_model.number_constraints);

   Iterate first_iterate(m->model->number_variables, m->model->number_constraints);

  //  ampl_model.get_initial_primal_point(first_iterate.primals);

   m->model->get_initial_primal_point(first_iterate.primals);
  std::cout << "Memory location " << d_nlp->x0 << std::endl;


  //  ampl_model.get_initial_dual_point(first_iterate.multipliers.constraints);

   m->model->get_initial_dual_point(first_iterate.multipliers.constraints);

  //  ampl_model.project_primals_onto_bounds(first_iterate.primals);

   m->model->project_primals_onto_bounds(first_iterate.primals);

  // //  // reformulate (scale, add slacks) if necessary
  // //  std::unique_ptr<Model> model = ModelFactory::reformulate(ampl_model, first_iterate, options);
  // //  std::unique_ptr<Model> model = ModelFactory::reformulate(m->model, first_iterate, uno_options);

    std::unique_ptr<Model> model = std::make_unique<ScaledModel>(*m->model, first_iterate, uno_options);

  // //  // enforce linear constraints at initial point
  // //  if (options.get_bool("enforce_linear_constraints")) {
  // //     Preprocessing::enforce_linear_constraints(options, *model, first_iterate.primals, first_iterate.multipliers);
  // //  }

   if (uno_options.get_bool("enforce_linear_constraints")) {
      // Preprocessing::enforce_linear_constraints(uno_options, *model, first_iterate.primals, first_iterate.multipliers);
      Preprocessing::enforce_linear_constraints(uno_options, *model, first_iterate.primals, first_iterate.multipliers);
   }

  // //  // create the constraint relaxation strategy
  // //  auto constraint_relaxation_strategy = ConstraintRelaxationStrategyFactory::create(*model, options);

   auto constraint_relaxation_strategy = ConstraintRelaxationStrategyFactory::create(*model, uno_options);

  // //  // create the globalization mechanism
  // //  auto mechanism = GlobalizationMechanismFactory::create(*constraint_relaxation_strategy, options);

   auto mechanism = GlobalizationMechanismFactory::create(*constraint_relaxation_strategy, uno_options);

  //  // instantiate the combination of ingredients and solve the problem
  //  Uno uno = Uno(*mechanism, options);
  //  Result result = uno.solve(*model, first_iterate, options);

   Uno uno = Uno(*mechanism, uno_options);
   Result result = uno.solve(*model, first_iterate, uno_options);


   // Write the solution to Casadi .....
   // Negate rc to match CasADi's definition
    // casadi_scal(nx_ + ng_, -1., get_ptr(m->rc));

  // Get primal solution
  casadi_copy(get_ptr(result.solution.primals), nx_, d_nlp->z);

  // Get optimal cost
  d_nlp->objective = result.solution.evaluations.objective;

  // Get dual solution
  // Get dual solution (simple bounds)
  for (casadi_int i=0; i<m->model->number_variables; ++i) {
    d_nlp->lam[i] = result.solution.multipliers.upper_bounds[i]-result.solution.multipliers.upper_bounds[i];
  }
  casadi_copy(get_ptr(result.solution.multipliers.constraints)+nx_, ng_, d_nlp->lam + nx_);
  // Copy optimal constraint values to output
  casadi_copy(get_ptr(result.solution.evaluations.constraints), ng_, d_nlp->z + nx_);


   // print the optimization summary
   std::string combination = uno_options.get_string("mechanism") + " " + uno_options.get_string("constraint-relaxation") + " " + uno_options.get_string("strategy")
         + " " + uno_options.get_string("subproblem");
   std::cout << "\nUno (" << combination << ")\n";
   std::cout << Timer::get_current_date();
   std::cout << "────────────────────────────────────────\n";
   const bool print_solution = uno_options.get_bool("print_solution");
   result.print(print_solution);

    return 0;
  }

  Dict UnoInterface::get_stats(void* mem) const {
    Dict stats = Nlpsol::get_stats(mem);
    // auto m = static_cast<UnoMemory*>(mem);


    return stats;
  }

  

} // namespace casadi
