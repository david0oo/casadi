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

#include "Uno_C_API.h"
#include "uno_interface.hpp"
#include "casadi/core/casadi_misc.hpp"

#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cassert>
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

  UnoInterface::UnoInterface(const std::string& name, const Function& nlp) : Nlpsol(name, nlp)
  {
  }

  UnoInterface::~UnoInterface()
  {
    clear_mem();
  }

  const Options UnoInterface::options_
  = {{&Nlpsol::options_},
     {{"uno",
       {OT_DICT,
        "Options to be passed to UNO"}}
     }
  };

  UnoMemory::UnoMemory(const UnoInterface& uno_interface)
      : self(uno_interface), NlpsolMemory() {
    this->return_status = "Unset";
    this->model = nullptr;
    this->solver = nullptr;
    this->uno_nlp = nullptr;
  }

  UnoMemory::~UnoMemory() {
    if (this->model)  uno_destroy_model(this->model);
    if (this->solver) uno_destroy_solver(this->solver);
    delete static_cast<UnoNlp*>(this->uno_nlp);
  }

  void UnoInterface::free_mem(void* mem) const {
    delete static_cast<UnoMemory*>(mem);
  }

  /*------------------------------------------
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
    create_function("nlp_f", {"x", "p"}, {"f"});
    create_function("nlp_g", {"x", "p"}, {"g"});
    create_function("nlp_grad_f", {"x", "p"}, {"grad:f:x"});
    Function gf_jg_fcn = create_function("nlp_jac_g", {"x", "p"}, {"jac:g:x"});
    jacg_sp_ = gf_jg_fcn.sparsity_out(0);

    Function hess_l_fcn = create_function("nlp_hess_l", {"x", "p", "lam:f", "lam:g"},
                                  {"triu:hess:gamma:x:x"},
                                  {{"gamma", {"f", "g"}}});
    hesslag_sp_ = hess_l_fcn.sparsity_out(0);
    casadi_assert(hesslag_sp_.is_triu(), "Hessian must be upper triangular");

    // Allocate persistent memory
    alloc_w(nx_, true); // wlbx_
    alloc_w(nx_, true); // wubx_
    alloc_w(ng_, true); // wlbg_
    alloc_w(ng_, true); // wubg_
  }

  static uno_int casadi_uno_termination_cb(uno_int, uno_int, const double*,
      const double*, const double*, const double*, double, double, double, double, void*);

  int UnoInterface::init_mem(void* mem) const {
    if (Nlpsol::init_mem(mem)) return 1;
    auto m = static_cast<UnoMemory*>(mem);

    if (verbose_) {
      uno_int uno_major, uno_minor, uno_patch;
      uno_get_version(&uno_major, &uno_minor, &uno_patch);
      casadi_message("Using Uno v" + str(uno_major) + "." + str(uno_minor) + "." + str(uno_patch));
    }

    m->uno_nlp = new UnoNlp(m);
    m->solver = uno_create_solver();
    uno_set_solver_callbacks(m->solver, nullptr, &casadi_uno_termination_cb, m);
    // define preset and insert options given through  
    UnoNlp::insert_casadi_options(m->solver, opts_);
   return 0;

  }
//----------------------------------------------------------

  void UnoInterface::set_work(void* mem, const double**& arg, double**& res,
                                 casadi_int*& iw, double*& w) const
  {
    // Set work in base classes
    Nlpsol::set_work(mem, arg, res, iw, w);
    auto m = static_cast<UnoMemory*>(mem);
  }

  int casadi_KN_puts(const char * const str, void * const userParams) {
    std::string s(str);
    uout() << s << std::flush;
    return s.size();
  }

  // Uno's C API impl checks `termination_callback(...) == 0` to mean "terminate"
  // (Uno_C_API.cpp ~ line 368), opposite of the header comment. So we return 0
  // to terminate, non-zero to continue.
  // Invokes the user's iteration_callback (Nlpsol::fcallback_) and propagates
  // exceptions through m->cb_exception, which solve() rethrows after uno_optimize.
  static uno_int casadi_uno_termination_cb(uno_int n, uno_int ng, const double* primals,
      const double* lower_mult, const double* upper_mult, const double* constraint_mult,
      double objective_multiplier, double inf_pr, double inf_du, double compl_res,
      void* user_data) {
    constexpr uno_int CONTINUE = 1;
    constexpr uno_int TERMINATE = 0;
    UnoMemory* m = static_cast<UnoMemory*>(user_data);
    const UnoInterface& self = m->self;
    if (self.fcallback_.is_null()) return CONTINUE;

    auto d_nlp = &m->d_nlp;
    casadi_copy(primals, self.nx_, d_nlp->z);
    for (casadi_int i = 0; i < self.nx_; ++i) {
      d_nlp->lam[i] = lower_mult[i] - upper_mult[i];
    }
    if (self.ng_ > 0) {
      casadi_copy(constraint_mult, self.ng_, d_nlp->lam + self.nx_);
    }
    // f and g values aren't directly available; nullptrs are fine for
    // iteration-interrupt-style callbacks that don't read them.
    std::fill_n(m->arg, self.fcallback_.n_in(), nullptr);
    m->arg[NLPSOL_X]     = d_nlp->z;
    m->arg[NLPSOL_LAM_X] = d_nlp->lam;
    m->arg[NLPSOL_LAM_G] = d_nlp->lam + self.nx_;
    std::fill_n(m->res, self.fcallback_.n_out(), nullptr);
    double ret_double = 0;
    m->res[0] = &ret_double;

    try {
      self.fcallback_(m->arg, m->res, m->iw, m->w, 0);
    } catch (KeyboardInterruptException&) {
      m->cb_exception = std::current_exception();
      return TERMINATE;
    } catch (std::exception& ex) {
      casadi_warning(std::string("intermediate_callback: ") + ex.what());
      if (!self.iteration_callback_ignore_errors_) {
        m->cb_exception = std::current_exception();
        return TERMINATE;
      }
      return CONTINUE;
    }
    return static_cast<casadi_int>(ret_double) ? TERMINATE : CONTINUE;
  }

inline const char* return_status_string(void* solver) {
    uno_int iterate_status = uno_get_solution_status(solver);
    if (iterate_status == UNO_FEASIBLE_KKT_POINT)
    {
      return "Converged with feasible KKT point";
    }
    else if (iterate_status == UNO_FEASIBLE_FJ_POINT)
    {
      return "Converged with feasible FJ point";
    }
    else if (iterate_status == UNO_INFEASIBLE_STATIONARY_POINT)
    {
      return "Converged with infeasible stationary point";
    }
    else if (iterate_status == UNO_FEASIBLE_SMALL_STEP)
    {
      return "Terminated with feasible small step";
    }
    else if (iterate_status == UNO_INFEASIBLE_SMALL_STEP)
    {
      return "Terminated with infeasible small step";
    }
    else if (iterate_status == UNO_UNBOUNDED)
    {
      return "Terminated with unbounded problem";
    }
    else if (iterate_status == UNO_NOT_OPTIMAL)
    {
      return "Terminated with not optimal point";
    }
    else
    {
      return "Terminated with an unknown status!";
    }
  }

  inline bool return_status_success(void* solver) {
    return uno_get_optimization_status(solver) == UNO_SUCCESS
        && uno_get_solution_status(solver) == UNO_FEASIBLE_KKT_POINT;
  }

  // Map Uno status pair -> CasADi UnifiedReturnStatus.
  inline UnifiedReturnStatus unified_status_from_uno(void* solver) {
    uno_int opt = uno_get_optimization_status(solver);
    if (opt == UNO_EVALUATION_ERROR)   return SOLVER_RET_NAN;
    if (opt == UNO_ITERATION_LIMIT)    return SOLVER_RET_LIMITED;
    if (opt == UNO_TIME_LIMIT)         return SOLVER_RET_LIMITED;
    if (opt == UNO_ALGORITHMIC_ERROR)  return SOLVER_RET_EXCEPTION;
    if (opt == UNO_USER_TERMINATION)   return SOLVER_RET_UNKNOWN;
    // opt == UNO_SUCCESS: refine via solution_status.
    uno_int sol = uno_get_solution_status(solver);
    if (sol == UNO_FEASIBLE_KKT_POINT)         return SOLVER_RET_SUCCESS;
    if (sol == UNO_FEASIBLE_FJ_POINT)          return SOLVER_RET_SUCCESS;
    if (sol == UNO_INFEASIBLE_STATIONARY_POINT) return SOLVER_RET_INFEASIBLE;
    if (sol == UNO_INFEASIBLE_SMALL_STEP)      return SOLVER_RET_INFEASIBLE;
    if (sol == UNO_FEASIBLE_SMALL_STEP)        return SOLVER_RET_LIMITED;
    if (sol == UNO_UNBOUNDED)                  return SOLVER_RET_UNKNOWN;
    return SOLVER_RET_UNKNOWN;
  }
    
    int UnoInterface::solve(void* mem) const 
    {
    auto m = static_cast<UnoMemory*>(mem);
    UnoNlp* nlp = static_cast<UnoNlp*>(m->uno_nlp);
    auto d_nlp = &m->d_nlp;
    
    // model creation; previous-call model (if any) is destroyed first
    if (m->model) { uno_destroy_model(m->model); m->model = nullptr; }
    const uno_int base_indexing = UNO_ZERO_BASED_INDEXING;
    void* model = uno_create_model(UNO_PROBLEM_NONLINEAR, nx_, d_nlp->lbz, d_nlp->ubz, base_indexing);
    m->model = model;
    // set NLP
    uno_int ret;
    ret = uno_set_user_data(model, nlp);

    // constraints
    const uno_int number_jacobian_nonzeros = static_cast<size_t>(this->jacg_sp_.nnz());
    std::vector<casadi_int> jac_row_indices = this->jacg_sp_.get_row();
    std::vector<casadi_int> jac_column_indices = this->jacg_sp_.get_col();
    std::vector<uno_int> jacobian_row_indices(jac_row_indices.size());
    std::vector<uno_int> jacobian_column_indices(jac_column_indices.size());

    for (uno_int i = 0; i < number_jacobian_nonzeros; ++i) {
        jacobian_row_indices[i] = static_cast<uno_int>(jac_row_indices[i]);
        jacobian_column_indices[i] = static_cast<uno_int>(jac_column_indices[i]);
    }

    // Hessian
    const uno_int number_hessian_nonzeros = static_cast<size_t>(this->hesslag_sp_.nnz());
    std::vector<casadi_int> hess_row_indices = this->hesslag_sp_.get_row();
    std::vector<casadi_int> hess_column_indices = this->hesslag_sp_.get_col();
    std::vector<uno_int> hessian_row_indices(hess_row_indices.size());
    std::vector<uno_int> hessian_column_indices(hess_column_indices.size());
    
    for (uno_int i = 0; i < number_hessian_nonzeros; ++i)
    {
      hessian_row_indices[i] = static_cast<uno_int>(hess_row_indices[i]);
      hessian_column_indices[i] = static_cast<uno_int>(hess_column_indices[i]);
    }
    const char hessian_triangular_part = UNO_UPPER_TRIANGLE;
    const uno_int lagrangian_sign_convention = UNO_MULTIPLIER_POSITIVE;

    // Uno's uno_set_* returns bool: true on success, false on failure.
    ret = uno_set_objective(model, UNO_MINIMIZE,
        UnoNlp::objective_function_wrapper, UnoNlp::objective_gradient_wrapper);
    casadi_assert(ret, "uno_set_objective failed");

    if (ng_ > 0) {
      ret = uno_set_constraints(model, ng_, UnoNlp::constraint_functions_wrapper,
            d_nlp->lbz+nx_, d_nlp->ubz+nx_, number_jacobian_nonzeros,
            jacobian_row_indices.data(), jacobian_column_indices.data(),
            UnoNlp::jacobian_wrapper);
      casadi_assert(ret, "uno_set_constraints failed");
    }

    ret = uno_set_lagrangian_hessian(model, number_hessian_nonzeros,
        hessian_triangular_part, hessian_row_indices.data(),
        hessian_column_indices.data(), UnoNlp::lagrangian_hessian_wrapper);
    casadi_assert(ret, "uno_set_lagrangian_hessian failed");

    ret = uno_set_lagrangian_sign_convention(model, lagrangian_sign_convention);
    casadi_assert(ret, "uno_set_lagrangian_sign_convention failed");

    ret = uno_set_initial_primal_iterate(model, d_nlp->x0);
    casadi_assert(ret, "uno_set_initial_primal_iterate failed");

    m->cb_exception = nullptr;
    uno_optimize(m->solver, model);
    if (m->cb_exception) std::rethrow_exception(m->cb_exception);

    uno_get_primal_solution(m->solver, d_nlp->z);
    uno_get_constraint_dual_solution(m->solver, d_nlp->lam+nx_);
    // lam_x neg when LBX active, pos when UBX active (Uno UNO_MULTIPLIER_POSITIVE).
    for (casadi_int i=0; i<nx_; ++i) {
      d_nlp->lam[i] = uno_get_lower_bound_dual_solution_component(m->solver, i)
                    - uno_get_upper_bound_dual_solution_component(m->solver, i);
    }

    m->return_status = return_status_string(m->solver);
    m->success = return_status_success(m->solver);
    m->unified_return_status = unified_status_from_uno(m->solver);
    d_nlp->objective = uno_get_solution_objective(m->solver);

    m->primal_infeasbility = uno_get_solution_primal_feasibility(m->solver);
    m->stationarity = uno_get_solution_stationarity(m->solver);
    m->complementarity = uno_get_solution_complementarity(m->solver);
    m->iter_count = uno_get_number_iterations(m->solver);

    return 0;
  }

  Dict UnoInterface::get_stats(void* mem) const {
    Dict stats = Nlpsol::get_stats(mem);
    auto m = static_cast<UnoMemory*>(mem);
    stats["return_status"] = m->return_status;
    stats["iter_count"] = m->iter_count;
    stats["primal_infeasbility"] = m->primal_infeasbility;
    stats["stationarity"] = m->stationarity;
    stats["complementarity"] = m->complementarity;
    
    return stats;
  }

} // namespace casadi