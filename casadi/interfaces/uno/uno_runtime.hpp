//
//    MIT No Attribution
//
//    Copyright (C) 2010-2026 Joel Andersson, Joris Gillis, Moritz Diehl, KU Leuven.
//
//    Permission is hereby granted, free of charge, to any person obtaining a copy of this
//    software and associated documentation files (the "Software"), to deal in the Software
//    without restriction, including without limitation the rights to use, copy, modify,
//    merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
//    permit persons to whom the Software is furnished to do so.
//

// C-REPLACE "casadi_nlpsol_prob<T1>" "struct casadi_nlpsol_prob"
// C-REPLACE "casadi_nlpsol_data<T1>" "struct casadi_nlpsol_data"
// C-REPLACE "casadi_oracle_data<T1>" "struct casadi_oracle_data"
// C-REPLACE "OracleCallback" "struct casadi_oracle_callback"
// C-REPLACE "calc_function" "casadi_oracle_call"
// The earlier C-REPLACE turns casadi_uno_data<T1> -> "struct casadi_uno_data"
// first; the resulting cast looks like "static_cast< struct casadi_uno_data* >".
// C-REPLACE "static_cast< struct casadi_uno_data* >" "(struct casadi_uno_data*)"
// C-REPLACE "SOLVER_RET_SUCCESS" "0"
// C-REPLACE "SOLVER_RET_UNKNOWN" "1"
// C-REPLACE "SOLVER_RET_LIMITED" "2"
// C-REPLACE "SOLVER_RET_NAN" "3"
// C-REPLACE "SOLVER_RET_INFEASIBLE" "4"
// C-REPLACE "SOLVER_RET_EXCEPTION" "5"

// SYMBOL "uno_prob"
template<typename T1>
struct casadi_uno_prob {
  // p_nlp lives here (not on casadi_uno_data) because it's per-Function-
  // instance constant -- nx / ng / np / detect_bounds.* don't change between
  // memory blocks. d->nlp.prob points at this field via the codegen_setup_
  // constants emission (or via a manual override in C++ set_work).
  casadi_nlpsol_prob<T1> p_nlp;
  // nx / ng duplicated outside p_nlp as uno_int (32-bit) so the runtime
  // helpers don't have to chain through nlp->nx (which would force the
  // codegen prob to depend on the per-call p_nlp local).
  uno_int nx;
  uno_int ng;
  const casadi_int* sp_a;
  const casadi_int* sp_h;
  const uno_int* jac_row;
  const uno_int* jac_col;
  uno_int n_jac;
  const uno_int* hess_row;
  const uno_int* hess_col;
  uno_int n_hess;
  OracleCallback nlp_f;
  OracleCallback nlp_g;
  OracleCallback nlp_grad_f;
  OracleCallback nlp_jac_g;
  OracleCallback nlp_hess_l;
  // Function pointers to register with Uno (kept on prob so the C++ vm path
  // can pass exception-aware shims and the codegen path can pass the bare
  // casadi_uno_*_wrapper symbols below).
  uno_objective_callback              obj_cb;
  uno_objective_gradient_callback     obj_grad_cb;
  uno_constraints_callback            constr_cb;
  uno_constraints_jacobian_callback   jac_cb;
  uno_lagrangian_hessian_callback     hess_cb;
};
// C-REPLACE "casadi_uno_prob<T1>" "struct casadi_uno_prob"

// SYMBOL "uno_data"
template<typename T1>
struct casadi_uno_data {
  const casadi_uno_prob<T1>* prob;
  void* solver;
  void* model;
  // Fed by casadi_uno_init() each call so codegen_body_exit's post-solve
  // nlp_grad call can find them (matches fatrop's data layout).
  const T1** arg;
  T1** res;
  casadi_int* iw;
  T1* w;
  int unified_return_status;
  int success;
  T1 primal_infeasibility;
  T1 stationarity;
  T1 complementarity;
  uno_int iter_count;
  // Persistent NLP scratch as by-value fields. The codegen path skips
  // Nlpsol::codegen_body_enter (which would otherwise emit these as
  // per-call function-scope locals) and writes through these instead; the
  // C++ vm path mirrors NlpsolMemory's d_nlp into nlp at set_work time.
  // (p_nlp lives on casadi_uno_prob -- it's per-Function-constant.)
  casadi_nlpsol_data<T1> nlp;
  casadi_oracle_data<T1> d_oracle;
};
// C-REPLACE "casadi_uno_data<T1>" "struct casadi_uno_data"

// SYMBOL "uno_obj_wrapper"
template<typename T1>
uno_int casadi_uno_obj_wrapper(uno_int n, const T1* x, T1* fval, void* user_data) {
  casadi_uno_data<T1>* d = static_cast< casadi_uno_data<T1>* >(user_data);
  casadi_oracle_data<T1>* d_oracle = d->nlp.oracle;
  d_oracle->arg[0] = x;
  d_oracle->arg[1] = d->nlp.p;
  d_oracle->res[0] = fval;
  return calc_function(&d->prob->nlp_f, d_oracle) == 0 ? 0 : 1;
}
// C-REPLACE "casadi_uno_obj_wrapper<T1>" "casadi_uno_obj_wrapper"

// SYMBOL "uno_obj_grad_wrapper"
template<typename T1>
uno_int casadi_uno_obj_grad_wrapper(uno_int n, const T1* x, T1* grad, void* user_data) {
  casadi_uno_data<T1>* d = static_cast< casadi_uno_data<T1>* >(user_data);
  casadi_oracle_data<T1>* d_oracle = d->nlp.oracle;
  d_oracle->arg[0] = x;
  d_oracle->arg[1] = d->nlp.p;
  d_oracle->res[0] = grad;
  return calc_function(&d->prob->nlp_grad_f, d_oracle) == 0 ? 0 : 1;
}
// C-REPLACE "casadi_uno_obj_grad_wrapper<T1>" "casadi_uno_obj_grad_wrapper"

// SYMBOL "uno_constr_wrapper"
template<typename T1>
uno_int casadi_uno_constr_wrapper(uno_int n, uno_int ng, const T1* x, T1* gval, void* user_data) {
  casadi_uno_data<T1>* d = static_cast< casadi_uno_data<T1>* >(user_data);
  casadi_oracle_data<T1>* d_oracle = d->nlp.oracle;
  d_oracle->arg[0] = x;
  d_oracle->arg[1] = d->nlp.p;
  d_oracle->res[0] = gval;
  return calc_function(&d->prob->nlp_g, d_oracle) == 0 ? 0 : 1;
}
// C-REPLACE "casadi_uno_constr_wrapper<T1>" "casadi_uno_constr_wrapper"

// SYMBOL "uno_jac_wrapper"
template<typename T1>
uno_int casadi_uno_jac_wrapper(uno_int n, uno_int nnz, const T1* x, T1* jvals, void* user_data) {
  casadi_uno_data<T1>* d = static_cast< casadi_uno_data<T1>* >(user_data);
  casadi_oracle_data<T1>* d_oracle = d->nlp.oracle;
  d_oracle->arg[0] = x;
  d_oracle->arg[1] = d->nlp.p;
  d_oracle->res[0] = jvals;
  return calc_function(&d->prob->nlp_jac_g, d_oracle) == 0 ? 0 : 1;
}
// C-REPLACE "casadi_uno_jac_wrapper<T1>" "casadi_uno_jac_wrapper"

// SYMBOL "uno_hess_wrapper"
template<typename T1>
uno_int casadi_uno_hess_wrapper(uno_int n, uno_int ng, uno_int nnz,
    const T1* x, T1 obj_mult, const T1* mults, T1* hvals, void* user_data) {
  casadi_uno_data<T1>* d = static_cast< casadi_uno_data<T1>* >(user_data);
  casadi_oracle_data<T1>* d_oracle = d->nlp.oracle;
  d_oracle->arg[0] = x;
  d_oracle->arg[1] = d->nlp.p;
  d_oracle->arg[2] = &obj_mult;
  d_oracle->arg[3] = mults;
  d_oracle->res[0] = hvals;
  return calc_function(&d->prob->nlp_hess_l, d_oracle) == 0 ? 0 : 1;
}
// C-REPLACE "casadi_uno_hess_wrapper<T1>" "casadi_uno_hess_wrapper"

// Termination cb that always continues. Uno's C API impl checks ==0 for terminate.
// SYMBOL "uno_term_cb"
template<typename T1>
uno_int casadi_uno_term_cb(uno_int n, uno_int ng, const T1* primals,
    const T1* lower_mult, const T1* upper_mult, const T1* constr_mult,
    T1 obj_mult, T1 inf_pr, T1 inf_du, T1 compl_res, void* user_data) {
  return 1;
}
// C-REPLACE "casadi_uno_term_cb<T1>" "casadi_uno_term_cb"

// One-shot, called from init_mem: just create the solver + register the
// no-op termination cb. Cheap. uno_create_solver is the heavyweight part
// (instantiates Uno's strategy combination, factory state, etc.) -- doing
// it once per memory block is a deliberate design choice, not lazy-init.
// SYMBOL "uno_init_mem"
template<typename T1>
int casadi_uno_init_mem(casadi_uno_data<T1>* d) {
  d->solver = uno_create_solver();
  uno_set_solver_callbacks(d->solver, 0, &casadi_uno_term_cb<T1>, d);
  d->model = 0;
  d->unified_return_status = SOLVER_RET_UNKNOWN;
  d->success = 0;
  return 0;
}

// Build the Uno model + bind every callback. Called from C++ init_mem
// (with placeholder bounds), or lazily from casadi_uno_solve in the
// codegen path. uno_create_model copies the bounds it gets, so the
// placeholder buffers can be throwaway.
// SYMBOL "uno_init_model"
template<typename T1>
void casadi_uno_init_model(casadi_uno_data<T1>* d,
    const T1* lb_x, const T1* ub_x, const T1* lb_g, const T1* ub_g) {
  const casadi_uno_prob<T1>* p = d->prob;
  uno_int nx = p->nx;
  uno_int ng = p->ng;

  d->model = uno_create_model(UNO_PROBLEM_NONLINEAR, nx,
      lb_x, ub_x, UNO_ZERO_BASED_INDEXING);
  uno_set_user_data(d->model, d);
  uno_set_objective(d->model, UNO_MINIMIZE, p->obj_cb, p->obj_grad_cb);
  if (ng > 0) {
    uno_set_constraints(d->model, ng, p->constr_cb, lb_g, ub_g,
        p->n_jac, p->jac_row, p->jac_col, p->jac_cb);
  }
  uno_set_lagrangian_hessian(d->model, p->n_hess, UNO_UPPER_TRIANGLE,
      p->hess_row, p->hess_col, p->hess_cb);
  uno_set_lagrangian_sign_convention(d->model, UNO_MULTIPLIER_POSITIVE);
}

// SYMBOL "uno_free_mem"
template<typename T1>
void casadi_uno_free_mem(casadi_uno_data<T1>* d) {
  if (d->model)  uno_destroy_model(d->model);
  if (d->solver) uno_destroy_solver(d->solver);
  d->model = 0;
  d->solver = 0;
}

// Per-call wiring: stash arg/res/iw/w on the data struct so codegen_body_exit
// (which post-solve fires nlp_grad to fill NLPSOL_F / NLPSOL_G) finds them.
// SYMBOL "uno_init"
template<typename T1>
void casadi_uno_init(casadi_uno_data<T1>* d, const T1*** arg, T1*** res,
                     casadi_int** iw, T1** w) {
  d->arg = *arg;
  d->res = *res;
  d->iw = *iw;
  d->w = *w;
}

// Per-call: build the model on the first call (codegen path), patch bounds +
// initial iterate, optimize, extract.
// SYMBOL "uno_solve"
template<typename T1>
void casadi_uno_solve(casadi_uno_data<T1>* d) {
  const casadi_uno_prob<T1>* p = d->prob;
  casadi_nlpsol_data<T1>* d_nlp = &d->nlp;
  uno_int nx = p->nx;
  uno_int ng = p->ng;
  casadi_int i;

  if (!d->model) {
    // Codegen path: model wasn't built in init_mem (no p_nlp scope there);
    // build it now using the real bounds from this first call.
    casadi_uno_init_model(d, d_nlp->lbz, d_nlp->ubz,
        d_nlp->lbz + nx, d_nlp->ubz + nx);
  } else {
    uno_set_variables_lower_bounds(d->model, d_nlp->lbz);
    uno_set_variables_upper_bounds(d->model, d_nlp->ubz);
    if (ng > 0) {
      uno_set_constraints_lower_bounds(d->model, d_nlp->lbz + nx);
      uno_set_constraints_upper_bounds(d->model, d_nlp->ubz + nx);
    }
  }
  uno_set_initial_primal_iterate(d->model, d_nlp->x0);

  uno_optimize(d->solver, d->model);

  uno_get_primal_solution(d->solver, d_nlp->z);
  uno_get_constraint_dual_solution(d->solver, d_nlp->lam + nx);
  for (i = 0; i < nx; ++i) {
    d_nlp->lam[i] = uno_get_lower_bound_dual_solution_component(d->solver, i)
                  - uno_get_upper_bound_dual_solution_component(d->solver, i);
  }
  d_nlp->objective         = uno_get_solution_objective(d->solver);
  d->primal_infeasibility  = uno_get_solution_primal_feasibility(d->solver);
  d->stationarity          = uno_get_solution_stationarity(d->solver);
  d->complementarity       = uno_get_solution_complementarity(d->solver);
  d->iter_count            = uno_get_number_iterations(d->solver);

  uno_int opt = uno_get_optimization_status(d->solver);
  uno_int sol = uno_get_solution_status(d->solver);
  d->success = (opt == UNO_SUCCESS && sol == UNO_FEASIBLE_KKT_POINT) ? 1 : 0;
  if      (opt == UNO_EVALUATION_ERROR)            d->unified_return_status = SOLVER_RET_NAN;
  else if (opt == UNO_ITERATION_LIMIT)             d->unified_return_status = SOLVER_RET_LIMITED;
  else if (opt == UNO_TIME_LIMIT)                  d->unified_return_status = SOLVER_RET_LIMITED;
  else if (opt == UNO_ALGORITHMIC_ERROR)           d->unified_return_status = SOLVER_RET_EXCEPTION;
  else if (opt == UNO_USER_TERMINATION)            d->unified_return_status = SOLVER_RET_UNKNOWN;
  else if (sol == UNO_FEASIBLE_KKT_POINT)          d->unified_return_status = SOLVER_RET_SUCCESS;
  else if (sol == UNO_FEASIBLE_FJ_POINT)           d->unified_return_status = SOLVER_RET_SUCCESS;
  else if (sol == UNO_INFEASIBLE_STATIONARY_POINT) d->unified_return_status = SOLVER_RET_INFEASIBLE;
  else if (sol == UNO_INFEASIBLE_SMALL_STEP)       d->unified_return_status = SOLVER_RET_INFEASIBLE;
  else if (sol == UNO_FEASIBLE_SMALL_STEP)         d->unified_return_status = SOLVER_RET_LIMITED;
  else                                             d->unified_return_status = SOLVER_RET_UNKNOWN;
}
