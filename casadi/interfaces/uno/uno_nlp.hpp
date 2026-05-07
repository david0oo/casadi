/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2026 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            KU Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    Released under the LGPL; see the file LICENSE in the top-level directory.
 */

#ifndef CASADI_UNO_NLP_HPP
#define CASADI_UNO_NLP_HPP

#include "Uno_C_API.h"
#include "casadi/core/generic_type.hpp"
#include <exception>
#include <string>

namespace casadi {

// Exception-aware C++ shims around the runtime callback wrappers.
// uno_set_objective / uno_set_constraints / uno_set_lagrangian_hessian on the
// C++ vm path point at these (one extra try/catch around the runtime
// casadi_uno_*_wrapper). The codegen path points uno_set_* at the runtime
// wrappers directly -- no try/catch in pure C.
namespace UnoNlp {

  uno_int objective_function_wrapper(uno_int n, const double* x,
      double* objective_value, void* user_data);
  uno_int objective_gradient_wrapper(uno_int n, const double* x,
      double* gradient, void* user_data);
  uno_int constraint_functions_wrapper(uno_int n, uno_int ng, const double* x,
      double* constraint_values, void* user_data);
  uno_int jacobian_wrapper(uno_int n, uno_int nnz, const double* x,
      double* jacobian_values, void* user_data);
  uno_int lagrangian_hessian_wrapper(uno_int n, uno_int ng, uno_int nnz,
      const double* x, double objective_multiplier, const double* multipliers,
      double* hessian_values, void* user_data);

  void set_uno_option(void* solver, const std::string& name, const GenericType& value);
  void insert_casadi_options(void* solver, Dict opts);

  // Returns the most recent KeyboardInterrupt/std::exception caught by any of
  // the wrappers above on the calling thread, or nullptr if there hasn't been
  // one. Cleared by clear_pending_exception().
  std::exception_ptr take_pending_exception();
  void clear_pending_exception();
  void stash_pending_exception(std::exception_ptr p);

}  // namespace UnoNlp

}  // namespace casadi

#endif  // CASADI_UNO_NLP_HPP
