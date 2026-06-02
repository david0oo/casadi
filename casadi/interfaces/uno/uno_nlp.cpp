#include "uno_nlp.hpp"
#include "uno_interface.hpp"
#include "casadi/core/casadi_interrupt.hpp"

namespace casadi {

// The runtime helpers live in uno_runtime.hpp (included into the casadi::
// namespace from uno_interface.hpp). The C++ side instantiates them with
// <double>; the codegen-emitted C calls them after C-REPLACE strips the
// template suffix. The shims below just wrap the runtime wrappers in
// try/catch so KeyboardInterrupt / std::exception can re-throw later.

namespace UnoNlp {
namespace {

thread_local std::exception_ptr s_pending_exception;

// Catches the user's exception, stashes it for solve() to rethrow, and returns
// non-zero to Uno (= UNO_EVALUATION_ERROR) so the solver bails cleanly instead
// of being caught as ALGORITHMIC_ERROR.
#define UNO_GUARD(EXPR, NAME) do {                                              \
  try { return (EXPR) ? 1 : 0; }                                                \
  catch (KeyboardInterruptException&) {                                         \
    s_pending_exception = std::current_exception();                             \
    return 1;                                                                   \
  }                                                                             \
  catch (std::exception& ex) {                                                  \
    casadi_warning(std::string(NAME) + ": " + ex.what());                       \
    s_pending_exception = std::current_exception();                             \
    return 1;                                                                   \
  }                                                                             \
} while (0)

}  // namespace

uno_int objective_function_wrapper(uno_int n, const double* x,
    double* objective_value, void* user_data) {
  UNO_GUARD(casadi_uno_obj_wrapper<double>(n, x, objective_value, user_data) != 0, "nlp_f");
}

uno_int objective_gradient_wrapper(uno_int n, const double* x,
    double* gradient, void* user_data) {
  UNO_GUARD(casadi_uno_obj_grad_wrapper<double>(n, x, gradient, user_data) != 0, "nlp_grad_f");
}

uno_int constraint_functions_wrapper(uno_int n, uno_int ng, const double* x,
    double* constraint_values, void* user_data) {
  UNO_GUARD(casadi_uno_constr_wrapper<double>(n, ng, x, constraint_values, user_data) != 0,
      "nlp_g");
}

uno_int jacobian_wrapper(uno_int n, uno_int nnz, const double* x,
    double* jacobian_values, void* user_data) {
  UNO_GUARD(casadi_uno_jac_wrapper<double>(n, nnz, x, jacobian_values, user_data) != 0,
      "nlp_jac_g");
}

uno_int lagrangian_hessian_wrapper(uno_int n, uno_int ng, uno_int nnz,
    const double* x, double objective_multiplier, const double* multipliers,
    double* hessian_values, void* user_data) {
  UNO_GUARD(casadi_uno_hess_wrapper<double>(n, ng, nnz, x, objective_multiplier,
      multipliers, hessian_values, user_data) != 0, "nlp_hess_l");
}

uno_int lagrangian_hessian_product_wrapper(uno_int n, uno_int ng, const double* x, 
  bool evaluate_at_x, double objective_multiplier, const double* multipliers, const double* vector,
    double* result, void* user_data) {
  UNO_GUARD(casadi_uno_hess_prod_wrapper<double>(n, ng, x, evaluate_at_x, objective_multiplier,
      multipliers, vector, result, user_data) != 0, "fwd1_nlp_grad_l");
}

void set_uno_option(void* solver, const std::string& name, const GenericType& value) {
  if (value.is_bool()) {
    uno_set_solver_bool_option(solver, name.c_str(), value.to_bool());
  } else if (value.is_int()) {
    uno_set_solver_integer_option(solver, name.c_str(), static_cast<uno_int>(value.to_int()));
  } else if (value.is_double()) {
    uno_set_solver_double_option(solver, name.c_str(), value.to_double());
  } else if (value.is_string()) {
    uno_set_solver_string_option(solver, name.c_str(), value.to_string().c_str());
  } else {
    casadi_assert(false, "Unsupported UNO option type for " + name);
  }
}

void insert_casadi_options(void* solver, Dict opts) {
  Dict casadi_options = Options::sanitize(opts);
  // We need to split this up since the preset overwrites some options.
  // e.g., when we want L-BFGS, the preset would overwrite the Hessian model to
  // exact Hessian 
  for (auto&& op : casadi_options) {
    if (op.first == "preset") {
      uno_set_solver_preset(solver, op.second.to_string().c_str());
      break;
    }
  }
  for (auto&& op : casadi_options) {
    if (op.first != "preset") {
      set_uno_option(solver, op.first, op.second);
    }
  }
}

std::exception_ptr take_pending_exception() {
  std::exception_ptr p = s_pending_exception;
  s_pending_exception = nullptr;
  return p;
}

void clear_pending_exception() {
  s_pending_exception = nullptr;
}

void stash_pending_exception(std::exception_ptr p) {
  s_pending_exception = p;
}

}  // namespace UnoNlp
}  // namespace casadi
