#include "uno_nlp.hpp"
#include "uno_interface.hpp"
#include "casadi/core/casadi_interrupt.hpp"

namespace casadi {

// Uno C API callbacks: 0 == success, positive == evaluation failure.
// Don't let exceptions out (Uno catches and reports UNO_ALGORITHMIC_ERROR);
// emit a casadi_warning with the message (so user-visible stderr keeps
// working) and return non-zero so Uno reports UNO_EVALUATION_ERROR.
// Re-throw KeyboardInterruptException so Ctrl-C / iteration callbacks
// still propagate.
#define UNO_GUARD(EXPR, NAME) do {                                              \
  try { return (EXPR) ? 1 : 0; }                                                \
  catch (KeyboardInterruptException&) { throw; }                                \
  catch (std::exception& ex) {                                                  \
    casadi_warning(std::string(NAME) + ": " + ex.what());                       \
    return 1;                                                                   \
  }                                                                             \
} while (0)

UnoNlp::UnoNlp(UnoMemory* mem) : mem_(mem) {}

uno_int UnoNlp::objective_function(const double* x, double* objective_value) {
  mem_->arg[0] = x;
  mem_->arg[1] = mem_->d_nlp.p;
  mem_->res[0] = objective_value;
  UNO_GUARD(mem_->self.calc_function(mem_, "nlp_f") != 0, "nlp_f");
}

uno_int UnoNlp::objective_function_wrapper(uno_int, const double* x,
    double* objective_value, void* user_data) {
  return static_cast<UnoNlp*>(user_data)->objective_function(x, objective_value);
}

uno_int UnoNlp::constraint_functions(const double* x, double* constraint_values) {
  mem_->arg[0] = x;
  mem_->arg[1] = mem_->d_nlp.p;
  mem_->res[0] = constraint_values;
  UNO_GUARD(mem_->self.calc_function(mem_, "nlp_g") != 0, "nlp_g");
}

uno_int UnoNlp::constraint_functions_wrapper(uno_int, uno_int, const double* x,
    double* constraint_values, void* user_data) {
  return static_cast<UnoNlp*>(user_data)->constraint_functions(x, constraint_values);
}

uno_int UnoNlp::objective_gradient(const double* x, double* gradient) {
  mem_->arg[0] = x;
  mem_->arg[1] = mem_->d_nlp.p;
  mem_->res[0] = gradient;
  UNO_GUARD(mem_->self.calc_function(mem_, "nlp_grad_f") != 0, "nlp_grad_f");
}

uno_int UnoNlp::objective_gradient_wrapper(uno_int, const double* x,
    double* gradient, void* user_data) {
  return static_cast<UnoNlp*>(user_data)->objective_gradient(x, gradient);
}

uno_int UnoNlp::jacobian(const double* x, double* jacobian_values) {
  mem_->arg[0] = x;
  mem_->arg[1] = mem_->d_nlp.p;
  mem_->res[0] = jacobian_values;
  UNO_GUARD(mem_->self.calc_function(mem_, "nlp_jac_g") != 0, "nlp_jac_g");
}

uno_int UnoNlp::jacobian_wrapper(uno_int, uno_int, const double* x,
    double* jacobian_values, void* user_data) {
  return static_cast<UnoNlp*>(user_data)->jacobian(x, jacobian_values);
}

uno_int UnoNlp::lagrangian_hessian(const double* x, double objective_multiplier,
    const double* multipliers, double* hessian_values) {
  mem_->arg[0] = x;
  mem_->arg[1] = mem_->d_nlp.p;
  mem_->arg[2] = &objective_multiplier;
  mem_->arg[3] = multipliers;
  mem_->res[0] = hessian_values;
  UNO_GUARD(mem_->self.calc_function(mem_, "nlp_hess_l") != 0, "nlp_hess_l");
}

uno_int UnoNlp::lagrangian_hessian_wrapper(uno_int, uno_int, uno_int,
    const double* x, double objective_multiplier, const double* multipliers,
    double* hessian_values, void* user_data) {
  return static_cast<UnoNlp*>(user_data)
      ->lagrangian_hessian(x, objective_multiplier, multipliers, hessian_values);
}

void UnoNlp::set_uno_option(void* solver, const std::string& name,
    const GenericType& value) {
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

void UnoNlp::insert_casadi_options(void* solver, Dict opts) {
  Dict casadi_options = Options::sanitize(opts);
  for (auto&& op : casadi_options) {
    if (op.first == "preset") {
      uno_set_solver_preset(solver, op.second.to_string().c_str());
    } else {
      set_uno_option(solver, op.first, op.second);
    }
  }
}

}  // namespace casadi
