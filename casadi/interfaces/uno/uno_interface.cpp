/*
 *    This file is part of CasADi.
 *    Released under the LGPL; see LICENSE in the top-level directory.
 */

#include "uno_interface.hpp"
#include "casadi/core/casadi_misc.hpp"
#include "casadi/core/casadi_interrupt.hpp"
#include "casadi/core/code_generator.hpp"

#include <uno_runtime_str.h>

#include <cstdlib>
#include <limits>

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

  UnoInterface::UnoInterface(const std::string& name, const Function& nlp)
      : Nlpsol(name, nlp) {}

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

  UnoMemory::UnoMemory(const UnoInterface& uno_interface)
      : self(uno_interface), NlpsolMemory() {
    return_status = "Unset";
    d_uno.solver = nullptr;
    d_uno.model = nullptr;
  }

  UnoMemory::~UnoMemory() {
    casadi_uno_free_mem<double>(&d_uno);
  }

  void UnoInterface::free_mem(void* mem) const {
    delete static_cast<UnoMemory*>(mem);
  }

  void UnoInterface::init(const Dict& opts) {
    Nlpsol::init(opts);

    calc_f_ = true;
    calc_g_ = true;

    for (auto&& op : opts) {
      if (op.first == "uno") opts_ = op.second;
    }

    create_function("nlp_f", {"x", "p"}, {"f"});
    create_function("nlp_g", {"x", "p"}, {"g"});
    if (!has_function("nlp_grad_f")) {
      create_function("nlp_grad_f", {"x", "p"}, {"grad:f:x"});
    }
    Function gf_jg_fcn = create_function("nlp_jac_g", {"x", "p"}, {"jac:g:x"});
    jacg_sp_ = gf_jg_fcn.sparsity_out(0);

    Function hess_l_fcn = create_function("nlp_hess_l", {"x", "p", "lam:f", "lam:g"},
                                  {"triu:hess:gamma:x:x"},
                                  {{"gamma", {"f", "g"}}});
    hesslag_sp_ = hess_l_fcn.sparsity_out(0);
    casadi_assert(hesslag_sp_.is_triu(), "Hessian must be upper triangular");

    // Tweaking options for increased efficiency
    Dict final_options;
    final_options["is_diff_in"] = std::vector<bool>{true, false, false, false};
    final_options["is_diff_out"] = std::vector<bool>{true};
    Dict func_opts;
    func_opts["final_options"] = final_options;

    // Setup NLP Hessian product
    Function grad_hess_fcn = create_function("nlp_grad_l", {"x", "p", "lam:f", "lam:g"},
                                    {"grad:gamma:x"}, {{"gamma", {"f", "g"}}}, func_opts);

    // only differentiate wrt first argument, i.e., x
    // inputs:
    // "x", "p", "lam_f", "lam_g", "out_grad_gamma_x",
    // "fwd_x", "fwd_p", "fwd_lam_f", "fwd_lam_g"
    // outputs:
    // "fwd_grad_gamma_x"
    Function ret_hess_prod = create_forward("nlp_grad_l", 1);

    {
      auto jr = jacg_sp_.get_row(),    jc = jacg_sp_.get_col();
      auto hr = hesslag_sp_.get_row(), hc = hesslag_sp_.get_col();
      jacobian_row_indices_.assign(jr.begin(), jr.end());
      jacobian_column_indices_.assign(jc.begin(), jc.end());
      hessian_row_indices_.assign(hr.begin(), hr.end());
      hessian_column_indices_.assign(hc.begin(), hc.end());
    }

    placeholder_lb_g_.assign(ng_, -std::numeric_limits<double>::infinity());
    placeholder_ub_g_.assign(ng_,  std::numeric_limits<double>::infinity());

    set_uno_prob();
  }

  void UnoInterface::set_uno_prob() {
    // Mirror Nlpsol's p_nlp_ into our prob's p_nlp by-value -- so the C++ vm
    // and codegen paths agree on a single canonical location. Set_work then
    // points m->d_uno.nlp.prob at &p_uno_.nlp.
    p_uno_.nlp = p_nlp_;
    p_uno_.nx = static_cast<uno_int>(nx_);
    p_uno_.ng = static_cast<uno_int>(ng_);
    p_uno_.sp_a = jacg_sp_;
    p_uno_.sp_h = hesslag_sp_;
    p_uno_.jac_row  = jacobian_row_indices_.data();
    p_uno_.jac_col  = jacobian_column_indices_.data();
    p_uno_.n_jac    = static_cast<uno_int>(jacobian_row_indices_.size());
    p_uno_.hess_row = hessian_row_indices_.data();
    p_uno_.hess_col = hessian_column_indices_.data();
    p_uno_.n_hess   = static_cast<uno_int>(hessian_row_indices_.size());
    // Bind the C++-side exception-aware shims (UnoNlp::*_wrapper).
    p_uno_.obj_cb       = &UnoNlp::objective_function_wrapper;
    p_uno_.obj_grad_cb  = &UnoNlp::objective_gradient_wrapper;
    p_uno_.constr_cb    = &UnoNlp::constraint_functions_wrapper;
    p_uno_.jac_cb       = &UnoNlp::jacobian_wrapper;
    p_uno_.hess_cb      = &UnoNlp::lagrangian_hessian_wrapper;
    p_uno_.hess_prod_cb = &UnoNlp::lagrangian_hessian_product_wrapper;
    p_uno_.nlp_f      = OracleCallback("nlp_f", this);
    p_uno_.nlp_g      = OracleCallback("nlp_g", this);
    p_uno_.nlp_grad_f = OracleCallback("nlp_grad_f", this);
    p_uno_.nlp_jac_g  = OracleCallback("nlp_jac_g", this);
    p_uno_.nlp_hess_l = OracleCallback("nlp_hess_l", this);
    p_uno_.fwd1_nlp_grad_l = OracleCallback("fwd1_nlp_grad_l", this);
  }

  // C++ termination cb -- invokes Nlpsol::fcallback_ for opti.callback /
  // iteration callback, stashes thrown exceptions for solve() to rethrow.
  // (Runtime header registers a no-op cb; C++ init_mem overrides with this.)
  // Uno's C API impl checks ==0 for "terminate", non-zero for "continue".
  static uno_int casadi_uno_term_cb_cpp(uno_int, uno_int, const double* primals,
      const double* lower_mult, const double* upper_mult, const double* constraint_mult,
      double, double, double, double, void* user_data) {
    constexpr uno_int CONTINUE = 1;
    constexpr uno_int TERMINATE = 0;
    auto* m = static_cast<UnoMemory*>(user_data);  // wired in init_mem
    const UnoInterface& self = m->self;
    if (self.fcallback_.is_null()) return CONTINUE;

    auto* d_nlp = &m->d_nlp;
    casadi_copy(primals, self.nx_, d_nlp->z);
    for (casadi_int i = 0; i < self.nx_; ++i) {
      d_nlp->lam[i] = lower_mult[i] - upper_mult[i];
    }
    if (self.ng_ > 0) {
      casadi_copy(constraint_mult, self.ng_, d_nlp->lam + self.nx_);
    }
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
      UnoNlp::stash_pending_exception(std::current_exception());
      return TERMINATE;
    } catch (std::exception& ex) {
      casadi_warning(std::string("intermediate_callback: ") + ex.what());
      if (!self.iteration_callback_ignore_errors_) {
        UnoNlp::stash_pending_exception(std::current_exception());
        return TERMINATE;
      }
      return CONTINUE;
    }
    return static_cast<casadi_int>(ret_double) ? TERMINATE : CONTINUE;
  }

  int UnoInterface::init_mem(void* mem) const {
    if (Nlpsol::init_mem(mem)) return 1;
    auto m = static_cast<UnoMemory*>(mem);

    if (verbose_) {
      uno_int uno_major, uno_minor, uno_patch;
      uno_get_version(&uno_major, &uno_minor, &uno_patch);
      casadi_message("Using Uno v" + str(uno_major) + "." + str(uno_minor) + "." + str(uno_patch));
    }

    m->d_uno.prob = &p_uno_;
    casadi_uno_init_mem<double>(&m->d_uno);
    casadi_uno_init_model<double>(&m->d_uno,
        placeholder_lb_g_.data(), placeholder_ub_g_.data());
    // Override the runtime's no-op termination cb with one that fires
    // Nlpsol::fcallback_ each iteration (opti.callback support). Pass UnoMemory*
    // (not casadi_uno_data*) so the cb can reach m->self.fcallback_ etc.
    uno_set_solver_callbacks(m->d_uno.solver, nullptr, &casadi_uno_term_cb_cpp, m);
    UnoNlp::insert_casadi_options(m->d_uno.solver, opts_);
    return 0;
  }

  void UnoInterface::set_work(void* mem, const double**& arg, double**& res,
                              casadi_int*& iw, double*& w) const {
    Nlpsol::set_work(mem, arg, res, iw, w);
    auto m = static_cast<UnoMemory*>(mem);
    // Mirror NlpsolMemory's d_nlp into the by-value field on casadi_uno_data
    // (the runtime helpers and codegen path both read d->nlp.* there). Then
    // retarget the prob pointer at our own p_uno_.nlp -- the canonical
    // copy that the codegen path also references.
    m->d_uno.nlp = m->d_nlp;
    m->d_uno.nlp.prob = &p_uno_.nlp;
    // Wiring trap (cf casadi_nlpsol_plugin skill): the OracleCallback path
    // dispatches via cb->oracle_->calc_function(d->m, ...). Without this set,
    // d->m is NULL on the first eval and the plugin segfaults.
    m->d_nlp.oracle->m = static_cast<void*>(m);
    m->d_uno.nlp.oracle = m->d_nlp.oracle;
  }

  namespace {
    inline const char* return_status_string(int sol) {
      switch (sol) {
      case UNO_FEASIBLE_KKT_POINT:          return "Converged with feasible KKT point";
      case UNO_FEASIBLE_FJ_POINT:           return "Converged with feasible FJ point";
      case UNO_INFEASIBLE_STATIONARY_POINT: return "Converged with infeasible stationary point";
      case UNO_FEASIBLE_SMALL_STEP:         return "Terminated with feasible small step";
      case UNO_INFEASIBLE_SMALL_STEP:       return "Terminated with infeasible small step";
      case UNO_UNBOUNDED:                   return "Terminated with unbounded problem";
      case UNO_NOT_OPTIMAL:                 return "Terminated with not optimal point";
      default:                              return "Terminated with an unknown status";
      }
    }
  }

  int UnoInterface::solve(void* mem) const {
    auto m = static_cast<UnoMemory*>(mem);
    UnoNlp::clear_pending_exception();

    casadi_uno_solve<double>(&m->d_uno);

    if (auto p = UnoNlp::take_pending_exception()) std::rethrow_exception(p);

    m->success = m->d_uno.success;
    m->unified_return_status = static_cast<UnifiedReturnStatus>(m->d_uno.unified_return_status);
    m->return_status = return_status_string(uno_get_solution_status(m->d_uno.solver));
    return 0;
  }

  Dict UnoInterface::get_stats(void* mem) const {
    Dict stats = Nlpsol::get_stats(mem);
    auto m = static_cast<UnoMemory*>(mem);
    stats["return_status"]       = m->return_status;
    stats["iter_count"]          = static_cast<casadi_int>(m->d_uno.iter_count);
    stats["primal_infeasbility"] = m->d_uno.primal_infeasibility;
    stats["stationarity"]        = m->d_uno.stationarity;
    stats["complementarity"]     = m->d_uno.complementarity;
    return stats;
  }

  void UnoInterface::serialize_body(SerializingStream& s) const {
    Nlpsol::serialize_body(s);
    s.version("UnoInterface", 1);
    s.pack("UnoInterface::jacg_sp",    jacg_sp_);
    s.pack("UnoInterface::hesslag_sp", hesslag_sp_);
    s.pack("UnoInterface::opts",       opts_);
  }

  UnoInterface::UnoInterface(DeserializingStream& s) : Nlpsol(s) {
    s.version("UnoInterface", 1);
    s.unpack("UnoInterface::jacg_sp",    jacg_sp_);
    s.unpack("UnoInterface::hesslag_sp", hesslag_sp_);
    s.unpack("UnoInterface::opts",       opts_);
    auto jr = jacg_sp_.get_row(),    jc = jacg_sp_.get_col();
    auto hr = hesslag_sp_.get_row(), hc = hesslag_sp_.get_col();
    jacobian_row_indices_.assign(jr.begin(), jr.end());
    jacobian_column_indices_.assign(jc.begin(), jc.end());
    hessian_row_indices_.assign(hr.begin(), hr.end());
    hessian_column_indices_.assign(hc.begin(), hc.end());
    placeholder_lb_g_.assign(ng_, -std::numeric_limits<double>::infinity());
    placeholder_ub_g_.assign(ng_,  std::numeric_limits<double>::infinity());
    set_uno_prob();
  }

  // ----- Codegen --------------------------------------------------------------

  void UnoInterface::codegen_init_mem(CodeGenerator& g) const {
    g.local("d", "struct casadi_uno_data*");
    g.init_local("d", "&" + codegen_mem(g));
    // Static prob: nx/ng + sparsity + callbacks are problem-invariant, so
    // the prob struct lives forever (one per generated function). Storing
    // it function-scope-static lets us call uno_create_model from init_mem,
    // matching the C++ vm path's "build everything at allocation time".
    g.local("p", "static struct casadi_uno_prob");
    set_uno_prob(g);
    g << "d->prob = &p;\n";
    // Wire d->nlp at the persistent NLP scratch on this memory block.
    // (Casadi convention has d_nlp/p_nlp/d_oracle as per-call function-scope
    // locals via Nlpsol::codegen_body_enter; uno opts out of that and uses
    // the by-value fields on casadi_uno_data instead.)
    g << "\n";
    Nlpsol::codegen_setup_constants(g, "d->nlp", "p.nlp", "d->d_oracle");
    g << "casadi_uno_init_mem(d);\n";
    g << "casadi_uno_init_model(d, "
      << g.constant(placeholder_lb_g_) << ", "
      << g.constant(placeholder_ub_g_) << ");\n";
    // Apply user options (statically known at codegen time).
    for (auto&& kv : opts_) {
      const std::string& key = kv.first;
      if (key == "preset") {
        g << "uno_set_solver_preset(d->solver, \"" << kv.second.to_string() << "\");\n";
      } else if (kv.second.is_bool()) {
        g << "uno_set_solver_bool_option(d->solver, \"" << key << "\", "
          << (kv.second.to_bool() ? "1" : "0") << ");\n";
      } else if (kv.second.is_int()) {
        g << "uno_set_solver_integer_option(d->solver, \"" << key << "\", "
          << kv.second.to_int() << ");\n";
      } else if (kv.second.is_double()) {
        g << "uno_set_solver_double_option(d->solver, \"" << key << "\", "
          << g.constant(kv.second.to_double()) << ");\n";
      } else if (kv.second.is_string()) {
        g << "uno_set_solver_string_option(d->solver, \"" << key << "\", \""
          << kv.second.to_string() << "\");\n";
      } else {
        casadi_error("Unsupported uno option type for '" + key + "'");
      }
    }
    g << "return 0;\n";
  }

  void UnoInterface::codegen_free_mem(CodeGenerator& g) const {
    g << "casadi_uno_free_mem(&" + codegen_mem(g) + ");\n";
  }

  void UnoInterface::codegen_declarations(CodeGenerator& g) const {
    Nlpsol::codegen_declarations(g);
    g.add_auxiliary(CodeGenerator::AUX_NLP);
    g.add_auxiliary(CodeGenerator::AUX_ORACLE_CALLBACK);
    g.add_auxiliary(CodeGenerator::AUX_INF);
    g.add_dependency(get_function("nlp_f"));
    g.add_dependency(get_function("nlp_g"));
    g.add_dependency(get_function("nlp_grad_f"));
    g.add_dependency(get_function("nlp_jac_g"));
    g.add_dependency(get_function("nlp_hess_l"));
    g.add_dependency(get_function("fwd1_nlp_grad_l"));
    g.add_include("Uno_C_API.h");
    g.auxiliaries << g.sanitize_source(uno_runtime_str, {"casadi_real"});
  }

  void UnoInterface::set_uno_prob(CodeGenerator& g) const {
    g << "p.nx       = " << nx_ << ";\n";
    g << "p.ng       = " << ng_ << ";\n";
    g << "p.sp_a     = " << g.sparsity(jacg_sp_)    << ";\n";
    g << "p.sp_h     = " << g.sparsity(hesslag_sp_) << ";\n";
    // Uno wants uno_int (32-bit) row/col arrays; g.constant emits casadi_int
    // (64-bit when WITH_LONGLONG_CORE), so use constant_copy with type "int"
    // to produce a typed local that matches. Guard the empty-vector case
    // (constant_copy emits an unused-variable that trips -Werror).
    auto emit_int_array = [&g](const std::string& name,
        const std::vector<uno_int>& v) {
      if (v.empty()) {
        g << "p." << name << " = 0;\n";
      } else {
        g.constant_copy("p_" + name, vector_static_cast<casadi_int>(v), "int");
        g << "p." << name << " = p_" << name << ";\n";
      }
    };
    emit_int_array("jac_row",  jacobian_row_indices_);
    emit_int_array("jac_col",  jacobian_column_indices_);
    g << "p.n_jac    = " << jacobian_row_indices_.size() << ";\n";
    emit_int_array("hess_row", hessian_row_indices_);
    emit_int_array("hess_col", hessian_column_indices_);
    g << "p.n_hess   = " << hessian_row_indices_.size() << ";\n";
    g.setup_callback("p.nlp_f",      get_function("nlp_f"));
    g.setup_callback("p.nlp_g",      get_function("nlp_g"));
    g.setup_callback("p.nlp_grad_f", get_function("nlp_grad_f"));
    g.setup_callback("p.nlp_jac_g",  get_function("nlp_jac_g"));
    g.setup_callback("p.nlp_hess_l", get_function("nlp_hess_l"));
    g.setup_callback("p.fwd1_nlp_grad_l", get_function("fwd1_nlp_grad_l"));
    // Codegen path uses the runtime wrappers directly (no exception handling
    // -- the C++ shim is an LGPL-side luxury, not available in pure C).
    g << "p.obj_cb       = &casadi_uno_obj_wrapper;\n";
    g << "p.obj_grad_cb  = &casadi_uno_obj_grad_wrapper;\n";
    g << "p.constr_cb    = &casadi_uno_constr_wrapper;\n";
    g << "p.jac_cb       = &casadi_uno_jac_wrapper;\n";
    g << "p.hess_cb      = &casadi_uno_hess_wrapper;\n";
    g << "p.hess_prod_cb = &casadi_uno_hess_prod_wrapper;\n";
  }

  void UnoInterface::codegen_body(CodeGenerator& g) const {
    // No codegen_body_enter / codegen_body_exit: d_nlp / p_nlp / d_oracle
    // live on casadi_uno_data, populated from codegen_init_mem (constants)
    // and the codegen_setup_per_call call below (per-call wiring).
    g.local("d", "struct casadi_uno_data*");
    g.init_local("d", "&" + codegen_mem(g));
    Nlpsol::codegen_setup_per_call(g, "d->nlp");
    g << "casadi_uno_init(d, &arg, &res, &iw, &w);\n";
    g << "casadi_oracle_init(&d->d_oracle, &arg, &res, &iw, &w);\n";
    g << "casadi_uno_solve(d);\n";
    Nlpsol::codegen_post_solve(g, "d->nlp");
    g << "return 0;\n";
  }

}  // namespace casadi
