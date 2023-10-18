/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2023 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            KU Leuven. All rights reserved.
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


#include "idas_interface.hpp"
#include "casadi/core/casadi_misc.hpp"

// Macro for error handling
#define THROWING(fcn, ...) \
idas_error(CASADI_STR(fcn), fcn(__VA_ARGS__))

namespace casadi {

extern "C"
int CASADI_INTEGRATOR_IDAS_EXPORT
    casadi_register_integrator_idas(Integrator::Plugin* plugin) {
  plugin->creator = IdasInterface::creator;
  plugin->name = "idas";
  plugin->doc = IdasInterface::meta_doc.c_str();
  plugin->version = CASADI_VERSION;
  plugin->options = &IdasInterface::options_;
  plugin->deserialize = &IdasInterface::deserialize;
  return 0;
}

extern "C"
void CASADI_INTEGRATOR_IDAS_EXPORT casadi_load_integrator_idas() {
  Integrator::registerPlugin(casadi_register_integrator_idas);
}

IdasInterface::IdasInterface(const std::string& name, const Function& dae,
    double t0, const std::vector<double>& tout) : SundialsInterface(name, dae, t0, tout) {
}

IdasInterface::~IdasInterface() {
  clear_mem();
}

const Options IdasInterface::options_
= {{&SundialsInterface::options_},
    {{"suppress_algebraic",
      {OT_BOOL,
      "Suppress algebraic variables in the error testing"}},
    {"calc_ic",
      {OT_BOOL,
      "Use IDACalcIC to get consistent initial conditions."}},
    {"constraints",
      {OT_INTVECTOR,
      "Constrain the solution y=[x,z]. 0 (default): no constraint on yi, "
      "1: yi >= 0.0, -1: yi <= 0.0, 2: yi > 0.0, -2: yi < 0.0."}},
    {"calc_icB",
      {OT_BOOL,
      "Use IDACalcIC to get consistent initial conditions for "
      "backwards system [default: equal to calc_ic]."}},
    {"abstolv",
      {OT_DOUBLEVECTOR,
      "Absolute tolerarance for each component"}},
    {"max_step_size",
      {OT_DOUBLE,
      "Maximim step size"}},
    {"first_time",
      {OT_DOUBLE,
      "First requested time as a fraction of the time interval"}},
    {"cj_scaling",
      {OT_BOOL,
      "IDAS scaling on cj for the user-defined linear solver module"}},
    {"init_xdot",
      {OT_DOUBLEVECTOR,
      "Initial values for the state derivatives"}}
    }
};

void IdasInterface::init(const Dict& opts) {
  if (verbose_) casadi_message(name_ + "::init");

  // Call the base class init
  SundialsInterface::init(opts);

  // Default options
  cj_scaling_ = true;
  calc_ic_ = true;
  suppress_algebraic_ = false;

  // Read options
  for (auto&& op : opts) {
    if (op.first=="init_xdot") {
      init_xdot_ = op.second;
    } else if (op.first=="cj_scaling") {
      cj_scaling_ = op.second;
    } else if (op.first=="calc_ic") {
      calc_ic_ = op.second;
    } else if (op.first=="suppress_algebraic") {
      suppress_algebraic_ = op.second;
    } else if (op.first=="constraints") {
      y_c_ = op.second;
    } else if (op.first=="abstolv") {
      abstolv_ = op.second;
    }
  }

  // Default dependent options
  calc_icB_ = calc_ic_;
  first_time_ = tout_.back();

  // Read dependent options
  for (auto&& op : opts) {
    if (op.first=="calc_icB") {
      calc_icB_ = op.second;
    } else if (op.first=="first_time") {
      first_time_ = op.second;
    }
  }

  // Get initial conditions for the state derivatives
  if (init_xdot_.empty()) {
    init_xdot_.resize(nx_, 0);
  } else {
    casadi_assert(
      init_xdot_.size()==nx_,
      "Option \"init_xdot\" has incorrect length. Expecting " + str(nx_) + ", "
      "but got " + str(init_xdot_.size()) + ". "
      "Note that this message may actually be generated by the augmented integrator. "
      "In that case, make use of the 'augmented_options' options "
      "to correct 'init_xdot' for the augmented integrator.");
  }

  // Constraints
  casadi_assert(y_c_.size() == nx_+nz_ || y_c_.empty(),
    "Constraint vector if supplied, must be of length nx+nz, but got "
    + str(y_c_.size()) + " and nx+nz = " + str(nx_+nz_) + ".");

  // For Jacobian calculation
  alloc_w(nx_ + nz_); // casadi_copy_block
}

int IdasInterface::resF(double t, N_Vector xz, N_Vector xzdot, N_Vector rr, void *user_data) {
  try {
    auto m = to_mem(user_data);
    auto& s = m->self;
    if (s.calc_daeF(m, t, NV_DATA_S(xz), NV_DATA_S(xz) + s.nx_,
      NV_DATA_S(rr), NV_DATA_S(rr) + s.nx_)) return 1;

    // Subtract state derivative to get residual
    casadi_axpy(s.nx_, -1., NV_DATA_S(xzdot), NV_DATA_S(rr));
    return 0;
  } catch(std::exception& e) { // non-recoverable error
    uerr() << "res failed: " << e.what() << std::endl;
    return -1;
  }
}

void IdasInterface::ehfun(int error_code, const char *module, const char *function,
    char *msg, void *eh_data) {
  try {
    //auto m = to_mem(eh_data);
    //auto& s = m->self;
    uerr() << msg << std::endl;
  } catch(std::exception& e) {
    uerr() << "ehfun failed: " << e.what() << std::endl;
  }
}

int IdasInterface::jtimesF(double t, N_Vector xz, N_Vector xzdot, N_Vector rr, N_Vector v,
    N_Vector Jv, double cj, void *user_data, N_Vector tmp1, N_Vector tmp2) {
  try {
    auto m = to_mem(user_data);
    auto& s = m->self;
    if (s.calc_jtimesF(m, t, NV_DATA_S(xz), NV_DATA_S(xz) + s.nx_,
      NV_DATA_S(v), NV_DATA_S(v) + s.nx_,
      NV_DATA_S(Jv), NV_DATA_S(Jv) + s.nx_)) return 1;

    // Subtract state derivative to get residual
    casadi_axpy(s.nx_, -cj, NV_DATA_S(v), NV_DATA_S(Jv));

    return 0;
  } catch(std::exception& e) { // non-recoverable error
    uerr() << "jtimesF failed: " << e.what() << std::endl;
    return -1;
  }
}

int IdasInterface::jtimesB(double t, N_Vector xz, N_Vector xzdot, N_Vector rxz,
    N_Vector rxzdot, N_Vector resvalB, N_Vector v, N_Vector Jv,
    double cjB, void *user_data, N_Vector tmp1B, N_Vector tmp2B) {
  try {
    auto m = to_mem(user_data);
    auto& s = m->self;
    if (s.calc_jtimesB(m, t, NV_DATA_S(xz), NV_DATA_S(xz) + s.nx_,
      NV_DATA_S(rxz), NV_DATA_S(rxz) + s.nrx_,
      NV_DATA_S(v), NV_DATA_S(v) + s.nrx_,
      NV_DATA_S(Jv), NV_DATA_S(Jv) + s.nrx_)) return 1;

    // Subtract state derivative to get residual
    casadi_axpy(s.nrx_, cjB, NV_DATA_S(v), NV_DATA_S(Jv));

    return 0;
  } catch(std::exception& e) { // non-recoverable error
    uerr() << "jtimesB failed: " << e.what() << std::endl;
    return -1;
  }
}

int IdasInterface::init_mem(void* mem) const {
  if (SundialsInterface::init_mem(mem)) return 1;
  auto m = to_mem(mem);

  // Create IDAS memory block
  m->mem = IDACreate();
  casadi_assert(m->mem!=nullptr, "IDACreate: Creation failed");

  // Set error handler function
  THROWING(IDASetErrHandlerFn, m->mem, ehfun, m);

  // Set user data
  THROWING(IDASetUserData, m->mem, m);

  // Allocate n-vectors for ivp
  m->xzdot = N_VNew_Serial(nx_+nz_);

  // Initialize Idas
  double t0 = 0;
  N_VConst(0.0, m->xz);
  N_VConst(0.0, m->xzdot);
  IDAInit(m->mem, resF, t0, m->xz, m->xzdot);
  if (verbose_) casadi_message("IDA initialized");

  // Include algebraic variables in error testing
  THROWING(IDASetSuppressAlg, m->mem, suppress_algebraic_);

  // Maxinum order for the multistep method
  THROWING(IDASetMaxOrd, m->mem, max_multistep_order_);

  // Initial step size
  if (step0_!=0) THROWING(IDASetInitStep, m->mem, step0_);

  // Set maximum step size
  if (max_step_size_!=0) THROWING(IDASetMaxStep, m->mem, max_step_size_);

  // Set constraints
  if (!y_c_.empty()) {
    N_Vector domain  = N_VNew_Serial(nx_+nz_);
    std::copy(y_c_.begin(), y_c_.end(), NV_DATA_S(domain));

    // Pass to IDA
    int flag = IDASetConstraints(m->mem, domain);
    casadi_assert_dev(flag==IDA_SUCCESS);

    // Free the temporary vector
    N_VDestroy_Serial(domain);
  }

  // Maximum order of method
  if (max_order_) THROWING(IDASetMaxOrd, m->mem, max_order_);

  // Coeff. in the nonlinear convergence test
  if (nonlin_conv_coeff_!=0) THROWING(IDASetNonlinConvCoef, m->mem, nonlin_conv_coeff_);

  // Scaling
  if (!abstolv_.empty()) {
    // Vector absolute tolerances
    N_Vector nv_abstol = N_VNew_Serial(abstolv_.size());
    std::copy(abstolv_.begin(), abstolv_.end(), NV_DATA_S(nv_abstol));
    THROWING(IDASVtolerances, m->mem, reltol_, nv_abstol);
    N_VDestroy_Serial(nv_abstol);
  } else if (scale_abstol_) {
    // Scale absolute tolerances with nominal values
    THROWING(IDASVtolerances, m->mem, reltol_, m->abstolv);
  } else {
    // Scalar absolute tolerances
    THROWING(IDASStolerances, m->mem, reltol_, abstol_);
  }

  // Maximum number of steps
  THROWING(IDASetMaxNumSteps, m->mem, max_num_steps_);

  // Set algebraic components
  N_Vector id = N_VNew_Serial(nx_+nz_);
  std::fill_n(NV_DATA_S(id), nx_, 1);
  std::fill_n(NV_DATA_S(id)+nx_, nz_, 0);

  // Pass this information to IDAS
  THROWING(IDASetId, m->mem, id);

  // Delete the allocated memory
  N_VDestroy_Serial(id);

  // attach a linear solver
  if (newton_scheme_==SD_DIRECT) {
    // Direct scheme
    IDAMem IDA_mem = IDAMem(m->mem);
    IDA_mem->ida_lmem   = m;
    IDA_mem->ida_lsetup = lsetupF;
    IDA_mem->ida_lsolve = lsolveF;
    IDA_mem->ida_setupNonNull = TRUE;
  } else {
    // Iterative scheme
    switch (newton_scheme_) {
    case SD_DIRECT: casadi_assert_dev(0);
    case SD_GMRES: THROWING(IDASpgmr, m->mem, max_krylov_); break;
    case SD_BCGSTAB: THROWING(IDASpbcg, m->mem, max_krylov_); break;
    case SD_TFQMR: THROWING(IDASptfqmr, m->mem, max_krylov_); break;
    }
    THROWING(IDASpilsSetJacTimesVecFn, m->mem, jtimesF);
    if (use_precon_) THROWING(IDASpilsSetPreconditioner, m->mem, psetupF, psolveF);
  }

  // Quadrature equations
  if (nq1_ > 0) {

    // Initialize quadratures in IDAS
    THROWING(IDAQuadInit, m->mem, rhsQF, m->q);

    // Should the quadrature errors be used for step size control?
    if (quad_err_con_) {
      THROWING(IDASetQuadErrCon, m->mem, true);

      // Quadrature error tolerances
      // TODO(Joel): vector absolute tolerances
      THROWING(IDAQuadSStolerances, m->mem, reltol_, abstol_);
    }
  }

  if (verbose_) casadi_message("Attached linear solver");

  // Adjoint sensitivity problem
  if (nadj_ > 0) {
    m->rxzdot = N_VNew_Serial(nrx_+nrz_);
    N_VConst(0.0, m->rxz);
    N_VConst(0.0, m->rxzdot);
  }
  if (verbose_) casadi_message("Initialized adjoint sensitivities");

  // Initialize adjoint sensitivities
  if (nadj_ > 0) {
    int interpType = interp_==SD_HERMITE ? IDA_HERMITE : IDA_POLYNOMIAL;
    THROWING(IDAAdjInit, m->mem, steps_per_checkpoint_, interpType);
  }

  m->first_callB = true;
  return 0;
}

void IdasInterface::reset(IntegratorMemory* mem,
    const double* _x, const double* _z, const double* _p) const {
  if (verbose_) casadi_message(name_ + "::reset");
  auto m = to_mem(mem);

  // Reset the base classes
  SundialsInterface::reset(mem, _x, _z, _p);

  // Re-initialize
  N_VConst(0.0, m->xzdot);
  std::copy(init_xdot_.begin(), init_xdot_.end(), NV_DATA_S(m->xzdot));

  THROWING(IDAReInit, m->mem, m->t, m->xz, m->xzdot);

  // Re-initialize quadratures
  if (nq1_ > 0) THROWING(IDAQuadReInit, m->mem, m->q);

  // Correct initial conditions, if necessary
  if (calc_ic_) {
    THROWING(IDACalcIC, m->mem, IDA_YA_YDP_INIT , first_time_);
    THROWING(IDAGetConsistentIC, m->mem, m->xz, m->xzdot);
  }

  // Re-initialize backward integration
  if (nadj_ > 0) THROWING(IDAAdjReInit, m->mem);
}

void IdasInterface::advance(IntegratorMemory* mem,
    const double* u, double* x, double* z, double* q) const {
  auto m = to_mem(mem);

  // Set controls
  casadi_copy(u, nu_, m->u);

  // Do not integrate past change in input signals or past the end
  THROWING(IDASetStopTime, m->mem, m->t_stop);

  // Integrate, unless already at desired time
  double ttol = 1e-9;   // tolerance
  if (fabs(m->t - m->t_next) >= ttol) {
    // Integrate forward ...
    double tret = m->t;
    if (nrx_>0) { // ... with taping
      THROWING(IDASolveF, m->mem, m->t_next, &tret, m->xz, m->xzdot, IDA_NORMAL, &m->ncheck);
    } else { // ... without taping
      THROWING(IDASolve, m->mem, m->t_next, &tret, m->xz, m->xzdot, IDA_NORMAL);
    }
    // Get quadratures
    if (nq1_ > 0) THROWING(IDAGetQuad, m->mem, &tret, m->q);
  }

  // Set function outputs
  casadi_copy(NV_DATA_S(m->xz), nx_, x);
  casadi_copy(NV_DATA_S(m->xz)+nx_, nz_, z);
  casadi_copy(NV_DATA_S(m->q), nq_, q);

  // Get stats
  THROWING(IDAGetIntegratorStats, m->mem, &m->nsteps, &m->nfevals, &m->nlinsetups,
    &m->netfails, &m->qlast, &m->qcur, &m->hinused, &m->hlast, &m->hcur, &m->tcur);
  THROWING(IDAGetNonlinSolvStats, m->mem, &m->nniters, &m->nncfails);

}

void IdasInterface::resetB(IntegratorMemory* mem) const {
  if (verbose_) casadi_message(name_ + "::resetB");
  auto m = to_mem(mem);

  // Reset initial guess
  N_VConst(0.0, m->rxz);

  // Reset the base classes
  SundialsInterface::resetB(mem);

  // Reset initial guess
  N_VConst(0.0, m->rxzdot);
}

bool IdasInterface::all_zero(const double* v, casadi_int n) {
  // Quick return if trivially zero
  if (v == 0 || n == 0) return true;
  // Loop over entries
  for (casadi_int i = 0; i < n; ++i) {
    if (v[i] != 0.) return false;
  }
  // All zero if reached here
  return true;
}

void IdasInterface::z_impulseB(IdasMemory* m, const double* rz) const {
  // Quick return if nothing to propagate
  if (all_zero(rz, nrz_)) return;
  // We have the following solved nonlinear system of equations:
  //   f_alg(x, z) == 0,
  // which implicitly defines z as a function of x
  // Linearize w.r.t. x:
  //   df_alg/dz * dz/dx + df_alg/dx == 0 <=> dz/dx = -inv(df_alg/dz)*df_alg/dx
  // Want to calculate:
  //   adj_x = (dz/dx)^T * adj_z = -(df_alg/dx)^T * inv((df_alg/dz)^T) * adj_z
  //   = -(df_alg/dx)^T * w,
  // where
  //   (df_alg/dz)^T * w = adj_z
  // Augment linear system to get the system we are able to factorize
  //   [(df_ode/dx)^T - cj*I, (df_alg/dx)^T; (df_ode/dz)^T, (df_alg/dz)^T] * [v; w] = [0; adj_z]
  // (Re)factorize linear system
  if (psetupF(m->t, m->xz, m->xzdot, nullptr, m->cj_last, m, nullptr, nullptr, nullptr))
    casadi_error("Linear system factorization for backwards initial conditions failed");
  // Right-hand-side for linear system in m->v2
  casadi_clear(m->v2, nrx_);
  casadi_copy(rz, nrz_, m->v2 + nrx_);
  // Solve transposed linear system of equations (Note: m->v2 not used since rxz null)
  if (solve_transposed(m, m->t, NV_DATA_S(m->xz), nullptr, m->v2, m->v2)) {
    casadi_error("Linear system solve for backwards initial conditions failed");
  }
  // Calculate: -adj_x = (df_alg/dx)^T * w
  casadi_clear(m->v2, nrx_);
  if (calc_daeB(m, m->t, NV_DATA_S(m->xz), NV_DATA_S(m->xz) + nx_,
      m->v2, m->v2 + nrx_, nullptr, m->v1, m->v1 + nrx_)) {
    casadi_error("Adjoint seed propagation for backwards initial conditions failed");
  }
  // Add contribution to backward state
  casadi_axpy(nrx_, -1., m->v1, NV_DATA_S(m->rxz));
}

void IdasInterface::impulseB(IntegratorMemory* mem,
    const double* rx, const double* rz, const double* rp) const {
  auto m = to_mem(mem);

  // Call method in base class
  SundialsInterface::impulseB(mem, rx, rz, rp);

  // Propagate impulse from rz to rx
  z_impulseB(m, rz);

  if (m->first_callB) {
    // Create backward problem
    THROWING(IDACreateB, m->mem, &m->whichB);
    THROWING(IDAInitB, m->mem, m->whichB, resB, m->t, m->rxz, m->rxzdot);
    THROWING(IDASStolerancesB, m->mem, m->whichB, reltol_, abstol_);
    THROWING(IDASetUserDataB, m->mem, m->whichB, m);
    THROWING(IDASetMaxNumStepsB, m->mem, m->whichB, max_num_steps_);


    // Set algebraic components
    N_Vector id = N_VNew_Serial(nrx_+nrz_);
    std::fill_n(NV_DATA_S(id), nrx_, 1);
    std::fill_n(NV_DATA_S(id)+nrx_, nrz_, 0);
    THROWING(IDASetIdB, m->mem, m->whichB, id);
    N_VDestroy_Serial(id);

    // attach linear solver
    if (newton_scheme_==SD_DIRECT) {
      // Direct scheme
      IDAMem IDA_mem = IDAMem(m->mem);
      IDAadjMem IDAADJ_mem = IDA_mem->ida_adj_mem;
      IDABMem IDAB_mem = IDAADJ_mem->IDAB_mem;
      IDAB_mem->ida_lmem   = m;
      IDAB_mem->IDA_mem->ida_lmem = m;
      IDAB_mem->IDA_mem->ida_lsetup = lsetupB;
      IDAB_mem->IDA_mem->ida_lsolve = lsolveB;
      IDAB_mem->IDA_mem->ida_setupNonNull = TRUE;
    } else {
      // Iterative scheme
      switch (newton_scheme_) {
      case SD_DIRECT: casadi_assert_dev(0);
      case SD_GMRES: THROWING(IDASpgmrB, m->mem, m->whichB, max_krylov_); break;
      case SD_BCGSTAB: THROWING(IDASpbcgB, m->mem, m->whichB, max_krylov_); break;
      case SD_TFQMR: THROWING(IDASptfqmrB, m->mem, m->whichB, max_krylov_); break;
      }
      THROWING(IDASpilsSetJacTimesVecFnB, m->mem, m->whichB, jtimesB);
      if (use_precon_) THROWING(IDASpilsSetPreconditionerB, m->mem, m->whichB, psetupB, psolveB);
    }

    // Quadratures for the adjoint problem
    THROWING(IDAQuadInitB, m->mem, m->whichB, rhsQB, m->ruq);
    if (quad_err_con_) {
      THROWING(IDASetQuadErrConB, m->mem, m->whichB, true);
      THROWING(IDAQuadSStolerancesB, m->mem, m->whichB, reltol_, abstol_);
    }

    // Mark initialized
    m->first_callB = false;
  } else {
    // Re-initialize
    THROWING(IDAReInitB, m->mem, m->whichB, m->t, m->rxz, m->rxzdot);
    if (nrq_ > 0 || nuq_ > 0) {
      // Workaround (bug in SUNDIALS)
      // THROWING(IDAQuadReInitB, m->mem, m->whichB[dir], m->rq[dir]);
      void* memB = IDAGetAdjIDABmem(m->mem, m->whichB);
      THROWING(IDAQuadReInit, memB, m->ruq);
    }
  }

  // Correct initial values for the integration if necessary
  if (calc_icB_ && m->k == nt() - 1) {
    THROWING(IDACalcICB, m->mem, m->whichB, t0_, m->xz, m->xzdot);
    THROWING(IDAGetConsistentICB, m->mem, m->whichB, m->rxz, m->rxzdot);
  }
}

void IdasInterface::retreat(IntegratorMemory* mem, const double* u,
    double* rx, double* rq, double* uq) const {
  auto m = to_mem(mem);

  // Set controls
  casadi_copy(u, nu_, m->u);

  // Integrate, unless already at desired time
  if (m->t_next < m->t) {
    double tret = m->t;
    THROWING(IDASolveB, m->mem, m->t_next, IDA_NORMAL);
    THROWING(IDAGetB, m->mem, m->whichB, &tret, m->rxz, m->rxzdot);
    if (nrq_ > 0 || nuq_ > 0) {
      THROWING(IDAGetQuadB, m->mem, m->whichB, &tret, m->ruq);
    }
    // Interpolate to get current state
    THROWING(IDAGetAdjY, m->mem, m->t_next, m->xz, m->xzdot);
  }

  // Save outputs
  casadi_copy(NV_DATA_S(m->rxz), nrx_, rx);
  casadi_copy(NV_DATA_S(m->ruq), nrq_, rq);
  casadi_copy(NV_DATA_S(m->ruq) + nrq_, nuq_, uq);

  // Get stats
  IDAMem IDA_mem = IDAMem(m->mem);
  IDAadjMem IDAADJ_mem = IDA_mem->ida_adj_mem;
  IDABMem IDAB_mem = IDAADJ_mem->IDAB_mem;
  THROWING(IDAGetIntegratorStats, IDAB_mem->IDA_mem, &m->nstepsB, &m->nfevalsB,
    &m->nlinsetupsB, &m->netfailsB, &m->qlastB, &m->qcurB, &m->hinusedB,
    &m->hlastB, &m->hcurB, &m->tcurB);
  THROWING(IDAGetNonlinSolvStats, IDAB_mem->IDA_mem, &m->nnitersB, &m->nncfailsB);
}

void IdasInterface::idas_error(const char* module, int flag) {
  // Successfull return or warning
  if (flag>=IDA_SUCCESS) return;
  // Construct error message
  char* flagname = IDAGetReturnFlagName(flag);
  std::stringstream ss;
  ss << module << " returned \"" << flagname << "\". Consult IDAS documentation.";
  free(flagname);  // NOLINT
  casadi_error(ss.str());
}

int IdasInterface::rhsQF(double t, N_Vector xz, N_Vector xzdot, N_Vector qdot, void *user_data) {
  try {
    auto m = to_mem(user_data);
    auto& s = m->self;
    if (s.calc_quadF(m, t, NV_DATA_S(xz), NV_DATA_S(xz) + s.nx_, NV_DATA_S(qdot))) return 1;

    return 0;
  } catch(std::exception& e) { // non-recoverable error
    uerr() << "rhsQ failed: " << e.what() << std::endl;
    return -1;
  }
}

int IdasInterface::resB(double t, N_Vector xz, N_Vector xzdot, N_Vector rxz,
    N_Vector rxzdot, N_Vector rr, void *user_data) {
  try {
    auto m = to_mem(user_data);
    auto& s = m->self;
    if (s.calc_daeB(m, t, NV_DATA_S(xz), NV_DATA_S(xz) + s.nx_,
      NV_DATA_S(rxz), NV_DATA_S(rxz) + s.nrx_, m->rp,
      NV_DATA_S(rr), NV_DATA_S(rr) + s.nrx_)) return 1;

    // Subtract state derivative to get residual
    casadi_axpy(s.nrx_, 1., NV_DATA_S(rxzdot), NV_DATA_S(rr));

    return 0;
  } catch(std::exception& e) { // non-recoverable error
    uerr() << "resB failed: " << e.what() << std::endl;
    return -1;
  }
}

int IdasInterface::rhsQB(double t, N_Vector xz, N_Vector xzdot, N_Vector rxz,
    N_Vector rxzdot, N_Vector ruqdot, void *user_data) {
  try {
    auto m = to_mem(user_data);
    auto& s = m->self;
    if (s.calc_quadB(m, t, NV_DATA_S(xz), NV_DATA_S(xz) + s.nx_,
      NV_DATA_S(rxz), NV_DATA_S(rxz) + s.nrx_,
      NV_DATA_S(ruqdot), NV_DATA_S(ruqdot) + s.nrq_)) return 1;

    // Negate (note definition of g)
    casadi_scal(s.nrq_ + s.nuq_, -1., NV_DATA_S(ruqdot));

    return 0;
  } catch(std::exception& e) { // non-recoverable error
    uerr() << "resQB failed: " << e.what() << std::endl;
    return -1;
  }
}

int IdasInterface::psolveF(double t, N_Vector xz, N_Vector xzdot, N_Vector rr,
    N_Vector rvec, N_Vector zvec, double cj, double delta, void *user_data, N_Vector tmp) {
  try {
    auto m = to_mem(user_data);
    auto& s = m->self;

    // Get right-hand sides in m->v1, ordered by sensitivity directions
    double* vx = NV_DATA_S(rvec);
    double* vz = vx + s.nx_;
    double* v_it = m->v1;
    for (int d = 0; d <= s.nfwd_; ++d) {
      casadi_copy(vx + d * s.nx1_, s.nx1_, v_it);
      v_it += s.nx1_;
      casadi_copy(vz + d * s.nz1_, s.nz1_, v_it);
      v_it += s.nz1_;
    }

    // Solve for undifferentiated right-hand-side, save to output
    if (s.linsolF_.solve(m->jacF, m->v1, 1, false, m->mem_linsolF))
      return 1;
    vx = NV_DATA_S(zvec); // possibly different from rvec
    vz = vx + s.nx_;
    casadi_copy(m->v1, s.nx1_, vx);
    casadi_copy(m->v1 + s.nx1_, s.nz1_, vz);

    // Sensitivity equations
    if (s.nfwd_ > 0) {
      // Second order correction
      if (s.second_order_correction_) {
        // The outputs will double as seeds for jtimesF
        casadi_clear(vx + s.nx1_, s.nx_ - s.nx1_);
        casadi_clear(vz + s.nz1_, s.nz_ - s.nz1_);
        if (s.calc_jtimesF(m, t, NV_DATA_S(xz), NV_DATA_S(xz) + s.nx_,
          vx, vz, m->v2, m->v2 + s.nx_)) return 1;

        // Subtract m->v2 (reordered) from m->v1
        v_it = m->v1 + s.nx1_ + s.nz1_;
        for (int d = 1; d <= s.nfwd_; ++d) {
          casadi_axpy(s.nx1_, -1., m->v2 + d*s.nx1_, v_it);
          v_it += s.nx1_;
          casadi_axpy(s.nz1_, -1., m->v2 + s.nx_ + d*s.nz1_, v_it);
          v_it += s.nz1_;
        }
      }

      // Solve for sensitivity right-hand-sides
      if (s.linsolF_.solve(m->jacF, m->v1 + s.nx1_ + s.nz1_, s.nfwd_,
        false, m->mem_linsolF)) return 1;

      // Save to output, reordered
      v_it = m->v1 + s.nx1_ + s.nz1_;
      for (int d = 1; d <= s.nfwd_; ++d) {
        casadi_copy(v_it, s.nx1_, vx + d * s.nx1_);
        v_it += s.nx1_;
        casadi_copy(v_it, s.nz1_, vz + d * s.nz1_);
        v_it += s.nz1_;
      }
    }

    return 0;
  } catch(std::exception& e) { // non-recoverable error
    uerr() << "psolve failed: " << e.what() << std::endl;
    return -1;
  }
}

int IdasInterface::solve_transposed(IdasMemory* m, double t, const double* xz, const double* rxz,
    const double* rhs, double* sol) const {
  // Get right-hand sides in m->v1, ordered by sensitivity directions
  double* v_it = m->v1;
  for (int d = 0; d <= nfwd_; ++d) {
    for (int a = 0; a < nadj_; ++a) {
      casadi_copy(rhs + (d * nadj_ + a) * nrx1_, nrx1_, v_it);
      v_it += nrx1_;
      casadi_copy(rhs + nrx_ + (d * nadj_ + a) * nrz1_, nrz1_, v_it);
      v_it += nrz1_;
    }
  }

  // Solve for undifferentiated right-hand-side, save to output
  if (linsolF_.solve(m->jacF, m->v1, nadj_, true, m->mem_linsolF)) return 1;
  for (int a = 0; a < nadj_; ++a) {
    casadi_copy(m->v1 + a * (nrx1_ + nrz1_), nrx1_, sol + a * nrx1_);
    casadi_copy(m->v1 + a * (nrx1_ + nrz1_) + nrx1_, nrz1_, sol + nrx_ + a * nrz1_);
  }

  // Sensitivity equations
  if (nfwd_ > 0) {
    // Second order correction
    if (second_order_correction_ && rxz) {
      // The outputs will double as seeds for jtimesB
      casadi_clear(sol + nrx1_ * nadj_, nrx_ - nrx1_ * nadj_);
      casadi_clear(sol + nrx_ + nrz1_ * nadj_, nrz_ - nrz1_ * nadj_);

      // Get second-order-correction, save to m->v2
      if (calc_jtimesB(m, t, xz, xz + nx_, rxz, rxz + nrx_,
        sol, sol + nrx_, m->v2, m->v2 + nrx_)) return 1;

      // Subtract m->v2 (reordered) from m->v1
      v_it = m->v1 + (nrx1_ + nrz1_) * nadj_;
      for (int d = 1; d <= nfwd_; ++d) {
        for (int a = 0; a < nadj_; ++a) {
          casadi_axpy(nrx1_, -1., m->v2 + nrx1_ * (d * nadj_ + a), v_it);
          v_it += nrx1_;
          casadi_axpy(nrz1_, -1., m->v2 + nrx_ + nrz1_ * (d * nadj_ + a), v_it);
          v_it += nrz1_;
        }
      }
    }

    // Solve for sensitivity right-hand-sides
    if (linsolF_.solve(m->jacF, m->v1 + nrx1_ * nadj_ + nrz1_ * nadj_,
      nadj_ * nfwd_, true, m->mem_linsolF)) return 1;

    // Save to output, reordered
    v_it = m->v1 + (nrx1_ + nrz1_) * nadj_;
    for (int d = 1; d <= nfwd_; ++d) {
      for (int a = 0; a < nadj_; ++a) {
        casadi_copy(v_it, nrx1_, sol + nrx1_ * (d * nadj_ + a));
        v_it += nrx1_;
        casadi_axpy(nrz1_, -1., m->v2 + nrx_ + nrz1_ * (d * nadj_ + a), v_it);
        v_it += nrz1_;
      }
    }
  }

  return 0;
}

int IdasInterface::psolveB(double t, N_Vector xz, N_Vector xzdot, N_Vector xzB,
    N_Vector xzdotB, N_Vector resvalB, N_Vector rvecB,
    N_Vector zvecB, double cjB, double deltaB,
    void *user_data, N_Vector tmpB) {
  try {
    auto m = to_mem(user_data);
    auto& s = m->self;
    return s.solve_transposed(m, t, NV_DATA_S(xz), NV_DATA_S(xzB),
      NV_DATA_S(rvecB), NV_DATA_S(zvecB));

  } catch(std::exception& e) { // non-recoverable error
    uerr() << "psolveB failed: " << e.what() << std::endl;
    return -1;
  }
}

template<typename T1>
void casadi_copy_block(const T1* x, const casadi_int* sp_x, T1* y, const casadi_int* sp_y,
    casadi_int r_begin, casadi_int c_begin, T1* w) {
  // x and y should be distinct
  casadi_int nrow_x, ncol_x, ncol_y, i_x, i_y, j, el, r_end;
  const casadi_int *colind_x, *row_x, *colind_y, *row_y;
  nrow_x = sp_x[0];
  ncol_x = sp_x[1];
  colind_x = sp_x+2; row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2; row_y = sp_y + 2 + ncol_y+1;
  // End of the rows to be copied
  r_end = r_begin + nrow_x;
  // w will correspond to a column of x, initialize to zero
  casadi_clear(w, nrow_x);
  // Loop over columns in x
  for (i_x = 0; i_x < ncol_x; ++i_x) {
    // Corresponding row in y
    i_y = i_x + c_begin;
    // Copy x entries to w
    for (el=colind_x[i_x]; el<colind_x[i_x + 1]; ++el) w[row_x[el]] = x[el];
    // Copy entries to y, if in interval
    for (el=colind_y[i_y]; el<colind_y[i_y + 1]; ++el) {
      j = row_y[el];
      if (j >= r_begin && j < r_end) y[el] = w[j - r_begin];
    }
    // Restore w
    for (el=colind_x[i_x]; el<colind_x[i_x + 1]; ++el) w[row_x[el]] = 0;
  }
}

int IdasInterface::psetupF(double t, N_Vector xz, N_Vector xzdot, N_Vector rr,
    double cj, void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
  try {
    auto m = to_mem(user_data);
    auto& s = m->self;

    // Sparsity patterns
    const Function& jacF = s.get_function("jacF");
    const Sparsity& sp_jac_ode_x = jacF.sparsity_out(JACF_ODE_X);
    const Sparsity& sp_jac_alg_x = jacF.sparsity_out(JACF_ALG_X);
    const Sparsity& sp_jac_ode_z = jacF.sparsity_out(JACF_ODE_Z);
    const Sparsity& sp_jac_alg_z = jacF.sparsity_out(JACF_ALG_Z);
    const Sparsity& sp_jacF = s.linsolF_.sparsity();

    // Calculate Jacobian blocks
    if (s.calc_jacF(m, t, NV_DATA_S(xz), NV_DATA_S(xz) + s.nx_,
      m->jac_ode_x, m->jac_alg_x, m->jac_ode_z, m->jac_alg_z)) return 1;

    // Copy to jacF structure
    casadi_int nx_jac = sp_jac_ode_x.size1();  // excludes sensitivity equations
    casadi_copy_block(m->jac_ode_x, sp_jac_ode_x, m->jacF, sp_jacF, 0, 0, m->w);
    casadi_copy_block(m->jac_alg_x, sp_jac_alg_x, m->jacF, sp_jacF, nx_jac, 0, m->w);
    casadi_copy_block(m->jac_ode_z, sp_jac_ode_z, m->jacF, sp_jacF, 0, nx_jac, m->w);
    casadi_copy_block(m->jac_alg_z, sp_jac_alg_z, m->jacF, sp_jacF, nx_jac, nx_jac, m->w);

    // Shift diagonal corresponding to jac_ode_x
    const casadi_int *colind = sp_jacF.colind(), *row = sp_jacF.row();
    for (casadi_int c = 0; c < nx_jac; ++c) {
      for (casadi_int k = colind[c]; k < colind[c + 1]; ++k) {
        if (row[k] == c) m->jacF[k] -= cj;
      }
    }

    // Factorize the linear system
    if (s.linsolF_.nfact(m->jacF, m->mem_linsolF)) return 1;
    m->cj_last = cj;

    return 0;
  } catch(std::exception& e) { // non-recoverable error
    uerr() << "psetup failed: " << e.what() << std::endl;
    return -1;
  }
}

int IdasInterface::psetupB(double t, N_Vector xz, N_Vector xzdot, N_Vector rxz, N_Vector rxzdot,
    N_Vector rresval, double cj, void *user_data, N_Vector tmp1B, N_Vector tmp2B, N_Vector tmp3B) {
  try {
    // We use the same linear solver for the forward problem as for the backward problem
    return psetupF(t, xz, nullptr, nullptr, -cj, user_data, tmp1B, tmp2B, tmp3B);

  } catch(std::exception& e) { // non-recoverable error
    uerr() << "psetupB failed: " << e.what() << std::endl;
    return -1;
  }
}

int IdasInterface::lsetupF(IDAMem IDA_mem, N_Vector xz, N_Vector xzdot, N_Vector resp,
    N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3) {
  // Current time
  double t = IDA_mem->ida_tn;

  // Multiple of df_dydot to be added to the matrix
  double cj = IDA_mem->ida_cj;

  // Call the preconditioner setup function (which sets up the linear solver)
  return psetupF(t, xz, xzdot, nullptr, cj, IDA_mem->ida_lmem,
    vtemp1, vtemp1, vtemp3);
}

int IdasInterface::lsetupB(IDAMem IDA_mem, N_Vector xzB, N_Vector xzdotB, N_Vector respB,
    N_Vector vtemp1B, N_Vector vtemp2B, N_Vector vtemp3B) {
  try {
    auto m = to_mem(IDA_mem->ida_lmem);
    //auto& s = m->self;
    IDAadjMem IDAADJ_mem;
    //IDABMem IDAB_mem;

    // Current time
    double t = IDA_mem->ida_tn; // TODO(Joel): is this correct?
    // Multiple of df_dydot to be added to the matrix
    double cj = IDA_mem->ida_cj;

    IDA_mem = static_cast<IDAMem>(IDA_mem->ida_user_data);

    IDAADJ_mem = IDA_mem->ida_adj_mem;
    //IDAB_mem = IDAADJ_mem->ia_bckpbCrt;

    // Get FORWARD solution from interpolation.
    if (IDAADJ_mem->ia_noInterp==FALSE) {
      int flag = IDAADJ_mem->ia_getY(IDA_mem, t, IDAADJ_mem->ia_yyTmp, IDAADJ_mem->ia_ypTmp,
                                  nullptr, nullptr);
      if (flag != IDA_SUCCESS) casadi_error("Could not interpolate forward states");
    }
    // Call the preconditioner setup function (which sets up the linear solver)
    return psetupB(t, IDAADJ_mem->ia_yyTmp, IDAADJ_mem->ia_ypTmp,
      xzB, xzdotB, nullptr, cj, static_cast<void*>(m), vtemp1B, vtemp1B, vtemp3B);

  } catch(std::exception& e) { // non-recoverable error
    uerr() << "lsetupB failed: " << e.what() << std::endl;
    return -1;
  }
}

int IdasInterface::lsolveF(IDAMem IDA_mem, N_Vector b, N_Vector weight, N_Vector xz,
    N_Vector xzdot, N_Vector rr) {
  try {
    auto m = to_mem(IDA_mem->ida_lmem);
    auto& s = m->self;

    // Current time
    double t = IDA_mem->ida_tn;

    // Multiple of df_dydot to be added to the matrix
    double cj = IDA_mem->ida_cj;

    // Accuracy
    double delta = 0.0;

    // Call the preconditioner solve function (which solves the linear system)
    int flag = psolveF(t, xz, xzdot, rr, b, b, cj,
      delta, static_cast<void*>(m), nullptr);
    if (flag) return flag;

    // Scale the correction to account for change in cj
    if (s.cj_scaling_) {
      double cjratio = IDA_mem->ida_cjratio;
      if (cjratio != 1.0) N_VScale(2.0/(1.0 + cjratio), b, b);
    }

    return 0;
  } catch(std::exception& e) { // non-recoverable error
    uerr() << "lsolve failed: " << e.what() << std::endl;
    return -1;
  }
}

int IdasInterface::lsolveB(IDAMem IDA_mem, N_Vector b, N_Vector weight, N_Vector xzB,
    N_Vector xzdotB, N_Vector rrB) {
  try {
    auto m = to_mem(IDA_mem->ida_lmem);
    auto& s = m->self;
    IDAadjMem IDAADJ_mem;
    //IDABMem IDAB_mem;
    int flag;

    // Current time
    double t = IDA_mem->ida_tn; // TODO(Joel): is this correct?
    // Multiple of df_dydot to be added to the matrix
    double cj = IDA_mem->ida_cj;
    double cjratio = IDA_mem->ida_cjratio;

    IDA_mem = (IDAMem) IDA_mem->ida_user_data;
    IDAADJ_mem = IDA_mem->ida_adj_mem;
    //IDAB_mem = IDAADJ_mem->ia_bckpbCrt;

    // Get FORWARD solution from interpolation.
    if (IDAADJ_mem->ia_noInterp==FALSE) {
      flag = IDAADJ_mem->ia_getY(IDA_mem, t, IDAADJ_mem->ia_yyTmp, IDAADJ_mem->ia_ypTmp,
        nullptr, nullptr);
      if (flag != IDA_SUCCESS) casadi_error("Could not interpolate forward states");
    }

    // Accuracy
    double delta = 0.0;

    // Call the preconditioner solve function (which solves the linear system)
    flag = psolveB(t, IDAADJ_mem->ia_yyTmp, IDAADJ_mem->ia_ypTmp, xzB, xzdotB,
      rrB, b, b, cj, delta, static_cast<void*>(m), nullptr);
    if (flag) return flag;

    // Scale the correction to account for change in cj
    if (s.cj_scaling_) {
      if (cjratio != 1.0) N_VScale(2.0/(1.0 + cjratio), b, b);
    }
    return 0;
  } catch(std::exception& e) { // non-recoverable error
    uerr() << "lsolveB failed: " << e.what() << std::endl;
    return -1;
  }
}

IdasMemory::IdasMemory(const IdasInterface& s) : self(s) {
  this->mem = nullptr;
  this->xzdot = nullptr;
  this->rxzdot = nullptr;
  this->cj_last = nan;

  // Reset checkpoints counter
  this->ncheck = 0;
}

IdasMemory::~IdasMemory() {
  if (this->mem) IDAFree(&this->mem);
  if (this->xzdot) N_VDestroy_Serial(this->xzdot);
  if (this->rxzdot) N_VDestroy_Serial(this->rxzdot);
  if (this->mem_linsolF >= 0) self.linsolF_.release(this->mem_linsolF);
}

IdasInterface::IdasInterface(DeserializingStream& s) : SundialsInterface(s) {
  int version = s.version("IdasInterface", 1, 2);
  s.unpack("IdasInterface::cj_scaling", cj_scaling_);
  s.unpack("IdasInterface::calc_ic", calc_ic_);
  s.unpack("IdasInterface::calc_icB", calc_icB_);
  s.unpack("IdasInterface::suppress_algebraic", suppress_algebraic_);
  s.unpack("IdasInterface::abstolv", abstolv_);
  s.unpack("IdasInterface::first_time", first_time_);
  s.unpack("IdasInterface::init_xdot", init_xdot_);

  if (version>=2) {
    s.unpack("IdasInterface::max_step_size", max_step_size_);
    s.unpack("IdasInterface::y_c", y_c_);
  } else {
    max_step_size_ = 0;
  }
}

void IdasInterface::serialize_body(SerializingStream &s) const {
  SundialsInterface::serialize_body(s);
  s.version("IdasInterface", 2);
  s.pack("IdasInterface::cj_scaling", cj_scaling_);
  s.pack("IdasInterface::calc_ic", calc_ic_);
  s.pack("IdasInterface::calc_icB", calc_icB_);
  s.pack("IdasInterface::suppress_algebraic", suppress_algebraic_);
  s.pack("IdasInterface::abstolv", abstolv_);
  s.pack("IdasInterface::first_time", first_time_);
  s.pack("IdasInterface::init_xdot", init_xdot_);
  s.pack("IdasInterface::max_step_size", max_step_size_);
  s.pack("IdasInterface::y_c", y_c_);
}

} // namespace casadi
