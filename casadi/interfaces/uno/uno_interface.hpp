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


#ifndef CASADI_UNO_INTERFACE_HPP
#define CASADI_UNO_INTERFACE_HPP

#include "uno_nlp.hpp"
#include <casadi/interfaces/uno/casadi_nlpsol_uno_export.h>

#include "casadi/core/nlpsol_impl.hpp"

/** \defgroup plugin_Nlpsol_uno Title
    \par

    David Kiessling

  Uno interface

    \identifier{22c} */

/** \pluginsection{Nlpsol,uno} */

/// \cond INTERNAL

namespace casadi {
  // Forward declaration
  class UnoInterface;

  /*------------------------
  Definition of UnoMemory
  ------------------------*/

  struct CASADI_NLPSOL_UNO_EXPORT UnoMemory : public NlpsolMemory {
    const UnoInterface& self;

    void* model;
    void* solver;
    void* uno_nlp;
    const char* return_status;
    int iter_count;
    double primal_infeasbility;
    double stationarity;
    double complementarity;
    // Set by uno_termination_cb when the user iteration_callback throws or
    // when iteration_callback_ignore_errors_ is false; rethrown after uno_optimize.
    std::exception_ptr cb_exception;
    UnoMemory(const UnoInterface& uno_interface);
    ~UnoMemory();
  };

  /*------------------------------
  Definition of class UnoInterface
  -------------------------------*/

  /** \brief \pluginbrief{Nlpsol,uno}
     @copydoc Nlpsol_doc
     @copydoc plugin_Nlpsol_uno
  */
  class CASADI_NLPSOL_UNO_EXPORT UnoInterface : public Nlpsol {
    friend class UnoNlp;
  public:
    explicit UnoInterface(const std::string& name, const Function& nlp);
    ~UnoInterface() override;

    const char* plugin_name() const override { return "uno"; }
    std::string class_name() const override { return "UnoInterface"; }

    static Nlpsol* creator(const std::string& name, const Function& nlp) {
      return new UnoInterface(name, nlp);
    }

    static const Options options_;
    const Options& get_options() const override { return options_; }

    void init(const Dict& opts) override;
    void* alloc_mem() const override { return new UnoMemory(*this); }
    int  init_mem(void* mem) const override;
    void free_mem(void* mem) const override;
    void set_work(void* mem, const double**& arg, double**& res,
                  casadi_int*& iw, double*& w) const override;
    int  solve(void* mem) const override;
    Dict get_stats(void* mem) const override;

    static const std::string meta_doc;

  private:
    // NLP sparsities discovered in init().
    Sparsity jacg_sp_;
    Sparsity hesslag_sp_;
    // Sparsity index arrays in Uno's int width, filled once in init().
    std::vector<uno_int> jacobian_row_indices_;
    std::vector<uno_int> jacobian_column_indices_;
    std::vector<uno_int> hessian_row_indices_;
    std::vector<uno_int> hessian_column_indices_;
    // Solver-specific options forwarded to uno (the {"uno": {...}} dict).
    Dict opts_;
  };

} // namespace casadi

/// \endcond
#endif // CASADI_UNO_INTERFACE_HPP