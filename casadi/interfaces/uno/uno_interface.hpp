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

#include <casadi/interfaces/uno/casadi_nlpsol_uno_export.h>
#include <optimization/Model.hpp>
#include <linear_algebra/RectangularMatrix.hpp>

#include "casadi/core/nlpsol_impl.hpp"

/** \defgroup plugin_Nlpsol_uno Title
    \par

  Uno interface

    \identifier{22c} */

/** \pluginsection{Nlpsol,knitro} */

/// \cond INTERNAL
namespace casadi {
  // Forward declaration
  class UnoInterface;

  struct CASADI_NLPSOL_UNO_EXPORT UnoMemory : public NlpsolMemory {
    
    /// Constructor
    UnoMemory();

    /// Destructor
    ~unoMemory();
  };

  /** \brief \pluginbrief{Nlpsol,uno}
     @copydoc Nlpsol_doc
     @copydoc plugin_Nlpsol_uno
  */
  class CASADI_NLPSOL_KNITRO_EXPORT UnoInterface : public Nlpsol {
  public:
    

  };

} // namespace casadi

/// \endcond
#endif // CASADI_UNO_INTERFACE_HPP
