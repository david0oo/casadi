/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            K.U. Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *    Copyright (C) 2022-2023 David Kiessling
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


      #include "feasiblesqpmethod.hpp"
      #include <string>

      const std::string casadi::Feasiblesqpmethod::meta_doc=
      "\n"
"An implementation of FP-SQP\n"
"\n"
"\n"
">List of available options\n"
"\n"
"+-----------------+-----------------+-----------------+-----------------+\n"
"|       Id        |      Type       |     Default     |   Description   |\n"
"+=================+=================+=================+=================+\n"
"| beta            | OT_DOUBLE         | 0.800           | Line-search     |\n"
"|                 |                 |                 | parameter,      |\n"
"|                 |                 |                 | restoration     |\n"
"|                 |                 |                 | factor of       |\n"
"|                 |                 |                 | stepsize        |\n"
"+-----------------+-----------------+-----------------+-----------------+\n"
"| c1              | OT_DOUBLE         | 0.000           | Armijo          |\n"
"|                 |                 |                 | condition,      |\n"
"|                 |                 |                 | coefficient of  |\n"
"|                 |                 |                 | decrease in     |\n"
"|                 |                 |                 | merit           |\n"
"+-----------------+-----------------+-----------------+-----------------+\n"
"| hessian_approxi | OT_STRING       | \"exact\"         | limited-        |\n"
"| mation          |                 |                 | memory|exact    |\n"
"+-----------------+-----------------+-----------------+-----------------+\n"
"| lbfgs_memory    | OT_INT      | 10              | Size of L-BFGS  |\n"
"|                 |                 |                 | memory.         |\n"
"+-----------------+-----------------+-----------------+-----------------+\n"
"| max_iter        | OT_INT      | 50              | Maximum number  |\n"
"|                 |                 |                 | of SQP          |\n"
"|                 |                 |                 | iterations      |\n"
"+-----------------+-----------------+-----------------+-----------------+\n"
"| max_iter_ls     | OT_INT      | 3               | Maximum number  |\n"
"|                 |                 |                 | of linesearch   |\n"
"|                 |                 |                 | iterations      |\n"
"+-----------------+-----------------+-----------------+-----------------+\n"
"| merit_memory    | OT_INT      | 4               | Size of memory  |\n"
"|                 |                 |                 | to store        |\n"
"|                 |                 |                 | history of      |\n"
"|                 |                 |                 | merit function  |\n"
"|                 |                 |                 | values          |\n"
"+-----------------+-----------------+-----------------+-----------------+\n"
"| min_step_size   | OT_DOUBLE         | 0.000           | The size (inf-  |\n"
"|                 |                 |                 | norm) of the    |\n"
"|                 |                 |                 | step size       |\n"
"|                 |                 |                 | should not      |\n"
"|                 |                 |                 | become smaller  |\n"
"|                 |                 |                 | than this.      |\n"
"+-----------------+-----------------+-----------------+-----------------+\n"
"| print_header    | OT_BOOL      | true            | Print the       |\n"
"|                 |                 |                 | header with     |\n"
"|                 |                 |                 | problem         |\n"
"|                 |                 |                 | statistics      |\n"
"+-----------------+-----------------+-----------------+-----------------+\n"
"| print_time      | OT_BOOL      | true            | Print           |\n"
"|                 |                 |                 | information     |\n"
"|                 |                 |                 | about execution |\n"
"|                 |                 |                 | time            |\n"
"+-----------------+-----------------+-----------------+-----------------+\n"
"| conic       | OT_STRING       | GenericType()   | The QP solver   |\n"
"|                 |                 |                 | to be used by   |\n"
"|                 |                 |                 | the SQP method  |\n"
"+-----------------+-----------------+-----------------+-----------------+\n"
"| conic_optio | OT_DICT   | GenericType()   | Options to be   |\n"
"| ns              |                 |                 | passed to the   |\n"
"|                 |                 |                 | QP solver       |\n"
"+-----------------+-----------------+-----------------+-----------------+\n"
"| regularize      | OT_BOOL      | false           | Automatic       |\n"
"|                 |                 |                 | regularization  |\n"
"|                 |                 |                 | of Lagrange     |\n"
"|                 |                 |                 | Hessian.        |\n"
"+-----------------+-----------------+-----------------+-----------------+\n"
"| tol_du          | OT_DOUBLE         | 0.000           | Stopping        |\n"
"|                 |                 |                 | criterion for   |\n"
"|                 |                 |                 | dual            |\n"
"|                 |                 |                 | infeasability   |\n"
"+-----------------+-----------------+-----------------+-----------------+\n"
"| tol_pr          | OT_DOUBLE         | 0.000           | Stopping        |\n"
"|                 |                 |                 | criterion for   |\n"
"|                 |                 |                 | primal          |\n"
"|                 |                 |                 | infeasibility   |\n"
"+-----------------+-----------------+-----------------+-----------------+\n"
"\n"
"\n"
">List of available monitors\n"
"\n"
"+-------------+\n"
"|     Id      |\n"
"+=============+\n"
"| bfgs        |\n"
"+-------------+\n"
"| dx          |\n"
"+-------------+\n"
"| eval_f      |\n"
"+-------------+\n"
"| eval_g      |\n"
"+-------------+\n"
"| eval_grad_f |\n"
"+-------------+\n"
"| eval_h      |\n"
"+-------------+\n"
"| eval_jac_g  |\n"
"+-------------+\n"
"| qp          |\n"
"+-------------+\n"
"\n"
"\n"
">List of available stats\n"
"\n"
"+--------------------+\n"
"|         Id         |\n"
"+====================+\n"
"| iter_count         |\n"
"+--------------------+\n"
"| iteration          |\n"
"+--------------------+\n"
"| iterations         |\n"
"+--------------------+\n"
"| n_eval_f           |\n"
"+--------------------+\n"
"| n_eval_g           |\n"
"+--------------------+\n"
"| n_eval_grad_f      |\n"
"+--------------------+\n"
"| n_eval_h           |\n"
"+--------------------+\n"
"| n_eval_jac_g       |\n"
"+--------------------+\n"
"| return_status      |\n"
"+--------------------+\n"
"| t_callback_fun     |\n"
"+--------------------+\n"
"| t_callback_prepare |\n"
"+--------------------+\n"
"| t_eval_f           |\n"
"+--------------------+\n"
"| t_eval_g           |\n"
"+--------------------+\n"
"| t_eval_grad_f      |\n"
"+--------------------+\n"
"| t_eval_h           |\n"
"+--------------------+\n"
"| t_eval_jac_g       |\n"
"+--------------------+\n"
"| t_mainloop         |\n"
"+--------------------+\n"
"\n"
"\n"
"\n"
"\n"
;
