"""
Test Problem for UNO solver
- Globalization is needed.
- Linear constraints as constraints, solver does not know about
 linearity
"""
import casadi as cs

x = cs.SX.sym("x")
f = cs.log(cs.exp(x) + cs.exp(-x))
g = x
nlp = {'x':x, 'f':f, 'g':g}

lbg = -2
ubg = 2
x0 = 2

solver = cs.nlpsol('s','uno', nlp)
res = solver(x0  = x0,ubg = ubg,lbg = lbg)
print(res['x'])
