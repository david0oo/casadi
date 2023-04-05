import casadi as cs

x = cs.MX.sym("x")
y = cs.MX.sym("y")

f = x**2 + y**2

g = x**2 + y**2

lbg = 0
ubg = 0

nlp = {"x":cs.vertcat(x,y), "f":f, "g":g}

solver = cs.nlpsol('s','uno', nlp)


res = solver(lbg=lbg, ubg=ubg)

