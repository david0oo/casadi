import casadi as cs

# x = cs.MX.sym("x")
# y = cs.MX.sym("y")

# f = x**2 + y**2
# g = x**2 + y**2

# lbg = 0
# ubg = 0

x = cs.SX.sym("x")
f = cs.log(cs.exp(x) + cs.exp(-x))
g = x
nlp = {'x':x, 'f':f, 'g':g}

lbx = -2
ubx = 2
x0 = 2

# nlp = {"x":cs.vertcat(x,y), "f":f, "g":g}

# opts = {"ipopt":{"linear_solver":"ma57"}}

solver = cs.nlpsol('s','uno', nlp)#, opts)


# res = solver(lbg=lbg, ubg=ubg, x0=cs.vertcat(0.5,0.5))
res = solver(x0  = x0,ubx = ubx,lbx = lbx)
print(res['x'])
