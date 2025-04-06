using iLQGames
using Plots
import iLQGames: dx



# parametes: number of states, number of inputs, sampling time, horizon
nx, nu, ΔT, game_horizon = 8, 4, 0.1, 40

# setup the dynamics
struct Drone <: ControlSystem{ΔT,nx,nu} end
# state: (px, py, phi, v)
dx(cs::Drone, x, u, t) = SVector(x[2], u[1], x[4], u[2], x[6], u[3], x[8], u[4])
dynamics = Drone()

costs = (FunctionPlayerCost((g, x, u, t) -> (((x[1])^2 + (x[3] - 2)^2 > 1) * ((x[1])^2 + (x[3] - 2)^2)^2 + 0.5 * (u[1]^2 + u[2]^2) + (((x[1] - x[5])^2 + (x[3] - x[7])^2) < 2) * (2 - ((x[1] - x[5])^2 + (x[3] - x[7])^2))^4)),
         FunctionPlayerCost((g, x, u, t) -> (((x[5])^2 + (x[7] - 2)^2 > 1) * ((x[5])^2 + (x[7] - 2)^2)^2 + 0.5 * (u[3]^2 + u[4]^2) + (((x[1] - x[5])^2 + (x[3] - x[7])^2) < 2) * (2 - ((x[1] - x[5])^2 + (x[3] - x[7])^2))^4)))

# indices of inputs that each player controls
player_inputs = (SVector(1,2), SVector(3,4))
# the horizon of the game
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)

# get a solver, choose initial conditions and solve (in about 9 ms with AD)
solver = iLQSolver(g)
x0 = SVector(-2, 0, -5, 0, 2, 0, -5, 0)
converged, trajectory, strategies = solve(g, solver, x0)

# animate the resulting trajectory. Use the `plot_traj` call without @animated to
# get a static plot instead.

x1, y1 = [trajectory.x[i][1] for i in 1:game_horizon], [trajectory.x[i][3] for i in 1:game_horizon]
x2, y2 = [trajectory.x[i][5] for i in 1:game_horizon], [trajectory.x[i][7] for i in 1:game_horizon]

n = 100
ϕ = range(0,stop=2*π,length=n)
x_cir = cos.(ϕ)
y_cir = sin.(ϕ) .+ 2

plt = plot(x1, y1, xaxis = ("x", (-10,10)), yaxis = ("y", (-10,10)), legend = false)
plt = plot(plt, [x2, x_cir], [y2, y_cir])

anim = @animate for i in 1:game_horizon
    scatter(plt, [x1[i], x2[i]], [y1[i], y2[i]], xaxis = ("x", (-10,10)), yaxis = ("y", (-10,10)); k=i, legend = false)
end
gif(anim, "test.gif", fps=10)