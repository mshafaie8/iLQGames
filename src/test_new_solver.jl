using iLQGames
import iLQGames: dx
using Plots
using ForwardDiff
using iLQGames:
    SystemTrajectory
using iLQGames:
    LinearSystem
using Infiltrator
using Optim
using LinearAlgebra
using Distributed
using Dates
using Statistics


"Forward Game Problem: generate expert demo"
# parametes: number of states, number of inputs, sampling time, horizon
nx, nu, ΔT, game_horizon = 8, 4, 0.1, 40

# setup the dynamics
struct DoubleUnicycle <: ControlSystem{ΔT,nx,nu} end
# state: (px, py, phi, v)
dx(cs::DoubleUnicycle, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4])
dynamics = DoubleUnicycle()
costs = (FunctionPlayerCost((g, x, u, t) -> (  8*(x[5]-0)^2  +  2*(u[1]^2 + u[2]^2) )),
        FunctionPlayerCost((g, x, u, t) -> (  4*(x[5]-x[1])^2  +  4*(x[8]-1)^2 + 2*(u[3]^2 + u[4]^2) ))   )

# indices of inputs that each player controls
player_inputs = (SVector(1,2), SVector(3,4))
# the horizon of the game
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)



# get a solver, choose initial conditions and solve (in about 9 ms with AD)
x0 = SVector(0, 0.5, pi/2, 1,       1, 0, pi/2, 1)


# if we want to solve an LQ game, then consider running the following 2 lines of codes:
solver = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf)
c, expert_traj, strategies = solve(g, solver, x0)