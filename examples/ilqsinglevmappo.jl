using iLQGames
using Plots
using LinearAlgebra
using CSV
import iLQGames: dx
using DelimitedFiles


mappo = readdlm("mappotraj.csv", ',', Float64)
mappo = mappo'



# parametes: number of states, number of inputs, sampling time, horizon
nx, nu, ΔT, game_horizon = 7, 4, 0.1, 20

# setup the dynamics
struct Drone <: ControlSystem{ΔT,nx,nu} end
# state: (px, ux, py, uy, pz, uz)
dx(cs::Drone, x, u, t) = SVector(x[2], u[1], x[4], u[2], x[6], u[3], u[4])
dynamics = Drone()

# Define the target circle areas
target_centers = [SVector(1.0, 3.0, 1.15), SVector(1.0, 0.0, 0.5), SVector(-1.0, 0.0, 0.5), SVector(-0.5, 1.5, 0.5), SVector(-1.0, 3.0, 0.5)]
normal_vectors = [SVector(1, -1, 0), SVector(0, -1, 0), SVector(-1, 1, 0), SVector(0, 1, 0), SVector(0, 1, 0)]
current_target = 1  # Track current area for both drones (initially 1 for both)
radius = 0.4  # Circle radius for areas

# Function to update target area if the drone reaches the current target
function update_target(x, current_gate, target_centers)
    target_pos = target_centers[current_gate]
    normal_vec = normal_vectors[current_gate]
    if dot((SVector(x[1], x[3], x[5]) - target_pos), normal_vec) >= 0
        if current_gate == length(target_centers)
            current_target = 1
        else
            current_target = current_gate + 1
        end
    end
end

# Function to define player cost
function drone_cost_function(current_target, target_centers, x)
    return (
        FunctionPlayerCost((g, x, u, t) -> (
            # Cost for drone 1 to reach its target area while behind the gate
            #(dot((SVector(x[1], x[3], x[5]) - target_centers[current_target]), normal_vectors[current_target]) < 0) * 
            ((x[1] - target_centers[current_target][1])^2 + (x[3] - target_centers[current_target][2])^2 + (x[5] - target_centers[current_target][3])^2) +
            #(dot((SVector(x[1], x[3], x[5]) - target_centers[current_target[1]]), normal_vectors[current_target[1]]) >= 0) * #(norm(SVector(x[1], x[3], x[5]) - target_centers[current_target[1]]) < radius) *
            #1.3 * ((x[1] - target_centers[(current_target[1]%5)+1][1])^2 + (x[3] - target_centers[(current_target[1]%5)+1][2])^2 + (x[5] - target_centers[(current_target[1]%5)+1][3])^2) + 
            0.5 * (u[1]^2 + u[2]^2 + u[3]^2) +  # Penalize control effort
            # Speed Control
            (norm(SVector(x[2], x[4], x[6])) > 1) * norm(SVector(x[2], x[4], x[6]))
        )),
        FunctionPlayerCost((g, x, u, t) -> (
            x[7] + u[4]
        ))
    )
end

# indices of inputs that each player controls
player_inputs = (SVector(1,2,3), SVector(4))
# Define initial conditions for both drones
# x0 = SVector(-6.0, 0.0, -10.0, 0.0, -4.0, 0.0, -10.0, 0.0)  
x0 = [-0.25, 0.0, 3.5, 0.0, 0.5, 0.0, 0.0] # Initial positions and velocities for both drones

# Simulate iteratively until both drones reach the final target area
maximum_horizon = 400
trajectory = []
while length(trajectory) < maximum_horizon
    g = GeneralGame(game_horizon, player_inputs, dynamics, drone_cost_function(current_target, target_centers, x0))
    solver = iLQSolver(g)
    # Solve for the next 20 steps
    converged, full_trajectory, strategies = solve(g, solver, SVector(x0...))
    
    # Execute one step
    push!(trajectory, full_trajectory.x[1])

    # Update the state after each step
    global x0 = collect(full_trajectory.x[2])

    # Update the target if the drones reached their current area
    update_target(x0, current_target, target_centers)
end

# Prepare for animation
n = 100
ϕ = range(0, stop=2π, length=n)
radius = 0.4
circles = [(cos.(ϕ) * radius, zeros(n), sin.(ϕ) * radius) for _ in 1:5]


function get_rot_mtx(theta)
    return (
        [cos.(theta) -sin.(theta) 0; sin.(theta) cos.(theta) 0; 0 0 1]
    )
end

thetas = SVector(π/4, 0.0, π/4, 0.0, 0.0)

anim = @animate for t in eachindex(trajectory)
    plot3d(
        xlims=(-3, 3), ylims=(-2, 5), zlim = (0, 3),
        xlabel="x", ylabel="y", legend=false, 
        title="Drone Trajectory"
    )
    
    # Plot the target areas (circles)
    for i in 1:5
        rotmtx = get_rot_mtx(thetas[i])
        gate = rotmtx * [axis for axis in circles[i]]
        plot3d!(gate[1] .+ target_centers[i][1], gate[2] .+ target_centers[i][2], gate[3] .+ target_centers[i][3], color=:gray, lw=1)
    end
    
    # Drone 1 and Drone 2 positions at each step
    plot!([trajectory[t][1]], [trajectory[t][3]], [trajectory[t][5]], marker=:circle, color=:blue, label="Drone 1")
end

# Save animation as gif
gif(anim, "drone_trajectory.gif", fps=10)