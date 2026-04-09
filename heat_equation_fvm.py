import numpy as np

def heat_flux(u_left, u_right, dx, kappa):
    """Compute the diffusive heat flux between adjacent cell values."""
    return -kappa * (u_right - u_left) / dx


def exponential_heat_flux(u_left, u_right, dx, kappa, alpha=-1.0):
    """Compute a nonlinear diffusive flux with gradient magnitude exponent alpha."""
    grad = (u_right - u_left) / dx
    return -kappa * np.sign(grad) * np.power(np.abs(grad) + 1e-12, alpha)


def saturated_heat_flux(u_left, u_right, dx, kappa,  grad_crit=1.0, alpha=1.0):
    """Compute a saturated heat flux with a critical gradient.

    If the temperature gradient magnitude exceeds grad_crit, the flux is clipped
    to the saturated heat flux magnitude q_sat.
    """
    grad = (u_right - u_left) / dx
    grad_mag = np.abs(grad)
    
    linear_flux = -kappa * grad
    
    q_crit = kappa * grad_crit
    q_sat = kappa * 0.8 * grad_crit
    exponential_flux = (q_crit - q_sat) * np.power( grad_mag / grad_crit, -alpha) + q_sat
    exponential_flux *= -np.sign(grad) 

    return np.where(grad_mag < grad_crit, linear_flux, exponential_flux)


def linear_initial_condition(x, start=1.0, end=2.0, length=1.0):
    """Return a linear temperature gradient from start to end over the domain."""
    return start + (end - start) * x / length


def flat_initial_condition(x, value=1.0):
    """Return a flat (constant) initial condition."""
    return np.full_like(x, value)


def sinusoidal_initial_condition(x, amplitude=2.0, phase=0.0, offset=5.0, length=1.0, modes=3):
    """Return a sinusoidal initial condition on the domain [0, length]."""
    return offset + amplitude * np.sin(2 * np.pi * modes * x / length + phase)


def solve_heat_equation_fvm(
    nx=50,
    nt=500,
    t_end=0.1,
    length=1.0,
    kappa=1.0,
    left_bc=0.0,
    right_bc=0.0,
    periodic=False,
    flux=saturated_heat_flux,
    initial_condition=linear_initial_condition,
):
    """Solve the 1D heat equation using a finite volume method.

    The equation is u_t = kappa * u_xx on 0 < x < length.
    By default, Dirichlet boundary conditions are imposed at x=0 and x=length.
    Set periodic=True to use periodic boundary conditions instead.

    initial_condition must be a callable that returns the field values at cell centers.
    flux must be a callable in this file that computes the diffusive flux.
    """
    if not callable(flux):
        raise ValueError("flux must be a callable function.")
    if not callable(initial_condition):
        raise ValueError("initial_condition must be a callable function.")

    dx = length / nx
    x_centers = (np.arange(nx) + 0.5) * dx
    dt = t_end / nt
    stability_limit = dx**2 / (2 * kappa)
    if dt > stability_limit:
        raise ValueError(
            f"Time step dt={dt:.4e} exceeds explicit stability limit {stability_limit:.4e}."
        )

    u = initial_condition(x_centers)

    solutions = [u.copy()]
    times = [0.0]

    for n in range(1, nt + 1):
        # Compute face fluxes for each cell interface
        u_ext = np.empty(nx + 2)
        if periodic:
            u_ext[0] = u[-1]
            u_ext[1:-1] = u
            u_ext[-1] = u[0]

            # For periodic boundaries, all faces are separated by a full cell width.
            dx_faces = np.full(nx + 1, dx)
        else:
            u_ext[0] = left_bc
            u_ext[1:-1] = u
            u_ext[-1] = right_bc

            # Face distances: dx/2 for boundary faces, dx for interior.
            dx_faces = np.full(nx + 1, dx)
            dx_faces[0] = dx / 2
            dx_faces[-1] = dx / 2

        face_fluxes = flux(u_ext[:-1], u_ext[1:], dx_faces, kappa)
        dudt = -(face_fluxes[1:] - face_fluxes[:-1]) / dx
        u = u + dt * dudt

        solutions.append(u.copy())
        times.append(n * dt)

    return x_centers, np.array(times), np.array(solutions)
