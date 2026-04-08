import numpy as np

def heat_flux(u_left, u_right, dx, alpha):
    """Compute the diffusive heat flux between adjacent cell values."""
    return -alpha * (u_right - u_left) / dx


def linear_initial_condition(x, start=1.0, end=1.0, length=1.0):
    """Return a linear temperature gradient from start to end over the domain."""
    return start + (end - start) * x / length


def solve_heat_equation_fvm(
    nx=50,
    nt=500,
    t_end=0.1,
    length=1.0,
    alpha=1.0,
    left_bc=0.0,
    right_bc=0.0,
    periodic=False,
    flux=heat_flux,
    initial_condition=linear_initial_condition,
):
    """Solve the 1D heat equation using a finite volume method.

    The equation is u_t = alpha * u_xx on 0 < x < length.
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
    stability_limit = dx**2 / (2 * alpha)
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
        else:
            u_ext[0] = left_bc
            u_ext[1:-1] = u
            u_ext[-1] = right_bc

        face_fluxes = flux(u_ext[:-1], u_ext[1:], dx, alpha)

        dudt = -(face_fluxes[1:] - face_fluxes[:-1]) / dx
        u = u + dt * dudt

        solutions.append(u.copy())
        times.append(n * dt)

    return x_centers, np.array(times), np.array(solutions)
