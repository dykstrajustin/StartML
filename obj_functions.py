import os
import numpy as np
import pandas as pd
from pysb.examples.earm_1_0 import model
from scipy.interpolate import splrep, sproot
from scipy import stats
from pysb.simulator import ScipyOdeSimulator


# load experimental data
data_path = os.path.join(os.path.dirname(__file__), 'data',
                         'EC-RP_IMS-RP_IC-RP_data_for_models.csv')

exp_data = pd.read_csv(data_path, index_col=False)

# timepoints for simulation. These must match the experimental data.
tspan = exp_data['Time'].values.copy()

rate_params = model.parameters_rules()
param_values = np.array([p.value for p in model.parameters])
rate_mask = np.array([p in rate_params for p in model.parameters])
starting_position = np.log10(param_values[rate_mask])

solver = ScipyOdeSimulator(model, tspan, integrator='lsoda', verbose=False,
                           use_analytic_jacobian=False, compiler='cython',
                           integrator_options={"rtol": 1e-6, "atol": 1e-6})

# Mean and variance of Td (delay time) and Ts (switching time) of MOMP, and
# yfinal (the last value of the IMS-RP trajectory)
momp_data = np.array([9810.0, 180.0, model.parameters['mSmac_0'].value])
momp_var = np.array([7245000.0, 3600.0, 1e4])

time_of_death = 3. * 3600
# std/mean
cv = .18
std_tod = cv * time_of_death


def calc_tod(traj):
    """ Calculates time of death of PARP trajectory

    Assumes PARP is 90% cleaved, if not returns nans.

    Parameters
    ----------
    traj : pd.Dataframe

    Returns
    -------
    time_of_switch : np.float
    """
    check_frac = True

    if check_frac:
        # assume that at least 90% of PARP must be cleaved
        frac_cleaved = traj['CPARP_total'].values[-1] /\
                       model.parameters['PARP_0'].value

        if frac_cleaved < .90:
            return np.nan

        frac_cleaved = traj['tBid_total'].values[-1] / \
                       model.parameters['Bid_0'].value

        if frac_cleaved < .90:
            return np.nan

        frac_cleaved = traj['cSmac_total'].values[-1] / \
                       model.parameters['mSmac_0'].value

        if frac_cleaved < .90:
            return np.nan

    momp_traj = traj['cSmac_total'].values
    momp_sim = _fit_spline(momp_traj)
    td = momp_sim[0]
    return td


def extract_momp(traj_dist):
    tds = []
    for sim, df in traj_dist.groupby('simulation'):
        tds.append(calc_tod(df))
    return np.array(tds)


def momp_distance(traj_dist):
    # get parp timing
    tds = extract_momp(traj_dist)
    n_sim = len(tds)
    # ensure not zeros
    tds = tds[~np.isnan(tds)]

    n_true = len(tds)
    # make sure 50% of cells actually reach death (90% cleavage)
    if n_true < n_sim/2:
        return 100000

    # generate sample from experimental values (must be same size as simulated,
    # thus why we generate each time function is called. Some cells may not die
    # in the simulations
    tod_exp = np.random.normal(time_of_death, std_tod, size=n_true)
    return stats.energy_distance(tds, tod_exp)
    # return 1 - stats.ks_2samp(tds, tod_exp)[1]
    # return wasserstein_distance(tds, tod_exp)


def likelihood(position):
    param_values[rate_mask] = 10 ** position.copy()
    traj = solver.run(param_values=param_values)

    # normalize trajectories
    bid_traj = traj.observables['tBid_total'] / model.parameters['Bid_0'].value
    cparp_traj = traj.observables['CPARP_total'] / model.parameters['PARP_0'].value
    momp_traj = traj.observables['cSmac_total']
    return calc_error(bid_traj, cparp_traj, momp_traj)


def compare_avgs(traj):
    traj.reset_index(inplace=True)
    # normalize trajectories
    bid_traj = pd.pivot_table(
        traj, values='tBid_total', index='time',
        columns='simulation'
    ).mean(axis=1).values / model.parameters['Bid_0'].value

    cparp_traj = pd.pivot_table(
        traj, values='CPARP_total', index='time',
        columns='simulation'
    ).mean(axis=1).values / model.parameters['PARP_0'].value

    momp_traj = pd.pivot_table(
        traj, values='cSmac_total', index='time',
        columns='simulation'
    ).mean(axis=1).values

    return calc_error(bid_traj, cparp_traj, momp_traj)


def calc_error(bid_traj, cparp_traj, momp_traj):
    # calculate chi^2 distance for each time course
    e1 = np.sum((exp_data['norm_IC-RP'] - bid_traj) ** 2 /
                (2 * exp_data['nrm_var_IC-RP'])) / len(bid_traj)

    e2 = np.sum((exp_data['norm_EC-RP'] - cparp_traj) ** 2 /
                (2 * exp_data['nrm_var_EC-RP'])) / len(cparp_traj)

    # Here we fit a spline to find where we get 50% release of MOMP reporter
    if np.nanmax(momp_traj) == 0:
        e3 = 10000000
    else:
        momp_sim = _fit_spline(momp_traj)
        e3 = np.sum((momp_data - momp_sim) ** 2 / (2 * momp_var)) / 3

    return e1 + e2 + e3


def _fit_spline(momp_traj):
    ysim_momp_norm = momp_traj / np.nanmax(momp_traj)
    st, sc, sk = splrep(tspan, ysim_momp_norm)
    try:
        t10 = sproot((st, sc - 0.10, sk))[0]
        t90 = sproot((st, sc - 0.90, sk))[0]
    except IndexError:
        t10 = 0
        t90 = 0

    # time of death  = halfway point between 10 and 90%
    td = (t10 + t90) / 2

    # time of switch is time between 90 and 10 %
    ts = t90 - t10
    return [td, ts, momp_traj[-1]]