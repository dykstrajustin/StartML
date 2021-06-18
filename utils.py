import numpy as np
from scipy.interpolate import splrep, sproot
from pysb import Model


def get_momp(tspan, traj, check_fraction=False, model=None):
    tod = [calc_tod(tspan, i, check_fraction, model) for i in traj.T]
    return np.array(tod)


def calc_tod(time_span, traj, check_fraction=False, model=None):
    """ Calculates time of death of PARP trajectory

    Assumes PARP is 90% cleaved, if not returns nans.

    Parameters
    ----------
    time_span: np.array
    traj : pd.Dataframe
    check_fraction: bool
    model: pysb.Model


    Returns
    -------
    (time_of_death, time_of_switch, fraction_activated_smac)
    (np.float, np.float, np.float)
    """
    if check_fraction:
        if not isinstance(model, Model):
            print("Must provide the earm model to function")
            return
        # assume that at least 90% of PARP must be cleaved
        frac_cleaved = traj['CPARP_total'].values[-1] / \
                       model.parameters['PARP_0'].value
        if frac_cleaved < .90:
            return [np.nan, np.nan, np.nan]


        frac_cleaved = traj['tBid_total'].values[-1] / \
                       model.parameters['Bid_0'].value

        if frac_cleaved < .95:
            return [np.nan, np.nan, np.nan]

        frac_cleaved = traj['cSmac_total'].values[-1] / \
                       model.parameters['mSmac_0'].value

        if frac_cleaved < .95:
            return [np.nan, np.nan, np.nan]

    ysim_momp_norm = traj['CPARP_total'].values / \
                     np.nanmax(traj['CPARP_total'].values)
    st, sc, sk = splrep(time_span, ysim_momp_norm)
    try:
        t10 = sproot((st, sc - 0.10, sk))[0]
        t90 = sproot((st, sc - 0.90, sk))[0]
        # time of death  = halfway point between 10 and 90%
        time_of_death = (t10 + t90) / 2

        # time of switch is time between 90 and 10 %
        time_of_switch = t90 - t10
    except IndexError:
        time_of_death = np.nan
        time_of_switch = np.nan
    # final fraction of aSMAC (last value)
    return [time_of_death, time_of_switch, traj['CPARP_total'].values[-1]]
