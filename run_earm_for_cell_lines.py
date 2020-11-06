import pandas as pd
import numpy as np
from pysb.examples.earm_1_0 import model
import matplotlib.pyplot as plt
from pysb.simulator import OpenCLSSASimulator, ScipyOdeSimulator

scales = pd.read_csv('normalized_ratios.csv')
scales.set_index('Cell line', inplace=True)
scales = scales.to_dict('index')
original_ic = {j.name: j.value for i, j in model.initial_conditions}
print(original_ic)
mapping = {
    'BAX': ['Bax_0'],
    'BCL2': ['Bcl2_0', 'Bcl2c_0'],
    'BID': ['Bid_0'],
    'CYCS': ['mCytoC_0'],
    'FADD': ['Bcl2_0'],
    'CASP3': ['pC3_0'],
    'CASP6': ['pC6_0'],
    'CASP8': ['pC8_0'],
    'CASP9': ['pC9_0'],
    'CFLAR': ['flip_0'],  # FLIP
    'XIAP': ['XIAP_0'],
    'DIABLO': ['mSmac_0'],  # SMAC
    'TNFRSF10A': [],  # DR4
    'TNFRSF10B': ['pR_0'],  # DR5
    'PARP1': ['PARP_0'],
    'APAF1': ['Apaf_0'],
    'BFAR': ['BAR_0']
}


def create_plot(save_name):
    n_sim = 100
    tspan = np.linspace(0, 20000, 101)
    name = 'CPARP_total'
    sim = OpenCLSSASimulator(model, tspan=tspan, verbose=True)
    traj = sim.run(tspan=tspan, number_sim=n_sim)
    traj.dataframe.to_csv('traj/{}.csv'.format(save_name.replace('.png', '')))
    result = np.array(traj.observables)[name]

    x = np.array([tr[:] for tr in result]).T

    plt.plot(tspan, x, '0.5', lw=2, alpha=0.25)  # individual trajectories
    plt.plot(tspan, x.mean(axis=1), 'b', lw=3, label='mean')
    plt.plot(tspan, x.max(axis=1), 'k--', lw=2, label="min/max")
    plt.plot(tspan, x.min(axis=1), 'k--', lw=2)

    sol = ScipyOdeSimulator(model, tspan)
    traj = sol.run()

    plt.plot(tspan, np.array(traj.observables)[name], label='ode', color='red')

    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()


cell_lines = {'HDLM-2', 'HL-60', 'RT4', 'A549', 'SiHa', 'THP-1', 'PC-3',
              'K-562', 'WM-115', 'REH', 'MOLT-4', 'HeLa', 'EFO-21', 'AN3-CA',
              'CAPAN-2', 'RPMI-8226', 'Daudi', 'MCF7'}
for cell_line in cell_lines:
    print(cell_line)
    save_name = cell_line.replace('-', '_')
    for protein, scale_factor in scales[cell_line].items():
        for ic in mapping[protein]:
            val = original_ic[ic]
            if scale_factor < 0:
                scale_factor = -1/scale_factor
            model.parameters[ic].value = scale_factor * val
    create_plot(f'{save_name}.png')
