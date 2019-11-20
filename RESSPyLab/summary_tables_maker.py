import pandas as pd
import numpy as np
from uvc_model import calc_phi_total


def summary_tables_maker_uvc(material_definition, x_file_paths, data, peeq='sat'):
    """ Prints to screen the summary tables for the material optimization in LaTeX format for the updated VC model.

    :param dict material_definition: Contains information about each material.
    :param list x_file_paths: (str) Path for the files that contain the x values for each material.
    :param list data: (list, pd.DataFrame) The test data used for calibration of each of the materials.
    :param str or float peeq: If 'sat' then calculates the metrics at model saturation, otherwise a finite equivalent
                                plastic strain.
    :return list: The first and second summary tables.

    Notes:
        - material_definition:
            'material_id': (list, str) Identifier for each material.
            'load_protocols': (list, str) Labels of the load protocols used, see [1] for definitions.
        - The metrics in Table 2 are defined in [2].
        - If a finite peeq is provided, the metrics are calculated assuming that peeq increases monotonically
        to the provided value.

    References:
        [1] de Castro e Sousa and Lignos (2017), On the inverse problem of classic nonlinear plasticity models.
        [2] de Castro e Sousa and Lignos (2018), Constrained optimization in metal plasticity inverse problems.
    """
    # Output column labels
    parameter_labels = [r'$E$[GPa]', r'$\sigma_{y,0}$[MPa]', r'$Q_\infty$[MPa]', r'$b$',
                        r'$D_\infty$[MPa]', r'$a$',
                        r'$C_1$[MPa]', r'$\gamma_1$', r'$C_2$[MPa]', r'$\gamma_2$',
                        r'$C_3$[MPa]', r'$\gamma_3$', r'$C_4$[MPa]', r'$\gamma_4$']
    metric_labels = [r'$\sigma_{y,0}$[MPa]', r'$\sigma_{sat}$[MPa]', r'$\sigma_{hard}$[MPa]',
                     r'$\rho^{sat}_{yield}$', r'$\rho^{sat}_{iso}$', r'$\rho^{sat}_{kin}$', r'$\rho^{sat}_{D}$']
    n_basic_param = 6
    tab_1, tab_2 = _table_maker(material_definition, x_file_paths, data, parameter_labels, metric_labels,
                                n_basic_param, calc_upd_metrics=True, peeq=peeq)

    return [tab_1, tab_2]


def summary_tables_maker_vc(material_definition, x_file_paths, data, peeq='sat'):
    """ Prints to screen the summary tables for the material optimization in LaTeX format for the original VC model.

    :param dict material_definition: Contains information about each material.
    :param list x_file_paths: (str) Path for the files that contain the x values for each material.
    :param list data: (list, pd.DataFrame) The test data used for calibration of each of the materials.
    :param str or float peeq: If 'sat' then calculates the metrics at model saturation, otherwise a finite equivalent
                                plastic strain.
    :return list: The first and second summary tables.

    Notes:
        - material_definition:
            'material_id': (list, str) Identifier for each material.
            'load_protocols': (list, str) Labels of the load protocols used, see [1] for definitions.
        - The metrics in Table 2 are defined in [2].
        - If a finite peeq is provided, the metrics are calculated assuming that peeq increases monotonically
        to the provided value.

    References:
        [1] de Castro e Sousa and Lignos (2017), On the inverse problem of classic nonlinear plasticity models.
        [2] de Castro e Sousa and Lignos (2018), Constrained optimization in metal plasticity inverse problems.
    """
    # Output column labels
    parameter_labels = [r'$E$[GPa]', r'$\sigma_{y,0}$[MPa]', r'$Q_\infty$[MPa]', r'$b$',
                        r'$C_1$[MPa]', r'$\gamma_1$', r'$C_2$[MPa]', r'$\gamma_2$',
                        r'$C_3$[MPa]', r'$\gamma_3$', r'$C_4$[MPa]', r'$\gamma_4$']
    metric_labels = [r'$\sigma_{y,0}$[MPa]', r'$\sigma_{sat}$[MPa]', r'$\sigma_{hard}$[MPa]',
                     r'$\rho^{sat}_{yield}$', r'$\rho^{sat}_{iso}$', r'$\rho^{sat}_{kin}$']
    n_basic_param = 4
    tab_1, tab_2 = _table_maker(material_definition, x_file_paths, data, parameter_labels, metric_labels,
                                n_basic_param, calc_upd_metrics=False, peeq=peeq)

    return [tab_1, tab_2]


def _table_maker(material_definition, x_file_paths, data, parameter_labels, metric_labels, num_basic_param,
                 calc_upd_metrics, peeq='sat'):
    """ Base function to generate the tables. """
    # Set some options for the display
    pd.set_option('display.max_columns', 12)
    pd.set_option('display.width', 300)
    pd.set_option('display.float_format', '{:0.2f}'.format)

    # Extract the properties from the definition
    material_id = material_definition['material_id']
    load_protocols = material_definition['load_protocols']

    # Make the first table
    phi_values = []
    summary_table = pd.DataFrame()
    for i, f in enumerate(x_file_paths):
        x = pd.read_csv(f, delimiter=' ')
        x = np.array(x.iloc[-1])
        # Sort the backstresses so that the largest gamma value is first
        gammas = x[num_basic_param + 1::2]
        ind = np.flipud(np.argsort(gammas))
        # Exchange the gammas
        x[num_basic_param + 1::2] = x[2 * ind + num_basic_param + 1]
        # Exchange the Cs
        x[num_basic_param::2] = x[2 * ind + num_basic_param]
        temp_table = pd.DataFrame(x, columns=(material_id[i],)).transpose()
        summary_table = summary_table.append(temp_table)
        if calc_upd_metrics:
            phi_values.append(calc_phi_total(x, data[i]))
        else:
            x_phi = np.insert(x, 4, [0., 1.])
            phi_values.append(calc_phi_total(x_phi, data[i]))
    # Rename the columns
    summary_table.columns = parameter_labels[:len(summary_table.columns)]
    # Add the phi values
    summary_table.insert(0, r'$\bar{\varphi}$[\%]', phi_values)
    # Add the load protocols
    summary_table.insert(0, r'LP', load_protocols)
    # Make the elastic modulus in GPa
    summary_table[parameter_labels[0]] = summary_table[parameter_labels[0]] / 1000.
    # Set the index name to Materials
    summary_table.index.name = 'Material'
    print summary_table.to_latex(escape=False)

    # Make the second table
    summary_table_2 = pd.DataFrame()
    for i, f in enumerate(x_file_paths):
        # Calculate the comparison metrics
        data_row = list(summary_table.iloc[i])
        s_y0 = data_row[3]
        hm = _hard_metric_at_peeq(data_row, num_basic_param, calc_upd_metrics, peeq)
        sigma_sat = hm['sigma_sat']
        sigma_hard = hm['sigma_hard']
        rho_yield = hm['rho_yield']
        rho_iso = hm['rho_iso']
        rho_kin = hm['rho_kin']
        rho_d = hm['rho_d']
        if calc_upd_metrics:
            new_row = np.array([s_y0, sigma_sat, sigma_hard, rho_yield, rho_iso, rho_kin, rho_d])
        else:
            new_row = np.array([s_y0, sigma_sat, sigma_hard, rho_yield, rho_iso, rho_kin])
        # Add the values to the table for each material
        temp_table = pd.DataFrame(new_row, columns=(material_id[i],)).transpose()
        summary_table_2 = summary_table_2.append(temp_table)
    # Rename the columns
    summary_table_2.columns = metric_labels
    # Set the index name to Materials
    summary_table_2.index.name = 'Material'
    print summary_table_2.to_latex(escape=False)

    return [summary_table, summary_table_2]


def _hard_metric_at_peeq(x, num_basic_param, calc_upd_metrics, peeq='sat'):
    """ Calculates the hardening metrics for both the original and updated Voce-Chaboche models.

    :param list x: Row of data from table_maker function.
    :param int num_basic_param: Number of non-backstress related parameters in the model.
    :param bool calc_upd_metrics: If True then calculates the rho_d metric, if False then sets it to 0.
    :param str or float peeq: If 'sat' then calculates the metrics at model saturation, otherwise a finite equivalent
                                plastic strain.
    :return dict: Hardening metrics.

    Notes:
        - If a finite peeq is provided, the metrics are calculated assuming that peeq increases monotonically
        to the provided value.
    """
    cols_before_kin = num_basic_param + 2
    num_backstresses = (len(x) - cols_before_kin) // 2
    s_y0 = x[3]
    if peeq == 'sat':
        # Calculate values assuming fully saturated
        q_inf = x[4]
        if calc_upd_metrics:
            d_inf = x[6]
        else:
            d_inf = 0.
        sum_kin = 0.
        for j in range(num_backstresses):
            c_j = x[cols_before_kin + 2 * j]
            g_j = x[cols_before_kin + 1 + 2 * j]
            sum_kin += c_j / g_j
    else:
        # Calculate values at finite equivalent plastic strain (monotonically increasing)
        q_inf = x[4] * (1. - np.exp(-x[5] * peeq))
        if calc_upd_metrics:
            d_inf = x[6] * (1. - np.exp(-x[7] * peeq))
        else:
            d_inf = 0.
        sum_kin = 0.
        for j in range(num_backstresses):
            c_j = x[cols_before_kin + 2 * j]
            g_j = x[cols_before_kin + 1 + 2 * j]
            sum_kin += c_j / g_j * (1. - np.exp(-g_j * peeq))

    # Calculate all the metrics
    sigma_sat = s_y0 + q_inf - d_inf + sum_kin
    sigma_hard = q_inf + sum_kin
    rho_yield = sigma_sat / s_y0
    rho_iso = q_inf / sigma_hard
    rho_kin = sum_kin / sigma_hard
    rho_d = d_inf / (q_inf + sum_kin)
    return {'sigma_sat': sigma_sat, 'sigma_hard': sigma_hard,
            'rho_yield': rho_yield, 'rho_iso': rho_iso, 'rho_kin': rho_kin, 'rho_d': rho_d}
