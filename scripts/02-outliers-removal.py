# Remove outliers based on the parameters specified on the outliers_removal_v2.ipynb notebook

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    import itesoAQ


def main():

    preprocessed_data = itesoAQ.OutliersRemovalTools()
    preprocessed_data.fit_data()
    preprocessed_data.remove_std_outliers_v2()
    preprocessed_data.export_to_csv()


if __name__ == "__main__":
    main()
