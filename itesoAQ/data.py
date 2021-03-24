################################################################################
# Module: Data gathering and treatment
# updated: 22/02/2021
################################################################################

import os
import pandas as pd
import numpy as np
import xlrd
import math
from . import utils

from datetime import datetime, timedelta


def daterange(start_date, end_date, interval='hour', lapse=1):
    """Function that creates a list with dates from start to end according to interval.

    Args:
        start_date (datetime): start datetime value in format yyyy-mm-dd hh:mm.
        end_date (datetime): end datetime value in format yyyy-mm-dd hh:mm.
        interval (str, optional): interval that will be added from start to end date, it can be day or hour. Defaults to 'hour'.
        lapse (int, optional): time lapse increment. Defaults to 1.

    Returns:
        list: list of dates from start_date to end_date according to interval and lapse.
    """

    if interval == 'hour':
        delta = timedelta(hours=lapse)
    elif interval == 'day':
        delta = timedelta(days=lapse)

    date_list = []
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += delta

    return date_list


def database_clean(interval='hour'):
    """Function that creates a new and clean database in .csv format from SIMAJ air quality database for the years 2014 to 2019.

    Args:
        interval (str, optional): it sets the interval for the new database, it can take 'hour' or 'day'. Defaults to 'hour'.
    """

    dir_gdl = '../data/raw/simaj/'

    # list for station names
    # est_list = ['AGU','ATM','CEN','LPIN','LDO','MIR','OBL','SFE','TLA','VAL']

    # check for file or directory in dir_gdl
    for file in os.listdir(dir_gdl):

        if file.endswith('.xlsx'):
            # SIMAJ data is in xls and in different sheets
            xls = xlrd.open_workbook(r'' + dir_gdl + file, on_demand=True)
            sheets = xls.sheet_names()  # creates list form sheet names

        else:
            continue

        year = file[6:10]  # gathers the year from the file name

        start_date = datetime(int(year), 1, 1, 00, 00)  # start date for array
        end_date = datetime(int(year), 12, 31, 23, 00)  # end date for array

        # creates array for every hour of a given year
        date_array = np.array(
            daterange(start_date, end_date, interval=interval))

        # creates new df in which air quality data for all stations will be saved
        df = pd.DataFrame(index=np.array(range(0, len(date_array))), columns=[
            'FECHA', 'HORA', 'O3', 'PM10', 'CO', 'NO2', 'SO2', 'TMP', 'HR', 'WS', 'WD'])

        # datetime id counter
        dt_id = 0

        # saves date and hour data to df
        for t in date_array:
            df['FECHA'].iloc[dt_id] = t.date()
            df['HORA'].iloc[dt_id] = '{:02d}:{:02d}'.format(t.hour, t.minute)
            dt_id += 1

        # sets index before stack
        df = df.set_index(['FECHA', 'HORA'])

        # stacks DataFrame so for every date there are 5 rows with criterion pollutants
        df = df.stack(dropna=False)
        df = df.reset_index().rename(columns={'level_2': 'PARAM'})

        # sets index depending on interval type
        if interval == 'hour':
            df = df.set_index(['FECHA', 'HORA', 'PARAM'])
        else:
            df = df.set_index(['FECHA', 'PARAM']).drop(columns='HORA')

        for s in sheets:

            # reads excel with data and sets empty cells as nan
            gdl_data = pd.read_excel(
                dir_gdl + file, sheet_name=s).replace(r'^\s*$', np.nan, regex=True)

            gdl_data.rename(
                columns={gdl_data.columns[0]: 'FECHA', gdl_data.columns[1]: 'HORA'}, inplace=True)

            # removes : from columns
            gdl_data.columns = [col.replace(':', '')
                                for col in gdl_data.columns]

            '''gdl_data = gdl_data[['FECHA', 'HORA', 'O3',
                                 'NO2', 'SO2', 'PM10', 
                                 'CO','TMP','HR','WS','WD']]'''  # filters data

            # fixes date
            if int(year) == 2018 and s == 'CEN':
                gdl_data['FECHA'].iloc[4112] = datetime.strptime(
                    '2018-06-21', '%Y-%m-%d')

            # sets FECHA column as date
            gdl_data['FECHA'] = gdl_data['FECHA'].dt.date

            # iterates over hour data and gives it the appropiate format
            for i in range(len(gdl_data)):

                try:
                    # tries to set hour in hh:mm format according to HORA column
                    gdl_data['HORA'].iloc[i] = '{:02d}:{:02d}'.format(
                        gdl_data['HORA'].iloc[i].hour, gdl_data['HORA'].iloc[i].minute)

                except:

                    # nested tries
                    try:
                        # if HORA is float or int it gives it datetime format
                        time_type = gdl_data['HORA'].iloc[i]
                        seconds = (time_type - 25569) * 86400.0
                        time_datetime = datetime.utcfromtimestamp(seconds)

                        gdl_data['HORA'].iloc[i] = '{:02d}:{:02d}'.format(
                            time_datetime.hour, time_datetime.minute)

                    except:

                        # if HORA is nan it takes previous date and hour and adds +1h and stores it in date and time
                        prev_date = gdl_data['FECHA'].iloc[i - 1]
                        prev_hour = gdl_data['HORA'].iloc[i - 1]

                        prev_datetime = str(prev_date) + ' ' + str(prev_hour)

                        date_datetime = datetime.strptime(
                            prev_datetime, '%Y-%m-%d %H:%M')

                        new_datetime = date_datetime + timedelta(hours=1)

                        gdl_data['FECHA'].iloc[i] = new_datetime.date()
                        gdl_data['HORA'].iloc[i] = '{:02d}:{:02d}'.format(
                            new_datetime.hour, new_datetime.minute)

            # stacks gdl DataFrame so for every date there are 10 rows with criterion pollutants
            gdl_stack = pd.DataFrame(gdl_data.set_index(
                ['FECHA', 'HORA']).stack([0], dropna=False))

            # changes name from stacked column with concentration information
            gdl_stack = gdl_stack.reset_index().rename(columns={'level_2': 'PARAM',
                                                                0: s})

            gdl_stack['FECHA'] = pd.to_datetime(gdl_stack['FECHA'])

            # because the data base contains dates out from the analyzed year the DataFrame is filtered
            gdl_stack = gdl_stack[gdl_stack['FECHA'].dt.year == int(
                file[6:10])]

            # removes sapces from sheet names and sets columns as numbers avoiding spaces
            gdl_stack[s] = pd.to_numeric(
                gdl_stack[s], errors='coerce')

            if interval == 'day':
                gdl_stack = gdl_stack.drop(columns=['HORA'])
                gdl_stack = gdl_stack.groupby(['FECHA', 'PARAM']).mean()
            else:
                # it groups data to avoid doble dates in air quality database
                gdl_stack = gdl_stack.groupby(
                    ['FECHA', 'HORA', 'PARAM']).mean()

            # adds data from gdl_stack for a specified year to all_data which
            # will contain information for every year
            df = pd.merge(df, gdl_stack, how='left',
                          left_index=True, right_index=True)

        # drops coloumn 0 which is created when stacking df
        df = df.drop(columns=0)
        # saves data for all years, stations and parameters
        df.to_csv('../data/processed/' +
                  file[6:10] + '_' + interval + '.csv')


def restructure_database(interval='hour'):
    """Function that takes cleaned databases for air quality data and restructures it into a single DataFrame.

    Args:
        interval (str, optional): it sets the interval for the new database, it can take 'hour' or 'day'. Defaults to 'hour'.

    Returns:
        pandas.DataFrame: pandas DataFrame with columns: FECHA, HORA (depending on interval), PARAM, EST_SIMAJ, CONC, LONG, LAT.
    """
    dir_gdl = '../data/processed/'

    simaj_reestructurado_all = pd.DataFrame()

    # check for file or directory in dir_gdl
    for file in os.listdir(dir_gdl):

        if file.endswith('.csv'):

            if interval in file:

                # read csv of air quality data according to interval
                simaj_data = pd.read_csv(dir_gdl + file)

                simaj_reestructurado = simaj_data.copy()

                # stack stations according to interval
                if interval == 'hour':

                    simaj_reestructurado = pd.DataFrame(simaj_reestructurado.set_index(
                        ['FECHA', 'HORA', 'PARAM']).stack(dropna=False))

                    simaj_reestructurado.reset_index(inplace=True)

                    simaj_reestructurado.rename(
                        columns={'level_3': 'EST_SIMAJ', 0: 'CONC'}, inplace=True)

                else:

                    simaj_reestructurado = pd.DataFrame(
                        simaj_reestructurado.set_index(['FECHA', 'PARAM']).stack(dropna=False))

                    simaj_reestructurado.reset_index(inplace=True)

                    simaj_reestructurado.rename(
                        columns={'level_2': 'EST_SIMAJ', 0: 'CONC'}, inplace=True)

                simaj_reestructurado_all = simaj_reestructurado_all.append(
                    simaj_reestructurado)

            # read df with stations coordinates
            stations_simaj = pd.read_csv('../data/raw/estaciones.csv')

        else:
            continue

    i = 0

    for est in stations_simaj['codigo']:
        # adds coordinates to df
        simaj_reestructurado_all.loc[simaj_reestructurado_all.EST_SIMAJ == est,
                                     'LONG'] = stations_simaj[stations_simaj['codigo'] == est]['long'][i]

        simaj_reestructurado_all.loc[simaj_reestructurado_all.EST_SIMAJ == est,
                                     'LAT'] = stations_simaj[stations_simaj['codigo'] == est]['lat'][i]

        i = i + 1

    return (simaj_reestructurado_all)


class OutliersRemovalTools:
    '''
    Class that will have different methods to deal with outliers.
    '''

    def __init__(self):
        import pandas as pd

        self.concat_df = pd.DataFrame()
        self.params_dict_df = dict()
        self.preprocessed_df = pd.DataFrame()
        self.parameters = None
        self.stations = ['AGU', 'ATM', 'CEN', 'LDO', 'LPIN', 'MIR', 'OBL', 'SFE', 'TLA', 'VAL']
        self.nulls_df = pd.DataFrame()
        self.dir_gdl = '../data/processed'

    def fit_data(self):
        import pandas as pd
        import datetime as dt
        import os

        '''
        Reads the .csv files located on the dir_gdl var. 
        Creates a new var called concat_df in whit the data will be stored as a DF.
        The concat_df will have its date values in a datetime format and the stations rows are ordered alphabetically.
        Creates a parameters_dict_df dictionary containing all parameters data separated.
        
        There's no input and output as it will update the attributes of the class.
        '''

        #Select the path to read the .csv files
        #self.dir_gdl = r'C:\Users\victo\PycharmProjects\DataScienceProj\DS-Proj\Air_modelling\ItesoAQ\data\processed'
        os.chdir(self.dir_gdl)

        #Start the iter to read each file
        counter = 0
        for file in os.listdir(self.dir_gdl):

            if file.endswith('.csv'):
                if counter == 0:
                    self.concat_df = pd.read_csv(file)
                    counter += 1
                    continue

                #concat previous files read with new ones
                self.concat_df = pd.concat([self.concat_df, pd.read_csv(file)])

        #Creating a list with unique var of the parameters
        self.parameters = self.concat_df['PARAM'].unique()

        #Updating 'FECHA' column in a datetime format
        self.concat_df['FECHA'] = [dt.datetime.combine(
            dt.date.fromisoformat(self.concat_df['FECHA'].iloc[i]),
            dt.time.fromisoformat(self.concat_df['HORA'].iloc[i])
                                            )
            for i in range(len(self.concat_df['FECHA']))]

        #Arrange concat_df stations alphabetically
        self.concat_df = self.concat_df[['FECHA', 'HORA', 'PARAM', 'AGU', 'ATM', 'CEN', 'LDO', 'LPIN', 'MIR', 'OBL', 'SFE', 'TLA', 'VAL']]

        #Create dict to store each parameters df
        for parameter in self.parameters:
            self.params_dict_df[parameter] = self.concat_df.groupby('PARAM').get_group(parameter)


    def remove_std_outliers_v1(self, std_factor=3):
        '''
        Method that will remove all of the values that are lower or higher than
        the sum of the average + - std_factor * std dev.
        The average and std dev is considered to be different on each station and on each parameter.
        The outliers will be replaced with a NaN.

        :param std_factor: factor to which multiply the std dev
        :return: updates the preprocessed_df class attribute
        '''

        import numpy as np
        import pandas as pd

        for parameter in self.params_dict_df:

            parameter_df = self.params_dict_df[parameter]
            parameter_df.reset_index(inplace=True)

            # Create a main_df which saves the index of the original df
            main_df = parameter_df[['FECHA', 'HORA', 'PARAM']]

            # # Create a np array that only has the data of each station
            param_arr = parameter_df[self.stations].to_numpy()

            # Create fvout_arr (first value out array) which has all the rows but the first one
            fvout_arr = param_arr[1:, :]

            # Create a lvout_arr (last value out array) which has all the rpws but the last one
            lvout_arr = param_arr[:-1, :]

            # create a delta_arr array that stores the value of the diff between fvout and lvout
            delta_arr = fvout_arr - lvout_arr

            # obtain the mean and std of the delta_arr values
            mean_delta_arr = np.nanmean(delta_arr)
            std_delta_arr = np.nanstd(delta_arr)

            # hscalar represents the highest value our parameter can have before we remove it
            # lscalar works the same but with the lowest value
            hscalar = mean_delta_arr + std_factor * std_delta_arr
            lscalar = mean_delta_arr - std_factor * std_delta_arr

            # get the index of the elements whose values are gt hscalar or lt lscalar
            # IMPORTANT!!: add a + 1 on the row index as we are going to delete the values from the main ndarray and not from delta_arr
            outliers = np.where((delta_arr <= lscalar) | (delta_arr >= hscalar))
            coordinates = list(zip(outliers[0] + 1, outliers[1]))

            for i in range(len(coordinates)):
                param_arr[coordinates[i]] = np.nan

            param_df = pd.DataFrame(columns=self.stations, data=param_arr)
            outliers_removed_df = main_df.join(param_df)

            self.preprocessed_df = self.preprocessed_df.append(outliers_removed_df)

        self.preprocessed_df.set_index(['FECHA', 'HORA', 'PARAM'], inplace=True)
        self.preprocessed_df.sort_index(inplace=True)
        self.preprocessed_df.reset_index(inplace=True)

    def remove_std_outliers_v2(self, std_factor=3):
        '''
        This method is a modified version of the remove_std_outliers_v1 method that
        deletes outliers that fits the following statements:

            There must be at least 2 consecutive values higher or lower than the 3 std scalar.
            from those 2 consecutive values, one must have a negative value and the other a positive value in order to be removed.

        :param std_factor: std dev factor in which the data will be considered as outlier.
        :return: Updates the preprocessed_df attribute which then can be exported to a .csv file.
        '''

        import numpy as np
        import pandas as pd

        for parameter in self.params_dict_df:

            parameter_df = self.params_dict_df[parameter]
            parameter_df.reset_index(inplace=True)

            # Create a main_df which saves the index of the original df
            main_df = parameter_df[['FECHA', 'HORA', 'PARAM']]

            # # Create a np array that only has the data of each station
            P_arr = parameter_df[self.stations].to_numpy()

            # Create fvout_arr (first value out array) which has all the rows but the first one
            fvout_arr = P_arr[1:, :]

            # Create a lvout_arr (last value out array) which has all the rpws but the last one
            lvout_arr = P_arr[:-1, :]

            # create a delta_arr array that stores the value of the diff between fvout and lvout
            delta_arr = fvout_arr - lvout_arr

            # obtain the mean and std of the delta_arr values
            mean_delta_arr = np.nanmean(delta_arr, axis=0)
            std_delta_arr = np.nanstd(delta_arr, axis=0)

            # hscalar represents the highest value our parameter can have before we remove it
            # lscalar works the same but with the lowest value
            hscalar = mean_delta_arr + std_factor * std_delta_arr
            lscalar = mean_delta_arr - std_factor * std_delta_arr

            # get the index of the elements whose values are gt hscalar or lt lscalar
            outliers = np.where((delta_arr <= lscalar) | (delta_arr >= hscalar))
            coordinates = list(zip(outliers[0], outliers[1]))

            if coordinates[-1][0] == delta_arr.shape[0] - 1:
                coordinates.pop(-1)

            # Create a new list to store the real outliers
            coord_v2 = list()

            for i in range(len(coordinates)):

                # check the next value from the first outliers list comparing it with the lscalar
                if delta_arr[(coordinates[i][0] + 1, coordinates[i][1])] <= lscalar[coordinates[i][1]]:

                    # check for differences on the signs between values, if they differ, add the loc to the new list
                    if np.sign(delta_arr[(coordinates[i][0], coordinates[i][1])]) != np.sign(
                            delta_arr[(coordinates[i][0] + 1, coordinates[i][1])]):
                        coord_v2.append((coordinates[i][0] + 1, coordinates[i][1]))

                # check the next value from the first outliers list comparing it with the hscalar
                elif delta_arr[(coordinates[i][0] + 1, coordinates[i][1])] >= hscalar[coordinates[i][1]]:

                    # check for differences on the signs between values, if they differ, add the loc to the new list
                    if np.sign(delta_arr[(coordinates[i][0], coordinates[i][1])]) != np.sign(
                            delta_arr[(coordinates[i][0] + 1, coordinates[i][1])]):
                        coord_v2.append((coordinates[i][0] + 1, coordinates[i][1]))
                else:

                    continue

            for i in range(len(coord_v2)):
                P_arr[coord_v2[i]] = np.nan

            param_df = pd.DataFrame(columns=self.stations, data=P_arr)
            outliers_removed_df = main_df.join(param_df)

            self.preprocessed_df = self.preprocessed_df.append(outliers_removed_df)

        self.preprocessed_df.set_index(['FECHA', 'HORA', 'PARAM'], inplace=True)
        self.preprocessed_df.sort_index(inplace=True)
        self.preprocessed_df.reset_index(inplace=True)

    def export_to_csv(self):
        '''
        :return: a .csv file with the preprocessed data.
        '''
        import os
        os.chdir(self.dir_gdl + r'\\Outliers-rmvd\\')
        self.preprocessed_df.to_csv(f'{self.preprocessed_df["FECHA"].iloc[0].year}-{self.preprocessed_df["FECHA"].iloc[-1].year}_3std_preprocessed.csv')

