import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score
from scipy.stats import multivariate_normal as mvn
from scipy.stats import pearsonr
import pandas as pd
import datetime
import os
import joblib
import format_dataset as fd
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib import cm 
import lime
import lime.lime_tabular

saved_pickle_model = "bem_output/r2_0.88_2026-03-20_11.pkl"
 
def load_dataset(cat_exoplanet='data/exoplanet.eu_catalog_20-01-26_15_03_11.csv', 
                cat_solar="data/solar_system_planets_catalog.csv", 
                feature_names=['mass', 'semi_major_axis',
                                'eccentricity', 'star_metallicity',
                                'star_radius', 'star_teff',
                                'star_mass', 'star_metallicity', 'radius'],
                remove_outliers=True, 
                solar=True):
    """
    Select exoplanet in the catalogue which have mass and radius measurements
    as well as stellar parameters. This dataset will be used to train and 
    test the RF.

    Inputs: 
    cat_exoplanet: CSV file from exoplanet.eu
    cat_solar: CSV file from Planetary sheet
    feature_names: list of features to select in the dataset.


    Returns:
    dataset_exo = pandas dataframe with exoplanets with mass & radius 
                  measurements. mass/radii are in Earth mass/radii
                  
    #TODO: Vraag aan solene wat de beste manier is om outliers in de functie te verwijderen. 
    # Zij deed t direct uit de opgeslagen data, maar als mensen hun eigen data willen gebruiken gaat t lastig worden.
    """
    print("\nLoadint the exoplanet datast and solar system planets: ")
    # Importing exoplanet dataset
    dataset_exo = pd.read_csv(cat_exoplanet, index_col = 0)

    # Importing Solar system dataset
    dataset_solar_system = pd.read_csv(cat_solar, index_col = 0)
    print(dataset_exo.columns)
    # Choosing features/data
    if not feature_names:
        print("No features selected, loading all features")
    else: 
        dataset_exo = dataset_exo[feature_names]
        dataset_solar_system = dataset_solar_system[feature_names]
 
    if remove_outliers:
        print("REmoving outliers...")
        outliers = pd.read_csv('data\outliers.txt', sep='\t', index_col=0)
        for outlier in outliers.index:
            dataset_exo = dataset_exo.drop(labels=outlier)
        print('Removed all outliers.')
    else:
        print('No outliers removed.')
        
    # Remove planets with NaN's
    dataset_exo = dataset_exo.dropna(axis=0, how='any')
    
    # Converting from Jupiter to Earth mass/radius
    print("Convert planets mass/radius from Jupiter to Earth.")
    dataset_exo = fd.jupiter_to_earth_mass(dataset_exo, 'mass')
    dataset_exo = fd.jupiter_to_earth_radius(dataset_exo, 'radius')
    
    # Add the solar system planets with the exoplanets
    if solar:
        dataset = pd.concat([dataset_exo, dataset_solar_system])
    else:
        dataset = dataset_exo
        
    # Remove planets with NaN's
    dataset = dataset.dropna(axis=0, how='any')
    
    # Add observables
    print("Computing planets equi temperature")
    dataset = fd.add_temp_eq_dataset(dataset)
    print('Computing stellar luminosity')
    dataset = fd.add_star_luminosity_dataset(dataset)
    
    # Number of planets in dataset
    print('\nNumber of planets: ', len(dataset))
    
    # Returning the dataset with selected features
    select_features = ['mass',
                       'semi_major_axis',
                       'temp_eq',
                       'star_luminosity',
                       'star_radius', 'star_teff',
                       'star_mass',
                       'radius']

    print('Selecting features:')
    print(select_features)
    dataset = dataset[select_features]

    return dataset
    

def load_dataset_errors(cat_exoplanet='data/exoplanet.eu_catalog_20-01-26_15_03_11.csv', cat_solar="data/solar_system_planets_catalog.csv", remove_outliers = False, reference_dataset=None, solar=True):
    """
    Select exoplanet in the catalogue which have uncertainty measurements as 
    well as stellar parameters. If there is no uncertainty measurement, the 
    uncertainty is set to the 0.9 quantile of the distribution of uncertainties.
    
    If the uncertainty is higher then the value, the planet will be removed.
    
    This dataset will be used to compute error bars for the test set.
    
    Input:
    cat_exoplanet = CSV file from exoplanet.eu
    cat_solar = CSV file from planetary sheet.
    
    Returns:
    dataset_exo = pandas dataframe with exoplanets with mass & radius measurements
                  the mass/radius are in Earth mass/radius.
    """
    print("\nLoading exoplanet dataset and solar system planets:")
    
    dataset_exo = pd.read_csv(cat_exoplanet, index_col=0)
    
    #TODO zoek uit wrm die index_col, en of t ook weg kan.
    dataset_exo = dataset_exo[['mass', 'mass_error_min', 'mass_error_max',
                               'radius',
                               'radius_error_min', 'radius_error_max',
                               'semi_major_axis', 'semi_major_axis_error_min',
                               'semi_major_axis_error_max',
                               'eccentricity', 'eccentricity_error_min',
                               'eccentricity_error_max',
                               'star_mass',
                               'star_mass_error_min', 'star_mass_error_max',
                               'star_radius', 'star_radius_error_min',
                               'star_radius_error_max',
                               'star_teff',
                               'star_teff_error_min', 'star_teff_error_max']]

    dataset_solar_system = pd.read_csv(cat_solar, index_col=0)
    #TODO zoek again uit wrm de index_col = 0
    
    dataset_solar_system = dataset_solar_system[['mass', 'mass_error', 
                                                 'semi_major_axis', 
                                                 'semi_major_axis_error', 
                                                 'eccentricity', 
                                                 'eccentricity_error',
                                                 'star_mass',
                                                 'star_mass_error',
                                                 'star_radius',
                                                 'star_radius_error',
                                                 'star_teff',
                                                 'star_teff_error', 
                                                 'radius', 'radius_error']]
    
    #Remove NaN's in features only
    #TODO: find out why not here axis=0 en how='any'
    dataset_exo = dataset_exo.dropna(subset=['mass', 'semi_major_axis', 
                                            'star_radius', 'star_mass', 
                                                'star_teff', 'radius'])

    dataset_solar_system = dataset_solar_system.dropna(subset=['mass',
                                                               'semi_major_axis',
                                                               'star_radius',
                                                               'star_mass',
                                                               'star_teff',
                                                               'radius'])

    # Replace inf by NaN
    dataset_exo = dataset_exo.replace([np.inf, -np.inf], np.nan)
    
    # Replace NaN values in the error features by the 0.9 quantile value
    error_columns = ['mass_error_min', 'mass_error_max',
                     'radius_error_min', 'radius_error_max',
                     'semi_major_axis_error_min', 'semi_major_axis_error_max',
                     'eccentricity_error_min', 'eccentricity_error_max',
                     'star_mass_error_min', 'star_mass_error_max',
                     'star_radius_error_min', 'star_radius_error_max',
                     'star_teff_error_min', 'star_teff_error_max']

    for error_col in error_columns:
        # Find the 0.9 quantile value of the error column
        max_error = dataset_exo[error_col].quantile(0.9)
        print(error_col, max_error)
        
        #Replace NaN by the 0.9 error value
        dataset_exo[error_col] = dataset_exo[error_col].replace(np.nan, max_error)
        
        

    # After filling error NaNs, drop rows missing any core feature
    core_features = ['mass', 'semi_major_axis', 'eccentricity', 
                 'star_mass', 'star_radius', 'star_teff', 'radius']
    dataset_exo = dataset_exo.dropna(subset=core_features)

    #Convert from Jupiter to Earth
    print("Converting planets mass/radius to Earth masses/radii")
    dataset_exo = fd.jupiter_to_earth_mass(dataset_exo, 'mass')
    dataset_exo = fd.jupiter_to_earth_mass(dataset_exo, 'mass_error_max')
    dataset_exo = fd.jupiter_to_earth_mass(dataset_exo, 'mass_error_min')
    dataset_exo = fd.jupiter_to_earth_radius(dataset_exo, 'radius')
    dataset_exo = fd.jupiter_to_earth_radius(dataset_exo, 'radius_error_max')
    dataset_exo = fd.jupiter_to_earth_radius(dataset_exo, 'radius_error_min')

    # Computes the average error column
    dataset_exo['mass_error'] = dataset_exo[['mass_error_min', 'mass_error_max']].mean(axis=1).abs()
    dataset_exo['radius_error'] = dataset_exo[['radius_error_min', 'radius_error_max']].mean(axis=1).abs()
    dataset_exo['semi_major_axis_error'] = dataset_exo[['semi_major_axis_error_min', 'semi_major_axis_error_max']].mean(axis=1).abs()
    dataset_exo['eccentricity_error'] = dataset_exo[['eccentricity_error_min', 'eccentricity_error_max']].mean(axis=1).abs()
    dataset_exo['star_mass_error'] = dataset_exo[['star_mass_error_min', 'star_mass_error_max']].mean(axis=1).abs()
    dataset_exo['star_radius_error'] = dataset_exo[['star_radius_error_min', 'star_radius_error_max']].mean(axis=1).abs()
    dataset_exo['star_teff_error'] = dataset_exo[['star_teff_error_min', 'star_teff_error_max']].mean(axis=1).abs()
    
    dataset_exo = dataset_exo[['mass', 'mass_error',
                               'semi_major_axis',
                               'semi_major_axis_error',
                               'eccentricity',
                               'eccentricity_error',
                               'star_mass',
                               'star_mass_error',
                               'star_radius',
                               'star_radius_error',
                               'star_teff',
                               'star_teff_error',
                               'radius', 'radius_error']]
    
    if solar:
        dataset = pd.concat([dataset_exo, dataset_solar_system])
    else:
        dataset = dataset_exo
        

        
        
    # Add observables
    print("Computing planets equi temperature")
    dataset = fd.add_temp_eq_error_dataset(dataset)
    print("Computing stellar luminosity")
    dataset = fd.add_star_luminosity_error_dataset(dataset)
    
    
    # Select the same features as the original dataset
    dataset = dataset[['mass', 'mass_error',
                       'star_luminosity',
                       'star_luminosity_error',
                       'temp_eq', 'temp_eq_error',
                       'semi_major_axis',
                       'semi_major_axis_error',
                       'star_mass', 'star_mass_error',
                       'star_radius', 'star_radius_error',
                       'star_teff', 'star_teff_error',
                       'radius', 'radius_error']]

    print("The selected features can be changed in [load_dataset_errors]")
    print("\n", dataset.head())
    
    dataset.dropna(axis=1, how='any')
    print("\nNumber of planets: ", len(dataset))
    # We want the same exoplanets as the load_dataset fcn.
    if reference_dataset is not None:
        print("Matching planets with reference dataset")
        dataset = dataset.loc[dataset.index.intersection(reference_dataset.index)]
        dataset = dataset.reindex(reference_dataset.index)
    print('Final dataset length with errors is: ', len(dataset))
    return dataset    

def load_dataset_RV(cat_exoplanet="data/exoplanet.eu_catalog_20-01-26_15_03_11.csv", 
                    feature_names=['mass', 'mass_error_min', 'mass_error_max',
                                   'semi_major_axis',
                                   'eccentricity',
                                   'star_metallicity',
                                   'star_radius',
                                   'star_teff', 'star_mass']):
    """
    Select exoplanets in the catalog which are detected using RV and do not 
    have mass measurement. This dataset will be used to llater predict their
    masses.
    
    INPUTS:
    cat_exoplanet = CSV file from exoplanet.eu.
    features = list of features to select in the datsaset.
    
    OUTPUTS:
    dataset_radial = pd struct with exoplanets detected using RV without radius
                     measurements. Mass in in earth mass.
    """
    
    print("\nLoading exoplanet dataset found with RVs:")
    dataset = pd.read_csv(cat_exoplanet, index_col=0)
    
    # Select detected by RV
    dataset_radial = dataset[dataset.detection_type == "Radial Velocity"]
    
    # the radius column in Null = NaN
    dataset_radial = dataset_radial[pd.isnull(dataset_radial['radius'])]
    
    # Choosing features/data
    if not feature_names:
        print("No features selected, loading all features")
        pass
    else:
        print("Selected features:")
        print(feature_names)
        
        dataset_radial = dataset_radial[feature_names]
        
        # Excluding exoplanets with missing data
        dataset_radial = dataset_radial.dropna(subset=['mass', 'semi_major_axis',
                                                       'eccentricity',
                                                       'star_metallicity',
                                                       'star_radius', 'star_teff',
                                                       'star_mass'])
        
    # Replace inf by NaN
    dataset_radial = dataset_radial.replace([np.inf, -np.inf], np.nan)
    
    # Replace NaN values in the error features by the 0.9 quantile value
    error_columns = ['mass_error_min', 'mass_error_max']
    
    for error_col in error_columns:
        # Find the 0.9 quantile value of the error columns
        # max_error = dataset_radial[error_col].replace(np.nan, max_error)
        max_error = 0.0
        print(error_col, max_error)
        # replace NaN by the 0.9 error value
        dataset_radial[error_col] = dataset_radial[error_col].replace(np.nan,
                                                                      max_error)
        
    # Converting from Jupiter to Earth masses/radii
    print("Converting planet's mass/radius in Earth masses/radii")
    dataset_radial = fd.jupiter_to_earth_mass(dataset_radial, 'mass')
    dataset_radial = fd.jupiter_to_earth_mass(dataset_radial, 'mass_error_max')
    dataset_radial = fd.jupiter_to_earth_mass(dataset_radial, 'mass_error_min')

    # Computes the average error column
    dataset_radial['mass_error'] = dataset_radial[['mass_error_min',
                                                   'mass_error_max']].mean(axis=1).abs()

    # Adding observables
    print('Computing planet\'s equilibrium temperature')
    dataset_radial = fd.add_temp_eq_dataset(dataset_radial)
    print('Computing stellar luminosity')
    dataset_radial = fd.add_star_luminosity_dataset(dataset_radial)

    print('\nNumber of planets: ', len(dataset_radial))

    # Remove the mass error column for Random forest
    dataset_radial = dataset_radial[['mass',
                                     'semi_major_axis',
                                     'temp_eq',
                                     'star_luminosity',
                                     'star_radius', 'star_teff',
                                     'star_mass']]

    return dataset_radial

def random_forest_regression(dataset, model=saved_pickle_model, fit=False):
    """
    We gonna do some random forest regression.
    """
    np.int = int
    dataset_exo = dataset[:-8]
    dataset_solar = dataset[-8:]

    features_needed = ['mass', 'semi_major_axis', 'temp_eq', 'star_luminosity', 'star_radius', 'star_teff', 'star_mass']

    features = dataset_exo[features_needed]
    label = dataset_exo['radius']
    
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        label,
                                                        test_size = 0.25,
                                                        random_state = 1)
    
    features_solar = dataset_solar[features_needed]
    label_solar = dataset_solar['radius']
    
    X_train_solar, X_test_solar, y_train_solar, y_test_solar = train_test_split(
        features_solar, label_solar, test_size=0.25, random_state=1
    )
    
    
    X_train = pd.concat([X_train, X_train_solar])
    X_test  = pd.concat([X_test,  X_test_solar])
    y_train = pd.concat([y_train, y_train_solar])
    y_test  = pd.concat([y_test,  y_test_solar])
    
    train_test_values = [X_train.values, X_test.values, 
                         y_train.values, y_test.values]
    
    train_test_sets = [X_train, X_test, y_train, y_test]
    print('we zijn voor de fit')
    if fit:
        #Setting up the grid of hyperparameters
        param_grid = {'n_estimators': np.arange(80, 200),
                      'max_depth': np.arange(4, 10),
                      'max_features': np.arange(3, 6),
                      'min_samples_split': np.arange(4, 5)}


        rf = GridSearchCV(RandomForestRegressor(),
                          param_grid=param_grid,
                          cv=3,
                          verbose=1,
                          n_jobs=-1)
        np.int = int
        print('rf is geweest')
        
        # Fitting the training set - finding the best hyperparameters
        rf.fit(X_train, y_train)
        # rf.fit(X_train, y_train)
        print('fit is geweest')
        # Best hyperparameters found by the grid search
        print(rf.best_params_)
        
        # Random forest model with the best hyperparameters
        regr = RandomForestRegressor(n_estimators=rf.best_params_['n_estimators'],
                                     max_depth = rf.best_params_['max_depth'],
                                     max_features=rf.best_params_['max_features'],
                                     min_samples_split=rf.best_params_['min_samples_split'],
                                     random_state=42,
                                     oob_score=True
                                     )
        
        #Saving the random forest model in a file
        outdir = 'bem_output'
        if not os.path.exists(outdir):
            os.mkdir(outdir)


        name_Rf = 'r2_' + str(round(rf.best_score_, 2)) + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H")) + '.pkl'
        name_Rf = os.path.join(outdir, name_Rf)

        joblib.dump(regr, name_Rf)
        print('RF model save in : ', name_Rf)

    else:
        #Loading the random forest model saved
        print("Loading random forest model: ", model)
        regr = joblib.load(model)
        
    # Fit the best random forest model to the training set
    regr.fit(X_train, y_train)
    
    #Predict the radius for the training and testing sets
    y_train_predict = regr.predict(X_train)
    y_test_predict = regr.predict(X_test)
    
    test_score = r2_score(y_test, y_test_predict)
    r2_sklearn = regr.score(X_test, y_test)
    print('R-2 score sklearn: ', r2_sklearn)
    pearson = pearsonr(y_test, y_test_predict)
    print(f'Test set, R-2 score: {test_score:>5.3}')
    print(f'\nTest set, Pearson correlation: {pearson[0]:.3}')

    # Mean squared errors of the train and test set
    print('Root mean squared errors')
    print('Train set: ', np.sqrt(np.mean((y_train-y_train_predict)**2)),
          '\nTest set:  ', np.sqrt(np.mean((y_test-y_test_predict)**2)))
    
    # Feature importance
    print('\nFeature importance')
    _ = [print(name, ': \t', value) 
         for name, value in zip(features_needed, regr.feature_importances_)]
    
    return regr, y_test_predict, train_test_values, train_test_sets

def computing_errorbars(regr, dataset_errors, train_test_set):
    """
    INPUTS:
    regr = random forest regression model.
    dataset_errors = pandas df with each feature and their uncertainty.
    train_test_sets = pd df with exoplanets and their feature name incl the values.
    
    OUTPUTS:
    radii_test_output_error = error on the predicted radius for the test set.
    radii_test_input_error = original uncertainty on the radius measurements.
    """
    
    # Original train and test sets
    X_train, X_test, y_train, y_test = train_test_set
    
    # Cross matching the test set with the dataset with errors to compute error
    # bars for the exoplanet which have input errors.
    dataset_errors = dataset_errors.loc[X_test.index.values.tolist()]
    
    # Remove an exoplanet in case there is still a NaN in one of the features
    dataset_errors = dataset_errors.dropna(axis=0, how='any')
    
    # Matrix with all the erros on the different features
    features_errors = dataset_errors.iloc[:, :-2].values
    
    # Radius vector
    radii_test = dataset_errors.iloc[:, -2].values
    
    #Error on the radius vector
    radii_test_input_error = dataset_errors.iloc[:, -1].values
    
    # Empty vector to store the error bars
    radii_test_output_error = np.zeros_like(radii_test_input_error)
    for i in range(radii_test.size):
        rerr = regr.predict(mvn(features_errors[i, ::2], np.diag(features_errors[i, 1::2]), allow_singular=True).rvs(1000)).std()

        radii_test_output_error[i] = rerr
        
    # Save the errorbars in a txt file
    outdir = 'bem_output'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        
    filename = 'bem_output/test_radius_RF_errorbars.dat'
    print("Error bars of the test set are savid in: ", filename)
    np.savetxt(filename, radii_test_output_error)
    
    return radii_test_output_error, radii_test_input_error

def predict_radius(my_planet=np.array([[1, 1, 0, 1, 5777, 1]]),
                   my_name=np.array(['My planet b']),
                   regr=None,
                   jupiter_mass=False,
                   error_bar=False):
    """
    Predict the radius of a planet given the planetary mass, semi major axis,
    eccentricity, stellar radius, star effective temperature and stellar mass.
    
    Inputs: my_planet = array with shape (1, 6)
                        np.array([[planetary mass, 
                                   semi major axis,
                                   eccentricity,
                                   star_radius,
                                   star_teff,
                                   star_mass]])
            my_name = array with shape (1, )
                      np.array(['my planet b'])
            regr = random forest regression model
            jupiter_mass = bool, True if the planet's mass is given in Jupiter 
                           mass.
            error_bar = bool, True if an error is provided for each parameter
                        such as:
                        my_planet = np.array([[planetary mass, planetary mass error,
                                   semi major axis, semi major axis error,
                                   eccentricity, eccentricity error,
                                   star_radius, star_radius error,
                                   star_teff, star_teff error,
                                   star_mass, star_mass error]]) 
                                   
    OUTPUTS: radius = planet's radius predicting with the RF model
             my_pred_planet = pandas dataframe with the input features used by 
                              the random forest model.
                              Can be used as input in plot_LIME_predictions()
                              The features are now:
                              'mass', 'semi_major_axis',
                              'temp_eq', 'star_luminosity',
                              'star_radius', 'star_teff',
                              'star_mass'    
    """
    
    if regr is None:
        # Loading the random forest model saved
        print("Loading random forest model: ", regr)
        regr = joblib.load(regr)
    else:
        pass
    
    if error_bar:
        print("\nPredicting radius for planet:\n")
        my_planet = pd.DataFrame(data=my_planet,
                                 index=my_name,
                                 columns=np.array(['mass','mass_error',
                                                   'semi_major_axis', 'semi_major_axis_error',
                                                   'eccentricity', 'eccentricity_error', 
                                                   'star_radius', 'star_radius_error',
                                                   'star_teff', 'star_teff_error',
                                                   'star_mass', 'star_mass_error']))
        if jupiter_mass:   
            # Changing mass units to Earth mass
            my_planet = fd.jupiter_to_earth_mass(my_planet, 'mass')
            my_planet = fd.jupiter_to_earth_radius(my_planet, 'mass_error')
            
        else:
            print("Planetary mass is given in Earth")
        
        # Computing equi temperature
        my_planet = fd.add_temp_eq_dataset(my_planet)
        # Computing stellar luminosity
        my_planet = fd.add_star_luminosity_dataset(my_planet)
        
        # Planet with error bars
        print("Planet with error bars\n", my_planet.iloc[0])
        
        # Radius error prediction
        my_pred_planet = my_planet[['mass', 'mass_error', 'semi_major_axis',
                                    'semi_major_axis_error', 'temp_eq', 
                                    'temp_eq_error', 'star_luminosity', 'star_luminosity_error',
                                        'star_radius', 'star_radius_error',
                                        'star_teff', 'star_teff_error',
                                        'star_mass', 'star_mass_error']]
        
        # Feature / feature error
        features_with_errors = my_pred_planet.iloc[0].values.reshape(1, -1)
        radius_error = regr.predict(mvn(features_with_errors[0, ::2],
                                        np.diag(features_with_errors[0, 1::2]),
                                        allow_singular=True).rvs(1000)).std()
        
        # Radius prediction
        my_pred_planet = my_planet[['mass', 'semi_major_axis', 'temp_eq', 
                                    'star_luminosity', 'star_radius', 'star_teff',
                                    'star_mass']]
        
        radius = regr.predict(my_pred_planet.iloc[0].values.reshape(1, -1))
        
        # Print pred radius
        print("Predicted radius (R_earth): ", radius, '±', radius_error)   
        return [radius, radius_error], my_pred_planet
    
    else:
        print('\nPredicting radius for planet:\n')
        my_planet = pd.DataFrame(data=my_planet,
                                 index=my_name,
                                 columns=np.array(['mass', 'semi_major_axis',
                                                   'eccentricity',
                                                   'star_radius',
                                                   'star_teff', 'star_mass']))
        # Changing mass units to Earth mass
        if jupiter_mass:
            my_planet = fd.jupiter_to_earth_mass(my_planet, 'mass')
        else:
            print('Planetary mass is given in Earth mass')

        # Computing equilibrium temperature
        my_planet = fd.add_temp_eq_dataset(my_planet)
        # Computing stellar luminosity
        my_planet = fd.add_star_luminosity_dataset(my_planet)
        # Select features
        my_pred_planet = my_planet[['mass', 'semi_major_axis',
                                    'temp_eq', 'star_luminosity',
                                    'star_radius', 'star_teff',
                                    'star_mass']]
        # Radius prediction
        print(my_pred_planet.iloc[0])
        radius = regr.predict(my_pred_planet.iloc[0].values.reshape(1, -1))
        print('Predicted radius (Rearth): ', radius)

        return radius, my_pred_planet
    
def plot_dataset(dataset, predicted_radii=[], rv=False):
    """
    Function plots the dataset.
    """
    if not rv:
        #Plot the original dataset
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        size = dataset.temp_eq
        plt.scatter(dataset.mass, dataset.radius, c=size, cmap=cm.magma_r,
                    s=4, label='Verification sample')
        plt.colorbar(label=r"Equilibrium temperature (K)")
        plt.xlabel(r"Mass ($M_\oplus$)")
        plt.ylabel(r"Radius ($R_\oplus$)")
        plt.legend(loc='lower right', markerscale=0, handletextpad=0, handlelength=0)
        plt.show()
        
    if rv:
        # Plot the radial velocity dataset
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        size = dataset.temp_eq
        plt.scatter(dataset.mass, predicted_radii, c=size,
                    cmap=cm.magma_r, s=4, label='RV sample')
        plt.colorbar(label=r'Equilibrium temperature (K)')
        plt.xlabel(r'Mass ($M_\oplus$)')
        plt.ylabel(r'Radius ($R_\oplus$)')
        plt.legend(loc='lower right', markerscale=0,
                   handletextpad=0.0, handlelength=0)
        
    
    
    
    return None

def plot_true_predicted(train_test_sets, radii_test_RF,
                        radii_test_output_error):
    """
    Plot the residuals on the test set between true radius and RF.
    """
    X_train, X_test, y_train, y_test = train_test_sets
    
    plt.figure()
    plt.errorbar(radii_test_RF, y_test.values, xerr=radii_test_output_error,
                 fmt='.', c='C1', elinewidth=0.5, label="Random forest")
    
    # 1:1 line and labels
    plt.plot(np.sort(y_test.values), np.sort(y_test.values), 'k-', lw=0.25)
    plt.ylabel(r'True radius ($R_\oplus$)')
    plt.ylabel(r'Predicted radius ($R_\oplus$)')
    plt.legend(loc='lower right')
    plt.show()
    return None
    
    
    

def plot_learning_curve(regr, dataset, save=False, fit=False):
    """
    Function plots the learning curve of the random forest regression model.
    Cross validation with 100 iterations to get smoother mean test and train
    test score curves, each time with 20% data randomly selected as a 
    validation set.
    
    INPUTS:
    regr = random forest regression model.
    dataset = pandas dataframe with features and labels.
    save = bool, writes (True) or not (False) the scores.
    fit = bool, computes the score if True
    
    OUTPUTS:
    written files
    """
    
    features = dataset.iloc[:, :-1].values
    label = dataset.iloc[:, -1].values #radius
    
    outdir = 'bem_output'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        
    if fit:
        cv = ShuffleSplit(n_splits=100, test_size=0.1, random_state=11)
        train_sizes, train_scores, test_scores= learning_curve(regr, 
                                                                X = features,
                                                                y = label,
                                                                cv=cv,
                                                                train_sizes=np.linspace(0.1, 1, 10),
                                                                n_jobs=-1,
                                                                verbose=1)
        
    else:
        train_sizes = np.loadtxt("bem_output/lc_train_sizes.dat")
        train_scores = np.loadtxt("bem_output/lc_train_scores.dat")
        test_scores = np.loadtxt("bem_output/lc_test_scores.dat")
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure()
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color = 'hotpink')
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', 
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', 
             label="Cross-validation score")
    plt.legend(loc='lower right')
    plt.show()
    
    if save:
        np.savetxt(os.path.join(outdir, "lc_train_sizes.dat"), train_sizes)
        np.savetxt(os.path.join(outdir, 'lc_train_scores.dat'), train_scores)
        np.savetxt(os.path.join(outdir, 'lc_test_scores.dat'), test_scores)
    return plt 
    
    

def plot_validation_curves(regr, dataset, name='features', save=False,
                           fit=False):
    """
    INPUTS: 
    regr = random forest regression model
    dataset = pandas dataframe with features and labels
    name = str, can be 'features', 'tree', 'depth'
    save = bool, writes (True) or not (False) the scores
    fit = bool, computes the score if True 
    
    OUTPUTS:
    Written files
    """
    features = dataset.iloc[:, :-1]
    label = dataset.iloc[:, -1]
    
    outdir = 'bem_output'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        
    if name == 'features':
        param_range = np.arange(features.shape[1]) + 1
        param_name = 'max_features'
    elif name == 'tree':
        param_range = np.array([10, 20, 35, 50, 100, 1000, 5000, 10000])
        param_name = 'n_estimators'
    elif name == 'depth':
        param_range = np.array([1, 2, 3, 4, 5, 6, 7,
                                8, 9, 10, 50, 100, 1000])
        param_name = 'max_depth'
    else:
        print("Error the parameter of the validation curve is incorrect")
        print("Names can be features, tree, depth")
        return None
    
    if fit:
        train_scores, test_scores = validation_curve(regr, features, label,
                                                     param_name=param_name,
                                                     param_range=param_range,
                                                     cv=3, scoring='r2', 
                                                     n_jobs=-1, verbose=1)
    else:
        if name == 'features':
            train_scores = np.loadtxt("bem_output/vc_features_train_scores.dat")
            test_scores = np.loadtxt("bem_output/vc_features_test_scores.dat")
        elif name == 'tree':
            train_scores = np.loadtxt("bem_output/vc_tree_train_scores.dat")
            test_scores = np.loadtxt("bem_output/vc_tree_test_scores.dat")
        elif name == 'depth':
            train_scores = np.loadtxt("bem_output/vc_depth_train_scores.dat")
            test_scores = np.loadtxt("bem_output/vc_depth_test_scores.dat")
        else:
            pass

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Validation Curve with Random Forest regressor")
    plt.xlabel(param_name)
    plt.ylabel("Score")

    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
    
    if save:
        if name == 'features':
            np.savetxt(os.path.join(outdir, "vc_features_train_scores.dat"),
                       train_scores)
            np.savetxt(os.path.join(outdir, "vc_features_test_scores.dat"),
                       test_scores)
        elif name == 'tree':
            np.savetxt(os.path.join(outdir, "vc_tree_train_scores.dat"),
                       train_scores)
            np.savetxt(os.path.join(outdir, "vc_tree_test_scores.dat"),
                       test_scores)
        elif name == 'depth':
            np.savetxt(os.path.join(outdir, "vc_depth_train_scores.dat"),
                       train_scores)
            np.savetxt(os.path.join(outdir, "vc_depth_test_scores.dat"),
                       test_scores)
        else:
            pass
    return None
        
    

def plot_LIME_predictions(regr, dataset, train_test_sets, 
                          planets=None,    
                          my_pred_planet=None,    
                          my_true_radius=None,    
                          feature_names=None):
    """
    Compute and plot the LIME explanations for the predictions of the Random Forest regression model. 

    Possibilities:
    Explain 1 planet from the test set
    Explain 1 custom planet (my_pred_planet) with true radius (my_true_radius)
    make grid plots for 6 planets from the test set (planets)
    
    """


    if planets is None:
        planets = []

    if my_pred_planet is None:
        my_pred_planet = pd.DataFrame()

    if feature_names is None:
        feature_names = [
            'mass',
            'semi_major_axis',
            'temp_eq',
            'star_luminosity',
            'star_radius',
            'star_teff',
            'star_mass',
        ]

    # data
    X_train, X_test, y_train, y_test = train_test_sets

    # makes sure pandas object with correct column names are used for LIME
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)

    # check if features are non continuous
    nb_unique_obj_in_features = np.array([
        X_train.iloc[:, i].nunique(dropna=False) for i in range(X_train.shape[1])
    ])
    # will be empty for continuous features 
    cat_features = np.argwhere(nb_unique_obj_in_features <= 10).flatten().tolist()

    # LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        categorical_features=cat_features,
        mode='regression',
        verbose=True
    )

    def _to_scalar(x):
        """
        Convert a 1D array with one element to a scalar.
        """
        arr = np.asarray(x).reshape(-1)
        return float(arr[0])

    def _pretty_names(exp_list):
        """
        Convert feature names to a more readable format.
        """
        names = []
        for name, _ in exp_list:
            name = (
                name.replace("<=", r"$\leq$")
                    .replace("<", r"$<$")
                    .replace(">", r"$>$")
                    .replace("_", " ")
                    .replace(".00", "")
            )
            names.append(name)
        return names

    def _plot_single_explanation(exp, title, true_radius=None):
        lime_radius = _to_scalar(exp.local_pred)
        rf_radius = _to_scalar(exp.predicted_value)

        exp_list = exp.as_list()
        vals = [x[1] for x in exp_list]
        names = _pretty_names(exp_list)

        vals.reverse()
        names.reverse()

        colors = ['C2' if v > 0 else 'C3' for v in vals]
        pos = np.arange(len(vals)) + 0.5

        plt.figure(figsize=(8, 5))
        plt.barh(pos, vals, align='center', color=colors, alpha=0.5)
        plt.yticks(pos, names)
        plt.xlabel("Weight")
        plt.title(title, loc='right')

        if true_radius is not None:
            textstr = "\n".join((
                rf"True radius={float(true_radius):.2f}$R_\oplus$",
                rf"RF radius={rf_radius:.2f}$R_\oplus$",
                rf"LIME radius={lime_radius:.2f}$R_\oplus$"
            ))
        else:
            textstr = "\n".join((
                rf"RF radius={rf_radius:.2f}$R_\oplus$",
                rf"LIME radius={lime_radius:.2f}$R_\oplus$"
            ))

        plt.text(
            0.02, 0.02, textstr,
            transform=plt.gca().transAxes,
            bbox={'boxstyle': 'round', 'facecolor': 'white'}
        )
        plt.tight_layout()
        plt.show()

        return exp

    # 1 eigen planeet verklaren
    if not my_pred_planet.empty:
        missing = [c for c in feature_names if c not in my_pred_planet.columns]
        if missing:
            raise ValueError(
                f"my_pred_planet mist deze kolommen: {missing}. "
                f"Verwacht exact: {feature_names}"
            )

        row = my_pred_planet[feature_names].iloc[0].values.astype(float)

        exp = explainer.explain_instance(
            data_row=row,
            predict_fn=regr.predict,
            num_features=len(feature_names)
        )

        title = my_pred_planet.index[0] if len(my_pred_planet.index) > 0 else "Custom planet"
        _plot_single_explanation(exp, title=title, true_radius=my_true_radius)

        return explainer, [exp]

    # 2 default testplaneten kiezen
    if len(planets) == 0:
        default_names = [
            'TRAPPIST-1 g',
            'HATS-35 b',
            'CoRoT-13 b',
            'Kepler-75 b',
            'WASP-17 b',
            'Kepler-20 c'
        ]
        found = []
        for name in default_names:
            matches = np.where(X_test.index == name)[0]
            if len(matches) > 0:
                found.append(int(matches[0]))
        planets = found[:6]

        if len(planets) == 0:
            planets = list(range(min(6, len(X_test))))

    explanations = []

    # 3 grid plots voor testplaneten
    n = len(planets)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), constrained_layout=True)
    axs = np.array(axs).reshape(-1)

    for ax in axs[n:]:
        ax.axis("off")

    for j, planet in enumerate(planets):
        exp = explainer.explain_instance(
            data_row=X_test.iloc[planet].values.astype(float),
            predict_fn=regr.predict,
            num_features=len(feature_names)
        )
        explanations.append(exp)

        lime_radius = _to_scalar(exp.local_pred)
        rf_radius = _to_scalar(exp.predicted_value)

        exp_list = exp.as_list()
        vals = [x[1] for x in exp_list]
        names = _pretty_names(exp_list)

        vals.reverse()
        names.reverse()

        colors = ['C2' if v > 0 else 'C3' for v in vals]
        pos = np.arange(len(vals)) + 0.5

        ax = axs[j]
        ax.barh(pos, vals, align='center', color=colors, alpha=0.5)
        ax.set_yticks(pos)
        ax.set_yticklabels(names)
        ax.set_xlabel("Weight")
        ax.set_title(str(X_test.iloc[planet].name), loc='right')

        true_radius = float(y_test.iloc[planet])
        textstr = "\n".join((
            rf"True radius={true_radius:.2f}$R_\oplus$",
            rf"RF radius={rf_radius:.2f}$R_\oplus$",
            rf"LIME radius={lime_radius:.2f}$R_\oplus$"
        ))

        ax.text(
            0.60, 0.05, textstr,
            transform=ax.transAxes,
            bbox={'boxstyle': 'round', 'facecolor': 'white'}
        )

    plt.show()

    # 4 mass-radius plot met gemarkeerde planeten
    plt.figure()
    plt.xscale('log')
    plt.yscale('log')

    size = X_test['temp_eq'].values if 'temp_eq' in X_test.columns else None
    sc = plt.scatter(
        X_test['mass'].values,
        y_test.values,
        c=size,
        cmap=cm.magma_r
    )

    if size is not None:
        plt.colorbar(sc, label=r'Equilibrium temperature (K)')

    plt.xlabel(r'Mass ($M_\oplus$)')
    plt.ylabel(r'Radius ($R_\oplus$)')

    for planet in planets:
        plt.plot(
            X_test.iloc[planet]['mass'],
            y_test.iloc[planet],
            'o',
            mfc='none',
            ms=12,
            label=str(X_test.iloc[planet].name)
        )

    if len(planets) > 0:
        plt.legend()

    plt.show()

    return explainer, explanations


        