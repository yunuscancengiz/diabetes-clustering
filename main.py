import os
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE, KMeansSMOTE, BorderlineSMOTE, RandomOverSampler, SVMSMOTE
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.style.use("ggplot")
from warnings import filterwarnings
filterwarnings('ignore')


class DiabetesClassification:
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(),
            'param_grid': {'penalty':['l1','l2','elasticnet','none'],
                'C' : np.logspace(-4,4,20),
                'solver': ['lbfgs','newton-cg','liblinear','sag','saga'],
                'max_iter'  : [100,1000,2500,5000]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(n_neighbors=5),
            'param_grid': {
                'n_neighbors': list(range(1, 12))
            }
        },
        'SVC': {
            'model': SVC(kernel='linear'),
            'param_grid': {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}
        }
    }


    def __init__(self):
        self.plots_path = os.path.join(os.getcwd(), 'plots')
        os.makedirs(self.plots_path, exist_ok=True)
        os.makedirs(os.path.join(self.plots_path, 'cluster'), exist_ok=True)
        os.makedirs(os.path.join(self.plots_path, 'confusion_matrix'), exist_ok=True)
        
        self.feature1 = 'Glucose'
        self.feature2 = 'Age'


    def main(self):
        df = self.read_dataset()
        df = self.missing_value_analysis(df=df)
        self.correlation_analysis(df=df)
        self.unique_value_analysis(df=df)
        scaled_df, scaler = self.standardization(df=df)
        cleaned_df = self.outlier_detection(scaled_df=scaled_df)
        self.check_target_imbalance(cleaned_df=cleaned_df)
        X_train, X_test, y_train, y_test = self.train_test_splitting(df=cleaned_df)
        balanced_dict = self.oversampling_with_smote(X_train=X_train, y_train=y_train)
        self.pairplot_with_outcome(df=cleaned_df)

        for smote_name, data in balanced_dict.items():
            X_balanced = data['X']
            y_balanced = data['y']

            comparison_df = self.k_means_clustering(smote_name=smote_name, X_balanced=X_balanced, y_balanced=y_balanced, feature1=self.feature1, feature2=self.feature2)
            self.k_means_classification_report(smote_name=smote_name, comparison_df=comparison_df)


        for model_name, data in self.models.items():
            model = data['model']
            param_grid = data['param_grid']

            self.train_and_evaluate_models(model_name=model_name, model=model, param_grid=param_grid, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            self.compare_accuracies(model_name=model_name, model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)



    def read_dataset(self, filename:str='diabetes_dataset.csv', unnecessary_columns:list=['External', 'Unnamed: 0', 'Insulin']) -> pd.DataFrame:
        '''
        Summary: 
            Reads dataset from csv file and prints basic information and statistics about dataset. Drops unnecessary columns.

        Params:
            filename (str): Filename to read the dataset. Default: diabetes_dataset.csv.
            unnecessary_columns: Columns to drop.
        
        Return:
            Returns dataset in pandas DataFrame format.
        '''

        df = pd.read_csv(filename, sep=';')

        # drop unnecessary columns if exist
        if len(unnecessary_columns) > 0:
            for col in unnecessary_columns:
                df.drop(col, axis=1, inplace=True)
                print(f'{col} named column dropped.')

        print(f'\nFirst part of the dataset: \n{df.head()}')
        print(f'\nBasic Informations: \n{df.info()}')
        print(f'\nBasic Statistics: \n{df.describe().T}')
        return df
    

    def missing_value_analysis(self, df:pd.DataFrame, columns:list=['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']) -> pd.DataFrame:
        '''
        Summary:
            Analyzes missing values and drops them. If the columns has meaningless 0 values replaces them with np.nan and handles them too. Plots the ratio of number of meaningless zeros to all data as piecharts and saves them into plots directory.

        Params:
            df (pandas DataFrame): dataset.
            columns (list): Columns that have meaningless zeros.

        Return:
            df (pandas Dataframe): Analyzed dataset in pandas DataFrame format.
        '''

        self.piechart(df=df, columns=columns)   # piechart of meaningless zeros vs all data
        df[columns] = df[columns].replace(0, np.nan)    # replace 0 with np.nan
        print(f'\nNumber of Missing Values Before Analyze: \n{df.isnull().sum()}\n')
        [print(f'(before) skew of {col}: {df[col].skew():.2f}') for col in columns]

        # drop missing values
        df.dropna(inplace=True, ignore_index=True)
        print(f'\nDataframe shape after drop missing values: {df.shape}')

        print(f'\nNumber of Missing Values After Analyze: \n{df.isnull().sum()}\n')
        [print(f'(after) skew of {col}: {df[col].skew():.2f}') for col in columns]
        return df


    def piechart(self, df:pd.DataFrame, columns:list):
        '''
        Summary:
            Plots piecharts and save them into plots directory.

        Params:
            Columns: Columns to plot.
        '''

        for col in columns:
            number_of_zeros = (df[col] == 0).sum()
            number_of_non_zeros = len(df) - number_of_zeros
            
            sizes = [number_of_zeros, number_of_non_zeros] 
            labels = ['Zeros', 'Non-Zeros']
            
            plt.figure(figsize=(6, 6))
            plt.pie(sizes, labels=labels, autopct='%1.1f', startangle=90, colors=['#A1D490', '#00509E'], explode=(0.1, 0))
            plt.title(f'Proportion of Zero vs Non-Zero Values in {col}')
            plt.savefig(f'{self.plots_path}/piechart_{col}.png')


    def correlation_analysis(self, df:pd.DataFrame):
        '''
        Summary:
            Plots correlation matrix as heatmap and saves the figure in plots directory.
        
        Params:
            df (pandas DataFrame): dataset.
        '''

        mask = np.zeros_like(df.corr())
        triangle_indices = np.triu_indices_from(mask)
        mask[triangle_indices] = True

        plt.figure(figsize=(16, 10))
        sns.heatmap(df.corr(), mask = mask, annot = True, annot_kws = {"size":14})
        plt.xticks(fontsize = 14, rotation=90)
        plt.yticks(fontsize = 14)
        plt.savefig(f'{self.plots_path}/correlation_heatmap.png')


    def unique_value_analysis(self, df:pd.DataFrame):
        '''
        Summary:
            Prints number of unique values for each column.
        
        Params:
            df (pandas DataFrame): dataset.
        '''
        print('\nUnique Values:')
        [print(f'{col}: {df[col].value_counts().shape[0]}') for col in df.columns]


    def standardization(self, df:pd.DataFrame):
        '''
        Summary:
            Normalizes data using StandardScaler.
        
        Params:
            df (pandas DataFrame): dataset

        Return:
            Scaled dataset and scaler.
        '''

        columns_to_scale = df.drop(columns=['Outcome']).columns     # keep Outcome away from scaling
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df[columns_to_scale])
        scaled_df = pd.DataFrame(scaled_array, columns=columns_to_scale)
        scaled_df = pd.concat([scaled_df, df['Outcome']], axis=1)
        print(f'\nFirst Part of Scaled Dataset: \n{scaled_df.head()}')
        return scaled_df, scaler
    

    def outlier_detection(self, scaled_df:pd.DataFrame) -> pd.DataFrame:
        '''
        Summary:
            - Plots boxplot to show outliers for each column and uses Outcome as hue parameter to display the distributions of outliers for classes. Saves the plot in plots directory.
            - Drops outliers from dataset.

        Params:
            scaled_df (pandas DataFrame): scaled dataset.
        
        Return:
            cleaned_df (pandas DataFrame): Returns cleaned dataframe    
        '''

        self.boxplot(scaled_df=scaled_df)

        print(f'\nDataset shape before outlier cleaning: {scaled_df.shape}')
        for col in scaled_df.columns:
            Q1 = np.percentile(scaled_df[col], 25)
            Q3 = np.percentile(scaled_df[col], 75)
            IQR = Q3 - Q1

            upper = np.where(scaled_df.loc[:, col] >= (Q3 + (IQR * 2.5)))
            lower = np.where(scaled_df.loc[:, col] <= (Q1 - (IQR * 2.5)))
            
            try:
                scaled_df.drop(upper[0], inplace=True)
            except:
                pass
            try:
                scaled_df.drop(lower[0], inplace=True)
            except:
                pass
        print(f'\nDataset shape after outlier cleaning: {scaled_df.shape}')
        cleaned_df = scaled_df.copy()
        return cleaned_df


    def boxplot(self, scaled_df:pd.DataFrame):
        '''
         Summary:
            - Plots boxplot to show outliers for each column and uses Outcome as hue parameter to display the distributions of outliers for classes. Saves the plot in plots directory.
        '''

        df_melted = pd.melt(scaled_df, id_vars = "Outcome", var_name = "features", value_name = "value")
        plt.figure(figsize = (15, 15))
        sns.boxplot(x = "features", y = "value", hue = "Outcome", data = df_melted)
        plt.savefig(f'{self.plots_path}/boxplot.png')


    def check_target_imbalance(self, cleaned_df:pd.DataFrame):
        '''
        Summary: 
            Checks target imbalance for Outcome column and prints the results. Also, plots the class distributions as countplot in plots directory.
        
        Params:
            cleaned_df (pandas DataFrame): scaled and cleaned dataset.
        '''
        
        print(f'\nClass distributions of Outcome column: \n{cleaned_df["Outcome"].value_counts()}')
        plt.figure()
        sns.countplot(data=cleaned_df, x='Outcome', palette='Set2')
        plt.title(f'Class Distribution of Outcome - 0: {cleaned_df["Outcome"].value_counts()[0]} | 1: {cleaned_df["Outcome"].value_counts()[1]}')
        plt.xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)')
        plt.ylabel('Count')
        plt.savefig(f'{self.plots_path}/countplot_class_distributions.png')


    def train_test_splitting(self, df:pd.DataFrame):
        '''
        Summary:
            Splits the dataset as train and test sets. 

        Params:
            df (pandas DataFrame): dataset to split.
        
        Return:
            X_train, X_test, y_train, y_test: Train and test sets for features and target columns.
        '''

        X = df.drop(columns=['Outcome'])
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test


    def oversampling_with_smote(self, X_train, y_train):
        '''
        Summary:
            Handles target imbalance with different SMOTE algorithms such as SMOTE, RandomOverSampler, BorderlineSMOTE, SVMSMOTE, K-MeansSMOTE

        Params:
            X_train: train set of features
            y_train: train set of target

        Return:
            balanced_dict (dict): Returns balanced sets as dictionary format.
        '''

        balanced_dict = {}

        #Random Oversampling
        random_os = RandomOverSampler(random_state = 42)
        X_random, y_random = random_os.fit_resample(X_train, y_train)
        balanced_dict['RandomOverSampler'] = {'X': X_random, 'y': y_random}

        #SMOTE
        smote_os = SMOTE(random_state = 42)
        X_smote, y_smote = smote_os.fit_resample(X_train, y_train)
        balanced_dict['SMOTE'] = {'X': X_smote, 'y': y_smote}

        #BorderlineSMOTE
        smote_border = BorderlineSMOTE(random_state = 42, kind = 'borderline-2')
        X_smoteborder, y_smoteborder = smote_border.fit_resample(X_train, y_train)
        balanced_dict['BorderlineSMOTE'] = {'X': X_smoteborder, 'y': y_smoteborder}

        #SVM SMOTE
        smote_svm = SVMSMOTE(random_state = 42)
        X_smotesvm, y_smotesvm = smote_svm.fit_resample(X_train, y_train)
        balanced_dict['SVMSMOTE'] = {'X': X_smotesvm, 'y': y_smotesvm}

        #K-Means SMOTE
        smote_kmeans = KMeansSMOTE(random_state = 42)
        X_smotekmeans, y_smotekmeans = smote_kmeans.fit_resample(X_train, y_train)
        balanced_dict['KMeansSMOTE'] = {'X': X_smotekmeans, 'y': y_smotekmeans}

        return balanced_dict
    

    def pairplot_with_outcome(self, df:pd.DataFrame):
        '''
        Summary:
            Plots a pairplot using Outcome column as hue parameter to examine if there is obvious clusters between two columns. That plot might be useful to choose which columns are going to be used for K-Means Clustering. Saves the figure in plots directory

        Params:
            df (pandas DataFrame): dataset.
        '''

        plt.figure()
        sns.pairplot(df, hue='Outcome', diag_kind='kde', plot_kws={'alpha':0.6})
        plt.savefig(f'{self.plots_path}/pairplot_hue_outcome.png')


    def k_means_clustering(self, smote_name, X_balanced, y_balanced, feature1, feature2):
        '''
        Summary: 
            Fits the feature train set with KMeans. Plots scatter plot to show clusters.

        Params:
            smote_name (str): name of SMOTE algorithm
            X_balanced, y_balanced: balanced training sets of fetaures and target column

        Return:
            comparison_df (pandas DataFrame): returns the df that has Cluster and Outcome columns.
        '''

        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X_balanced[[feature1, feature2]])

        centroids = kmeans.cluster_centers_
        y_kmeans = kmeans.predict(X_balanced[[feature1, feature2]])
        comparison_df = pd.DataFrame({
            'Cluster': y_kmeans,
            'Outcome': y_balanced
        })

        self.scatterplot_for_clustering(smote_name=smote_name, X_balanced=X_balanced, feature1=feature1, feature2=feature2, centroids=centroids, y_kmeans=y_kmeans)
        return comparison_df
    

    def scatterplot_for_clustering(self, smote_name:str, X_balanced, feature1:str, feature2:str, centroids, y_kmeans):
        '''
        Summary:
            Plots the clusters as scatter plot and saves the plot in plots/cluster/ directory.

        Params:
            smote_name (str): Name of SMOTE algorithm
            X_balanced: balanced feature set
            feature1 (str): first feature name to cluster
            feature2 (str): second feature name to cluster
            centroids: centroids of clusters
            y_kmeans: clustered target  
        '''

        cluster_plots_path = os.path.join(self.plots_path, 'cluster')
        plt.figure(figsize=(8, 6))
        plt.scatter(X_balanced[feature1], X_balanced[feature2], c=y_kmeans, cmap='viridis', s=50, alpha=0.6, label='Clusters')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='X', label='Centroids')
        plt.title(f'K-Means Clustering with {smote_name} Data')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.legend()
        plt.savefig(f'{cluster_plots_path}/{smote_name}_cluster.png')


    def k_means_classification_report(self, smote_name:str, comparison_df:pd.DataFrame):
        '''
        Summary:
            Compares predicteds with actuals then creates classification report and plots confusion matrix
        
        Params:
            smote_name: name of SMOTE algorithm
            comparison_df (pandas DataFrame): dataframe with Cluster and Outcome columns
        '''
        
        cluster_comparison = comparison_df.groupby('Cluster')['Outcome'].value_counts().unstack()
        cluster_labels = cluster_comparison.idxmax(axis=1)  # Majority class

        # re-label predictions
        comparison_df['Predicted'] = comparison_df['Cluster'].map(cluster_labels)
        y_true = comparison_df['Outcome']
        y_pred = comparison_df['Predicted']
        print(f'\n({smote_name}) Classification Report: \n{classification_report(y_true, y_pred)}')

        cm = confusion_matrix(y_true, y_pred)
        print(f'\n({smote_name}) Confusion Matrix: \n{cm}')

        self.plot_confusion_matrix(name=smote_name, cm=cm)
        


    def plot_confusion_matrix(self, name:str, cm):
        '''
        Summary:
            Plots confusion matrix and plots them in plots/confusion_matrix/ directory
            
        Params:
            name (str): name of SMOTE algorithm or model
            cm: confusion matrix
        '''

        confusion_matrix_plots_path = os.path.join(self.plots_path, 'confusion_matrix')
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Outcome 0', 'Outcome 1'], yticklabels=['Outcome 0', 'Outcome 1'])
        plt.title(f'({name}) Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'{confusion_matrix_plots_path}/{name}_confusion_matrix.png')


    def train_and_evaluate_models(self, model_name, model, param_grid:dict, X_train, y_train, X_test, y_test):
        '''
        Summary: 
            Trains models for LogisticRegression, KNN, SVM then prints classification report and confusion matrix. Plots confusion matrix for each model and saves them in plots/confusion_matrix/ directory

        Params:
            model_name (str): name of model
            model: classifier
            X_train, y_train, X_test, y_test: train and test sets for features and target columns
        '''
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print(f"\nModel: {model_name}")
        print(f'Classification Report: \n{classification_report(y_test, y_pred)}')

        cm = confusion_matrix(y_test, y_pred)
        print(f'\nConfusion Matrix: \n{cm}')

        self.plot_confusion_matrix(name=model_name, cm=cm)
        self.hyperparameter_optimization(model=model, model_name=model_name, param_grid=param_grid, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


    def compare_accuracies(self, model_name, model,  X_train, y_train, X_test, y_test):
        '''
        Summary:
            Compares models accuracy scores and plots the results then saves the plot in plots directory.

        Params:
            model_name (str): name of model
            model: classifier
            X_train, y_train, X_test, y_test: train and test sets for features and target columns
        '''

        accuracies = {}
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies[model_name] = accuracy_score(y_test, y_pred)

        self.plot_accuracies(model_name=model_name, accuracies=accuracies)
        self.plot_roc_curve(model_name=model_name, y_test=y_test, y_pred=y_pred)


    def plot_accuracies(self, model_name:str, accuracies:dict):
        '''
        Summary:
            Plots accuracy scores of models to compare. Saves the plot in plots directory.
        
        Params:
            model_name: name of the model
            accuracies (dict): dictionary of accuracy scores.
        '''

        plt.figure(figsize=(8, 6))
        plt.bar(accuracies.keys(), accuracies.values(), color=['#A1D490', '#00509E', '#FF6F61'])
        plt.title('Model Accuracy Comparison', fontsize=16)
        plt.ylabel('Accuracy Score', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.ylim(0, 1)  # Accuracy skoru için 0-1 aralığı
        for i, v in enumerate(accuracies.values()):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
        plt.savefig(f'{self.plots_path}/{model_name}_accuracies.png')


    def plot_roc_curve(self, model_name:str, y_test, y_pred):
        '''
        Summary:
            Plots the roc curve and save the plot in plots/ directory

        Params:
            model_name: name of model
            y_test: actual targets
            y_pred: predicted targets
        '''

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend()
        plt.savefig(f'{self.plots_path}/{model_name}_roc_curve.png')

    # optimizing hyperparameters for increasing model accuracy
    def hyperparameter_optimization(self, model, model_name:str, param_grid:dict, X_train, X_test, y_train, y_test):
        '''
        Summary:
            Optimize hyperparameters using GridsearchCV and print best model resutls according to accuracy score.

        Params:
            model: model to optimize
            model_name (str): name of model
            param_grid (dict): parameters to optimize
            X_train, X_Test, y_train, y_test: train and test sets of features and target columns 
        
        Return:
            optimized_results (dict): Returns optimization results as dict.
        '''

        X = pd.concat([X_train, X_test], axis=0)
        y = pd.concat([y_train, y_test], axis=0)

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=True, return_train_score=False)
        grid.fit(X, y)

        optimized_results = {}
        optimized_results['best_score'] = grid.best_score_
        optimized_results['best_params'] = grid.best_params_
        optimized_results['best_estimator'] = grid.best_estimator_

        print(f'\nHyperparameter Optimization Results of {model_name}: \n')
        print(optimized_results)
        return optimized_results





if __name__ == '__main__':
    classifier = DiabetesClassification()
    classifier.main()