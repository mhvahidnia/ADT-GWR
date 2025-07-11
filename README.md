**Instructions for Using the ADT-GWR Python Program
**

1.	Install Required Packages
  At the beginning of the script, install any required packages listed.
    o	If using Google Colab, you only need to install the mgwr package for the standard GWR model before running the full code:

      !pip install mgwr

2.	Load Your Dataset
  In the ‘Data Preparation’ section, update the following line with the path to your dataset:

   data = pd.read_csv('/content/drive/MyDrive/GWR/Depression.csv')

   Your .csv file must follow this structure:
      o	The first two columns must contain geographic coordinates (X, Y)
      o	The third column is the dependent variable (e.g., Depression)
      o	All remaining columns should be independent variables

        Example:
        X          Y         Depression  PopDensity  PopFemale  NoHealthInsurance
        -82.3345   34.4119   23.1        61.48899           53.4              7.98
        -82.4113   34.32681  22.8        58.3391            48.1              10.96
        -82.5836   34.25005  23.2        35.92264           50.92             9.31
        -82.3685   34.22126  21.8        76.091             52.13             13.17

3.	Set the Neighborhood Range
  In the ‘Setting the range of neighborhood’ section, you can customize the neighborhood point range by adjusting:

  	neighbors_range = range(15, 51, 5)

  	o	This example sets a range from 15 to 50 with steps of 5.

4.	Save ADT-GWR Coefficients
  In the ‘Saving the coefficients’ section, modify the path to save the ADT-GWR coefficients:

    best_betas_df.to_csv('/your/custom/path/ADT_GWR_Betas.csv', index=False)

5.	Save Standard GWR Coefficients
  In the ‘Standard GWR’ section, update the file path to save the standard GWR coefficients:

  	gwr_betas_df.to_csv('/your/custom/path/Standard_GWR_Betas.csv', index=False)

6.	Customize Plot Titles
  In the ‘Residual Plot’ section, you can change the default plot title to reflect your dataset:

    plt.title('Your Dataset Name - Residuals Comparison')

7.	Run the Program
  Once you've made the necessary changes, simply run the full script to execute the ADT-GWR model.
