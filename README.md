# Mercedes-Benz-Greener-Manufacturing
Applied stacking of predictive models (linear regression, random forest regressor, decision tree regressor and XGBoost) to predict the manufacturing time (continuous) based on a number of given variables.  

## Libraries:
1. pandas
2. numpy
3. matplotlib.pylot
4. seaborn
5. sklearn
6. plotly
7. xgboost

## Files used:
1. The dataset the model was built on can be downloaded using [train.csv](https://github.com/pulkiB/Mercedes-Benz-Greener-Manufacturing/blob/master/train.csv)
2. The dataset used to test the model can be downloaded using [test.csv](https://github.com/pulkiB/Mercedes-Benz-Greener-Manufacturing/blob/master/test.csv)
3. My Python notebook where I built my model is [here](https://github.com/pulkiB/Mercedes-Benz-Greener-Manufacturing/blob/master/Mercedes_Greener_Manufacturing.ipynb)

## Part 1: Data Cleaning
I first determined the column types and counted them.
```
dtype_df= train.dtypes.reset_index()
dtype_df.columns = ["Count", "type"]
dtype_df.groupby("type").count()
```

I then checked for any null values or NaN values in my train set. There weren't any.

Examining the dataset, I noticed that the first 10 columns (ID, y, X0, X1, X2, X3, X4, X5, X6, X8) were either float or object types, while the others appeared to be binary.
To confirm whether the other columns were binary, I found the distribution of variables in each of these columns. Most columns were indeed binary (0,1), while the rest only contained 0 values. We did not need to consider the columns that only contained 0 in my model.

## Part 2: Data Analysis and Visualization
I plotted the distribution of 'y' values against each individual column (X0, X1, X2, X3, X4, X5, X6, X8) to determine which variables had the highest level of variance with y. From the plots, we can see that X0 and X2 had a noticable effect.

I then plotted the distribution of binary values per column. This was followed by a comparitive heatmap of the mean 'y' value distribution per binary value

## Part 3: Building predictive models
Now that I knew the distribution of our data, I needed to understand which variables had the greatest impact on the 'y' variable. To do so, I implemented Stacking algorithms.
However, since I was going to be implemnting multiple predictive algorithms, I first created a class, that would simplify this building process, called sklearnHelper.
The function `get_oof` was used to account for overfitting. It created multiple folds that would train the different levels of the model.

Now that I had all the tools necessary for building the model, I entered my model parameters and the models were built.
This first level used the Random Forest Regression, Decision Tree Regression and Linear Regression. I chose these models because 'y' was a continuous variable and no classication could be used.
To understand more of what was going on, I found out the variable importances and plotted the top 10 importances from Random Forest and Decision Tree.
```
      importance
X314    0.119788
X127    0.104398
X261    0.101774
X313    0.034566
X29     0.028221
X279    0.024657
X232    0.024630
X316    0.024279
X0      0.023902
X263    0.023875
      importance
X314    0.612542
X315    0.134614
X136    0.114786
X118    0.085955
X5      0.010145
X47     0.007738
X2      0.005885
X311    0.004386
X8      0.002923
X220    0.002831
```
We see that 'X314' is of the great importance in both models.

Using plotly, I plotted the mean importances from the two models.

Finally, these base level predictions became the training set for the the higher order prediction. I used XGBoost to build this final predictive model.

Credit must be given to @Anisotropic whose kernel on stacking really helped me out. It can be found [here](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)
