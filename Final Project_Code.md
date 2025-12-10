# Final Project-Data Wrangling


``` python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math as m
import random
```

``` python
death_how = pd.read_csv("/Users/seanpatnett/Downloads/NCHS_-_Leading_Causes_of_Death__United_States.csv")
death_how.columns = [col.lower().replace(" ", "_") for col in death_how.columns]
```

check for duplicates or null values

``` python
death_how.duplicated().sum() #0
death_how.isnull().sum() #0
```

    year                       0
    113_cause_name             0
    cause_name                 0
    state                      0
    deaths                     0
    age-adjusted_death_rate    0
    dtype: int64

``` python
num_of_death = pd.DataFrame(death_how.groupby(['state', 'year'])['deaths'].agg(['sum'])).reset_index()
```

Going to look at how death in the 5 most populus midwestern states moved
across time.

``` python
midwest_move = num_of_death.loc[(num_of_death['state'] == 'Illinois') | (num_of_death['state'] == 'Ohio') | (num_of_death['state'] == 'Michigan') | (num_of_death['state'] == 'Minnesota') | (num_of_death['state'] == 'Wisconsin'), :]
```

``` python
tick_locations = [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018]
plt.figure(figsize=[10,6])
sns.scatterplot(data=midwest_move, x='year', y='sum', hue='state')
plt.xticks(tick_locations)
plt.ylabel('Death Total')
plt.xlabel('Year')
plt.title('How does the death count vary across the midwest')
plt.show()
```

![](Final%20Project_Code_files/figure-commonmark/cell-7-output-1.png)

I noticed that death spiked in 2008, I know the 2008 housing crisis, was
detrimental. I am going to look into what caused the spike then. If
there is anything.

``` python
suicide_tracker = death_how.loc[death_how['cause_name'] == 'Suicide', ['year','state', 'cause_name', 'deaths']]
```

Now redo midwest_move variable to keep states consistent-

Making a suicide tracker for the midwest (comparing this to the chart
above)

``` python
midwest_s_move = suicide_tracker.loc[(suicide_tracker['state'] == 'Illinois') | (suicide_tracker['state'] == 'Ohio') | (suicide_tracker['state'] == 'Michigan') | (suicide_tracker['state'] == 'Minnesota') | (suicide_tracker['state'] == 'Wisconsin'), :] #going to use in scatterplot


tick_locations = [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018]
plt.figure(figsize=[10,6])
sns.scatterplot(data=midwest_s_move, x='year', y='deaths', hue='state')
plt.xticks(tick_locations)
plt.ylabel('Suicide Tracker')
plt.xlabel('Year')
plt.title('Did suicide trend up during 2008?')
plt.show()
```

![](Final%20Project_Code_files/figure-commonmark/cell-9-output-1.png)

- Question 1: Do I see a noticeable jump in suicide related deaths
  between 2007-2010 which might coorelate with the 2008 financial
  crisis?

``` python
suicide_tracker_usa = death_how.loc[(death_how['cause_name'] == 'Suicide') & (death_how['state'] == 'United States'), ['year','state', 'cause_name', 'deaths']] #suicide tracker for new visualization

plt.figure(figsize=[10,14]) 
sns.barplot(data=suicide_tracker_usa, y='year', x='deaths', orient='h')
#horizontal barplot
plt.xlabel('Suicide Tracker')
plt.ylabel('Year')
plt.title('Did suicide trend up during 2008?')
plt.show()
#doesnt really show much going to rethink approach
```

![](Final%20Project_Code_files/figure-commonmark/cell-10-output-1.png)

I need to find the sum of total deaths and the proportion of suicide
deaths each year, then I can see if there is a jump

``` python
suicide_tracker = death_how.loc[death_how['cause_name'] == 'Suicide', ['year','state', 'cause_name', 'deaths']] #ensuring it's updated

d1 = pd.DataFrame(death_how.groupby('year')['deaths'].sum()).reset_index()
d2 = pd.DataFrame(suicide_tracker.groupby('year')['deaths'].sum()).reset_index().rename(columns={'deaths': 'suicide_related_death'})


d4 = d1.merge(d2, on='year', how='inner') #merge two dfs to create proportion column
d4['suicide_prop_%'] = (d4['suicide_related_death']/d4['deaths']) * 100
```

Now undergoing the visualizations

``` python
tick_locations = [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018]
plt.figure(figsize=[10,6])
sns.scatterplot(data=d4, x='year', y='suicide_prop_%')
plt.xticks(tick_locations)
plt.ylabel('Suicide Proportion Tracker(%)')
plt.xlabel('Year')
plt.title('Did suicide trend up between 2007-2010?')
plt.show()
#scatterplot first
```

![](Final%20Project_Code_files/figure-commonmark/cell-12-output-1.png)

``` python
plt.figure(figsize=[10,14])
sns.barplot(data=d4, y='year', x='suicide_prop_%', orient='h')

plt.xlabel('Suicide Tracker')
plt.ylabel('Year')
plt.title('Did suicide trend up during Financial Recession')
plt.show()
#barplot check
```

![](Final%20Project_Code_files/figure-commonmark/cell-13-output-1.png)

Noticed that the rise between 2007-2010 was fairly large in regards to
suicide related deaths in proportion to total deaths.

Question 2: What is the largest cause of death in America?

``` python
import statsmodels.formula.api as smf
d5 = death_how.loc[death_how['cause_name'] != 'All causes', :]
model1 = smf.ols(data=d5, formula='deaths~C(cause_name) + 0').fit()
model1.summary()
```

|                   |                  |                     |             |
|-------------------|------------------|---------------------|-------------|
| Dep. Variable:    | deaths           | R-squared:          | 0.042       |
| Model:            | OLS              | Adj. R-squared:     | 0.041       |
| Method:           | Least Squares    | F-statistic:        | 48.03       |
| Date:             | Tue, 09 Dec 2025 | Prob (F-statistic): | 1.48e-85    |
| Time:             | 20:55:06         | Log-Likelihood:     | -1.1844e+05 |
| No. Observations: | 9880             | AIC:                | 2.369e+05   |
| Df Residuals:     | 9870             | BIC:                | 2.370e+05   |
| Df Model:         | 9                |                     |             |
| Covariance Type:  | nonrobust        |                     |             |

OLS Regression Results

|  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | coef | std err | t | P\>\|t\| | \[0.025 | 0.975\] |
| C(cause_name)\[Alzheimer's disease\] | 3025.9433 | 1237.812 | 2.445 | 0.015 | 599.578 | 5452.308 |
| C(cause_name)\[CLRD\] | 5252.8887 | 1237.812 | 4.244 | 0.000 | 2826.524 | 7679.254 |
| C(cause_name)\[Cancer\] | 2.195e+04 | 1237.812 | 17.733 | 0.000 | 1.95e+04 | 2.44e+04 |
| C(cause_name)\[Diabetes\] | 2833.8927 | 1237.812 | 2.289 | 0.022 | 407.528 | 5260.258 |
| C(cause_name)\[Heart disease\] | 2.474e+04 | 1237.812 | 19.989 | 0.000 | 2.23e+04 | 2.72e+04 |
| C(cause_name)\[Influenza and pneumonia\] | 2215.8725 | 1237.812 | 1.790 | 0.073 | -210.492 | 4642.237 |
| C(cause_name)\[Kidney disease\] | 1738.0830 | 1237.812 | 1.404 | 0.160 | -688.282 | 4164.448 |
| C(cause_name)\[Stroke\] | 5519.2773 | 1237.812 | 4.459 | 0.000 | 3092.912 | 7945.642 |
| C(cause_name)\[Suicide\] | 1410.9636 | 1237.812 | 1.140 | 0.254 | -1015.401 | 3837.328 |
| C(cause_name)\[Unintentional injuries\] | 4752.6721 | 1237.812 | 3.840 | 0.000 | 2326.307 | 7179.037 |

|                |           |                   |              |
|----------------|-----------|-------------------|--------------|
| Omnibus:       | 17874.550 | Durbin-Watson:    | 1.985        |
| Prob(Omnibus): | 0.000     | Jarque-Bera (JB): | 16912393.273 |
| Skew:          | 13.510    | Prob(JB):         | 0.00         |
| Kurtosis:      | 203.880   | Cond. No.         | 1.00         |

<br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

Noticed that Heart Disease, Cancer, and Stroke were the largest factors
in deaths in America. But Heart Disease and Cancer were significantly
higher than both crossing above 20,000.

Now going to view a few visualizations, sampling random states to see if
this trend continues across the US.

``` python
usa = death_how.loc[((death_how['state'] == 'Illinois') | (death_how['state'] == 'Ohio') | (death_how['state'] == 'Michigan') | (death_how['state'] == 'Minnesota') | (death_how['state'] == 'Wisconsin')) & (death_how['cause_name'] != 'All causes'), :]

plt.figure(figsize=[9,9])
sns.heatmap(pd.crosstab(usa['cause_name'], usa['state'], values=usa['deaths'], aggfunc='mean'), cbar=True, linewidths=0.8, annot=True, cmap='viridis')
```

![](Final%20Project_Code_files/figure-commonmark/cell-15-output-1.png)

``` python
e = death_how.loc[(death_how['cause_name'] != 'All causes') & (death_how['year'] > 2010) & (death_how['state'] == 'United States'), ['year','state', 'cause_name', 'deaths']]
plt.figure(figsize=[12,12])
sns.heatmap(pd.crosstab(e['cause_name'], e['year'], values=e['deaths'], aggfunc='mean'), cbar=True, linewidths=0.8, annot=True, cmap='viridis')
plt.xlabel("Years")
plt.ylabel("Cause of Death")
plt.title("How did causes of death trend")
```

    Text(0.5, 1.0, 'How did causes of death trend')

![](Final%20Project_Code_files/figure-commonmark/cell-16-output-2.png)

``` python
death_how1 = death_how.loc[((death_how['state'] == 'New York') | (death_how['state'] == 'Alaska') | (death_how['state'] == 'Florida') | (death_how['state'] == 'Oregon') | (death_how['state'] == 'California')) & (death_how['cause_name'] != 'All causes') , :]


sns.heatmap(pd.crosstab(death_how1['cause_name'], death_how1['state'], values=death_how1['deaths'], aggfunc='mean'), cbar=True, linewidths=1, annot=True, cmap='viridis')
plt.xlabel("State Sampled")
plt.ylabel("Cause of Death")
plt.title("How did causes of death vary in select states")
```

    Text(0.5, 1.0, 'How did causes of death vary in select states')

![](Final%20Project_Code_files/figure-commonmark/cell-17-output-2.png)

Question 3: Is death per 100,000 people rising and how does cause of
death vary by state ?

``` python
us = death_how.loc[(death_how['state'] == 'United States') & (death_how['cause_name'] == 'All causes'), :]


tick_locations = [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018]
sns.scatterplot(data=us, x='year', y='age-adjusted_death_rate')
plt.xticks(tick_locations)
plt.ylabel('Death Rate per 100,000')
plt.xlabel('Year')
plt.title('How has Death Rate shifted?')
plt.show()
```

![](Final%20Project_Code_files/figure-commonmark/cell-18-output-1.png)

We see that death per 100,000 people is actually decreasing at a high
rate.

Doing a second dataset: Second part of project, going to use to compare
America to the three other largest economies by GDP

``` python
#import data
world_work = pd.read_csv("/Users/seanpatnett/Downloads/Data Wrangling_MSBA/Final Project Presentation/UnifiedDataset.csv")
world_work.columns = [col.lower().replace(" ", "_") for col in world_work.columns]
```

Now let’s get america alone, and key columns

``` python
america = world_work.loc[(world_work['country'] == 'United States')  & (world_work['gender'] == 'Both sexes') & (world_work['year'] > 1997), ['gender', 'year', 'birth_rate', 'death_rate', 'life_expectancy', 'homicide_rate',
 'gdp_per_capita', 'income_per_capita',   'basic_sanization_services_total' ]].reset_index().drop(columns='index', axis=1)
```

``` python
#America solo, homicide rate
tick_locations = [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018]
sns.scatterplot(data=america, x='year', y='homicide_rate')
plt.title('Homicide Rate through 21st century')
plt.ylabel("Homicide Rate")
plt.xlabel("Year")
plt.xticks(tick_locations)
```

    ([<matplotlib.axis.XTick at 0x127521210>,
      <matplotlib.axis.XTick at 0x127521d50>,
      <matplotlib.axis.XTick at 0x12756fc90>,
      <matplotlib.axis.XTick at 0x127572210>,
      <matplotlib.axis.XTick at 0x1275787d0>,
      <matplotlib.axis.XTick at 0x1275717d0>,
      <matplotlib.axis.XTick at 0x12757bf10>,
      <matplotlib.axis.XTick at 0x12757e590>,
      <matplotlib.axis.XTick at 0x127584b50>,
      <matplotlib.axis.XTick at 0x127587050>,
      <matplotlib.axis.XTick at 0x12757cfd0>],
     [Text(1998, 0, '1998'),
      Text(2000, 0, '2000'),
      Text(2002, 0, '2002'),
      Text(2004, 0, '2004'),
      Text(2006, 0, '2006'),
      Text(2008, 0, '2008'),
      Text(2010, 0, '2010'),
      Text(2012, 0, '2012'),
      Text(2014, 0, '2014'),
      Text(2016, 0, '2016'),
      Text(2018, 0, '2018')])

![](Final%20Project_Code_files/figure-commonmark/cell-21-output-2.png)

``` python
#America solo, birth rate
tick_locations = [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018]
sns.scatterplot(data=america, x='year', y='birth_rate')
plt.title('Birth Rate through 21st century')
plt.ylabel("Birth Rate")
plt.xlabel("Year")
plt.xticks(tick_locations)
```

    ([<matplotlib.axis.XTick at 0x12756ff10>,
      <matplotlib.axis.XTick at 0x127571d50>,
      <matplotlib.axis.XTick at 0x126943e90>,
      <matplotlib.axis.XTick at 0x12690ad10>,
      <matplotlib.axis.XTick at 0x126915250>,
      <matplotlib.axis.XTick at 0x126917890>,
      <matplotlib.axis.XTick at 0x126928c10>,
      <matplotlib.axis.XTick at 0x126929cd0>,
      <matplotlib.axis.XTick at 0x126920bd0>,
      <matplotlib.axis.XTick at 0x1269317d0>,
      <matplotlib.axis.XTick at 0x126932c50>],
     [Text(1998, 0, '1998'),
      Text(2000, 0, '2000'),
      Text(2002, 0, '2002'),
      Text(2004, 0, '2004'),
      Text(2006, 0, '2006'),
      Text(2008, 0, '2008'),
      Text(2010, 0, '2010'),
      Text(2012, 0, '2012'),
      Text(2014, 0, '2014'),
      Text(2016, 0, '2016'),
      Text(2018, 0, '2018')])

![](Final%20Project_Code_files/figure-commonmark/cell-22-output-2.png)

``` python
#America solo, income per capita
tick_locations = [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018]
sns.scatterplot(data=america, x='year', y='income_per_capita')
plt.title('Income per Capita through 21st century')
plt.ylabel("Income per Capita")
plt.xlabel("Year")
plt.xticks(tick_locations)
```

    ([<matplotlib.axis.XTick at 0x1268e3790>,
      <matplotlib.axis.XTick at 0x1268e9d50>,
      <matplotlib.axis.XTick at 0x126867050>,
      <matplotlib.axis.XTick at 0x12688e150>,
      <matplotlib.axis.XTick at 0x126877e50>,
      <matplotlib.axis.XTick at 0x1264dc550>,
      <matplotlib.axis.XTick at 0x1275b1450>,
      <matplotlib.axis.XTick at 0x1275b5490>,
      <matplotlib.axis.XTick at 0x1275b77d0>,
      <matplotlib.axis.XTick at 0x1266c39d0>,
      <matplotlib.axis.XTick at 0x1275c6b10>],
     [Text(1998, 0, '1998'),
      Text(2000, 0, '2000'),
      Text(2002, 0, '2002'),
      Text(2004, 0, '2004'),
      Text(2006, 0, '2006'),
      Text(2008, 0, '2008'),
      Text(2010, 0, '2010'),
      Text(2012, 0, '2012'),
      Text(2014, 0, '2014'),
      Text(2016, 0, '2016'),
      Text(2018, 0, '2018')])

![](Final%20Project_Code_files/figure-commonmark/cell-23-output-2.png)

``` python
#Not relevant to project -----------------------------
correlation = round(america['life_expectancy'].corr(america['homicide_rate']), 2)
print(f'The coorelation between life expectancy and homeicide rate is strongly negatively coorelated at {correlation}.')
#Not relevant to project -----------------------------
```

    The coorelation between life expectancy and homeicide rate is strongly negatively coorelated at -0.92.

Now subsetting the 4 largest economies:

``` python
#subsetting 
north_america = world_work.loc[((world_work['country'] == 'United States') | (world_work['country'] == 'China')| (world_work['country'] == 'Germany') | (world_work['country'] == 'Japan')) & (world_work['gender'] == 'Both sexes') & (world_work['year'] > 1997), ['year', 'country', 'birth_rate', 'death_rate', 'life_expectancy', 'homicide_rate',
 'gdp_per_capita', 'income_per_capita',  'total_population' ,'basic_sanization_services_total']].reset_index().drop(columns='index', axis=1)
```

``` python
#comparing homicide rate
tick_locations = [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018]
sns.scatterplot(data=north_america, x='year', y='homicide_rate', hue='country')
plt.title('Homicide Rate through 21st century: Comparing 4 largest economies')
plt.ylabel("Homicide Rate")
plt.xlabel("Year")
plt.xticks(tick_locations)
```

    ([<matplotlib.axis.XTick at 0x1275f6c50>,
      <matplotlib.axis.XTick at 0x1275d9d50>,
      <matplotlib.axis.XTick at 0x12761dbd0>,
      <matplotlib.axis.XTick at 0x1276240d0>,
      <matplotlib.axis.XTick at 0x127626510>,
      <matplotlib.axis.XTick at 0x127625a50>,
      <matplotlib.axis.XTick at 0x12762dcd0>,
      <matplotlib.axis.XTick at 0x127630050>,
      <matplotlib.axis.XTick at 0x127632510>,
      <matplotlib.axis.XTick at 0x12763c9d0>,
      <matplotlib.axis.XTick at 0x127627f90>],
     [Text(1998, 0, '1998'),
      Text(2000, 0, '2000'),
      Text(2002, 0, '2002'),
      Text(2004, 0, '2004'),
      Text(2006, 0, '2006'),
      Text(2008, 0, '2008'),
      Text(2010, 0, '2010'),
      Text(2012, 0, '2012'),
      Text(2014, 0, '2014'),
      Text(2016, 0, '2016'),
      Text(2018, 0, '2018')])

![](Final%20Project_Code_files/figure-commonmark/cell-26-output-2.png)

``` python
#comparing birth rate
tick_locations = [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018]
sns.scatterplot(data=north_america, x='year', y='birth_rate', hue='country')
plt.title('Birth Rate through 21st century: Comparing 4 largest economies')
plt.ylabel("Birth Rate ")
plt.xlabel("Year")
plt.xticks(tick_locations)
```

    ([<matplotlib.axis.XTick at 0x12768d990>,
      <matplotlib.axis.XTick at 0x1276b4590>,
      <matplotlib.axis.XTick at 0x1276b6150>,
      <matplotlib.axis.XTick at 0x1276bc810>,
      <matplotlib.axis.XTick at 0x1276bed10>,
      <matplotlib.axis.XTick at 0x1276c1290>,
      <matplotlib.axis.XTick at 0x1276c36d0>,
      <matplotlib.axis.XTick at 0x1276b7250>,
      <matplotlib.axis.XTick at 0x1276c6710>,
      <matplotlib.axis.XTick at 0x1276ccb90>,
      <matplotlib.axis.XTick at 0x1276cf090>],
     [Text(1998, 0, '1998'),
      Text(2000, 0, '2000'),
      Text(2002, 0, '2002'),
      Text(2004, 0, '2004'),
      Text(2006, 0, '2006'),
      Text(2008, 0, '2008'),
      Text(2010, 0, '2010'),
      Text(2012, 0, '2012'),
      Text(2014, 0, '2014'),
      Text(2016, 0, '2016'),
      Text(2018, 0, '2018')])

![](Final%20Project_Code_files/figure-commonmark/cell-27-output-2.png)

``` python
#comparing GDP
tick_locations = [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018]
sns.scatterplot(data=north_america, x='year', y='gdp_per_capita', hue='country')
plt.title('GDP per Capita through 21st century: Comparing 4 largest economies')
plt.ylabel("GDP per Capita ")
plt.xlabel("Year")
plt.xticks(tick_locations)
'gdp_per_capita'
```

    'gdp_per_capita'

![](Final%20Project_Code_files/figure-commonmark/cell-28-output-2.png)

``` python
#comparing Income
tick_locations = [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018]
sns.scatterplot(data=north_america, x='year', y='income_per_capita', hue='country')
plt.title('Income Per Capita through 21st century: Comparing 4 largest economies')
plt.ylabel("Income Per Capita")
plt.xlabel("Year")
plt.xticks(tick_locations)
```

    ([<matplotlib.axis.XTick at 0x1277b1d90>,
      <matplotlib.axis.XTick at 0x1277da910>,
      <matplotlib.axis.XTick at 0x1277e4910>,
      <matplotlib.axis.XTick at 0x1277e6d10>,
      <matplotlib.axis.XTick at 0x1277a8490>,
      <matplotlib.axis.XTick at 0x1277ea950>,
      <matplotlib.axis.XTick at 0x1277f0e50>,
      <matplotlib.axis.XTick at 0x1277f3290>,
      <matplotlib.axis.XTick at 0x1277f9690>,
      <matplotlib.axis.XTick at 0x1277e9490>,
      <matplotlib.axis.XTick at 0x1278008d0>],
     [Text(1998, 0, '1998'),
      Text(2000, 0, '2000'),
      Text(2002, 0, '2002'),
      Text(2004, 0, '2004'),
      Text(2006, 0, '2006'),
      Text(2008, 0, '2008'),
      Text(2010, 0, '2010'),
      Text(2012, 0, '2012'),
      Text(2014, 0, '2014'),
      Text(2016, 0, '2016'),
      Text(2018, 0, '2018')])

![](Final%20Project_Code_files/figure-commonmark/cell-29-output-2.png)

``` python
#comparing Life Expectancy
tick_locations = [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018]
sns.scatterplot(data=north_america, x='year', y='life_expectancy', hue='country')
plt.title('Life Expectancy through 21st century: Comparing 4 largest economies')
plt.ylabel("Life Expectancy")
plt.xlabel("Year")
plt.xticks(tick_locations)
```

    ([<matplotlib.axis.XTick at 0x127843090>,
      <matplotlib.axis.XTick at 0x1278543d0>,
      <matplotlib.axis.XTick at 0x12787e250>,
      <matplotlib.axis.XTick at 0x127885a90>,
      <matplotlib.axis.XTick at 0x127888090>,
      <matplotlib.axis.XTick at 0x12788a490>,
      <matplotlib.axis.XTick at 0x12788cbd0>,
      <matplotlib.axis.XTick at 0x127834dd0>,
      <matplotlib.axis.XTick at 0x1278bf690>,
      <matplotlib.axis.XTick at 0x1278c1210>,
      <matplotlib.axis.XTick at 0x1278c3710>],
     [Text(1998, 0, '1998'),
      Text(2000, 0, '2000'),
      Text(2002, 0, '2002'),
      Text(2004, 0, '2004'),
      Text(2006, 0, '2006'),
      Text(2008, 0, '2008'),
      Text(2010, 0, '2010'),
      Text(2012, 0, '2012'),
      Text(2014, 0, '2014'),
      Text(2016, 0, '2016'),
      Text(2018, 0, '2018')])

![](Final%20Project_Code_files/figure-commonmark/cell-30-output-2.png)

``` python
#comparing total population average
sns.barplot(data=north_america, y='country', x='total_population', hue='country')
plt.title('Total Population through 21st century: Comparing 4 largest economies')
plt.xlabel("Total Population(Measured in Billions)")
plt.xlim(60000000, 1400000000)
plt.ylabel("Country")
```

    Text(0, 0.5, 'Country')

![](Final%20Project_Code_files/figure-commonmark/cell-31-output-2.png)

Going to compare aging of population for last bench mark. Need to add to
my original subset frames.

``` python
america12 = world_work.loc[(world_work['country'] == 'United States')  & (world_work['gender'] == 'Both sexes') & (world_work['year'] > 1997), ['gender', 'year', 'birth_rate', 'death_rate', 'life_expectancy', 'homicide_rate',
 'gdp_per_capita', 'income_per_capita',
 '%_population_aged_0-14',
 '%_population_aged_15-64',
 '%_population_aged_65+',
]].reset_index().drop(columns='index', axis=1)
```

``` python
north_america12 = world_work.loc[((world_work['country'] == 'United States') | (world_work['country'] == 'China')| (world_work['country'] == 'Germany') | (world_work['country'] == 'Japan')) & (world_work['gender'] == 'Both sexes') & (world_work['year'] > 1997), ['country', 'gender', 'year', 'birth_rate', 'death_rate', 'life_expectancy', 'homicide_rate',
 'gdp_per_capita', 'income_per_capita',
 '%_population_aged_0-14',
 '%_population_aged_15-64',
 '%_population_aged_65+',
]].reset_index().drop(columns='index', axis=1)
```

Now need to melt data(make it longer so I do scatterplots)

``` python
america12_melt = america12.melt(id_vars=['gender', 'year', 'birth_rate', 'death_rate', 'life_expectancy', 'homicide_rate','gdp_per_capita', 'income_per_capita'], var_name='population', value_name='population_by_age')
```

``` python
north_america12_melt = north_america12.melt(id_vars=['country', 'gender', 'year', 'birth_rate', 'death_rate', 'life_expectancy', 'homicide_rate','gdp_per_capita', 'income_per_capita'], var_name='population', value_name='population_by_age')
```

``` python
sns.scatterplot(data=america12_melt, x='year', y='population_by_age', hue='population')
```

![](Final%20Project_Code_files/figure-commonmark/cell-36-output-1.png)

``` python
color = {
    'China': 'orange',
    'Japan': 'green',
    'United States': 'blue',
    'Germany': 'red'
}
plt.figure(figsize=[10,10])
sns.scatterplot(data=north_america12_melt, x='year', y='population_by_age', hue='country', palette=color, style='population')
plt.legend()
```

![](Final%20Project_Code_files/figure-commonmark/cell-37-output-1.png)
