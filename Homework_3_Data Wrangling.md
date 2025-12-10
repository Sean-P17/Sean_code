# Final Project-Data Wrangling


# Looking at American trends

### Import packages

``` python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math as m
import random
```

### Importing datasets

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | year | 113_cause_name | cause_name | state | deaths | age-adjusted_death_rate |
|----|----|----|----|----|----|----|
| 0 | 2017 | Accidents (unintentional injuries) (V01-X59,Y8... | Unintentional injuries | United States | 169936 | 49.4 |
| 1 | 2017 | Accidents (unintentional injuries) (V01-X59,Y8... | Unintentional injuries | Alabama | 2703 | 53.8 |
| 2 | 2017 | Accidents (unintentional injuries) (V01-X59,Y8... | Unintentional injuries | Alaska | 436 | 63.7 |
| 3 | 2017 | Accidents (unintentional injuries) (V01-X59,Y8... | Unintentional injuries | Arizona | 4184 | 56.2 |
| 4 | 2017 | Accidents (unintentional injuries) (V01-X59,Y8... | Unintentional injuries | Arkansas | 1625 | 51.8 |

</div>

Checking for potential data inconsistencies.

    year                       0
    113_cause_name             0
    cause_name                 0
    state                      0
    deaths                     0
    age-adjusted_death_rate    0
    dtype: int64

Creating a Dataframe for death by year in each state.

Going to look at how death in the 5 most populus midwestern states moved
across time.

``` python
midwest_move = num_of_death.loc[(num_of_death['state'] == 'Illinois') | (num_of_death['state'] == 'Ohio') | (num_of_death['state'] == 'Michigan') | (num_of_death['state'] == 'Minnesota') | (num_of_death['state'] == 'Wisconsin'), :]
```

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-7-output-1.png)

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

#| echo: false
tick_locations = [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018]
plt.figure(figsize=[10,6])
sns.scatterplot(data=midwest_s_move, x='year', y='deaths', hue='state')
plt.xticks(tick_locations)
plt.ylabel('Suicide Tracker')
plt.xlabel('Year')
plt.title('Did suicide trend up during 2008?')
plt.show()
```

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-9-output-1.png)

#### Question 1: Do I see a noticeable jump in suicide related deaths between 2007-2010 which might coorelate with the 2008 financial crisis?

``` python
suicide_tracker_usa = death_how.loc[(death_how['cause_name'] == 'Suicide') & (death_how['state'] == 'United States'), ['year','state', 'cause_name', 'deaths']] #suicide tracker for new visualization

#| echo: false
plt.figure(figsize=[10,14]) 
sns.barplot(data=suicide_tracker_usa, y='year', x='deaths', orient='h')
#horizontal barplot
plt.xlabel('Suicide Tracker')
plt.ylabel('Year')
plt.title('Did suicide trend up during 2008?')
plt.show()
#doesnt really show much going to rethink approach
```

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-10-output-1.png)

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

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-12-output-1.png)

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-13-output-1.png)

Noticed that the rise between 2007-2010 was fairly large in regards to
suicide related deaths in proportion to total deaths.

#### Question 2: What is the largest cause of death in America?

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
| Date:             | Wed, 10 Dec 2025 | Prob (F-statistic): | 1.48e-85    |
| Time:             | 09:17:17         | Log-Likelihood:     | -1.1844e+05 |
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

First using a heatmap to visualize rate of deaths by cause in a select
few states

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-15-output-1.png)

looking at how death by cause moved throughout the 2010s.

    Text(0.5, 1.0, 'How did causes of death trend')

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-16-output-2.png)

Repeating a heatmap, but this time with random states to check for
signifcant variance.

    Text(0.5, 1.0, 'How did causes of death vary in select states')

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-17-output-2.png)

#### Question 3: Is death per 100,000 people rising and how does cause of death vary by state ?

Looking at death rate per 100,000 movement

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-18-output-1.png)

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

Analyzing homicide rate:

    ([<matplotlib.axis.XTick at 0x115d20b50>,
      <matplotlib.axis.XTick at 0x115d50810>,
      <matplotlib.axis.XTick at 0x115d526d0>,
      <matplotlib.axis.XTick at 0x115d5cd50>,
      <matplotlib.axis.XTick at 0x115d5f290>,
      <matplotlib.axis.XTick at 0x115d5d890>,
      <matplotlib.axis.XTick at 0x115d62890>,
      <matplotlib.axis.XTick at 0x115d68ed0>,
      <matplotlib.axis.XTick at 0x115d6b310>,
      <matplotlib.axis.XTick at 0x115d6d750>,
      <matplotlib.axis.XTick at 0x115d61fd0>],
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

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-21-output-2.png)

Birth Rate for America

    ([<matplotlib.axis.XTick at 0x115a05150>,
      <matplotlib.axis.XTick at 0x1153465d0>,
      <matplotlib.axis.XTick at 0x115345cd0>,
      <matplotlib.axis.XTick at 0x11534ff90>,
      <matplotlib.axis.XTick at 0x115358f50>,
      <matplotlib.axis.XTick at 0x11535b7d0>,
      <matplotlib.axis.XTick at 0x115374750>,
      <matplotlib.axis.XTick at 0x115377f10>,
      <matplotlib.axis.XTick at 0x115387fd0>,
      <matplotlib.axis.XTick at 0x11535d6d0>,
      <matplotlib.axis.XTick at 0x1153453d0>],
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

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-22-output-2.png)

How is income trending?

    ([<matplotlib.axis.XTick at 0x11533a910>,
      <matplotlib.axis.XTick at 0x115219c90>,
      <matplotlib.axis.XTick at 0x11521b750>,
      <matplotlib.axis.XTick at 0x1152208d0>,
      <matplotlib.axis.XTick at 0x115219590>,
      <matplotlib.axis.XTick at 0x114fc7990>,
      <matplotlib.axis.XTick at 0x114fe3550>,
      <matplotlib.axis.XTick at 0x114fe3b10>,
      <matplotlib.axis.XTick at 0x114fcbbd0>,
      <matplotlib.axis.XTick at 0x1151a6a90>,
      <matplotlib.axis.XTick at 0x115205950>],
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

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-23-output-2.png)

    The coorelation between life expectancy and homeicide rate is strongly negatively coorelated at -0.92.

Now subsetting the 4 largest economies:

``` python
#subsetting 
north_america = world_work.loc[((world_work['country'] == 'United States') | (world_work['country'] == 'China')| (world_work['country'] == 'Germany') | (world_work['country'] == 'Japan')) & (world_work['gender'] == 'Both sexes') & (world_work['year'] > 1997), ['year', 'country', 'birth_rate', 'death_rate', 'life_expectancy', 'homicide_rate',
 'gdp_per_capita', 'income_per_capita',  'total_population' ,'basic_sanization_services_total']].reset_index().drop(columns='index', axis=1)
```

Homicide Rate comparision by country over the years

    ([<matplotlib.axis.XTick at 0x114fd5650>,
      <matplotlib.axis.XTick at 0x115245e10>,
      <matplotlib.axis.XTick at 0x114ff5c50>,
      <matplotlib.axis.XTick at 0x114feb3d0>,
      <matplotlib.axis.XTick at 0x114feaa10>,
      <matplotlib.axis.XTick at 0x115245ad0>,
      <matplotlib.axis.XTick at 0x114f46690>,
      <matplotlib.axis.XTick at 0x114f67c10>,
      <matplotlib.axis.XTick at 0x114fda790>,
      <matplotlib.axis.XTick at 0x114ffbc10>,
      <matplotlib.axis.XTick at 0x114f45fd0>],
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

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-26-output-2.png)

Birth Rate Comparisions:

    ([<matplotlib.axis.XTick at 0x115dedad0>,
      <matplotlib.axis.XTick at 0x115e086d0>,
      <matplotlib.axis.XTick at 0x115e0a390>,
      <matplotlib.axis.XTick at 0x115e10790>,
      <matplotlib.axis.XTick at 0x115e12c90>,
      <matplotlib.axis.XTick at 0x115e1d290>,
      <matplotlib.axis.XTick at 0x115e1f890>,
      <matplotlib.axis.XTick at 0x115e0b310>,
      <matplotlib.axis.XTick at 0x115e269d0>,
      <matplotlib.axis.XTick at 0x115e28f90>,
      <matplotlib.axis.XTick at 0x115e2b450>],
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

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-27-output-2.png)

Comparing GDP across nations

    'gdp_per_capita'

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-28-output-2.png)

How does Income vary:

    ([<matplotlib.axis.XTick at 0x115f19210>,
      <matplotlib.axis.XTick at 0x115f3dd10>,
      <matplotlib.axis.XTick at 0x115f3fad0>,
      <matplotlib.axis.XTick at 0x115f46150>,
      <matplotlib.axis.XTick at 0x115f442d0>,
      <matplotlib.axis.XTick at 0x115f49f50>,
      <matplotlib.axis.XTick at 0x115f4c410>,
      <matplotlib.axis.XTick at 0x115f4e750>,
      <matplotlib.axis.XTick at 0x115f50bd0>,
      <matplotlib.axis.XTick at 0x115f49510>,
      <matplotlib.axis.XTick at 0x115f53d10>],
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

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-29-output-2.png)

Comparing Life Expectancy:

    ([<matplotlib.axis.XTick at 0x115e66990>,
      <matplotlib.axis.XTick at 0x115fcd190>,
      <matplotlib.axis.XTick at 0x115fcea90>,
      <matplotlib.axis.XTick at 0x115fd1050>,
      <matplotlib.axis.XTick at 0x115fd35d0>,
      <matplotlib.axis.XTick at 0x115fddb10>,
      <matplotlib.axis.XTick at 0x115fdfe50>,
      <matplotlib.axis.XTick at 0x115fce1d0>,
      <matplotlib.axis.XTick at 0x11600e4d0>,
      <matplotlib.axis.XTick at 0x116018050>,
      <matplotlib.axis.XTick at 0x11601a5d0>],
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

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-30-output-2.png)

Comparing total population

    Text(0, 0.5, 'Country')

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-31-output-2.png)

Going to compare aging of population for last bench mark. Need to add to
my original subset frames.

Now need to melt data(make it longer so I do scatterplots)

``` python
america12_melt = america12.melt(id_vars=['gender', 'year', 'birth_rate', 'death_rate', 'life_expectancy', 'homicide_rate','gdp_per_capita', 'income_per_capita'], var_name='population', value_name='population_by_age')
```

    Text(0, 0.5, 'Population by Age')

![](Homework_3_Data%20Wrangling_files/figure-commonmark/cell-34-output-2.png)
