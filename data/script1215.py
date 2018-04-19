
# coding: utf-8

# # Motivation
# I'll use this script to provide introduction to data analysis using SQL language, which should be a must tool for every data scientist - both for getting access to data, but more interesting, as a simple tool for advance data analysis.
# The logic behind SQL is very similar to any other tool or language that used for data analysis (excel, Pandas),  and for those that used to work with data, should be very intuitive. 
# 
# Feel free to response with questions, and I'll try to explain more if need.

# # Important Definitions
# SQL is a conceptual language for working with data stored in databases. In our case, SQLite is the specific implementation.
# Eventually, we will use SQL lunguage to write queries that would pull data from the DB, manipulate it, sort it, and extract it.
# 
# The most important component of the DB  is its tables - that's where all the data stored. Usually the data would be devided to many tables, and not stored all in one place (so designing the data stracture propely is very important). Most of this script would handle how to work with tables.
# Other than tables, there are some other very useful concepts/features that we won't talk about:
# * table creation
# * inserting / updating data in the DB
# * functions - gets a value as an input, and returns manipulation of that value (for example function that remove white spaces)

# In[ ]:


#Improts 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

path = "../input/"  #Insert path here
database = path + 'database.sqlite'


# # First we will create the connection to the DB, and see what tables we have
# The basic structure of the query is very simple:
# You define what you want to see after the SELECT, * means all possible columns
# You choose the table after the FROM
# You add the conditions for the data you want to use from the table(s) after the WHERE
# 
# The stracture, and the order of the sections matter, while spaces, new lines, capital words and indentation are there to make the code easier to read.

# In[ ]:


conn = sqlite3.connect(database)

tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table';""", conn)
tables


# # List of countries
# This is the most basic query.
# The only must parts of a qeury is the SELECT and the FROM (assuming you want to pull from a table)

# In[ ]:


countries = pd.read_sql("""SELECT *
                        FROM Country;""", conn)
countries


# # List of leagues and their country 
# JOIN is used when you want to connect two tables to each other. It works when you have a common key in each of them.
# Understanding the concept of Keys is crucial for connecting (joining) between data set (tables). 
# A key is uniquely identifies each record (row) in a table. 
# It can consinst of one value (cell) - usually ID, or from a combination of values that are unique in the table.
# 
# When joinin between different tables, you must:
# * Decide what type of join to use. The most common are:
# * * (INNER) JOIN - keep only records that match the condition (after the ON) in both the tables, and records in both tables that do not match wouldn't appear in the output
# * * LEFT JOIN - keep all the values from the first (left) table - in conjunction with the matching rows from the right table. The columns from the right table, that don't have matching value in the left, would have NULL values. 
# * Specify the common value that is used to connect the tables (the id of the country in that case). 
# * Make sure that at least one of the values has to be a key in its table. In our case, it's the Country.id. The League.country_id is not unique, as there can be more than one league in the same country
# 
# JOINs, and using them incorrectly, is the most common and dangerious mistake when writing complicated queries

# In[ ]:


leagues = pd.read_sql("""SELECT *
                        FROM League
                        JOIN Country ON Country.id = League.country_id;""", conn)
leagues


# # List of teams
# ORDER BY defines the sorting of the output - ascending or descending (DESC)
# 
# LIMIT, limits the number of rows in the output - after the sorting

# In[ ]:


teams = pd.read_sql("""SELECT *
                        FROM Team
                        ORDER BY team_long_name
                        LIMIT 10;""", conn)
teams


# # List of matches
# In this exapmle we will show only the columns that interests us, so instead of * we will use the exact names.
# 
# Some of the cells have the same name (Country.name,League.name). We will rename them using AS.
# 
# As you can see, this query has much more joins. The reasons is because the DB is designed in a star
# structure - one table (Match) with all the "performance" and metrics, but only keys and IDs,
# while all the descriptive information stored in other tables (Country, League, Team)
# 
# Note that Team is joined twice. This is a tricky one, as while we are using the same table name, we basically bring two different copies (and rename them using AS). The reason is that we need to bring information about two different values (home_team_api_id, away_team_api_id), and if we join them to the same table, it would mean that they are equal to each other.
# 
# You will also note that the Team tables are joined using left join. The reason is decided that I would prefer to keep the matches in the output - even if on of the teams doesn't appear in the Team table.
# 
# ORDER defines the order of the output, and comes before the LIMIT and after the WHERE

# In[ ]:


detailed_matches = pd.read_sql("""SELECT Match.id, 
                                        Country.name AS country_name, 
                                        League.name AS league_name, 
                                        season, 
                                        stage, 
                                        date,
                                        HT.team_long_name AS  home_team,
                                        AT.team_long_name AS away_team,
                                        home_team_goal, 
                                        away_team_goal                                        
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE country_name = 'Spain'
                                ORDER by date
                                LIMIT 10;""", conn)
detailed_matches


# # Let's do some basic analytics
# Here we are starting to look at the data at more aggregated level. Instead of looking on the raw data we will start to grouping it to different levels we want to examine.
# In this example, we will base it on the previous query, remove the match and date information, and look at it at the country-league-season level.
# 
# The functionality we will use for that is GROUP BY, that comes between the WHERE and ORDER
# 
# Once you chose what level you want to analyse, we can devide the select statement to two:
# * Dimensions - those are the values we describing, same that we group by later.
# * Metrics - all the metrics have to be aggregated using functions.. 
# The common functions are: sum(), count(), count(distinct), avg(), min(), max()
# 
# Note - it is very important to use the same dimensions both in the select, and in the GROUP BY. Otherwise the output might be wrong.
# 
# Another functionality that can be used after grouping, is HAVING. This adds another layer of filtering the data, this time the output of the table **after** the grouping. A lot of times it is used to clean the output.
# 

# In[ ]:


leages_by_season = pd.read_sql("""SELECT Country.name AS country_name, 
                                        League.name AS league_name, 
                                        season,
                                        count(distinct stage) AS number_of_stages,
                                        count(distinct HT.team_long_name) AS number_of_teams,
                                        avg(home_team_goal) AS avg_home_team_scors, 
                                        avg(away_team_goal) AS avg_away_team_goals, 
                                        avg(home_team_goal-away_team_goal) AS avg_goal_dif, 
                                        avg(home_team_goal+away_team_goal) AS avg_goals, 
                                        sum(home_team_goal+away_team_goal) AS total_goals                                       
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE country_name in ('Spain', 'Germany', 'France', 'Italy', 'England')
                                GROUP BY Country.name, League.name, season
                                HAVING count(distinct stage) > 10
                                ORDER BY Country.name, League.name, season DESC
                                ;""", conn)
leages_by_season


# In[ ]:


df = pd.DataFrame(index=np.sort(leages_by_season['season'].unique()), columns=leages_by_season['country_name'].unique())

df.loc[:,'Germany'] = list(leages_by_season.loc[leages_by_season['country_name']=='Germany','avg_goals'])
df.loc[:,'Spain']   = list(leages_by_season.loc[leages_by_season['country_name']=='Spain','avg_goals'])
df.loc[:,'France']   = list(leages_by_season.loc[leages_by_season['country_name']=='France','avg_goals'])
df.loc[:,'Italy']   = list(leages_by_season.loc[leages_by_season['country_name']=='Italy','avg_goals'])
df.loc[:,'England']   = list(leages_by_season.loc[leages_by_season['country_name']=='England','avg_goals'])

df.plot(figsize=(12,5),title='Average Goals per Game Over Time')


# In[ ]:


df = pd.DataFrame(index=np.sort(leages_by_season['season'].unique()), columns=leages_by_season['country_name'].unique())

df.loc[:,'Germany'] = list(leages_by_season.loc[leages_by_season['country_name']=='Germany','avg_goal_dif'])
df.loc[:,'Spain']   = list(leages_by_season.loc[leages_by_season['country_name']=='Spain','avg_goal_dif'])
df.loc[:,'France']   = list(leages_by_season.loc[leages_by_season['country_name']=='France','avg_goal_dif'])
df.loc[:,'Italy']   = list(leages_by_season.loc[leages_by_season['country_name']=='Italy','avg_goal_dif'])
df.loc[:,'England']   = list(leages_by_season.loc[leages_by_season['country_name']=='England','avg_goal_dif'])

df.plot(figsize=(12,5),title='Average Goals Difference Home vs Out')


# # Query Run Order
# Now that we are familiar with most of the functionalities being used in a query, it is very important to understand the order that code runs.
# 
# First, order of how we write it (reminder):
# * SELECT
# * FROM
# * JOIN
# * WHERE
# * GROUP BY
# * HAVING
# * ORDER BY
# * LIMIT
# 
# Now, the actul order that things happens.
# First, you can think of this part as creating a new temporal table in the memory:
# * Define which tables to use, and connect them (FROM + JOIN)
# * Keep only the rows that apply to the conditions (WHERE)
# * Group the data by the required level (if need) (GROUP BY)
# * Choose what information you want to have in the new table. It can have just rawdata (if no grouping), or combination of dimensions (from the grouping), and metrics
# Now, you chose that to show from the table:
# * Order the output of the new table (ORDER BY)
# * Add more conditions that would filter the new created table (HAVING) 
# * Limit to number of rows - would cut it according the soring and the having filtering (LIMIT)
# 

# # Sub Queries and Functions 
# 
# Using subqueries is an essential tool in SQL, as it allows manipulating the data in very advanced ways without the need of any external scripts, and especially important when your tables stractured in such a way that you can't be joined directly.
# 
# In our example, I'm trying to join between a table that holds player's basic details (name, height, weight), to a table that holds more attributes. The problem is that while the first table holds one row for each player, the key in the second table is player+season, so if we do a regular join, the result would be a Cartesian product, and each player's basic details would appear as many times as this player appears in the attributes table. The problem with of course is that the average would be skewed towards players that appear many times in the attribute table.
# 
# The solution, is to use a subquery.  We would need to group the attributes table, to a different key - player level only (without season). Of course we would need to decide first how we would want to combine all the attributes to a single row. I used average, but one can also decide on maximum, latest season and etc. 
# Once both tables have the same keys, we can join them together (think of the subquery as any other table, only temporal), knowing that we won't have duplicated rows after the join.
# 
# In addition, you can see here two examples of how to use functions:
# * Conditional function is an important tool for data manipulation. While IF statement is very popular in other languages, SQLite is not supporting it, and it's implemented using CASE + WHEN + ELSE statement. 
# As you can see, based on the input of the data, the query would return different results.
# 
# * ROUND - straight sorward.
# Every SQL languages comes with a lot of usefull functions by default.

# In[ ]:


players_height = pd.read_sql("""SELECT CASE
                                        WHEN ROUND(height)<165 then 165
                                        WHEN ROUND(height)>195 then 195
                                        ELSE ROUND(height)
                                        END AS calc_height, 
                                        COUNT(height) AS distribution, 
                                        (avg(PA_Grouped.avg_overall_rating)) AS avg_overall_rating,
                                        (avg(PA_Grouped.avg_potential)) AS avg_potential,
                                        AVG(weight) AS avg_weight 
                            FROM PLAYER
                            LEFT JOIN (SELECT Player_Attributes.player_api_id, 
                                        avg(Player_Attributes.overall_rating) AS avg_overall_rating,
                                        avg(Player_Attributes.potential) AS avg_potential  
                                        FROM Player_Attributes
                                        GROUP BY Player_Attributes.player_api_id) 
                                        AS PA_Grouped ON PLAYER.player_api_id = PA_Grouped.player_api_id
                            GROUP BY calc_height
                            ORDER BY calc_height
                                ;""", conn)
players_height


# In[ ]:


players_height.plot(x=['calc_height'],y=['avg_overall_rating'],figsize=(12,5),title='Potential vs Height')

