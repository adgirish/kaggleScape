
# coding: utf-8

# In [another notebook](https://www.kaggle.com/dvasyukova/d/kaggle/us-baby-names/persistent-vs-spike-fade-names) I took a look at names that suddenly gain popularity seemingly appearing out of nowhere. I tracked some of them to their likely causes uncovering TV series or movie characters, actors, singers, people from reality shows and sport stars. Then I was interested in finding such influences that made the biggest impact on name popularities.
# 
# ### My approach is as follows:
# 
# - First normalize raw counts within each year and gender so that we look at popularity values like "babies with this name per million born".
# - Then compute year-to-year differences of popularity values.
# - For each name and gender find the maximum popularity gain and the year it happened. Put these on a chart.
# - Google around for possible explanations of points on the chart. I started with the largest gains, then continued with whatever caught my eye on the chart.
# - For a measure of impact strength I use the name's maximum year-to-year popularity gain. This is straightforward and easily quantifiable though it's not a perfect approach for a few reasons. First, it gives an advantage to names that were already on the rise when the impact happened. Second, some of the influences might span a number of consecutive years and are disadvantaged by using only one year. I pool together spelling variants of the same name (like Jamie and Jaime). Please drop me a comment if you think of a better approach to estimating impact strength.
# 
# ### And the winners are:
# 
# 1. The "Linda" song topped the charts in 1947 and gave the name Linda a popularity boost of 23 extra baby girls per thousand.
# 1. Child actress Shirley Temple was Hollywood's number one box-office star from 1935 to 1938. 19 extra baby girls per thousand received this name in 1935 compared to the previous year. The name actually saw a significant popularity boost in 1934 as well and would make number one if we count both years.
# 1. Jaime Sommers - the lead character of "The Bionic Woman" TV series - brought the names Jaime, Jamie and their many variations a boost of 11 extra babies per thousand in 1976.
# 
# The first chart below lists names on their peak years. Where I've found an explanation I've added the name in bold and colored letters, the explanation should be displayed when hovering over the dot. 
# 
# A lot of that chart is still unexplored - feel free to dig around. I'll especially appreciate it if you find the reasons for some big spikes I haven't figured out - like Ashley in 1983, Dewey in 1898, Jason in 1973 or Nicholas in 1978. There's also something interesting in the name Kelly - it appears to have received two distinct impacts.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from bokeh.plotting import figure, output_notebook, show, ColumnDataSource
from bokeh.models import HoverTool, LabelSet, FixedTicker
from bokeh.charts import Bar
from IPython.display import HTML, display

output_notebook()


# ## Prepare the data

# In[ ]:


df = pd.read_csv('../input/NationalNames.csv', index_col='Id')


# Compute name popularity: how many babies per million born a year receive this name?

# In[ ]:


df['Popularity'] = df.Count.values*1000000/df.groupby(['Year','Gender'])['Count'].transform(sum).values


# Compute popularity differences year to year.

# In[ ]:


def popularity_diff(group):
    yd = group.Year.diff().fillna(0)
    popd = group.Popularity.diff()
    popd.loc[yd>1] = group.Popularity.loc[yd>1]
    popd.iloc[0] = group.Popularity.iloc[0] if group.Year.iloc[0] > 1880 else 0
    return popd

df['PopDiff'] = df.groupby(['Name','Gender']).apply(popularity_diff).reset_index().set_index('Id')['Popularity']


# Compute the name's ranks within a year:
# 
# - popularity rank
# - trending rank - rank names based on popularity gain compared to last year

# In[ ]:


df['TrendingRank'] = df.groupby(['Year','Gender'])['PopDiff'].rank(ascending=False)
df['PopularityRank'] = df.groupby(['Year','Gender'])['Popularity'].rank(ascending=False)


# Compute some statistics on names. Sort names by their max popularity difference.

# In[ ]:


gr = df.groupby(['Name','Gender'])
names = gr.agg({'PopDiff':{'MaxPopDiff':'max',
                           'MinPopDiff':'min'},
                'Count':{'TotalCount':'sum'},
                'Year':{'FirstYear':'min',
                        'LastYear':'max',
                        'NYears':'count'},
                'Popularity':{'MaxPopularity':'max'}})
names.columns = names.columns.droplevel(0)
def bestyear(group, col):
    years, counts = group['Year'].values, group[col].values
    ind = np.argmax(counts)
    return years[ind]
names['BestYearPop'] = gr.apply(bestyear,'Popularity')
names['BestYearPopDiff'] = gr.apply(bestyear,'PopDiff')
names = names.sort_values(by='MaxPopDiff',ascending=False)


# Select top 300 names for plotting. I included the explanations I googled here so that they show up on the chart.

# In[ ]:


data = names.head(300).reset_index()
data['Rank'] = data.groupby('BestYearPopDiff')['MaxPopDiff'].rank(ascending=False)
data['size'] = 6 + np.log(data.MaxPopDiff)
data['alpha'] = np.clip(0.1+data.MaxPopDiff/data.MaxPopDiff.max(),0,1)
data['color'] = 'blue'
data.loc[data.Gender=='F','color']='red'
data['text_color']='#555555'
data['cause'] = ''
data.loc[data.Name=='Linda','cause'] = '"Linda" is a popular song written about then one year old future star Linda McCartney. It was written by Jack Lawrence, and published in 1946.'
data.loc[data.Name=='Shirley','cause'] = 'Shirley Temple was a child actress wildly popular since 1935 for films Bright Eyes, Curly Top and Heidi'
data.loc[data.Name.isin(['Michelle','Michele']),'cause'] = '"Michelle" is a love ballad by the Beatles. It is featured on their Rubber Soul album, released in December 1965. "Michelle" won the Grammy Award for Song of the Year in 1967 and has since become one of the best known and often recorded of all Beatles songs.'
data.loc[data.Name=='Amanda','cause']='"Amanda" is a 1973 song written by Bob McDill and recorded by both Don Williams (1973) and Waylon Jennings (1974). In April 1979 the song was issued as a single, and it soon became one of the biggest country hits of 1979.'
data.loc[data.Name.isin(['Jaime','Jamie']),'cause']='Jaime Sommers is an 1970s iconic television leading female science fiction action character who takes on special high-risk government missions using her superhuman bionic powers in the American television series The Bionic Woman (1976–1978).'
data.loc[data.Name.isin(['Katina','Catina']),'cause']='In 1972 the name Katina was used for a newborn baby on the soap opera "Where the Heart Is"'
data.loc[data.Name.isin(['Judy','Judith']),'cause']='Judy Garland stars as Dorothy in the Wizard of Oz movie (1939)'
data.loc[data.Name=='Whitney','cause']='The singer Whitney Houston was No. 1 artist of the year and her album was the No. 1 album of the year on the 1986 Billboard year-end charts.'
data.loc[data.Name=='Ashanti','cause']='In April 2002 the singer Ashanti released her eponymous debut album, which featured the hit song "Foolish", and sold over 503,000 copies in its first week of release throughout the U.S.'
data.loc[data.Name=='Woodrow','cause']='Woodrow Wilson ran for president of the USA in 1912.'
data.loc[data.Name=='Jacqueline','cause']='Jacqueline Kennedy becomes First Lady'
data.loc[data.cause!='','text_color'] = data['color']


# In[ ]:


source_noexpl = ColumnDataSource(data=data.loc[data.cause==''])
source_expl = ColumnDataSource(data=data.loc[data.cause!=''])

hover = HoverTool(
        tooltips="""
        <div>
            <div>
                <span style="font-size: 17px; font-weight: bold;">@Name</span>
                <span style="font-size: 15px; color: #966;">@BestYearPopDiff</span>
            </div>
            <div style='max-width: 300px'>
                <span style="font-size: 15px;">@cause</span>
            </div>
        </div>
        """
    )

p = figure(plot_width=800, plot_height=2000, tools=[hover,'pan'],
           title="Top {} trending names from {} to {}".format(data.shape[0],df.Year.min()+1, df.Year.max()))

p.circle('Rank', 'BestYearPopDiff', size='size', color='color',source=source_noexpl, alpha='alpha')
p.circle('Rank', 'BestYearPopDiff', size='size', color='color',source=source_expl, alpha='alpha')

labels_noexpl = LabelSet(x="Rank", y="BestYearPopDiff", text="Name", x_offset=8., y_offset=-7.,
                  text_font_size="10pt", text_color="text_color", text_font_style='normal',
                  source=source_noexpl, text_align='left')
labels_expl = LabelSet(x="Rank", y="BestYearPopDiff", text="Name", x_offset=8., y_offset=-7.,
                  text_font_size="10pt", text_color="text_color", text_font_style='bold',
                  source=source_expl, text_align='left')
p.add_layout(labels_noexpl)
p.add_layout(labels_expl)
p.yaxis[0].ticker=FixedTicker(ticks=np.arange(1880,2015,5))
show(p)


# In[ ]:


def report_name(name, gender):
    stats = names.loc[(name,gender)]
    html = """
    <p> {boygirl} name <strong>{name}</strong> has been in use for 
    {NYears:.0f} years from {FirstYear:.0f} to {LastYear:.0f}.</p>
    <p> It was most popular in {BestYearPop:.0f} when {MaxPopularity:.0f} babies in a million were named {name}.</p>
    <p> Its largest popularity raise was in <strong>{BestYearPopDiff:.0f}</strong> 
    with {MaxPopDiff:.0f} babies per million more named {name} than in previous year.</p>
    """.format(**{'boygirl':'Boy' if gender=='M' else 'Girl', 
                'name':name},
              **stats)
    display(HTML(html))
    data = df.loc[(df.Name==name)&(df.Gender==gender)]
    fig, ax = plt.subplots(2,1,figsize=(12,6),sharex=True,
                           gridspec_kw = {'height_ratios':[3, 1]})
    ax[0].bar(data.Year, data['Popularity'], width = 1, alpha=0.6,
           color = 'r' if gender=='F' else 'b')
    ax[0].set_ylabel('Babies per million')
    ax[1].semilogy(data.Year+0.5, data.TrendingRank,label='Trending rank')
    ax[1].semilogy(data.Year+0.5, data.PopularityRank,label='Popularity rank')
    ax[1].set_ylim(0.5,1100)
    ax[1].invert_yaxis()
    ax[1].set_yticklabels([str(int(x)) for x in ax[1].get_yticks()]);
    ax[1].legend()
    fig.suptitle(name,fontsize='large')
    return ax


# In[ ]:


impacts = []


# ## Linda
# 
# > "[Linda](https://en.wikipedia.org/wiki/Linda_%281946_song%29)" is a popular song written about then one year old future star Linda McCartney. It was written by Jack Lawrence, and published in 1946.
# 
# > The recording by Ray Noble and Buddy Clark was recorded on November 15, 1946 and released by Columbia Records. It first reached the Billboard magazine charts on March 21, 1947 and lasted thirteen weeks on the chart, peaking at number one.
# 
# > The recording by Charlie Spivak was released by RCA Victor Records. It first reached the Billboard magazine charts on March 28, 1947, and lasted nine weeks on the chart, peaking at number six.
# 
# The name Linda had been gaining popularity since 1935 and was number one trending name since 1940. This process seemed to be peaking out when the song arrived and gave the name a huge popularity boost.
# 
# Spelling variant Lynda has similar dynamics but is about 20 times less popular. Linda as a boy name also received a boost as well as somewhat similar names Glinda, Arlinda and Jolinda.

# In[ ]:


ax = report_name('Linda','F')
ax[0].set_xlim(1930,1980);
ax[0].axvspan(1947+3/12,1947+3/12+13/52, alpha = 0.5, label='Linda song on the charts')
ax[0].legend(fontsize='large');


# In[ ]:


impacts.append({'Cause':'"Linda" song',
                'Year':1947,
                'Names':'Linda, Lynda',
                'PopularityGain':names.loc[[('Linda','F'),
                                            ('Lynda','F'),
                                            ('Linda','M')],'MaxPopDiff'].sum()})


# ## Shirley
# 
# > [Shirley Temple](https://en.wikipedia.org/wiki/Shirley_Temple) was an American film and television actress ... most famous as Hollywood's number one box-office star from 1935 to 1938.
# 
# > Temple began her film career in 1932 at three years old. In 1934, she found international fame in "Bright Eyes", a feature film designed specifically for her talents, and film hits such as "Curly Top" and "Heidi" followed year after year during the mid-to-late 1930s.
# 
# > Licensed merchandise that capitalized on her wholesome image included dolls, dishes, and clothing.
# 
# Less popular name variants that also peaked in 1935 include Shirlee, Shirlie and also the rare Shirle, Shirla, Sherlie and Shirli.
# 
# The gains of 1934 should probably be counted here as well, but I'll stick to one year for now.

# In[ ]:


ax = report_name('Shirley','F')
ax[0].set_xlim(1910,1970)


# In[ ]:


impacts.append({'Cause':'Shirley Temple, child actress',
                'Year':1935,
                'Names':'Shirley, Shirlee, Shrilie',
                'PopularityGain':names.loc[[('Shirley','F'),
                                            ('Shirlee','F'),
                                            ('Shirlie','F'),
                                            ('Shirley','M')],'MaxPopDiff'].sum()})


# ## Michelle
# 
# > "[Michelle](https://en.wikipedia.org/wiki/Michelle_%28song%29)" is a love ballad by the Beatles, composed principally by Paul McCartney, with the middle eight co-written with John Lennon. It is featured on their Rubber Soul album, released in December 1965. The song is unique among Beatles recordings in that some of its lyrics are in French. "Michelle" won the Grammy Award for Song of the Year in 1967 and has since become one of the best known and often recorded of all Beatles songs.
# 
# Spelling variant Michele was also popular and has similar dynamics.

# In[ ]:


ax = report_name('Michelle','F')
ax[0].set_xlim(1940,2015)


# In[ ]:


impacts.append({'Cause':'"Michelle", Beatles song',
                'Year':1966,
                'Names':'Michelle, Michele',
                'PopularityGain':names.loc[[('Michelle','F'),
                                            ('Michele','F')],'MaxPopDiff'].sum()})


# ## Amanda
# 
# > "[Amanda](https://en.wikipedia.org/wiki/Amanda_%28Don_Williams_song%29)" is a 1973 song written by Bob McDill and recorded by both Don Williams (1973) and Waylon Jennings (1974). "Amanda" was Waylon Jennings's eighth solo number one on the country chart. The single stayed at number one for three weeks on the Billboard Hot Country Singles chart.
# 
# > As recorded by Jennings, "Amanda" had been a track on his 1974 album The Ramblin' Man, but was not released as a single at that time; two other tracks, "I'm a Ramblin' Man" and "Rainy Day Woman," were. More than 4½ years later, new overdubs were added to the original track and placed on his first greatest hits album. In April 1979 the song was issued as a single, and it soon became one of the biggest country hits of 1979.
# 
# Amanda was number one trending name in 1975 and 1979, probably due to recordings by different artists. I'll count the 1979 part as it is bigger.

# In[ ]:


ax = report_name('Amanda','F')
ax[0].set_xlim(1960,2015);


# In[ ]:


impacts.append({'Cause':'"Amanda" song',
                'Year':1979,
                'Names':'Amanda',
                'PopularityGain':names.loc[[('Amanda','F')],'MaxPopDiff'].sum()})


# ## Jaime
# 
# > [Jaime Sommers](https://en.wikipedia.org/wiki/Jaime_Sommers_%28The_Bionic_Woman%29) is an 1970s iconic television leading female science fiction action character - portrayed by American Emmy Award-winning actress Lindsay Wagner - who takes on special high-risk government missions using her superhuman bionic powers in the American television series The Bionic Woman (1976–1978). 
# 
# > The character of Jaime Sommers became a pop culture icon of the 1970s. In 2004, the Jaime Sommers character was listed in Bravo's 100 Greatest TV Characters. AOL named her one of the 100 Most Memorable Female TV Characters.
# 
# Name Jaime along with a more popular spelling variant Jamie had a huge popularity boost in 1976 when the TV series started. There is also a huge number of similar names that also peaked that year: Jami, Jaimie, Jayme, Jaimee, Jamey, Jaymie, Jaimi, Jamy, Jamye, Jaimy.
# 
# Names Jaime and Jamie for boys didn't react to this significantly.

# In[ ]:


ax=report_name('Jaime','F')
ax[0].set_xlim(1940,2015);


# In[ ]:


ax=report_name('Jamie','F')
ax[0].set_xlim(1940,2015);


# In[ ]:


impacts.append({'Cause':'"Bionic Woman" TV series',
                'Year':1976,
                'Names':'Jaime, Jamie',
                'PopularityGain':names.loc[[('Jaime','F'),
                                            ('Jamie','F'),
                                            ('Jami','F'),('Jaimie','F'),('Jayme','F'),('Jaimee','F'),
                                            ('Jamey','F'),('Jaymie','F'),('Jaimi','F'),('Jamy','F'),
                                            ('Jamye','F'),('Jaimy','F'),],'MaxPopDiff'].sum()})


# ## Judith, Judy
# 
# > [The Wizard of Oz](https://en.wikipedia.org/wiki/The_Wizard_of_Oz_%281939_film%29) is a 1939 American musical comedy-drama fantasy film produced by Metro-Goldwyn-Mayer, and the most well-known and commercially successful adaptation based on the 1900 novel The Wonderful Wizard of Oz by L. Frank Baum. The film stars [Judy Garland](https://en.wikipedia.org/wiki/Judy_Garland) as Dorothy Gale.
# 
# Judy Garland's popularity probably affected baby names for more than this one year, but I'll again only count 1939 for simplicity.

# In[ ]:


ax=report_name('Judith','F')
ax[0].set_xlim(1920,1980);


# In[ ]:


ax=report_name('Judy','F')
ax[0].set_xlim(1920,1980);


# In[ ]:


impacts.append({'Cause':'Judy Garland in "Wizard of Oz"',
                'Year':1939,
                'Names':'Judith, Judy',
                'PopularityGain':names.loc[[('Judith','F'),
                                            ('Judy','F'),
                                            ('Judie','F')],'MaxPopDiff'].sum()})


# ## Katina
# 
# One of the [spike-fade names](https://www.kaggle.com/dvasyukova/d/kaggle/us-baby-names/persistent-vs-spike-fade-names) that was popular enough to make it to this chart.
# 
# In 1972 the name Katina is given to a newborn baby on the soap opera "[Where the Heart Is](https://en.wikipedia.org/wiki/Where_the_Heart_Is_%28US_TV_series%29)".

# In[ ]:


ax=report_name('Katina','F')
ax[0].set_xlim(1960,2015);


# In[ ]:


impacts.append({'Cause':'Baby on "Where the Heart Is" soap opera',
                'Year':1972,
                'Names':'Katina, Catina',
                'PopularityGain':names.loc[[('Katina','F'),
                                            ('Catina','F')],'MaxPopDiff'].sum()})


# ## Whitney
# 
# > [Whitney Houston](https://en.wikipedia.org/wiki/Whitney_Houston) releases her debut album "Whitney Houston" in February 1985.  Houston was No. 1 artist of the year and Whitney Houston was the No. 1 album of the year on the 1986 Billboard year-end charts.

# In[ ]:


ax=report_name('Whitney','F')
ax[0].set_xlim(1960,2015);


# In[ ]:


impacts.append({'Cause':'Whitney Houston, singer',
                'Year':1986,
                'Names':'Whitney',
                'PopularityGain':names.loc[[('Whitney','F')],'MaxPopDiff'].sum()})


# ## Ashanti
# 
# > Ashanti Shequoiya Douglas (born October 13, 1980), known simply as [Ashanti](https://en.wikipedia.org/wiki/Ashanti_%28singer%29), is an American singer, songwriter, record producer, dancer and actress. Ashanti is known for her eponymous debut album, which featured the hit song "Foolish", and sold over 503,000 copies in its first week of release throughout the U.S. in April 2002. In 2003, the self-titled debut album won Ashanti her first Grammy Award for Best Contemporary R&B album. 

# In[ ]:


ax=report_name('Ashanti','F')
ax[0].set_xlim(1980,2015);


# In[ ]:


impacts.append({'Cause':'Ashanti, singer',
                'Year':1986,
                'Names':'Ashanti',
                'PopularityGain':names.loc[[('Ashanti','F'),('Ashanty','F')],'MaxPopDiff'].sum()})


# ## Jacqueline
# 
# Jacqueline Kennedy becomes First Lady in 1961.

# In[ ]:


ax=report_name('Jacqueline','F')
ax[0].set_xlim(1920,2015);


# In[ ]:


ax=report_name('Jackie','F')
ax[0].set_xlim(1920,2015);


# In[ ]:


impacts.append({'Cause':'Jacqueline Kennedy, First Lady',
                'Year':1961,
                'Names':'Jacqueline, Jackie',
                'PopularityGain':names.loc[[('Jacqueline','F'),('Jackie','F'),('Jacquelyn','F'),
                                            ('Jacquline','F'),('Jacquelin','F'),('Jackqueline','F')],'MaxPopDiff'].sum()})


# ## Woodrow
# 
# Woodrow Wilson ran for president in 1912 and has been in office from 1913 to 1921.
# 
# It looks like names Woodrow and Wilson both had increased popularity during his presidency term.

# In[ ]:


ax=report_name('Woodrow','M')
ax[0].set_xlim(1900,1960);


# In[ ]:


ax=report_name('Wilson','M')
ax[0].set_xlim(1900,1960);


# In[ ]:


impacts.append({'Cause':'Woodrow Wilson running for president',
                'Year':1912,
                'Names':'Woodrow, Wilson',
                'PopularityGain':names.loc[[('Woodrow','M'),('Wilson','M'),('Woodroe','M'),
                                            ('Woodrow','F')],'MaxPopDiff'].sum()})


# # And the winners are...

# In[ ]:


pd.DataFrame(impacts).sort_values(by='PopularityGain',ascending=False).head(3)


# In[ ]:


res = pd.DataFrame(impacts).sort_values(by='PopularityGain')
fig, ax = plt.subplots(figsize=(10,6))
h = np.arange(res.shape[0])
ax.barh(h,res.PopularityGain)
ax.set_yticks(h+0.4)
ax.set_yticklabels(res.Cause.str.cat(res.Year.astype(str),sep=' '))
for (y,n) in zip(h, res.Names):
    ax.text(300,y+0.4, n, verticalalignment='center',color='white')
ax.set_xlabel('Popularity gain, babies per million');
ax.set_ylim(0,h.max()+1);

