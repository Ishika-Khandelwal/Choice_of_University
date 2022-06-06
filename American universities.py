import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#to center every figure in the notebook.
#from: https://stackoverflow.com/questions/18380168/center-output-plots-in-the-notebook
from IPython.core.display import HTML as Center

Center(""" <style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style> """)

with open('universities_data.csv') as file:
    universities_df=pd.read_csv(file)

    type(universities_df)

    universities_df.head()

    universities_df.shape

    print('The dataset contains {} rows and {} columns'.format(universities_df.shape[0],universities_df.shape[1]))

    universities_df.info(max_cols=len(universities_df))

    universities_df.isna().sum().sort_values(ascending=False)

    perc_nan=universities_df.isna().sum()/len(universities_df)*100

    ax=perc_nan[perc_nan>=20].sort_values(ascending=False).plot.bar(title='Percentage of NaN values',figsize=(12,5));
ax.set_ylabel('% of NaN elements');

colum_off=universities_df.isna().sum()[universities_df.isna().sum()>=(0.2*len(universities_df))]
list_colum_off=colum_off.index.to_list()
universitiesnw_df=universities_df.copy()
universitiesnw_df.drop(list_colum_off,axis=1,inplace=True)

interesting_columns=['Name', 'year', 'Highest degree offered', "Offers Bachelor's degree",
       "Offers Master's degree",
       "Offers Doctor's degree - research/scholarship",
       "Offers Doctor's degree - professional practice", 'Applicants total',
       'Admissions total', 'Enrolled total', 'Estimated enrollment, total',
       'Tuition and fees, 2013-14',
       'Total price for in-state students living on campus 2013-14',
       'Total price for out-of-state students living on campus 2013-14',
       'State abbreviation', 'Control of institution', 'Total enrollment',
       'Full-time enrollment', 'Part-time enrollment',
       'Undergraduate enrollment', 'Graduate enrollment',
       'Full-time undergraduate enrollment',
       'Part-time undergraduate enrollment',
       'Percent of total enrollment that are women',
       'Percent of undergraduate enrollment that are women',
       'Percent of graduate enrollment that are women',
       'Graduation rate - Bachelor degree within 4 years, total',
       'Graduation rate - Bachelor degree within 5 years, total',
       'Graduation rate - Bachelor degree within 6 years, total',
       ]

universitiesnw_df=universitiesnw_df[interesting_columns]

universitiesnw_df[universitiesnw_df['Total enrollment'].isna()][['Name','Applicants total','Admissions total','Enrolled total','Total enrollment']]

a=universitiesnw_df[universitiesnw_df['Name']=='University of North Georgia'].index[0]
b=universitiesnw_df[universitiesnw_df['Name']=='Texas A & M University-Galveston'].index[0]
universitiesnw_df=universitiesnw_df.drop([a,b],axis=0)
print('The data frame now has {} columns out of the {} original columns, and {} rows out of the {} original rows.'.format(universitiesnw_df.shape[1],universities_df.shape[1],universitiesnw_df.shape[0],universities_df.shape[0]))
col=universitiesnw_df.select_dtypes(include=['float64','int64']).columns

lt=list()
for i in col:
    y=any(x < 0 for x in universitiesnw_df[i])
    if y==True:
        lt.append(y)
print('There are {} negative values in the data frame.'.format(len(lt)))

universitiesnw_df.describe()

total_zero=(universitiesnw_df[universitiesnw_df.loc[0:]==0]).count().sum()
print('This data set contains {} zero values.'.format(total_zero))

universitiesnw_df.replace(0,np.nan,inplace=True)
total_zero_nw=universitiesnw_df[universitiesnw_df.loc[0:]==0].count().sum()
print('This data set contains {} zero values.'.format(total_zero_nw))
universitiesnw_df[['Name','Applicants total']].sort_values('Applicants total').head()

def remove_space(header):
    
    '''This function takes all the spaces between the words of column names and replaces them
    with '_' . 
    
    The argument header corresponds to a column name. '''
    
    list1=list()
    words_header=header.split()    
    size=int(len(words_header))
    
    for i in range(len(words_header)):
        if i<size-1:
            list1.append(words_header[i]+'_')            
        else:
            list1.append(words_header[i])
                
    separator = ''
    final=separator.join(list1)    
    
    return final

def remove_sp_char(header):
    
    '''This function takes all the special characters found in column names and replaces them
    with other character accordingly to the case. 
    
    The argument header corresponds to a column name. '''
    
    if "'" in header:
        header=header.replace("'",'')
    if "," in header:
        header=header.replace(",",'')
    if "_-_" in header:
        header=header.replace("_-_",'_')
    if "/" in header:
        header=header.replace("/",'_or_')
    if ":" in header:
        header=header.replace(":",'')
    if "-" in header:
        header=header.replace("-",'_')
        
    return header

list_new_head=list()
headers=universitiesnw_df.columns

for header in headers:
    header1=remove_space(header) # Spaces are replaced.
    header1=header1.casefold()   # All capitalized letters are changed.
    header1=remove_sp_char(header1) # Special characters are replaced.
      
    if "degrese" in header1:    # One column name has a typo.  
        header1=header1.replace("degrese",'degrees')
            
    list_new_head.append(header1)

universitiesnw_df.columns=list_new_head
universitiesnw_df.rename(columns={'state_abbreviation':'state'}, inplace=True)
universitiesnw_df[['state']].head(2)

#matplotlib

matplotlib.rcParams['figure.facecolor']='whitesmoke'
from IPython.display import display
with pd.option_context('display.max_columns',None):
    display(universitiesnw_df.describe())

high_app_df=universitiesnw_df[['name','applicants_total']].sort_values('applicants_total',ascending=False).head(20)
high_app_df;
plt.figure(figsize=(12,8))
matplotlib.rcParams['font.size']=14
sns.barplot(x='applicants_total',y='name',data=high_app_df)
plt.title('Top 20 American Universities with the Most Applications in 2013')
plt.xlabel('Number of applications')
plt.ylabel('');

plt.figure(figsize=(16,6))

plt.subplot(1,3,1)
sns.histplot(universitiesnw_df.applicants_total,bins=50)
plt.title('''Histogram of Number of Applications. 
Mean: {:.1f}, Median: {:.1f}'''.format(universitiesnw_df.applicants_total.mean(),universitiesnw_df.applicants_total.median()));
plt.xlabel('Number of Applications')
plt.axis([0,30000,0,350])
plt.xticks(rotation=10)
plt.grid()

plt.subplot(1,3,2)
sns.histplot(universitiesnw_df.admissions_total,bins=50)
plt.title('''Histogram of Number of Admissions. 
Mean: {:.1f}, Median: {:.1f}'''.format(universitiesnw_df.admissions_total.mean(),universitiesnw_df.admissions_total.median()));
plt.axis([0,10000,0,350])
plt.xlabel('Number of Admissions')
plt.xticks(rotation=10)
plt.grid()

plt.subplot(1,3,3)
sns.histplot(universitiesnw_df.enrolled_total,bins=50)
plt.title('''Histogram of Number of Enrollments. 
Mean: {:.1f}, Median: {:.1f}'''.format(universitiesnw_df.enrolled_total.mean(),universitiesnw_df.enrolled_total.median()));
plt.axis([0,5000,0,350])
plt.xlabel('Number of Enrollments')
plt.grid()
plt.xticks(rotation=10)
plt.tight_layout(pad=2);


plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.title('APPLICATIONS VS ADMISSIONS')
sns.scatterplot(y=universitiesnw_df.admissions_total,x=universitiesnw_df.applicants_total,hue=universitiesnw_df.control_of_institution)
plt.ylabel('Number of Admissions')
plt.xlabel('Number of Applications')
plt.grid()

plt.subplot(1,2,2)
plt.title('ADMISSIONS VS ENROLLMENTS')
sns.scatterplot(x='admissions_total',y='enrolled_total',data=universitiesnw_df,hue='control_of_institution')
plt.ylabel('Number of Enrollments')
plt.xlabel('Number of Admissions')
plt.grid()

plt.tight_layout(pad=2)

universitiesnw_df['acceptance_rate']=(universitiesnw_df.admissions_total/universitiesnw_df.applicants_total*100).round(2)
universitiesnw_df['enrollment_rate']=(universitiesnw_df.enrolled_total/universitiesnw_df.admissions_total*100).round(2)
plt.figure(figsize=(12,5))
sns.scatterplot(x='applicants_total',y='enrollment_rate',data=universitiesnw_df)
plt.title('APPLICATIONS VS ENROLLMENT RATE')
plt.ylabel('Enrollment Rate %')
plt.xlabel('Number of Applications');
plt.figure(figsize=(16,4))

plt.subplot(1,2,1)
ind = np.arange(len(high_acceptance)) #number of universities
width = 0.35       #space

plt.bar(ind, high_acceptance.acceptance_rate, width, label='Acceptance Rate')
plt.bar(ind + width, high_acceptance.enrollment_rate, width,label='Enrollment Rate')
plt.title('''Acceptance and Enrollment Rates.
25 Universities With the Highest Acceptance Rate ''')
plt.ylabel('Rates %')
plt.xticks(ind + width,high_acceptance.name.values,rotation=90 )
plt.legend(loc='best');

plt.subplot(1,2,2)
ind = np.arange(len(low_acceptance)) #number of universities
width = 0.35       #space

plt.bar(ind, low_acceptance.acceptance_rate, width, label='Acceptance Rate')
plt.bar(ind + width, low_acceptance.enrollment_rate, width,label='Enrollment Rate')
plt.title('''Acceptance and Enrollment Rates.
25 Universities With the Lowest Acceptance Rate ''')
plt.ylabel('Rates %')
plt.xticks(ind + width,high_acceptance.name.values,rotation=90 )
plt.legend(loc='best');

#spliting the number of applications according to the type of control: private or public.
uni_private_df=universitiesnw_df[universitiesnw_df.control_of_institution=='Private not-for-profit']
uni_private_df=uni_private_df[uni_private_df.applicants_total.notnull()]

uni_public_df=universitiesnw_df[universitiesnw_df.control_of_institution=='Public']
uni_public_df=uni_public_df[uni_public_df.applicants_total.notnull()]

plt.figure(figsize=(16,7))

plt.subplot(1,2,1)
plt.hist([uni_public_df.applicants_total,uni_private_df.applicants_total],stacked=True,bins=25)
plt.axis([0,31000,0,700])
plt.title('Distribution of Applications')
plt.xlabel('Number of Applications')
plt.ylabel('Universities')
plt.legend(['Public universities. ({})'.format(len(uni_public_df)),'Private universities. ({})'.format(len(uni_private_df))]);

plt.subplot(1,2,2)
sns.barplot(x=universitiesnw_df.control_of_institution,y=universitiesnw_df.applicants_total);
plt.title('''Average and Variation of Applications 
According to the Type of Control''')
plt.xlabel('')
plt.ylabel('Number of Applications');
plt.tight_layout(pad=1)

print('The minimum number of applications for private universities was {}; whereas, for public universities was {}.'.format(int(uni_private_df.applicants_total.min()),int(uni_public_df.applicants_total.min())))
print('The maximum number of applications for private universities was {}; whereas, for public universities was {}.'.format(int(uni_private_df.applicants_total.max()), int(uni_public_df.applicants_total.max())))
g=sns.jointplot(x=universitiesnw_df.enrollment_rate,y=universitiesnw_df.applicants_total,hue=universitiesnw_df.control_of_institution,height=6);

#
plt.figure(figsize=(16,6))
sns.scatterplot(x='acceptance_rate',y='enrollment_rate',data=universitiesnw_df,hue=universitiesnw_df.control_of_institution)
plt.title('ACCEPTANCE VS ENROLLMENT RATES')
plt.ylabel('Enrollment Rate %')
plt.xlabel('Acceptance Rate %');

high_acceptance=universitiesnw_df[universitiesnw_df.acceptance_rate.notnull()][['name','acceptance_rate','enrollment_rate']].sort_values('acceptance_rate',ascending=False).head(25)
low_acceptance=universitiesnw_df[universitiesnw_df.acceptance_rate.notnull()][['name','acceptance_rate','enrollment_rate']].sort_values('acceptance_rate',ascending=False).tail(25)

plt.figure(figsize=(16,4))

plt.subplot(1,2,1)
ind = np.arange(len(high_acceptance)) #number of universities
width = 0.35       #space

plt.bar(ind, high_acceptance.acceptance_rate, width, label='Acceptance Rate')
plt.bar(ind + width, high_acceptance.enrollment_rate, width,label='Enrollment Rate')
plt.title('''Acceptance and Enrollment Rates.
25 Universities With the Highest Acceptance Rate ''')
plt.ylabel('Rates %')
plt.xticks(ind + width,high_acceptance.name.values,rotation=90 )
plt.legend(loc='best');

plt.subplot(1,2,2)
ind = np.arange(len(low_acceptance)) #number of universities
width = 0.35       #space

plt.bar(ind, low_acceptance.acceptance_rate, width, label='Acceptance Rate')
plt.bar(ind + width, low_acceptance.enrollment_rate, width,label='Enrollment Rate')
plt.title('''Acceptance and Enrollment Rates.
25 Universities With the Lowest Acceptance Rate ''')
plt.ylabel('Rates %')
plt.xticks(ind + width,high_acceptance.name.values,rotation=90 )
plt.legend(loc='best');

#spliting the number of applications according to the type of control: private or public.
uni_private_df=universitiesnw_df[universitiesnw_df.control_of_institution=='Private not-for-profit']
uni_private_df=uni_private_df[uni_private_df.applicants_total.notnull()]

uni_public_df=universitiesnw_df[universitiesnw_df.control_of_institution=='Public']
uni_public_df=uni_public_df[uni_public_df.applicants_total.notnull()]

plt.figure(figsize=(16,7))

plt.subplot(1,2,1)
plt.hist([uni_public_df.applicants_total,uni_private_df.applicants_total],stacked=True,bins=25)
plt.axis([0,31000,0,700])
plt.title('Distribution of Applications')
plt.xlabel('Number of Applications')
plt.ylabel('Universities')
plt.legend(['Public universities. ({})'.format(len(uni_public_df)),'Private universities. ({})'.format(len(uni_private_df))]);

plt.subplot(1,2,2)
sns.barplot(x=universitiesnw_df.control_of_institution,y=universitiesnw_df.applicants_total);
plt.title('''Average and Variation of Applications 
According to the Type of Control''')
plt.xlabel('')
plt.ylabel('Number of Applications');
plt.tight_layout(pad=1)

print('The minimum number of applications for private universities was {}; whereas, for public universities was {}.'.format(int(uni_private_df.applicants_total.min()),int(uni_public_df.applicants_total.min())))
print('The maximum number of applications for private universities was {}; whereas, for public universities was {}.'.format(int(uni_private_df.applicants_total.max()), int(uni_public_df.applicants_total.max())))
g=(g.set_axis_labels("Enrollment Rate %","Applications"));

#Do students prefer universities with low tuition and fees?
g=sns.jointplot(x=universitiesnw_df.tuition_and_fees_2013_14,y=universitiesnw_df.applicants_total,hue=universitiesnw_df.control_of_institution,height=6);
g=(g.set_axis_labels("Tuition and Fees $","Applications"))
g=sns.jointplot(x=universitiesnw_df.tuition_and_fees_2013_14,y=universitiesnw_df.enrollment_rate,hue=universitiesnw_df.control_of_institution,height=9);
g=(g.set_axis_labels('Tuition and Fees $','Enrollment rate'))

#Do students prefer a university for its low cost of on-campus living?
plt.figure(figsize=(16,7))
plt.subplot(1,2,1)
sns.barplot(y=universitiesnw_df.total_price_for_in_state_students_living_on_campus_2013_14,x=universitiesnw_df.control_of_institution)
plt.title('''Average and variation of the Cost for 
In-State Students Living on Campus (2013-2014)''')        
plt.xlabel('')
plt.ylabel('Cost of living on campus $')


plt.subplot(1,2,2)
sns.scatterplot(x=universitiesnw_df.total_price_for_in_state_students_living_on_campus_2013_14,y=universitiesnw_df.enrollment_rate,hue=universitiesnw_df.control_of_institution);
plt.title('''Cost for In-State Students Living 
on Campus vs Enrollment Rate (2013-2014)''')
plt.xlabel('Cost of living on campus $')
plt.ylabel('Enrollment Rate')


plt.tight_layout(pad=2)
plt.figure(figsize=(16,7))
plt.subplot(1,2,2)
sns.scatterplot(x=universitiesnw_df.total_price_for_out_of_state_students_living_on_campus_2013_14,y=universitiesnw_df.enrollment_rate,hue=universitiesnw_df.control_of_institution);
plt.title('''Cost for Out-State Students Living 
on Campus vs Enrollment Rate (2013-2014)''')
plt.xlabel('Cost of living on campus $')
plt.ylabel('Enrollment Rate')

plt.subplot(1,2,1)
sns.barplot(y=universitiesnw_df.total_price_for_out_of_state_students_living_on_campus_2013_14,x=universitiesnw_df.control_of_institution)
plt.title('''Average and variation of the Cost 
for Out-State Students Living on Campus (2013-2014)''')        
plt.xlabel('')
plt.ylabel('Cost of living on campus $')

plt.tight_layout(pad=2)
region=pd.read_csv('region.csv')
universitiesnw_df=universitiesnw_df.merge(region,on='state')
in_state_df=universitiesnw_df[['name','enrollment_rate','total_price_for_in_state_students_living_on_campus_2013_14','control_of_institution','state','region']]
in_state_df=in_state_df.rename(columns={'total_price_for_in_state_students_living_on_campus_2013_14':'price_living'})
in_state_df['from']='In-State'
out_state_df=universitiesnw_df[['name','enrollment_rate','total_price_for_out_of_state_students_living_on_campus_2013_14','control_of_institution','state','region']]
out_state_df=out_state_df.rename(columns={'total_price_for_out_of_state_students_living_on_campus_2013_14':'price_living'})
out_state_df['from']='Out-State'
in_out_state_df=in_state_df.append(out_state_df,ignore_index = True)
plt.figure(figsize=(16,6))

#with standard deviation
plt.subplot(1,2,1)
sns.barplot(x='control_of_institution',y='price_living',data=in_out_state_df,hue='from',ci="sd",palette='hot');
plt.title('''Average and variation of the 
cost of on-campus living (2013-2014)''')
plt.xlabel('')
plt.ylabel('Cost of on-campus living $')

plt.subplot(1,2,2)
sns.scatterplot(x='price_living',y='enrollment_rate',data=in_out_state_df,hue='from',palette='hot')
plt.title('Cost of On-Campus Living vs Enrollment Rate')
plt.xlabel('Cost of On-Campus Living $')
plt.ylabel('Enrollment Rate');

plt.figure(figsize=(16,7))
sns.barplot(x='region',y='price_living',data=in_out_state_df,hue='control_of_institution',ci="sd",palette='Accent');
plt.title('Average and Variation of Cost of On-Campus Living (2013-2014)')
plt.grid(axis='y')
plt.xlabel('')
plt.ylabel('Cost of on-campus living $');

#Do students prefer universities from highly populated states?

plt.figure(figsize=(16,8))
sns.barplot(x=universitiesnw_df.state,y=universitiesnw_df.enrollment_rate);
plt.grid(axis='y')

plt.xticks(rotation=90);with open('states_population.csv') as file:
    
population_df=pd.read_csv(file)

population_df.head()

universitiesnw_df=universitiesnw_df.merge(population_df,on='state')
universitiesnw_df=universitiesnw_df.rename(columns={'population_2013':'population'})

#pattern between state population and enrollment rate.

plt.figure(figsize=(16,8))
ax=sns.scatterplot(y='population',x='enrollment_rate',data=universitiesnw_df,hue='control_of_institution');
plt.title('Population vs Enrollment Rate')
plt.grid(axis='y')
ax.ticklabel_format(style='plain')
plt.ylabel('Population')
plt.xlabel('Enrollment Rate %');

#Do students prefer a university because it belongs to a state with a high GDP per capita?
with open('states_gdp.csv') as file:
    gdp_df=pd.read_csv(file)
    gdp_df.head()

gdp_df.drop(columns=['code'],inplace=True)
universitiesnw_df=universitiesnw_df.merge(gdp_df,on='state')
universitiesnw_df[['state','gdp_million','population']].head()
universitiesnw_df['gdp_capita']=universitiesnw_df.gdp_million/universitiesnw_df.population*1e6
gdp_state_df=universitiesnw_df.groupby('state')[['region','gdp_capita']].mean().sort_values('gdp_capita',ascending=False)
gdp_f25=gdp_state_df.head(25)
gdp_l25=gdp_state_df.tail(25)

plt.figure(figsize=(16,8))
ax=sns.barplot(x=gdp_f25.gdp_capita,y=gdp_f25.index);
ax.set_xlim((0,180000)); plt.title('GDP per Capita of American States'); plt.xlabel('GDP per Capita $');
plt.grid(axis='x',alpha=0.75)

plt.figure(figsize=(16,8))
ax=sns.barplot(x=gdp_l25.gdp_capita,y=gdp_f25.index);
ax.set_xlim((0,180000)); plt.xlabel('GDP per Capita $'); plt.ylabel(''), plt.grid(axis='x',alpha=0.75);
gdp_f25.head(2)

plt.figure(figsize=(16,7))
sns.scatterplot(x='gdp_capita',y='enrollment_rate',data=universitiesnw_df);
plt.plot([78000,78000], [0, 110], c='magenta',lw=3,marker='*',ls='--')
plt.title('GDP per Capita vs Enrollment Rate')
plt.grid()
plt.xlabel('GDP per Capita $')
plt.ylabel('Enrollment Rate %');

plt.figure(figsize=(16,7))
sns.scatterplot(x='gdp_capita',y='enrollment_rate',data=universitiesnw_df);
plt.axis([30000,80000,0, 101]);
plt.grid();
plt.title('GDP per Capita vs Enrollment Rate')
plt.xlabel('GDP per Capita $')
plt.ylabel('Enrollment Rate %'); sns.despine();

names=universitiesnw_df.columns[universitiesnw_df.columns.str.startswith('offers')].values
degree=universitiesnw_df[universitiesnw_df[names]=='Yes'][names].count().sort_values(ascending=False)

plt.figure(figsize=(10,8))
ax=sns.barplot(x=degree,y=degree.index)
ax.set_yticklabels(("Bacherlor's Degree","Master's Degree",
                    "Doctor's Degree: Research/Scholarship",
                    "Doctor's Degree: Professional Practice"));
plt.title('Degrees Offered')
plt.xlabel('Universities')
plt.grid(axis='x');

#Do students prefer a university based on the possibility of a higher, additional academic degree in the same university?
hg_degree=universitiesnw_df.highest_degree_offered.value_counts()plt.figure(figsize=(16,8))
plt.pie(hg_degree,labels=hg_degree.index,
       autopct='%.1f%%',startangle=140,colors = ['violet','aqua','pink','lightsalmon','moccasin','dodgerblue'])
plt.title('Highest Degree Offered');
plt.figure(figsize=(16,8))
ax=sns.scatterplot(y='highest_degree_offered',x='enrollment_rate',data=universitiesnw_df);
plt.title('Highest Degree Offered vs Enrollment Rate')
plt.ylabel('')
plt.xlabel('Enrollment Rate %')
plt.grid(axis='x')
ax.set_yticklabels(('''Doctor's Degree: 
Research/Scholarship''',
                    '''Doctor's Degree: Research/
Scholarship & Professional 
Practice''',
                    "Bacherlor's Degree",
                    '''Doctor's Degree: 
Professional Practice''',
                    "Master's Degree",
                    "Doctor's Degree: Other"));


###A high number of applications does not imply that a university is preferred among students. In fact, the universities that receive a lower number of applications are the ones with a higher enrollment rate. Obviously, there are some exceptions, but this is the strongest tendency.

#Based on the lack of a strong pattern among admissions and the enrollment rate, we can say that students do not necessarily prefer a university because of its high acceptance rate or, in other words, the students'preference is not based on how easy it is for them to be admitted to a university.

#By analyzing the enrollment rate, we saw that this rate, on average, is higher for public universities than the average for private universities. So, there is a strong students' preference for public universities.

#When it comes to tuition and fees, students prefer affordable universities. Additionally, the reason or one of the reasons for the students' preference for public universities is that public universities are much more affordable than the majority of private universities.

#In all the analyses made to find a pattern about costs for on-campus living, we found a high enrollment rate more frequently when costs are affordable. This means that students, in-state and out-state students, prefer universities with affordable costs of on-campus living.

#The majority of public universities offer a much more affordable price for in-state students than private universities.

#The average cost of living for out-state students that public universities offer is higher than that for in-state students. However, the average cost that private universities offer does not make a distinction between in-state and out-state students.

#Since there was no firm trend when analyzing the state population with enrolment rates, we cannot say that students prefer universities of crowded states.

#Students do not prefer a university because of the GDP per capita of the state where the university locates. In other words, students do not choose a university based on the overall well-being of states.

#When students look for a university to study for a Bachelor's degree, they do not frequently choose the university thinking about a future possibility of pursuing a higher degree at the same university.

#To get more accurate results, it's necessary to have the information of other years, expand the number of universities, and add information about their ranking.





