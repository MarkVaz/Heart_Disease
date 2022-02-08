# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 22:59:25 2022

@author: MC
"""

import streamlit as st


Header = st.container()
Intro = st.container()
Exploration = st.container()
Analysis = st.container()
Conclusion = st.container()

with Header:
    st.title('No One Else')
    st.header('A Data Science project on Heart Disease')
    st.header('By Mark Vaz')
    
    st.markdown('Heart Disease is the **#2** leading cause of death in Canada, 50,000 Canadians each year are diagnosed with heart failure.')
    st.markdown('The indicators shown in this project may seem foreign to many, but **9 in 10** Canadians over the age of 20 have at least one risk factor for heart disease')
    st.markdown('There are many factors that lead towards heart disease, most people do not take the nescessary precautions to ensure their safety.')   
    st.markdown('I have watched two men in my family pass away from Heart related issues, my grandfather 5 years ago from a heart attack and my father who passed at the age of 58 last year.')
    st.markdown('I am not a heart surgeon, so to do my part I hope to be able to update this project in the years to come as I improve in my ML skills')
    
with Intro:
    st.header('Data Exploration')
    st.markdown('This Data was obtained through Kaggle and I take no credit in the work of compiling this Data.') 
    st.markdown('You can find more information about the data set below:')
    st.markdown('https://www.kaggle.com/ronitf/heart-disease-uci',unsafe_allow_html=True)
    
    st.subheader('Attribute Information:')
    st.markdown("- age: The person's age in years")
    st.markdown("- sex: The person's sex (1 = male, 0 = female)")
    st.markdown("- cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)")
    st.markdown("- trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)")
    st.markdown("- chol: The person's cholesterol measurement in mg/dl")
    st.markdown("- fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)")
    st.markdown("- restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)")
    st.markdown("- thalach: The person's maximum heart rate achieved")
    st.markdown("- exang: Exercise induced angina (1 = yes; 0 = no)")
    st.markdown('- oldpeak: ST depression induced by exercise relative to rest')
    st.markdown("- slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)")
    st.markdown("- ca: The number of major vessels (0-3)")
    st.markdown("- thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)")
    st.markdown("- target: Heart disease (0 = no, 1 = yes)")
    
with Exploration:
    st.subheader("Let's Start!")
    st.markdown("Import the nescessary python **libraries**")    
    with st.echo():
        import pandas as pd
        import altair as alt
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_validate, train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        
    st.markdown('Read in the data with **pandas**')
    
    with st.echo():

        df = pd.read_csv('Data/heart.csv')
        
st.markdown("Let's look at our Data Frame")

#Displays df       
st.dataframe(df)

#Displays shape of df
st.text(str(df.shape))

#Streamlit can not show series easily so the following code dispalys what df.index() would show but as a data frame
df_datatypes = pd.DataFrame(df.dtypes).reset_index().rename(columns = {'index':'column_name', 0:'dtype'} )
df_null_count = df.count().to_frame().reset_index().rename(columns ={'index':'column_name',0:'Non-Null_Values'})

df_info =pd.concat([df_datatypes,df_null_count], axis =1)
df_info =df_info.loc[:,~df_info.columns.duplicated()]
df_info =df_info.astype(str)
st.dataframe(df_info)


st.markdown("In this Data Frame we have both continuous value and categorical data columns. Let's take a look first at how male and females are represented.")

with st.echo():
    sex_chart = (alt.Chart(df, width = 400, height =400)
                 .mark_bar(size=80)
                 .encode(alt.X('target:O')
                         ,alt.Y('count():N')
                         ,alt.Color('sex:O'))
                 .properties(title = 'Distribution of Sex'))

st.text('light blue = female')
st.text('dark blue = male')
st.text('target = 0 patient shows no sign of a heart disease')
st.text('target = 1, patient has a heart disease')
    
st.altair_chart(sex_chart)

st.markdown('It appears that there are more men in the study than women, but the ratio of women with heart disease compared to not is much higher than the ratio for men ')

st.markdown("Now let's look at the continuous data")

df_heart_disease = df[df['target']==1]
df_no_disease = df[df['target']==0]

with st.echo():
    age_chart1 = (alt.Chart(df_heart_disease, width = 800, height =400)
                 .mark_bar()
                 .encode(alt.X('age:N')
                         ,alt.Y('count():N')
                         ,alt.Color('target:O'))
                 .properties(title = 'Distribution of Ages with Heart Disease'))
    
    age_chart0 = (alt.Chart(df_no_disease, width = 800, height =400)
                 .mark_bar()
                 .encode(alt.X('age:N')
                         ,alt.Y('count():N')
                         ,alt.Color('target:O'))
                 .properties(title = 'Distribution of Ages with No Disease'))
    
    
    
st.altair_chart(age_chart1)
st.altair_chart(age_chart0)

st.markdown('It is interesting that the chart marking people with Heart Disease has numerous peaks, while the other has a fairly uniform peak. Perhaps this is because for the group with heart disease they needed medical attention and so the ages are more random. The distribution of Ages without the disease seems to peak around the time most people would start testing for heart disease for preventative measures')
st.markdown("There are 5 other columns to look at regarding continuous data")

c_values = df[["age","trestbps","chol","thalach",'oldpeak']]

c_values_desc = c_values.describe()

st.dataframe(c_values_desc)

st.markdown("Now that we have an understanding of our continuous data let's transform some of our categorical data")

st.markdown("Some of the columns have integers representing categories, let's change those to their string values")

#'thal' does not have a categorical value for 0 so we assume they count as 1 which is normal. Two entries were found with 'thal' = 0
df.at[48,['thal']] = 1
df.at[281,['thal']] = 1

with st.echo():
    
    sex_dict = {0:'female',1:'male'}
    cp_dict = {0:'typical_angina',1:'atypical_angina',2:'non_anginal_pain',3:'asymptomatic'}
    slope_dict ={0:'upsloping',1:'flat',2:'downsloping'}
    thal_dict = {1:'normal',2:'fixed_defect',3:'reversable_defect'}
    restecg_dict ={0:'normal',1:'abnormality',2:'lv_hypertrophy'}
    
    df.sex =df.sex.map(sex_dict)

    df.cp = df.cp.map(cp_dict)

    df.slope = df.slope.map(slope_dict)

    df.thal = df.thal.map(thal_dict)

    df.restecg = df.restecg.map(restecg_dict)


st.dataframe(df)

st.markdown('Since categorical data can make classification models accuaracy fall especially with a low sample size that we have lets change these categories into binary column values. This generally **increases** accuracy by removing categorical data and increasing the number of **features**')
    
with st.echo():
    #get_dummies turns categorical columns into seperate binary value columns. We did this for 5 columns
    df_wide = pd.get_dummies(df, columns = ['sex','cp','slope','thal','restecg'])
    
st.dataframe(df_wide)

st.header('Time for Classification')
st.markdown('Now that we transformed and explored our data lets begin the Classification Model Training')
st.markdown('First thing is to split our data into Training and Test data')
    
with st.echo():
    #We split the data into train and test data, random state of 1962 chosen to honour my father with the year he was born
    train_df, test_df = train_test_split(df_wide, test_size =0.2, random_state =1962)
    
    X_train = train_df.drop(columns = ['target'])
    y_train = train_df['target']

    X_test = train_df.drop(columns = ['target'])
    y_test = train_df['target']
    
st.markdown('We will be testing three classification models for this project:')
st.markdown('1.KNeighborsClassifier')
st.markdown('2.SVC')
st.markdown('3.RandomForestClassifier')

st.subheader("KNeighborsClassifier")



with st.echo():
    results_dict_KNC = {"n_neighbors": [],"mean_train_score": [],"mean_cv_score": []}

    
    #iterate over k values to tune for hyper parameters
    for k in range(2,50, 2):
        model = KNeighborsClassifier(n_neighbors = k)
    
        scores = cross_validate(model, X_train, y_train, cv = 10, return_train_score = True)
    
        results_dict_KNC["n_neighbors"].append(k)
    
        mean_train_score = scores['train_score'].mean()
    
        mean_test_score = scores['test_score'].mean()
    
        results_dict_KNC["mean_train_score"].append(mean_train_score)
    
        results_dict_KNC["mean_cv_score"].append(mean_test_score)
    
    #Convert the results dict to a data frame
    knn_plot_df = pd.DataFrame.from_dict(results_dict_KNC)

st.dataframe(knn_plot_df.sort_values(by = ['mean_cv_score'], ascending = False)) 

st.markdown('Our highest score is when using k = 40')

st.markdown('Now lets test it on our test data')

with st.echo():
    best_model_knn = KNeighborsClassifier(n_neighbors = 40)

    best_model_knn.fit(X_train,y_train)

    test_score_knn = best_model_knn.score(X_test, y_test)


st.text(str(round(test_score_knn,2)))

st.markdown(' The score 0.67 is not a great score, but it is a start lets see if our next classifier model does any better! ')
    

st.subheader('SVC')

with st.echo():
    
    #There are two hyper parameters for SVC so a nested for loop is nescessary for hyper parameter tuning
    hyperparameters = {
    "gamma": [0.1, 1.0, 10.0, 100.0],
    "C": [0.1, 1.0, 10.0, 100.0]
    }
    param_scores = {"gamma": [], "C": [], "train_accuracy": [], "valid_accuracy": []}

    gamma_list = hyperparameters["gamma"]
    C_list = hyperparameters["C"]

    for g in gamma_list:
        for c in C_list:
            model_svc = SVC(gamma = g ,C = c, random_state =1962)
    
            svc_scores = cross_validate(model_svc, X_train, y_train, cv = 5, return_train_score = True)
    
            param_scores["C"].append(c)
        
            param_scores["gamma"].append(g)
    
            mean_train_score_svc = svc_scores['train_score'].mean()
    
            mean_test_score_svc = svc_scores['test_score'].mean()
    
            param_scores["train_accuracy"].append(mean_train_score_svc)
    
            param_scores["valid_accuracy"].append(mean_test_score_svc)
            
    #convert results dictionary to df
    param_scores_df = pd.DataFrame.from_dict(param_scores).sort_values(by = 'valid_accuracy' ,ascending =False)
    
st.dataframe(param_scores_df)

st.markdown('The best hyperparameters are gamma = 0.1 and C = .1 with an accuracy of 0.5331')

with st.echo():
    best_svc = SVC(gamma = 0.1, C = 0.1, random_state = 1962)

    best_svc.fit(X_train, y_train)

    svc_test_score = best_svc.score(X_test, y_test)

st.text(str(round(svc_test_score,2)))

st.markdown('Looks like SVC is not the model we are looking for with a test score of 0.53')

st.subheader('RandomForestClassifier')

with st.echo():
    
    results_dict_rfc = {"n_estimators": [],"mean_train_score": [],"mean_cv_score": []}


    for k in range(2,20, 2):
        model_rfc = RandomForestClassifier(n_estimators = k)
    
        scores_rfc = cross_validate(model_rfc, X_train, y_train, cv = 10, return_train_score = True)
    
        results_dict_rfc["n_estimators"].append(k)
    
        mean_train_score_rfc = scores_rfc['train_score'].mean()
    
        mean_test_score_rfc = scores_rfc['test_score'].mean()
    
        results_dict_rfc["mean_train_score"].append(mean_train_score_rfc)
    
        results_dict_rfc["mean_cv_score"].append(mean_test_score_rfc)
        
        rfc_plot_df = pd.DataFrame.from_dict(results_dict_rfc)


rfc_df = rfc_plot_df.sort_values(by = ['mean_cv_score'], ascending = False)

rfc_df = rfc_df[rfc_df['mean_train_score'] != 1.0]

rfc_df = rfc_df.reset_index().drop(columns = ['index'])

top_n = rfc_df.iloc[0]['n_estimators']

top_score = round(rfc_df.iloc[0]['mean_cv_score'],2)

st.dataframe(rfc_df)

st.markdown('The best value of n is '+str(top_n)+' with an accuracy of '+ str(top_score))

with st.echo():
    
    best_rfc = RandomForestClassifier(n_estimators = int(top_n))

    best_rfc.fit(X_train, y_train)

    rfc_test_score = best_rfc.score(X_test, y_test)
    
st.text(str(round(rfc_test_score,4)))

st.markdown('With a score of '+str(round(rfc_test_score,4))+' looks like the RandomForestClassifier may be the best for our dataset!')
st.markdown('When running this random forest classifier test values are really high. This can be due to a couple of reasons in my opinion. Firstly the Data set is really small, with around 300 rows and 25 features this Data Set was very simple for RFC to get near to a 1.0 test score. Secondly the Data set does have some outliers in the column values and RFC handles outliers fairly well, compared to the other classification models used.')
st.markdown('One may rejoice at the sight of this score, but one has to remember that using RFC means that classifying is based within the limits of the values given.')
st.markdown("So if for example someone has an extremely high **thol** value if it is above the highest value in our test set the model may miss the mark.  ")


st.header('Conclusion')
st.markdown('In conclusion I believe that it is possible to classify this dataset. Additional fine tuning and more models could be used to get better results. The RandomForestClassifer works for this small Data Set but if one were to scale this project a new model may have to be chosen.')
st.markdown('Heart Disease in general is very preventable, we can all do simple things to decrease the risk of Heart Disease affecting ourselves or the ones we love.')
st.markdown('I hope in the future I can bring this project into a place where I can make sure that **No One Else** has to go through what I have in my family due to heart disease.')
st.markdown('Thanks for checking out my project! Any and all suggestions or critiques can be sent to markchris.vaz@gmail.com')
    


    
    
    
    


    
    
    
    

        


    
    






        
    
    
