
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import warnings
# Create your views here.
from Remote_User.models import ClientRegister_Model,heart_disease_model,heart_disease_prediction_model,detection_ratio_model,detection_results_model


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            detection_results_model.objects.all().delete()
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')

def Find_Heart_Disease_Ratio(request):
    detection_ratio_model.objects.all().delete()
    ratio = ""
    kword = 'Heart Disease'
    print(kword)
    obj = heart_disease_prediction_model.objects.all().filter(Q(prediction=kword))
    obj1 = heart_disease_prediction_model.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio_model.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'No Heart Disease'
    print(kword1)
    obj1 = heart_disease_prediction_model.objects.all().filter(Q(prediction=kword1))
    obj11 = heart_disease_prediction_model.objects.all()
    count1 = obj1.count();
    count11 = obj11.count();
    ratio1 = (count1 / count11) * 100
    if ratio1 != 0:
        detection_ratio_model.objects.create(names=kword1, ratio=ratio1)



    obj = detection_ratio_model.objects.all()
    return render(request, 'SProvider/Find_Heart_Disease_Ratio.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic = heart_disease_prediction_model.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})


def charts(request,chart_type):
    chart1 = detection_ratio_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_results_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def Find_Heart_Disease_Status_Details(request):

    status=''
    type=''
    obj1 =heart_disease_model.objects.values('age',
'sex',
'cp',
'trestbps',
'chol',
'fbs',
'resecg',
'thalach',
'exang',
'oldpeak',
'slope',
'ca',
'thal',
'target'
    )

    heart_disease_prediction_model.objects.all().delete()
    for t in obj1:

        age= t['age']
        sex= t['sex']
        cp= t['cp']
        trestbps= t['trestbps']
        chol= t['chol']
        fbs= t['fbs']
        resecg= t['resecg']
        thalach= t['thalach']
        exang= t['exang']
        oldpeak= t['oldpeak']
        slope= t['slope']
        ca= t['ca']
        thal= t['thal']
        target= t['target']


        chol1=int(chol)

        if chol1 >= 200 and chol1<= 350:
            hd = "Heart Disease"
        if chol1 > 100 and chol1 <= 200:
            hd = "No Heart Disease"
        elif chol1 <= 100 and chol1>=0:
            hd = "Heart Disease"


        heart_disease_prediction_model.objects.create(age=age,
        sex=sex,
        cp=cp,
        trestbps=trestbps,
        chol=chol,
        fbs=fbs,
        resecg=resecg,
        thalach=thalach,
        exang=exang,
        oldpeak=oldpeak,
        slope=slope,
        ca=ca,
        thal=thal,
        target=target,
        prediction=hd
            )

    obj =heart_disease_prediction_model.objects.all()
    return render(request, 'SProvider/Find_Heart_Disease_Status_Details.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =detection_results_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="TrainedData.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = heart_disease_prediction_model.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1

        ws.write(row_num, 0, my_row.age, font_style)
        ws.write(row_num, 1, my_row.sex, font_style)
        ws.write(row_num, 2, my_row.cp, font_style)
        ws.write(row_num, 3, my_row.trestbps, font_style)
        ws.write(row_num, 4, my_row.chol, font_style)
        ws.write(row_num, 5, my_row.fbs, font_style)
        ws.write(row_num, 6, my_row.resecg, font_style)
        ws.write(row_num, 7, my_row.thalach, font_style)
        ws.write(row_num, 8, my_row.exang, font_style)
        ws.write(row_num, 9, my_row.oldpeak, font_style)
        ws.write(row_num, 10, my_row.slope, font_style)
        ws.write(row_num, 11, my_row.ca, font_style)
        ws.write(row_num, 12, my_row.thal, font_style)
        ws.write(row_num, 13, my_row.target, font_style)
        ws.write(row_num, 14, my_row.prediction, font_style)


    wb.save(response)
    return response

def train_model(request):
    obj=''
    detection_results_model.objects.all().delete()
    warnings.filterwarnings('ignore')

    sns.set()
    # %matplotlib inline
    # import dataset
    heart_df = pd.read_csv('./heart.csv')
    heart_df.head(10)
    # description about dataset
    heart_df.describe()
    heart_df.shape
    heart_df.isnull().sum()
    heart_df.notnull().sum()
    heart_df.dtypes
    # Plotting the distribution plot.
    #plt.figure(figsize=(20, 25))
    plotnumber = 1

    for column in heart_df:
        if plotnumber < 14:
            ax = plt.subplot(4, 4, plotnumber)
            sns.distplot(heart_df[column])
            plt.xlabel(column, fontsize=20)
            plt.ylabel('Values', fontsize=20)
        plotnumber += 1
    #plt.show()
    # Correlation matrix

    plt.figure(figsize=(16, 8))

    corr = heart_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2g', linewidths=1)
   # plt.show()
    # checking the variance
    heart_df.var()
    heart_df['trestbps'] = np.log(heart_df['trestbps'])
    heart_df['chol'] = np.log(heart_df['chol'])
    heart_df['thalach'] = np.log(heart_df['thalach'])

    np.var(heart_df[["trestbps", 'chol', 'thalach']])
    heart_df.isnull().sum()
    x = heart_df.drop('target', axis=1)
    y = heart_df['target']
    # spliting the dataset

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=123)
    accuracies = {}

    # LogisticRegression Model *********************************

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies['Logistic Regression'] = acc * 100
    print("Logistic Regression")
    print("Accuracy score of the model is:", accuracy_score(y_test, y_pred) * 100, "%")
    print("Confusion matrix of the model", confusion_matrix(y_test, y_pred))

    print("Classification Report", classification_report(y_test, y_pred))



    detection_results_model.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, y_pred) * 100)

    #KNeighborsClassifier Model*********************************

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=8)

    knn.fit(x_train, y_train)

    y_pred1 = knn.predict(x_test)

    acc1 = accuracy_score(y_test, y_pred)
    accuracies['KNeighborsClassifier'] = acc1 * 100
    print("KNeighborsClassifier")
    accuracy_score(y_train, knn.predict(x_train))
    print("Accuracy score of the model is:", accuracy_score(y_test, y_pred1) * 100, "%")
    print("Confusion matrix of the model", confusion_matrix(y_test, y_pred1))

    print("Classification Report", classification_report(y_test, y_pred1))

    detection_results_model.objects.create(names="KNeighborsClassifier", ratio=accuracy_score(y_test, y_pred1) * 100)

    # SVC*********************************

    from sklearn.svm import SVC

    svc = SVC()
    svc.fit(x_train, y_train)

    y_pred2 = svc.predict(x_test)

    acc2 = accuracy_score(y_test, y_pred2)
    accuracies['SVC'] = acc2 * 100
    print("SVC")
    accuracy_score(y_train, svc.predict(x_train))

    print("Accuracy score of the model is:", accuracy_score(y_test, y_pred2) * 100, "%")
    print("Confusion matrix of the model", confusion_matrix(y_test, y_pred2))

    print("Classification Report", classification_report(y_test, y_pred2))

    detection_results_model.objects.create(names="SVC", ratio=accuracy_score(y_test, y_pred2) * 100)

    # DecisionTreeClassifier Model*********************************

    from sklearn.tree import DecisionTreeClassifier

    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)

    y_pred3 = dtc.predict(x_test)
    acc3 = accuracy_score(y_test, y_pred)
    accuracies['DecisionTreeClassifier'] = acc3 * 100
    print("DecisionTreeClassifier")
    accuracy_score(y_train, dtc.predict(x_train))
    print("Accuracy score of the model is:", accuracy_score(y_test, y_pred3) * 100, "%")
    print("Confusion matrix of the model", confusion_matrix(y_test, y_pred3))

    print("Classification Report", classification_report(y_test, y_pred3))

    detection_results_model.objects.create(names="Decision Tree Classifier", ratio=accuracy_score(y_test, y_pred3) * 100)

    # GridSearchCV Model*********************************

    from sklearn.model_selection import GridSearchCV

    grid_params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': range(2, 10, 1),
        'min_samples_leaf': range(2, 10, 1)
    }

    grid_search = GridSearchCV(dtc, grid_params, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    grid_search.best_score_
    y_pred4 = dtc.predict(x_test)
    acc4 = accuracy_score(y_test, y_pred4)
    print("GridSearchCV")

    accuracies['GridSearchCV'] = acc4 * 100
    print("Accuracy score of the model is:", accuracy_score(y_test, y_pred4) * 100, "%")
    print("Confusion matrix of the model", confusion_matrix(y_test, y_pred4))

    print("Classification Report", classification_report(y_test, y_pred4))

    detection_results_model.objects.create(names="GridSearchCV", ratio=accuracy_score(y_test, y_pred4) * 100)



    # RandomForestClassifier Model*********************************

    from sklearn.ensemble import RandomForestClassifier

    rfc = RandomForestClassifier(criterion='gini', max_depth=7, max_features='sqrt', min_samples_leaf=2,
                                 min_samples_split=4, n_estimators=180)
    rfc.fit(x_train, y_train)

    y_pred5 = rfc.predict(x_test)

    acc5 = accuracy_score(y_test, y_pred5)
    accuracies['RandomForestClassifier'] = acc5 * 100

    print("RandomForestClassifier")
    accuracy_score(y_train, rfc.predict(x_train))
    print("Accuracy score of the model is:", accuracy_score(y_test, y_pred5) * 100, "%")
    print("Confusion matrix of the model", confusion_matrix(y_test, y_pred5))

    print("Classification Report", classification_report(y_test, y_pred5))

    detection_results_model.objects.create(names="Random Forest Classifier", ratio=accuracy_score(y_test, y_pred5) * 100)

    # GradientBoostingClassifier Model*********************************

    from sklearn.ensemble import GradientBoostingClassifier

    gbc = GradientBoostingClassifier()

    parameters = {
        'loss': ['deviance', 'exponential'],
        'learning_rate': [0.001, 0.1, 1, 10],
        'n_estimators': [100, 150, 180, 200]
    }

    gbc = GridSearchCV(gbc, parameters, cv=5, n_jobs=-1, verbose=1)
    gbc.fit(x_train, y_train)

    y_pred6 = gbc.predict(x_test)

    acc6 = accuracy_score(y_test, y_pred6)

    print("GradientBoosting")
    accuracies['GradientBoosting'] = acc5 * 100
    print("Accuracy score of the model is:", accuracy_score(y_test, y_pred6) * 100, "%")
    print("Confusion matrix of the model", confusion_matrix(y_test, y_pred6))

    print("Classification Report", classification_report(y_test, y_pred6))

    colors = ["purple", "green", "orange", "magenta", "blue", "black"]

    detection_results_model.objects.create(names="Gradient Boosting", ratio=accuracy_score(y_test, y_pred6) * 100)

    obj = detection_results_model.objects.all()
    return render(request,'SProvider/train_model.html', {'objs': obj})














