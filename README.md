# Treatment-of-liver-metastases-with-SBRT-Analysis-of-variables-and-results-using-Machine-Learning-si

Treatment of liver metastases with SBRT: Analysis of variables and results using Machine Learning sing Machine Learning

1.	INTRODUCTION
One of the main challenges of medicine is the fight against cancer. It is estimated that this disease will affect 33% of the population at some point in their lives and therefore, its research and development is fundamental for our society.
In recent years, technological advances have allowed the development and application of new non-invasive cancer treatments, such as radiotherapy using the SBRT technique (Stereotactic Body Radiation Therapy). SBRT delivers high doses of radiation to small and well-defined targets in an extreme hypofractionated (and accelerated) scheme with a very high biological effectiveness obtaining very good initial clinical results in terms of local tumor control and acceptable rate of late complications. This contributes greatly to improving disease control and response as well as the patient's quality of life. However, the indication for this treatment depends on several factors such as the number of lesions, the clinical conditions of the patient and the presence of other foci of disease, among others, so it is not always easy to define the treatment parameters.
In parallel, the increase in computing capacity in recent years has allowed the analysis and processing of large volumes of data. This development has given rise to the creation of predictive algorithms in any area allowing obtaining from personalized recommendation systems, prediction of economic models to early detection of diseases.
Therefore, taking into account that the most frequent patients with disseminated disease are located in the liver and lung, we want to conduct a study with the data generated in patients with liver metastases in a certain hospital, in which we will analyze through ML techniques the different relationships between the variables used in SBRT treatments and thus create predictive models that allow us to obtain more knowledge to optimize according to the conditions of each patient, the treatment parameters to predict their response.

Objective
Develop a machine learning classification model in order to predict, based on patient characteristics and treatment parameters, whether the patient will develop local recurrence or not after SBRT treatment. We also analyze the weight of each variable in the final outcome, and the relationships between variables.




2.	WORKFLOW
This project has been divided into different jupyter notebooks:
2.1	Step_1: Feature selection.

Table I. data_Recidiva		
Name of variables	Meaning	Type
Sexo	Patient’s gender	str
Edad	Patient’s age	int
Num_lesion_total	total number of injuries*	int
tratamientos_vez	Number of treatments performed at the same time	int
tratatos_anteriomente	Has the patient been treated for an injury before?	str
Mts-tto	Time between discovery of metastases and the treatment	int
Mst-dco	Time between diagnosis of primary cancer and discovery of metastases	int
Histología	Time between treatment and discovery of metastases	str
NeumotoraxHematoma	Does the patient have hepatic hematoma?	str
numeroQtPrevias	Number of chemotherapies prior to treatment	int
TtoLocalPrevio	has the patient had any local treatment prior to radiotherapy?	str
mtsSNC	Does the patient have metastases in the central nervous system?	str
mtdOtrosNiveles	Does the patient have metastases in another area?	str
VolPTV	Planning treatment volume	int
BEDmax_test	Maximum effective biological dose	int
BEDmim_test	Minimum effective biological dose	int
VolHigSano	Healthy liver volume	int
ControlRespiratorio	Respiratory control during treatment	str
GradoTox	Patient’s degree of toxicity	int
Segmento	Liver segment	str
ReciLocal	Does the patient present local recurrence after treatment?	int


Table I. data_Recidiva		
Name of variables	Meaning	Type
Sexo	Patient’s gender	str
Edad	Patient’s age	int
Num_lesion_total	total number of injuries*	int
tratamientos_vez	Number of treatments performed at the same time	int
tratatos_anteriomente	Has the patient been treated for an injury before?	str
Mts-tto	Time between discovery of metastases and the treatment	int
Mst-dco	Time between diagnosis of primary cancer and discovery of metastases	int
Histología	Time between treatment and discovery of metastases	str
NeumotoraxHematoma	Does the patient have hepatic hematoma?	str
numeroQtPrevias	Number of chemotherapies prior to treatment	int
TtoLocalPrevio	has the patient had any local treatment prior to radiotherapy?	str
mtsSNC	Does the patient have metastases in the central nervous system?	str
mtdOtrosNiveles	Does the patient have metastases in another area?	str
VolPTV	Planning treatment volume	int
BEDmax_test	Maximum effective biological dose	int
BEDmim_test	Minimum effective biological dose	int
VolHigSano	Healthy liver volume	int
ControlRespiratorio	Respiratory control during treatment	str
GradoTox	Patient’s degree of toxicity	int
Segmento	Liver segment	str
ReciLocal	Does the patient present local recurrence after treatment?	int
* Each injurie is a metastasis
The starting point is an anonymized database of 150 patients with more than 400 treated metastases provided by e-mail by means of an excel file named Base Liver SPSS 2019 by lesions anonymized.xlsx.
The information contained in this database ranges from patient demographics (age, sex), date of diagnosis of the primary tumor and metastases, histology (age, sex), date of diagnosis of the primary tumor and metastases, histology, previous treatments (chemotherapy, surgery, etc.), and previous treatments (chemotherapy, surgery...), dosimetric parameters of the radiotherapeutic treatment (prescribed dose, maximum dose, maximum dose, dose prescribed dose, maximum dose, minimum dose, biological equivalent dose), response to treatment, date of progression, site of progression, survival time, cause of death, etc
Taking into account that the objective is to predict the result of the treatment, ReciLocal, we prepare the dataset by creating new variables and making a feature selection resulting in a dataset of dimension (405 x 21) shown above in Table I: data_Recidiva. 
It is important to clarify that each rows correspond with one treatment, therefore if a patient has had two treatments, he will have two rows. 
This notebook also includes an Exploratory Data Analysis through visualizations and statistics. This step allows on the one hand visualizing the distribution of our data, the outliers or spurious values and on the other hand to examine the correlations between our variables. 
This analysis has allowed us to conclude:
- Generally, the higher the radiation doses in the treatment, the lower the chance of local recurrence. 
- All patients with toxicity grade 4, present local recurrence. That toxicity is linked to previous chemotherapy treatment.
- All patients who have had more than 5 lesions treated at the same time do not present local recurrence. Therefore, we can say that if the patient has 5 or more lesions it is better to treat them all at the same time
- The target classes are imbalanced: No local recurrence: 84% vs local recurrence: 16%
- 
![image](https://user-images.githubusercontent.com/74373030/125138705-6dedeb80-e10f-11eb-9aca-8bd33c1e923c.png)



When we are faced with a data imbalanced problem, we must take into account that it affects to the algorithms in their process of generalization of the information, harming the minority classes and therefore we will have to take into account the metric to choose (For an analysis of the metrics I have uploaded a notebook called metrics, in which different metrics are analyzed and it is shown that although the accuracy metric gives good results it is not good for imbalanced data problems.)
In order to achieve our goal of obtaining a classification model that predicts treatment response, we will train different models by performing different data balancing strategies:
1.	Data_level approach: We modify the distribution of the data removing noisy observations or retaing those harder to classify or creating synthetic new observations or simply to add or remove randomly observations:  Undersampling and Over sampling (step_3 notebooks)

2.	Ensemble algorithms: Combine weak learner through bagging and boosting to make more robust predictioons: Boosting and bagging. In an ensemble we construct multiple classifiers from the original data and then we aggregate their predictions. (combining classifiers generally improve the generalization ability) The problem is that the classifiers tend to optimize the accuracy so we are not tackling with imbalanced data issue. So we have to combine it with either a data level approach or a cost sensitive approach (step_4 notebook).
3.	Cost sensitive approach: We introduce a cost into the minimization function of the different algorithms : Higher miss-classification costs(step_5 notebook)

2.2 Step_2: Preprocessing data 
In this notebook we prepare the dataset for use it in different training models. We split the data into train and testset (taking into account the imbalanced issue we use a version of k-fold cross-validation that preserves the imbalanced class distribution in each fold. It is called stratified k-fold cross-validation and will enforce the class distribution in each split of the data to match the distribution in the complete training dataset), the scaling through the RobustScaler function because is more robust to outliers and the categorical variables encoding  by OneHotEncoding().
On the other hand, an analysis of the different over and undersampling techniques with the final data size of each technique is carried out. Here I show some examples of the final distribution of the samples after applying the different sampling techniques compared to the initial data: raw data. In each graph the final size is also shown.
![image](https://user-images.githubusercontent.com/74373030/125139080-2a47b180-e110-11eb-801a-efe3cfe5df90.png)
2.3 Step_3: Data Level approach 

![image](https://user-images.githubusercontent.com/74373030/125139098-33388300-e110-11eb-9d43-2f4c93a484b7.png)


Oversampling:

Explain at notebooks
Undersampling
Above is a summary of the best results (see notebook for more information).
The choice of metric depends on what we want to predict. In our case we are interested in minimizing the prediction errors of the minority class.( False Negative Rate, FNR) and since we want it to be independent of the threshold we use as metric the roc-auc which is also not affected by the imbalanced classes.
Area Under the ROC: AUC. AUC provides an aggregate measure of performance across all possible classification thresholds.
●Higher AUC indicates the model is better at predicting both classes
AUC=1 for a perfect model 0.5 for a random model (varia entre 1 y 0.5)
advantages of this metric show you visibility of the model performance across probability thresholds and gives you versatility to select this threshold depending we want to minimize. 

Combination
We don`t manage to get good results
2.4 Step 4: Ensemble models
Improve model performance combining several classifiers. Their decisions can be combined or aggregated. The idea is to improve generalization. Such classifier make erros, but each classification makes different errors. Thus classifiers complement each other.
 2.5 Step 5: Cost Sensitive Learning (CSL) 
CSL is a type of learning that takes (misclassification) costs into account. This method minimizes the total misclassification cost instead of the error rate classification.
