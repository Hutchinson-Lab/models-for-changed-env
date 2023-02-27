
'''

'''
ds_meta = {

	"banknote-authentication": 
	{
		"title" : "Banknote Authentication Data Set",
		"abstract" : "Data were extracted from images that were taken for the evaluation of an authentication procedure for banknotes.",
		"home-url" : "https://archive.ics.uci.edu/ml/datasets/banknote+authentication",
		"url" : "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
		"local-path" : "",
		"filename" : "data_banknote_authentication.txt",
		"filetype" : ".txt",
		"number-of-features" : 4,
		"feature-column-position" : [i for i in range(0,4)],
		"categorical-feature-column-position": [],
		"label-column-position" : 4,
		"positive-label" : "1",
		"negative-label" : "0",
		"delimiter" : ",",
		"missing-data-identifier" : "",

	},
		
	"credit-approval": 
	{
		"title" : "Credit Approval Data Set",
		"abstract" : "This data concerns credit card applications.",
		"home-url" : "https://archive.ics.uci.edu/ml/datasets/Credit+Approval",
		"url" : "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data",		
		"local-path" : "",
		"filename" : "crx.data",
		"filetype" : ".data",
		"number-of-features" : 15,
		"feature-column-position" : [i for i in range(0,15)],
		"categorical-feature-column-position": [0,3,4,5,6,8,9,11,12],
		"label-column-position" : 15, 
		"positive-label" : "+",
		"negative-label" : "-",
		"delimiter" : ",",
		"missing-data-identifier" : "?",

	},

	"german-credit": 
	{
		"title" : "German Credit Data Set (Statlog)", # dat set description indicates that a cost matrix is required
		"abstract" : "This dataset classifies people described by a set of attributes as good or bad credit risks.",
		"home-url" : "https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)",
		"url" : "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric",		
		"local-path" : "",
		"filename" : "german.data-numeric",
		"filetype" : ".dat",
		"number-of-features" : 24,
		"feature-column-position" : [i for i in range(0,24)],
		"categorical-feature-column-position": [],
		"label-column-position" : 24, 
		"positive-label" : "1", #Good
		"negative-label" : 2, #Bad
		"delimiter" : "",
		"missing-data-identifier" : "?",

	},

	"australian-credit": 
	{
		"title" : "Australian Credit Approval Data Set (Statlog (",
		"abstract" : "his file concerns credit card applications. ",
		"home-url" : "https://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)",
		"url" : "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat",		
		"local-path" : "",
		"filename" : "australian.dat",
		"filetype" : ".dat",
		"number-of-features" : 14,
		"feature-column-position" : [i for i in range(0,14)],
		"categorical-feature-column-position": [0,3,4,5, 7,8,10,11],
		"label-column-position" : 14, 
		"positive-label" : "1",
		"negative-label" : "0",
		"delimiter" : "",
		"missing-data-identifier" : "?",

	},

	"audit": 
	{
		"title" : "Audit Data Set",
		"abstract" : "To build a predictor for classifying suspicious firms.",
		"home-url" : "https://archive.ics.uci.edu/ml/datasets/Audit+Data",
		"url" : "https://archive.ics.uci.edu/ml/machine-learning-databases/00475/audit_data.zip",
		"local-path" : "audit_data/trial.csv",
		"filename" : "trial.csv",
		"filetype" : ".csv",
		"number-of-features" : 17,
		"feature-column-position" : [i for i in range(0,17)],
		"categorical-feature-column-position": [1,],
		"label-column-position" : 17, 
		"positive-label" : "1",
		"negative-label" : "0",
		"delimiter" : ",",
		"missing-data-identifier" : "",

	},

	"spambase": 
	{
		"title" : "Spambase Data Set",
		"abstract" : "Classifying Email as Spam or Non-Spam.",
		"home-url" : "https://archive.ics.uci.edu/ml/datasets/Spambase",
		"url" : "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
		"local-path" : "",
		"filename" : "spambase.data",
		"filetype" : ".data",
		"number-of-features" : 57,
		"feature-column-position" : [i for i in range(0,57)],
		"categorical-feature-column-position": [],
		"label-column-position" : 57, 
		"positive-label" : "1",
		"negative-label" : "0",
		"delimiter" : ",",
		"missing-data-identifier" : "",

	},


	"heart-disease": 
	{
		"title" : "Heart Disease Data Set",
		"abstract" : "Classifying Email as Spam or Non-Spam.",
		"home-url" : "https://archive.ics.uci.edu/ml/datasets/heart+disease",
		"url" : "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
		"local-path" : "ad-dataset/ad.data",
		"filename" : "ad.data",
		"filetype" : ".data",
		"number-of-features" : 13,
		"feature-column-position" : [i for i in range(0,13)],
		"categorical-feature-column-position": [],
		"label-column-position" : 13, 
		"positive-label" : [1,2,3,4], # multiple identifiers denoting same class label
		"negative-label" : "0",
		"delimiter" : ",",
		"missing-data-identifier" : "?",

	},

	"heart-failure": 
	{
		"title" : "Heart Failure Data Set",
		"abstract" : "This dataset contains the medical records of 299 patients who had heart failure.",
		"home-url" : "https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records",
		"url" : "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv",
		"local-path" : "",
		"filename" : "heart_failure_clinical_records_dataset.csv",
		"filetype" : ".csv",
		"number-of-features" : 12,
		"feature-column-position" : [i for i in range(0,12)],
		"categorical-feature-column-position": [], # all categorical features are boolean/binary so no need to one-hot encode
		"label-column-position" : 12, 
		"positive-label" : "1",
		"negative-label" : "0",
		"delimiter" : ",",
		"missing-data-identifier" : "",

	},


	"parkinsons": 
	{
		"title" : "Parkinson's Data Set",
		"abstract" : "Oxford Parkinson's Disease Detection Dataset.",
		"home-url" : "https://archive.ics.uci.edu/ml/datasets/Parkinsons",
		"url" : "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
		"local-path" : "",
		"filename" : "parkinsons.data",
		"filetype" : ".data",
		"number-of-features" : 21,
		"feature-column-position" : [i for i in range(1,17)]+[i for i in range(18,24)],
		"categorical-feature-column-position": [], 
		"label-column-position" : 17, 
		"positive-label" : "1",
		"negative-label" : "0",
		"delimiter" : ",",
		"missing-data-identifier" : "",

	},

	"habermans-survival": 
	{
		"title" : "Haberman's Survival Data Set",
		"abstract" : "Dataset contains cases from study conducted on the survival of patients who had undergone surgery for breast cancer.",
		"home-url" : "https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival",
		"url" : "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data",
		"local-path" : "",
		"filename" : "haberman.data",
		"filetype" : ".data",
		"number-of-features" : 3,
		"feature-column-position" : [i for i in range(0,3)],
		"categorical-feature-column-position": [], 
		"label-column-position" : 3, 
		"positive-label" : "1", # patient survived longer than 5 years.
		"negative-label" : 2, # patient passed away within 5 years.
		"delimiter" : ",",
		"missing-data-identifier" : "",

	},

	"mushroom": 
	{
		"title" : "Mushroom Data Set",
		"abstract" : "From Audobon Society Field Guide; mushrooms described in terms of physical characteristics; classification: poisonous or edible.",
		"home-url" : "https://archive.ics.uci.edu/ml/datasets/Mushroom",
		"url" : "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
		"local-path" : "",
		"filename" : "agaricus-lepiota.data",
		"filetype" : ".data",
		"number-of-features" : 22,
		"feature-column-position" : [i for i in range(1,23)],
		"categorical-feature-column-position": [i for i in range(1,23)], 
		"label-column-position" : 0, 
		"positive-label" : "p", # p = poisonous
		"negative-label" : "e", # e = edible
		"delimiter" : ",",
		"missing-data-identifier" : "?",

	},

	"raisin": 
	{
		"title" : "Raisin Data Set",
		"abstract" : "Images of the Kecimen and Besni raisin varieties were obtained with CVS. A total of 900 raisins were used, including 450 from both varieties, and 7 morphological features were extracted.",
		"home-url" : "https://archive.ics.uci.edu/ml/datasets/Raisin+Dataset",
		"url" : "https://archive.ics.uci.edu/ml/machine-learning-databases/00617/Raisin_Dataset.zip",
		"local-path" : "Raisin_Dataset/Raisin_Dataset.xlsx",
		"filename" : "Raisin_Dataset.xlsx",
		"filetype" : ".xlsx",
		"number-of-features" : 7,
		"feature-column-position" : [i for i in range(0,7)],
		"categorical-feature-column-position": [], 
		"label-column-position" : 7, 
		"positive-label" : "Kecimen", # Type Kecimen.
		"negative-label" : "Besni", # Yype Besn.
		"delimiter" : ",",
		"missing-data-identifier" : "",

	},
	

	"climate-model-crashes": 
	{
		"title" : "Climate Model Simulation Crashes Data Set",
		"abstract" : "Given Latin hypercube samples of 18 climate model input parameter values, predict climate model simulation crashes and determine the parameter value combinations that cause the failures.",
		"home-url" : "https://archive.ics.uci.edu/ml/datasets/climate+model+simulation+crashes",
		"url" : "https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat",
		"local-path" : "",
		"filename" : "pop_failures.dat",
		"filetype" : ".dat",
		"number-of-features" : 18,
		"feature-column-position" : [i for i in range(2,20)],
		"categorical-feature-column-position": [], 
		"label-column-position" : 20, 
		"positive-label" : "0", # 0 = failure/crash.
		"negative-label" : "1", # 1 = succes/no crash.
		"delimiter" : ",",
		"missing-data-identifier" : "",

	},

	"ionosphere":
	{
		"title" : "Ionosphere Data Set",
		"abstract" : "Classification of radar returns from the ionosphere.",
		"home-url" : "https://archive.ics.uci.edu/ml/datasets/Ionosphere",
		"url" : "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
		"local-path" : "",
		"filename" : "ionosphere.data",
		"filetype" : ".data",
		"number-of-features" : 34,
		"feature-column-position" : [i for i in range(0,34)],
		"categorical-feature-column-position": [], 
		"label-column-position" : 34, 
		"positive-label" : "g", # g = good.
		"negative-label" : "b", # b = bad.
		"delimiter" : ",",
		"missing-data-identifier" : "",

	},

	"tic-tac-toe":
	{
		"title" : "Tic-Tac-Toe Endgame Data Set",
		"abstract" : "Binary classification task on possible configurations of tic-tac-toe game.",
		"home-url" : "https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame",
		"url" : "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data",
		"local-path" : "",
		"filename" : "tic-tac-toe.data",
		"filetype" : ".data",
		"number-of-features" : 9,
		"feature-column-position" : [i for i in range(0,9)],
		"categorical-feature-column-position": [i for i in range(0,9)], 
		"label-column-position" : 9, 
		"positive-label" : "positive", # postive position.
		"negative-label" : "negative", # negative position.
		"delimiter" : ",",
		"missing-data-identifier" : "",

	},




}
